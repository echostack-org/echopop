"""
Length-TS acoustic inversion using empirical target strength–length relationships.

Implements ``InversionLengthTS``, which converts vertically integrated backscatter (S_A)
to fish density and abundance via a log-linear TS-length regression, and provides the
standalone ``ts_length_regression`` helper for computing mean backscattering cross-sections.
"""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from .. import validators as val
from ..survey import quantize_length_data
from .inversion_base import InversionBase
from .utils import impute_missing_sigma_bs


class InversionLengthTS(InversionBase):
    r"""
    Perform acoustic inversion using a log-linear length–TS relationship.

    This class implements acoustic inversion by relating fish length to acoustic backscatter
    through empirical TS-length relationships. It calculates stratified mean backscattering
    cross-sections and converts :math:`S_\\text{A}` (vertically integrated backscatter) to
    estimates of number density.

    Parameters
    ----------
    model_parameters : Dict[str, Any]
        Dictionary containing model configuration. Required keys:

        - ``'ts_length_regression'``: Dictionary with ``'slope'`` and ``'intercept'`` for the
          TS-length regression equation.

        - ``'stratify_by'``: str or List[str] - stratification columns (e.g., ``'stratum_number'``)

        Optional keys:

        - ``'expected_strata'``: An array of specific strata to expect in the data

        - ``'impute_missing_strata'``: A boolean argument that elects whether or not to impute
          missing strata values

        - ``'haul_replicates':`` A boolean argument that elects whether or not to use haul
          numbers/ids as replicates instead of individuals to account for pseudoreplication.

    Attributes
    ----------
    sigma_bs_haul : |pd.DataFrame| or None
        Haul-level backscattering cross-sections
    sigma_bs_strata : |pd.DataFrame| or None
        Stratum-level backscattering cross-sections

    Examples
    --------
    >>> # Define model parameters
    >>> params = {
    ...     "ts_length_regression": {"slope": 20.0, "intercept": -68.0},
    ...     "stratify_by": "stratum_ks",
    ...     "strata": [1, 2, 3, 4, 5],
    ...     "impute_missing_strata": True
    ... }
    >>>
    >>> # Initialize inverter
    >>> inverter = InversionLengthTS(params)
    >>>
    >>> # Set haul-level sigma_bs from biological data
    >>> inverter.set_haul_sigma_bs([specimen_df, length_df])
    >>>
    >>> # Perform inversion on NASC data
    >>> result = inverter.invert(nasc_df)
    >>> print(result['number_density'].sum())
    1234567.89

    Notes
    -----
    The inversion process follows these steps:

    1. Quantize length data and compute TS using the regression equation
    2. Convert TS to linear backscattering cross-section (:math:`\\sigma_\\text{bs}`)
    3. Calculate stratified mean sigma_bs values
    4. Impute missing strata if specified
    5. Convert vertically integrated backscatter to number density where:

    .. math::
        \\rho_\\text{v} = \\frac{S_\\text{A}}{4\\pi \\times \\sigma_\\text{bs}}

    where :math:`\\rho_\\text{v}` is number density (animals :math:`\\text{nmi}^{-2}`) and
    :math:`S_\\text{A}` is the vertically integrated backscatter over an area (:math:`\\text{m}^2\\,
    \\text{nmi}^{-2}`) that is commonly called the nautical area scattering coefficient (NASC).
    """

    def __new__(
        cls,
        model_parameters: dict[str, Any],
    ):
        """Create and validate a new InversionLengthTS instance."""
        # Validate
        try:
            # ---- Check
            valid_args = val.ValidateLengthTS.create(**model_parameters)
        # Break creation
        except ValidationError as e:
            raise e from None

        # Create instance
        self = super().__new__(cls)

        # Update attributes
        self.model_params = valid_args

        # Generate
        return self

    def __init__(self, model_parameters: dict[str, Any]):

        # Set inversion method
        self.inversion_method = "length_TS_regression"

        # Initialize attributes
        self.sigma_bs_haul = None
        self.sigma_bs_strata = None

    def set_haul_sigma_bs(self, length_data: pd.DataFrame | list[pd.DataFrame]) -> pd.DataFrame:
        """
        Compute the mean linear scattering coefficient (sigma_bs) for each haul.

        This method processes length data from biological samples to calculate haul-specific
        backscattering cross-sections using the TS-length regression relationship.

        Parameters
        ----------
        length_data : pd.DataFrame or list of pd.DataFrame
            Length data containing fish measurements. Can be:

            - Single DataFrame with length measurements per haul

            - List of DataFrames to be concatenated

            - Dictionary of DataFrames (values will be used)

            Required columns:

            - All columns specified in model_params["stratify_by"]

            - "haul_num": Haul identifier

            - "length": Fish length measurements

            - "length_count" (optional): Pre-aggregated counts per length

        Notes
        -----
        The method performs the following steps:

        1. Quantizes length data by stratification variables and haul

        2. Applies TS-length regression to convert lengths to target strength

        3. Converts TS to linear backscattering cross-section (sigma_bs)

        4. Calculates length-weighted average sigma_bs per haul

        Examples
        --------
        >>> # Single DataFrame
        >>> inverter.set_haul_sigma_bs(specimen_df)
        >>>
        >>> # Multiple DataFrames
        >>> inverter.set_haul_sigma_bs([specimen_df, length_df])
        """
        # Create copy
        df_length = length_data.copy()

        # Prepare the calculation depending on if `df` is a single DataFrame or Dictionary
        if isinstance(df_length, pd.DataFrame):
            df_length = [df_length]
        elif isinstance(df_length, dict):
            df_length = [d for d in df_length.values()]

        # Quantize the lengths
        haul_counts = pd.concat(
            [
                quantize_length_data(
                    d, self.model_params["stratify_by"] + [self.model_params["haul_column"]]
                )
                for d in df_length
            ],
            axis=1,
        )

        # Fill any mismatched indices
        haul_counts = haul_counts.fillna(0.0).sum(axis=1).reset_index(name="length_count")

        # Compute the average TS
        haul_counts["TS"] = ts_length_regression(
            haul_counts["length"],
            self.model_params["ts_length_regression"]["slope"],
            self.model_params["ts_length_regression"]["intercept"],
        )

        # Linearize
        haul_counts["sigma_bs"] = 10.0 ** (haul_counts["TS"] / 10.0)

        # Weighted average across hauls
        sigma_bs_haul = haul_counts.groupby(
            self.model_params["stratify_by"] + [self.model_params["haul_column"]]
        )[["sigma_bs", "length_count"]].apply(
            lambda x: np.average(x.sigma_bs, weights=x.length_count)
        )

        # Return the output
        return sigma_bs_haul.to_frame("sigma_bs")

    def get_stratified_sigma_bs(
        self,
        length_data: pd.DataFrame | list[pd.DataFrame],
        haul_replicates: bool = True,
    ) -> None:
        """
        Calculate stratified mean backscattering cross-sections (sigma_bs) by stratum.

        This method computes stratum-level sigma_bs values from either all individuals directly, or
        by hauls to account for pseudoreplication. The latter would treat each haul as an
        individual replicate (i.e. the unit of replication is the haul)[1]_. The resulting values
        are used in acoustic inversion to convert NASC to number density (animals nmi^-2).

        Parameters
        ----------
        length_data : pd.DataFrame or list of pd.DataFrame
            Length data for computing sigma_bs.

            Required columns (when provided):

            - All columns specified in model_params["stratify_by"]

            - "length": Fish length measurements

            - "length_count" (optional): Pre-aggregated counts per length

            - "haul_num": When `haul_replicates=True`, otherwise it is optional.

        haul_replicates : bool
            When True, individual hauls are used as replicates instead of individual lengths. This
            involves averaging sigma_bs for each haul.

        Returns
        -------
        pd.DataFrame
            DataFrame with stratified sigma_bs values. Index contains stratum identifiers,
            and 'sigma_bs' column contains the mean backscattering cross-section values
            for each stratum.

        Notes
        -----
        Using hauls as the unit of replication is recommended to avoid pseudoreplication when
        individuals within hauls are not independent[1]_.

        The sigma_bs calculation follows:

        1. Convert length to TS using regression: TS = slope * log10(length) + intercept

        2. Convert TS to linear scale: sigma_bs = 10^(TS/10)

        3. Calculate weighted average by length frequency within each stratum

        Examples
        --------
        >>> # Direct calculation from new data
        >>> sigma_bs = inverter.get_stratified_sigma_bs(specimen_df)

        References
        ----------
        .. [1] Hurlbert, S.H. (1984). Pseudoreplication and the Design of Ecological Field
        Experiments. *Ecological Monographs*, 54(2), 187-211. https://doi.org/10.2307/1942661
        """
        # Create copy
        df_length = length_data.copy()

        # Grab mean hauls if `df_length` is not specified
        if df_length is None and self.sigma_bs_haul is not None:
            self.sigma_bs_strata = self.sigma_bs_haul.unstack(
                self.model_params["stratify_by"]
            ).mean(axis=0)
        # ---- Otherwise, apply the appropriate groupby operation
        else:
            # ---- Prepare the calculation depending on if `df` is a single DataFrame or Dictionary
            if isinstance(df_length, pd.DataFrame):
                df_length = [df_length]
            elif isinstance(df_length, dict):
                df_length = [d for d in df_length.values()]

            # Quantize the length counts across all datasets per length value
            df_length_counts = (
                pd.concat(
                    [quantize_length_data(d, self.model_params["stratify_by"]) for d in df_length],
                    axis=1,
                )
                .fillna(0.0)
                .sum(axis=1)
                .reset_index(name="length_count")
            )

            # Compute the average TS
            df_length_counts["TS"] = ts_length_regression(
                df_length_counts["length"],
                self.model_params["ts_length_regression"]["slope"],
                self.model_params["ts_length_regression"]["intercept"],
            )

            # Linearize
            df_length_counts["sigma_bs"] = 10.0 ** (df_length_counts["TS"] / 10.0)

            # Aggregate by stratum
            self.sigma_bs_strata = df_length_counts.groupby(self.model_params["stratify_by"])[
                ["length_count", "sigma_bs"]
            ].apply(lambda x: np.average(x.sigma_bs, weights=x.length_count))

        # Use hauls as replicates
        if haul_replicates:
            # ---- Store the values
            self.sigma_bs_haul = self.set_haul_sigma_bs(df_length)
            # ---- Set the mean sigma_bs for each stratum
            self.sigma_bs_strata = (
                self.sigma_bs_haul["sigma_bs"]
                .unstack(self.model_params["stratify_by"])
                .mean(axis=0)
                .to_frame("sigma_bs")
            )
        # Use individuals as replicates
        else:
            # ---- Prepare the calculation depending on if `df` is a single DataFrame
            if isinstance(df_length, pd.DataFrame):
                df_length = [df_length]
            # ---- Quantize the length counts across all datasets per length value
            df_length_counts = (
                pd.concat(
                    [quantize_length_data(d, self.model_params["stratify_by"]) for d in df_length],
                    axis=1,
                )
                .fillna(0.0)
                .sum(axis=1)
                .reset_index(name="length_count")
            )
            # ---- Compute the average TS
            df_length_counts["TS"] = ts_length_regression(
                df_length_counts["length"],
                self.model_params["ts_length_regression"]["slope"],
                self.model_params["ts_length_regression"]["intercept"],
            )
            # ---- Linearize
            df_length_counts["sigma_bs"] = 10.0 ** (df_length_counts["TS"] / 10.0)
            # ---- Aggregate by stratum
            self.sigma_bs_strata = (
                df_length_counts.groupby(self.model_params["stratify_by"])[
                    ["length_count", "sigma_bs"]
                ].apply(lambda x: np.average(x.sigma_bs, weights=x.length_count))
            ).to_frame("sigma_bs")

    def invert(self, nasc_data: pd.DataFrame, length_data: pd.DataFrame | list[pd.DataFrame]):
        """Invert NASC data to number density using the TS-length regression."""
        # Create copy
        df_nasc = nasc_data.copy()

        # When stratified
        if "stratify_by" in self.model_params:

            # Get the unique strata
            if self.model_params["expected_strata"] is not None:
                unique_strata = np.unique(self.model_params["expected_strata"])
            else:
                unique_strata = np.unique(df_nasc[self.model_params["stratify_by"]])

            # Get the mean linear scattering coefficient for each stratum
            self.get_stratified_sigma_bs(length_data, self.model_params["haul_replicates"])

            # Impute, if defined
            if (
                "impute_missing_strata" in self.model_params
                and self.model_params["impute_missing_strata"]
            ):
                self.sigma_bs_strata = impute_missing_sigma_bs(unique_strata, self.sigma_bs_strata)

            # Index `df_nasc` to align with `sigma_bs_strata`
            df_nasc.set_index(self.model_params["stratify_by"], inplace=True)

            # Compute number density
            df_nasc["number_density"] = (
                (
                    df_nasc["nasc_proportion"].fillna(0.0)
                    * df_nasc["nasc"]
                    / (4 * np.pi * self.sigma_bs_strata["sigma_bs"].reindex_like(df_nasc))
                )
                .astype(float)
                .fillna(0.0)
            )

            # Return the NASC DataFrame with the inverted number densities
            return df_nasc.reset_index()


def ts_length_regression(length: np.ndarray | float, slope: float, intercept: float) -> np.ndarray:
    r"""
    Convert length values into acoustic target strength (TS, dB re. 1 :math:`\text{m}^{-2}`).

    Parameters
    ----------
    length : Union[np.ndarray, float]
        Length value(s) typically represented in 'cm' that will be converted into acoustic target
        strength (TS, dB re. 1 :math:`\\text{m}^{-2}`).
    slope : float
        TS-length regression slope coefficient
    intercept : float
        TS-length regression intercept coefficient

    Returns
    -------
    np.ndarray
        Target strength values in dB re. 1 :math:`\\text{m}^{-2}`

    Examples
    --------
    Single length

    >>> ts = ts_length_regression(20.0, slope=20.0, intercept=-68.0)
    >>> print(f"TS for 20cm fish: {ts:.2f} dB")
    TS for 20cm fish: -42.00 dB

    Multiple lengths

    >>> # Multiple length values
    >>> lengths = np.array([10, 15, 20, 25, 30])
    >>> ts_values = ts_length_regression(lengths, slope=20.0, intercept=-68.0)
    >>> print("Lengths:", lengths)
    >>> print("TS values:", ts_values)
    Lengths: [10 15 20 25 30]
    TS values: [-48.   -44.77 -42.   -39.82 -38.   ]

    Notes
    -----
    The TS-length relationship follows the standard log-linear form:

    .. math::
        \\text{TS} = \\beta_1 \\times \\log_{10}(L) + \\beta_0

    where :math:`L` is the length, :math:`\\beta_1` is the slope, and :math:`\\beta_0` is the
    :math:`y`-intercept. This is commonly used in fisheries acoustics where the relationship between
    fish length and acoustic backscatter follows this logarithmic pattern.
    """
    return slope * np.log10(length) + intercept
