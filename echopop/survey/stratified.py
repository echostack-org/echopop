"""Stratified survey analysis including kriging, bootstrap resampling, and biomass estimation."""

from typing import Any, Literal

import awkward as awk
import geopy.distance
import numpy as np
import pandas as pd

from .. import utils
from ..ingest import join_geostrata_by_latitude
from . import statistics


class JollyHampton:
    """
    Jolly-Hampton stratified survey analysis with bootstrap resampling.

    This class implements stratified sampling analysis using the Jolly-Hampton method with
    bootstrap resampling for uncertainty estimation. It handles virtual transect creation,
    stratified bootstrap sampling, and confidence interval estimation.

    Parameters
    ----------
    model_parameters : Dict[str, Any]
        Dictionary containing model configuration parameters:

        - "transects_per_latitude": Number of transects per degree latitude
        - "strata_transect_proportion": Proportion of transects to sample per stratum
        - "num_replicates": Number of bootstrap replicates

    resample_seed : int, optional
        Random seed for reproducible bootstrap resampling, by default None.

    Attributes
    ----------
    model_params : Dict[str, Any]
        Stored model parameters.
    rng : np.random.Generator
        Random number generator for bootstrap sampling.
    bootstrap_replicates : pd.DataFrame or None
        DataFrame containing bootstrap replicates (set after stratified_bootstrap).
    variable : str or None
        Name of the response variable being analyzed.
    transect_summary : pd.DataFrame
        Summary statistics for each transect within strata.
    strata_summary : pd.DataFrame
        Summary statistics for each stratum.
    survey_summary : Dict[str, pd.DataFrame]
        Dictionary containing survey-level summaries for strata and overall survey.

    Notes
    -----
    The Jolly and Hampton algorithm is commonly used in fisheries acoustic surveys for estimating
    fish biomass and abundance with stratified sampling designs.

    References
    ----------
    Jolly, G.M., and Hampton, I. (1990). A stratified random transect design for acoustic surveys
    of fish stocks. Canadian Journal of Fisheries and Aquatic Sciences, 47(7), 1282-1291.
    https://doi.org/10.1139/f90-147

    Examples
    --------
    >>> model_params = {
    ...     "transects_per_latitude": 5,
    ...     "strata_transect_proportion": 0.75,
    ...     "num_replicates": 100,
    ... }
    >>> analysis = JollyHampton(model_params, resample_seed=42)
    >>> virtual_data = analysis.create_virtual_transects(
    ...     mesh_data, geostratum_df,
    ...     stratify_by=["geostratum_inpfc"],
    ...     variable="biomass"
    ... )
    >>> analysis.stratified_bootstrap(
    ...     virtual_data,
    ...     stratify_by=["geostratum_inpfc"],
    ...     variable="biomass"
    ... )
    >>> summary = analysis.summarize(ci_percentile=0.95)
    """

    def __init__(
        self,
        model_parameters: dict[str, Any],
        resample_seed: int | None = None,
    ):
        # Ingest model parameters
        self.model_params = model_parameters

        # Initialize the random number generator
        self.rng = np.random.default_rng(resample_seed)

        # Initialize attributes
        # ---- Bootstrapped quantities for each replicate
        self.bootstrap_replicates = None
        # ---- Population statistic variable
        self.variable = None
        # ---- Summary statistics for each transect
        self.transect_summary = None
        # ---- Summary statistics for each stratum
        self.strata_summary = None
        # ---- Summary (global) statistics for the entire survey
        self.survey_summary = None

    def _partition_data_into_transects(
        self,
        data_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Partition gridded dataset into virtual transects based on latitude.

        Creates virtual transects by discretizing latitude coordinates and assigning sequential
        transect numbers. This is commonly used in acoustic surveys where data is collected on a
        regular grid.

        Parameters
        ----------
        data_df : pd.DataFrame
            Input DataFrame containing gridded data with 'latitude' column.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tuple containing:

            - Modified data DataFrame with added 'transect_num' column

            - Lookup DataFrame mapping latitude to transect numbers

        Notes
        -----
        The number of transects per latitude degree is controlled by the 'transects_per_latitude'
        parameter in ``model_parameters``.
        """
        # Get model parameters
        mp = self.model_params

        # Create a copy to avoid modifying original data
        data_copy = data_df.copy()

        # Partition the dataset based on latitude - round to nearest transect latitude
        data_copy.loc[:, "latitude"] = (
            utils.round_half_up(data_copy.loc[:, "latitude"] * mp["transects_per_latitude"] + 0.5)
            / mp["transects_per_latitude"]
        )

        # Create unique key pairs for latitude and transect
        unique_latitude_transect_key = pd.DataFrame(
            {
                "latitude": np.unique(data_copy["latitude"]),
                "transect_num": np.arange(0, len(np.unique(data_copy["latitude"])), 1) + 1,
            }
        ).set_index("latitude")

        # Temporarily set index for efficient joining
        data_copy.set_index("latitude", inplace=True)

        # Append the transect numbers
        data_copy["transect_num"] = unique_latitude_transect_key["transect_num"]

        # Return the partitioned gridded dataset and unique latitude-transect key
        return data_copy.reset_index(), unique_latitude_transect_key.reset_index()

    def _format_virtual_transects(
        self,
        data_df: pd.DataFrame,
        variable: str,
    ) -> pd.DataFrame:
        """
        Initialize virtual transects DataFrame from gridded data.

        Selects relevant columns and prepares the DataFrame for virtual transect calculations by
        renaming area column and setting transect_num as index.

        Parameters
        ----------
        data_df : pd.DataFrame
            Input DataFrame with partitioned transect data.
        variable : str
            Name of the response variable column.

        Returns
        -------
        pd.DataFrame
            Filtered and indexed DataFrame ready for virtual transect calculations.
        """
        # Supply required columns
        required_cols = ["transect_num", "longitude", "latitude", "area", variable]

        # Validate that all required columns are present
        missing_cols = [col for col in required_cols if col not in data_df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        return (
            data_df[required_cols]
            .rename(columns={"area": "area_interval"})
            .sort_values(["transect_num"])
            .set_index(["transect_num"])
        )

    def _generate_virtual_transects(
        self,
        virtual_df: pd.DataFrame,
        latitude_key: pd.DataFrame,
        variable: str,
    ) -> pd.DataFrame:
        """
        Compute transect distances and areas for virtual transects.

        Calculates mean latitude, transect distance (using geopy), transect area, and sums the
        response variable for each virtual transect.

        Parameters
        ----------
        virtual_df : pd.DataFrame
            DataFrame with virtual transect data indexed by transect_num.
        latitude_key : pd.DataFrame
            Lookup DataFrame mapping latitude to transect numbers.
        variable : str
            Name of the response variable column.

        Returns
        -------
        pd.DataFrame
            DataFrame with computed transect metrics including distance, area,
            and summed response variable.

        Notes
        -----
        Transect distances are calculated using great circle distance (nautical miles). Transect
        areas account for edge effects where end transects get half the area of interior transects.
        """
        # Initialize with mean latitude values for each transect
        virtual_transect_data = virtual_df.groupby(level=0)["latitude"].mean().to_frame("latitude")

        # Compute transect distances using geopy great circle distance
        virtual_transect_data["transect_distance"] = virtual_df.groupby(level=0).apply(
            lambda x: geopy.distance.distance(
                (x.latitude.min(), x.longitude.min()), (x.latitude.max(), x.longitude.max())
            ).nm,
            include_groups=False,
        )

        # Get the average latitude spacing (in nmi)
        latitude_spacing = np.diff(latitude_key["latitude"]).mean() * 60

        # Determine edge cases (first and last intervals)
        is_edge_transect = virtual_transect_data.index.isin(
            [virtual_df.index.min(), virtual_df.index.max()]
        )

        # Compute the transect areas
        virtual_transect_data.loc[:, "transect_area"] = np.where(
            is_edge_transect,
            virtual_transect_data["transect_distance"] * latitude_spacing / 2,
            virtual_transect_data["transect_distance"] * latitude_spacing,
        )

        # Sum the response variable for each transect
        virtual_transect_data[variable] = virtual_df.groupby(level=0)[variable].sum()

        return virtual_transect_data

    def create_virtual_transects(
        self,
        mesh_data: pd.DataFrame,
        geostrata: pd.DataFrame,
        stratum_dim: str,
        variable: str,
    ) -> pd.DataFrame:
        """
        Create virtual transects from gridded data and assign to strata.

        This method converts gridded data into virtual transects by partitioning based on latitude,
        computing transect metrics, and assigning transects to geographical strata.

        Parameters
        ----------
        mesh_data : pd.DataFrame
            Input DataFrame containing gridded survey data with columns: latitude, longitude, area,
            and the response variable.
        geostrata : pd.DataFrame
            DataFrame containing geographical stratum boundaries and definitions.
        stratum_dum : str
            Stratification column name (typically geographical strata).
        variable : str
            Name of the response variable column (e.g., 'biomass', 'abundance').

        Returns
        -------
        pd.DataFrame
            DataFrame containing virtual transects with computed distances, areas, response
            variable sums, and stratum assignments.

        Notes
        -----
        The virtual transects represent aggregated data along latitude bands, which is a common
        approach in systematic acoustic surveys.
        """
        # Assign transect numbers and get the latitude-transect key
        data_proc, latitude_key = self._partition_data_into_transects(mesh_data.copy())

        # Initialize the virtual transects DataFrame
        transects_df = self._format_virtual_transects(data_proc, variable)

        # Compute the summary metrics (distance, area, variable sums)
        virtual_df = self._generate_virtual_transects(transects_df, latitude_key, variable)

        # Stratify the virtual transects
        virtual_df = join_geostrata_by_latitude(
            data=virtual_df, geostrata=geostrata, stratum_name=stratum_dim
        )

        return virtual_df.reset_index()

    def _prepare_bootstrap_arrays(
        self,
    ) -> tuple[awk.Array, awk.Array, awk.Array]:
        """
        Prepare awkward arrays for bootstrap resampling.

        Creates nested arrays containing bootstrap samples of transects for each stratum, organized
        by distances, areas, and response variable values.

        Returns
        -------
        Tuple[awk.Array, awk.Array, awk.Array]
            Tuple containing awkward arrays for:

            - Sampled distances

            - Sampled areas

            - Sampled response variable values

        Notes
        -----
        The bootstrap sampling is done without replacement within each stratum. Sample sizes are
        determined by the 'num_transects_to_sample' for each stratum.
        """
        # Get the transect summary with reset index for easier indexing
        transect_df = self.transect_summary.reset_index(["transect_num"])

        # Helper function for sorting
        def _sort_mixed(arr: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    np.concatenate(
                        [
                            np.sort(
                                row[np.array([not isinstance(v, str) for v in row])].astype(float)
                            ),
                            np.sort(row[np.array([isinstance(v, str) for v in row])]),
                        ]
                    )
                    for row in arr
                ]
            )

        # Generate bootstrap samples of transect indices for each stratum
        transect_samples = [
            _sort_mixed(
                np.array(
                    [
                        self.rng.choice(
                            # transect_df.loc[j, "transect_num"].values,
                            transect_df.loc[[j], "transect_num"].to_numpy(),
                            size=self.strata_summary.loc[j, "num_transects_to_sample"],
                            replace=False,
                        )
                        for _ in range(self.model_params["num_replicates"])
                    ]
                )
            )
            for j in transect_df.index.unique()
        ]

        # Prepare indexed data for efficient lookup
        sampled_data = transect_df.reset_index().set_index(["transect_num"])

        # Use advanced indexing to get the sampled values
        # ---- Distances
        sampled_distances = awk.Array(
            [
                [sampled_data.loc[replicate, "distance"].to_numpy() for replicate in stratum]
                for stratum in transect_samples
            ]
        )
        # ---- Areas
        sampled_areas = awk.Array(
            [
                [sampled_data.loc[replicate, "area"].to_numpy() for replicate in stratum]
                for stratum in transect_samples
            ]
        )
        # ---- Variable
        sampled_values = awk.Array(
            [
                [sampled_data.loc[replicate, self.variable].to_numpy() for replicate in stratum]
                for stratum in transect_samples
            ]
        )

        return sampled_distances, sampled_areas, sampled_values

    def _format_bootstrap_replicates(
        self,
        mean_arr_np: np.ndarray,
        total_arr_np: np.ndarray,
        variance_arr_np: np.ndarray,
        length_arr_np: np.ndarray,
        area_arr_np: np.ndarray,
    ) -> pd.DataFrame:
        """
        Format bootstrap results into a structured DataFrame.

        Organizes the various bootstrap estimators into a pivoted DataFrame with proper indexing
        and column names for downstream analysis.

        Parameters
        ----------
        mean_arr_np : np.ndarray
            Distance-weighted means for each bootstrap replicate and stratum.
        total_arr_np : np.ndarray
            Total response variable values for each bootstrap replicate and stratum.
        variance_arr_np : np.ndarray
            Distance-weighted variances for each bootstrap replicate and stratum.
        length_arr_np : np.ndarray
            Total distances for each bootstrap replicate and stratum.
        area_arr_np : np.ndarray
            Total areas for each bootstrap replicate and stratum.

        Returns
        -------
        pd.DataFrame
            Formatted DataFrame with bootstrap replicates organized by stratum and replicate
            number, containing all computed metrics.

        Notes
        -----
        The DataFrame is pivoted so that strata become columns and replicates become rows,
        facilitating downstream statistical analysis.
        """
        # Get sampling fractions to scale totals back to stratum level
        sampling_fractions = (
            self.strata_summary["num_transects_to_sample"] / self.strata_summary["transect_counts"]
        ).to_numpy()

        # Scale totals to represent full stratum estimates
        stratum_values = total_arr_np / sampling_fractions

        # Compute densities
        stratum_densities = stratum_values / self.strata_summary["area"].to_numpy()

        # Create comprehensive DataFrame with all metrics
        metrics_data = {
            self.variable: stratum_values,
            f"{self.variable}_density": stratum_densities,
            "distance": length_arr_np,
            "area": area_arr_np,
            f"distance_weighted_{self.variable}_density": mean_arr_np,
            "distance_weighted_variance": variance_arr_np,
        }

        # Combine all metrics into single DataFrame
        resamples = pd.concat(
            [
                pd.DataFrame(data).unstack().to_frame(name=name)
                for name, data in metrics_data.items()
            ],
            axis=1,
        )

        # Get proper index names
        stratification_group = self.strata_summary.index.names[0]

        # Rename the indices to include 'replicate'
        resamples.index.set_names([stratification_group, "replicate"], inplace=True)

        # Pivot to get strata as columns
        resamples_pvt = resamples.reset_index().pivot_table(
            columns=stratification_group, index="replicate"
        )

        # Update column names with actual stratum values
        # --- Create the mapping
        stratum_mapping = {
            i: stratum for i, stratum in enumerate(self.strata_summary.index.unique())
        }
        # ---- Apply the names
        resamples_pvt = resamples_pvt.rename(columns=stratum_mapping, level=1)

        return resamples_pvt

    def _compute_transect_statistics(
        self,
        data_df: pd.DataFrame,
        stratum_dim: str,
    ) -> None:
        """
        Summarize transect-level data within strata.

        Computes summary statistics for each transect including distance, area, response variable
        totals, and density metrics. Results are stored in the 'transect_summary' attribute.

        Parameters
        ----------
        data_df : pd.DataFrame
            Input DataFrame containing virtual transect data.
        stratum_dim : str
            Stratification variable (column name).

        Raises
        ------
        KeyError
            If required columns for distance or area calculation are missing.

        Notes
        -----
        This method handles multiple approaches for calculating transect distances and areas,
        falling back to coordinate-based calculations if pre-computed values are not available.
        """
        # Create a grouped DataFrame from the DataFrame input
        grouped_df = data_df.groupby([stratum_dim, "transect_num"], observed=True)

        # Sum the variable for each stratum contained within the input dataset
        survey_values = grouped_df[self.variable].sum().to_frame()

        # Start with response variable sums
        survey_values = grouped_df[self.variable].sum().to_frame()

        # Handle distance calculation
        if "transect_distance" in data_df.columns:
            survey_values["distance"] = grouped_df["transect_distance"].sum()
        # If not already present, compute the distance now and generate the output
        elif all(col in data_df.columns for col in ["longitude", "latitude"]):
            survey_values["distance"] = grouped_df[["latitude", "longitude"]].apply(
                lambda x: geopy.distance.distance(
                    (x.latitude.min(), x.longitude.min()), (x.latitude.max(), x.longitude.max())
                ).nm
            )
        # Cannot proceed
        else:
            raise KeyError(
                "Input DataFrame must contain either 'transect_distance' or "
                "both 'longitude' and 'latitude' columns for distance calculation."
            )

        # Handle area calculation
        if "transect_area" in data_df.columns:
            survey_values["area"] = grouped_df["transect_area"].sum()
        # If not already present, compute the area now and generate the output
        elif "transect_spacing" in data_df.columns:
            survey_values["area"] = (
                grouped_df["transect_spacing"].mean() * survey_values["distance"]
            )
        # Cannot proceed
        else:
            raise KeyError(
                "Input DataFrame must contain either 'transect_area' or "
                "'transect_spacing' for area calculation."
            )

        # Calculate areal density
        survey_values[f"{self.variable}_areal_density"] = (
            survey_values[self.variable] / survey_values["area"]
        )

        # Calculate the line density
        survey_values[f"{self.variable}_distance_density"] = (
            survey_values[self.variable] / survey_values["distance"]
        )

        # Store results
        self.transect_summary = survey_values

    def _compute_strata_statistics(
        self,
        stratum_dim: str,
    ) -> None:
        """
        Summarize data at the stratum level.

        Computes stratum-level statistics including transect counts, sampling fractions, totals,
        and density metrics. Results are stored in the strata_summary attribute.

        Parameters
        ----------
        stratum_dim : str
            Stratification variable (column name).

        Notes
        -----
        The number of transects to sample per stratum is determined by the
        'strata_transect_proportion' parameter, which is applied to the total transect count in
        each stratum.
        """
        # Get model parameters
        mp = self.model_params

        # Create grouped data by strata
        grouped_df = self.transect_summary.groupby([stratum_dim], observed=True)

        # Start with transect counts
        strata_summary = grouped_df[self.variable].count().to_frame("transect_counts")

        # Calculate number of transects to sample per stratum
        strata_summary["num_transects_to_sample"] = utils.round_half_up(
            strata_summary["transect_counts"] * mp["strata_transect_proportion"]
        ).astype(int)

        # Ensure at least 1 transect is sampled per stratum
        strata_summary["num_transects_to_sample"] = np.maximum(
            strata_summary["num_transects_to_sample"], 1
        )

        # Calculate stratum totals
        strata_summary[["distance", "area", self.variable]] = grouped_df[
            ["distance", "area", self.variable]
        ].sum()

        # Calculate density metrics
        strata_summary = pd.concat(
            [strata_summary]
            + [
                (strata_summary[self.variable] / strata_summary[s]).to_frame(
                    f"{self.variable}{n}density"
                )
                for s, n in zip(["distance", "area"], ["_distance_", "_"], strict=False)
            ],
            axis=1,
        )

        # Store results
        self.strata_summary = strata_summary

    def _compute_survey_statistics(self) -> None:
        """
        Compute survey-wide summary statistics.

        Calculates overall survey statistics including weighted means, variances, and coefficient
        of variation. Results are stored in the survey_summary attribute as a dictionary with
        'strata' and 'survey' keys.

        Notes
        -----
        Survey-wide statistics use area-weighted calculations to properly account for stratum size
        differences. The coefficient of variation is calculated using the ratio of weighted
        standard deviation to weighted mean.
        """
        # Get local references for cleaner code
        tdf = self.transect_summary.copy()
        sdf = self.strata_summary.copy()

        # Calculate distance-based weights for variance estimation
        distance_weights = tdf["distance"] / tdf.groupby(level=0, observed=False)["distance"].mean()

        # Calculate distance-weighted variable density
        values_per_distance = tdf[self.variable] / tdf["distance"]

        # Stratum-level distance-weighted means
        weighted_variable = (tdf[self.variable] * tdf["distance"]).groupby(
            level=0, observed=False
        ).sum() / sdf["distance"]

        # Calculate squared deviations from stratum means
        squared_deviation = (values_per_distance - weighted_variable) ** 2

        # Apply distance weights to deviations
        weighted_deviation = (
            (distance_weights**2 * squared_deviation).groupby(level=0, observed=False).sum()
        )

        # Calculate degrees of freedom with single-transect adjustment
        dof_adjustment = np.where(sdf["transect_counts"] == 1, 0, 1)

        # Calculate the effective degrees of freedom
        effective_dof = sdf["transect_counts"] * (sdf["transect_counts"] - dof_adjustment)

        # Compute stratum variances
        survey_variance = weighted_deviation / effective_dof

        # Get the weighted variance [area]
        weighted_variance = (survey_variance * sdf["area"] ** 2).sum()

        # Standard deviation
        weighted_std = np.sqrt(weighted_variance)

        # Weighted mean
        weighted_mean = (weighted_variable * sdf["area"]).sum()

        # Survey coefficient of variation
        survey_cv = weighted_std / weighted_mean

        # Define the variable names for stratum-level summary
        strata_cols = [self.variable, f"{self.variable}_density"]

        # Prepare stratum-level summary
        sdf_strata = sdf[strata_cols].unstack().to_frame().T

        # Prepare survey-level summary
        sdf_survey = pd.DataFrame(
            [
                [
                    sdf_strata.sum(axis=1).iloc[0],
                    (sdf[f"{self.variable}_density"] * sdf["area"]).sum() / sdf["area"].sum(),
                    survey_cv,
                ]
            ],
            columns=strata_cols + ["cv"],
        )

        # Store results
        self.survey_summary = {"strata": sdf_strata, "survey": sdf_survey}

    def _dof(self) -> np.ndarray:
        """
        Calculate degrees of freedom for variance estimation.

        Computes effective degrees of freedom (d.o.f) for each stratum, accounting for the case
        where only one transect is sampled (which would give zero d.o.f).

        Returns
        -------
        np.ndarray
            Array of degrees of freedom values for each stratum.

        Notes
        -----
        For strata with only one transect to sample, the degrees of freedom is set to 1 to avoid
        division by zero in variance calculations.
        """
        # Adjustment for single-transect strata
        sample_offset = np.where(self.strata_summary["num_transects_to_sample"] == 1, 0, 1)

        # Calculate effective sample size/degrees of freedom
        return self.strata_summary["num_transects_to_sample"] * (
            self.strata_summary["num_transects_to_sample"] - sample_offset
        )

    @staticmethod
    def _compute_variance(
        sampled_values: awk.Array,
        sampled_distances: awk.Array,
        mean_values: awk.Array,
        stratified_weights: awk.Array,
        sample_dof: np.ndarray,
    ) -> awk.Array:
        """
        Compute distance-weighted variance using bootstrap samples.

        Calculates variance estimates for each bootstrap replicate using distance-weighted
        deviations from the mean.

        Parameters
        ----------
        sampled_values : awk.Array
            Bootstrap samples of response variable values.
        sampled_distances : awk.Array
            Bootstrap samples of transect distances.
        mean_values : awk.Array
            Distance-weighted means for each replicate.
        stratified_weights : awk.Array
            Distance-based weights for variance calculation.
        sample_dof : np.ndarray
            Degrees of freedom for each stratum.

        Returns
        -------
        awk.Array
            Variance estimates for each bootstrap replicate and stratum.

        Notes
        -----
        This method implements the distance-weighted variance calculation used in acoustic survey
        analysis, where variance is weighted by the square of the distance weights.
        """
        # Convert values to per-distance basis, handling division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            values_adjusted = awk.where(
                sampled_distances != 0, sampled_values / sampled_distances, 0.0
            )

        # Calculate squared deviations from mean
        squared_deviation = (values_adjusted - mean_values[..., None]) ** 2

        # Apply distance weights squared
        weighted_squared_deviation = awk.sum(stratified_weights**2 * squared_deviation, axis=-1)

        # Return variance (weighted sum divided by DOF)
        return weighted_squared_deviation / awk.Array(sample_dof)[:, None]

    def stratified_bootstrap(
        self,
        data: pd.DataFrame,
        stratum_dim: str,
        variable: str,
    ) -> None:
        """
        Perform stratified bootstrap resampling analysis.

        Executes the complete bootstrap analysis pipeline including transect summarization, stratum
        summarization, bootstrap sample generation, and computation of bootstrap statistics.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing virtual transect data with all required columns for analysis.
        stratum_dim : str
            Column name defining the stratification (e.g., 'geostratum_inpfc').
        variable : str
            Name of the response variable column (e.g., 'biomass', 'abundance').

        Notes
        -----
        This method populates the 'bootstrap_replicates' attribute with bootstrap replicates
        organized by stratum and replicate. The bootstrap sampling is done without replacement
        within each stratum.

        The method performs the following steps:

        1. Summarize transect-level data

        2. Summarize stratum-level data

        3. Generate bootstrap sample arrays

        4. Compute distance-weighted means and variances

        5. Format results into structured DataFrame
        """
        # Store the response variable name
        self.variable = variable

        # Summarize transect data within strata
        self._compute_transect_statistics(data, stratum_dim)

        # Summarize at stratum level
        self._compute_strata_statistics(stratum_dim)

        # Summarize at the survey level
        self._compute_survey_statistics()

        # Calculate degrees of freedom for variance calculations
        sample_dof = self._dof()

        # Generate bootstrap sample arrays
        sampled_distances, sampled_areas, sampled_values = self._prepare_bootstrap_arrays()

        # Compute distance-based stratified weights, handling division by zero
        mean_distances = awk.mean(sampled_distances, axis=-1, keepdims=True)

        # Calculate the stratified weights
        with np.errstate(divide="ignore", invalid="ignore"):
            stratified_weights = awk.where(
                mean_distances != 0, sampled_distances / mean_distances, 0.0
            )

        # Compute total distance for each bootstrap replicate
        total_distances = awk.sum(sampled_distances, axis=-1)

        # Compute the weighted mean value for each stratum
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_arr = awk.where(
                total_distances != 0,
                awk.sum(sampled_values * sampled_distances, axis=-1) / total_distances,
                0.0,
            )

        # Compute distance-weighted variances
        variance_arr = self._compute_variance(
            sampled_values, sampled_distances, mean_arr, stratified_weights, sample_dof
        )

        # Gather all arrays into list
        arrays_to_convert = [
            total_distances,  # Total distances (reuse computed value)
            mean_arr,  # Distance-weighted means
            awk.sum(sampled_areas, axis=-1),  # Total areas
            awk.sum(sampled_values, axis=-1),  # Total values
            variance_arr,  # Variances
        ]

        # Convert awkward arrays to numpy and transpose for DataFrame format
        length_arr_np, mean_arr_np, area_arr_np, total_arr_np, variance_arr_np = [
            awk.to_numpy(arr).T for arr in arrays_to_convert
        ]

        # Format and store bootstrap replicates
        self.bootstrap_replicates = self._format_bootstrap_replicates(
            mean_arr_np, total_arr_np, variance_arr_np, length_arr_np, area_arr_np
        )

    def summarize(
        self,
        ci_percentile: float = 0.95,
        ci_method: Literal[
            "bc", "bca", "empirical", "normal", "percentile", "t", "t-jackknife"
        ] = "t-jackknife",
    ) -> pd.DataFrame:
        """
        Generate summary statistics and confidence intervals from bootstrap results.

        Computes confidence intervals for both stratum-level and survey-wide estimates using the
        specified bootstrap method.

        Parameters
        ----------
        ci_percentile : float, default 0.95
            Confidence level for interval estimation (between 0 and 1).
        ci_method : {"bc", "bca", "empirical", "normal", "percentile", "t", "t-jackknife"},
        default "t-jackknife"
            Bootstrap confidence interval method:

            - "bc": Bias-corrected

            - "bca": Bias-corrected and accelerated

            - "empirical": Empirical bootstrap

            - "normal": Normal approximation

            - "percentile": Simple percentile method

            - "t": t-distribution based

            - "t-jackknife": Jackknife studentized (recommended)

        Returns
        -------
        pd.DataFrame
            DataFrame containing confidence intervals and summary statistics for both stratum-level
            and survey-wide estimates (including the coefficient of variation). Includes columns
            for confidence bounds, means, and bias estimates. These quantities are calculated using
            the area-weighted stratified approach.

        Raises
        ------
        RuntimeError
            If stratified_bootstrap() has not been run prior to calling this method.

        Notes
        -----
        The method computes area-weighted survey statistics and includes coefficient of variation
        estimates. The t-jackknife method is recommended as it provides better coverage properties
        for small sample sizes common in survey data.

        Examples
        --------
        >>> summary_stats = analysis.summarize(
        ...     ci_percentile=0.95,
        ...     ci_method='t-jackknife'
        ... )
        >>> print(summary_stats)
        """
        # Validate that bootstrap has been run
        if self.bootstrap_replicates is None:
            raise RuntimeError(
                "Must run `stratified_bootstrap()` before calling summarize(). The "
                "'bootstrap_replicates' attribute is None."
            )

        # Get bootstrap replicates
        bdf = self.bootstrap_replicates.copy()

        # Compute area-weighted survey statistics
        area_weights = self.strata_summary["area"]

        # Area-weighted variance and standard deviation
        weighted_variance = (bdf["distance_weighted_variance"] * area_weights**2).sum(axis=1)
        weighted_stdev = np.sqrt(weighted_variance)

        # Area-weighted mean
        weighted_mean = (bdf[f"distance_weighted_{self.variable}_density"] * area_weights).sum(
            axis=1
        )

        # Coefficient of variation
        bootstrap_cv = weighted_stdev / weighted_mean

        # Add survey-wide statistics to bootstrap DataFrame
        self.bootstrap_replicates = bdf.assign(
            area_weighted_variance=weighted_variance,
            **{f"area_weighted_{self.variable}": weighted_mean},
            cv=bootstrap_cv,
        )

        # Prepare data for confidence interval calculations
        ci_variables = [self.variable, f"{self.variable}_density"]

        # Compute confidence intervals for stratum-level estimates
        strata_ci_estimates = statistics.confidence_interval(
            bootstrap_samples=self.bootstrap_replicates[ci_variables],
            population_values=self.survey_summary["strata"],
            ci_method=ci_method,
            ci_percentile=ci_percentile,
        )

        # Prepare survey-wide bootstrap estimates
        survey_bootstrap = pd.DataFrame(
            {
                self.variable: self.bootstrap_replicates[self.variable].sum(axis=1),
                f"{self.variable}_density": (
                    (self.bootstrap_replicates[f"{self.variable}_density"] * area_weights).sum(
                        axis=1
                    )
                    / area_weights.sum()
                ),
                "cv": self.bootstrap_replicates["cv"],
            }
        )

        # Compute confidence intervals for survey-wide estimates
        survey_ci_estimates = statistics.confidence_interval(
            bootstrap_samples=survey_bootstrap,
            population_values=self.survey_summary["survey"],
            ci_method=ci_method,
            ci_percentile=ci_percentile,
        )

        # Set survey index name
        survey_ci_estimates.index = ["survey"]

        # Combine stratum and survey results
        ci_estimates = pd.concat([strata_ci_estimates, survey_ci_estimates])

        return ci_estimates
