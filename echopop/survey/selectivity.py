"""Correct biological distributions based on length-based net selectivity."""

import numpy as np
import pandas as pd

from ..validators import ValidateSelectivityParams


def get_l50_sr(
    params: dict[str, float],
) -> tuple[float, float]:
    """
        Validate and convert selectivity parameters to an (L50, SR) pair.

    Parameters
    ----------
        params : dict
            A dictionary containing exactly one of the following parameter sets:

            - Logistic coefficients: ``{'slope', 'intercept'}``

            - Selection parameters: ``{'l50', 'sr'}``

    Returns
    -------
        l50 : float
            The length at 50% retention.
        sr : float
            The selection range (length interval between 25% and 75% retention).
    """
    # Define internal constant
    _K = 2.0 * np.log(3.0)

    # Validate and normalize all net selectivity parameters to (L50, SR)
    validated = ValidateSelectivityParams.create(**params)
    # ---- If L50 already present
    if validated.get("l50") is not None:
        return float(validated["l50"]), float(validated["sr"])
    # ---- Make conversion
    else:
        slope = float(validated["slope"])
        intercept = float(validated["intercept"])
        return -intercept / slope, _K / slope


def assign_selectivity_expansion(
    biodata: pd.DataFrame,
    net_selectivity_params: dict[str, float] | dict[str, dict[str, float]],
    net_column: str,
    minimum_selectivity: float = 1e-12,
) -> pd.DataFrame:
    r"""
    Compute per-fish logistic selectivity expansion factors and add them as a new column.

    The caller is responsible for ensuring ``biodata`` already contains both a length column and a
    gear column (e.g., via a prior merge of the individual-level and haul-level frames). This
    function only performs the selectivity computation.

    Parameters
    ----------
    biodata : pd.DataFrame
        Individual-level DataFrame containing at least ``net_column`` and 'length'.
    net_selectivity_params : dict
        A dictionary defining the selectivity parameters. This can take two shapes:

        **1. Global (single parameter set)**
            Applied uniformly to every row in `biodata`.
            Example: ``{'l50': 12.5, 'sr': 3.0}``

        **2. Gear-Specific (multiple net-types with unique parameters)**
            A mapping of net types to their respective parameter dictionaries.
            Example:

            .. code-block:: python

                {
                    'AWT': {'l50': 15.0, 'sr': 4.5},
                    'IKMT': {'intercept': -10.2, 'slope': 0.8}
                }

        Each parameter set must consist of either **selection parameters** (``'l50'``, ``'sr'``) or
        **logistic regression coefficients** (``'slope'``, ``'intercept'``). When a row's net-type
        is not found in ``net_selectivity_params``, its selectivity expansion (:math:`S(L)`) will be
        ``NaN``.

    net_column : str
        Column identifying the net-type for each individual.
    minimum_selectivity : float, default 1e-12
        Lower bound applied to ``S(L)`` before inversion to prevent division by zero.

    Returns
    -------
    pd.DataFrame
        Copy of ``biodata`` with the following columns added:

        - ``l50`` : The length-at-50%-retention applied to that row.

        - ``sr`` : The selection range applied to that row.

        - ``selectivity_expansion`` : The calculated expansion factor (:math:`\\phi(L)`).

    Raises
    ------
    KeyError
        If `net_column` or 'length' are missing from `biodata`.
    ValueError
        If the parameter dictionary contains mixed or incomplete sets.

    Notes
    -----
    The probability of retention :math:`S(L)` for a fish of length :math:`L` is
    modeled using the logistic selection ogive:

    .. math::

        S(L) = \\left[ 1 + \\exp\\left( \\frac{2\\ln(3)(L_{50} - L)}{SR} \\right) \\right]^{-1}

    If the user provides logistic regression coefficients (intercept :math:`\\alpha`
    and slope :math:`\\beta`), they are converted to selection parameters via:

    .. math::

        L_{50} = -\\frac{\\alpha}{\\beta}, \\quad SR = \\frac{2\\ln(3)}{\\beta}

    The resulting expansion factor :math:`\\phi(L)` is the inverse of the
    retention probability, capped by a minimum threshold :math:`S_{\\min}`:

    .. math::

        \\phi(L) = \\frac{1}{\\max(S(L), S_{\\min})},

    where :math:`S_{\\min}` is the minimum selectivity used for computational purposes.

    References
    ----------
    Wileman, D. A., Ferro, R. S. T., Fonteyne, R., & Millar, R. B. (Eds.). (1996). *Manual of
    methods of measuring the selectivity of towed fishing gears*. ICES Cooperative Research Report
    No. 215.
    """
    # Validate DataFrame columns
    if net_column not in biodata.columns:
        raise KeyError(f"Column '{net_column}' not found in 'biodata'.")
    if "length" not in biodata.columns:
        raise KeyError("Column 'length' not found in 'biodata'.")

    # Create DataFrame copy
    biodata = biodata.copy()

    # Define internal constant
    _K = 2.0 * np.log(3.0)

    # Detect whether dictionary is nested (multiple net-types) or flat (single net-type)
    is_nested = any(isinstance(v, dict) for v in net_selectivity_params.values())

    # Flattened -- single net-type
    if not is_nested:
        # ---- Validate
        l50, sr = get_l50_sr(net_selectivity_params)
        # ---- Apply to all rows
        biodata["l50"] = l50
        biodata["sr"] = sr
    # Multiple net-types
    else:
        # ---- Validate and normalize all net selectivity parameters to (L50, SR)
        net_l50_sr = {}
        # ---- Iterate across all net-types
        for net_name, input_params in net_selectivity_params.items():
            net_l50_sr[net_name] = get_l50_sr(input_params)
        # ---- L50
        biodata["l50"] = biodata[net_column].map(
            lambda n: net_l50_sr[n][0] if n in net_l50_sr else np.nan
        )
        # ---- SR
        biodata["sr"] = biodata[net_column].map(
            lambda n: net_l50_sr[n][1] if n in net_l50_sr else np.nan
        )

    # Compute retention probability S(L)
    selectivity = 1.0 / (1.0 + np.exp(_K * (biodata["l50"] - biodata["length"]) / biodata["sr"]))

    # Apply lower bound threshold to avoid division by zero
    bounded_selectivity = np.maximum(selectivity, minimum_selectivity)

    # Calculate expansion weight (1 / S(L))
    biodata["selectivity_expansion"] = 1.0 / bounded_selectivity

    # Return the annotated DataFrame
    return biodata
