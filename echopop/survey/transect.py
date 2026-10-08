"""Transect-level data processing and aggregation for acoustic survey analysis."""

import numpy as np
import pandas as pd


def compute_interval_distance(
    nasc_data: pd.DataFrame,
    interval_threshold: float = 0.05,
) -> None:
    """
    Calculate along-transect interval distances and add to DataFrame.

    Parameters
    ----------
    nasc_data : pd.DataFrame
        DataFrame containing NASC data with distance and spacing information. Must contain columns:
        'distance_s', 'distance_e', 'transect_spacing'
    interval_threshold : float, default 0.05
        Along-transect interval threshold for detecting erroneous values. Values that deviate from
        the median interval by more than this threshold will be replaced with the median interval.

    Returns
    -------
    None
        Modifies the input DataFrame in place by adding a 'distance_interval' column containing
        along-transect interval distances.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'distance_s': [0, 1, 2, 3],
    ...     'distance_e': [1, 2, 3, 4],
    ...     'transect_spacing': [0.1, 0.1, 0.1, 0.1]
    ... })
    >>> compute_interval_distance(df)
    >>> 'distance_interval' in df.columns
    True

    Notes
    -----
    This function calculates the along-track transect interval length. It identifies and corrects
    potentially erroneous values at transect endpoints by comparing intervals to the median and
    replacing outliers with the median of all calculated intervals.

    The interval calculation uses diff(periods=-1) to compute forward differences, making each
    interval represent the distance to the next measurement point.
    The final interval is initially calculated as distance_e - distance_s and is also replaced
    with the median if it exceeds the deviation threshold.
    """
    # Calculate the along-transect interval distance
    # ---- Use forward difference to get distance to next point
    nasc_data["distance_interval"] = nasc_data["distance_s"].diff(periods=-1).abs()

    # Handle the final interval (no next point available)
    # ---- Use distance_e - distance_s for the last measurement
    nasc_data.loc[nasc_data.index[-1], "distance_interval"] = (
        nasc_data["distance_e"].iloc[-1] - nasc_data["distance_s"].iloc[-1]
    )

    # Identify and correct erroneous interval lengths
    # ---- Calculate median interval for comparison
    median_interval = np.nanmedian(nasc_data["distance_interval"])

    # Find intervals that deviate significantly from median
    deviation_mask = np.abs(nasc_data["distance_interval"] - median_interval) > interval_threshold

    # Replace erroneous intervals with median distance interval
    nasc_data.loc[deviation_mask, "distance_interval"] = median_interval
