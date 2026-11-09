"""
FEAT general-purpose workflow utility functions.

Provides spatial, interpolation, and data-manipulation helpers used across FEAT survey workflows,
including transect extent extraction, kriging mesh preparation, and adaptive nearest-neighbor
search utilities.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import interpolate

from echopop import utils


def get_survey_western_extents(
    transects: pd.DataFrame,
    coordinate_names: tuple[str, str],
    latitude_threshold: float,
) -> pd.DataFrame:
    """
    Get the western extents of each survey transect line.

    Get the western extents of each survey transect that can be used to constrain the adaptive
    nearest neighbors search algorithm incorporated into the kriging interpolation algorithm.

    Parameters
    ----------
    transects : pd.DataFrame
        A dataframe containing georeferenced coordinates associated with a particular variable (e.g.
        biomass). This DataFrame must have at least two valid columns comprising the overall 2D
        coordinates (e.g. 'x' and 'y'). Furthermore, this function requires that `transects`
        also contain a column called 'latitude'.
    coordinates_names : Tuple[str, str], default = ('x', 'y')
        A tuple containing the 'transects' column names defining the coordinates. The order of
        this input matters where they should be defined as the (horizontal axis, vertical axis).
    latitude_threshold : float
        A threshold that is applied to the georeferenced coordinates that further constrains any
        extrapolation that occurs during the kriging analysis.

    Returns
    -------
    pd.DataFrame
        A DataFrame comprising three columns: 'transect_num', and the column names supplied by
        the argument `coordinate_names`.
    """
    # Apply the latitude filter
    transect_thresholded = transects.loc[transects["latitude"] < latitude_threshold]

    # Parse the western-most coordinate indices of each transect
    western_extent_idx = transect_thresholded.groupby(["transect_num"])[
        coordinate_names[0]
    ].idxmin()

    # Subset the DataFrame
    transect_western_extent = transect_thresholded.loc[western_extent_idx].reset_index(drop=True)

    # Return the reduced DataFrame
    return transect_western_extent.filter(["transect_num", *coordinate_names])


def western_boundary_search_strategy(
    kriging_mesh: pd.DataFrame,
    western_extent: pd.DataFrame,
    coordinate_names: tuple[str, str],
    sparse_radii: np.ndarray[int],
    valid_distances: np.ndarray[int],
    local_points: np.ndarray[float],
    distance_matrix_masked: np.ndarray[float],
    nearby_indices: np.ndarray[int],
    k_min: int,
    k_max: int,
    search_radius: float,
    wr_indices: np.ndarray[int],
    oos_indices: np.ndarray[np.number],
    oos_weights: np.ndarray[float],
    **kwargs,
) -> tuple[np.ndarray[np.number], np.ndarray[np.number], np.ndarray[np.number]]:
    """
    Search strategy that applies western boundary constraints for transect-based surveys.

    Parameters
    ----------
    kriging_mesh : pd.DataFrame
        Kriging mesh used for interpolated data values via geostatistics.
    western_extent : pd.DataFrame
        DataFrame with the western-most extent of each transect line used for re-weighting the
        out-of-sample/extrapolated kriged values.
    coordinate_names : Tuple[str, str]
        Names of the coordinate columns when using DataFrames. Expected format: (x_col, y_col).
    sparse_radii : np.ndarray[int]
        Indices where there are fewer than `k_min` nearest neighbors.
    valid_distances : np.ndarray[int]
        The number of masked distance matrix values where extrapolation is required.
    local_points : np.ndarray[float]
        An array with the sorted distances (from nearest to furthest) relative to each point.
    distance_matrix_masked : np.ndarray[float]
        An array with the search-radius-masked nearest neighbor distances.
    nearby_indices : np.ndarray[int]
        Indices of points that require extrapolation.
    k_min : int
        The minimum number of nearest neighbors required for including values for kriging within
        the search radius.
    k_max : int
        The maximum number of nearest neighbors required for including values for kriging detected
        within the search radius.
    search_radius : float
        The adaptive search radius that identifies the *k*-nearest neighbors around each
        georeferenced value that are subsequently kriged.
    wr_indices : np.ndarray[int]
        Indices of within-radius (WR) (i.e. < `k_max`) points.
    oos_indices : np.ndarray[np.number]
        Template array based on the size of the data input and `k_min` that will contain indices
        where extrapolation is required where there are fewer than `k_min` nearest neighbors.
    oos_weights : np.ndarray[float]
        Weights applied to extraplolated values.

    Returns
    -------
    Tuple[np.ndarray[np.number], np.ndarray[np.number], np.ndarray[np.number]]
        A tuple with updated values for `wr_indices`, `oos_indices`, and `oos_weights` via a
        search strategy that uses an extrapolation re-weighting based on transect extents.
    """
    # Parse ordered coordinate names
    x_name, y_name = coordinate_names

    # Index the mesh grid coordinates for bounding the search radius expansion/extrapolation
    # ---- y-coordinates with array transformation to access matrix operations
    mesh_y = kriging_mesh[y_name].to_numpy()[sparse_radii].reshape(-1, 1)
    # ---- x-coordinates
    mesh_x = kriging_mesh[x_name].to_numpy()[sparse_radii]

    # Calculate the mesh distance from the western boundary of the survey transects
    # ---- Find closest point
    mesh_western_distance = np.abs(mesh_y - western_extent["y"].to_numpy()).argmin(axis=1)
    # ---- Calculate the western limits (x-axis)
    western_limit = western_extent.iloc[np.ravel(mesh_western_distance)][x_name]
    # ---- Compute bounding threshold (for tapered extrapolation function)
    western_threshold = western_limit - search_radius
    # ---- Create a thresholded mask for lazy operations
    western_limit_mask = mesh_x < western_threshold

    # Adjust values that don't fall outside the western extent
    if np.any(~western_limit_mask):
        # ---- Grab all values that don't fall outside the western extent
        soft_extrapolation_index = sparse_radii[~western_limit_mask]
        # ---- Find the local points where there are at least some valid points
        partial_indices = soft_extrapolation_index[valid_distances[soft_extrapolation_index] > 0]
        # ---- Update the current values in `wr_indices`
        if len(partial_indices) > 0:
            # -------- Create boolean mask for within-range/sample points
            wr_mask = local_points[partial_indices, :k_max] < search_radius
            # -------- Create temporary matrix for within-range samples
            wr_tmp = wr_indices[partial_indices].copy()
            # -------- Create temporary matrix for oos samples
            oos_tmp = wr_indices[partial_indices].copy()
            # -------- Update temporary matrix by applying `wr_mask` for wr points
            wr_tmp[~wr_mask] = np.nan
            # -------- Update temporary matrix by applying `wr_mask` for oos points
            oos_tmp[wr_mask] = np.nan
            # -------- Assign the OOS values to `oos_indices`
            oos_indices[partial_indices] = np.sort(oos_tmp[:, :k_min])
            # -------- Apply the mask to the remaining `wr_indices` values
            wr_indices[partial_indices] = np.sort(wr_tmp[:, :k_max])

        # ---- Find the local points where there are no valid points within the search radius
        full_extrap_indices = soft_extrapolation_index[
            valid_distances[soft_extrapolation_index] == 0
        ]
        if len(full_extrap_indices) > 0:
            # -------- Update `oos_indices`
            oos_indices[full_extrap_indices] = wr_indices[full_extrap_indices, :k_min]
            # -------- Update `wr_indices`
            wr_indices[full_extrap_indices] = np.nan

    # Taper function for extrapolating values outside the search radius
    if np.any(western_limit_mask):
        # ---- Index these values
        extrapolation_index = sparse_radii[western_limit_mask]
        # ---- Compute the OOS kriging weights
        oos_mean = np.apply_along_axis(np.nanmean, 1, local_points[extrapolation_index, :k_min])
        # ---- Exponentiate the OOS mean
        oos_exp = np.exp(-oos_mean / search_radius)
        # ---- Update the OOS weights
        oos_weights[extrapolation_index] = oos_exp
        # ---- Get the outside indices that correspond to this tapered extrapolation
        sparse_extrapolation_index = nearby_indices[western_limit_mask].astype(float)
        # ---- Apply indices as a mask to the NaN-masked distance matrix
        extrapolated_distance = np.take_along_axis(
            distance_matrix_masked[sparse_radii][western_limit_mask],
            sparse_extrapolation_index.astype(int),
            axis=1,
        )
        # ---- Create NaN mask
        extrapolated_nan_mask = ~np.isnan(extrapolated_distance)
        # -------- Apply mask to indices
        sparse_extrapolation_index_nan = sparse_extrapolation_index.copy()
        sparse_extrapolation_index_nan[extrapolated_nan_mask] = np.nan
        # -------- Update `out_of_sample_indices` matrix
        oos_indices[extrapolation_index] = np.sort(sparse_extrapolation_index_nan)
        # ---- Get inside indices that apply to these points
        # -------- Create NaN mask for within-sample values
        interpolated_nan_mask = np.isnan(extrapolated_distance)
        # -------- Apply mask to indices
        sparse_interpolation_index_nan = sparse_extrapolation_index.copy()
        sparse_interpolation_index_nan[interpolated_nan_mask] = np.nan
        # -------- Pad NaN to match `within_sample_indices` matrix
        sparse_interpolation_pad = np.pad(
            sparse_interpolation_index_nan,
            [(0, 0), (0, k_max - k_min)],
            mode="constant",
            constant_values=np.nan,
        )
        # -------- Updated `within_sample_indices` matrix
        wr_indices[extrapolation_index] = np.sort(sparse_interpolation_pad)

    # Return Tuple
    return wr_indices, oos_indices, oos_weights


def convert_afsc_nasc_to_feat(
    nasc_data: pd.DataFrame,
    default_interval_distance: float = 0.5,
    default_transect_spacing: float = 10.0,
    inclusion_filter: dict[str, Any] = None,
    exclusion_filter: dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Convert AFSC-MACE to NWFSC-FEAT transect NASC format.

    Parameters
    ----------
    nasc_data : pd.DataFrame
        DataFrame containing AFSC NASC data.
    default_interval_distance : float, optional
        Default distance interval for transects, by default 0.5.
    default_transect_spacing : float, optional
        Default transect spacing, by default 10.0.
    inclusion_filter : Dict[str, Any], optional
        Filter to include specific rows based on column values, by default {}.
    exclusion_filter : Dict[str, Any], optional
        Filter to exclude specific rows based on column values, by default {}.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame corresponding to the expected NWFSC-FEAT format.
    """
    # Apply inclusion filter if provided
    if exclusion_filter is None:
        exclusion_filter = {}
    if inclusion_filter is None:
        inclusion_filter = {}
    df = utils.apply_filters(
        nasc_data, include_filter=inclusion_filter, exclude_filter=exclusion_filter
    )

    # Create distance intervals
    df = df.rename(columns={"distance": "distance_s"})

    # End of interval
    df["distance_e"] = df["distance_s"] + default_interval_distance

    # Replace NaN values in transect_spacing with the default value
    df["transect_spacing"] = df["transect_spacing"].fillna(default_transect_spacing)

    return df


def filter_transect_intervals(
    nasc_data: pd.DataFrame,
    transect_filter: pd.DataFrame | Path,
    survey_filter: str | None = None,
    transect_filter_sheet: str | None = None,
) -> pd.DataFrame:
    """
    Filter transect intervals based on log start and end values.

    Parameters
    ----------
    nasc_data : pandas.DataFrame
        DataFrame containing NASC data with columns 'transect_num', 'distance_s', and 'distance_e'
    transect_filter : Union[pandas.DataFrame, Path]
        DataFrame containing transect filter data with columns 'transect_num', 'log_start',
        and 'log_end', or a filepath that reads in a file.
    survey_filter : str, optional
        Query survey string to filter ``transect_filter`` (e.g., "region_id == 'A'")
    transect_filter_sheet : str, optional
        Optional sheetname if a filename is input

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only rows that overlap with the specified transect intervals

    Examples
    --------
    >>> nasc_data = pd.DataFrame({
    ...     'transect_num': [1, 1, 2, 2],
    ...     'distance_s': [0, 1, 0, 1],
    ...     'distance_e': [1, 2, 1, 2],
    ...     'nasc': [10, 20, 30, 40]
    ... })
    >>> filter_data = pd.DataFrame({
    ...     'transect_num': [1, 2],
    ...     'log_start': [0.5, 0.5],
    ...     'log_end': [1.5, 1.5],
    ...     'region_id': ['A', 'B']
    ... })
    >>> result = filter_transect_intervals(nasc_data, filter_data)
    """
    # Make copies to avoid modifying the inputs
    nasc_df = nasc_data.copy()

    # Read in transect filter file
    if isinstance(transect_filter, Path):
        # Read in the defined file
        transect_filter = pd.read_excel(
            transect_filter, sheet_name=transect_filter_sheet, index_col=None, header=0
        )

    # Lowercase column names in transect filter DataFrame
    transect_filter.columns = transect_filter.columns.str.lower()

    # Rename 'transect' to 'transect_num' if it exists
    if "transect" in transect_filter.columns and "transect_num" not in transect_filter.columns:
        transect_filter = transect_filter.rename(columns={"transect": "transect_num"})

    # Validation check
    if not set(transect_filter.columns) >= set(["transect_num", "log_start", "log_end"]):
        # ---- Find mismatch
        missing = list(set(["transect_num", "log_start", "log_end"]) - set(transect_filter.columns))
        # ---- Format
        missing_str = [f"'{v}'" for v in missing]
        # ---- Print
        raise KeyError(
            f"The following columns could not be formatted from 'transect_filter': "
            f"{', '.join(missing_str)}."
        )

    # Apply a filter, if needed
    if survey_filter is not None:
        # Extract tokens from string
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", survey_filter)

        # Provide typical Python operator keywords
        keywords = {"and", "or", "not", "in", "notin", "True", "False"}

        # Check for column names
        column_names = [
            t
            for t in tokens
            if t not in keywords and not t.isnumeric() and t in transect_filter.columns
        ]

        # Check if all referenced columns exist
        missing = [col for col in column_names if col not in transect_filter.columns]

        # Raise error, if needed
        if missing:
            raise ValueError(f"Invalid column(s): {', '.join(missing)}")
        else:
            transect_filter = transect_filter.query(survey_filter).sort_values(["transect_num"])

    # Sort transect filter by vessel log distance start values
    transect_filter = transect_filter.sort_values("log_start").reset_index(drop=True)

    # Get arrays for easier processing
    filter_transect_nums = transect_filter["transect_num"].values
    filter_log_starts = transect_filter["log_start"].values
    filter_log_ends = transect_filter["log_end"].values
    unique_filter_transects = np.unique(filter_transect_nums)

    # Get NASC data arrays
    nasc_transect_nums = nasc_df["transect_num"].values
    nasc_distance_starts = nasc_df["distance_s"].values
    nasc_distance_ends = nasc_df["distance_e"].values

    # Initialize removal list
    indices_to_remove = []

    # Process each unique transect that has filter intervals
    for transect in unique_filter_transects:
        # ---- Find filter interval indices for this transect
        filter_interval_indices = np.where(filter_transect_nums == transect)[0]
        num_intervals = len(filter_interval_indices)
        # ---- Find NASC data indices for this transect
        nasc_data_indices = np.where(nasc_transect_nums == transect)[0]
        # ---- Skip if empty
        if len(nasc_data_indices) == 0:
            continue
        # ---- Initialize current transect removal indices
        current_removal_indices = []
        # ---- Iterate through gaps
        if num_intervals > 1:
            # ---- Multiple intervals case
            for interval in range(num_intervals):
                if interval == 0:
                    # ---- Case 1: Remove data before first interval
                    condition_indices = np.where(
                        nasc_distance_ends[nasc_data_indices]
                        < filter_log_starts[filter_interval_indices[0]]
                    )[0]
                elif interval == num_intervals - 1:
                    # ---- Case 2: Remove data after last interval OR between last two intervals
                    condition_indices = np.where(
                        (
                            nasc_distance_starts[nasc_data_indices]
                            > filter_log_ends[filter_interval_indices[num_intervals - 1]]
                        )
                        | (
                            (
                                nasc_distance_starts[nasc_data_indices]
                                > filter_log_ends[filter_interval_indices[interval - 1]]
                            )
                            & (
                                nasc_distance_ends[nasc_data_indices]
                                < filter_log_starts[filter_interval_indices[interval]]
                            )
                        )
                    )[0]
                else:
                    # ---- Case 3: Remove data between intervals j-1 and j
                    condition_indices = np.where(
                        (
                            nasc_distance_starts[nasc_data_indices]
                            > filter_log_ends[filter_interval_indices[interval - 1]]
                        )
                        & (
                            nasc_distance_ends[nasc_data_indices]
                            < filter_log_starts[filter_interval_indices[interval]]
                        )
                    )[0]
                # ---- Add matching NASC indices to removal list
                if len(condition_indices) > 0:
                    current_removal_indices.extend(nasc_data_indices[condition_indices])
        else:
            # ---- Single interval case: Remove data before start OR after end
            condition_indices = np.where(
                (
                    nasc_distance_ends[nasc_data_indices]
                    < filter_log_starts[filter_interval_indices[0]]
                )
                | (
                    nasc_distance_starts[nasc_data_indices]
                    > filter_log_ends[filter_interval_indices[0]]
                )
            )[0]
            if len(condition_indices) > 0:
                current_removal_indices.extend(nasc_data_indices[condition_indices])
        # ---- Add all removal indices for this transect
        indices_to_remove.extend(current_removal_indices)

    # Create boolean mask for rows to keep
    keep_mask = np.ones(len(nasc_df), dtype=bool)
    keep_mask[indices_to_remove] = False

    # Return the filtered dataframe
    return nasc_df[keep_mask].reset_index(drop=True)


# Helper functions for setting E-W/N-S assignments
def region_13_conditions(x, boundary_column: str, position: str):
    """
    Map mesh regions 1 and 3 boundaries based on encoded transect boundary values.

    This helper function interprets the encoded boundary values for regions 1 and 3,
    which use latitude-parallel transects. The encoding uses decimal fractions to
    indicate spatial positions (e.g., .1 for west, .4 for east, .6 for south, .9 for north).

    Parameters
    ----------
    x : pd.DataFrame
        DataFrame subset containing transect data for a specific transect number
    boundary_column : str
        Name of the column containing the encoded boundary values
    position : str
        Position identifier ('east', 'west', 'north', 'south') used for output column naming

    Returns
    -------
    pd.Series
        Series containing longitude and latitude coordinates for the specified position,
        with column names formatted as 'longitude_{position}' and 'latitude_{position}'

    Notes
    -----
    The encoding scheme:
    - x.1: Western boundary (minimum longitude)
    - x.4: Eastern boundary (maximum longitude)
    - x.6: Southern boundary (minimum latitude)
    - x.9: Northern boundary (maximum latitude)
    """
    # Calculate the floor
    x_floor = np.round((x[boundary_column].iloc[0] % 1) * 10)

    # Assign output based on conditions
    if x_floor == 1.0:
        return pd.Series(
            {
                f"longitude_{position}": x["longitude"].min(),
                f"latitude_{position}": x["latitude"].iloc[x["longitude"].argmin()],
            }
        )
    elif x_floor == 4.0:
        return pd.Series(
            {
                f"longitude_{position}": x["longitude"].max(),
                f"latitude_{position}": x["latitude"].iloc[x["longitude"].argmax()],
            }
        )
    elif x_floor == 6.0:
        return pd.Series(
            {
                f"longitude_{position}": x["longitude"].iloc[
                    x["latitude"].argmin(), f"latitude_{position}" : x["latitude"].min()
                ]
            }
        )
    elif x_floor == 9.0:
        return pd.Series(
            {
                f"longitude_{position}": x["longitude"].iloc[
                    x["latitude"].argmax(), f"latitude_{position}" : x["latitude"].max()
                ]
            }
        )


def region_2_conditions(x, boundary_column: str, position: str):
    """
    Map mesh region 2 boundaries based on encoded transect boundary values.

    This helper function interprets the encoded boundary values for region 2,
    which uses longitude-parallel transects. The encoding uses decimal fractions to
    indicate spatial positions.

    Parameters
    ----------
    x : pd.DataFrame
        DataFrame subset containing transect data for a specific transect number
    boundary_column : str
        Name of the column containing the encoded boundary values
    position : str
        Position identifier ('north', 'south') used for output column naming

    Returns
    -------
    pd.Series
        Series containing longitude and latitude coordinates for the specified position,
        with column names formatted as 'longitude_{position}' and 'latitude_{position}'

    Notes
    -----
    The encoding scheme for region 2:
    - x.1: Western boundary (minimum longitude)
    - x.4: Eastern boundary (maximum longitude)
    - x.6: Southern boundary (minimum latitude)
    - x.9: Northern boundary (maximum latitude)

    Region 2 transects run parallel to longitudes (north-south orientation).
    """
    # Calculate the floor
    x_floor = np.round((x[boundary_column].iloc[0] % 1) * 10)

    # Assign output based on conditions
    if x_floor == 1.0:
        return pd.Series(
            {
                f"latitude_{position}": x["latitude"].iloc[x["longitude"].argmin()],
                f"longitude_{position}": x["longitude"].min(),
            }
        )
    elif x_floor == 4.0:
        return pd.Series(
            {
                f"latitude_{position}": x["latitude"].iloc[x["longitude"].argmax()],
                f"longitude_{position}": x["longitude"].max(),
            }
        )
    elif x_floor == 6.0:
        return pd.Series(
            {
                f"latitude_{position}": x["latitude"].min(),
                f"longitude_{position}": x["longitude"].iloc[x["latitude"].argmin()],
            }
        )
    elif x_floor == 9.0:
        return pd.Series(
            {
                f"latitude_{position}": x["latitude"].max(),
                f"longitude_{position}": x["longitude"].iloc[x["latitude"].argmax()],
            }
        )


def transect_ends_crop(
    transects: pd.DataFrame,
    mesh: pd.DataFrame,
    latitude_resolution: float,
    transect_mesh_region_function: Callable,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crop the kriging mesh by interpolating the eastern and western transect extents.

    Crop the kriging mesh by interpolating the eastern and western extents of survey transects
    partitioned into discrete regions via user-defined sorting functions. This function processes
    survey transect data to define spatial boundaries for mesh cropping. It divides transects into
    three regions, interpolates their boundaries, and identifies mesh cells that fall within the
    survey extent.

    Parameters
    ----------
    transects : pd.DataFrame
        Georeferenced survey transect data used for defining the spatial extent for the kriging
        mesh grid. Must contain columns: 'transect_num', 'longitude', 'latitude'.
    mesh : pd.DataFrame
        Complete kriging mesh DataFrame that is subsequently cropped. Must contain columns:
        'longitude', 'latitude'.
    latitude_resolution : float
        The latitudinal resolution (in degrees) used for the interpolation. This determines
        the spacing between interpolation points and affects the precision of boundary detection.
    transect_mesh_region_function : Callable
        A sorting function that maps specific transect numbers to their respective discretized
        mesh regions. The outputs of this function are expected to be:

        - 'transect_start' (np.number): the first transect number of a particular region,
        - 'transect_end' (np.number): the last transect number of a particular region,
        - 'transect_lower_bound' (List[np.number]): a list of encoded transect numbers indicating
          whether the transect is north, south, west, or east,
        - 'transect_upper_bound' (List[np.number]): a list of encoded transect numbers indicating
          whether the transect is north, south, west, or east.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple comprising:

        - Cropped kriging mesh DataFrame containing only cells within the survey extent
        - Annotated transect data with each transect number's respective mesh region assignment

    Examples
    --------
    >>> from echopop.nwfsc_feat.FEAT import transect_mesh_region_2019
    >>> cropped_mesh, annotated_transects = transect_ends_crop(
    ...     transects, mesh, 0.05, transect_mesh_region_2019
    ... )
    >>> print(f"Original mesh size: {len(mesh)}")
    >>> print(f"Cropped mesh size: {len(cropped_mesh)}")
    """
    # Create dictionary from user-defined transect-mesh region assignment
    mesh_region_dict = {
        region: {
            name: value
            for name, value in zip(
                ["start", "end", "upper", "lower"],
                transect_mesh_region_function(region),
                strict=False,
            )
        }
        for region in [1, 2, 3]
    }

    # Create an equivalent DataFrame for merging
    mesh_region_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "transect_num": np.arange(values["start"], values["end"] + 1),
                    "mesh_region": region,
                    "transect_lower_bound": values["lower"],
                    "transect_upper_bound": values["upper"],
                }
            )
            for region, values in mesh_region_dict.items()
        ]
    )
    # ---- Merge
    transect_df = transects.merge(mesh_region_df, on="transect_num", how="inner")
    # ---- Set indices
    transect_df = transect_df.set_index(["mesh_region"])
    mesh_region_df = mesh_region_df.set_index(["mesh_region"])

    # Generate arrays that will be used for interpolation for each region
    # ---- Region 1
    region_1_latitude = np.arange(
        transect_df.loc[1, "latitude"].min(),
        transect_df.loc[1, "latitude"].max(),
        latitude_resolution,
    )
    # ---- Region 2
    # -------- Compute longitude resolution
    longitude_resolution_region_2 = latitude_resolution * np.cos(
        np.radians(transect_df.loc[2, "latitude"].mean())
    )
    # -------- Complete the array
    region_2_longitude = np.arange(
        transect_df.loc[2, "longitude"].min(),
        transect_df.loc[2, "longitude"].max(),
        longitude_resolution_region_2,
    )
    # ---- Region 3
    # -------- Get start and end transect numbers
    t3_start = mesh_region_dict[3]["start"]
    t3_end = mesh_region_dict[3]["end"]
    # -------- Complete the array
    region_3_latitude = np.arange(
        transect_df.loc[3]
        .loc[lambda x: x["transect_num"].isin([t3_start, t3_end]), "latitude"]
        .min(),
        transect_df.loc[3]
        .loc[lambda x: x["transect_num"].isin([t3_start, t3_end]), "latitude"]
        .max(),
        latitude_resolution,
    )

    # Process region 1
    # ---- Get eastern and western extents
    region_1_extents = pd.concat(
        [
            transect_df.loc[1]
            .groupby(["transect_num"])
            .apply(region_13_conditions, "transect_lower_bound", "east", include_groups=False),
            transect_df.loc[1]
            .groupby(["transect_num"])
            .apply(region_13_conditions, "transect_upper_bound", "west", include_groups=False),
        ],
        axis=1,
    )
    # ---- Generate interpolated coordinates
    region_1_interp = pd.DataFrame(
        {
            "longitude_west": interpolate.interp1d(
                region_1_extents["latitude_west"],
                region_1_extents["longitude_west"],
                kind="linear",
                bounds_error=False,
            )(region_1_latitude),
            "longitude_east": interpolate.interp1d(
                region_1_extents["latitude_east"],
                region_1_extents["longitude_east"],
                kind="linear",
                bounds_error=False,
            )(region_1_latitude),
        }
    )
    # ---- Calculate the longitudinal interval
    delta_longitude_region_1 = latitude_resolution * np.cos(np.radians(region_1_latitude))
    # ---- Initialize Region 1 index
    region_1_index = []
    # ---- Iterate through
    for i in range(len(region_1_latitude)):
        # -------- Find the mesh indices that are within the survey extent
        idx = np.where(
            (
                mesh["longitude"]
                >= region_1_interp.loc[i, "longitude_west"] - delta_longitude_region_1[i]
            )
            & (
                mesh["longitude"]
                <= region_1_interp.loc[i, "longitude_east"] + delta_longitude_region_1[i]
            )
            & (mesh["latitude"] >= region_1_latitude[i] - latitude_resolution)
            & (mesh["latitude"] < region_1_latitude[i] + latitude_resolution)
        )
        # -------- Append the indices
        region_1_index.append(idx[0])
    # ---- Assign the unique indices
    region_1_index_unique = np.unique(np.concatenate(region_1_index))

    # Process region 2
    # ---- Get northern and southern extents
    region_2_extents = pd.concat(
        [
            transect_df.loc[2]
            .groupby(["transect_num"])
            .apply(region_2_conditions, "transect_upper_bound", "south", include_groups=False),
            transect_df.loc[2]
            .groupby(["transect_num"])
            .apply(region_2_conditions, "transect_lower_bound", "north", include_groups=False),
        ],
        axis=1,
    )
    # ---- Generate interpolated coordinates
    region_2_interp = pd.DataFrame(
        {
            "latitude_south": interpolate.interp1d(
                region_2_extents["longitude_south"],
                region_2_extents["latitude_south"],
                kind="linear",
                bounds_error=False,
            )(region_2_longitude),
            "latitude_north": interpolate.interp1d(
                region_2_extents["longitude_north"],
                region_2_extents["latitude_north"],
                kind="linear",
                bounds_error=False,
            )(region_2_longitude),
        }
    )
    # ---- Calculate the longitudinal interval
    delta_longitude_region_2 = latitude_resolution * np.cos(
        np.radians(transect_df.loc[2, "latitude"].mean())
    )
    # ---- Initialize Region 2 index
    region_2_index = []
    # ---- Iterate through
    for i in range(len(region_2_longitude)):
        # -------- Find the mesh indices that are within the survey extent: southern limit
        if ~np.isnan(region_2_interp.loc[i, "latitude_south"]) or ~np.isnan(
            region_2_interp.loc[i, "latitude_north"]
        ):
            # -------- Compute the indices for the northern- and southernmost coordinates
            # -------- North
            lon_n_min = np.argmin(
                np.abs(region_2_longitude[i] - region_2_extents["longitude_north"])
            )
            # -------- South
            lon_s_min = np.argmin(
                np.abs(region_2_longitude[i] - region_2_extents["longitude_south"])
            )
            # -------- Slope
            slope = (
                region_2_extents["latitude_north"].iloc[lon_n_min]
                - region_2_extents["latitude_south"].iloc[lon_s_min]
            ) / (
                region_2_extents["longitude_north"].iloc[lon_n_min]
                - region_2_extents["longitude_south"].iloc[lon_s_min]
            )
            # -------- Set a new border threshold
            latitude_slope_i = (
                slope
                * (region_2_longitude[i] - region_2_extents["longitude_south"].iloc[lon_s_min])
                + region_2_extents["latitude_south"].iloc[lon_s_min]
            )
            if np.isnan(region_2_interp.loc[i, "latitude_south"]):
                # -------- Find the mesh indices that are within the survey extent
                idx = np.where(
                    (mesh["longitude"] >= region_2_longitude[i] - delta_longitude_region_2)
                    & (mesh["longitude"] <= region_2_longitude[i] + delta_longitude_region_2)
                    & (mesh["latitude"] >= latitude_slope_i - latitude_resolution)
                    & (
                        mesh["latitude"]
                        < region_2_interp.loc[i, "latitude_north"] + latitude_resolution
                    )
                )
            elif np.isnan(region_2_interp.loc[i, "latitude_north"]):
                # -------- Find the mesh indices that are within the survey extent
                idx = np.where(
                    (mesh["longitude"] >= region_2_longitude[i] - delta_longitude_region_2)
                    & (mesh["longitude"] <= region_2_longitude[i] + delta_longitude_region_2)
                    & (
                        mesh["latitude"]
                        >= region_2_interp.loc[i, "latitude_south"] - latitude_resolution
                    )
                    & (mesh["latitude"] < latitude_slope_i + latitude_resolution)
                )
            else:
                # -------- Find the mesh indices that are within the survey extent
                idx = np.where(
                    (mesh["longitude"] >= region_2_longitude[i] - delta_longitude_region_2)
                    & (mesh["longitude"] <= region_2_longitude[i] + delta_longitude_region_2)
                    & (
                        mesh["latitude"]
                        >= region_2_interp.loc[i, "latitude_south"] - latitude_resolution
                    )
                    & (
                        mesh["latitude"]
                        < region_2_interp.loc[i, "latitude_north"] + latitude_resolution
                    )
                )
            # -------- Append the indices
            region_2_index.append(idx[0])
    # ---- Assign the unique indices
    region_2_index_unique = np.unique(np.concatenate(region_2_index))

    # Process region 3
    # ---- Get eastern and western extents
    region_3_extents = pd.concat(
        [
            transect_df.loc[3]
            .groupby(["transect_num"])
            .apply(region_13_conditions, "transect_upper_bound", "west", include_groups=False),
            transect_df.loc[3]
            .groupby(["transect_num"])
            .apply(region_13_conditions, "transect_lower_bound", "east", include_groups=False),
        ],
        axis=1,
    )
    # ---- Generate interpolated coordinates
    region_3_interp = pd.DataFrame(
        {
            "longitude_west": interpolate.interp1d(
                region_3_extents["latitude_west"],
                region_3_extents["longitude_west"],
                kind="linear",
                bounds_error=False,
            )(region_3_latitude),
            "longitude_east": interpolate.interp1d(
                region_3_extents["latitude_east"],
                region_3_extents["longitude_east"],
                kind="linear",
                bounds_error=False,
            )(region_3_latitude),
        }
    )
    # ---- Calculate the longitudinal interval
    delta_longitude_region_3 = latitude_resolution * np.cos(np.radians(region_3_latitude))
    # ---- Initialize Region 1 index
    region_3_index = []
    # -------- Compute the indices for the eastern- and western-most coordinates
    # -------- West
    lat_w_max = np.argmax(region_3_extents["latitude_west"])
    # -------- East
    lat_e_max = np.argmax(region_3_extents["latitude_east"])
    # -------- Slope
    slope = (
        region_3_extents.iloc[lat_w_max]["longitude_west"]
        - region_3_extents.iloc[lat_e_max]["longitude_east"]
    ) / (
        region_3_extents.iloc[lat_w_max]["latitude_west"]
        - region_3_extents.iloc[lat_e_max]["latitude_east"]
    )
    # ---- Iterate through
    for i in range(len(region_3_latitude)):
        if ~np.isnan(region_3_interp.loc[i, "longitude_west"]) or ~np.isnan(
            region_3_interp.loc[i, "longitude_east"]
        ):
            # -------- Set a new border threshold
            longitude_slope_i = (
                slope * (region_3_latitude[i] - region_3_extents.iloc[lat_e_max]["latitude_east"])
                + region_3_extents.iloc[lat_w_max]["longitude_east"]
            )
            if np.isnan(region_3_interp.loc[i, "longitude_west"]):
                # -------- Find the mesh indices that are within the survey extent
                idx = np.where(
                    (mesh["longitude"] >= longitude_slope_i - delta_longitude_region_3[i])
                    & (
                        mesh["longitude"]
                        <= region_3_interp.loc[i, "longitude_east"] + delta_longitude_region_3[i]
                    )
                    & (mesh["latitude"] >= region_3_latitude[i] - latitude_resolution)
                    & (mesh["latitude"] < region_3_latitude[i] + latitude_resolution)
                )
            elif np.isnan(region_3_interp.loc[i, "longitude_east"]):
                # -------- Find the mesh indices that are within the survey extent
                idx = np.where(
                    (
                        mesh["longitude"]
                        >= region_3_interp.loc[i, "longitude_west"] - delta_longitude_region_3[i]
                    )
                    & (mesh["longitude"] <= longitude_slope_i + delta_longitude_region_3[i])
                    & (mesh["latitude"] >= region_3_latitude[i] - latitude_resolution)
                    & (mesh["latitude"] < region_3_latitude[i] + latitude_resolution)
                )
            else:
                # -------- Find the mesh indices that are within the survey extent
                idx = np.where(
                    (
                        mesh["longitude"]
                        >= region_3_interp.loc[i, "longitude_west"] - delta_longitude_region_3[i]
                    )
                    & (
                        mesh["longitude"]
                        <= region_3_interp.loc[i, "longitude_east"] + delta_longitude_region_3[i]
                    )
                    & (mesh["latitude"] >= region_3_latitude[i] - latitude_resolution)
                    & (mesh["latitude"] < region_3_latitude[i] + latitude_resolution)
                )
            # -------- Append the indices
            region_3_index.append(idx[0])
    # ---- Assign the unique indices
    region_3_index_unique = np.unique(np.concatenate(region_3_index))

    # Concatenate the indices
    mesh_indices = np.concatenate(
        [region_1_index_unique, region_2_index_unique, region_3_index_unique]
    )

    # Return the DataFrames
    return mesh.loc[mesh_indices], transect_df.reset_index()
