"""
NASC data ingestion and preprocessing for echopop.

This module handles all ingestion pathways for Nautical Area Scattering Coefficient (NASC) data
produced by Echoview and other acoustic backscatter processing software. It supports both raw
multi-file export directories and pre-consolidated single-sheet workbooks. Key responsibilities
include merging interval and region export files, parsing region-name codes into structured labels
(species class, haul number, country), constructing transect-region-haul keys, and producing a clean
``pandas.DataFrame`` ready for stratification and population estimation.
"""

import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.echoview import (
    ECHOVIEW_DATABASE_EXPORT_FILESET,
    ECHOVIEW_EXPORT_ROW_SORT,
    ECHOVIEW_TO_ECHOPOP,
)
from ..utils.base import add_haul_uids


def map_transect_num(
    ev_export_paths: dict[str, Generator], transect_pattern: str = r"T(\d+)"
) -> pd.DataFrame:
    """
    Extract transect numbers from Echoview export filenames without using loops.

    Parameters
    ----------
    ev_export_paths : Dict[str, Generator]
        Dictionary with keys "analysis", "cells", "intervals", "layers", with values being
        `pathlib.Path.glob` generators pointing toward Echoview export *.csv files.
    transect_pattern : str
        Regex pattern to extract transect numbers from file names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for file_type, file_path, and transect_num.
    """
    # Compile the transect pattern regex
    compiled_pattern = re.compile(transect_pattern)

    # Extract all file info using a list comprehension and flatten the result
    records = [
        {
            "file_type": file_type,
            "file_path": path,
            "transect_num": float(compiled_pattern.search(str(path)).group(1)),
        }
        for file_type, paths in ev_export_paths.items()
        for path in list(paths)
        if compiled_pattern.search(str(path))
    ]

    # Convert to DataFrame
    return pd.DataFrame(records)


def validate_transect_exports(transect_files: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that each transect number has all required export file types without using loops.

    For each transect number, there should be exactly four files: "analysis", "cells", "intervals",
    and "layers".

    Parameters
    ----------
    transect_files : pd.DataFrame
        DataFrame with columns "file_type", "file_path", and "transect_num".

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only valid transect numbers that have all required file types.
    """
    # Create a pivot table to count file types by transect number
    type_count = pd.crosstab(transect_files["transect_num"], transect_files["file_type"])

    # Check that each transect has all required file types
    valid_mask = (type_count[list(ECHOVIEW_DATABASE_EXPORT_FILESET)] == 1).all(axis=1)
    valid_transects = type_count.index[valid_mask].tolist()

    # Filter the original DataFrame to include only valid transect numbers
    filtered_transects = transect_files[transect_files["transect_num"].isin(valid_transects)].copy()

    # Log the removed transect numbers
    all_transects = transect_files["transect_num"].unique()
    removed_transects = np.setdiff1d(all_transects, valid_transects)
    if len(removed_transects) > 0:
        print(f"Removed transect numbers with incomplete file sets: {sorted(removed_transects)}")

    return filtered_transects


def read_nasc_file(
    filename: str | Path,
    sheetname: str,
    impute_coordinates: bool = True,
    column_name_map: dict[str, str] | None = None,
    haul_uid_config: dict[str, Any] | None = None,
):
    """
    Read NASC data from a consolidated XLSX file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the Excel file
    sheetname : str
        Name of the sheet to read
    impute_coordinates : bool
        Instruct whether bad spatial coordinates should be imputed or not
    column_name_map : dict, optional
        Dictionary mapping original column names to new column names
    haul_uid_config : Dict[str, Any]
        Optional keyword arguments to override defaults or DataFrame values:

        - ship_id (dict): Region-specific IDs, e.g., {'US': 10, 'CAN': 20}.

        - survey_id (dict): Region-specific IDs, e.g., {'US': 1, 'CAN': 2}.

        - species_id (int/str): A global species code override.

        - haul_offset (int/float): A value subtracted from 'haul_num' for records identified as
          'CAN' (where haul_num - offset >= 0).

    Examples
    --------
    >>> column_map = {"transect": "transect_num", "region id": "region_id"}
    >>> df = read_nasc_file("data.xlsx", "Sheet1", column_map)

    >>> # Without column mapping
    >>> df = read_nasc_file("data.xlsx", "Sheet1")

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with renamed columns and imputed coordinates
    """
    # Read in the defined file
    consolidated_file = pd.read_excel(filename, sheet_name=sheetname, index_col=None, header=0)

    # Set column names to lowercase
    consolidated_file.columns = consolidated_file.columns.str.lower()

    # Rename columns
    if column_name_map:
        consolidated_file.rename(columns=column_name_map, inplace=True)

    # Fix latitude and longitude
    # ---- Latitude
    if "latitude" in consolidated_file.columns and impute_coordinates:
        impute_bad_coordinates(consolidated_file, "latitude")
    # ---- Longitude
    if "longitude" in consolidated_file.columns and impute_coordinates:
        impute_bad_coordinates(consolidated_file, "longitude")

    # Reformat haul datatype
    consolidated_file["haul_num"] = consolidated_file["haul_num"].astype(float)

    # Add UID column
    add_haul_uids(consolidated_file, _dataset_type="NASC", **(haul_uid_config or {}))

    # Return the cleaned DataFrame
    return consolidated_file


def read_afsc_nasc_file(
    filename: str | Path,
    sheetname: str,
    impute_coordinates: bool = True,
    column_name_map: dict[str, str] | None = None,
):
    """
    Read AFSC NASC data from a consolidated XLSX file.

    This is for the AFSC files that are missing the haul number column.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the Excel file
    sheetname : str
        Name of the sheet to read
    impute_coordinates : bool
        Instruct whether bad spatial coordinates should be imputed or not
    column_name_map : dict, optional
        Dictionary mapping original column names to new column names

    Examples
    --------
    >>> column_map = {"transect": "transect_num", "region id": "region_id"}
    >>> df = read_nasc_file("data.xlsx", "Sheet1", column_map)

    >>> # Without column mapping
    >>> df = read_nasc_file("data.xlsx", "Sheet1")

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with renamed columns and imputed coordinates
    """
    # Read in the defined file
    consolidated_file = pd.read_excel(filename, sheet_name=sheetname, index_col=None, header=0)

    # Set column names to lowercase
    consolidated_file.columns = consolidated_file.columns.str.lower()

    # Rename columns
    if column_name_map:
        consolidated_file.rename(columns=column_name_map, inplace=True)

    # Fix latitude and longitude
    # ---- Latitude
    if "latitude" in consolidated_file.columns and impute_coordinates:
        impute_bad_coordinates(consolidated_file, "latitude")
    # ---- Longitude
    if "longitude" in consolidated_file.columns and impute_coordinates:
        impute_bad_coordinates(consolidated_file, "longitude")

    # Return the cleaned DataFrame
    return consolidated_file


def clean_echoview_cells_df(cells_df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame | None:
    """
    Clean the various region class and name entries.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Cells export DataFrame
    inplace : bool, default False
        If True, perform operation in-place and return None

    Returns
    -------
    pd.DataFrame or None
        DataFrame with cleaned strings for the region class and name strings if inplace=False
        None if inplace=True
    """
    # Work on the original or a copy based on inplace parameter
    df = cells_df if inplace else cells_df.copy()

    # Check column overlap
    region_cols = list(set({"region_class", "region_name"}).intersection(df.columns))

    if not region_cols:
        # No relevant columns found, return unchanged
        return None if inplace else df

    # Adjust "region_class" and/or "region_name" if needed
    df[region_cols] = (
        df[region_cols].replace('"', "", regex=True).replace(r"^\s*(.*?)\s*$", r"\1", regex=True)
    )

    # Set to lowercase
    if "region_class" in df.columns:
        df["region_class"] = df["region_class"].str.lower()

    # Return None for inplace=True, df otherwise
    return None if inplace else df


def impute_bad_coordinates(data: pd.DataFrame, column: str) -> None:
    """
    Impute bad or missing coordinates in a DataFrame.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame containing georeferenced data with latitude and longitude coordinates
    column: str
        Name of the column containing coordinates to impute

    Returns
    -------
    None
        The function modifies the pandas.DataFrame in-place
    """
    # Find indices where values are "999.0" or `NaN`
    invalid_coords = data.index[(data[column] == 999.0) | (data[column].isna())].to_numpy()

    # Break from helper if there are no invalid coordinates
    if len(invalid_coords) == 0:
        return

    # Evaluate cases where there are multiple steps between indices (i.e. multiple groups)
    delta_invalid_coords = np.where(np.diff(invalid_coords) > 1)[0]

    # Digitize into discrete groups of "bad" data
    idx_groups = (
        np.digitize(invalid_coords, invalid_coords[delta_invalid_coords], right=True)
        if len(delta_invalid_coords) > 0
        else np.zeros(len(invalid_coords), dtype=int)
    )

    # Iterate across the bad data groups
    for group in np.unique(idx_groups):
        # Get the matching group index
        in_group_idx = np.where(idx_groups == group)[0]
        # Parse the index
        in_group = invalid_coords[in_group_idx]

        # Case 1: Single or consecutive bad coordinates at head of transect
        if any(in_group == 0):
            # Rolling imputation
            for i in range(in_group.size - 1, -1, -1):
                data.loc[i, column] = 2 * data.loc[i + 1, column] - data.loc[i + 2, column]

        # Case 2: Single or consecutive bad coordinates at tail of transect
        elif any(in_group + 2 > len(data)):
            # Get starting index
            start_index = in_group.min()
            # Get ending index
            end_index = in_group.max()
            # Rolling imputation
            for i in range(start_index, end_index + 1, 1):
                data.loc[i, column] = 2 * data.loc[i - 1, column] - data.loc[i - 2, column]

        # Case 3: Single or consecutive bad coordinates in the middle of transect
        else:
            # Get starting index
            start_index = in_group.min() - 1
            # Get ending index
            end_index = in_group.max() + 1
            # Calculate the mean difference between "good/valid" coordinates
            step_mean = (data.loc[end_index, column] - data.loc[start_index, column]) / (
                in_group.size + 1
            )
            # Impute over "bad" points
            data.loc[in_group, column] = data.loc[start_index, column] + step_mean * np.arange(
                1, in_group.size + 1, 1
            )


def read_echoview_export(filename: Path, validator: Any | None = None) -> pd.DataFrame:
    """
    Read a generic Echoview export CSV file.

    Handles analysis, cells, layers, and intervals export types.

    Parameters
    ----------
    filename : pathlib.Path
        Full path to the NASC CSV file.
    validator : Any
        File-specific validator, if defined.

    Returns
    -------
    pd.DataFrame
        Cleaned and formatted data.
    """
    # Read the CSV file
    # df = read_csv_file(filename)
    df = pd.read_csv(filename, index_col=None, header=0, skipinitialspace=True)

    # Set column names to lowercase
    df.columns = df.columns.str.lower()

    # Disambiguate possible overlapping longitude column names
    # ---- Intersecting names
    lon_columns = set(["lon_s", "lon_m", "lon_e"]).intersection(df.columns)
    # ---- If only 1 is present
    if len(lon_columns) == 1:
        col = next(iter(lon_columns))
        df.rename(columns={col: "longitude"}, inplace=True)
    # ---- If > 1 is present and includes 'lon_m'
    elif len(lon_columns) > 1 and "lon_s" in lon_columns:
        df.rename(columns={"lon_s": "longitude"}, inplace=True)

    # Disambiguate possible overlapping longitude column names
    # ---- Intersecting names
    lat_columns = set(["lat_s", "lat_m", "lat_e"]).intersection(df.columns)
    # ---- If only 1 is present
    if len(lat_columns) == 1:
        col = next(iter(lat_columns))
        df.rename(columns={col: "latitude"}, inplace=True)
    # ---- If > 1 is present and includes 'lon_m'
    elif len(lat_columns) > 1 and "lat_s" in lat_columns:
        df.rename(columns={"lat_s": "latitude"}, inplace=True)

    # Rename columns used by Echopop
    df.rename(columns=ECHOVIEW_TO_ECHOPOP, inplace=True)

    # TODO: Validation step would be here

    return df


def sort_echoview_export_df(export_df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Sort the file rows and reset indices.

    Parameters
    ----------
    export_df : pd.DataFrame
        Echoview export DataFrame that can be either "analysis", "cells", "intervals", or "layers".
    inplace : bool, default False
        If True, perform operation in-place and return None

    Returns
    -------
    pd.DataFrame or None
        Sorted and reindexed DataFrame if inplace=False, None otherwise
    """
    # Work on the original or a copy based on inplace parameter
    df = export_df if inplace else export_df.copy()

    # Check columns against the appropriate sorting columns
    sort_cols = [col for col in ECHOVIEW_EXPORT_ROW_SORT if col in df.columns]

    # Sort the columns
    df.sort_values(sort_cols, inplace=True)

    # Reindex
    df.reset_index(drop=True, inplace=True)

    # Return None for inplace=True, df otherwise
    return None if inplace else df


def update_transect_spacing(
    transect_data: pd.DataFrame,
    default_transect_spacing: float,
    latitude_threshold: float = 60.0,
    inplace: bool = False,
) -> pd.DataFrame | None:
    """
    Calculate and update the maximum spacing between transects.

    Parameters
    ----------
    transect_data : pd.DataFrame
        DataFrame containing transect data with 'transect_num', 'latitude' columns
    default_transect_spacing : float
        Default spacing to use (in nautical miles)
    latitude_threshold : float, default 60.0
        Maximum latitude to consider for spacing calculations
    inplace : bool, default False
        If True, modify the input DataFrame in-place and return None

    Returns
    -------
    pd.DataFrame or None
        Updated DataFrame if inplace=False, None otherwise
    """
    # Work on a copy or the original based on inplace parameter
    df = transect_data if inplace else transect_data.copy()

    # Get unique transect numbers
    transect_number = np.unique(df["transect_num"])

    # Initialize max transect spacing column
    df["transect_spacing"] = default_transect_spacing

    # Iterate through the transects to determine the maximum spacing
    for i in range(len(transect_number)):
        if i >= 2:
            # ---- For 2 transects prior to the current transect
            lag_2_index = df.index[
                (df["transect_num"] == transect_number[i - 2])
                & (df["latitude"] < latitude_threshold)
            ]
            # ---- For 1 transect prior to the current transect
            lag_1_index = df.index[(df["transect_num"] == transect_number[i - 1])]
            # ---- Current transect
            current_index = df.index[
                (df["transect_num"] == transect_number[i]) & (df["latitude"] < latitude_threshold)
            ]

            # Check if we have data for all three transects
            if len(lag_2_index) > 0 and len(lag_1_index) > 0 and len(current_index) > 0:
                # ---- Calculate the mean transect latitude (lag-2)
                lag_2_latitude = df.loc[lag_2_index, "latitude"].mean()
                # ---- Calculate the mean transect latitude (current)
                current_latitude = df.loc[current_index, "latitude"].mean()
                # ---- Compute the difference in the latitudes of adjacent transects
                delta_latitude = np.abs(current_latitude - lag_2_latitude)
                # ---- Get latitude range for the lag-2 transect
                latitude_2_range = (
                    df.loc[lag_2_index, "latitude"].max() - df.loc[lag_2_index, "latitude"].min()
                )
                # ---- Get latitude range for current transect
                latitude_range = (
                    df.loc[current_index, "latitude"].max()
                    - df.loc[current_index, "latitude"].min()
                )
                # ---- Assign maximum spacing
                if (
                    (delta_latitude <= 2.0 * default_transect_spacing * 1.1 / 30.0)
                    & (latitude_2_range < 1 / 6)
                    & (latitude_range < 1 / 6)
                ):
                    df.loc[lag_1_index, "transect_spacing"] = delta_latitude * 30.0

    # Return the updated dataframe or None if inplace
    return None if inplace else df


def read_echoview_nasc(
    filename: Path,
    transect_num: float,
    impute_coordinates: bool = True,
    validator: Any | None = None,
) -> pd.DataFrame:
    """
    Read and pre-process a single Echoview NASC export CSV for one transect.

    Handles coordinate imputation and transect number assignment.

    Parameters
    ----------
    filename : pathlib.Path
        Full path to the NASC CSV file.
    transect_num : float
        Transect number to use for filtering or labeling.
    impute_coordinates : bool
        Instruct whether bad spatial coordinates should be imputed or not

    Returns
    -------
    pd.DataFrame
        Cleaned and formatted DataFrame
    """
    # Read in the defined CSV file
    nasc_df = read_echoview_export(filename, validator)

    # Add transect number
    nasc_df["transect_num"] = transect_num

    # Fix latitude and longitude
    # ---- Latitude
    if "latitude" in nasc_df.columns and impute_coordinates:
        impute_bad_coordinates(nasc_df, "latitude")
    # ---- Longitude
    if "longitude" in nasc_df.columns and impute_coordinates:
        impute_bad_coordinates(nasc_df, "longitude")

    # Return the cleaned DataFrame
    return nasc_df


def echoview_nasc_to_df(
    filtered_df: pd.DataFrame, impute_coordinates: bool = True
) -> list[pd.DataFrame]:
    """
    Read and return Echoview NASC DataFrames for each file listed in the input DataFrame.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        DataFrame with columns "file_path" and "transect_num"
    impute_coordinates : bool
        Instruct whether bad spatial coordinates should be imputed or not

    Returns
    -------
    list[pd.DataFrame]
        List of parsed and validated DataFrames for each file.
    """
    return [
        read_echoview_nasc(row["file_path"], row["transect_num"], impute_coordinates)
        for _, row in filtered_df.iterrows()
    ]


# ! This function is necessary because of how `pandas.DataFrame.merge()` can unexpectedly change
# ! column dtypes. This avoids that issue
def merge_exports(
    df_intervals: pd.DataFrame, df_cells: pd.DataFrame, df_layers: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge Echoview export dataframes with explicit outer joins.

    Parameters
    ----------
    df_intervals : pd.DataFrame
        Intervals export dataframe
    df_cells : pd.DataFrame
        Cells export dataframe
    df_layers : pd.DataFrame
        Layers export dataframe

    Returns
    -------
    pd.DataFrame
        Merged dataframe with preserved data types
    """
    # Store original datatypes
    # ---- Intervals
    intervals_dtypes = df_intervals.dtypes.to_dict()
    # ---- Cells
    cells_dtypes = df_cells.dtypes.to_dict()
    # ---- Layers
    layers_dtypes = df_layers.dtypes.to_dict()

    # Combine all dtypes, prioritizing cells over intervals over layers
    all_dtypes = {**layers_dtypes, **intervals_dtypes, **cells_dtypes}

    # Explicit merges with hard-coded keys
    merged_df = df_cells.merge(df_intervals, how="outer").merge(df_layers, how="outer")

    # Drop NA
    merged_df.dropna(subset=["interval", "nasc"], inplace=True)

    # Restore the original datatypes
    merged_df = merged_df.astype(all_dtypes)

    # Return the output
    return merged_df


def merge_echoview_nasc(
    file_directory: Path,
    filename_transect_pattern: str = r"T(\d+)",
    default_transect_spacing: float = 10.0,
    default_latitude_threshold: float = 60.0,
    impute_coordinates: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Ingest and merge all Echoview NASC files (intervals, cells, layers).

    Parameters
    ----------
    file_directory : Path
        Directory containing Echoview export files (``\\*.csv``).
    filename_transect_pattern: str, default = ``r"T(\\d+)"``
        Regular expression used for extracting the transect number from the filename.
    default_transect_spacing : float, default = 10.
        Default spacing (nmi) to impute where missing.
    default_latitude_threshold : float, default = 60.
        Default latitude threshold used for determining how far north transect spacings should be
        calculated versus using the default value.
    impute_coordinates : bool
        Instruct whether bad spatial coordinates should be imputed or not

    Returns
    -------
    Tuple[|pd.DataFrame|, |pd.DataFrame|]
        A tuple containing two pandas.DataFrames:

        - ``df_intervals``: The processed intervals DataFrame with added transect spacing

        - ``merged_exports_df``: The merged DataFrame from intervals, cells, and layers
    """
    # Get all echoview NASC files: analysis, cells, intervals, layers
    ev_export_paths: dict = {
        "analysis": file_directory.glob("*(analysis).csv"),  # Removed leading / only
        "cells": file_directory.glob("*(cells).csv"),
        "intervals": file_directory.glob("*(intervals).csv"),
        "layers": file_directory.glob("*(layers).csv"),
    }

    # Create a DataFrame with mapped transect numbers to validate the presence of complete filesets
    # ---- Get the trasnect numbers
    transect_num_df = map_transect_num(ev_export_paths, filename_transect_pattern)
    # ---- Validate the filesets
    valid_transect_num_df = validate_transect_exports(transect_num_df)

    # Read and concatenate the Echoview exports (assuming a database format)
    # ---- Cells
    df_cells: pd.DataFrame = pd.concat(
        echoview_nasc_to_df(
            valid_transect_num_df[valid_transect_num_df["file_type"] == "cells"], impute_coordinates
        )
    )
    # ---- Intervals
    df_intervals = pd.concat(
        echoview_nasc_to_df(
            valid_transect_num_df[valid_transect_num_df["file_type"] == "intervals"],
            impute_coordinates,
        )
    )
    # ---- Layers
    df_layers: pd.DataFrame = pd.concat(
        echoview_nasc_to_df(
            valid_transect_num_df[valid_transect_num_df["file_type"] == "layers"],
            impute_coordinates,
        )
    )

    # Clean the cells export file
    clean_echoview_cells_df(df_cells, inplace=True)

    # Sort and reindex the intervals DataFrame
    sort_echoview_export_df(df_intervals, inplace=True)

    # Impute and update the transect spacing
    update_transect_spacing(
        df_intervals, default_transect_spacing, default_latitude_threshold, inplace=True
    )

    # Merge the acoustic export files
    # ! This function is necessary because of how `pandas.DataFrame.merge()` can unexpectedly change
    # ! column dtypes. This avoids that issue
    merged_exports_df = merge_exports(df_intervals, df_cells, df_layers)

    # Resort the merged DataFrame
    sort_echoview_export_df(merged_exports_df, inplace=True)

    # Return two DataFrames: the complete intervals and the cells that will be integrated
    return df_intervals, merged_exports_df


def read_transect_region_haul_key(
    filename: Path, sheetname: str, column_name_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """
    Load the key that maps hauls to export regions to transect numbers.

    This function reads a CSV or Excel file containing the mapping between transect numbers,
    region IDs, and haul numbers. It can handle both file formats and allows column renaming.

    Parameters
    ----------
    filename : Path
        Path to the CSV or Excel file containing the mapping data.
    sheetname : str
        Name of the sheet to read (only used for Excel files).
    column_name_map : Optional[Dict[str, str]], default None
        Dictionary for renaming columns, where keys are original column names and values are new
        column names.

    Returns
    -------
    |pd.DataFrame|
        DataFrame containing only the columns ``"transect_num"``, ``"region_id"``, and
        ``"haul_num"``.

    Notes
    -----
    The input file must contain columns that can be mapped to ``"transect_num"``, ``"region_id"``,
    and ``"haul_num"``, either directly or via the ``column_name_map``.
    """
    # Determine appropriate file reader
    if filename.suffix == ".csv":
        # transect_region_df = read_csv_file(filename)
        transect_region_df = pd.read_csv(filename, index_col=None, header=0, skipinitialspace=True)

    else:
        # transect_region_df = read_xlsx_file(filename, sheetname)
        transect_region_df = pd.read_excel(filename, sheetname, index_col=None, header=0)

    # Lowercase
    transect_region_df.columns = transect_region_df.columns.str.lower()

    # Rename column names, if defined
    if column_name_map:
        transect_region_df.rename(columns=column_name_map, inplace=True)

    # TODO: Insert validation here

    # Return a filtered DataFrame
    return transect_region_df.filter(["transect_num", "region_id", "haul_num"])


def write_transect_region_key(
    transect_region_haul_key: pd.DataFrame,
    filename: str,
    verbose: bool,
) -> None:
    """
    Write transect-region-haul key.

    Parameters
    ----------
    transect_region_haul_key : pd.DataFrame
    filename: str
    verbose : bool
    """
    pass


def compile_patterns(pattern_dict: dict[str, str]) -> dict[str, str]:
    """
    Compile regex patterns from a pattern dictionary.

    Processes a dictionary of pattern specifications and returns a dictionary
    of compiled regex patterns for efficient matching.

    Parameters
    ----------
    pattern_dict : Dict[str, str]
        Dictionary where keys are component names and values are either:
        - Dict mapping labels to regex pattern strings
        - Set of regex pattern strings

    Returns
    -------
    Dict[str, str]
        Dictionary where keys are component names and values are lists of
        compiled regex patterns

    Notes
    -----
    Handles both dictionary-based patterns (where label is the key and pattern is the value)
    and set-based patterns (where patterns are directly in a set).
    """
    # Initialize pattern dictionary
    compiled_patterns = {}

    # Iterate through the pattern dictionary to compile everything into regex
    for part_name, part_patterns in pattern_dict.items():
        compiled_patterns[part_name] = []

        if isinstance(part_patterns, dict):
            for _label, pattern in part_patterns.items():
                compiled_patterns[part_name].append(re.compile(pattern, re.IGNORECASE))
        elif isinstance(part_patterns, set):
            for pattern in part_patterns:
                compiled_patterns[part_name].append(re.compile(pattern, re.IGNORECASE))

    # Return the output dictionary
    return compiled_patterns


def extract_parts_and_labels(region_name: str, compiled_patterns: dict, pattern_dict: dict) -> dict:
    """
    Extract components and their labels from a region name using compiled patterns.

    Analyzes a region name string to identify components based on the provided
    patterns and returns a dictionary of extracted labels.

    Parameters
    ----------
    region_name : str
        The region name string to analyze
    compiled_patterns : Dict
        Dictionary of compiled regex patterns from compile_patterns()
    pattern_dict : Dict
        Original pattern dictionary for label lookup

    Returns
    -------
    Dict
        Dictionary mapping component names to their extracted labels

    Notes
    -----
    This function progressively processes the region name, removing matched
    portions as it goes to avoid duplicate matches.
    """
    # Initialize the labels dictionary
    labels = {}
    remaining_name = region_name

    for part_name, patterns in compiled_patterns.items():
        for pattern in patterns:
            match = pattern.search(remaining_name)
            if match:
                matched_value = match.group(0)

                if isinstance(pattern_dict[part_name], dict):
                    label = next(
                        (
                            key
                            for key, value in pattern_dict[part_name].items()
                            if value == pattern.pattern
                        ),
                        matched_value,
                    )
                elif isinstance(pattern_dict[part_name], set):
                    label = matched_value

                labels[part_name] = label if label != "None" else matched_value
                remaining_name = remaining_name.replace(matched_value, "", 1)
                break

    # Return the labels
    return labels


def extract_region_components(df: pd.DataFrame, pattern_dict: dict[str, str]) -> pd.DataFrame:
    """
    Extract and process components from region names using pattern dictionary.

    Takes a DataFrame with region names and extracts components based on
    provided patterns, returning a new DataFrame with the extracted components.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'region_name' column
    pattern_dict : Dict
        Dictionary of pattern specifications for component extraction

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by region_name with columns for each extracted component

    Notes
    -----
    Only processes unique region names to improve efficiency. The resulting
    DataFrame uses lowercase column names derived from the pattern dictionary keys.
    """
    # Initialize an input DataFrame
    unique_regions = pd.DataFrame({"region_name": df["region_name"].unique()})

    # Compile the regex patterns to parse export region names
    regex_patterns = compile_patterns(pattern_dict)

    # Process each row to extract components
    def process_row(row):
        extracted = extract_parts_and_labels(row["region_name"], regex_patterns, pattern_dict)
        result = {"region_name": row["region_name"]}
        result.update(
            {part_name.lower(): extracted.get(part_name) for part_name in pattern_dict.keys()}
        )
        return pd.Series(result)

    # Apply to all rows and set index
    result = unique_regions.apply(process_row, axis=1)

    # Return the indexed DataFrame
    return result.set_index("region_name")


def process_extracted_data(
    extracted_df: pd.DataFrame, can_haul_offset: int | None = None
) -> pd.DataFrame:
    """
    Process extracted data with type conversions and Canadian haul offsets.

    Takes a DataFrame of extracted components and applies type conversions
    and Canadian haul number offsets.

    Parameters
    ----------
    extracted_df : pd.DataFrame
        DataFrame with extracted region components
    can_haul_offset : Optional[int], default None
        Offset to add to haul numbers for Canadian regions

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with correct data types and haul number offsets

    Notes
    -----
    Applies type conversions based on a predefined mapping and adds the
    specified offset to haul numbers where country is 'CAN'.
    """
    # Define and apply type conversions
    valid_dtypes = {
        "region_class": str,
        "haul_num": float,
        "country": str,
        "ship": float,
    }

    processed = extracted_df.apply(
        lambda col: col.astype(valid_dtypes.get(col.name, type(col.iloc[0])))
    )

    # Apply Canadian haul offset if applicable
    if "country" in processed.columns:
        processed.loc[processed["country"] == "CAN", "haul_num"] = (
            processed.loc[processed["country"] == "CAN", "haul_num"] + can_haul_offset
        )

    return processed


def compute_region_layer_depths(
    transect_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate summary statistics for acoustic transect layers.

    Parameters
    ----------
    transect_data : pd.DataFrame
        DataFrame containing transect measurements. Must include the following columns:
        - max_depth: Maximum depth measurements
        - layer_depth_min: Minimum depth of the layer
        - layer_depth_max: Maximum depth of the layer
        - All columns specified in index_variable

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following columns:
        - All columns specified in index_variable
        - bottom_depth : Maximum depth for each group
        - layer_mean_depth : Average depth of the layer
        - layer_height : Height of the layer (max - min depth)

    Raises
    ------
    TypeError
        If index_variable is neither a string nor a list
    ValueError
        If any required columns are missing from transect_data

    """
    # Convert string input to list
    # if isinstance(index_variable, str):
    #     index_variable = [index_variable]
    # elif not isinstance(index_variable, list):
    #     raise TypeError(
    #         f"index_variable must be either a str or list, got {type(index_variable)}"
    #     )
    index_variable = ["transect_num", "interval"]
    # Validate required columns
    required_columns = {"max_depth", "layer_depth_min", "layer_depth_max"}.union(index_variable)
    missing_columns = required_columns - set(transect_data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in transect_data: {sorted(missing_columns)}")

    # Calculate summary statistics
    grouped = transect_data.groupby(index_variable)

    # Create summary DataFrame
    summary = pd.DataFrame()

    # Calculate bottom depth
    summary["bottom_depth"] = grouped["max_depth"].max()

    # Calculate layer statistics
    layer_stats = grouped.agg({"layer_depth_min": "min", "layer_depth_max": "max"})

    summary["layer_mean_depth"] = (
        layer_stats["layer_depth_min"] + layer_stats["layer_depth_max"]
    ) / 2
    summary["layer_height"] = layer_stats["layer_depth_max"] - layer_stats["layer_depth_min"]

    return summary.reset_index()


def generate_transect_region_haul_key(
    region_data: pd.DataFrame, filter_list: list[str]
) -> pd.DataFrame:
    """
    Filter DataFrame by region class patterns and create a mapping.

    Filters the DataFrame to include only rows with region classes in the provided filter list and
    creates a mapping of unique regions.

    Parameters
    ----------
    region_data : |pd.DataFrame|
        |pd.DataFrame| with processed region data
    filter_list : List[str]
        List of region class names to include in the filter

    Returns
    -------
    |pd.DataFrame|
        Filtered and grouped |pd.DataFrame| containing unique region mappings

    Notes
    -----
    The returned |pd.DataFrame| is grouped by ``'transect_num'``, ``'haul_num'``, and
    ``'region_id'``, with first instances of ``'region_class'`` and ``'region_name'``, sorted by
    ``'haul_num'``.
    """
    # Convert filter list to lowercase for case-insensitive comparison
    filter_set = {name.lower() for name in filter_list}

    # Create df copy
    region_data_copy = region_data.copy()

    # Change class names to lowercase
    region_data_copy.loc[:, "region_class"] = region_data_copy.loc[:, "region_class"].str.lower()

    # Filter
    region_data_filtered = region_data_copy[
        region_data_copy["region_class"].str.lower().isin(filter_set)
    ].copy()

    # Create final mapping
    return (
        region_data_filtered.groupby(["transect_num", "haul_num", "region_id"])[
            ["region_class", "region_name"]
        ]
        .first()
        .reset_index()
        .sort_values(["haul_num"])
    )


def process_region_names(
    nasc_cells: pd.DataFrame,
    region_name_expr: dict,
    can_haul_offset: int | None = None,
) -> pd.DataFrame:
    """
    Process region names in a DataFrame using regex patterns.

    Coordinates the extraction and processing of region name components from a |pd.DataFrame|
    according to specified patterns, with optional filtering and mapping.

    Parameters
    ----------
    nasc_cells : |pd.DataFrame|
        A |pd.DataFrame| comprising NASC export data that must include the column ``'region_name'``
        to function.
    region_name_expr : Dict
        Dictionary of pattern specifications for component extraction:

        - Keys are component names (e.g., ``'REGION_CLASS'``, ``'HAUL_NUM'``, ``'COUNTRY'``)

        - Values are either:

          * Dict mapping labels to regex pattern strings
          * Set of regex pattern strings

    can_haul_offset : Optional[int], Default None
        Offset to add to haul numbers for Canadian regions

    Returns
    -------
    |pd.DataFrame|
        If ``filter_list`` is provided: a mapping of unique regions filtered by region class.
        Otherwise,  the original |pd.DataFrame| with extracted components added.

    Example
    -------
    >>> region_name_expr = {
    ...     "REGION_CLASS": {
    ...         "Hake": "^(?:h(?![a-z]|1a)|hake(?![_]))",
    ...         "Hake Mix": "^(?:hm(?![a-z]|1a)|hake_mix(?![_]))"
    ...     },
    ...     "HAUL_NUM": {"[0-9]+"},
    ...     "COUNTRY": {"CAN": "^[cC]", "US": "^[uU]"}
    ... }
    >>> process_region_names(df, region_name_expr, filter_list=["Hake", "Hake Mix"])
    """
    # Step 1: Extract components from region names
    extracted_regions = extract_region_components(nasc_cells, region_name_expr)

    # Step 2: Process the extracted data
    processed_regions = process_extracted_data(extracted_regions, can_haul_offset)

    # Step 3: Map to original data
    # ---- Index DataFrame based on region names
    mapped_cells = nasc_cells.set_index("region_name")
    # ---- Concatenate the processed region metadata
    mapped_cells.loc[:, processed_regions.columns] = processed_regions
    # ---- Reset the index
    mapped_cells.reset_index(inplace=True)

    # Return the mapped DataFrame
    return mapped_cells


def consolidate_echvoiew_nasc(
    nasc_data: pd.DataFrame,
    interval_data: pd.DataFrame,
    region_class_names: list[str],
    impute_region_ids: bool = True,
    transect_region_haul_key: pd.DataFrame | None = None,
    haul_uid_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Consolidate Echoview NASC data with interval and region information.

    Parameters
    ----------
    nasc_data : |pd.DataFrame|
        DataFrame containing merged Echoview data with columns:

        - ``'region_class'`` : Region classification names

        - ``'region_id'`` : Region identifier

        - ``'nasc'`` : Nautical area scattering coefficient

        - ``'transect_num'`` : Transect number

        - ``'interval'`` : Interval identifier

    interval_data : |pd.DataFrame|
        DataFrame containing interval information with columns:

        - ``'interval'`` : Interval identifier

        - ``'transect_num'`` : Transect number

        - ``'distance_s'`` : Starting distance

        - ``'distance_e'`` : Ending distance

        - ``'latitude'`` : Latitude coordinates

        - ``'longitude'`` : Longitude coordinates

        - ``'transect_spacing'`` : Spacing between transects

    region_class_names : List[str]
        List of region class names to include in the analysis
    impute_region_ids : bool, optional
        Whether to impute region IDs for overlapping regions, by default True
    transect_region_haul_key : |pd.DataFrame|, optional
        DataFrame containing haul information to merge, by default None
    haul_uid_config : Dict[str, Any]
        Optional keyword arguments to override defaults or DataFrame values:

        - ship_id (dict): Region-specific IDs, e.g., {'US': 10, 'CAN': 20}.

        - survey_id (dict): Region-specific IDs, e.g., {'US': 1, 'CAN': 2}.

        - species_id (int/str): A global species code override.

        - haul_offset (int/float): A value subtracted from 'haul_num' for records identified as
          'CAN' (where haul_num - offset >= 0).

    Returns
    -------
    |pd.DataFrame|
        Consolidated DataFrame containing:

        - ``'transect_num'`` : Transect number

        - ``'region_id'`` : Region identifier (``999`` for ``NaN``)

        - ``'distance_s'`` : Starting distance

        - ``'distance_e'`` : Ending distance

        - ``'latitude'`` : Latitude coordinates

        - ``'longitude'`` : Longitude coordinates

        - ``'transect_spacing'`` : Spacing between transects

        - ``'layer_mean_depth'`` : Mean layer depth

        - ``'layer_height'`` : Layer height

        - ``'bottom_depth'`` : Bottom depth

        - ``'nasc'`` : Summed nautical area scattering coefficient

        - ``'haul_num'`` : Haul number (``0`` for ``NaN``)

        - ``'uid'`` : Unique ship-survey-species-haul identifier

    Notes
    -----
    All numeric columns (``'nasc'``, ``'layer_mean_depth'``, ``'layer_height'``,
    ``'bottom_depth'``) are filled with ``0.0`` for ``NaN`` values.
    """
    # Create DataFrame copy
    df_copy = nasc_data.copy()
    # ---- Change region class names to lowercase
    df_copy.loc[:, "region_class"] = df_copy.loc[:, "region_class"].str.lower()

    # Create set of region class names
    region_class_name_set = {name.lower() for name in region_class_names}

    # Filter the NASC values for only the target region class names
    transect_regions = df_copy[df_copy["region_class"].str.lower().isin(region_class_name_set)]
    # ---- Further sorting
    transect_regions = transect_regions.sort_values(
        ["transect_num", "interval", "region_id"], ignore_index=True
    )

    # Impute region IDs for cases where multiple regions overlap within the same interval
    if impute_region_ids:
        transect_regions.loc[:, "region_id"] = transect_regions.groupby(
            ["interval", "transect_num"]
        )["region_id"].transform("first")

    # Get the layer depth means and intervals
    transect_layer_summary = compute_region_layer_depths(transect_regions)

    # Vertically sum backscatter
    nasc_intervals = (
        transect_regions.groupby(["transect_num", "interval", "region_id"])["nasc"]
        .sum()
        .reset_index()
    )

    # Create copy of interval export template
    interval_copy = interval_data.copy().set_index(["interval", "transect_num"]).sort_index()

    # Merge haul information into the integrated NASC DataFrame
    nasc_hauls = nasc_intervals.merge(transect_region_haul_key).set_index(
        ["interval", "transect_num"]
    )

    # Check for country codes
    if "country" in transect_regions.columns:
        # ---- Get country codes
        country_indices = transect_regions.groupby(["interval", "transect_num"])["country"].first()
        nasc_hauls["country"] = (
            nasc_hauls.index.to_frame()[["interval", "transect_num"]]
            .apply(lambda x: country_indices.get((x.interval, x.transect_num)), axis=1)
            .values
        )
        # ---- Update the input haul UID configuration
        if "country" not in (haul_uid_config or {}) and "ship" not in nasc_hauls.columns:
            nasc_hauls["ship"] = np.where(nasc_hauls["country"] == "US", 1, 2)
            nasc_hauls["ship"] = nasc_hauls["ship"].values

    # Append the haul numbers and integrated NASC values to the interval template
    interval_copy.loc[:, nasc_hauls.columns] = nasc_hauls
    # ---- Reset the index
    interval_copy.reset_index(inplace=True)

    # Update UID entries as they apply
    if "ship" in interval_copy.columns:
        interval_copy["ship"] = interval_copy["ship"].fillna(1)
    interval_copy["haul_num"] = interval_copy["haul_num"].fillna(0).astype(float)

    # Add UID label
    add_haul_uids(interval_copy, _dataset_type="NASC", **(haul_uid_config or {}))

    # Combine with the transect layer summaries
    full_interval_strata_df = interval_copy.merge(transect_layer_summary, how="left")
    # ---- Sort
    full_interval_strata_df = full_interval_strata_df.sort_values(
        ["transect_num", "distance_s", "distance_e"]
    )
    # ---- Fill NaN region id column with 999
    full_interval_strata_df["region_id"] = (
        full_interval_strata_df["region_id"].fillna(999).astype(int)
    )
    # ---- Fill haul with 0's
    full_interval_strata_df["haul_num"] = full_interval_strata_df["haul_num"].fillna(0)
    # ---- Fill float/continuous columns
    full_interval_strata_df[["nasc", "layer_mean_depth", "layer_height", "bottom_depth"]] = (
        full_interval_strata_df[["nasc", "layer_mean_depth", "layer_height", "bottom_depth"]]
        .fillna(0)
        .astype(float)
    )
    # ---- Drop unused columns
    output_nasc = full_interval_strata_df.filter(
        [
            "transect_num",
            "region_id",
            "distance_s",
            "distance_e",
            "latitude",
            "longitude",
            "transect_spacing",
            "layer_mean_depth",
            "layer_height",
            "bottom_depth",
            "nasc",
            "haul_num",
            "uid",
        ]
    )

    # Return the consdolidated NASC file
    return output_nasc
