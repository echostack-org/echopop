"""
Biological data ingestion for echopop.

This module loads specimen, length, and catch data from consolidated Excel workbooks, from separate
CSV database-view exports, or directly from a database. It handles ship/survey sub-setting,
column renaming, label remapping, and haul UID construction so that downstream survey and
apportionment functions receive uniformly structured either individual ``pandas.DataFrame``
objects, or keyed by dataset-type within a dictionary.
"""

import copy
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from ..utils.base import add_haul_uids


def load_single_biological_sheet(
    biodata_filepath: Path,
    sheet_name: str,
    column_name_map: dict | None = None,
    survey_subset: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load and process a single biological data sheet.

    Parameters
    ----------
    biodata_filepath : pathlib.Path
        Path to the Excel file
    sheet_name : str
        Name of the sheet to load
    column_name_map : dict, default={}
        Dictionary mapping original column names to new column names
    survey_subset : dict, optional
        Subset dictionary containing ships and species_code settings
        Format: {"ships": {ship_id: {"survey": survey_id, "haul_offset": offset}}, "species_code":
        [codes]}

    Returns
    -------
    |pd.DataFrame|
        Processed and validated biological data
    """
    # Read Excel file into memory
    df_initial = pd.read_excel(biodata_filepath, sheet_name=sheet_name, index_col=None, header=0)

    # Force the column names to be lower case
    df_initial.columns = df_initial.columns.str.lower()

    # Rename the columns
    if column_name_map:
        df_initial.rename(columns=column_name_map, inplace=True)

    # Apply ship and survey filtering
    df_filtered = apply_ship_survey_filters(df_initial, survey_subset)

    return df_filtered


def load_single_biological_view(
    biodata_filepath: Path,
    column_name_map: dict | None = None,
    survey_subset: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load and process a single biological data view.

    Parameters
    ----------
    biodata_filepath : pathlib.Path
        Path to the Excel file
    column_name_map : dict, default={}
        Dictionary mapping original column names to new column names
    survey_subset : dict, optional
        Subset dictionary containing ships and species_code settings
        Format: {"ships": {ship_id: {"survey": survey_id, "haul_offset": offset}}, "species_code":
        [codes]}

    Returns
    -------
    |pd.DataFrame|
        Processed and validated biological data
    """
    # Read *.csv file into memory
    df_initial = pd.read_csv(biodata_filepath, index_col=None, header=0)

    # Force the column names to be lower case
    df_initial.columns = df_initial.columns.str.lower()

    # Rename the columns
    if column_name_map:
        df_initial.rename(columns=column_name_map, inplace=True)

    # Apply ship and survey filtering
    df_filtered = apply_ship_survey_filters(df_initial, survey_subset)

    return df_filtered


def load_biodata_db_views(
    db_credentials: dict[str, str],
    biodata_table_map: dict[str, str],
    column_name_map: dict[str, str] = None,
    subset_dict: dict | None = None,
    biodata_label_map: dict[str, dict] | None = None,
    haul_uid_config: dict[str, Any] = None,
) -> dict[str, pd.DataFrame] | None:
    """
    Load biological data from a postgres database.

    Parameters
    ----------
    db_credentials : dict
        Dictionary containing database credentials
        (e.g., {"host": "localhost", "port": "5432", "dbname": "fisheries", "schema": "biodata"
        "user": "<USERNAME>", "password": "<PASSWORD>"})
    biodata_table_map : dict
        Dictionary mapping dataset names to database table names
        (e.g., {"specimen": "biodata_specimen", "length": "biodata_length",
        "catch": "biodata_catch"})
    column_name_map : dict, optional
        Dictionary mapping original column names to new column names
        (e.g., {"frequency": "length_count", "haul": "haul_num"})
    subset_dict : dict, optional
        Subset dictionary containing ships and species_code for filtering
        Format: {"ships": {ship_id: {"survey": survey_id, "haul_offset": offset}}, "species_code":
        [codes]}
    biodata_label_map : dict, optional
        Dictionary mapping column names to value replacement dictionaries
        (e.g., {"sex": {1: "male", 2: "female", 3: "unsexed"}})
    haul_uid_config : Dict[str, Any]
        Optional keyword arguments to override defaults or DataFrame values:

        - ship_id (dict): Region-specific IDs, e.g., ``{'US': 10, 'CAN': 20}``.

        - survey_id (dict): Region-specific IDs, e.g., ``{'US': 1, 'CAN': 2}``.

        - species_id (int/str): A global species code override.

        - haul_offset (int/float): A value subtracted from ``'haul_num'`` for records identified as
          'CAN' (where ``haul_num - offset >= 0``).

    Returns
    -------
    dict
        Dictionary containing processed biological DataFrames keyed by dataset name

    Examples
    --------
    >>> subset = {"ships": {160: {"survey": 201906}}, "species_code": [22500]}
    >>> col_map = {"frequency": "length_count", "haul": "haul_num"}
    >>> label_map = {"sex": {1: "male", 2: "female", 3: "unsexed"}}
    """
    try:
        db_url = (
            f"postgresql+psycopg://"
            f"{db_credentials['user']}:{db_credentials['password']}@"
            f"{db_credentials['host']}:{db_credentials['port']}/"
            f"{db_credentials['dbname']}"
        )

        engine = create_engine(db_url)

        biodata_dict = {}

        with engine.connect() as connection:
            for data_set, table in biodata_table_map.items():
                query = f"SELECT * FROM {db_credentials['schema']}.{table};"

                df_initial = pd.read_sql_query(query, connection)

                # Force the column names to be lower case
                df_initial.columns = df_initial.columns.str.lower()

                # Rename the columns
                if column_name_map:
                    df_initial.rename(columns=column_name_map, inplace=True)

                # # Validate data types for ship and survey before filtering
                df_initial["ship"] = pd.to_numeric(df_initial["ship"])
                df_initial["survey"] = pd.to_numeric(df_initial["survey"])

                biodata_dict[data_set] = apply_ship_survey_filters(df_initial, subset_dict)

        # Apply label mappings if provided
        if biodata_label_map:
            for col, mapping in biodata_label_map.items():
                for _name, df in biodata_dict.items():
                    if isinstance(df, pd.DataFrame) and col in df.columns:
                        df[col] = df[col].map(mapping).fillna(df[col])

        # # Validate data types
        biodata_dict["specimen"]["length"] = pd.to_numeric(biodata_dict["specimen"]["length"])
        biodata_dict["specimen"]["weight"] = pd.to_numeric(biodata_dict["specimen"]["weight"])

        # Reformat haul datatype
        biodata_dict = {
            k: v.assign(haul_num=v["haul_num"].astype(float)) for k, v in biodata_dict.items()
        }

        # Add UID labels
        _ = {
            k: add_haul_uids(v, _dataset_type=f"biodata.{k}", **(haul_uid_config or {}))
            for k, v in biodata_dict.items()
        }

        return biodata_dict

    except Exception as e:
        print(f"Database error: {e}")

    finally:
        if "engine" in locals():
            engine.dispose()


def load_biodata_views(
    biodata_filepaths: dict[str, Path],
    column_name_map: dict[str, str] = None,
    survey_subset: dict | None = None,
    biodata_label_map: dict[str, dict] | None = None,
    haul_uid_config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load biological data from a database view.

    Load biological data from materialized views of catch data and a file including specimen-only
    values.

    Parameters
    ----------
    biodata_filepaths : pathlib.Path
        A dictionary of filepaths to ``*.csv`` files containing biological data
    column_name_map : dict, optional
        Dictionary mapping original column names to new column names
        (e.g., ``{"frequency": "length_count", "haul": "haul_num"}``)
    survey_subset : dict, optional
        Subset dictionary containing ships and species_code for filtering
        Format: ``{"ships": {ship_id: {"survey": survey_id, "haul_offset": offset}}, "species_code":
        [codes]}``
    biodata_label_map : dict, optional
        Dictionary mapping column names to value replacement dictionaries (e.g.,
        ``{"sex": {1: "male", 2: "female", 3: "unsexed"}}``)
    haul_uid_config : Dict[str, Any]
        Optional keyword arguments to override defaults or DataFrame values:

        - ship_id (dict): Region-specific IDs, e.g., ``{'US': 10, 'CAN': 20}``.

        - survey_id (dict): Region-specific IDs, e.g., ``{'US': 1, 'CAN': 2}``.

        - species_id (int/str): A global species code override.

        - haul_offset (int/float): A value subtracted from ``'haul_num'`` for records identified as
          'CAN' (where ``haul_num - offset >= 0``).

    Returns
    -------
    dict
        Dictionary containing processed biological DataFrames keyed by dataset name
    """
    # Validate across files
    if not all([v.exists() for v in biodata_filepaths.values()]):
        raise FileNotFoundError("Not all files in 'biodata_filepaths' could be found.")

    # Load each biological dataset
    biodata_dict = {
        dataset_name: load_single_biological_view(file, column_name_map, survey_subset)
        for dataset_name, file in biodata_filepaths.items()
    }

    # Apply label mappings if provided
    if biodata_label_map:
        # ---- For each column mapping in the label map
        for col, mapping in biodata_label_map.items():
            # ---- Apply to each dataframe that has that column
            for _name, df in biodata_dict.items():
                if isinstance(df, pd.DataFrame) and col in df.columns:
                    df[col] = df[col].map(mapping).fillna(df[col])

    # Reformat haul datatype
    biodata_dict = {
        k: v.assign(haul_num=v["haul_num"].astype(float)) for k, v in biodata_dict.items()
    }

    # Add UID labels
    _ = {
        k: add_haul_uids(v, _dataset_type=f"biodata.{k}", **(haul_uid_config or {}))
        for k, v in biodata_dict.items()
    }

    return biodata_dict


def load_biological_data(
    biodata_filepath: Path,
    biodata_sheet_map: dict[str, str],
    column_name_map: dict[str, str] = None,
    survey_subset: dict | None = None,
    biodata_label_map: dict[str, dict] | None = None,
    haul_uid_config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load biological data from a single Excel file with multiple sheets.

    Parameters
    ----------
    biodata_filepath : pathlib.Path
        Path to the Excel file containing biological data
    biodata_sheet_map : dict
        Dictionary mapping dataset names to sheet names
        (e.g., ``{"specimen": "biodata_specimen", "length": "biodata_length", "catch":
        "biodata_catch"}``)
    column_name_map : dict, optional
        Dictionary mapping original column names to new column names
        (e.g., ``{"frequency": "length_count", "haul": "haul_num"}``)
    survey_subset : dict, optional
        Subset dictionary containing ships and species_code for filtering
        Format: ``{"ships": {ship_id: {"survey": survey_id, "haul_offset": offset}}, "species_code":
        [codes]}``
    biodata_label_map : dict, optional
        Dictionary mapping column names to value replacement dictionaries (e.g.,
        ``{"sex": {1: "male", 2: "female", 3: "unsexed"}}``)
    haul_uid_config : Dict[str, Any]
        Optional keyword arguments to override defaults or DataFrame values:

        - ship_id (dict): Region-specific IDs, e.g., {'US': 10, 'CAN': 20}.

        - survey_id (dict): Region-specific IDs, e.g., {'US': 1, 'CAN': 2}.

        - species_id (int/str): A global species code override.

        - haul_offset (int/float): A value subtracted from 'haul_num' for records identified as
          'CAN' (where haul_num - offset >= 0).

    Returns
    -------
    dict
        Dictionary containing processed biological DataFrames keyed by dataset name

    Examples
    --------
    >>> sheet_map = {"catch": "biodata_catch", "length": "biodata_length"}
    >>> subset = {"ships": {160: {"survey": 201906}}, "species_code": [22500]}
    >>> col_map = {"frequency": "length_count", "haul": "haul_num"}
    >>> label_map = {"sex": {1: "male", 2: "female", 3: "unsexed"}}
    >>> bio_data = load_biological_data("biodata.xlsx", sheet_map, col_map, subset, label_map)
    """
    if not biodata_filepath.exists():
        raise FileNotFoundError(f"Biological data file not found: {biodata_filepath}")

    # Load each biological dataset
    biodata_dict = {
        dataset_name: load_single_biological_sheet(
            biodata_filepath, sheet_name, column_name_map, survey_subset
        )
        for dataset_name, sheet_name in biodata_sheet_map.items()
    }

    # Apply label mappings if provided
    if biodata_label_map:
        # ---- For each column mapping in the label map
        for col, mapping in biodata_label_map.items():
            # ---- Apply to each dataframe that has that column
            for _name, df in biodata_dict.items():
                if isinstance(df, pd.DataFrame) and col in df.columns:
                    df[col] = df[col].map(mapping).fillna(df[col])

    # Reformat haul datatype
    biodata_dict = {
        k: v.assign(haul_num=v["haul_num"].astype(float)) for k, v in biodata_dict.items()
    }

    # Add UID labels
    _ = {
        k: add_haul_uids(v, _dataset_type=f"biodata.{k}", **(haul_uid_config or {}))
        for k, v in biodata_dict.items()
    }

    return biodata_dict


def apply_ship_survey_filters(
    biodata: pd.DataFrame,
    survey_subset: dict | None = None,
) -> pd.DataFrame:
    """
    Apply ship ID, survey ID, and haul offset filters to biological data.

    Parameters
    ----------
    biodata : |pd.DataFrame|
        Biological data that will be filtered.
    survey_subset : dict, optional
        Subset dictionary containing ships and species_code settings.
        Format: ``{"ships": {ship_id: {"survey": survey_id, "haul_offset": offset}}, "species_code":
        [codes]}``

    Returns
    -------
    |pd.DataFrame|
        Filtered DataFrame
    """
    # Create copy
    biodata_filtered = biodata.copy()  # Fixed: df_initial -> df

    # Check if subset_dict exists
    if survey_subset is None:
        return biodata_filtered

    # Extract ship configuration from nested structure
    if "ships" in survey_subset:
        # ---- Get general ship ID's
        ship_ids = [*survey_subset["ships"].keys()]
        # ---- Get ship-specific configurations
        ship_config = survey_subset["ships"]
        # ---- Apply ship-based filter
        if "ship" in biodata_filtered.columns:
            biodata_filtered = biodata_filtered.loc[biodata_filtered["ship"].isin(ship_ids)]
        # ---- Collect survey IDs, if present
        survey_ids = list(
            itertools.chain.from_iterable(
                [
                    v["survey"] if isinstance(v["survey"], list) else [v["survey"]]
                    for v in ship_config.values()
                    if "survey" in v
                ]
            )
        )
        # ---- Apply survey-based filter
        if survey_ids and "survey" in biodata_filtered.columns:
            biodata_filtered = biodata_filtered.loc[biodata_filtered["survey"].isin(survey_ids)]
        # ---- Collect haul offsets, if any are present
        haul_offsets = {k: v["haul_offset"] for k, v in ship_config.items() if "haul_offset" in v}
        # ---- Apply haul number offsets, if defined
        if haul_offsets and "ship" in biodata_filtered.columns:
            biodata_filtered["haul_num"] = biodata_filtered["haul_num"] + biodata_filtered[
                "ship"
            ].map(haul_offsets).fillna(0)

    # Filter species
    if "species_code" in survey_subset:
        biodata_filtered = biodata_filtered.loc[
            biodata_filtered["species_code"].isin(survey_subset["species_code"])
        ]

    return biodata_filtered


def generate_composite_key(
    biodata: dict[str, pd.DataFrame],
    index_columns: list[str],  # uid columns
    adjust: dict[str, np.number] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Build a composite UID key from one or more index columns across all biodata DataFrames.

    Parameters
    ----------
    biodata : dict[str, pandas.DataFrame]
        Dictionary of biological DataFrames keyed by dataset name.
    index_columns : list[str]
        Column names used to construct the composite key (e.g. ``['ship', 'haul_num']``).
    adjust : dict[str, numeric], optional
        Mapping of column name to a numeric offset applied before key construction.

    Returns
    -------
    pandas.DataFrame
        Unique index-column / UID pairs suitable for merging into downstream DataFrames.
    """
    # Validate that all columns exist in the underlying biodata
    valid_id_columns = {
        k: set(index_columns) <= set(v.columns) or "uid" in v.columns for k, v in biodata.items()
    }
    # ---- Collect invalid entries
    invalid_entries = [k for k, v in valid_id_columns.items() if not v]
    # ---- Raise KeyError
    if invalid_entries:
        id_col_str = "', '".join(index_columns)
        entr_str = "', '".join(invalid_entries)
        raise KeyError(
            f"Both defined ID columns ('{id_col_str}') or preset 'uid' column missing from "
            f"the following biodata DataFrames: '{entr_str}'."
        )

    # Create copy
    biodata_copy = copy.deepcopy(biodata)

    # Apply adjustments
    adjust = adjust or {}
    for column, adjustment in adjust.items():
        # ---- Format new name
        new_name = column + "_adj"
        # ---- Apply adjustment
        biodata_copy = {
            k: v.assign(
                **{
                    new_name: np.where(
                        v[column] + adjustment > 0, v[column] + adjustment, v[column]
                    )
                }
            )
            for k, v in biodata_copy.items()
        }

    # Identify the constructor columns
    constructor_columns = [c if c not in adjust else c + "_adj" for c in index_columns]

    # Join the columns to create unique id column
    biodata_copy = {
        k: v.assign(uid=v[constructor_columns].astype(str).agg("-".join, axis=1)).filter(
            list(biodata[k].columns) + ["uid"]
        )
        for k, v in biodata_copy.items()
    }

    # Get unique columns + uid for merging into downstream DataFrames
    unique_uid = pd.concat(
        [d.filter(index_columns + ["uid"]) for d in biodata_copy.values()]
    ).drop_duplicates()

    # Return the unique IDs
    return unique_uid


def apply_composite_key(
    biodata: pd.DataFrame,
    composite_key: pd.DataFrame,
    haul_offset: dict[np.number, np.number] = (),
) -> pd.DataFrame:
    """
    Merge a composite UID key into a biological DataFrame.

    Parameters
    ----------
    biodata : pandas.DataFrame
        Biological data that will receive the UID column.
    composite_key : pandas.DataFrame
        DataFrame of unique index-column / UID pairs produced by
        :func:`generate_composite_key`.
    haul_offset : tuple of (numeric, numeric), optional
        Two-element sequence ``(ship_id, offset)``; the offset is added to
        ``haul_num`` for rows belonging to ``ship_id`` before the merge.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with a ``'uid'`` column appended via a left merge.
    """
    # Apply adjustments to the original composite key columns
    composite_id = composite_key.copy()
    if len(haul_offset) == 2:
        composite_id["haul_num"] = np.where(
            composite_id["ship"] == haul_offset[0],
            composite_id["haul_num"] + haul_offset[1],
            composite_id["haul_num"],
        )

    # Drop "uid" from data
    if "uid" in biodata.columns:
        biodata.drop(columns=["uid"], inplace=True)

    # Left-merge
    shared_columns = [c for c in composite_id.columns if c in biodata.columns and c != "uid"]
    data_uid = biodata.merge(
        composite_id.filter(shared_columns + ["uid"]), on=shared_columns, how="left"
    )

    # Return
    return data_uid
