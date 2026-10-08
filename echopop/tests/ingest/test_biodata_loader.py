from pathlib import Path

import pandas as pd
import pytest

from echopop.ingest.biological import (
    apply_ship_survey_filters,
    load_biodata_db_views,
    load_biological_data,
)


def test_load_biological_data_basic(bio_excel_file, bio_sheet_map):
    """Test basic loading of biological data without optional parameters."""
    # Pass an empty dict instead of None for column_name_map
    result = load_biological_data(
        bio_excel_file, bio_sheet_map, column_name_map={"haul": "haul_num"}
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == set(bio_sheet_map.keys())

    for df in result.values():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


def test_load_biological_data_with_column_map(bio_excel_file, bio_sheet_map, bio_column_map):
    """Test loading with column name mapping."""
    result = load_biological_data(bio_excel_file, bio_sheet_map, column_name_map=bio_column_map)

    if "length" in result:
        assert "length_count" in result["length"].columns
        assert "haul_num" in result["length"].columns
        assert "haul" not in result["length"].columns

    if "catch" in result:
        assert "haul_weight" in result["catch"].columns
        assert "haul_num" in result["catch"].columns
        assert "weight_in_haul" not in result["catch"].columns


def test_load_biological_data_with_subset(bio_excel_file, bio_sheet_map, subset_dict):
    """Test loading with subset filtering."""
    # Pass an empty dict for column_name_map
    result = load_biological_data(
        bio_excel_file,
        bio_sheet_map,
        column_name_map={"haul": "haul_num"},
        survey_subset=subset_dict,
    )

    for df in result.values():
        if "species_code" in df.columns:
            assert df["species_code"].unique() == [22500]

        if "ship_id" in df.columns:
            assert set(df["ship_id"].unique()).issubset({160, 584})


def test_load_biological_data_with_label_map(bio_excel_file, bio_sheet_map, label_map):
    """Test loading with label mapping."""
    result = load_biological_data(
        bio_excel_file,
        bio_sheet_map,
        column_name_map={"haul": "haul_num"},
        biodata_label_map=label_map,
    )

    for df in result.values():
        if "sex" in df.columns:
            assert set(df["sex"].unique()).issubset({"male", "female", "unsexed"})
            assert not any(pd.api.types.is_numeric_dtype(val) for val in df["sex"].unique())


def test_load_biological_data_file_not_found(bio_sheet_map):
    """Test error handling for non-existent file."""
    non_existent_file = Path("non_existent_file.xlsx")

    with pytest.raises(FileNotFoundError):
        load_biological_data(non_existent_file, bio_sheet_map)


# Ship/survey filter tests
def test_apply_ship_survey_filters(biological_data, subset_dict):
    """Test ship/survey filtering."""
    df = biological_data["length"]
    result = apply_ship_survey_filters(df, subset_dict)

    assert result["species_code"].unique() == [22500]
    assert set(result["ship_id"].unique()).issubset({160, 584})


def test_apply_ship_survey_filters_no_subset(biological_data):
    """Test with no subset dict."""
    df = biological_data["length"]
    result = apply_ship_survey_filters(df, None)

    assert result is not df  # Not the same object
    pd.testing.assert_frame_equal(result, df)  # But same content


# Ingest from database tests
def test_load_biological_data_basic_from_db(database_credentials, bio_data_table_map):
    """Test basic loading of biological data without optional parameters."""
    result = load_biodata_db_views(database_credentials, bio_data_table_map)

    assert isinstance(result, dict)

    for df in result.values():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


def test_load_biological_data_with_column_map_from_db(
    database_credentials, bio_data_table_map, column_name_map
):
    """Test loading with column name mapping."""
    result = load_biodata_db_views(
        database_credentials, bio_data_table_map, column_name_map=column_name_map
    )

    if "catch" in result:
        assert "haul_weight" in result["catch"].columns
        assert result["catch"].loc[3, "haul_weight"] == 250.0
        assert "haul_num" in result["catch"].columns
        assert "weight_in_haul" not in result["catch"].columns

    if "specimen" in result:
        assert "species_code" in result["specimen"].columns
        assert result["specimen"].loc[2, "species_code"] == 22500
        assert "haul_num" in result["catch"].columns


def test_load_biological_data_with_subset_from_db(
    database_credentials, bio_data_table_map, pg_subset_dict, column_name_map
):
    """Test loading with subset filtering."""
    result = load_biodata_db_views(
        database_credentials,
        bio_data_table_map,
        subset_dict=pg_subset_dict,
        column_name_map=column_name_map,
    )

    for df in result.values():
        if "species_code" in df.columns:
            assert (df["species_code"] == 22500).all()

        if "ship" in df.columns:
            assert (df["ship"] == 101).all()
