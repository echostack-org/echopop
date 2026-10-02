"""Regression tests for exported kriged mesh results."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from echopop.reports import Reporter


@pytest.mark.parametrize("include_et_id", [False, True])
@pytest.mark.parametrize("stratum_as_index", [False, True])
def test_kriged_mesh_results_identifiers(tmp_path, include_et_id, stratum_as_index):
    """Preserve mesh identifiers and estimates through stratum alignment and Excel export."""
    mesh = pd.DataFrame(
        {
            "longitude": [-124.5, -125.5, -126.5],
            "latitude": [46.5, 47.5, 48.5],
            "mesh_stratum": [20, 10, 20],
            "nasc": [100.0, 200.0, 300.0],
            "abundance_male": [10.0, 20.0, 30.0],
            "abundance_female": [15.0, 25.0, 35.0],
            "abundance": [25.0, 45.0, 65.0],
            "biomass_male": [40.0, 80.0, 120.0],
            "biomass_female": [60.0, 100.0, 140.0],
            "biomass": [100.0, 180.0, 260.0],
            "cell_cv": [0.1, 0.2, 0.3],
        },
        index=[7, 2, 9],
    )
    if include_et_id:
        mesh["et_id"] = [503, 101, 907]
    sigma_bs = pd.DataFrame({"bio_stratum": [10, 20], "sigma_bs": [0.01, 0.02]})
    if stratum_as_index:
        mesh = mesh.set_index("mesh_stratum")
        sigma_bs = sigma_bs.set_index("bio_stratum")
    original_mesh = mesh.copy(deep=True)
    original_sigma_bs = sigma_bs.copy(deep=True)

    Reporter(tmp_path, verbose=False).kriged_mesh_results_report(
        filename="mesh.xlsx",
        sheetname="Mesh",
        kriged_data=mesh,
        kriged_stratum="mesh_stratum",
        kriged_variable="biomass",
        sigma_bs_data=sigma_bs,
        sigma_bs_stratum="bio_stratum",
    )

    actual = pd.read_excel(tmp_path / "mesh.xlsx", sheet_name="Mesh")
    expected = pd.DataFrame(
        {
            "Lon": [-124.5, -125.5, -126.5],
            "Lat": [46.5, 47.5, 48.5],
            "stratum": [20, 10, 20],
            "NASC": [100.0, 200.0, 300.0],
            "ntk_male": [10.0, 20.0, 30.0],
            "ntk_female": [15.0, 25.0, 35.0],
            "ntk_total": [25.0, 45.0, 65.0],
            "wgt_male": [40.0, 80.0, 120.0],
            "wgt_female": [60.0, 100.0, 140.0],
            "wgt_total": [100.0, 180.0, 260.0],
            "sig_b": 4.0 * np.pi * np.array([0.02, 0.01, 0.02]),
            "krig_CV": [0.1, 0.2, 0.3],
            "krig_SD": [10.0, 36.0, 78.0],
        }
    )
    if include_et_id:
        expected.insert(0, "ET_ID", [503, 101, 907])
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    pd.testing.assert_frame_equal(mesh, original_mesh)
    pd.testing.assert_frame_equal(sigma_bs, original_sigma_bs)


@pytest.mark.parametrize("include_et_id", [False, True])
@pytest.mark.parametrize("nonzero_only", [False, True])
def test_kriged_aged_biomass_mesh_identifiers(tmp_path, include_et_id, nonzero_only):
    """Retain identifiers in every sex sheet for full and filtered aged meshes."""
    mesh = pd.DataFrame(
        {
            "longitude": [-124.5, -125.5, -126.5],
            "latitude": [46.5, 47.5, 48.5],
            "mesh_stratum": [20, 10, 20],
            "biomass": [100.0, 0.0, 260.0],
            "biomass_male": [40.0, 0.0, 120.0],
            "biomass_female": [60.0, 0.0, 140.0],
        },
        index=[7, 2, 9],
    )
    if include_et_id:
        mesh["et_id"] = [503, 101, 907]
    if nonzero_only:
        mesh = mesh[mesh["biomass"] > 0.0]
    weight_data = xr.DataArray(
        [[[[1.0, 3.0], [2.0, 1.0]], [[3.0, 1.0], [2.0, 3.0]]]],
        dims=["length_bin", "age_bin", "sex", "bio_stratum"],
        coords={
            "length_bin": pd.IntervalIndex.from_tuples([(10.0, 20.0)]),
            "age_bin": pd.CategoricalIndex(pd.IntervalIndex.from_tuples([(0.5, 1.5), (1.5, 2.5)])),
            "sex": ["male", "female"],
            "bio_stratum": [10, 20],
        },
    )
    original_mesh = mesh.copy(deep=True)
    original_weights = weight_data.copy(deep=True)
    sheetnames = {"all": "All", "male": "Male", "female": "Female"}

    Reporter(tmp_path, verbose=False).kriged_aged_biomass_mesh_report(
        filename="aged_mesh.xlsx",
        sheetnames=sheetnames,
        kriged_data=mesh,
        weight_data=weight_data,
        kriged_stratum_link={"mesh_stratum": "bio_stratum"},
    )

    sheets = pd.read_excel(tmp_path / "aged_mesh.xlsx", sheet_name=None, header=1)
    assert set(sheets) == set(sheetnames.values())
    age1_proportions = {
        "all": {10: 0.375, 20: 0.5},
        "male": {10: 0.25, 20: 0.75},
        "female": {10: 0.5, 20: 0.25},
    }
    for sex, sheetname in sheetnames.items():
        biomass_column = "biomass" if sex == "all" else f"biomass_{sex}"
        output_column = "wgt_total" if sex == "all" else f"wgt_{sex}"
        expected = mesh[["latitude", "longitude", "mesh_stratum", biomass_column]].rename(
            columns={
                "latitude": "Lat",
                "longitude": "Lon",
                "mesh_stratum": "stratum",
                biomass_column: output_column,
            }
        )
        proportions = mesh["mesh_stratum"].map(age1_proportions[sex])
        expected[1] = mesh[biomass_column] * proportions
        expected[2] = mesh[biomass_column] * (1.0 - proportions)
        if include_et_id:
            expected.insert(0, "ET_ID", mesh["et_id"])
        pd.testing.assert_frame_equal(
            sheets[sheetname], expected.reset_index(drop=True), check_dtype=False
        )
    pd.testing.assert_frame_equal(mesh, original_mesh)
    xr.testing.assert_identical(weight_data, original_weights)


@pytest.mark.parametrize("include_et_id", [False, True])
def test_kriging_input_identifiers(tmp_path, include_et_id):
    """Preserve input identifiers and values in the exported workbook."""
    transect_data = pd.DataFrame(
        {
            "latitude": [46.5, 47.5, 48.5],
            "longitude": [-124.5, -125.5, -126.5],
            "biomass_density": [10.0, 0.0, 25.0],
            "nasc": [100.0, 0.0, 250.0],
            "number_density": [20.0, 0.0, 50.0],
        },
        index=[7, 2, 9],
    )
    if include_et_id:
        transect_data["et_id"] = [503, 101, 907]
    original = transect_data.copy(deep=True)

    Reporter(tmp_path, verbose=False).kriging_input_report(
        filename="kriging_input.xlsx", sheetname="Input", transect_data=transect_data
    )

    actual = pd.read_excel(tmp_path / "kriging_input.xlsx", sheet_name="Input")
    expected = pd.DataFrame(
        {
            "Lat": [46.5, 47.5, 48.5],
            "Lon": [-124.5, -125.5, -126.5],
            "Biomass density": [10.0, 0.0, 25.0],
            "NASC": [100.0, 0.0, 250.0],
            "Number density": [20.0, 0.0, 50.0],
        }
    )
    if include_et_id:
        expected.insert(0, "ET_ID", [503, 101, 907])
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    pd.testing.assert_frame_equal(transect_data, original)
