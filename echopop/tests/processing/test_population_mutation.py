"""
Regression tests for population calculations that update caller-owned DataFrames.
This is to address pandas>=3 warnings re inplace=True removal.
"""

import numpy as np
import pandas as pd
import xarray as xr

from echopop.survey.apportionment import mesh_biomass_to_nasc
from echopop.survey.biology import compute_abundance, compute_biomass


def test_grouped_population_calculations_modify_original_dataframe():
    data = pd.DataFrame(
        {
            "stratum": [2, 1],
            "area_interval": [3.0, 2.0],
            "number_density": [10.0, 20.0],
        },
        index=[8, 4],
    )
    original = data
    proportions = xr.Dataset(
        {
            "proportion": xr.DataArray(
                [[[0.25], [0.75]], [[0.6], [0.4]]],
                dims=["stratum", "sex", "length_bin"],
                coords={"stratum": [1, 2], "sex": ["female", "male"], "length_bin": [10]},
            )
        }
    )

    assert compute_abundance(data, number_proportions=proportions) is None
    assert data is original
    assert data["stratum"].tolist() == [2, 1]
    assert data.index.equals(pd.RangeIndex(2))
    np.testing.assert_allclose(data["abundance"], [30.0, 40.0])
    np.testing.assert_allclose(data["abundance_female"], [18.0, 10.0])
    np.testing.assert_allclose(data["abundance_male"], [12.0, 30.0])

    weights = xr.DataArray(
        [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        dims=["stratum", "sex"],
        coords={"stratum": [1, 2], "sex": ["female", "male", "all"]},
    )
    assert compute_biomass(data, stratum_weights=weights) is None
    assert data is original
    assert data["stratum"].tolist() == [2, 1]
    assert data.index.equals(pd.RangeIndex(2))
    np.testing.assert_allclose(data["biomass_female"], [36.0, 20.0])
    np.testing.assert_allclose(data["biomass_male"], [24.0, 60.0])
    np.testing.assert_allclose(data["biomass"], [60.0, 80.0])


def test_mesh_conversion_modifies_and_sorts_original_dataframe():
    mesh = pd.DataFrame({"mesh_stratum": [2, 1], "biomass": [100.0, 80.0]}, index=[8, 4])
    proportions = xr.Dataset(
        {
            "proportion": xr.DataArray(
                [[0.25, 0.75], [0.6, 0.4]],
                dims=["stratum", "sex"],
                coords={"stratum": [1, 2], "sex": ["female", "male"]},
            )
        }
    )
    weights = xr.DataArray([2.0, 5.0], dims="stratum", coords={"stratum": [1, 2]})
    sigma_bs = pd.DataFrame({"sigma_bs": [0.1, 0.2]}, index=pd.Index([1, 2], name="stratum"))

    result = mesh_biomass_to_nasc(
        mesh,
        biodata=proportions,
        mesh_biodata_link={"mesh_stratum": "stratum"},
        stratum_weights=weights,
        stratum_sigma_bs=sigma_bs,
        group_columns=["stratum", "sex"],
    )

    assert result is None
    assert mesh["mesh_stratum"].tolist() == [1, 2]
    assert mesh.index.equals(pd.RangeIndex(2))
    np.testing.assert_allclose(mesh["biomass_female"], [20.0, 60.0])
    np.testing.assert_allclose(mesh["biomass_male"], [60.0, 40.0])
    np.testing.assert_allclose(mesh["abundance"], [40.0, 20.0])
    np.testing.assert_allclose(mesh["nasc"], np.array([4.0, 4.0]) * 4.0 * np.pi)
