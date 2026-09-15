import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from echopop.survey import selectivity


def expected_selectivity_expansion(length, l50, sr, minimum_selectivity=1e-12):
    """Compute the expected selectivity expansion from the logistic formula."""
    selectivity_value = 1.0 / (1.0 + np.exp(2.0 * np.log(3.0) * (l50 - length) / sr))
    return 1.0 / max(selectivity_value, minimum_selectivity)


def test_get_l50_sr_from_regression_coefficients():
    """Regression coefficients are converted to L50 and SR."""
    expected_l50 = 14.6
    expected_sr = 15.3
    slope = 2.0 * np.log(3.0) / expected_sr
    intercept = -expected_l50 * slope

    l50, sr = selectivity.get_l50_sr({"intercept": intercept, "slope": slope})

    assert l50 == pytest.approx(expected_l50)
    assert sr == pytest.approx(expected_sr)


def test_get_l50_sr_from_metric_parameters():
    """L50/SR inputs are returned unchanged."""
    l50, sr = selectivity.get_l50_sr({"l50": 14.6, "sr": 15.3})

    assert l50 == 14.6
    assert sr == 15.3


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"intercept": -2.09},
            "Missing parameters: Must provide a complete set of 'intercept'/'slope' or 'l50'/'sr'.",
        ),
        (
            {"intercept": -2.09, "l50": 14.6},
            "Missing parameters: Must provide a complete set of 'intercept'/'slope' or 'l50'/'sr'.",
        ),
        (
            {"intercept": -2.09, "slope": 0.14, "l50": 14.6, "sr": 15.3},
            "Ambiguous parameters: Provide either 'intercept'/'slope' or 'l50'/'sr'.",
        ),
    ],
)
def test_get_l50_sr_rejects_invalid_parameter_sets(params, message):
    """Invalid selectivity parameter combinations raise validation errors."""
    with pytest.raises(ValidationError, match=message):
        selectivity.get_l50_sr(params)


def test_assign_selectivity_expansion_with_global_parameters():
    """Global L50/SR parameters match the logistic formula across representative lengths."""
    biodata = pd.DataFrame(
        {
            "net_num": [317, 317, 317],
            "length": [10.0, 14.6, 29.9],
        }
    )
    l50 = 14.6
    sr = 15.3

    expanded_biodata = selectivity.assign_selectivity_expansion(
        biodata=biodata,
        net_selectivity_params={"l50": l50, "sr": sr},
        net_column="net_num",
    )
    expected = [expected_selectivity_expansion(length, l50, sr) for length in biodata["length"]]

    assert expanded_biodata["l50"].tolist() == [l50, l50, l50]
    assert expanded_biodata["sr"].tolist() == [sr, sr, sr]
    assert expanded_biodata["selectivity_expansion"].tolist() == pytest.approx(expected)


def test_assign_selectivity_expansion_from_regression_coefficients_matches_formula():
    """Regression-coefficient inputs produce expansion values from the same logistic formula."""
    l50 = 14.6
    sr = 15.3
    slope = 2.0 * np.log(3.0) / sr
    intercept = -l50 * slope
    biodata = pd.DataFrame(
        {
            "net_num": [317, 317, 317],
            "length": [10.0, 14.6, 29.9],
        }
    )

    expanded_biodata = selectivity.assign_selectivity_expansion(
        biodata=biodata,
        net_selectivity_params={"intercept": intercept, "slope": slope},
        net_column="net_num",
    )
    expected = [expected_selectivity_expansion(length, l50, sr) for length in biodata["length"]]

    assert expanded_biodata["l50"].tolist() == pytest.approx([l50, l50, l50])
    assert expanded_biodata["sr"].tolist() == pytest.approx([sr, sr, sr])
    assert expanded_biodata["selectivity_expansion"].tolist() == pytest.approx(expected)


def test_assign_selectivity_expansion_with_nested_parameters_and_unmapped_net():
    """Per-net parameters support both input styles and preserve unmapped nets as NaN."""
    expected_l50_b = 20.0
    expected_sr_b = 10.0
    slope_b = 2.0 * np.log(3.0) / expected_sr_b
    intercept_b = -expected_l50_b * slope_b

    biodata = pd.DataFrame(
        {
            "net_num": ["AWT", "IKMT", "UNKNOWN"],
            "length": [10.0, 20.0, 15.0],
        }
    )

    expanded_biodata = selectivity.assign_selectivity_expansion(
        biodata=biodata,
        net_selectivity_params={
            "AWT": {"l50": 10.0, "sr": 4.0},
            "IKMT": {"intercept": intercept_b, "slope": slope_b},
        },
        net_column="net_num",
    )
    expected = [
        expected_selectivity_expansion(10.0, 10.0, 4.0),
        expected_selectivity_expansion(20.0, expected_l50_b, expected_sr_b),
    ]

    assert expanded_biodata["l50"].tolist()[:2] == pytest.approx([10.0, expected_l50_b])
    assert expanded_biodata["sr"].tolist()[:2] == pytest.approx([4.0, expected_sr_b])
    assert pd.isna(expanded_biodata.loc[2, "l50"])
    assert pd.isna(expanded_biodata.loc[2, "sr"])
    assert expanded_biodata["selectivity_expansion"].tolist()[:2] == pytest.approx(expected)
    assert pd.isna(expanded_biodata.loc[2, "selectivity_expansion"])


def test_assign_selectivity_expansion_respects_minimum_selectivity():
    """Very small selectivity values are bounded before inversion."""
    biodata = pd.DataFrame({"net_num": [317], "length": [0.0]})

    expanded_biodata = selectivity.assign_selectivity_expansion(
        biodata=biodata,
        net_selectivity_params={"l50": 100.0, "sr": 1.0},
        net_column="net_num",
        minimum_selectivity=1e-6,
    )

    assert expanded_biodata.loc[0, "selectivity_expansion"] == pytest.approx(1e6)


def test_assign_selectivity_expansion_requires_net_column():
    """Missing net identifier column raises a KeyError."""
    with pytest.raises(KeyError, match="Column 'net_num' not found in 'biodata'."):
        selectivity.assign_selectivity_expansion(
            biodata=pd.DataFrame({"length": [10.0]}),
            net_selectivity_params={"l50": 14.6, "sr": 15.3},
            net_column="net_num",
        )


def test_assign_selectivity_expansion_requires_length_column():
    """Missing length column raises a KeyError."""
    with pytest.raises(KeyError, match="Column 'length' not found in 'biodata'."):
        selectivity.assign_selectivity_expansion(
            biodata=pd.DataFrame({"net_num": [317]}),
            net_selectivity_params={"l50": 14.6, "sr": 15.3},
            net_column="net_num",
        )
