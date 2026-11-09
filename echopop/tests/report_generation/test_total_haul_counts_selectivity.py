"""Regression tests for combined specimen and length haul counts."""

import pandas as pd
import pytest

from echopop.reports import Reporter


@pytest.fixture
def biodata():
    specimen = pd.DataFrame(
        {
            "haul_num": [1, 1, 2, 2],
            "sex": ["male", "female", "male", "unsexed"],
            "length": [15.0] * 4,
            "length_bin": pd.cut([15.0] * 4, [10, 20, 30]),
            # Corrected specimens mixed with an opted-out gear's count of one.
            "length_count": [1.25, 2.5, 1.0, 1.75],
        }
    )
    length = specimen.copy()
    length["length_count"] = [10.5, 20.25, 30.0, 40.75]
    return {"specimen": specimen, "length": length}


@pytest.mark.parametrize("mode", [None, False, True])
def test_total_haul_workbook(tmp_path, biodata, mode):
    original = {key: data.copy(deep=True) for key, data in biodata.items()}
    options = {} if mode is None else {"net_selectivity": mode}
    Reporter(tmp_path, verbose=False).total_length_haul_counts_report(
        "counts.xlsx",
        {sex: sex for sex in ["male", "female", "all"]},
        biodata,
        **options,
    )
    tables = pd.read_excel(tmp_path / "counts.xlsx", sheet_name=None, header=1, index_col=0)
    if mode:
        expected = {
            "male": [11.75, 31.0, 42.75],
            "female": [22.75, 0.0, 22.75],
            "all": [34.5, 73.5, 108.0],
        }
    else:
        # Preserve the existing integer conversion when correction is disabled.
        expected = {
            "male": [11.0, 31.0, 42.0],
            "female": [21.0, 0.0, 21.0],
            "all": [32.0, 72.0, 104.0],
        }
    for sex, (haul1, haul2, total) in expected.items():
        table = tables[sex]
        assert table.loc[15.0, 1] == pytest.approx(haul1)
        assert table.loc[15.0, 2] == pytest.approx(haul2)
        assert table.loc[15.0, "Subtotal"] == pytest.approx(total)
        assert table.loc["Subtotal", 1] == pytest.approx(haul1)
        assert table.loc["Subtotal", 2] == pytest.approx(haul2)
        assert table.loc["Total", 1] == pytest.approx(total)
    for key in biodata:
        pd.testing.assert_frame_equal(biodata[key], original[key])


def test_total_missing_expanded_counts(tmp_path, biodata):
    report = Reporter(tmp_path, verbose=False)
    sheets = {sex: sex for sex in ["male", "female", "all"]}
    biodata["specimen"] = biodata["specimen"].drop(columns="length_count")
    report.total_length_haul_counts_report("observed.xlsx", sheets, biodata)
    with pytest.raises(KeyError, match="length_count"):
        report.total_length_haul_counts_report(
            "expanded.xlsx", sheets, biodata, net_selectivity=True,
        )


def test_total_invalid_selectivity_flag(tmp_path, biodata):
    with pytest.raises(TypeError, match="net_selectivity"):
        Reporter(tmp_path, verbose=False).total_length_haul_counts_report(
            "counts.xlsx", {}, biodata, net_selectivity="false",
        )
