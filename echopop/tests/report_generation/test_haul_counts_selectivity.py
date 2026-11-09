"""Regression tests for observed and expanded specimen haul reports."""

import pandas as pd
import pytest

from echopop.reports import Reporter
from echopop.reports.reporter import pivot_haul_tables


@pytest.fixture
def specimens():
    return pd.DataFrame(
        {
            "haul_num": [1, 1, 2, 2],
            "sex": ["male", "female", "male", "unsexed"],
            "length": [15.0, 15.0, 15.0, 15.0],
            "length_bin": pd.cut([15.0] * 4, [10, 20, 30]),
            # A mix of expanded specimens and an opted-out gear's count of one.
            "length_count": [1.25, 2.5, 1.0, 1.75],
        }
    )


@pytest.mark.parametrize("mode", [None, False, True])
def test_aged_haul_workbook(tmp_path, specimens, mode):
    original = specimens.copy(deep=True)
    options = {} if mode is None else {"net_selectivity": mode}
    Reporter(tmp_path, verbose=False).aged_length_haul_counts_report(
        "counts.xlsx",
        {sex: sex for sex in ["male", "female", "all"]},
        specimens,
        **options,
    )
    tables = pd.read_excel(tmp_path / "counts.xlsx", sheet_name=None, header=1, index_col=0)
    expected = (
        {
            "male": [1.25, 1.0, 2.25],
            "female": [2.5, 0.0, 2.5],
            "all": [3.75, 2.75, 6.5],
        }
        if mode
        else {
            "male": [1.0, 1.0, 2.0],
            "female": [1.0, 0.0, 1.0],
            "all": [2.0, 2.0, 4.0],
        }
    )
    for sex, (haul1, haul2, total) in expected.items():
        table = tables[sex]
        assert table.loc[15.0, 1] == pytest.approx(haul1)
        assert table.loc[15.0, 2] == pytest.approx(haul2)
        assert table.loc[15.0, "Subtotal"] == pytest.approx(total)
        assert table.loc["Subtotal", 1] == pytest.approx(haul1)
        assert table.loc["Subtotal", 2] == pytest.approx(haul2)
        assert table.loc["Total", 1] == pytest.approx(total)
    pd.testing.assert_frame_equal(specimens, original)


def test_missing_expanded_counts(tmp_path, specimens):
    report = Reporter(tmp_path, verbose=False)
    sheets = {sex: sex for sex in ["male", "female", "all"]}
    data = specimens.drop(columns="length_count")
    report.aged_length_haul_counts_report("observed.xlsx", sheets, data)
    with pytest.raises(KeyError, match="length_count"):
        report.aged_length_haul_counts_report(
            "expanded.xlsx",
            sheets,
            data,
            net_selectivity=True,
        )


def test_fractional_haul_subtotals():
    columns = pd.MultiIndex.from_product(
        [
            ["male"],
            pd.CategoricalIndex(pd.IntervalIndex.from_breaks([10, 20, 30])),
        ]
    )
    table = pivot_haul_tables(
        pd.DataFrame([[1.25, 2.5], [1.0, 1.75]], index=[1, 2], columns=columns),
        "male",
    )
    assert table.loc[15.0, "Subtotal"] == pytest.approx(2.25)
    assert table.loc[25.0, "Subtotal"] == pytest.approx(4.25)
    assert table.loc["Subtotal", 1] == pytest.approx(3.75)
    assert table.loc["Subtotal", 2] == pytest.approx(2.75)
