import warnings

import numpy as np
import pandas as pd
import pytest

import echopop.utils as utils


@pytest.mark.parametrize(
    "bins, values, edges, codes",
    [
        (
            [2, 4, 6],
            [-100, 1, 2, 3, 3.01, 5, 5.01, 6, 7, 100, np.nan],
            [1, 3, 5, 7],
            [-1, -1, 0, 0, 1, 1, 2, 2, 2, -1, -1],
        ),
        (
            [20, 30, 50],
            [0, 15, 20, 25, 25.01, 30, 40, 40.01, 50, 60, 100, np.nan],
            [15, 25, 40, 60],
            [-1, -1, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1],
        ),
        (
            [0.25, 0.75],
            [-np.inf, -0.01, 0.25, 0.5, 0.51, 0.75, 1.01, np.inf, np.nan],
            [0, 0.5, 1],
            [-1, -1, 0, 0, 1, 1, -1, -1, -1],
        ),
    ],
    ids=["uniform", "uneven", "two-float-bins-and-infinities"],
)
def test_binify_outputs(bins, values, edges, codes):
    frame = pd.DataFrame({"length": values}, index=np.arange(len(values)) + 10)
    original = frame.copy()
    assert utils.binify(frame, bins, "length") is None
    expected = pd.Categorical.from_codes(
        codes,
        categories=pd.IntervalIndex.from_breaks(np.asarray(edges, dtype=float), closed="right"),
        ordered=True,
    )
    pd.testing.assert_series_equal(
        frame["length_bin"], pd.Series(expected, index=frame.index, name="length_bin")
    )
    pd.testing.assert_frame_equal(frame[["length"]], original)
    assert list(frame) == ["length", "length_bin"]


@pytest.mark.parametrize(
    "column, bins, values, codes",
    [
        ("age", np.linspace(1, 22, 22), [0, 1, 2, 21, 22, 23], [-1, 0, 1, 20, 21, -1]),
        (
            "length",
            np.linspace(2, 80, 40),
            [0, 1, 2, 3, 4, 80, 81, 82],
            [-1, -1, 0, 0, 1, 39, 39, -1],
        ),
    ],
)
def test_workflow_bins(column, bins, values, codes):
    frame = pd.DataFrame({column: values})
    utils.binify(frame, bins, column)
    np.testing.assert_array_equal(frame[f"{column}_bin"].cat.codes, codes)
    np.testing.assert_allclose(frame[f"{column}_bin"].cat.categories.mid, bins)


def test_dictionary_and_multiple_columns():
    target = pd.DataFrame({"length": [20, 50], "age": [1, 2]})
    missing = pd.DataFrame({"other": [1]})
    original_missing = missing.copy()
    data = {"target": target, "missing": missing, "metadata": {"source": "test"}}
    utils.binify(data, [20, 30, 50], "length")
    utils.binify(data, [1, 2], "age")
    assert target["length_bin"].tolist() == [pd.Interval(15, 25), pd.Interval(40, 60)]
    assert target["age_bin"].tolist() == [pd.Interval(0.5, 1.5), pd.Interval(1.5, 2.5)]
    pd.testing.assert_frame_equal(missing, original_missing)
    assert data["metadata"] == {"source": "test"}


@pytest.mark.parametrize("has_column", [True, False])
def test_empty_frame(has_column):
    frame = pd.DataFrame({"age": pd.Series(dtype="Int64")}) if has_column else pd.DataFrame()
    utils.binify(frame, [1, 2], "age")
    assert frame.empty
    assert ("age_bin" in frame) == has_column
    if has_column:
        pd.testing.assert_index_equal(
            frame["age_bin"].cat.categories,
            pd.IntervalIndex.from_breaks([0.5, 1.5, 2.5]),
        )


def test_missing_column_leaves_frame_unchanged():
    frame = pd.DataFrame({"other": [1, 2]})
    original = frame.copy()
    utils.binify(frame, [1, 2], "age")
    pd.testing.assert_frame_equal(frame, original)


def test_nullable_values():
    frame = pd.DataFrame({"age": pd.Series([1, pd.NA, 2], dtype="Int64")})
    utils.binify(frame, [1, 2], "age")
    np.testing.assert_array_equal(frame["age_bin"].cat.codes, [0, -1, 1])


@pytest.mark.parametrize("bins", [[1], [], [[1, 2]], [2, 1], [1, 1], [1, np.nan], [1, np.inf]])
def test_invalid_bins(bins):
    frame = pd.DataFrame({"age": [1]})
    with pytest.raises(ValueError, match="bins must"):
        utils.binify(frame, bins, "age")
    assert list(frame) == ["age"]


def test_invalid_data():
    with pytest.raises(TypeError, match="data must be DataFrame or dict of DataFrames"):
        utils.binify("invalid", [1, 2], "age")


@pytest.mark.parametrize(
    "column, bins, values, expected_codes",
    [
        ("age", [1, 2], [0, 1, 2, 3, None], [-1, 0, 1, -1, -1]),
        ("length", [2, 4], [1, 2, 5, 6, None], [-1, 0, 1, -1, -1]),
    ],
)
def test_warns_for_out_of_range_values(column, bins, values, expected_codes):
    frame = pd.DataFrame({column: pd.Series(values, dtype="Int64")})
    with pytest.warns(
        UserWarning, match=f"2 nonmissing value.*'{column}'.*binning range"
    ) as caught:
        utils.binify(frame, bins, column)
    assert len(caught) == 1
    np.testing.assert_array_equal(frame[f"{column}_bin"].cat.codes, expected_codes)
    assert len(frame) == len(values)


def test_missing_and_in_range_values_do_not_warn():
    frame = pd.DataFrame({"age": pd.Series([1, 2, pd.NA], dtype="Int64")})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        utils.binify(frame, [1, 2], "age")
    assert not caught


def test_dictionary_warns_once_per_affected_frame():
    frames = {
        "first": pd.DataFrame({"age": [0, 3]}),
        "second": pd.DataFrame({"age": [3]}),
        "valid": pd.DataFrame({"age": [1, 2]}),
        "missing_column": pd.DataFrame({"other": [0]}),
    }
    with pytest.warns(UserWarning) as caught:
        utils.binify(frames, [1, 2], "age")
    assert len(caught) == 2
    assert str(caught[0].message).startswith("2 nonmissing")
    assert str(caught[1].message).startswith("1 nonmissing")
