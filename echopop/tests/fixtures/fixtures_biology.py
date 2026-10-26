import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def length_weight_data():
    """Create sample length-weight data for regression testing."""
    return pd.DataFrame(
        {
            "length": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
            "weight": [5.0, 15.0, 35.0, 70.0, 120.0, 190.0],
            "species_code": [22500, 22500, 22500, 22500, 22500, 22500],
            "haul": [1, 1, 2, 2, 3, 3],
        }
    )


@pytest.fixture
def grouped_length_weight_data():
    """Create sample data with grouping variables for testing groupby operations."""
    return pd.DataFrame(
        {
            "length": [10.0, 15.0, 20.0, 25.0, 12.0, 18.0, 24.0, 28.0],
            "weight": [5.0, 15.0, 35.0, 70.0, 8.0, 25.0, 50.0, 85.0],
            "sex": ["male", "male", "male", "male", "female", "female", "female", "female"],
            "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "species_code": [22500, 22500, 22500, 22500, 22500, 22500, 22500, 22500],
        }
    )


@pytest.fixture
def data_with_missing_values():
    """Create data with missing length and weight values."""
    return pd.DataFrame(
        {
            "length": [10.0, np.nan, 20.0, 25.0, np.nan],
            "weight": [5.0, 15.0, np.nan, 70.0, 120.0],
            "species_code": [22500, 22500, 22500, 22500, 22500],
        }
    )


@pytest.fixture
def minimal_data():
    """Create minimal two-point dataset for edge case testing."""
    return pd.DataFrame({"length": [10.0, 20.0], "weight": [5.0, 35.0]})


@pytest.fixture
def single_row_data():
    """Create single-row dataset for error testing."""
    return pd.DataFrame({"length": [10.0], "weight": [5.0]})


@pytest.fixture
def large_dataset():
    """Create larger dataset for performance testing."""
    np.random.seed(42)
    n = 1000
    # Generate realistic length-weight relationship with some noise
    lengths = np.random.uniform(5, 50, n)
    weights = 0.01 * (lengths**3.0) * np.random.uniform(0.8, 1.2, n)

    return pd.DataFrame(
        {
            "length": lengths,
            "weight": weights,
            "haul": np.random.randint(1, 21, n),
            "species_code": np.full(n, 22500),
        }
    )


@pytest.fixture
def zero_negative_data():
    """Create data with zero and negative values for error testing."""
    return pd.DataFrame({"length": [0.0, -5.0, 10.0, 15.0], "weight": [5.0, 10.0, 0.0, -2.0]})


@pytest.fixture
def realistic_fish_data():
    """Create realistic fish length-weight data."""
    return pd.DataFrame(
        {
            "length": [25.5, 32.1, 28.7, 35.9, 41.2, 38.6, 44.8, 51.3],
            "weight": [120.5, 245.8, 175.2, 325.1, 485.7, 410.3, 625.9, 850.4],
            "sex": ["female", "male", "female", "male", "female", "male", "female", "male"],
            "stratum": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "species_code": [22500, 22500, 22500, 22500, 22500, 22500, 22500, 22500],
            "haul": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )


@pytest.fixture
def sample_specimen_data(sample_length_distribution):
    """Create sample specimen data with length and weight."""
    data = pd.DataFrame(
        {
            "length": [12.5, 18.3, 22.7, 27.1, 15.8, 24.2, 19.5, 26.8, 14.2, 21.3],
            "weight": [25.3, 45.7, 68.2, 89.5, 32.1, 72.8, 52.1, 85.2, 28.9, 61.4],
            "sex": [
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
            ],
            "species_id": [22500] * 10,
            "haul_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        }
    )
    # Add length bins
    data["length_bin"] = pd.cut(
        data["length"], bins=sample_length_distribution["interval"].cat.categories
    )
    return data


@pytest.fixture
def specimen_data_with_bins(sample_specimen_data, sample_length_distribution):
    """Create specimen data with length_bin column already added."""
    data = sample_specimen_data.copy()
    # Add length bins manually for testing
    data["length_bin"] = pd.cut(
        data["length"], bins=sample_length_distribution["interval"].cat.categories
    )
    return data


@pytest.fixture
def sample_length_distribution():
    """Create sample length distribution with bins and intervals."""
    bins = np.array([10, 15, 20, 25, 30])
    binwidth = np.mean(np.diff(bins) / 2.0)
    centered_bins = np.concatenate([[bins[0] - binwidth], bins + binwidth])
    intervals = pd.cut(bins, centered_bins)

    return pd.DataFrame({"bin": bins, "interval": intervals})


@pytest.fixture
def sample_length_bin_edges():
    """Create explicit edges for length bins centered on 10, 15, ..., 30."""
    return np.array([7.5, 12.5, 17.5, 22.5, 27.5, 32.5])


@pytest.fixture
def single_regression_coefficients():
    """Create single set of regression coefficients."""
    return pd.Series([3.0, -5.0], index=["slope", "intercept"])


@pytest.fixture
def grouped_regression_coefficients():
    """Create grouped regression coefficients (by sex)."""
    return pd.DataFrame(
        {"slope": [3.1, 2.9], "intercept": [-5.1, -4.9]},
        index=pd.Index(["male", "female"], name="sex"),
    )


@pytest.fixture
def reduced_specimen_data(sample_length_distribution):
    """Create minimal specimen data for edge case testing."""
    data = pd.DataFrame({"length": [15.0, 25.0], "weight": [30.0, 75.0], "sex": ["male", "female"]})
    # Add length bins
    data["length_bin"] = pd.cut(
        data["length"], bins=sample_length_distribution["interval"].cat.categories
    )
    return data


@pytest.fixture
def specimen_data_missing_weights(sample_length_distribution):
    """Create specimen data with some missing weights."""
    data = pd.DataFrame(
        {
            "length": [12.5, 18.3, 22.7, 27.1, 15.8],
            "weight": [25.3, np.nan, 68.2, np.nan, 32.1],
            "sex": ["male", "female", "male", "female", "male"],
        }
    )
    # Add length bins
    data["length_bin"] = pd.cut(
        data["length"], bins=sample_length_distribution["interval"].cat.categories
    )
    return data


@pytest.fixture
def large_specimen_dataset(sample_length_distribution):
    """Create larger specimen dataset for performance testing with length_bin column."""
    np.random.seed(42)
    n = 500

    # Generate realistic length-weight relationship
    lengths = np.random.uniform(10, 30, n)
    weights = 0.01 * (lengths**3.0) * np.random.uniform(0.8, 1.2, n)

    data = pd.DataFrame(
        {
            "length": lengths,
            "weight": weights,
            "sex": np.random.choice(["male", "female"], n),
            "species_id": np.full(n, 22500),
            "haul_id": np.random.randint(1, 11, n),
        }
    )

    # Add length_bin column
    data["length_bin"] = pd.cut(
        data["length"],
        bins=sample_length_distribution["interval"].cat.categories,
        labels=sample_length_distribution["interval"],
    )

    return data


@pytest.fixture
def empty_specimen_data():
    """Create empty specimen DataFrame with correct columns."""
    return pd.DataFrame(columns=["stratum_num", "length", "weight", "sex", "length_bin"])


@pytest.fixture
def null_specimen_data():
    """Create empty specimen DataFrame with correct columns."""
    return pd.DataFrame(columns=["length", "weight", "sex", "species_id", "haul_id", "length_bin"])


@pytest.fixture
def uneven_specimen_data(sample_length_distribution):
    """Create specimen data with uneven distribution across bins with length_bin column."""
    data = pd.DataFrame(
        {
            "length": [12.0, 12.1, 12.2, 22.5, 27.8],  # Most data in first bin
            "weight": [24.5, 25.1, 24.8, 68.0, 92.3],
            "sex": ["male", "male", "female", "female", "male"],
        }
    )

    # Add length_bin column
    data["length_bin"] = pd.cut(
        data["length"],
        bins=sample_length_distribution["interval"].cat.categories,
        labels=sample_length_distribution["interval"],
    )

    return data


@pytest.fixture
def coefficients_with_multiple_groups():
    """Create regression coefficients with multiple grouping variables."""
    index = pd.MultiIndex.from_tuples(
        [("male", "A"), ("male", "B"), ("female", "A"), ("female", "B")], names=["sex", "stratum"]
    )

    return pd.DataFrame(
        {"slope": [3.0, 3.1, 2.9, 3.0], "intercept": [-5.0, -5.1, -4.9, -5.0]}, index=index
    )


@pytest.fixture
def specimen_data_multiple_groups(sample_length_distribution):
    """Create specimen data with multiple grouping variables and length_bin column."""
    data = pd.DataFrame(
        {
            "length": [12.5, 18.3, 22.7, 27.1, 15.8, 24.2, 19.5, 26.8],
            "weight": [25.3, 45.7, 68.2, 89.5, 32.1, 72.8, 52.1, 85.2],
            "sex": ["male", "male", "female", "female", "male", "female", "male", "female"],
            "stratum": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )

    # Add length_bin column
    data["length_bin"] = pd.cut(
        data["length"],
        bins=sample_length_distribution["interval"].cat.categories,
        labels=sample_length_distribution["interval"],
    )

    return data


@pytest.fixture
def invalid_length_distribution():
    """Create invalid length distribution missing required columns."""
    return pd.DataFrame({"size": [10, 15, 20, 25], "range": ["small", "medium", "large", "xlarge"]})


@pytest.fixture
def nasc_data_simple():
    """Return basic NASC data matching the real usage pattern."""
    return pd.DataFrame(
        {
            "number_density": [100.0, 200.0, 150.0, 75.0],
            "area_interval": [2.5, 3.0, 2.8, 1.5],
            "stratum_ks": [1, 2, 1, 2],
        }
    )


@pytest.fixture
def average_weight_series():
    """Average weight as a Series indexed by stratum_ks (real usage pattern)."""
    return pd.Series([0.4, 0.6], index=[1, 2], name="avg_weight")


@pytest.fixture
def scalar_weight():
    """Scalar weight value."""
    return 0.5
