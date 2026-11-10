import numpy as np
import pandas as pd
import pytest
from scipy import interpolate as interp


@pytest.fixture
def sample_specimen_data():
    """Create sample specimen data for testing."""
    return pd.DataFrame(
        {
            "stratum_num": [1, 1, 2, 2, 1, 2, 1, 2, 1, 2],
            "length_bin": pd.Categorical(
                [
                    "(10,15]",
                    "(15,20]",
                    "(20,25]",
                    "(25,30]",
                    "(10,15]",
                    "(15,20]",
                    "(20,25]",
                    "(25,30]",
                    "(15,20]",
                    "(20,25]",
                ]
            ),
            "age_bin": pd.Categorical([1, 2, 3, 4, 1, 2, 3, 4, 2, 3]),
            "sex": [
                "male",
                "female",
                "male",
                "female",
                "unsexed",
                "male",
                "female",
                "unsexed",
                "male",
                "female",
            ],
            "length": [12.5, 18.3, 22.7, 27.1, 13.2, 17.8, 23.5, 28.4, 19.1, 21.6],
            "weight": [25.3, 45.7, 68.2, 89.5, 28.1, 42.3, 71.8, 92.1, 48.2, 63.7],
        }
    )


@pytest.fixture
def sample_length_data():
    """Create sample length data for testing."""
    return pd.DataFrame(
        {
            "stratum_num": [1, 1, 2, 2, 1, 2, 1, 2],
            "length_bin": pd.Categorical(
                [
                    "(10,15]",
                    "(15,20]",
                    "(20,25]",
                    "(25,30]",
                    "(10,15]",
                    "(15,20]",
                    "(20,25]",
                    "(25,30]",
                ]
            ),
            "sex": ["male", "female", "male", "female", "unsexed", "male", "female", "unsexed"],
            "length": [12.8, 17.5, 22.1, 26.9, 14.1, 18.7, 24.3, 27.8],
            "length_count": [15, 23, 18, 12, 8, 19, 14, 9],
        }
    )


@pytest.fixture
def minimal_specimen_data():
    """Create minimal specimen data for edge case testing."""
    return pd.DataFrame(
        {
            "stratum_num": [1, 1],
            "length_bin": pd.Categorical(["(10,15]", "(15,20]"]),
            "length": [12.0, 17.0],
            "sex": ["male", "female"],
        }
    )


@pytest.fixture
def null_specimen_data():
    """Create empty specimen DataFrame."""
    return pd.DataFrame(columns=["stratum_num", "length_bin", "age_bin", "sex", "length", "weight"])


@pytest.fixture
def data_with_missing_columns():
    """Create data missing some expected columns."""
    return pd.DataFrame(
        {
            "stratum_num": [1, 2, 1, 2],
            "length_bin": pd.Categorical(["(10,15]", "(15,20]", "(20,25]", "(25,30]"]),
            "length": [12.5, 18.3, 22.7, 27.1],
            "other_col": ["A", "B", "C", "D"],
        }
    )


@pytest.fixture
def data_all_unsexed():
    """Create data where all fish are unsexed."""
    return pd.DataFrame(
        {
            "stratum_num": [1, 1, 2, 2],
            "length_bin": pd.Categorical(["(10,15]", "(15,20]", "(20,25]", "(25,30]"]),
            "sex": ["unsexed", "unsexed", "unsexed", "unsexed"],
            "length": [12.5, 18.3, 22.7, 27.1],
        }
    )


@pytest.fixture
def expanded_specimen_dataset():
    """Create larger dataset for performance testing."""
    np.random.seed(42)
    n = 1000

    return pd.DataFrame(
        {
            "stratum_num": np.random.randint(1, 6, n),
            "length_bin": pd.Categorical(
                np.random.choice(["(10,15]", "(15,20]", "(20,25]", "(25,30]", "(30,35]"], n)
            ),
            "age_bin": pd.Categorical(np.random.randint(1, 6, n)),
            "sex": np.random.choice(["male", "female", "unsexed"], n),
            "length": np.random.uniform(10, 35, n),
        }
    )


@pytest.fixture
def data_multiple_groups():
    """Create data with multiple grouping variables."""
    return pd.DataFrame(
        {
            "stratum_num": [1, 1, 2, 2, 1, 2],
            "length_bin": pd.Categorical(
                ["(10,15]", "(15,20]", "(20,25]", "(25,30]", "(10,15]", "(15,20]"]
            ),
            "age_bin": pd.Categorical([1, 2, 3, 4, 1, 2]),
            "sex": ["male", "female", "male", "female", "unsexed", "male"],
            "species": ["A", "A", "B", "B", "A", "B"],
            "region": ["North", "South", "North", "South", "North", "South"],
            "length": [12.5, 18.3, 22.7, 27.1, 13.2, 17.8],
        }
    )


@pytest.fixture
def aged_dataframe():
    """Create sample aged count data with stratum, length, age, and sex."""
    data = {
        "stratum_num": [1, 1, 1, 2, 2, 2],
        "length_bin": ["(10, 20]", "(20, 30]", "(30, 40]", "(10, 20]", "(20, 30]", "(30, 40]"],
        "age_bin": ["(1, 2]", "(2, 3]", "(3, 4]", "(1, 2]", "(2, 3]", "(3, 4]"],
        "sex": ["female", "male", "unsexed", "female", "male", "unsexed"],
        "species_id": [1, 1, 1, 1, 1, 1],
        "count": [10, 20, 5, 15, 25, 10],
    }
    return pd.DataFrame(data)


@pytest.fixture
def aged_dataarray(aged_dataframe):
    """Create sample aged count data with stratum, length, age, and sex."""
    dat_da = aged_dataframe.set_index(["stratum_num", "length_bin", "age_bin", "sex"])[
        "count"
    ].to_xarray()
    dat_da = dat_da.assign_coords({"variable": "count"})
    return dat_da


@pytest.fixture
def unaged_dataframe():
    """Create sample unaged count data with stratum, length and sex."""
    data = {
        "stratum_num": [1, 1, 1, 2, 2, 2],
        "length_bin": ["(10, 20]", "(20, 30]", "(30, 40]", "(10, 20]", "(20, 30]", "(30, 40]"],
        "sex": ["female", "male", "unsexed", "female", "male", "unsexed"],
        "species_id": [1, 1, 1, 1, 1, 1],
        "count": [30, 40, 10, 35, 45, 15],
    }
    return pd.DataFrame(data)


@pytest.fixture
def unaged_dataarray(unaged_dataframe):
    """Create sample aged count data with stratum, length, age, and sex."""
    dat_da = unaged_dataframe.set_index(["stratum_num", "length_bin", "sex"])["count"].to_xarray()
    dat_da = dat_da.assign_coords({"variable": "count"})

    return dat_da


@pytest.fixture
def length_dataset_with_bins(grouped_length_weight_data):
    """Create dataset with length bins and counts using pandas Interval."""
    from pandas import Interval

    df = grouped_length_weight_data.copy()

    # Add length bins using pandas Interval objects
    bins = []
    for length in df["length"]:
        bin_min = length - 2.5
        bin_max = length + 2.5
        bins.append(Interval(bin_min, bin_max, closed="right"))

    df["length_bin"] = bins

    # Add length counts
    df["length_count"] = [10, 15, 20, 5, 8, 12, 18, 6]

    return df


@pytest.fixture
def length_weight_dataset_with_bins(grouped_length_weight_data):
    """Create length-weight dataset with bins and fitted weights using pandas Interval."""
    from pandas import Interval

    df = grouped_length_weight_data.copy()

    # Add length bins using pandas Interval
    bins = []
    for length in df["length"]:
        bin_min = length - 2.5
        bin_max = length + 2.5
        bins.append(Interval(bin_min, bin_max, closed="right"))

    df["length_bin"] = bins

    # Add fitted weights (simulate a simple length-weight relationship)
    df["weight_fitted"] = df["length"] ** 3 * 0.01

    return df


@pytest.fixture
def length_weight_dataset_wide_format(grouped_length_weight_data):
    """Create length-weight dataset in long format for binned_weights function."""
    from pandas import Interval

    df = grouped_length_weight_data.copy()

    # Add length bins using pandas Interval
    bins = []
    for length in df["length"]:
        bin_min = length - 2.5
        bin_max = length + 2.5
        bins.append(Interval(bin_min, bin_max, closed="right"))

    df["length_bin"] = bins

    # Add fitted weights (simulate a simple length-weight relationship)
    df["weight_fitted"] = df["length"] ** 3 * 0.01

    # Return long format DataFrame (not pivot table) as expected by binned_weights function
    # This should match the long format returned by length_binned_weights
    return (
        df[["sex", "length_bin", "weight_fitted"]]
        .set_index(["sex", "length_bin"])["weight_fitted"]
        .to_xarray()
    )


@pytest.fixture
def real_interpolators():
    """Create real interpolator functions for testing."""
    # Male interpolator: f(x) = x * 3.0
    x_male = np.array([10.0, 20.0, 30.0])
    y_male = np.array([30.0, 60.0, 90.0])
    male_interp = interp.interp1d(
        x_male, y_male, kind="linear", bounds_error=False, fill_value=(y_male[0], y_male[-1])
    )

    # Female interpolator: f(x) = x * 2.0
    x_female = np.array([10.0, 20.0, 30.0])
    y_female = np.array([20.0, 40.0, 60.0])
    female_interp = interp.interp1d(
        x_female,
        y_female,
        kind="linear",
        bounds_error=False,
        fill_value=(y_female[0], y_female[-1]),
    )

    # Global interpolator: f(x) = x * 2.5
    x_global = np.array([10.0, 20.0, 30.0])
    y_global = np.array([25.0, 50.0, 75.0])
    global_interp = interp.interp1d(
        x_global,
        y_global,
        kind="linear",
        bounds_error=False,
        fill_value=(y_global[0], y_global[-1]),
    )

    return {"male": male_interp, "female": female_interp, "_global_": global_interp}


@pytest.fixture
def proportion_dict():
    """Create sample proportion dictionaries for testing."""
    # First group - aged data
    aged_data = pd.DataFrame(
        {
            "stratum_num": [1, 1, 1, 2, 2, 2],
            "sex": ["female", "male", "male", "female", "male", "female"],
            "length_bin": ["(10, 20]", "(10, 20]", "(20, 30]", "(10, 20]", "(20, 30]", "(20, 30]"],
            "proportion": [0.15, 0.2, 0.05, 0.025, 0.075, 0.05],
        }
    )

    # Second group - unaged data
    unaged_data = pd.DataFrame(
        {
            "stratum_num": [1, 1, 1, 2, 2, 2],
            "sex": ["female", "male", "male", "female", "male", "female"],
            "length_bin": ["(10, 20]", "(10, 20]", "(20, 30]", "(10, 20]", "(20, 30]", "(20, 30]"],
            "proportion": [0.125, 0.175, 0.1, 0.05, 0.1, 0.05],
        }
    )

    return {"aged": aged_data, "unaged": unaged_data}


@pytest.fixture
def weight_table():
    """Create sample weight table for testing."""
    return pd.DataFrame(
        {
            "sex": ["female", "male", "all", "female", "male", "all"],
            "length_bin": ["(10, 20]", "(10, 20]", "(10, 20]", "(20, 30]", "(20, 30]", "(20, 30]"],
            "weight_fitted": [0.5, 0.4, 0.45, 1.2, 1.0, 1.1],
        }
    )


@pytest.fixture
def proportion_test_dict():
    """Create sample proportion dictionary for testing stratum_averaged_weight."""
    # First group - aged data
    aged_data = (
        pd.DataFrame(
            {
                "stratum_num": [1, 1, 1, 2, 2, 2],
                "sex": ["female", "male", "male", "female", "male", "female"],
                "length_bin": [
                    "(10, 20]",
                    "(10, 20]",
                    "(20, 30]",
                    "(10, 20]",
                    "(20, 30]",
                    "(20, 30]",
                ],
                "proportion": [0.15, 0.2, 0.05, 0.025, 0.075, 0.05],
            }
        )
        .set_index(["stratum_num", "sex", "length_bin"])
        .to_xarray()
    )

    # Second group - unaged data
    unaged_data = (
        pd.DataFrame(
            {
                "stratum_num": [1, 1, 1, 2, 2, 2],
                "sex": ["female", "male", "male", "female", "male", "female"],
                "length_bin": [
                    "(10, 20]",
                    "(10, 20]",
                    "(20, 30]",
                    "(10, 20]",
                    "(20, 30]",
                    "(20, 30]",
                ],
                "proportion": [0.125, 0.175, 0.1, 0.05, 0.1, 0.05],
            }
        )
        .set_index(["stratum_num", "sex", "length_bin"])
        .to_xarray()
    )

    return {"aged": aged_data, "unaged": unaged_data}


@pytest.fixture
def test_weight_table():
    """Create sample weight table for testing stratum_averaged_weight in wide format."""
    # Create wide format table with length_bin as index and sex as columns
    data = {
        "female": [0.5, 1.2],
        "male": [0.4, 1.0],
        "all": [0.45, 1.1],
    }
    index = pd.Index(["(10, 20]", "(20, 30]"], name="length_bin")
    data_df = pd.DataFrame(data, index=index)
    data_df.columns = data_df.columns.rename("sex")

    return (
        data_df.stack()
        .to_frame(name="weight_fitted")
        .to_xarray()
        .to_dataarray(dim="weight_fitted")
        .squeeze()
        .rename("weight_fitted")
        .reset_coords("weight_fitted", drop=True)
    )


@pytest.fixture
def weights_df_fixture():
    """Create a DataFrame with the correct structure for weight distributions."""
    # Create multi-level columns for sex and stratum_num
    columns = pd.MultiIndex.from_tuples(
        [("female", 1), ("female", 2), ("male", 1), ("male", 2)], names=["sex", "stratum_num"]
    )

    # Create simple data
    data = [[10.5, 15.2, 8.3, 12.7]]

    return pd.DataFrame(data=data, columns=columns)


@pytest.fixture
def weight_distr_dict(weights_df_fixture):
    """Create a dictionary with weight distribution DataFrames."""
    # Create a variation for the unaged group
    aged_da = (
        weights_df_fixture.T.rename({0: "weight"}, axis=1)
        .to_xarray()
        .to_dataarray()
        .squeeze()
        .rename("weight")
        .reset_coords("variable", drop=True)
    )
    unaged_da = (
        weights_df_fixture.T.rename({0: "weight"}, axis=1)
        .to_xarray()
        .assign(weight=lambda x: x.weight + 2)
        .to_dataarray()
        .squeeze()
        .rename("weight")
        .reset_coords("variable", drop=True)
    )

    return {"aged": aged_da, "unaged": unaged_da}


@pytest.fixture
def catch_data_df():
    """Create simple catch data with the correct structure."""
    return pd.DataFrame(
        {
            "stratum_num": [1, 1, 2, 2],
            "haul_num": [101, 102, 201, 202],
            "weight": [50.0, 30.0, 65.0, 45.0],  # Changed from haul_weight to weight
        }
    )


@pytest.fixture
def weights_df_multilevel():
    """Create a multi-level column DataFrame with stratum weights for testing."""
    # Create multi-level columns with stratum_num
    columns = pd.MultiIndex.from_tuples(
        [
            ("sex", "male", "stratum_num", 1),
            ("sex", "male", "stratum_num", 2),
            ("sex", "female", "stratum_num", 1),
            ("sex", "female", "stratum_num", 2),
            ("sex", "unsexed", "stratum_num", 1),
            ("sex", "unsexed", "stratum_num", 2),
        ],
        names=["group", "category", "strata", "stratum_num"],
    )

    # Create data with one row
    data = np.array([[10.5, 15.2, 8.3, 12.7, 3.1, 4.5]])

    return pd.DataFrame(data=data, columns=columns)


@pytest.fixture
def reference_stratum_weights():
    """Create reference stratum weights DataFrame for testing."""
    return pd.DataFrame({"stratum_num": [1, 2], "weight": [100.0, 150.0]})


@pytest.fixture
def empty_weights_df_multilevel():
    """Create an empty multi-level column DataFrame for testing edge cases."""
    columns = pd.MultiIndex.from_tuples(
        [
            ("sex", "male", "stratum_num", 1),
            ("sex", "male", "stratum_num", 2),
        ],
        names=["group", "category", "strata", "stratum_num"],
    )

    return pd.DataFrame(columns=columns)


@pytest.fixture
def weights_df_missing_stratum():
    """Create a multi-level DataFrame without stratum_num level for testing."""
    columns = pd.MultiIndex.from_tuples(
        [
            ("sex", "male", "region", 1),
            ("sex", "male", "region", 2),
            ("sex", "female", "region", 1),
            ("sex", "female", "region", 2),
        ],
        names=["group", "category", "geography", "region_id"],
    )

    data = np.array([[10.5, 15.2, 8.3, 12.7]])

    return pd.DataFrame(data=data, columns=columns)


@pytest.fixture
def expected_proportions():
    """Create expected output for weight_proportions function."""
    # Calculated based on the test data:
    # For stratum 1: total weight = 80.0 + 21.9 = 101.9
    # For stratum 2: total weight = 110.0 + 32.4 = 142.4

    # For male, stratum 1: 10.5 / 101.9 = ~0.103
    # For male, stratum 2: 15.2 / 142.4 = ~0.107
    # Similar calculations for female and unsexed

    # Structure follows the data_pvt in the function
    index = pd.MultiIndex.from_tuples(
        [("sex", "male"), ("sex", "female"), ("sex", "unsexed")], names=["group", "category"]
    )

    return pd.DataFrame(
        {1: [10.5 / 101.9, 8.3 / 101.9, 3.1 / 101.9], 2: [15.2 / 142.4, 12.7 / 142.4, 4.5 / 142.4]},
        index=index,
    )


@pytest.fixture
def proportion_dict_fixture():
    """Create proportion_dict with 'unaged' key for group."""
    unaged_data = pd.DataFrame(
        {
            "stratum_num": [1, 1, 2, 2],
            "sex": ["male", "female", "male", "female"],
            "length_bin": [30, 30, 30, 40],
            "proportion": [0.3, 0.2, 0.3, 0.2],
        }
    )

    # Return as dictionary with 'unaged' key
    return {"unaged": unaged_data}


@pytest.fixture
def binned_weight_table_fixture():
    """Create binned_weight_table Series for the 'all' column."""
    # Return just the "all" column as a Series with length_bin as index
    index = pd.Index([30, 40], name="length_bin")
    return pd.Series([0.5, 1.2], index=index, name="all")


@pytest.fixture
def standardized_weight_reference():
    """Create reference data with proper MultiIndex structure."""
    # Create index for reference data
    idx = pd.MultiIndex.from_product(
        [
            ["(20.0, 30.0]", "(30.0, 40.0]"],  # length_bin
            ["(1.5, 2.5]"],  # age_bin
            ["male", "female", "unsexed"],  # sex
        ],
        names=["length_bin", "age_bin", "sex"],
    )

    # Create MultiIndex columns with stratum_num
    cols = pd.Index([1, 2], name="stratum_num")

    # Create data
    data = np.zeros((len(idx), len(cols)))
    data[0, 0] = 0.3  # First row, stratum 1
    data[1, 0] = 0.2  # Second row, stratum 1
    data[2, 0] = 0.1  # Third row, stratum 1
    data[0, 1] = 0.4  # First row, stratum 2
    data[1, 1] = 0.3  # Second row, stratum 2
    data[2, 1] = 0.1  # Third row, stratum 2

    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture
def standardized_data_fixture():
    """Create a properly structured weight_data fixture."""
    # Create a DataFrame with sex as row index and stratum_num as columns
    data = {1: [0.15, 0.25], 2: [0.08, 0.12]}  # stratum 1 values  # stratum 2 values

    # Create DataFrame with proper index name
    df = pd.DataFrame(data, index=pd.Index(["female", "male"], name="sex"))

    # Ensure column index is properly named
    df.columns.name = "stratum_num"

    return df


@pytest.fixture
def simple_weights_df():
    """Create a simple multi-level column DataFrame with stratum_num in columns."""
    # Create simple data with one row
    data = pd.DataFrame(
        {
            "sex": ["female", "female", "male", "male"],
            "stratum_num": [1, 2, 1, 2],
            "weight": [10.5, 15.2, 8.3, 12.7],
        }
    ).set_index(["stratum_num", "sex"])

    return data.to_xarray()["weight"]


@pytest.fixture
def simple_stratum_weights():
    """Create simple stratum weights DataFrame for testing."""
    return pd.DataFrame({"stratum_num": [1, 2], "weight": [100.0, 150.0]})


@pytest.fixture
def number_proportions_data():
    """Create realistic test data for number proportions."""
    np.random.seed(42)  # For reproducible tests

    # Create length bins
    length_bins = pd.interval_range(start=10, end=80, freq=5, closed="left")

    # Create test data structure
    data = []
    for stratum in [1, 2, 3]:
        for sex in ["female", "male"]:
            for age in [1, 2, 3, 4]:
                for length_bin in length_bins[:10]:  # Use subset for testing
                    proportion = np.random.exponential(0.1)  # Realistic skewed distribution
                    data.append(
                        {
                            "stratum_ks": stratum,
                            "sex": sex,
                            "age_bin": age,
                            "length_bin": length_bin,
                            "proportion": proportion,
                        }
                    )

    df = pd.DataFrame(data)

    # Normalize proportions within each stratum
    df["proportion"] = df.groupby("stratum_ks")["proportion"].transform(lambda x: x / x.sum())

    return df


@pytest.fixture
def weight_proportions_data():
    """Create realistic test data for weight proportions."""
    np.random.seed(43)

    # Create length bins
    length_bins = pd.interval_range(start=10, end=80, freq=5, closed="left")
    strata = [1, 2, 3]

    # Create hierarchical index
    index_tuples = []
    values = []

    for length_bin in length_bins[:10]:
        for sex in ["female", "male"]:
            for age in [1, 2, 3, 4]:
                index_tuples.append((length_bin, sex, age))
                # Weight proportions are typically higher for larger fish
                base_weight = length_bin.mid**2.5  # Allometric scaling
                values.append([base_weight * np.random.exponential(0.1) for _ in strata])

    # Create MultiIndex
    index = pd.MultiIndex.from_tuples(index_tuples, names=["length_bin", "sex", "age_bin"])

    # Create DataFrame
    df = pd.DataFrame(values, index=index, columns=[f"stratum_{i}" for i in strata])

    # Normalize columns
    df = df.div(df.sum(axis=0), axis=1)

    return df


@pytest.fixture
def ts_parameters():
    """Create test target strength parameters."""
    return {"slope": 20.0, "intercept": -68.0}


@pytest.fixture
def stratify_by():
    """Return standard stratification columns."""
    return ["stratum_ks"]


@pytest.fixture
def age1_filter():
    """Return standard age-1 inclusion filter."""
    return {"age_bin": [1]}


@pytest.fixture
def female_filter():
    """Female-only inclusion filter."""
    return {"sex": ["female"]}


@pytest.fixture
def length_threshold():
    """Return standard length threshold for testing."""
    return 15.0


@pytest.fixture
def weight_threshold():
    """Return standard weight proportion threshold."""
    return 1e-10


@pytest.fixture
def number_proportions_dict(number_proportions_data):
    """Return a dictionary of number proportions (aged/unaged format)."""
    return {
        "aged": number_proportions_data,
        "unaged": number_proportions_data.drop(columns=["age_bin"]),
    }


@pytest.fixture
def combined_filters():
    """Complex filter combining age and sex."""
    return {"age_bin": [1], "sex": ["female"]}


@pytest.fixture
def small_length_bins():
    """Small length bins for exclusion testing."""
    return pd.interval_range(start=10, end=20, freq=5, closed="left")


@pytest.fixture
def length_exclusion_filter(small_length_bins):
    """Length exclusion filter for thresholding tests."""
    return {"length_bin": small_length_bins.tolist()}
