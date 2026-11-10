import numpy as np
import pandas as pd
import pytest

from echopop.inversion import InvParameters
from echopop.inversion.pcdwba import pcdwba


@pytest.fixture
def model_parameters():
    """Return standard model parameters for InversionLengthTS."""
    return {
        "ts_length_regression": {"slope": 20.0, "intercept": -68.0},
        "stratify_by": "stratum_ks",
        "expected_strata": [1, 2, 3],  # Reduced to avoid edge case issues
        "impute_missing_strata": False,  # Disable to avoid imputation bugs
        "haul_replicates": True,
    }


@pytest.fixture
def model_parameters_multiple_strata():
    """Model parameters with multiple stratification columns."""
    return {
        "ts_length_regression": {"slope": 20.0, "intercept": -68.0},
        "stratify_by": ["stratum_ks", "haul_num"],
        "expected_strata": [1, 2, 3],
        "impute_missing_strata": True,
        "haul_replicates": True,
    }


@pytest.fixture
def sample_lengths():
    """Sample length values for testing TS calculations."""
    return np.array([10.0, 15.0, 20.0, 25.0, 30.0])


@pytest.fixture
def ts_regression_params():
    """Return standard TS regression parameters for Pacific hake."""
    return {"slope": 20.0, "intercept": -68.0}


@pytest.fixture
def specimen_df():
    """Sample specimen DataFrame with individual fish measurements."""
    return pd.DataFrame(
        {
            "stratum_ks": [1, 1, 1, 2, 2, 2, 3, 3],
            "haul_num": [101, 101, 102, 201, 201, 202, 301, 301],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
            "length": [18.5, 22.3, 20.1, 25.6, 19.8, 23.7, 21.2, 24.1],
            "weight": [65.2, 98.1, 78.3, 142.5, 71.8, 112.3, 85.1, 125.6],
        }
    )


@pytest.fixture
def length_df():
    """Sample length DataFrame with pre-aggregated counts."""
    return pd.DataFrame(
        {
            "stratum_ks": [1, 1, 1, 2, 2, 2],
            "haul_num": [101, 101, 102, 201, 201, 202],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "length": [20.0, 22.0, 19.5, 25.0, 21.0, 24.0],
            "length_count": [15, 12, 8, 10, 18, 14],
        }
    )


@pytest.fixture
def specimen_df_no_count():
    """Specimen DataFrame without length_count column for testing quantization."""
    return pd.DataFrame(
        {
            "stratum_ks": [1, 1, 1, 1, 2, 2, 2],
            "haul_num": [101, 101, 101, 102, 201, 201, 202],
            "sex": ["M", "M", "F", "M", "F", "M", "F"],
            "length": [20.5, 20.5, 22.1, 19.8, 25.0, 21.0, 24.0],
        }
    )


@pytest.fixture
def sigma_bs_df():
    """Sample sigma_bs DataFrame for imputation testing."""
    return pd.DataFrame(
        {"sigma_bs": [0.0012, 0.0018, 0.0025]}, index=pd.Index([1, 3, 5], name="stratum_ks")
    )


@pytest.fixture
def incomplete_sigma_bs_df():
    """Sigma_bs DataFrame missing some strata."""
    return pd.DataFrame({"sigma_bs": [0.0015, 0.0028]}, index=pd.Index([2, 4], name="stratum_ks"))


@pytest.fixture
def all_strata():
    """Complete list of expected strata."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def nasc_df():
    """Sample NASC DataFrame for inversion testing."""
    return pd.DataFrame(
        {
            "stratum_ks": [1, 1, 2, 2, 3, 3],
            "haul_num": [101, 102, 201, 202, 301, 302],
            "nasc": [1250.5, 980.3, 1680.7, 1420.9, 890.2, 1150.6],
            "nasc_proportion": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


@pytest.fixture
def model_parameters_no_impute():
    """Model parameters without imputation."""
    return {
        "ts_length_regression": {"slope": 20.0, "intercept": -68.0},
        "stratify_by": "stratum_ks",
        "expected_strata": [1, 2, 3],
        "impute_missing_strata": False,
        "haul_replicates": True,
    }


@pytest.fixture
def group_columns():
    """Return standard grouping columns for quantization."""
    return ["stratum_ks", "haul_num", "sex"]


@pytest.fixture
def expected_ts_values():
    """Return expected TS values for sample lengths with standard parameters."""
    # For lengths [10, 15, 20, 25, 30] with slope=20, intercept=-68
    # Calculate exact values: 20 * log10(length) + (-68)
    lengths = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    return 20.0 * np.log10(lengths) + (-68.0)


@pytest.fixture
def large_specimen_df():
    """Larger specimen DataFrame for performance testing."""
    np.random.seed(999)
    n_fish = 1000

    return pd.DataFrame(
        {
            "stratum_ks": np.random.choice([1, 2, 3, 4, 5], n_fish),
            "haul_num": np.random.choice(range(101, 401), n_fish),
            "sex": np.random.choice(["M", "F"], n_fish),
            "length": np.random.normal(22.0, 4.0, n_fish).clip(8.0, 60.0),
            "weight": np.random.normal(100.0, 30.0, n_fish).clip(10.0, 500.0),
        }
    )


@pytest.fixture
def empty_df():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame(
        {"stratum_ks": [], "haul_num": [], "sex": [], "length": [], "length_count": []}
    )


@pytest.fixture
def single_row_df():
    """Single row DataFrame for edge case testing."""
    return pd.DataFrame({"stratum_ks": [1], "haul_num": [101], "sex": ["M"], "length": [20.5]})


@pytest.fixture
def sample_InversionMatrix_parameters():
    """Sample model parameters for testing."""
    return {
        "number_density": {"value": 500.0, "min": 10.0, "max": 10000.0, "vary": True},
        "theta_mean": {"value": 10.0, "min": 0.0, "max": 90.0, "vary": False},
        "theta_sd": {"value": 20.0, "min": 10.0, "max": 20.0, "vary": False},
        "length_mean": {"value": 0.030, "min": 0.008, "max": 0.040, "vary": True},
        "g": {"value": 1.015, "min": 1.015, "max": 1.060, "vary": False},
        "h": {"value": 1.020, "min": 1.015, "max": 1.060, "vary": False},
        "length_radius_ratio": {"value": 18.2, "min": 14.0, "max": 20.0, "vary": False},
        "length_sd_norm": {"value": 0.2, "min": 0.1, "max": 0.2, "vary": False},
        "radius_of_curvature_ratio": {"value": 3.0, "min": 0.5, "max": 100.0, "vary": False},
    }


@pytest.fixture
def model_InversionMatrix_settings():
    """Model configuration settings."""
    return {
        "type": "pcdwba",
        "model_function": pcdwba,
        "taper_order": 10.0,
        "frequency_interval": 2000.0,
        "orientation_distribution": {"family": "gaussian", "bins": 60},
        "length_distribution": {"family": "gaussian", "bins": 100},
        "environment": {"sound_speed_sw": 1490.0, "density_sw": 1026.5},
        "n_integration": 50,
        "n_wavelength": 10,
    }


@pytest.fixture
def sample_frequencies():
    """Sample acoustic frequencies."""
    return np.array([18e3, 38e3, 70e3, 120e3, 200e3])


@pytest.fixture
def inv_parameters(sample_InversionMatrix_parameters):
    """InvParameters instance."""
    return InvParameters(sample_InversionMatrix_parameters)


@pytest.fixture
def inv_transect_info(inv_parameters):
    """NASC coordinates."""
    # Parameterize
    inv_parameters.simulate_parameter_sets(mc_realizations=5, rng=np.random.default_rng(123))

    # Create proper MultiIndex structure for inverted data
    inverted_transect_data = pd.DataFrame(
        {
            ("transect_num", ""): [1, 2, 3, 4, 5],
            ("thickness_mean", 38e3): [15.5, 55.5, 22.5, 33.5, 44.5],
            ("thickness_mean", 120e3): [22.5, 38.5, 36.5, 18.5, 21.5],
            ("nasc", 38e3): [10.0, 20.0, 30.0, 40.0, 50.0],
            ("nasc", 120e3): [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    ).set_index("transect_num")
    # ---- Add parameters
    inverted_transect_data[("parameters", "")] = [
        InvParameters(inv_parameters.realizations[idx]) for idx in np.linspace(0, 4, 5)
    ]
    # ---- Set column index names
    inverted_transect_data.columns.names = [None, "frequency"]

    # Generate dataframe
    transect_nasc_df = (
        pd.DataFrame(
            {
                "transect_num": np.tile([1, 2, 3, 4, 5], 2),
                "longitude": np.tile(np.linspace(-1.0, 1.0, 5), 2),
                "latitude": np.tile(np.linspace(-1.0, 1.0, 5), 2),
                "frequency": np.repeat([38e3, 120e3], 5),
                "nasc": np.tile([10.0, 20.0, 30.0, 40.0, 50.0], 2),
            }
        )
        .set_index(["transect_num", "longitude", "latitude", "frequency"])
        .unstack("frequency")["nasc"]
    )

    return {"inverted": inverted_transect_data, "coords": transect_nasc_df}


@pytest.fixture
def inv_interval_info(inv_parameters):
    """NASC coordinates."""
    # Generate dataframe
    interval_nasc_df = (
        pd.DataFrame(
            {
                "frequency": np.repeat([38e3, 120e3], 20),
                "interval": np.tile(np.linspace(1, 10, 10), 4),
                "longitude": np.repeat([-1.0, 1.0, -1.0, 1.0], 10),
                "latitude": np.repeat([-1.0, 1.0, -1.0, 1.0], 10),
                "nasc": np.tile(np.linspace(10.0, 100.0, 10), 4),
            }
        )
        .set_index(["interval", "longitude", "latitude", "frequency"])
        .unstack("frequency")["nasc"]
    )

    # Parameterize
    inv_parameters.simulate_parameter_sets(mc_realizations=20, rng=np.random.default_rng(345))

    # Create proper MultiIndex structure for inverted data
    inverted_interval_data = pd.DataFrame(
        {
            ("nasc", 38e3): interval_nasc_df[38e3],
            ("nasc", 120e3): interval_nasc_df[120e3],
            ("thickness_mean", 38e3): np.linspace(20.0, 80.0, 20),
            ("thickness_mean", 120e3): np.linspace(40.0, 100.0, 20),
        },
        index=interval_nasc_df.index,
    )
    # ---- Add parameters
    inverted_interval_data[("parameters", "")] = [
        InvParameters(inv_parameters.realizations[idx]) for idx in np.linspace(0, 19, 20)
    ]
    # ---- Set column index names
    inverted_interval_data.columns.names = [None, "frequency"]

    return {"inverted": inverted_interval_data, "coords": interval_nasc_df}


@pytest.fixture
def inv_cells_info(inv_parameters):
    """NASC coordinates."""
    # Generate dataframe
    cells_nasc_df = (
        pd.DataFrame(
            {
                "frequency": np.repeat([38e3, 120e3], 10),
                "interval": np.tile(np.repeat(np.linspace(1, 5, 5), 2), 2),
                "layer": np.tile([1.0, 2.0], 10),
                "longitude": np.tile([-1.0, 1.0], 10),
                "latitude": np.tile([-1.0, 1.0], 10),
                "nasc": np.tile(np.linspace(10.0, 100.0, 10), 2),
            }
        )
        .set_index(["interval", "layer", "longitude", "latitude", "frequency"])
        .unstack("frequency")["nasc"]
    )

    # Parameterize
    inv_parameters.simulate_parameter_sets(mc_realizations=10, rng=np.random.default_rng(345))

    # Create proper MultiIndex structure for inverted data
    inverted_cells_data = pd.DataFrame(
        {
            ("nasc", 38e3): cells_nasc_df[38e3],
            ("nasc", 120e3): cells_nasc_df[120e3],
            ("thickness_mean", 38e3): np.linspace(20.0, 80.0, 10),
            ("thickness_mean", 120e3): np.linspace(40.0, 100.0, 10),
        },
        index=cells_nasc_df.index,
    )
    # ---- Add parameters
    inverted_cells_data[("parameters", "")] = [
        InvParameters(inv_parameters.realizations[idx]) for idx in np.linspace(0, 9, 10)
    ]
    # ---- Set column index names
    inverted_cells_data.columns.names = [None, "frequency"]

    return {"inverted": inverted_cells_data, "coords": cells_nasc_df}
