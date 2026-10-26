from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from lmfit import Parameters

from echopop import inversion, utils
from echopop.survey import apportionment, biology, proportions, stratified, transect
from echopop.geostatistics import cropping, kriging, variogram
from echopop.ingest import (
    join_geostrata_by_latitude,
    join_strata_by_haul,
    load_biological_data,
    load_geostrata,
    load_isobath_data,
    load_kriging_variogram_params,
    load_mesh_data,
    load_strata,
    nasc,
)

# ==================================================================================================
# ==================================================================================================
# DEFINE DATA ROOT DIRECTORY
# --------------------------
DATA_ROOT = Path("C:/Data/EchopopData/echopop_2019")

# ==================================================================================================
# ==================================================================================================
# DATA INGESTION
# ==================================================================================================
# Organize NASC file
# ------------------

# Merge exports
df_intervals, df_exports = nasc.merge_echoview_nasc(
    file_directory=DATA_ROOT / "raw_nasc/",
    filename_transect_pattern=r"T(\d+)",
    default_transect_spacing=10.0,
    default_latitude_threshold=60.0,
)

# ==================================================================================================
# Read in transect-region-haul keys
# ---------------------------------
TRANSECT_REGION_FILEPATH_ALL_AGES = (
    DATA_ROOT / "Stratification/US_CAN_2019_transect_region_haul_age1+ auto_final.xlsx"
)
TRANSECT_REGION_FILEPATH_NO_AGE1 = (
    DATA_ROOT / "Stratification/US_CAN_2019_transect_region_haul_age2+ auto_20191205.xlsx"
)
TRANSECT_REGION_FILE_RENAME = {
    "tranect": "transect_num",
    "region id": "region_id",
    "trawl #": "haul_num",
}
TRANSECT_REGION_SHEETNAME_ALL_AGES = "Sheet1"
TRANSECT_REGION_SHEETNAME_NO_AGE1 = "Sheet1"

# Read in the transect-region-haul key files for each group
transect_region_haul_key_all_ages = nasc.read_transect_region_haul_key(
    filename=TRANSECT_REGION_FILEPATH_ALL_AGES,
    sheetname=TRANSECT_REGION_SHEETNAME_ALL_AGES,
    column_name_map=TRANSECT_REGION_FILE_RENAME,
)

transect_region_haul_key_no_age1 = nasc.read_transect_region_haul_key(
    TRANSECT_REGION_FILEPATH_NO_AGE1, TRANSECT_REGION_SHEETNAME_NO_AGE1, TRANSECT_REGION_FILE_RENAME
)

# ==================================================================================================
# Read in transect-region-haul keys
# ---------------------------------
region_name_expr = {
    "REGION_CLASS": {
        "Age-1 Hake": "^(?:h1a(?![a-z]|m))",
        "Age-1 Hake Mix": "^(?:h1am(?![a-z]|1a))",
        "Hake": "^(?:h(?![a-z]|1a)|hake(?![_]))",
        "Hake Mix": "^(?:hm(?![a-z]|1a)|hake_mix(?![_]))",
    },
    "HAUL_NUM": {
        "[0-9]+",
    },
    "COUNTRY": {
        "CAN": "^[cC]",
        "US": "^[uU]",
    },
}

# Process the region name codes to define the region classes
# e.g. H5C - Region 2 corresponds to "Hake, Haul #5, Canada"
df_exports_with_regions = nasc.process_region_names(
    nasc_cells=df_exports,
    region_name_expr=REGION_NAME_EXPR_DICT,
    can_haul_offset=200,
)

# ==================================================================================================
# [OPTIONAL] Generate transect-region-haul key from compiled values
# ---------------------------------

# Generate transect-region-haul key from compiled values
df_transect_region_haul_key_no_age1 = nasc.generate_transect_region_haul_key(
    region_data=df_exports_with_regions, filter_list=["Hake", "Hake Mix"]
)

df_transect_region_haul_key_all_ages = nasc.generate_transect_region_haul_key(
    region_data=df_exports_with_regions, filter_list=["Age-1 Hake", "Age-1", "Hake", "Hake Mix"]
)

# ==================================================================================================
# Consolidate the Echvoiew NASC export files
# ------------------------------------------
df_nasc_no_age1 = nasc.consolidate_echvoiew_nasc(
    nasc_data=df_exports_with_regions,
    interval_data=df_intervals,
    region_class_names=["Hake", "Hake Mix"],
    impute_region_ids=True,
    transect_region_haul_key=transect_region_haul_key_no_age1,
)

df_nasc_all_ages = nasc.consolidate_echvoiew_nasc(
    nasc_data=df_exports_with_regions,
    interval_data=df_intervals,
    region_class_names=["Age-1 Hake", "Age-1", "Hake", "Hake Mix"],
    impute_region_ids=True,
    transect_region_haul_key=transect_region_haul_key_all_ages,
)

# ==================================================================================================
# [OPTIONAL] Read in a pre-consolidated NASC data file
# ----------------------------------------------------
FEAT_TO_ECHOPOP_COLUMNS = {
    "transect": "transect_num",
    "region id": "region_id",
    "vessel_log_start": "distance_s",
    "vessel_log_end": "distance_e",
    "spacing": "transect_spacing",
    "layer mean depth": "layer_mean_depth",
    "layer height": "layer_height",
    "bottom depth": "bottom_depth",
    "assigned haul": "haul_num",
}

#
df_nasc_all_ages = nasc.read_nasc_file(
    filename=DATA_ROOT / "Exports/US_CAN_NASC_2019_table_all_ages.xlsx",
    sheetname="Sheet1",
    column_name_map=FEAT_TO_ECHOPOP_COLUMNS,
)

# ==================================================================================================
# Load in the biolodical data
# ---------------------------
BIODATA_SHEETS = {
    "catch": "biodata_catch",
    "length": "biodata_length",
    "specimen": "biodata_specimen",
}
SUBSET_DICT = {
    "ships": {160: {"survey": 201906}, 584: {"survey": 2019097, "haul_offset": 200}},
    "species_code": [22500],
}
FEAT_TO_ECHOPOP_BIODATA_COLUMNS = {
    "frequency": "length_count",
    "haul": "haul_num",
    "weight_in_haul": "weight",
}
BIODATA_SEX = {"sex": {1: "male", 2: "female", 3: "unsexed"}}

#
dict_df_bio = load_biological_data(
    biodata_filepath=DATA_ROOT / "Biological/1995-2025_Survey_Biodata.xlsx",
    biodata_sheet_map=BIODATA_SHEETS,
    column_name_map=FEAT_TO_ECHOPOP_BIODATA_COLUMNS,
    survey_subset=SUBSET_DICT,
    biodata_label_map=BIODATA_SEX,
)

# ==================================================================================================
# Load in strata files
# --------------------
STRATA_SHEETS = {
    "inpfc": "INPFC",
    "ks": "Base KS",
}
FEAT_TO_ECHOPOP_STRATA_COLUMNS = {
    "fraction_hake": "nasc_proportion",
    "haul": "haul_num",
    "stratum": "stratum_num",
}

#
df_dict_strata = load_strata(
    strata_filepath=DATA_ROOT / "Stratification/US_CAN strata 2019_final.xlsx",
    strata_sheet_map=STRATA_SHEETS,
    column_name_map=FEAT_TO_ECHOPOP_STRATA_COLUMNS,
)

# ==================================================================================================
# Load in geographical strata files
# ---------------------------------
GEOSTRATA_SHEETS = {
    "inpfc": "INPFC",
    "ks": "stratification1",
}
FEAT_TO_ECHOPOP_GEOSTRATA_COLUMNS = {
    "latitude (upper limit)": "northlimit_latitude",
    "stratum": "stratum_num",
}

#
df_dict_geostrata = load_geostrata(
    geostrata_filepath=DATA_ROOT / "Stratification/Stratification_geographic_Lat_2019_final.xlsx",
    geostrata_sheet_map=GEOSTRATA_SHEETS,
    column_name_map=FEAT_TO_ECHOPOP_GEOSTRATA_COLUMNS,
)

# ==================================================================================================
# Stratify data based on haul numbers
# -----------------------------------

# Add INPFC
# ---- NASC
df_nasc_all_ages = join_strata_by_haul(
    data=df_nasc_all_ages, strata=df_dict_strata["inpfc"], stratum_name="stratum_inpfc"
)
# ---- Biodata
dict_df_bio = join_strata_by_haul(
    dict_df_bio, df_dict_strata["inpfc"], stratum_name="stratum_inpfc"
)

# Add KS
# ---- NASC
df_nasc_all_ages = join_strata_by_haul(
    df_nasc_all_ages, df_dict_strata["ks"], stratum_name="stratum_ks"
)
df_nasc_no_age1 = join_strata_by_haul(
    df_nasc_no_age1, df_dict_strata["ks"], stratum_name="stratum_ks"
)
# ---- Biodata
dict_df_bio = join_strata_by_haul(dict_df_bio, df_dict_strata["ks"], stratum_name="stratum_ks")

# ==================================================================================================
# Load kriging mesh file
# ----------------------

FEAT_TO_ECHOPOP_MESH_COLUMNS = {
    "centroid_latitude": "latitude",
    "centroid_longitude": "longitude",
    "fraction_cell_in_polygon": "fraction",
}

#
df_mesh = load_mesh_data(
    mesh_filepath=DATA_ROOT
    / "Kriging_files/Kriging_grid_files/krig_grid2_5nm_cut_centroids_2013.xlsx",
    sheet_name="krigedgrid2_5nm_forChu",
    column_name_map=FEAT_TO_ECHOPOP_MESH_COLUMNS,
)

# ==================================================================================================
# [OPTIONAL] Stratify data based on latitude intervals
# ----------------------------------------------------
# INPFC (from geostrata)
df_nasc_all_ages = join_geostrata_by_latitude(
    df_nasc_all_ages, df_dict_geostrata["inpfc"], stratum_name="geostratum_inpfc"
)
df_nasc_no_age1 = join_geostrata_by_latitude(
    df_nasc_no_age1, df_dict_geostrata["inpfc"], stratum_name="geostratum_inpfc"
)
# KS (from geostrata)
df_nasc_all_ages = join_geostrata_by_latitude(
    df_nasc_all_ages, df_dict_geostrata["ks"], stratum_name="geostratum_ks"
)
df_nasc_no_age1 = join_geostrata_by_latitude(
    df_nasc_no_age1, df_dict_geostrata["ks"], stratum_name="geostratum_ks"
)

# MESH
# ---- DataFrame merged with geographically distributed stratum number (KS or INPFC)
# -------- INPFC (from geostrata)
df_mesh = join_geostrata_by_latitude(
    df_mesh, df_dict_geostrata["inpfc"], stratum_name="geostratum_inpfc"
)
# -------- KS (from geostrata)
df_mesh = join_geostrata_by_latitude(df_mesh, df_dict_geostrata["ks"], stratum_name="geostratum_ks")

# ==================================================================================================
# Load kriging and variogram parameters
# -------------------------------------

FEAT_TO_ECHOPOP_GEOSTATS_PARAMS_COLUMNS = {
    "hole": "hole_effect_range",
    "lscl": "correlation_range",
    "nugt": "nugget",
    "powr": "decay_power",
    "ratio": "aspect_ratio",
    "res": "lag_resolution",
    "srad": "search_radius",
}

#
dict_kriging_params, dict_variogram_params = load_kriging_variogram_params(
    geostatistic_params_filepath=(
        DATA_ROOT / "Kriging_files/default_vario_krig_settings_2019_US_CAN.xlsx"
    ),
    sheet_name="Sheet1",
    column_name_map=FEAT_TO_ECHOPOP_GEOSTATS_PARAMS_COLUMNS,
)

# ==================================================================================================
# ==================================================================================================
# DATA PROCESSING
# ==================================================================================================
# Generate binned distributions [age, length]
# -------------------------------------------
AGE_BIN_EDGES = np.arange(0.5, 23.0, 1.0)
LENGTH_BIN_EDGES = np.arange(1.0, 82.0, 2.0)

#
# ---- Length
for dataset in ("specimen", "length"):
    dict_df_bio[dataset]["length_bin"] = pd.cut(dict_df_bio[dataset]["length"], bins=LENGTH_BIN_EDGES)

# Age
dict_df_bio["specimen"]["age_bin"] = pd.cut(dict_df_bio["specimen"]["age"], bins=AGE_BIN_EDGES)

# ==================================================================================================
# Fit length-weight regression to the binned data
# -----------------------------------------------

# Dictionary for length-weight regression coefficients
dict_length_weight_coefs = {}

# For all fish
dict_length_weight_coefs["all"] = (
    dict_df_bio["specimen"]
    .assign(sex="all")
    .groupby(["sex"])
    .apply(biology.fit_length_weight_regression, include_groups=False)
)

# Sex-specific
dict_length_weight_coefs["sex"] = (
    dict_df_bio["specimen"]
    .groupby(["sex"])
    .apply(biology.fit_length_weight_regression, include_groups=False)
)

# ==================================================================================================
# Compute the mean weights per length bin
# ---------------------------------------

# Sex-specific (grouped coefficients)
da_binned_weights_sex = biology.length_binned_weights(
    data=dict_df_bio["specimen"],
    length_bin_edges=LENGTH_BIN_EDGES,
    regression_coefficients=dict_length_weight_coefs["sex"],
    impute_bins=True,
    minimum_count_threshold=5,
)

# All fish (single coefficient set)
da_binned_weights_all = biology.length_binned_weights(
    data=dict_df_bio["specimen"].assign(sex="all"),
    length_bin_edges=LENGTH_BIN_EDGES,
    regression_coefficients=dict_length_weight_coefs["all"],
    impute_bins=True,
    minimum_count_threshold=5,
)

# Combine the pivot tables by adding the "all" column to the sex-specific table
da_binned_weight_table = xr.concat(
    [da_binned_weights_sex, da_binned_weights_all],
    dim = "sex"
)
# ==================================================================================================
# Compute the count distributions per age- and length-bins
# --------------------------------------------------------

# DATASET CONTAINER
ds_counts = xr.Dataset()

# AGED
ds_counts["aged"] = proportions.compute_binned_counts(
    data=dict_df_bio["specimen"].dropna(subset=["age", "length", "weight"]),
    groupby_cols=["stratum_ks", "length_bin", "age_bin", "sex"],
    count_col="length",
    agg_func="size",
)

# UNAGED
ds_counts["unaged"] = proportions.compute_binned_counts(
    data=dict_df_bio["length"].copy().dropna(subset=["length"]),
    groupby_cols=["stratum_ks", "length_bin", "sex"],
    count_col="length_count",
    agg_func="sum",
)

# ==================================================================================================
# Compute the number proportions
# ------------------------------
dict_ds_number_proportion = proportions.number_proportions(
    data=ds_counts,
    stratum_dim="stratum_ks",
    exclude_filters={"aged": {"sex": "unsexed"}},
)

# ==================================================================================================
# Distribute (bin) weight over age, length, and sex
# -------------------------------------------------
# Pre-allocate a dataset
ds_da_weight_dist = xr.Dataset()

# Aged
ds_da_weight_dist["aged"] = proportions.binned_weights(
    length_data=dict_df_bio["specimen"],
    include_filter={"sex": ["female", "male"]},
    interpolate_regression=False,
    group_columns=["stratum_ks", "sex", "age_bin"],
)

# Unaged
ds_da_weight_dist["unaged"] = proportions.binned_weights(
    length_data=dict_df_bio["length"],
    include_filter={"sex": ["female", "male"]},
    interpolate_regression=True,
    length_weight_data=da_binned_weight_table,
    group_columns=["stratum_ks", "sex"],
)

# ==================================================================================================
# Calculate the average weights pre stratum when combining different datasets
# ---------------------------------------------------------------------------
da_averaged_weight = proportions.stratum_averaged_weight(
    number_proportions=dict_ds_number_proportion,
    length_weight_data=da_binned_weight_table,
    stratum_dim="stratum_ks"
)

# ==================================================================================================
# Compute the length-binned weight proportions for aged fish
# ----------------------------------------------------------

# Initialize Dictionary container
dict_da_weight_proportion = {}

# Aged
dict_da_weight_proportion["aged"] = proportions.weight_proportions(
    weight_data=ds_da_weight_dist["aged"], 
    catch_data=dict_df_bio["catch"], 
    stratum_dim="stratum_ks"
)

# ==================================================================================================
# Compute the standardized weight proportionsfor unaged fish
# ----------------------------------------------------------
dict_da_weight_proportion["unaged"] = proportions.fitted_weight_proportions(
    weight_data=ds_da_weight_dist["unaged"],
    aged_weight_proportions=dict_da_weight_proportion["aged"],
    number_proportions=dict_ds_number_proportion["unaged"],
    binned_weights=da_binned_weights_all,
    stratum_dim="stratum_ks"
)

# ==================================================================================================
# ==================================================================================================
# NASC TO POPULATION ESTIMATE CONVERSION
# ==================================================================================================
# Initialize the Inversion class
# ------------------------------
MODEL_PARAMETERS = {
    "ts_length_regression": {"slope": 20.0, "intercept": -68.0},
    "stratify_by": ["stratum_ks"],
    "expected_strata": df_dict_strata["ks"].stratum_num.unique(),
    "impute_missing_strata": True,
    "haul_replicates": True,
}

# Initiate object to perform inversion
invert_hake = inversion.InversionLengthTS(MODEL_PARAMETERS)

# ==================================================================================================
# Invert number density
# ---------------------

# If the above haul-averaged `sigma_bs` values were calculated, then the inversion can can
# completed without calling in additional biodata
df_nasc_all_ages = invert_hake.invert(
    nasc_data=df_nasc_all_ages, length_data=[dict_df_bio["length"], dict_df_bio["specimen"]]
)
df_nasc_no_age1 = invert_hake.invert(
    nasc_data=df_nasc_no_age1, length_data=[dict_df_bio["length"], dict_df_bio["specimen"]]
)
# ---- The average `sigma_bs` for each stratum can be inspected at:
invert_hake.sigma_bs_strata

# ==================================================================================================
# Set transect interval distances
# -------------------------------

# Calculate along-transect interval distances which is required for getting the area-per-interval
# and therefore going from number density to abundance
transect.compute_interval_distance(nasc_data=df_nasc_all_ages, interval_threshold=0.05)
transect.compute_interval_distance(nasc_data=df_nasc_no_age1, interval_threshold=0.05)

# ==================================================================================================
# Calculate transect interval areas
# ---------------------------------
df_nasc_all_ages["area_interval"] = (
    df_nasc_all_ages["transect_spacing"] * df_nasc_all_ages["distance_interval"]
)
df_nasc_no_age1["area_interval"] = (
    df_nasc_no_age1["transect_spacing"] * df_nasc_no_age1["distance_interval"]
)

# ==================================================================================================
# Calculate (and apportion) number densities to abundance, and number densities/abundance for each
# sex
# --------------------------------------------------------------------------------------------------

biology.compute_abundance(
    transect_data=df_nasc_no_age1,
    exclude_filter={"sex": "unsexed"},
    number_proportions=dict_ds_number_proportion,
)

# ==================================================================================================
# Calculate (and apportion) biomass densities and biomass (from number density and abundance,
# respectively) for the overall transect dataset as well as for each sex
# --------------------------------------------------------------------------------------------------

biology.compute_biomass(
    transect_data=df_nasc_no_age1,
    stratum_weights=da_averaged_weight,
)

# ==================================================================================================
# Get proportions for each stratum specific to age-1
# --------------------------------------------------

# Age-1 NASC proportions
da_age1_nasc_proportions = proportions.get_nasc_proportions_slice(
    number_proportions=dict_ds_number_proportion["aged"],
    stratum_dim="stratum_ks",
    ts_length_regression_parameters={"slope": 20.0, "intercept": -68.0},
    include_filter={"age_bin": [1]},
)

# Age-1 number proportions
da_age1_number_proportions = proportions.get_number_proportions_slice(
    number_proportions=dict_ds_number_proportion["aged"],
    stratum_dim="stratum_ks",
    include_filter={"age_bin": [1]},
)

# Age-1 weight proportions
da_age1_weight_proportions = proportions.get_weight_proportions_slice(
    weight_proportions=dict_da_weight_proportion["aged"],
    stratum_dim="stratum_ks",
    include_filter={"age_bin": [1]},
    number_proportions=dict_ds_number_proportion,
    length_threshold_min=10.0,
    weight_proportion_threshold=1e-10,
)

# ==================================================================================================
# Apply the calculated proportions to the abundance, biomass, and NASC estimates
# ------------------------------------------------------------------------------

df_nasc_no_age1_prt = apportionment.remove_group_from_estimates(
    transect_data=df_nasc_no_age1,
    group_proportions=xr.Dataset({
        "nasc": da_age1_nasc_proportions,
        "abundance": da_age1_number_proportions,
        "biomass": da_age1_weight_proportions,
    }),
)

# ==================================================================================================
# Distribute transect abundances across age-length-sex bins
# ---------------------------------------------------------

dict_ds_transect_abundance_table = apportionment.distribute_population_estimates(
    data=df_nasc_no_age1_prt,
    proportions = dict_ds_number_proportion,
    variable = "abundance",
    group_columns= ["sex", "age_bin", "length_bin", "stratum_ks"]
)

# ==================================================================================================
# Distribute transect biomasses across age-length-sex bins
# ---------------------------------------------------------

dict_ds_transect_biomass_table = apportionment.distribute_population_estimates(
    data=df_nasc_no_age1_prt,
    proportions = dict_ds_number_proportion,
    variable="biomass",
    group_columns= ["sex", "age_bin", "length_bin", "stratum_ks"]
)

# ==================================================================================================
# Distribute transect biomasses across age-length-sex bins for aged fish only
# ---------------------------------------------------------------------------

ds_transect_aged_biomass_table = apportionment.distribute_population_estimates(
    data=df_nasc_no_age1_prt,
    proportions=dict_da_weight_proportion["aged"],
    variable="biomass",
    group_columns = ["sex", "age_bin", "length_bin", "stratum_ks"]
)

# ==================================================================================================
# ==================================================================================================
# GEOSTATISTICS
# ==================================================================================================
# Load reference line (isobath)
# -----------------------------

df_isobath = load_isobath_data(
    isobath_filepath=DATA_ROOT
    / "Kriging_files/Kriging_grid_files/transformation_isobath_coordinates.xlsx",
    sheet_name="Smoothing_EasyKrig",
)

# ==================================================================================================
# Transform the geospatial coordinates for the transect data
# ----------------------------------------------------------
df_nasc_no_age1_prt, delta_longitude, delta_latitude = cropping.transform_coordinates(
    data=df_nasc_no_age1,
    reference=df_isobath,
    x_offset=-124.78338,
    y_offset=45.0,
)

# ==================================================================================================
# Transform the geospatial coordinates for the mesh data
# ------------------------------------------------------
df_mesh, _, _ = cropping.transform_coordinates(
    data=df_mesh,
    reference=df_isobath,
    x_offset=-124.78338,
    y_offset=45.0,
    delta_x=delta_longitude,
    delta_y=delta_latitude,
)

# ==================================================================================================
# Initialize Variogram class
# --------------------------

# Initialize
vgm = variogram.Variogram(
    lag_resolution=0.002,
    n_lags=30,
    coordinate_names=("x", "y"),
)

# ==================================================================================================
# Calculate the empirical variogram
# ---------------------------------
vgm.calculate_empirical_variogram(
    data=df_nasc_no_age1_prt,
    variable="biomass_density",
    azimuth_filter=True,
    azimuth_angle_threshold=180.0,
)

# ==================================================================================================
# Fit theoretical/modeled variogram to the transect data
# ------------------------------------------------------

# Set up `lmfit` parameters
# lmfit.Parameters tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
variogram_parameters_lmfit = Parameters()
variogram_parameters_lmfit.add_many(
    ("nugget", dict_variogram_params["nugget"], True, 0.0),
    ("sill", dict_variogram_params["sill"], True, 0.0),
    ("correlation_range", dict_variogram_params["correlation_range"], True, 0.0),
    ("hole_effect_range", dict_variogram_params["hole_effect_range"], True, 0.0),
    ("decay_power", dict_variogram_params["decay_power"], True, 1.25, 1.75),
)

# Set up optimization parameters used for fitting the variogram
dict_optimization = {
    "max_nfev": None,
    "ftol": 1e-08,
    "gtol": 1e-8,
    "xtol": 1e-8,
    "diff_step": None,
    "tr_solver": "exact",
    "x_scale": 1.0,
    "jac": "2-point",
}

# Get the best-fit variogram parameters
best_fit_parameters = vgm.fit_variogram_model(
    model=["exponential", "bessel"],
    model_parameters=variogram_parameters_lmfit,
    optimizer_kwargs=dict_optimization,
)
print(best_fit_parameters)

# ==================================================================================================
# Initialize the kriging class object
# -----------------------------------

# Define the requisite kriging parameters
KRIGING_PARAMETERS = {
    "search_radius": best_fit_parameters["correlation_range"] * 3,
    "aspect_ratio": 0.001,
    "k_min": 3,
    "k_max": 10,
}

# Define the requisite variogram parameters and arguments
VARIOGRAM_PARAMETERS = {"model": ["exponential", "bessel"], **best_fit_parameters}

krg = kriging.Kriging(
    mesh=df_mesh,
    kriging_params=KRIGING_PARAMETERS,
    variogram_params=VARIOGRAM_PARAMETERS,
    coordinate_names=("x", "y"),
)

# ==================================================================================================
# Mesh cropping using the hull convex
# -----------------------------------

krg.crop_mesh(
    crop_function=utils.feat_functions.transect_ends_crop,
    transects=df_nasc_no_age1_prt,
    latitude_resolution=1.25 / 60.0,
    transect_mesh_region_function=utils.feat_parameters.transect_mesh_region_2019,
)

# ==================================================================================================
# [FEAT] Get the western extent of the transect bounds
# ----------------------------------------------------
transect_western_extents = utils.feat_functions.get_survey_western_extents(
    transects=df_nasc_no_age1_prt, coordinate_names=("x", "y"), latitude_threshold=51.0
)

# ==================================================================================================
# [FEAT] Register the custom search strategy
# ------------------------------------------
krg.register_search_strategy("FEAT_strategy", utils.feat_functions.western_boundary_search_strategy)
# ---- Verify that method was registered
krg.list_search_strategies()

# ==================================================================================================
# Krige the biomass density to get kriged biomass
# -----------------------------------------------

# Define the required keyword arguments for 'FEAT_strategy'
# ---- Only `transect_western_extents` is needed for this particular function since the
# `kriging_mesh` and `coordinate_names` arguments are inherited from the class instance
FEAT_STRATEGY_KWARGS = {
    "western_extent": transect_western_extents,
}

# Krige
df_kriged_results = krg.krige(
    transects=df_nasc_no_age1_prt,
    variable="biomass_density",
    extrapolate=False,
    default_mesh_cell_area=6.25,
    adaptive_search_strategy="FEAT_strategy",
    custom_search_kwargs=FEAT_STRATEGY_KWARGS,
)

# Print the global survey CV
print(f"Global kriging survey CV: {krg.survey_cv:.3f}.")

# ##################################################################################################
# Back-calculate sex-specific biomass and abundance, and total NASC from the kriged biomass
# density estimates
# -----------------

# Compute biomass
df_kriged_results["biomass"] = df_kriged_results["biomass_density"] * df_kriged_results["area"]

# Convert biomass to abundance to NASC
apportionment.mesh_biomass_to_nasc(
    mesh_data=df_kriged_results,
    biodata=dict_da_weight_proportion,
    group_columns=["sex", "stratum_ks"],
    mesh_biodata_link={"geostratum_ks": "stratum_ks"},
    stratum_weights=da_averaged_weight.sel(sex="all"),
    stratum_sigma_bs=invert_hake.sigma_bs_strata,
)

# ##################################################################################################
# Distribute kriged abundance estimates over length and age/length
# ----------------------------------------------------------------

dict_ds_kriged_abundance_table = apportionment.distribute_population_estimates(
    data=df_kriged_results,
    proportions = dict_ds_number_proportion,
    variable = "abundance",
    group_columns = ["sex", "age_bin", "length_bin", "stratum_ks"],
    data_proportions_link={"geostratum_ks": "stratum_ks"}
)

# ##################################################################################################
# Distribute kriged biomass estimates over length and age/length
# --------------------------------------------------------------

dict_ds_kriged_biomass_table = apportionment.distribute_population_estimates(
    data=df_kriged_results,
    proportions = dict_da_weight_proportion,
    variable = "biomass",
    group_columns = ["sex", "age_bin", "length_bin", "stratum_ks"],
    data_proportions_link={"geostratum_ks": "stratum_ks"}
)

# ##################################################################################################
# Standardize the unaged abundance estimates to be distributed over age
# ---------------------------------------------------------------------

dict_ds_kriged_abundance_table["standardized_unaged"] = apportionment.distribute_unaged_from_aged(
    population_table = dict_ds_kriged_abundance_table["unaged"],
    reference_table = dict_ds_kriged_abundance_table["aged"],
    stratum_dim="stratum_ks",
    impute = False
)

# ##################################################################################################
# Standardize the unaged abundance estimates to be distributed over age
# ---------------------------------------------------------------------

dict_ds_kriged_biomass_table["standardized_unaged"] = apportionment.distribute_unaged_from_aged(
    population_table = dict_ds_kriged_biomass_table["unaged"],
    reference_table = dict_ds_kriged_biomass_table["aged"],
    stratum_dim="stratum_ks",
    impute=True,
    impute_variable=["age_bin"],
)


# ##################################################################################################
# Consolidate the kriged abundance estimates into a single DataFrame table
# ------------------------------------------------------------------------

da_kriged_abundance_table = apportionment.sum_population_tables(
    population_tables={
        "aged": dict_ds_kriged_abundance_table["aged"],
        "unaged": dict_ds_kriged_abundance_table["standardized_unaged"]
    },
)

# ##################################################################################################
# Consolidate the kriged biomass estimates into a single DataFrame table
# -----------------------------------------------------------------------

da_kriged_biomass_table = apportionment.sum_population_tables(
    population_tables={
        "aged": dict_ds_kriged_biomass_table["aged"],
        "unaged": dict_ds_kriged_biomass_table["standardized_unaged"]
    },
)

# ##################################################################################################
# Redistribute the kriged abundance estimates
# -------------------------------------------

# Re-allocate the age-1 abundance estimates
da_kriged_abundance_table_noage1 = apportionment.reallocate_excluded_estimates(
    population_table=da_kriged_abundance_table,
    exclusion_filter={"age_bin": [1]},
    group_columns=["sex"],
)

####################################################################################################
# Redistribute the kriged biomass estimates
# -----------------------------------------

# Re-allocate the age-1 abundance estimates
da_kriged_biomass_table_noage1 = apportionment.reallocate_excluded_estimates(
    population_table=da_kriged_biomass_table,
    exclusion_filter={"age_bin": [1]},
    group_columns=["sex"],
)

# ##################################################################################################
# Instantiate stratified analysis
# -------------------------------
# Model parameters for stratified analysis initialization
JOLLYHAMPTON_PARAMETERS = {
    "transects_per_latitude": 5,
    "strata_transect_proportion": 0.75,
    "num_replicates": 1000,
}

jh = stratified.JollyHampton(JOLLYHAMPTON_PARAMETERS)

# ##################################################################################################
# Compute coefficient of variation (CV) and other estimators for transect biomass
# -------------------------------------------------------------------------------

# Run bootstrapping procedure
jh.stratified_bootstrap(
    data=df_nasc_no_age1_prt, stratum_dim="geostratum_inpfc", variable="biomass"
)

# Store replicates
transect_replicates = jh.bootstrap_replicates

# Compute summary statistics for each stratum and overall survey
transect_results = jh.summarize(ci_percentile=0.95, ci_method="t-jackknife")
print(transect_results)

# ##################################################################################################
# Create virtual transects for the gridded kriged biomass results
# ---------------------------------------------------------------

kriged_transects = jh.create_virtual_transects(
    mesh_data=df_kriged_results,
    geostrata=df_dict_geostrata["inpfc"],
    stratum_dim="geostratum_inpfc",
    variable="biomass",
)

# ##################################################################################################
# Compute CV and other estimators for kriged biomass
# --------------------------------------------------

# Run bootstrapping procedure
jh.stratified_bootstrap(
    data=kriged_transects, stratum_dim="geostratum_inpfc", variable="biomass"
)

# Store replicates
kriged_replicates = jh.bootstrap_replicates

# Compute summary statistics for each stratum and overall survey
kriged_results = jh.summarize(ci_percentile=0.95, ci_method="t-jackknife")
print(kriged_results)

# ##################################################################################################
# Compare transect and kriged outputs
# -----------------------------------

print(
    kriged_results.xs("mean", axis=1, level="metric")
    - transect_results.xs("mean", axis=1, level="metric")
)
