####################################################################################################
# 2019 
# ----
from pathlib import Path
####################################################################################################
# PARAMETER ENTRY
# ---------------
# ** ENTER FILE INFORMATION FOR ALL INGESTED DATASETS.
# ** ADDITIONAL PARAMETERIZATIONS THROUGHOUT THE SCRIPT SHOULD BE EDITED BASED ON SPECIFIC NEEDS. 
# ** MAKE SURE TO EDIT WITH CARE.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DATA ROOT DIRECTORY
DATA_ROOT = Path("C:/Data/EchopopData/echopop_2019")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NASC EXPORTS FILE(S)
NASC_EXPORTS_FILES = DATA_ROOT / "Exports/US&CAN_detailsa_2019_table2y+_ALL_final - updated.xlsx"
# NASC EXPORTS SHEET
NASC_EXPORTS_SHEET = "Sheet1"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BIODATA FILE
BIODATA_FILE = DATA_ROOT / "Biological/1995-2025_Survey_Biodata_missing_age.xlsx"
# BIODATA SHEETS
# ---- Assign the sheetnames to 'catch', 'length', 'specimen'
BIODATA_SHEETS = {
    "catch": "biodata_catch",
    "length": "biodata_length",
    "specimen": "biodata_specimen",
}
# BIODATA PROCESSING
# ---- This is used to parse the biodata master spreadsheet, which is required for aligning the 
# ---- biodata with ancillary files such as stratification and transect-haul mappings. This should 
# ---- define the "ships" based on their IDs with the associated survey IDs. If an offset should be 
# ---- added to the haul numbers, that must also be defined here. The target species should also be 
# ---- defined here. 
CAN_HAUL_OFFSET = 200
SPECIES_ID = 22500 # numeric species code for Pacific hake
SHIP_US = 160 # US ship ID
SHIP_CAN = 584 # CAN ship ID
SURVEY_US = 201906 # US survey identifier
SURVEY_CAN = 2019097 # CAN survey identifier

BIODATA_SHIP_SPECIES = {
    "ships": {
        SHIP_US: {
            "country": "US",
            "survey": SURVEY_US 
        },
        SHIP_CAN: {
            "country": "CAN",
            "survey": SURVEY_CAN,
            "haul_offset": CAN_HAUL_OFFSET
        }
    },
    "species_code": [SPECIES_ID]
}
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HAUL STRATIFICATION FILE
HAUL_STRATA_FILE = DATA_ROOT / "Stratification/US_CAN strata 2019_final.xlsx"
# HAUL STRATIFICATION SHEET MAP
# ---- Valid keys are limited to "ks" and "inpfc"
HAUL_STRATA_SHEETS = {
    "inpfc": "INPFC",
    "ks": "Base KS",
}
# GEOGRAPHIC STRATIFICATION FILE
GEOSTRATA_FILE = DATA_ROOT / "Stratification/Stratification_geographic_Lat_2019_final.xlsx"
# GEOGRAPHIC STRATIFICATION SHEET MAP
# ---- Valid keys are limited to "ks" and "inpfc"
GEOSTRATA_SHEETS = {
    "inpfc": "INPFC",
    "ks": "stratification1",
}
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# KRIGING MESH FILE 
KRIGING_MESH_FILE = (
    DATA_ROOT / "Kriging_files/Kriging_grid_files/krig_grid2_5nm_cut_centroids_2013.xlsx"
)
# KRIGING MESH SHEET
KRIGING_MESH_SHEET = "krigedgrid2_5nm_forChu"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# KRIGING AND VARIOGRAM PARAMETERS FILE
KRIGING_VARIOGRAM_PARAMETERS_FILE = (
    DATA_ROOT / "Kriging_files/default_vario_krig_settings_2019_US_CAN.xlsx"
)
# KRIGING AND VARIOGRAM PARAMETERS SHEET
KRIGING_VARIGORAM_PARAMETERS_SHEET = "Sheet1"
# USE DEFAULT VALUES OR OPTIMIZE?
OPTIMIZE_VARIOGRAM = False
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 200m ISOBATH FILE
ISOBATH_FILE = (
    DATA_ROOT / "Kriging_files/Kriging_grid_files/transformation_isobath_coordinates.xlsx"
)
# 200m ISOBATH SHEET
ISOBATH_SHEET = "Smoothing_EasyKrig"
####################################################################################################
####################################################################################################
# !!! START OF PROCESSING SCRIPT !!
# !!! EDIT CODE BELOW WITH CARE !!
####################################################################################################
####################################################################################################
import logging
import numpy as np
import pandas as pd
import xarray as xr
import echopop.ingest as ingestion
from echopop import inversion, utils
from echopop.survey import biology, proportions, transect
# ==================================================================================================
# DATA INGESTION 
# ==================================================================================================
# FORMAT HAUL-BASED UID 
HAUL_UID_CONFIG = {
    "ship_id": {"US": SHIP_US, "CAN": SHIP_CAN},
    "survey_id": {"US": SURVEY_US, "CAN": SURVEY_CAN},
    "species_id": SPECIES_ID,
    "haul_offset": CAN_HAUL_OFFSET
}

# DEFINE COLUMN MAPPING
FEAT_TO_ECHOPOP_COLUMNS = {
    "transect": "transect_num",
    "region id": "region_id",
    "vl start": "distance_s",
    "vl end": "distance_e",
    "spacing": "transect_spacing",
    "layer mean depth": "layer_mean_depth",
    "layer height": "layer_height",
    "bottom depth": "bottom_depth",
    "assigned haul": "haul_num",
}

# Read file
df_nasc = ingestion.nasc.read_nasc_file(
    filename=NASC_EXPORTS_FILES,
    sheetname=NASC_EXPORTS_SHEET,
    column_name_map=FEAT_TO_ECHOPOP_COLUMNS,
    haul_uid_config=HAUL_UID_CONFIG,
)

# DROP TRANSECTS
df_nasc = utils.apply_filters(df_nasc, include_filter={"transect_num": np.arange(1, 200)})
# ==================================================================================================
# INGEST BIODATA
# BIODATA DATAFRAME COLUMN NAME MAPPING
FEAT_TO_ECHOPOP_BIODATA_COLUMNS = {
    "frequency": "length_count",
    "haul": "haul_num",
    "haul ": "haul_num",
    "weight_in_haul": "weight",
}

# BIODATA LABEL MAPPING
BIODATA_SEX = {
    "sex": {
        1: "male",
        2: "female",
        3: "unsexed"
    }
}

# READ IN DATA
dict_df_bio = ingestion.load_biological_data(
    biodata_filepath=BIODATA_FILE, 
    biodata_sheet_map=BIODATA_SHEETS, 
    column_name_map=FEAT_TO_ECHOPOP_BIODATA_COLUMNS, 
    survey_subset=BIODATA_SHIP_SPECIES, 
    biodata_label_map=BIODATA_SEX,
    haul_uid_config=HAUL_UID_CONFIG,
)

# ==================================================================================================
# INGEST STRATIFICATION DATA

# HAUL-BASED STRATIFICATION DATAFRAME COLUMN NAME MAPPING
FEAT_TO_ECHOPOP_STRATA_COLUMNS = {
    "fraction_hake": "nasc_proportion",
    "haul": "haul_num",
    "stratum": "stratum_num",
}

# READ IN STRATA FILE 
df_dict_strata = ingestion.load_strata(
    strata_filepath=HAUL_STRATA_FILE, 
    strata_sheet_map=HAUL_STRATA_SHEETS, 
    column_name_map=FEAT_TO_ECHOPOP_STRATA_COLUMNS,
    haul_uid_config=HAUL_UID_CONFIG,
)

# GEOGRAPHIC-BASED STRATIFICATION DATAFRAME COLUMN NAME MAPPING
FEAT_TO_ECHOPOP_GEOSTRATA_COLUMNS = {
    "latitude (upper limit)": "northlimit_latitude",
    "stratum": "stratum_num",
}

# READ IN GEOSTRATA FILE
df_dict_geostrata = ingestion.load_geostrata(
    geostrata_filepath=GEOSTRATA_FILE, 
    geostrata_sheet_map=GEOSTRATA_SHEETS, 
    column_name_map=FEAT_TO_ECHOPOP_GEOSTRATA_COLUMNS
)

# ==================================================================================================
# LOAD KRIGING MESH FILE

# KRIGING MESH DATAFRAME COLUMN NAME MAPPING
FEAT_TO_ECHOPOP_MESH_COLUMNS = {
    "centroid_latitude": "latitude",
    "centroid_longitude": "longitude",
    "fraction_cell_in_polygon": "fraction",
}

# LOAD MESH
df_mesh = ingestion.load_mesh_data(
    mesh_filepath=KRIGING_MESH_FILE, 
    sheet_name=KRIGING_MESH_SHEET, 
    column_name_map=FEAT_TO_ECHOPOP_MESH_COLUMNS
)

# ==================================================================================================
# LOAD ISOBATH FILE
df_isobath = ingestion.load_isobath_data(
    isobath_filepath=ISOBATH_FILE,
    sheet_name=ISOBATH_SHEET
)

# ==================================================================================================
# LOAD KRIGING AND VARIOGRAM PARAMETERS

# PARAMETERS DATAFRAME COLUMN NAME MAPPING
FEAT_TO_ECHOPOP_GEOSTATS_PARAMS_COLUMNS = {
    "hole": "hole_effect_range",
    "lscl": "correlation_range",
    "nugt": "nugget",
    "powr": "decay_power",
    "ratio": "aspect_ratio",
    "res": "lag_resolution",
    "srad": "search_radius",
}

# LOAD IN PARAMETERS
dict_kriging_params, dict_variogram_params = ingestion.load_kriging_variogram_params(
    geostatistic_params_filepath=KRIGING_VARIOGRAM_PARAMETERS_FILE,
    sheet_name=KRIGING_VARIGORAM_PARAMETERS_SHEET,
    column_name_map=FEAT_TO_ECHOPOP_GEOSTATS_PARAMS_COLUMNS
)

# ==================================================================================================
# INITIAL DATA PROCESSING
# ==================================================================================================
# APPLY STRATIFICATION DEFINITIONS TO BIODATA AND NASC

# HAUL-BASED STRATA
# ---- BIODATA [INPFC]
dict_df_bio = ingestion.join_strata_by_uid(
    data=dict_df_bio,
    strata=df_dict_strata["inpfc"],
    default_stratum=0,
    stratum_name="stratum_inpfc",
)
# ---- BIODATA [KS]
dict_df_bio = ingestion.join_strata_by_uid(
    data=dict_df_bio, strata=df_dict_strata["ks"], default_stratum=0, stratum_name="stratum_ks"
)
# ---- NASC [INPFC]
df_nasc = ingestion.join_strata_by_uid(
    data=df_nasc, strata=df_dict_strata["inpfc"], default_stratum=0, stratum_name="stratum_inpfc"
)
# ---- NASC [KS]
df_nasc = ingestion.join_strata_by_uid(
    data=df_nasc, strata=df_dict_strata["ks"], default_stratum=0, stratum_name="stratum_ks"
)

# GEOGRAPHIC-BASED STRATA
# ---- NASC [INPFC]
df_nasc = ingestion.join_geostrata_by_latitude(
    data=df_nasc,
    geostrata=df_dict_geostrata["inpfc"],
    stratum_name="geostratum_inpfc"
)
# ---- NASC [KS]
df_nasc = ingestion.join_geostrata_by_latitude(
    data=df_nasc,
    geostrata=df_dict_geostrata["ks"],
    stratum_name="geostratum_ks"
)
# ---- MESH [INPFC]
df_mesh = ingestion.join_geostrata_by_latitude(
    data=df_mesh, 
    geostrata=df_dict_geostrata["inpfc"], 
    stratum_name="geostratum_inpfc"
)
# ---- MESH [KS]
df_mesh = ingestion.join_geostrata_by_latitude(
    data=df_mesh, 
    geostrata=df_dict_geostrata["ks"], 
    stratum_name="geostratum_ks"
)
# ==================================================================================================
# BIN DATA USING EXPLICIT EDGES

# AGE-BINS
AGE_BIN_EDGES = np.arange(0.5, 23.0, 1.0)
dict_df_bio["specimen"]["age_bin"] = pd.cut(dict_df_bio["specimen"]["age"], bins=AGE_BIN_EDGES)

# LENGTH-BINS
LENGTH_BIN_EDGES = np.arange(1.0, 82.0, 2.0)
for dataset in ("specimen", "length"):
    dict_df_bio[dataset]["length_bin"] = pd.cut(dict_df_bio[dataset]["length"], bins=LENGTH_BIN_EDGES)

# ==================================================================================================
# FIT LENGTH-WEIGHT REGRESSION

# CREATE DICTIONARY CONTAINER
dict_length_weight_coefs = {}

# ALL FISH
dict_length_weight_coefs["all"] = dict_df_bio["specimen"].assign(sex="all").groupby(["sex"]).apply(
    biology.fit_length_weight_regression,
    include_groups=False
)

# SEX-SPECIFIC
dict_length_weight_coefs["sex"] = dict_df_bio["specimen"].groupby(["sex"]).apply(
    biology.fit_length_weight_regression,
    include_groups=False
)

# ==================================================================================================
# COMPUTE MEAN WEIGHTS PER LENGTH BIN

# SEX-SPECIFIC
da_binned_weights_sex = biology.length_binned_weights(
    data=dict_df_bio["specimen"],
    length_bin_edges=LENGTH_BIN_EDGES,
    regression_coefficients=dict_length_weight_coefs["sex"],
    impute_bins=True,
    minimum_count_threshold=5
)

# ALL FISH
da_binned_weights_all = biology.length_binned_weights(
    data=dict_df_bio["specimen"].assign(sex="all"),
    length_bin_edges=LENGTH_BIN_EDGES,
    regression_coefficients=dict_length_weight_coefs["all"],
    impute_bins=True,
    minimum_count_threshold=5,
)

# COMBINE
da_binned_weight_table = xr.concat(
    [da_binned_weights_sex, da_binned_weights_all],
    dim = "sex"
)
# ==================================================================================================
# COMPUTE COUNT DISTRIBUTIONS PER AGE- AND LENGTH-BINS
logging.info(
    "Computing the counts per age- and length-bins across sex.\n"
    "     Stratifying by: 'stratum_ks'"
    "     Grouping by: 'sex'"
    )

# LENGTH DATASET
da_counts_length = proportions.compute_binned_counts(
    data=dict_df_bio["length"].copy().dropna(subset=["length"]),
    groupby_cols=["stratum_ks", "length_bin", "sex"],
    count_col="length_count",
    agg_func="sum",
)

# SPECIMEN DATASET
da_counts_specimen = proportions.compute_binned_counts(
    data=dict_df_bio["specimen"].dropna(subset=["length"]),
    groupby_cols=["stratum_ks", "length_bin", "sex"],
    count_col="length",
    agg_func="size",
) 

# COMBINE INTO A SINGLE DATASET
da_counts = da_counts_length + da_counts_specimen

# ==================================================================================================
# COMPUTE NUMBER PROPORTIONS
ds_number_proportion = proportions.number_proportions(
    data=da_counts,
    stratum_dim="stratum_ks",
)

# ==================================================================================================
# COMPUTE BINNED WEIGHTS

# SPECIMEN
da_weight_dist_specimen = proportions.binned_weights(
    length_data=dict_df_bio["specimen"],
    include_filter={"sex": ["female", "male"]},
    interpolate_regression=False,
    group_columns=["stratum_ks", "sex"],
)

# LENGTH
da_weight_dist_length = proportions.binned_weights(
    length_data=dict_df_bio["length"],
    include_filter={"sex": ["female", "male"]},
    interpolate_regression=True,
    length_weight_data=da_binned_weight_table,
    group_columns=["stratum_ks", "sex"],
)

# COMBINE
da_weight_dist = da_weight_dist_specimen + da_weight_dist_length

# ==================================================================================================
# COMPUTE WEIGHT PROPORTIONS
da_weight_proportions_initial = proportions.weight_proportions(
    weight_data=da_weight_dist, 
    catch_data=dict_df_bio["catch"], 
    stratum_dim="stratum_ks"
)

# READJUST/NORMALIZE
da_weight_proportions = (
    da_weight_proportions_initial["proportion"]
    / da_weight_proportions_initial["proportion"].sum(dim=["length_bin", "sex"])
)

# ==================================================================================================
# NASC TO BIOMASS CONVERSION
# ==================================================================================================
# INVERSION
# DEFINE INVERSION MODEL PARAMETERS
MODEL_PARAMETERS = {
    "ts_length_regression": {
        "slope": 20.,
        "intercept": -68.
    },
    "stratify_by": ["stratum_ks"],
    "expected_strata": df_dict_strata["ks"].stratum_num.unique(),
    "impute_missing_strata": True,
    "haul_replicates": True,
    "haul_column": "uid",
}

# INITIALIZE INVERSION OBJECT
invert_hake = inversion.InversionLengthTS(MODEL_PARAMETERS)

# INVERT NUMBER DENSITY
df_nasc = invert_hake.invert(
    nasc_data=df_nasc, length_data=[dict_df_bio["length"], dict_df_bio["specimen"]]
)

# ==================================================================================================
# CONVERT TO BIOMASS

# SET TRANSECT INTERVAL DISTANCES
transect.compute_interval_distance(nasc_data=df_nasc, interval_threshold=0.05)

# SET TRANSECT INTERVAL AREAS
df_nasc["area_interval"] = (
    df_nasc["transect_spacing"] * df_nasc["distance_interval"]
)

# COMPUTE ABUNDANCE
biology.compute_abundance(
    transect_data=df_nasc,
    exclude_filter={"sex": "unsexed"},
    number_proportions=ds_number_proportion,
)

# COMPUTE STRATUM-AVERAGED WEIGHTS
da_averaged_weight = proportions.stratum_averaged_weight(
    number_proportions=ds_number_proportion,
    length_weight_data=da_binned_weight_table,
    stratum_dim="stratum_ks"
)

# COMPUTE BIOMASS
biology.compute_biomass(
    transect_data=df_nasc,
    stratum_weights=da_averaged_weight,
)