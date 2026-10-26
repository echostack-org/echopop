3####################################################################################################
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DATA ROOT DIRECTORY
DATA_ROOT = Path("C:/Data/EchopopData/echopop_2019")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ALREADY PROCESSED NASC FILE ? 
# ---- When False, the raw NASC exports will be processed. When True, the pre-formatted NASC 
# ---- spreadsheet will be read in. This also requires defining `NASC_EXPORTS_SHEET`
NASC_PREPROCESSED = True
# NASC EXPORTS FILE(S)
NASC_EXPORTS_FILES = DATA_ROOT / "Exports/US&CAN_detailsa_2019_table2y+_ALL_final - updated.xlsx"
# NASC EXPORTS SHEET
NASC_EXPORTS_SHEET = "Sheet1"
# REMOVE AGE-1 (I.E., AGE-2+ ONLY)?
REMOVE_AGE1 = True
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BIODATA FILE
BIODATA_FILES = {
    "catch": DATA_ROOT / "Biological/database_view/echopop_catch.csv",
    "specimen": DATA_ROOT / "Biological/database_view/echopop_fish.csv",
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

# NET SELECTIVITY PARAMETERIZATION
NET_SELECTIVITY = {
    "l50": 10.9,
    "sr": 14.0,
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
import numpy as np
import pandas as pd
import xarray as xr
import echopop.ingest as ingestion
from echopop import geostatistics, inversion, utils
from echopop.survey import apportionment, biology, proportions, selectivity, transect
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

# INGEST NASC DATA 
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
    "haul_weight": "weight",
    "species_id": "species_code",
}

# READ IN DATA
dict_df_bio = ingestion.load_biodata_views(
    biodata_filepaths=BIODATA_FILES,
    column_name_map=FEAT_TO_ECHOPOP_BIODATA_COLUMNS,
    survey_subset=BIODATA_SHIP_SPECIES, 
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
# ----> Replace length values
dict_df_bio["specimen"] = (
    dict_df_bio["specimen"].dropna(subset=["length"])
)
dict_df_bio["specimen"]["length"] = (
    dict_df_bio["specimen"].copy()["length_bin"].map(lambda x: x.mid).astype(float)
)

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
# APPLY NET SELECTIVITY CORRECTION
# ---- Assign gear-type to specimen-level data
specimen_data = dict_df_bio["specimen"].merge(
    dict_df_bio["catch"][["uid", "gear"]].drop_duplicates("uid"),
    on="uid",
    how="left",
)
# ---- Assign selectivity expansion to each fish
specimen_data_selectivity = selectivity.assign_selectivity_expansion(
    specimen_data,
    NET_SELECTIVITY,
    net_column = "gear",
)

# ==================================================================================================
# COMPUTE COUNT DISTRIBUTIONS PER AGE- AND LENGTH-BINS [AT HAUL-LEVEL]
da_count_distribution_hauls = proportions.compute_binned_counts(
    data=specimen_data_selectivity.dropna(subset=["length"]),
    groupby_cols=["stratum_ks", "length_bin", "age_bin", "sex", "uid"],
    count_col="selectivity_expansion",
    agg_func="sum",
)

# ==================================================================================================
# COMPUTE COUNT DISTRIBUTIONS PER AGE- AND LENGTH-BINS [AT STRATUM-LEVEL]
da_count_distribution_strata = da_count_distribution_hauls.sum(dim=["uid"])

# ==================================================================================================
# NUMBER PROPORTIONS
ds_count_proportions = proportions.number_proportions(
    data=da_count_distribution_strata,
    stratum_dim="stratum_ks",
    exclude_filters={"sex": "unsexed"}
)

# ==================================================================================================
# FITTED WEIGHT PROPORTIONS
ds_weight_proportions = proportions.fitted_weight_proportions_combined(
    number_proportions=ds_count_proportions,
    binned_weights=da_binned_weight_table,
    stratum_dim="stratum_ks",
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
    nasc_data=df_nasc, length_data=dict_df_bio["specimen"]
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
    number_proportions=ds_count_proportions,
)

# COMPUTE STRATUM-AVERAGED WEIGHTS
da_averaged_weight = proportions.stratum_averaged_weight(
    number_proportions=ds_count_proportions,
    length_weight_data=da_binned_weight_table,
    stratum_dim="stratum_ks"
)

# COMPUTE BIOMASS
biology.compute_biomass(
    transect_data=df_nasc,
    stratum_weights=da_averaged_weight,
)

# AGE-1 CONTRIBUTION REMOVAL
if REMOVE_AGE1:
    # NASC
    age1_nasc_proportions = proportions.get_nasc_proportions_slice(
        number_proportions=ds_count_proportions,
        stratum_dim="stratum_ks",
        ts_length_regression_parameters={"slope": 20.0, "intercept": -68.0},
        include_filter={"age_bin": [1]},
    )

    # NUMBER
    age1_number_proportions = proportions.get_number_proportions_slice(
        number_proportions=ds_count_proportions,
        stratum_dim="stratum_ks",
        include_filter={"age_bin": [1]},
    )

    # WEIGHT
    age1_weight_proportions = proportions.get_weight_proportions_slice(
        weight_proportions=ds_weight_proportions,
        stratum_dim="stratum_ks",
        include_filter={"age_bin": [1]},
        number_proportions=ds_count_proportions,
        length_threshold_min=10.0,
        weight_proportion_threshold=1e-10,
    )

    # APPLY REMOVAL
    df_nasc_proc = apportionment.remove_group_from_estimates(
        transect_data=df_nasc,
        group_proportions=xr.Dataset({
            "nasc": age1_nasc_proportions,
            "abundance": age1_number_proportions,
            "biomass": age1_weight_proportions,
        }),
    )
else:
    df_nasc_proc = df_nasc.copy()
    
# ==================================================================================================
# GEOSTATISTICS
# ==================================================================================================
# ==================================================================================================
# COORDINATE TRANSFORMATION
# NASC
df_nasc_proc, delta_longitude, delta_latitude = geostatistics.transform_coordinates(
    data = df_nasc_proc,
    reference = df_isobath,
    x_offset = -124.78338,
    y_offset = 45.,   
)

# MESH
df_mesh, _, _ = geostatistics.transform_coordinates(
    data = df_mesh,
    reference = df_isobath,
    x_offset = -124.78338,
    y_offset = 45.,   
    delta_x=delta_longitude,
    delta_y=delta_latitude
)

# ==================================================================================================
# KRIGING ANALYSIS
# KRIGING PARAMETERS CONTAINER
KRIGING_PARAMETERS = {
    "search_radius": dict_variogram_params["correlation_range"] * 3,
    "aspect_ratio": 0.001,
    "k_min": 3,
    "k_max": 10,
}  

# VARIOGRAM PARAMETERS CONTAINER
VARIOGRAM_PARAMETERS = {
    "model": ["exponential", "bessel"],
    "sill": dict_variogram_params["sill"],
    "nugget": dict_variogram_params["nugget"],
    "correlation_range": dict_variogram_params["correlation_range"],
    "hole_effect_range": dict_variogram_params["hole_effect_range"],
    "decay_power": dict_variogram_params["decay_power"],
}

# INITIALIZE CLASS OBJECT
krg = geostatistics.Kriging(
    mesh=df_mesh,
    kriging_params=KRIGING_PARAMETERS,
    variogram_params=VARIOGRAM_PARAMETERS,
    coordinate_names=("x", "y"),
)

# REGISTER KRIGING METHOD
krg.register_search_strategy("FEAT_strategy", utils.feat_functions.western_boundary_search_strategy)
# ---- Parameterize
transect_western_extents = utils.feat_functions.get_survey_western_extents(
    transects=df_nasc_proc, coordinate_names=("x", "y"), latitude_threshold=51.0
)
FEAT_STRATEGY_KWARGS = {
    "western_extent": transect_western_extents,
}

# RUN KRIGING
df_kriged_results = krg.krige(
    transects=df_nasc_proc,
    variable="biomass_density",
    extrapolate=True,
    default_mesh_cell_area=6.25,
    adaptive_search_strategy="FEAT_strategy",
    custom_search_kwargs=FEAT_STRATEGY_KWARGS
)

# ==================================================================================================
# CONVERT BIOMASS DENSITY TO NASC
# CONVERT TO BIOMASS
df_kriged_results["biomass"] = df_kriged_results["biomass_density"] * df_kriged_results["area"]

# BIOMASS TO NASC
apportionment.mesh_biomass_to_nasc(
    mesh_data=df_kriged_results,
    biodata=ds_weight_proportions,
    group_columns=["sex", "stratum_ks"],
    mesh_biodata_link={"geostratum_ks": "stratum_ks"},
    stratum_weights=da_averaged_weight.sel(sex="all"),
    stratum_sigma_bs=invert_hake.sigma_bs_strata,  
)

# ==================================================================================================
# DISTRIBUTE POPULATION ESTIMATES ACROSS AGE AND LENGTH BINS
# ABUNDANCE [ALL]
ds_kriged_abundance_table = apportionment.distribute_population_estimates(
    data=df_kriged_results,
    proportions = ds_count_proportions,
    variable = "abundance",
    group_columns = ["sex", "age_bin", "length_bin", "stratum_ks"],
    data_proportions_link={"geostratum_ks": "stratum_ks"}
)

# BIOMASS [ALL]
ds_kriged_biomass_table = apportionment.distribute_population_estimates(
    data = df_kriged_results,
    proportions = ds_weight_proportions,
    variable = "biomass",
    group_columns = ["sex", "age_bin", "length_bin", "stratum_ks"],
    data_proportions_link={"geostratum_ks": "stratum_ks"}
)

# CONSOLIDATE
# ---- ABUNDANCE
da_kriged_abundance_table = apportionment.sum_population_tables(
    population_tables={
        "all": ds_kriged_abundance_table.fillna(0.) ,
    },
)
# ---- BIOMASS
da_kriged_biomass_table = apportionment.sum_population_tables(
    population_tables={
        "all": ds_kriged_biomass_table.fillna(0.)
    },
)

# AGE-1 REALLOCATION ?
if REMOVE_AGE1:
    # REDISTRIBUTE AGE-1 ABUNDANCES
    da_kriged_abundance_table_proc = apportionment.reallocate_excluded_estimates(
        population_table=da_kriged_abundance_table,
        exclusion_filter={"age_bin": [1]},
        group_columns=["sex"],
    )

    # REDISTRIBTUE AGE-1 BIOMASS
    da_kriged_biomass_table_proc = apportionment.reallocate_excluded_estimates(
        population_table=da_kriged_biomass_table,
        exclusion_filter={"age_bin": [1]},
        group_columns=["sex"],
    )
else:
    da_kriged_abundance_table_proc = da_kriged_abundance_table
    da_kriged_biomass_table_proc = da_kriged_biomass_table