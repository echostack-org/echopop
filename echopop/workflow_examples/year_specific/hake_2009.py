####################################################################################################
# 2009
# ----
from pathlib import Path
from echopop.workflow_examples import cli_utils
####################################################################################################
# PARAMETER ENTRY
# ---------------
# ** ENTER FILE INFORMATION FOR ALL INGESTED DATASETS.
# ** ADDITIONAL PARAMETERIZATIONS THROUGHOUT THE SCRIPT SHOULD BE EDITED BASED ON SPECIFIC NEEDS. 
# ** MAKE SURE TO EDIT WITH CARE.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRINT CONSOLE LOGGING MESSAGES
# ---- When set to `True`, logging information will be printed in the terminal/console as the 
# ---- script progresses
try: 
    # ---- FOR CLI USE
    VERBOSE = cli_utils.get_verbose()
except Exception:
    # ---- FOR INTERACTIVE REPL USE
    VERBOSE = True
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DATA ROOT DIRECTORY
DATA_ROOT = Path("C:/Data/EchopopData/file_all_years_update_20260303/2009")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# REPORTS SAVE DIRECTORY
REPORTS_DIR = DATA_ROOT / "output_echopop"
# COMPARE TO ECHOPRO REPORTS?
try:
    # ---- FOR CLI USE
    COMPARE = cli_utils.get_compare()
    ECHOPRO_REPORTS_DIR = DATA_ROOT / "output_echopro"
    COMPARISONS_DIR = DATA_ROOT / "comparisons"
    SHOW_PLOT = False
except Exception:
    # ---- FOR INTERACTIVE REPL USE
    COMPARE = False
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ALREADY PROCESSED NASC FILE ? 
# ---- When False, the raw NASC exports will be processed. When True, the pre-formatted NASC 
# ---- spreadsheet will be read in. This also requires defining `NASC_EXPORTS_SHEET`
NASC_PREPROCESSED = True
# NASC EXPORTS FILE(S)
NASC_EXPORTS_FILES = DATA_ROOT / "input/Exports/US&CAN_detailsa_2009_table2y+_ALL_final.xlsx"
# NASC EXPORTS SHEET
NASC_EXPORTS_SHEET = "Sheet1"
# REMOVE AGE-1 (I.E., AGE-2+ ONLY)?
REMOVE_AGE1 = True
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRANSECT REGION HAUL MAPPING FILE
TRANSECT_REGION_HAUL_FILE = (
    DATA_ROOT / "input/Stratification/US&CAN_T_reg_haul_final.xlsx"
)
# TRANSECT REGION HAUL MAPPING SHEET
TRANSECT_REGION_HAUL_SHEET = "US&CAN_T_reg_haul_final"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRANSECT BOUNDARY FILE
TRANSECT_BOUNDARY_FILE = DATA_ROOT / "input/Kriging_files/Kriging_grid_files/Transect Bounds to 2011.xlsx"
# TRANSECT BOUNDARY SHEET
TRANSECT_BOUNDARY_SHEET = "1995-2011"
# SURVEY FILTER
SURVEY_FILTER = "survey in [200903, 200952]"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BIODATA FILE
BIODATA_FILE = DATA_ROOT / "input/Biological/1995-2025_Survey_Biodata.xlsx"
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
CAN_HAUL_OFFSET = 900
SPECIES_ID = 22500 # numeric species code for Pacific hake
SHIP_US = 21 # US ship ID
SHIP_CAN = 499 # CAN ship ID
SURVEY_US = 200903 # US survey identifier
SURVEY_CAN = 200952 # CAN survey identifier

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
# BIODATA PROCESSING: AGE-1 DOMINATED HAULS
# ---- This is a list of age-1 dominated haul numbers that should be designated for removal. If no
# ---- hauls should be removed, then set `AGE1_DOMINATED_HAULS` to `[]`
AGE1_DOMINATED_HAULS = []
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HAUL STRATIFICATION FILE
HAUL_STRATA_FILE = (
    DATA_ROOT / 
    "input/Stratification/US&CAN strata 2009.xlsx"
)
# HAUL STRATIFICATION SHEET MAP
# ---- Valid keys are limited to "ks" and "inpfc"
HAUL_STRATA_SHEETS = {
    "inpfc": "INPFC",
    "ks": "Base KS",
}
# GEOGRAPHIC STRATIFICATION FILE
GEOSTRATA_FILE = (
    DATA_ROOT / 
    "input/Stratification/Stratification_geographic_Lat_2009.xlsx"
)
# GEOGRAPHIC STRATIFICATION SHEET MAP
# ---- Valid keys are limited to "ks" and "inpfc"
GEOSTRATA_SHEETS = {
    "inpfc": "INPFC",
    "ks": "stratification1",
}
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# KRIGING MESH FILE 
KRIGING_MESH_FILE = (
    DATA_ROOT / "input/Kriging_files/Kriging_grid_files/krig_grid2_5nm_cut_centroids_2013.xlsx"
)
# KRIGING MESH SHEET
KRIGING_MESH_SHEET = "krigedgrid2_5nm_forChu"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# KRIGING AND VARIOGRAM PARAMETERS FILE
KRIGING_VARIOGRAM_PARAMETERS_FILE = (
    DATA_ROOT / "input" / "Kriging_files/default_vario_krig_settings_final.xlsx"
)
# KRIGING AND VARIOGRAM PARAMETERS SHEET
KRIGING_VARIGORAM_PARAMETERS_SHEET = "Sheet1"
# USE DEFAULT VALUES OR OPTIMIZE?
OPTIMIZE_VARIOGRAM = False
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 200m ISOBATH FILE
ISOBATH_FILE = (
    DATA_ROOT / "input/Kriging_files/Kriging_grid_files/transformation_isobath_coordinates.xlsx"
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
from lmfit import Parameters
from echopop.utils import feat_functions as feat
from echopop.reports import Reporter, compare
import echopop.ingest as ingestion
from echopop import geostatistics, inversion, utils
from echopop.survey import apportionment, biology, proportions, stratified, transect
####################################################################################################
# FORMAT LOGGER
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
logging.basicConfig(
    level=logging.INFO if VERBOSE else logging.WARNING,
    format="%(message)s")
# ==================================================================================================
# DATA INGESTION 
# ==================================================================================================
# FORMAT HAUL-BASED UID 
logging.info(f"Creating haul-based UID...")
HAUL_UID_CONFIG = {
    "ship_id": {"US": SHIP_US, "CAN": SHIP_CAN},
    "survey_id": {"US": SURVEY_US, "CAN": SURVEY_CAN},
    "species_id": SPECIES_ID,
    "haul_offset": CAN_HAUL_OFFSET
}

# INGEST NASC DATA 
if NASC_PREPROCESSED:
    logging.info(f"Reading pre-generated NASC export file: '{NASC_EXPORTS_FILES.as_posix()}'.")

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
else:
    logging.info(
        f"Beginning NASC export ingestion for files in: '{NASC_EXPORTS_FILES.as_posix()}'."
    )

    # MERGE EXPORTS
    logging.info(
        "---- Merging NASC exports...\n"
        "     Filename transect pattern: 'T(\\d+)'\n"
        "     Default transect spacing: 10.0 nmi\n"
        "     Default latitude threshold: 60.0 deg."
    )    

    df_intervals, df_exports = ingestion.nasc.merge_echoview_nasc(
        file_directory = NASC_EXPORTS_FILES,
        filename_transect_pattern = "T(\\d+)",
        default_transect_spacing = 10.0,
        default_latitude_threshold = 60.0,
    )

    # GENERATE TRANSECT-REGION-HAUL KEY
    logging.info(
        "---- Loading transect-region-haul key mapping\n"
    )

    # TRANSECT REGION HAUL KEY NAME MAPPING
    TRANSECT_REGION_FILE_RENAME = {
        "transect": "transect_num",
        "region id": "region_id",
        "assigned haul": "haul_num",
    }

    # LOAD
    df_transect_region_haul_key = ingestion.nasc.read_transect_region_haul_key(
        filename=TRANSECT_REGION_HAUL_FILE,
        sheetname=TRANSECT_REGION_HAUL_SHEET,
        column_name_map=TRANSECT_REGION_FILE_RENAME
    )
    
    # GENERATE TRANSECT-REGION-HAUL KEY
    if REMOVE_AGE1:
        CLASS_REGIONS = ["Hake", "Hake Mix"]
    else:
        CLASS_REGIONS = ["Age-1 Hake", "Age-1 Hake Mix", "Hake", "Hake Mix"]
    logging.info(
        f"---- Generating transect-region-haul key mapping\n"
        f"     Searching for the export regions: {', '.join(CLASS_REGIONS)}"
    )
    
    # AGE-1 DOMINATED HAUL REMOVAL
    if REMOVE_AGE1:
        logging.info(
            f"The following age-1 dominated haul numbers have been designated for removal from "
            f"transect-region-haul key mapping:\n"
            f"{', '.join(map(str, AGE1_DOMINATED_HAULS))}."
        )
        df_transect_region_haul_key = utils.apply_filters(
            df_transect_region_haul_key, exclude_filter={"haul_num": AGE1_DOMINATED_HAULS}
        )

    # CONSOLIDATE THE EXPORTS WITH TRANSECT-REGION-HAUL MAPPINGS
    logging.info(
        "---- Finalizing NASC export ingestion\n"
        "     Searching for the export regions: 'Age-1 Hake', 'Age-1 Hake Mix', 'Hake', 'Hake Mix'\n"
        "     Imputing overlapping region IDs within each interval: True"
    )
    df_nasc = ingestion.nasc.consolidate_echvoiew_nasc(
        nasc_data=df_exports,
        interval_data=df_intervals,
        region_class_names=CLASS_REGIONS,
        impute_region_ids=True,
        transect_region_haul_key=df_transect_region_haul_key,
        haul_uid_config=HAUL_UID_CONFIG
    )

# FILTER
logging.info(
    f"---- Filtering out off-effort transect intervals based on: {TRANSECT_BOUNDARY_FILE}\n"
    f"     Survey filter: '{SURVEY_FILTER}'"
)
df_nasc = feat.filter_transect_intervals(
    nasc_data=df_nasc, 
    transect_filter=TRANSECT_BOUNDARY_FILE,
    transect_filter_sheet=TRANSECT_BOUNDARY_SHEET,
    survey_filter=SURVEY_FILTER
)
logging.info(
    "NASC ingestion complete\n"
    "'df_nasc' created."
)

# DROP TRANSECTS
df_nasc = utils.apply_filters(df_nasc, include_filter={"transect_num": np.arange(1, 144)})
# ==================================================================================================
# INGEST BIODATA
logging.info(
    f"Beginning biodata ingestion for: '{BIODATA_FILE.as_posix()}'."
)

# BIODATA DATAFRAME COLUMN NAME MAPPING
FEAT_TO_ECHOPOP_BIODATA_COLUMNS = {
    "frequency": "length_count",
    "haul": "haul_num",
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
logging.info(
    "Biodata ingestion complete\n"
    "'dict_df_bio' created."
)

# AGE-1 DOMINATED HAUL REMOVAL
if len(AGE1_DOMINATED_HAULS) > 0:
    logging.info(
        f"The following age-1 dominated haul numbers have been designated for removal from "
        f"biodata:\n"
        f"{', '.join(map(str, AGE1_DOMINATED_HAULS))}."
    )
    dict_df_bio = {
        key: utils.apply_filters(dataset, exclude_filter={"haul_num": AGE1_DOMINATED_HAULS})
        for key, dataset in dict_df_bio.items()
    }
    logging.info(
        f"The following age-1 dominated haul numbers were successfully removed from the biodata:\n"
        f"{', '.join(map(str, AGE1_DOMINATED_HAULS))}."
    )
# ==================================================================================================
# INGEST STRATIFICATION DATA
logging.info(
    "Loading stratification files..."
)

# HAUL-BASED STRATIFICATION DATAFRAME COLUMN NAME MAPPING
FEAT_TO_ECHOPOP_STRATA_COLUMNS = {
    "wt": "nasc_proportion",
    "haul": "haul_num",
    "haul ": "haul_num",
    "strata": "stratum_num",
    "length": "stratum_length"
}

# READ IN STRATA FILE 
logging.info(
    f"Load in haul-based stratification: '{HAUL_STRATA_FILE.as_posix()}'."
)
df_dict_strata = ingestion.load_strata(
    strata_filepath=HAUL_STRATA_FILE, 
    strata_sheet_map=HAUL_STRATA_SHEETS, 
    column_name_map=FEAT_TO_ECHOPOP_STRATA_COLUMNS,
    haul_uid_config=HAUL_UID_CONFIG,
)
logging.info(
    "Haul-based stratification loading complete\n"
    "'df_dict_strata' created."
)

# GEOGRAPHIC-BASED STRATIFICATION DATAFRAME COLUMN NAME MAPPING
FEAT_TO_ECHOPOP_GEOSTRATA_COLUMNS = {
    "latitude (upper limit)": "northlimit_latitude",
    "strata": "stratum_num",
    "strata index": "stratum_num",
}

# READ IN GEOSTRATA FILE
logging.info(
    f"Load in geographic-based stratification: '{GEOSTRATA_FILE.as_posix()}'."
)
df_dict_geostrata = ingestion.load_geostrata(
    geostrata_filepath=GEOSTRATA_FILE, 
    geostrata_sheet_map=GEOSTRATA_SHEETS, 
    column_name_map=FEAT_TO_ECHOPOP_GEOSTRATA_COLUMNS
)
logging.info(
    "Geographic-based stratification loading complete\n"
    "'df_dict_geostrata' created."
)
# ==================================================================================================
# LOAD KRIGING MESH FILE
logging.info(
    f"Loading kriging mesh file: '{KRIGING_MESH_FILE.as_posix()}'."
)

# KRIGING MESH DATAFRAME COLUMN NAME MAPPING
FEAT_TO_ECHOPOP_MESH_COLUMNS = {
    "latitude of centroid": "latitude",
    "longitude of centroid": "longitude",
    "cell portion": "fraction",
}

# LOAD MESH
df_mesh = ingestion.load_mesh_data(
    mesh_filepath=KRIGING_MESH_FILE, 
    sheet_name=KRIGING_MESH_SHEET, 
    column_name_map=FEAT_TO_ECHOPOP_MESH_COLUMNS
)
logging.info(
    "Kriging mesh loading complete\n"
    "'df_mesh' created."
)
# ==================================================================================================
# LOAD ISOBATH FILE
logging.info(
    f"Loading isobath file: '{ISOBATH_FILE}'."
)
df_isobath = ingestion.load_isobath_data(
    isobath_filepath=ISOBATH_FILE,
    sheet_name=ISOBATH_SHEET
)
logging.info(
    f"Iosbath loading complete\n"
    "'df_isobath' created."
)
# ==================================================================================================
# LOAD KRIGING AND VARIOGRAM PARAMETERS
logging.info(
    f"Loading variogram and kriging parameters: '{KRIGING_VARIOGRAM_PARAMETERS_FILE.as_posix()}'."
)

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
logging.info(
    "Variogram and kriging parameter loading complete\n"
    "---- 'dict_variogram_params' created [variogram]\n"
    "---- 'dict_kriging_params' created [kriging]"
)
# ==================================================================================================
# INITIAL DATA PROCESSING
# ==================================================================================================
# APPLY STRATIFICATION DEFINITIONS TO BIODATA AND NASC
logging.info("Applying strata to datasets...")

# HAUL-BASED STRATA
logging.info(
    "Applying haul-based strata to 'dict_df_bio' and 'df_nasc'.\n"
    "     Using column 'uid' as haul-based unique identifier\n"
    "     Default stratum: 0\n"
    "     New columns:\n"
    "         INPFC: 'stratum_inpfc'\n"
    "         KS: 'stratum_ks'"
)
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
logging.info(
    "Applying geographic-based strata to 'df_nasc' and 'df_mesh.\n"
    "     New columns:\n"
    "         INPFC: 'geostratum_inpfc'\n"
    "         KS: 'geostratum_ks'"
)
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
logging.info("Strata application complete!")
# ==================================================================================================
# BIN DATA USING EXPLICIT EDGES
logging.info(
    "Binning biodata ['dict_df_bio'] into discrete age and length bins\n"
    "     Age bin edges: [0.5, 1.5, ..., 21.5, 22.5]\n"
    "     Length bin edges: [1.0, 3.0, ..., 79.0, 81.0]\n"
    "     New columns:\n"
    "         Age: 'age_bin'\n"
    "         Length: 'length_bin'"
)

# AGE-BINS
AGE_BIN_EDGES = np.arange(0.5, 23.0, 1.0)
dict_df_bio["specimen"]["age_bin"] = pd.cut(dict_df_bio["specimen"]["age"], bins=AGE_BIN_EDGES)

# LENGTH-BINS
LENGTH_BIN_EDGES = np.arange(1.0, 82.0, 2.0)
for dataset in ("specimen", "length"):
    dict_df_bio[dataset]["length_bin"] = pd.cut(dict_df_bio[dataset]["length"], bins=LENGTH_BIN_EDGES)
logging.info("Age and length binning complete!")
# ==================================================================================================
# FIT LENGTH-WEIGHT REGRESSION
logging.info("Fitting length-weight regression for each sex and for all fish.")

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
logging.info("Fitting length-weight regression for each sex and for all fish complete!")
# ==================================================================================================
# COMPUTE MEAN WEIGHTS PER LENGTH BIN
logging.info(
    "Computing the mean weight per length bin for each sex and for all fish.\n"
    "     Impute missing length bins using modeled weights: True"
    "     Minimum specimen count per bin: 5"
    )

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
# COMPUTE NUMBER PROPORTIONS
logging.info(
    "Computing number proportions across age and length bins\n"
    "     Stratifying by: 'stratum_ks'\n"
    "     Excluding: 'sex'='unsexed' from 'dict_df_counts['aged']'"
    )
dict_ds_number_proportion = proportions.number_proportions(
    data=ds_counts,
    stratum_dim="stratum_ks",
    exclude_filters={"aged": {"sex": "unsexed"}},
)
logging.info(
    "Number proportions calculation complete\n"
    "'dict_df_number_proportions' created\n"
    )
# ==================================================================================================
# COMPUTE BINNED WEIGHTS
logging.info(
    "Computing the summed weights per age- and length-bins across sex.\n"
    "     Stratifying by: 'stratum_ks'"
    "     Grouping by: 'sex'"
    "     Excluding: 'sex'='unsexed'"
    )

# DATASET CONTAINER
ds_da_weight_dist = xr.Dataset()

# AGED
ds_da_weight_dist["aged"] = proportions.binned_weights(
    length_data=dict_df_bio["specimen"],
    include_filter={"sex": ["female", "male"]},
    interpolate_regression=False,
    group_columns=["stratum_ks", "sex", "age_bin"],
)

# UNAGED
logging.info(
    "Unaged binned weights require additional processing steps.\n"
    "     Interpolating binned length-weight regression estimates: True"
    )
ds_da_weight_dist["unaged"] = proportions.binned_weights(
    length_data=dict_df_bio["length"],
    include_filter={"sex": ["female", "male"]},
    interpolate_regression=True,
    length_weight_data=da_binned_weight_table,
    group_columns=["stratum_ks", "sex"],
)
# ==================================================================================================
# COMPUTE WEIGHT PROPORTIONS
logging.info(
    "Computing weight proportions across age and length bins\n"
    "     Stratifying by: 'stratum_ks'"
    "     Grouping by: 'sex'"
    )

# DATAARRAY CONTAINER
dict_da_weight_proportion = {}

# AGED WEIGHT PROPORTIONS
logging.info("Computing aged weight proportions...")
dict_da_weight_proportion["aged"] = proportions.weight_proportions(
    weight_data=ds_da_weight_dist["aged"], 
    catch_data=dict_df_bio["catch"], 
    stratum_dim = "stratum_ks"
)

# UNAGED WEIGHT PROPORTIONS
logging.info(
    "Computing unaged weight proportions\n"
    "     Scaling weight proportions in reference to the aged estimates"
    )
dict_da_weight_proportion["unaged"] = proportions.fitted_weight_proportions(
    weight_data=ds_da_weight_dist["unaged"],
    aged_weight_proportions=dict_da_weight_proportion["aged"],
    number_proportions=dict_ds_number_proportion["unaged"],
    binned_weights=da_binned_weights_all,
    stratum_dim="stratum_ks"
)

# ==================================================================================================
# NASC TO BIOMASS CONVERSION
# ==================================================================================================
# INVERSION
logging.info(
    "Beginning inversion based on hake-specific TS-length regression coefficients\n"
    "     Model: 20.0 x log[10](L) + -68.0\n"
    "     Stratifying by: 'stratum_ks'\n"
    "     Imputing missing strata: True\n"
    "     Treating hauls as replicates: True"
    )

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
logging.info("Inversion-class object 'invert_hake' created...")

# INVERT NUMBER DENSITY
df_nasc = invert_hake.invert(
    nasc_data=df_nasc, length_data=[dict_df_bio["length"], dict_df_bio["specimen"]]
)
logging.info(
    "Number density inversion complete\n"
    "     New column in 'df_nasc':\n"
    "         Number density (animals nm^-2): 'number_density'"
    )
# ==================================================================================================
# CONVERT TO BIOMASS
logging.info("Converting number density estimates into biomass estimates")

# SET TRANSECT INTERVAL DISTANCES
logging.info(
    "Defining transect interval distances...\n"
    "     Along-transect interval distance (nmi) threshold: 0.5 nmi"
)
transect.compute_interval_distance(nasc_data=df_nasc, interval_threshold=0.05)

# SET TRANSECT INTERVAL AREAS
logging.info("Defining transect interval areas...")
df_nasc["area_interval"] = (
    df_nasc["transect_spacing"] * df_nasc["distance_interval"]
)

# COMPUTE ABUNDANCE
logging.info(
    "Compute interval abundances...\n"
    "     Stratifying by: 'stratum_ks'\n"
    "     Grouping by: 'sex'\n"
    "     Excluding: 'sex'='unsexed' from 'dict_df_number_proportions'"    
)
biology.compute_abundance(
    transect_data=df_nasc,
    exclude_filter={"sex": "unsexed"},
    number_proportions=dict_ds_number_proportion,
)
# COMPUTE STRATUM-AVERAGED WEIGHTS
da_averaged_weight = proportions.stratum_averaged_weight(
    number_proportions=dict_ds_number_proportion,
    length_weight_data=da_binned_weight_table,
    stratum_dim="stratum_ks"
)


# COMPUTE BIOMASS
logging.info(
    "Compute interval biomass...\n"
    "     Stratifying by: 'stratum_ks'\n"
    "     Grouping by: 'sex'\n"  
)
biology.compute_biomass(
    transect_data=df_nasc,
    stratum_weights=da_averaged_weight,
)
# AGE-1 CONTRIBUTION REMOVAL
if REMOVE_AGE1:
    logging.info(
        "Removing age-1 contributions from NASC, abundance, and biomass estimates...\n"
        "     Stratifying by: 'stratum_ks'\n"
        "     Minimum length threshold for weight proportions: 10.0 cm\n"
        "     Minimum weight proportion threshold: 1E-10"
    )

    # NASC
    age1_nasc_proportions = proportions.get_nasc_proportions_slice(
        number_proportions=dict_ds_number_proportion["aged"],
        stratum_dim="stratum_ks",
        ts_length_regression_parameters={"slope": 20.0, "intercept": -68.0},
        include_filter={"age_bin": [1]},
    )

    # NUMBER
    age1_number_proportions = proportions.get_number_proportions_slice(
        number_proportions=dict_ds_number_proportion["aged"],
        stratum_dim="stratum_ks",
        include_filter={"age_bin": [1]},
    )

    # WEIGHT
    age1_weight_proportions = proportions.get_weight_proportions_slice(
        weight_proportions=dict_da_weight_proportion["aged"],
        stratum_dim="stratum_ks",
        include_filter={"age_bin": [1]},
        number_proportions=dict_ds_number_proportion,
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
    logging.info("Age-1 contribution removal complete\n" "'df_nasc_proc' created.")
else:
    df_nasc_proc = df_nasc.copy()
logging.info(
    "NASC to biomass conversion complete\n"
    "     New columns in 'df_nasc':\n"
    "         Sex-specific number densities (animals nmi^-2): "
    "'number_density_female'/'number_density_male'\n"
    "         Abundance (animals): 'abundance'/'abundance_female'/'abundance_male'\n"
    "         Biomass density (kg nmi^-2): 'biomass_density'/'biomass_density_female'/"
    "'biomass_density_male'\n"
    "         Biomass (kg): 'biomass'/'biomass_female'/'biomass_male'"
    )
# SUMMARIZE TRANSECT RESULTS
logging.info(
    f"----------------------\n"
    f"Transect-based results\n"
    f"     Total NASC: {df_nasc_proc['nasc'].sum():.1f} m²nmi⁻²\n"
    f"     Mean number density: {df_nasc_proc['number_density'].mean():.1f} animals nmi⁻²\n"
    f"     Total abundance: {df_nasc_proc['abundance'].sum():.0f} fish\n"
    f"     Mean biomass density: {df_nasc_proc['biomass_density'].mean() * 1e-6:.3f} kmt nmi⁻²\n"
    f"     Total biomass: {df_nasc_proc['biomass'].sum() * 1e-6:.1f} kmt"
)
# ==================================================================================================
# DISTRIBUTE POPULATION ESTIMATES ACROSS AGE AND LENGTH BINS
logging.info(
    "Distribute population estimates across age- and length-bins\n"
    "     Stratifying by: 'stratum_ks'\n"
    "     Grouping by: 'sex'"
)

# ABUNDANCE
logging.info("Distributing abundances...")
dict_ds_transect_abundance_table = apportionment.distribute_population_estimates(
    data = df_nasc,
    proportions = dict_ds_number_proportion,
    variable = "abundance",
    group_columns = ["sex", "age_bin", "length_bin", "stratum_ks"]
)

logging.info("Abundance distributions complete\n'dict_ds_transect_abundance_table' created.")
# BIOMASS [ALL]
logging.info("Distributing biomass...")
dict_ds_transect_biomass_table = apportionment.distribute_population_estimates(
    data=df_nasc,
    proportions=dict_da_weight_proportion,
    variable = "biomass",
    group_columns = ["sex", "age_bin", "length_bin", "stratum_ks"]
)

dict_ds_transect_biomass_table[
    "standardized_unaged"
] = apportionment.distribute_unaged_from_aged(
    population_table = dict_ds_transect_biomass_table["unaged"],
    reference_table = dict_ds_transect_biomass_table["aged"],
    stratum_dim = "stratum_ks",
    impute = False 
)

da_transect_biomass_table = apportionment.sum_population_tables(
    population_tables={
        "aged": dict_ds_transect_biomass_table["aged"],
        "unaged": dict_ds_transect_biomass_table["standardized_unaged"]
    },
)

logging.info("Biomass distribution complete\n'dict_ds_transect_biomass_table' created.")
# BIOMASS [AGED-ONLY]
logging.info("Distributing biomass...\n     Aged-only weight proportions: True")
df_transect_aged_biomass_table = apportionment.distribute_population_estimates(
    data=df_nasc_proc,
    proportions=dict_da_weight_proportion["aged"],
    variable="biomass",
    group_columns = ["sex", "age_bin", "length_bin", "stratum_ks"]
)
# ==================================================================================================
# GEOSTATISTICS
# ==================================================================================================
# INITIAL
logging.info("Beginning geostatistical analysis...")
# ==================================================================================================
# COORDINATE TRANSFORMATION
logging.info(
    "Transform spatial coordinates for 'df_nasc' and 'df_mesh'\n"
    "     Reference coordinates: 'df_isobath'\n"
    "     Longitudinal offset: -124.78338 deg.E\n"
    "     Latitudinal offset: 45.0 deg.N"
)

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
logging.info(
    "Coordinate transformation complete\n"
    "     New columns:\n"
    "          Transformed longitude: 'x'\n"
    "          Transformed latitude: 'y'\n"
)
# ==================================================================================================
# VARIOGRAM ANALYSIS
if OPTIMIZE_VARIOGRAM:
    logging.info(
        "Beginning variogram analysis\n"
        "     Normalized lag resolution: 0.002\n"
        "     Number of lags: 30\n"
    )

    # INITIALIZE VARIOGRAM-CLASS OBJECT
    vgm = geostatistics.Variogram(
        lag_resolution=0.002,
        n_lags=30,
        coordinate_names=("x", "y"),
    )
    logging.info("Variogram-class object 'vgm' created...")

    # EMPIRICAL VARIOGRAM
    logging.info(
        "Computing the empirical variogram\n"
        "     Variable: 'biomass_density'\n"
        "     Applying azimuth angle filter: True\n"
        "     Azimuth angle filter: 180.0 deg.\n"
    )
    vgm.calculate_empirical_variogram(
        data=df_nasc_proc,
        variable="biomass_density",
        azimuth_filter=True,
        azimuth_angle_threshold=180.,
    )

    # SET UP FITTING PARAMETERS
    # ----- lmfit.Parameters tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    logging.info(
        f"Optimizing variogram parameters using non-linear least-squares\n"
        f"     Model: Exponential-Bessel (['exponential', 'bessel'])\n"
        f"     Initial values:\n"
        f"          Nugget: {dict_variogram_params["nugget"]}\n"
        f"          Sill: {dict_variogram_params["sill"]}\n"
        f"          Correlation range: {dict_variogram_params["correlation_range"]}\n"
        f"          Hole effect range: {dict_variogram_params["hole_effect_range"]}\n"
        f"          Decay power exponent: {dict_variogram_params["decay_power"]}"
    )
    variogram_parameters_lmfit = Parameters()
    variogram_parameters_lmfit.add_many(
        ("nugget", dict_variogram_params["nugget"], True, 0.),
        ("sill", dict_variogram_params["sill"], True, 0.),
        ("correlation_range", dict_variogram_params["correlation_range"], True, 0.),
        ("hole_effect_range", dict_variogram_params["hole_effect_range"], True, 0.),
        ("decay_power", dict_variogram_params["decay_power"], True, 1.25, 1.75),
    )

    # OPTIMIZATION PARAMETERS
    OPTIM_ARGS = {
        "max_nfev": None, "ftol": 1e-08, "gtol": 1e-8, "xtol": 1e-8, "diff_step": None, 
        "tr_solver": "exact", "x_scale": 1., "jac": "2-point"
    }
    logging.info(
        f"Optimization arguments:\n"
        f"{OPTIM_ARGS}"
    )

    # RUN MINIMIZER
    best_fit_parameters = vgm.fit_variogram_model(
        model=["exponential", "bessel"],
        model_parameters=variogram_parameters_lmfit,
        optimizer_kwargs=OPTIM_ARGS,
    )
    logging.info(
        f"Variogram parameter fitting complete\n"
        f"     Best-fit parameters:\n"
        f"     {best_fit_parameters}"
    )
else:
    best_fit_parameters = {
        "nugget": dict_variogram_params["nugget"],
        "sill": dict_variogram_params["sill"],
        "hole_effect_range": dict_variogram_params["hole_effect_range"],
        "correlation_range": dict_variogram_params["correlation_range"],
        "decay_power": dict_variogram_params["decay_power"]
    }
# ==================================================================================================
# KRIGING ANALYSIS
logging.info(
    f"Beginning kriging analysis\n"
    f"     Using best-fit variogram model and parameters: True\n"
    f"     Kriging parameters:\n"
    f"          Normalized search radius: {best_fit_parameters["correlation_range"] * 3}\n"
    f"          Minimum nearest-neighbor count: 3\n"
    f"          Maximum nearest-neighbor count: 10\n"
    f"          Anisotropic aspect ratio: 0.001"
)

# KRIGING PARAMETERS CONTAINER
KRIGING_PARAMETERS = {
    "search_radius": best_fit_parameters["correlation_range"] * 3,
    "aspect_ratio": 0.001,
    "k_min": 3,
    "k_max": 10,
}  

# VARIOGRAM PARAMETERS CONTAINER
VARIOGRAM_PARAMETERS = {
    "model": ["exponential", "bessel"],
    **best_fit_parameters
}

# INITIALIZE CLASS OBJECT
krg = geostatistics.Kriging(
    mesh=df_mesh,
    kriging_params=KRIGING_PARAMETERS,
    variogram_params=VARIOGRAM_PARAMETERS,
    coordinate_names=("x", "y"),
)
logging.info("Kriging-class object 'krg' created...")

# REGISTER KRIGING METHOD
krg.register_search_strategy("FEAT_strategy", feat.western_boundary_search_strategy)
# ---- Parameterize
transect_western_extents = feat.get_survey_western_extents(
    transects=df_nasc_proc, coordinate_names=("x", "y"), latitude_threshold=51.0
)
FEAT_STRATEGY_KWARGS = {
    "western_extent": transect_western_extents,
}

# RUN KRIGING
logging.info(
    "Interpolating population estimates using ordinary kriging\n"
    "     Variable: 'biomass_density'\n"
    "     Extrapolation (full mesh): True\n"
    "     Default mesh cell area: 6.25 nmi^2\n"
)
df_kriged_results = krg.krige(
    transects=df_nasc_proc,
    variable="biomass_density",
    extrapolate=True,
    default_mesh_cell_area=6.25,
    adaptive_search_strategy="FEAT_strategy",
    custom_search_kwargs=FEAT_STRATEGY_KWARGS
)
logging.info(
    f"Kriging complete\n"
    f"'df_kriged_results' created.\n"
    f"Global survey CV: {krg.survey_cv:.3f}"
)
# ==================================================================================================
# CONVERT BIOMASS DENSITY TO NASC
logging.info(
    "Converting biomass density estimates into NASC\n"
    "     Stratifying by: 'geostratum_ks'/'stratum_ks'\n"
    "     Grouping by: 'sex'\n"
    "     Using stratum weights for all fish: True"
)

# CONVERT TO BIOMASS
df_kriged_results["biomass"] = df_kriged_results["biomass_density"] * df_kriged_results["area"]
logging.info("New column in 'df_kriged_results': 'biomass'")

# BIOMASS TO NASC
apportionment.mesh_biomass_to_nasc(
    mesh_data=df_kriged_results,
    biodata=dict_da_weight_proportion,
    group_columns=["sex", "stratum_ks"],
    mesh_biodata_link={"geostratum_ks": "stratum_ks"},
    stratum_weights=da_averaged_weight.sel(sex="all"),
    stratum_sigma_bs=invert_hake.sigma_bs_strata,  
)
logging.info(
    "Biomass density to NASC conversion complete\n"
    "    New columns in `df_kriged_results`\n"
    "        Biomass (kg): 'biomass'/'biomass_female'/'biomass_male'\n"
    "        Abundance (animals): 'abundance'/'abundance_female'/'abundance_male'\n"
    "        NASC (m^2 nmi^-2): 'nasc'"
)

# SUMMARIZE KRIGING RESULTS
logging.info(
    f"----------------------\n"
    f"Kriging-based results\n"
    f"     Total derived NASC: {df_kriged_results['nasc'].sum():.1f} m²nmi⁻²\n"
    f"     Total derived abundance: {df_kriged_results['abundance'].sum():.0f} fish\n"
    f"     Total biomass: {df_kriged_results['biomass'].sum() * 1e-6:.1f} kmt"
)
# ==================================================================================================
# DISTRIBUTE POPULATION ESTIMATES ACROSS AGE AND LENGTH BINS
logging.info(
    "Distribute kriged population estimates across age- and length-bins\n"
    "     Stratifying by: 'stratum_ks'\n"
    "     Grouping by: 'sex'"
)

# ABUNDANCE [ALL]
logging.info("Distributing abundances...")
dict_ds_kriged_abundance_table = apportionment.distribute_population_estimates(
    data=df_kriged_results,
    proportions = dict_ds_number_proportion,
    variable = "abundance",
    group_columns = ["sex", "age_bin", "length_bin", "stratum_ks"],
    data_proportions_link={"geostratum_ks": "stratum_ks"}
)
logging.info("Abundance distributions complete\n'dict_kriged_abundance_table' created.")

# SCALE UNAGED ABUNDANCE
logging.info(
    "Scaling unaged abundance...\n"     
    "     Reference: Aged abundances\n"
    "     Imputing missing bins: False"
)
dict_ds_kriged_abundance_table["standardized_unaged"] = apportionment.distribute_unaged_from_aged(
    population_table = dict_ds_kriged_abundance_table["unaged"],
    reference_table = dict_ds_kriged_abundance_table["aged"],
    stratum_dim = "stratum_ks",
    impute = False 
)

# BIOMASS [ALL]
logging.info("Distributing biomass...")
dict_ds_kriged_biomass_table = apportionment.distribute_population_estimates(
    data = df_kriged_results,
    proportions = dict_da_weight_proportion,
    variable = "biomass",
    group_columns = ["sex", "age_bin", "length_bin", "stratum_ks"],
    data_proportions_link={"geostratum_ks": "stratum_ks"}
)
logging.info("Biomass distribution complete\n'dict_kriged_biomass_table' created.")

# SCALE UNAGED BIOMASS
logging.info(
    "Scaling unaged biomass...\n"     
    "     Reference: Aged biomass\n"
    "     Imputing missing bins: True"
)
dict_ds_kriged_biomass_table["standardized_unaged"] = apportionment.distribute_unaged_from_aged(
    population_table = dict_ds_kriged_biomass_table["unaged"],
    reference_table = dict_ds_kriged_biomass_table["aged"],
    stratum_dim = "stratum_ks",
    impute=True,
    impute_variable=["age_bin"],
)

# CONSOLIDATE
# ---- ABUNDANCE
logging.info("Consolidating abundance tables...")
da_kriged_abundance_table = apportionment.sum_population_tables(
    population_tables={
        "aged": dict_ds_kriged_abundance_table["aged"],
        "unaged": dict_ds_kriged_abundance_table["standardized_unaged"]
    },
)
logging.info("Abundance table complete\n'df_kriged_abundance_table' created.")
# ---- Biomass
logging.info("Consolidating biomass tables...")
da_kriged_biomass_table = apportionment.sum_population_tables(
    population_tables={
        "aged": dict_ds_kriged_biomass_table["aged"],
        "unaged": dict_ds_kriged_biomass_table["standardized_unaged"]
    },
)
logging.info("Biomass table complete\n'df_kriged_biomass_table' created.")

# AGE-1 REALLOCATION ?
if REMOVE_AGE1:
    # REDISTRIBUTE AGE-1 ABUNDANCES
    logging.info("Redistributing kriged age-1 abundances and biomasses...")
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
    logging.info(
        "Kriged age-1 abundance and biomass estimates redistributed\n"
        "'da_kriged_abundance_table_proc' and 'da_kriged_biomass_table_proc' created."
    )
else:
    da_kriged_abundance_table_proc = da_kriged_abundance_table
    da_kriged_biomass_table_proc = da_kriged_biomass_table
# ==================================================================================================
# JOLLY AND HAMPTON (1990) ANALYSIS
# ==================================================================================================
logging.info("Beginning stratified analysis to estimate uncertainties (Jolly and Hampton, 1990)...")

# ANALYSIS PARAMETERS CONTAINER
JOLLYHAMPTON_PARAMETERS = {
    "transects_per_latitude": 5,
    "strata_transect_proportion": 0.75,
    "num_replicates": 1000,
}

# INITIALIZE JOLLYHAMPTON CLASS OBJECT
jh = stratified.JollyHampton(JOLLYHAMPTON_PARAMETERS)
logging.info("Stratified-analysis-class object 'jh' created...")

# RUN ON TRANSECT DATA
logging.info(
    "Running Jolly and Hampton (1990) algorithm for transect data\n"
    "     Variable: 'biomass'\n"
    "     Number of bootstrap replicates: 1000\n"
    "     Stratum transect sampling proportion: 0.75\n"
    "     Stratifying by: 'geostratum_ks'"
)
jh.stratified_bootstrap(data=df_nasc_proc, 
                        stratum_dim="geostratum_inpfc", 
                        variable="biomass")
logging.info(
    "Summarizing results....\n"
    "     Confindence interval percentile: 0.95\n"
    "     Confidence interval method: Jackknife studentized interval ('t-jackknife')"
)
df_jh_transect_results = jh.summarize(ci_percentile=0.95, ci_method="t-jackknife")
logging.info("Stratified transect analysis results complete\n'df_jh_transect_results' created.")

# REPORT
logging.info(
    f"Mean transect CV [95% CI]: "
    f"{df_jh_transect_results.loc[('survey', 'cv')]['mean']:.3f} "
    f"[{df_jh_transect_results.loc[('survey', 'cv')]['low']:.3f}, "
    f"{df_jh_transect_results.loc[('survey', 'cv')]['high']:.3f}]\n"
    f"    Resampling/bootstrapping bias: "
    f"{df_jh_transect_results.loc[('survey', 'cv')]['bias']:.3f}"
)

# RUN ON KRIGED DATA
# ---- Create virtual transects
logging.info(
    "Creating virtual transects for kriged mesh data\n"
    "     Stratifying by: 'geostratum_inpfc'\n"
    "     Virtual transects per latitude: 5"
)
kriged_transects = jh.create_virtual_transects(
    mesh_data=df_kriged_results,
    geostrata=df_dict_geostrata["inpfc"], 
    stratum_dim="geostratum_inpfc",
    variable="biomass",
)
# ---- Run rest of flow
logging.info(
    "Running Jolly and Hampton (1990) algorithm for kriged data\n"
    "     Variable: 'biomass'\n"
    "     Number of bootstrap replicates: 1000\n"
    "     Virtual transect sampling proportion: 0.75\n"
    "     Stratifying by: 'geostratum_ks'"
)
jh.stratified_bootstrap(data=kriged_transects, 
                        stratum_dim="geostratum_inpfc", 
                        variable="biomass")
logging.info(
    "Summarizing results....\n"
    "     Confindence interval percentile: 0.95\n"
    "     Confidence interval method: Jackknife studentized interval ('t-jackknife')"
)
df_jh_kriged_results = jh.summarize(ci_percentile=0.95, ci_method="t-jackknife")
logging.info("Stratified kriged analysis results complete\n'df_jh_kriged_results' created.")

# REPORT
logging.info(
    f"Mean kriging CV [95% CI]: "
    f"{df_jh_kriged_results.loc[('survey', 'cv')]['mean']:.3f} "
    f"[{df_jh_kriged_results.loc[('survey', 'cv')]['low']:.3f}, "
    f"{df_jh_kriged_results.loc[('survey', 'cv')]['high']:.3f}]\n"
    f"    Resampling/bootstrapping bias: "
    f"{df_jh_kriged_results.loc[('survey', 'cv')]['bias']:.3f}"
)
# ==================================================================================================
# REPORT GENERATION
# ==================================================================================================
logging.info(
    f"Writing reports to: '{REPORTS_DIR.as_posix()}'."
)
reporter = Reporter(REPORTS_DIR, verbose=VERBOSE)

# AGED-LENGTH HAUL
reporter.aged_length_haul_counts_report(
    filename="aged_length_haul_counts.xlsx",
    sheetnames={"male": "Sheet1", "female": "Sheet2", "all": "Sheet3"},
    bio_data=dict_df_bio["specimen"].dropna(subset=["age", "length", "weight"])
)

# TOTAL LENGTH HAUL COUNTS
reporter.total_length_haul_counts_report(
    filename="total_length_haul_counts.xlsx",
    sheetnames={"male": "Sheet1", "female": "Sheet2", "all": "Sheet3"},
    bio_data=dict_df_bio
)

# KRIGED AGED BIOMASS

# All values
reporter.kriged_aged_biomass_mesh_report(
    filename="kriged_aged_biomass_mesh_full.xlsx",
    sheetnames={"all": "Sheet1", "male": "Sheet2", "female": "Sheet3"},
    kriged_data=df_kriged_results,
    weight_data=ds_da_weight_dist["aged"],
    kriged_stratum_link={"geostratum_ks": "stratum_ks"},
)

# Nonzero values
reporter.kriged_aged_biomass_mesh_report(
    filename="kriged_aged_biomass_mesh_nonzero.xlsx",
    sheetnames={"all": "Sheet1", "male": "Sheet2", "female": "Sheet3"},
    kriged_data=df_kriged_results[df_kriged_results["biomass"] > 0.],
    weight_data=ds_da_weight_dist["aged"],
    kriged_stratum_link={"geostratum_ks": "stratum_ks"},
)

# KRIGERD MESH RESULTS

# All values
reporter.kriged_mesh_results_report(
    filename="kriged_biomass_mesh_full.xlsx",
    sheetname="Sheet1",
    kriged_data=df_kriged_results,
    kriged_stratum="geostratum_ks",
    kriged_variable="biomass",
    sigma_bs_data=invert_hake.sigma_bs_strata,
    sigma_bs_stratum="stratum_ks",
)

# Nonzero values
reporter.kriged_mesh_results_report(
    filename="kriged_biomass_mesh_nonzero.xlsx",
    sheetname="Sheet1",
    kriged_data=df_kriged_results[df_kriged_results["abundance"] > 0.],
    kriged_stratum="geostratum_ks",
    kriged_variable="biomass",
    sigma_bs_data=invert_hake.sigma_bs_strata,
    sigma_bs_stratum="stratum_ks",
)

# KRIGED LENGTH-AGE ABUNDANCES
reporter.kriged_length_age_abundance_report(
    filename="kriged_length_age_abundance_report.xlsx",
    sheetnames={"male": "Sheet1", "female": "Sheet2", "all": "Sheet3"},
    datatables=dict_ds_kriged_abundance_table,
)

# KRIGED LENGTH-AGE BIOMASS
reporter.kriged_length_age_biomass_report(
    filename="kriged_length_age_biomass_report.xlsx",
    sheetnames={"male": "Sheet1", "female": "Sheet2", "all": "Sheet3"},
    datatable=da_kriged_biomass_table_proc,
)

# KRIGING INPUT
reporter.kriging_input_report(
    filename="kriging_input_report.xlsx",
    sheetname="Sheet1",
    transect_data=df_nasc_proc,
)

# TRANSECT LENGTH-AGE ABUNDANCES
reporter.transect_length_age_abundance_report(
    filename="transect_length_age_abundance_report.xlsx",
    sheetnames={"male": "Sheet1", "female": "Sheet2", "all": "Sheet3"},
    datatables=dict_ds_transect_abundance_table,
)

# TRANSECT LENGTH-AGE BIOMASS
reporter.transect_length_age_biomass_report(
    filename="transect_length_age_biomass_report.xlsx",
    sheetnames={"male": "Sheet1", "female": "Sheet2", "all": "Sheet3"},
    datatable=da_transect_biomass_table,
)

# TRANSECT AGED BIOMASS

# Full values
reporter.transect_aged_biomass_report(
    filename="transect_aged_biomass_report_full.xlsx",
    sheetnames={"all": "Sheet1", "male": "Sheet2", "female": "Sheet3"},
    transect_data=df_nasc_proc,
    weight_data=ds_da_weight_dist["aged"],
)

# Nonzero values
reporter.transect_aged_biomass_report(
    filename="transect_aged_biomass_report_nonzero.xlsx",
    sheetnames={"all": "Sheet1", "male": "Sheet2", "female": "Sheet3"},
    transect_data=df_nasc_proc[df_nasc_proc["biomass"] > 0.],
    weight_data=ds_da_weight_dist["aged"],
)

# TRANSECT RESULTS

# Full values
reporter.transect_population_results_report(
    filename="transect_population_results_full.xlsx",
    sheetname="Sheet1",
    transect_data=df_nasc_proc,
    weight_strata_data=da_averaged_weight,
    sigma_bs_stratum=invert_hake.sigma_bs_strata,
    stratum_name="stratum_ks",
)


# Nonzero values
reporter.transect_population_results_report(
    filename="transect_population_results_nonzero.xlsx",
    sheetname="Sheet1",
    transect_data=df_nasc_proc[df_nasc_proc["biomass"] > 0.],
    weight_strata_data=da_averaged_weight,
    sigma_bs_stratum=invert_hake.sigma_bs_strata,
    stratum_name="stratum_ks",
)


# ==================================================================================================
# [OPTIONAL] REPORT COMPARISONS WITH ECHOPRO
# ==================================================================================================
if COMPARE:
    # Dictionary map
    ECHOPRO_TO_ECHOPOP_FILE_MAP = {
        "aged_length_haul_counts": {
            "echopro": "aged_len_haul_counts_table.xlsx",
            "echopop": "aged_length_haul_counts.xlsx"
        },
        "total_length_haul_counts": {
            "echopro": "total_len_haul_counts_table.xlsx",
            "echopop": "total_length_haul_counts.xlsx"
        },
        "aged_kriged_mesh_biomass_full": {
            "echopro": "EchoPro_kriged_aged_output-2021_1.xlsx",
            "echopop": "kriged_aged_biomass_mesh_full.xlsx"
        },
        "aged_kriged_mesh_biomass_subset": {
            "echopro": "EchoPro_kriged_aged_output-2021_0.xlsx",
            "echopop": "kriged_aged_biomass_mesh_nonzero.xlsx"
        },
        "kriged_mesh_biomass_full": {
            "echopro": "EchoPro_kriged_output-28-Jan-2026_0.xlsx",
            "echopop": "kriged_biomass_mesh_full.xlsx"
        },
        "kriged_mesh_biomass_subset": {
            "echopro": "EchoPro_kriged_output-28-Jan-2026_1.xlsx",
            "echopop": "kriged_biomass_mesh_nonzero.xlsx"
        },
        "kriging_input": {
            "echopro": "kriging_input.xlsx",
            "echopop": "kriging_input_report.xlsx"
        },
        "kriged_length_age_abundance": {
            "echopro": "kriged_len_age_abundance_table.xlsx",
            "echopop": "kriged_length_age_abundance_report.xlsx"
        },
        "kriged_length_age_biomass": {
            "echopro": "kriged_len_age_biomass_table.xlsx",
            "echopop": "kriged_length_age_biomass_report.xlsx"
        },
        "aged_transect_biomass_full": {
            "echopro": "EchoPro_un-kriged_aged_output-2021_0.xlsx",
            "echopop": "transect_aged_biomass_report_full.xlsx"
        },
        "aged_transect_biomass_subset": {
            "echopro": "EchoPro_un-kriged_aged_output-2021_1.xlsx",
            "echopop": "transect_aged_biomass_report_nonzero.xlsx"
        },
        "transect_length_age_abundance": {
            "echopro": "un-kriged_len_age_abundance_table.xlsx",
            "echopop": "transect_length_age_abundance_report.xlsx"
        },
        "transect_length_age_biomass": {
            "echopro": "un-kriged_len_age_biomass_table.xlsx",
            "echopop": "transect_length_age_biomass_report.xlsx"
        },
        "transect_results_full": {
            "echopro": "EchoPro_un-kriged_output-28-Jan-2026_0.xlsx",
            "echopop": "transect_population_results_full.xlsx"
        },
        "transect_results_subset": {
            "echopro": "EchoPro_un-kriged_output-28-Jan-2026_1.xlsx",
            "echopop": "transect_population_results_nonzero.xlsx"
        }
    }
    
    # AGED LENGTH HAUL COUNTS
    echopro_aged_length_haul_counts = compare.read_pivot_table_report(
        ECHOPRO_REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["aged_length_haul_counts"]["echopro"]
    )
    echopop_aged_length_haul_counts = compare.read_pivot_table_report(
        REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["aged_length_haul_counts"]["echopop"]
    )
    compare.plot_haul_count_comparisons(
        echopro=echopro_aged_length_haul_counts, 
        echopop=echopop_aged_length_haul_counts,
        save_filepath=COMPARISONS_DIR / "aged_length_haul_counts.png",
        show_plot=SHOW_PLOT
    )
    
    # TOTAL LENGTH HAUL COUNTS
    echopro_total_length_haul_counts = compare.read_pivot_table_report(
        ECHOPRO_REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["total_length_haul_counts"]["echopro"]
    )
    echopop_total_length_haul_counts = compare.read_pivot_table_report(
        REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["total_length_haul_counts"]["echopop"]
    )
    compare.plot_haul_count_comparisons(
        echopro=echopro_total_length_haul_counts, 
        echopop=echopop_total_length_haul_counts,
        save_filepath=COMPARISONS_DIR / "total_length_haul_counts.png",
        show_plot=SHOW_PLOT
    )
    
    # KRIGED LENGTH-AGE ABUNDANCE
    echopro_kriged_abundance_table = compare.read_pivot_table_report(
        filepath=(
            ECHOPRO_REPORTS_DIR / 
            ECHOPRO_TO_ECHOPOP_FILE_MAP["kriged_length_age_abundance"]["echopro"]
        )
    )
    echopop_kriged_abundance_table = compare.read_pivot_table_report(
        filepath=REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["kriged_length_age_abundance"]["echopop"]
    )
    compare.plot_population_table_comparisons(
        echopro=echopro_kriged_abundance_table, 
        echopop=echopop_kriged_abundance_table, 
        save_filepath=COMPARISONS_DIR / "kriged_length_age_abundance.png",
        show_plot=SHOW_PLOT
    )
    
    # TRANSECT LENGTH-AGE ABUNDANCE
    echopro_transect_abundance_table = compare.read_pivot_table_report(
        filepath=(
            ECHOPRO_REPORTS_DIR / 
            ECHOPRO_TO_ECHOPOP_FILE_MAP["transect_length_age_abundance"]["echopro"]
        )
    )
    echopop_transect_abundance_table = compare.read_pivot_table_report(
        filepath=(
            REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["transect_length_age_abundance"]["echopop"]
        )
    )
    compare.plot_population_table_comparisons(
        echopro=echopro_transect_abundance_table, 
        echopop=echopop_transect_abundance_table, 
        log_transform=True,
        save_filepath=COMPARISONS_DIR / "transect_length_age_abundance.png",
        show_plot=SHOW_PLOT
    )

    # KRIGED AGED BIOMASS
    echopro_kriged_biomass_table = compare.read_pivot_table_report(
        filepath=(
            ECHOPRO_REPORTS_DIR / 
            ECHOPRO_TO_ECHOPOP_FILE_MAP["kriged_length_age_biomass"]["echopro"]
        )
    )
    echopop_kriged_biomass_table = compare.read_pivot_table_report(
        filepath=REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["kriged_length_age_biomass"]["echopop"]
    )
    compare.plot_population_table_comparisons(
        echopro=echopro_kriged_biomass_table, 
        echopop=echopop_kriged_biomass_table, 
        save_filepath=COMPARISONS_DIR / "kriged_length_age_biomass.png",
        show_plot=SHOW_PLOT
    )

    # TRANSECT AGED BIOMASS
    echopro_transect_biomass_table = compare.read_pivot_table_report(
        filepath=(
            ECHOPRO_REPORTS_DIR / 
            ECHOPRO_TO_ECHOPOP_FILE_MAP["transect_length_age_biomass"]["echopro"]
        )
    )
    echopop_transect_biomass_table = compare.read_pivot_table_report(
        filepath=REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["transect_length_age_biomass"]["echopop"]
    )
    compare.plot_population_table_comparisons(
        echopro=echopro_transect_biomass_table, 
        echopop=echopop_transect_biomass_table, 
        save_filepath=COMPARISONS_DIR / "transect_length_age_biomass.png",
        show_plot=SHOW_PLOT
    )

    # KRIGING INPUTS
    echopro_kriging_input = compare.read_geodata(
        filepath=ECHOPRO_REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["kriging_input"]["echopro"]
    )
    echopop_kriging_input = compare.read_geodata(
        filepath=REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["kriging_input"]["echopop"]
    ) 

    compare.plot_geodata(
        echopro=echopro_kriging_input,
        echopop=echopop_kriging_input,
        save_filepath=COMPARISONS_DIR / "kriging_input.png",
        show_plot=SHOW_PLOT
    )

    # TRANSECT POPULATION ESTIMATES
    echopro_transect_estimates = compare.read_geodata(
        filepath=(
            ECHOPRO_REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["transect_results_full"]["echopro"]
        ),
    )

    echopop_transect_estimates = compare.read_geodata(
        filepath=REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["transect_results_full"]["echopop"],
    )

    compare.plot_geodata(
        echopro=echopro_transect_estimates,
        echopop=echopop_transect_estimates,
        save_filepath={
            ("abundance", "abundance_male", "abundance_female"): 
                COMPARISONS_DIR / "transect_abundances.png",
            ("number_density", "number_density_male", "number_density_female"):
                COMPARISONS_DIR / "transect_number_densities.png",
            ("biomass", "biomass_male", "biomass_female"):
                COMPARISONS_DIR / "transect_biomasses.png",
            ("biomass_density", "biomass_density_male", "biomass_density_female"):
                COMPARISONS_DIR / "transect_biomass_densities.png",
            "nasc": COMPARISONS_DIR / "transect_nasc.png",
        },
        show_plot=SHOW_PLOT
    )

    # KRIGING POPULATION ESTIMATES
    echopro_kriged_estimates = compare.read_geodata(
        filepath=(
            ECHOPRO_REPORTS_DIR / 
            ECHOPRO_TO_ECHOPOP_FILE_MAP["kriged_mesh_biomass_full"]["echopro"]
        ),
    )

    echopop_kriged_estimates = compare.read_geodata(
        filepath=REPORTS_DIR / ECHOPRO_TO_ECHOPOP_FILE_MAP["kriged_mesh_biomass_full"]["echopop"],
    )
    
    logging.info(
        f"Kriged population estimate differences [total]: \n"
        f"  BIOMASS\n"
        f"  -------\n"
        f"  EchoPro: {round(echopro_kriged_estimates["biomass"].sum() * 1e-6)} kmt\n"
        f"  Echopop: {round(echopop_kriged_estimates["biomass"].sum() * 1e-6)} kmt\n"
        f"  ABUNDANCE\n"
        f"  ---------\n"
        f"  EchoPro: {round(echopro_kriged_estimates["abundance"].sum())} fish\n"
        f"  Echopop: {round(echopop_kriged_estimates["abundance"].sum())} fish"
    )

    compare.plot_geodata(
        echopro=echopro_kriged_estimates,
        echopop=echopop_kriged_estimates,
        save_filepath={
            ("abundance", "abundance_male", "abundance_female"): 
                COMPARISONS_DIR / "kriged_abundances.png",
            ("biomass", "biomass_male", "biomass_female"):
                COMPARISONS_DIR / "kriged_biomasses.png",
            "nasc": COMPARISONS_DIR / "kriged_nasc.png",
        },
        show_plot=SHOW_PLOT
    )



