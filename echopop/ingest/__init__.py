"""
Data ingestion sub-package for echopop.

This sub-package handles loading and preprocessing of all external input files used in
acoustic-trawl survey analysis: biological data (specimen, length, catch), NASC exports from
Echoview, kriging mesh and isobath grids, haul and geographic stratification tables,
variogram/kriging parameter files, and Sv CSV exports. Each module is responsible for a single data
domain and returns clean, standardised ``pandas.DataFrame`` objects ready for downstream analysis.
"""

from . import nasc, sv
from .biological import (
    apply_composite_key,
    generate_composite_key,
    load_biodata_db_views,
    load_biodata_views,
    load_biological_data,
)
from .mesh import load_isobath_data, load_mesh_data
from .params import load_kriging_variogram_params
from .strata import (
    join_geostrata_by_latitude,
    join_strata_by_haul,
    join_strata_by_uid,
    load_geostrata,
    load_strata,
)

__all__ = [
    "apply_composite_key",
    "generate_composite_key",
    "load_biological_data",
    "load_isobath_data",
    "load_biodata_views",
    "load_biodata_db_views",
    "load_mesh_data",
    "load_kriging_variogram_params",
    "join_geostrata_by_latitude",
    "join_strata_by_haul",
    "join_strata_by_uid",
    "load_strata",
    "load_geostrata",
    # Submodules
    "biological",
    "mesh",
    "params",
    "nasc",
    "strata",
    "sv",
]
