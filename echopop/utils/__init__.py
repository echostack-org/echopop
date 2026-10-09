"""
Utility functions for Echopop.

Utility functions that are either generic (e.g., keyword filtering) or specific to FEAT methods and
survey-specific parameterization.
"""

from . import feat_parameters
from .base import (
    binify,
    apply_filters,
    create_grouped_table,
    create_pivot_table,
    group_interpolator_creator,
    round_half_up,
)
from .feat_functions import (
    convert_afsc_nasc_to_feat,
    filter_transect_intervals,
    get_survey_western_extents,
    transect_ends_crop,
    western_boundary_search_strategy,
)

__all__ = [
    # Generic utilities
    "apply_filters",
    "binify",
    "create_grouped_table",
    "create_pivot_table",
    "group_interpolator_creator",
    "round_half_up",
    # FEAT-specific functions
    "convert_afsc_nasc_to_feat",
    "filter_transect_intervals",
    "get_survey_western_extents",
    "transect_ends_crop",
    "western_boundary_search_strategy",
    # Submodules
    "base",
    "feat_functions",
    "feat_parameters",
]
