"""
General-purpose utility functions used across the echopop package.

This module provides shared helpers for data filtering, binning, interpolation, pivot-table
construction, grouped-series creation, and haul UID construction. Functions here are intentionally
domain-agnostic and are imported freely by ``ingest``, ``survey``, ``geostatistics``, and
``workflows`` sub-packages.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import interpolate as interp

from ..core.validators import BaseDictionary
from ..validators import ValidateHaulUID


def binify(
    data: pd.DataFrame | dict[str, pd.DataFrame],
    bins: np.ndarray,
    bin_column: str,
) -> None:
    """Add an ordered interval column to each DataFrame in place.

    Parameters
    ----------
    data : pandas.DataFrame or dict of pandas.DataFrame
        Frames to modify. Frames without ``bin_column`` and non-frame dictionary
        entries are skipped.
    bins : array-like
        At least two finite, strictly increasing bin locations. Internal edges
        are midpoints between consecutive locations, as in MATLAB ``hist``.
        Uniformly spaced locations are the interval centers. For uneven spacing,
        the interval centers generally differ from the supplied locations.
    bin_column : str
        Column to discretize. Adds ``<bin_column>_bin`` as ordered categorical
        pandas intervals, preserving original columns and index.

    Returns
    -------
    None
        DataFrames are modified in place.

    Warns
    -----
    UserWarning
        If nonmissing input values fall outside the binning intervals. One warning
        is emitted per affected DataFrame, reporting the column, count, and range.

    Notes
    -----
    The first and last edges extend half the adjacent spacing beyond the first
    and last locations. Intervals include their right boundary, but exclude their
    left boundary. Missing and out-of-range values receive missing bin assignments;
    rows are not removed. Unlike MATLAB ``hist``, outer bins are finite.

    Examples
    --------
    >>> frame = pd.DataFrame({"length": [20, 30, 50, 100]})
    >>> binify(frame, np.array([20, 30, 50]), "length")
    >>> frame["length_bin"].cat.categories
    IntervalIndex([(15.0, 25.0], (25.0, 40.0], (40.0, 60.0]], dtype='interval[float64, right]')
    """
    if isinstance(data, pd.DataFrame):
        frames = (data,)
    elif isinstance(data, dict):
        frames = (frame for frame in data.values() if isinstance(frame, pd.DataFrame))
    else:
        raise TypeError(f"data must be DataFrame or dict of DataFrames, got {type(data)}")

    bins = np.asarray(bins, dtype=float)
    if (
        bins.ndim != 1
        or bins.size < 2
        or not np.isfinite(bins).all()
        or not (bins[1:] > bins[:-1]).all()
    ):
        raise ValueError(
            "bins must be one-dimensional, finite, and strictly increasing, "
            "with at least two values"
        )

    half_spacing = np.diff(bins) / 2.0
    edges = np.concatenate([
        [bins[0] - half_spacing[0]],
        bins[:-1] + half_spacing,
        [bins[-1] + half_spacing[-1]],
    ])
    intervals = pd.IntervalIndex.from_breaks(edges, closed="right")
    for frame in frames:
        if bin_column in frame.columns:
            binned = pd.cut(frame[bin_column], bins=intervals)
            out_of_range_count = int((frame[bin_column].notna() & binned.isna()).sum())
            if out_of_range_count:
                warnings.warn(
                    f"{out_of_range_count} nonmissing value(s) in '{bin_column}' fall outside "
                    f"the binning range ({edges[0]}, {edges[-1]}]; "
                    "their bin assignments will be missing. Rows are retained.",
                    UserWarning,
                    stacklevel=2,
                )
            frame[f"{bin_column}_bin"] = binned


def _filter_rows(
    df: pd.Series | pd.DataFrame,
    filter_dict: dict[str, Any],
    include: bool,
    replace_value: np.number | str | None = None,
) -> pd.DataFrame:
    """Filter DataFrame rows based on index or column values."""
    # Get index DataFrame
    index_df = df.index.to_frame(index=False)

    # Get native column
    df_reset = df.reset_index()

    # Filter only for keys that exist in columns or index
    valid_filters = {
        k: v for k, v in filter_dict.items() if k in df_reset.columns or k in index_df.columns
    }

    if not valid_filters:
        return df

    # Create mask
    mask = np.logical_and.reduce(
        [
            (
                df_reset[col].isin(np.atleast_1d(vals))
                if col in df_reset.columns
                else index_df[col]
                .apply(
                    lambda x, _vals=vals: any(
                        (v in x if isinstance(x, pd.Interval) else v == x)
                        for v in np.atleast_1d(_vals)
                    )
                )
                .values
            )
            for col, vals in valid_filters.items()
        ]
    )

    # Apply inclusion/exclusion logic
    if not include and replace_value is None:
        mask = ~mask
    elif replace_value is not None:
        # ---- Replace values
        if isinstance(df, pd.Series):
            df_reset.loc[mask, 0] = replace_value
        else:
            df_reset.loc[mask, df.columns] = replace_value
        # ---- Set mask to include all values
        mask[:] = True

    # Check for matching names
    index_cols = df.index.names
    # ---- Apply mask
    if all(name is None for name in index_cols):
        return df_reset[mask].filter(df.columns)
    else:
        if isinstance(df, pd.Series):
            return df_reset[mask].set_index(index_cols).iloc[:, 0]
        else:
            return df_reset[mask].set_index(index_cols).filter(df.columns)


def _filter_columns(
    df: pd.DataFrame,
    filter_dict: dict[str, Any],
    include: bool,
    replace_value: str | None = None,
) -> pd.DataFrame:
    """Filter DataFrame columns based on column index values."""
    # Get column DataFrame
    col_index_df = df.columns.to_frame(index=False)

    # Filter only for keys that exist in column names/levels
    valid_filters = {k: v for k, v in filter_dict.items() if k in col_index_df.columns}

    if not valid_filters:
        return df

    # Create column mask
    col_mask = np.logical_and.reduce(
        [col_index_df[col].isin(np.atleast_1d(vals)) for col, vals in valid_filters.items()]
    )

    # Replace values if specified
    if replace_value is not None:
        df.loc[:, col_mask] = replace_value
        # ---- Return the modified DataFrame
        return df

    # Apply inclusion/exclusion logic
    if not include:
        col_mask = ~col_mask

    # Apply column filter
    return df.loc[:, col_mask]


def apply_filters(
    data: pd.Series | pd.DataFrame,
    include_filter: dict[str, Any] | None = None,
    exclude_filter: dict[str, Any] | None = None,
    replace_value: np.number | None = None,
) -> pd.DataFrame:
    """
    Apply inclusion and exclusion filters to a DataFrame.

    Filters rows and columns based on index/column values, supporting both inclusion and exclusion
    criteria with single values or lists of values. Handles interval-based filtering for categorical
    interval indices.

    Parameters
    ----------
    data : |pd.DataFrame| pr |pd.Series|
        Input DataFrame or Series to filter
    include_filter : Dict[str, Any], optional
        Dictionary of column/index:value(s) pairs. Rows/columns will be kept if they match. If
        value is a list, rows/columns matching any value in the list will be kept.
    exclude_filter : Dict[str, Any], optional
        Dictionary of column/index:value(s) pairs. Rows/columns will be excluded if they match. If
        value is a list, rows/columns matching any value in the list will be excluded.
    replace_value : |np.number|, optional
        If provided, replaces values in excluded columns with this value instead of dropping them.

    Returns
    -------
    |pd.DataFrame|
        Filtered DataFrame

    Examples
    --------
    >>> # Row filtering: Keep only females and males from sex column
    >>> apply_filters(df, include_filter={"sex": ["female", "male"]})
    >>> # Row filtering: Exclude unsexed specimens and small fish
    >>> apply_filters(df, exclude_filter={"sex": "unsexed", "length": 10})
    >>> # Index filtering: Keep rows where length_bin contains values 1-5
    >>> apply_filters(df, include_filter={"length_bin": [1, 2, 3, 4, 5]})
    >>> # Column filtering: Keep only female and male columns (wide format)
    >>> apply_filters(df, include_filter={"sex": ["female", "male"]})
    >>> # Multi-index filtering: Filter by age_bin, length_bin, and sex simultaneously
    >>> apply_filters(df, include_filter={"age_bin": [1], "length_bin": [1, 2, 3], "sex": ["male"]})
    >>> # Combined filtering: Include certain length bins but exclude unsexed
    >>> apply_filters(df, include_filter={"length_bin": [1, 2, 3]},\
        exclude_filter={"sex": "unsexed"})
    """
    # Create copy
    result = data.copy()

    # Inclusion filter
    if include_filter:
        if not isinstance(result, pd.Series):
            result = _filter_columns(result, include_filter, True, replace_value)
        result = _filter_rows(result, include_filter, True, replace_value)

    # Exclusion filter
    if exclude_filter:
        if not isinstance(result, pd.Series):
            result = _filter_columns(result, exclude_filter, False, replace_value)
        result = _filter_rows(result, exclude_filter, False, replace_value)

    # Return masked DataFrame
    return result


def is_df_wide(df):
    """Check if DataFrame is in wide format using pandas structural information."""
    # Wide format indicators:
    # 1. Columns have a name (indicating they represent categories)
    # 2. Columns are MultiIndex
    # 3. Index contains the ID variables

    return (
        isinstance(df.columns, pd.MultiIndex)
        or df.columns.name is not None
        or (df.index.name is not None or isinstance(df.index, pd.MultiIndex))
    )


def group_interpolator_creator(
    grouped_data: pd.DataFrame,
    independent_var: str,
    dependent_var: str,
    contrast_vars: str | list[str] | None = None,
) -> dict:
    """
    Create interpolator functions grouped by one or more contrast variables.

    Generates scipy interpolation functions for length-weight relationships, either as a single
    global interpolator or grouped by specified contrast variables (e.g., sex, age class).

    Parameters
    ----------
    grouped_data : pd.DataFrame
        Data containing independent and dependent variables
    independent_var : str
        Column name of the independent variable (e.g., 'length')
    dependent_var : str
        Column name of the dependent variable (e.g., 'weight_fitted')
    contrast_vars : str, List[str], or None, optional
        Column name(s) to group by (e.g., 'sex' or ['sex', 'age_bin']).
        If None or empty list, a single global interpolator is created.

    Returns
    -------
    Dict
        Dictionary of interpolator functions. When contrast_vars is provided, keys are contrast
        variable values. When contrast_vars is None or empty, contains a single entry with key
        '_global_'.

    Notes
    -----
    The interpolation is linear and extrapolates using the endpoints when values outside the range
    are requested. Requires at least 2 points per group for valid interpolation.
    """
    # Check if we have contrast variables to group by
    if contrast_vars is None or (isinstance(contrast_vars, list) and len(contrast_vars) == 0):
        # Create a single global interpolator
        if len(grouped_data) < 2:
            # Not enough points for interpolation
            return {"_global_": None}

        # Sort the data
        sorted_data = grouped_data.sort_values(by=independent_var)

        # Create the interpolator
        global_interp = interp.interp1d(
            sorted_data[independent_var],
            sorted_data[dependent_var],
            kind="linear",
            bounds_error=False,
            fill_value=(sorted_data[dependent_var].iloc[0], sorted_data[dependent_var].iloc[-1]),
        )

        # Return a dictionary with a special key for the global interpolator
        return {"_global_": global_interp}

    # Convert single string to list for consistent handling
    if isinstance(contrast_vars, str):
        contrast_vars = [contrast_vars]

    # Interpolator generation helper function
    def interpolator_factory(sub_group):
        if len(sub_group) < 2:
            # Need at least 2 points for interpolation
            return None

        # Sort the grouped values
        sub_group_sort = sub_group.sort_values(by=independent_var)

        # Return the interpolation object for the specific sub-group
        return interp.interp1d(
            sub_group_sort[independent_var],
            sub_group_sort[dependent_var],
            kind="linear",
            bounds_error=False,
            fill_value=(
                sub_group_sort[dependent_var].iloc[0],
                sub_group_sort[dependent_var].iloc[-1],
            ),
        )

    # Produce a dictionary of interpolator functions
    interpolators = (
        grouped_data.groupby(contrast_vars)
        .apply(interpolator_factory, include_groups=False)
        .to_dict()
    )

    return interpolators


def create_grouped_series(
    proportions_dict: dict[str, pd.DataFrame], group_cols: list[str], value_col: str
) -> pd.DataFrame:
    """
    Create and combine grouped series from a dictionary of DataFrames.

    Parameters
    ----------
    proportions_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with proportion data, keyed by group name
    group_cols : List[str]
        Columns to group by (e.g., ["stratum_num", "sex"])
    value_col : str
        Column name containing values to be aggregated (e.g., "proportion")

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with aggregated values and original dictionary keys as 'group' column

    Examples
    --------
    >>> props_dict = {'aged': aged_df, 'unaged': unaged_df}
    >>> grouped = create_grouped_series(
    ...    props_dict,
    ...    ["stratum_num", "sex"],
    ...    "proportion"
    ... )
    """
    series = [
        df.groupby(group_cols, observed=False)[value_col].sum().reset_index().assign(group=key)
        for key, df in proportions_dict.items()
    ]
    return pd.concat(series, axis=0)


def create_pivot_table(
    df: pd.DataFrame,
    index_cols: list[str],
    strat_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    """
    Create a pivot table with standard settings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to pivot
    index_cols : List[str]
        Column names to use as pivot table index
    strat_cols : List[str]
        Column names to use as pivot table columns
    value_col : str
        Column name containing values to aggregate

    Returns
    -------
    pd.DataFrame
        Pivot table with values aggregated by sum, converted to float, and NaN filled with 0

    Examples
    --------
    >>> pivot = create_pivot_table(
    ...     df=grouped_data,
    ...     index_cols=["group", "sex"],
    ...     strat_cols=["stratum_num"],
    ...     value_col="proportion"
    ... )
    """
    return (
        df.pivot_table(
            index=index_cols, columns=strat_cols, values=value_col, aggfunc="sum", observed=False
        )
        .astype(float)
        .fillna(0.0)
    )


def create_grouped_table(
    proportions_dict: dict[str, pd.DataFrame],
    group_cols: list[str],
    index_cols: list[str],
    strat_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    """
    Create grouped series and then convert to pivot table.

    Parameters
    ----------
    proportions_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with proportion data, keyed by group name
    group_cols : List[str]
        Columns to group by (e.g., ["stratum_num", "sex"])
    index_cols : List[str]
        Column names to use as pivot table index
    strat_cols : List[str]
        Column names to use as pivot table columns
    value_col : str
        Column name containing values to aggregate

    Returns
    -------
    pd.DataFrame
        Pivot table with aggregated values by group

    Examples
    --------
    >>> props_dict = {'aged': aged_df, 'unaged': unaged_df}
    >>> grouped_table = create_grouped_table(
    ...     props_dict,
    ...     group_cols=["stratum_num", "sex"],
    ...     index_cols=["group"],
    ...     strat_cols=["stratum_num"],
    ...     value_col="proportion"
    ... )
    """
    # First create grouped series
    grouped_df = create_grouped_series(proportions_dict, group_cols, value_col)

    # Then convert to pivot table
    return create_pivot_table(grouped_df, index_cols, strat_cols, value_col)


def is_pivot_table(df: pd.DataFrame):
    """
    Determine whether a ``pandas.DataFrame`` comprises a MultiIndex along either axis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing some type of dataset.
    """
    # Check for a MultiIndex
    is_multiindex = isinstance(df.columns, pd.MultiIndex) or isinstance(df.index, pd.MultiIndex)

    # Check for number of levels
    is_multilevel = df.columns.nlevels > 1 or df.index.nlevels > 1

    # Check for index naming
    is_named_index = None not in list(df.index.names)

    # Check for index type
    is_index_typed = not isinstance(df.index, pd.RangeIndex)

    # Return boolean checksum
    if (is_multiindex + is_multilevel + is_named_index + is_index_typed) > 0:
        return True
    else:
        return False


def round_half_up(n: pd.Series | pd.DataFrame):
    r"""
    Round values to the nearest integer using the "round half up" rule.

    This function rounds each value in the input Series or array to the nearest integer,
    with halfway cases (i.e., values exactly halfway between two integers) rounded away from zero.
    For positive numbers, values ending in .5 are rounded up. For negative numbers, values ending
    in .5 are rounded down (toward zero).

    Mathematically, for each element :math:`x` in the input:

    .. math::
        \text{round\_half\_up}(x) =
        \begin{cases}
            \lfloor x + 0.5 \rfloor & \text{if } x \geq 0 \\
            \lceil x - 0.5 \rceil & \text{if } x < 0
        \end{cases}

    Parameters
    ----------
    n : pandas.Series or pandas.DataFrame column
        Series or column of numeric values to round.

    Returns
    -------
    pandas.Series
        Series of rounded integer values.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> s = pd.Series([1.5, 2.3, -1.5, -2.7, 0.5, -0.5])
    >>> round_half_up(s)
    0    2.0
    1    2.0
    2   -1.0
    3   -3.0
    4    1.0
    5    0.0
    dtype: float64

    Notes
    -----
    This rounding method is different from NumPy's default :func:`numpy.round`, which uses "round
    half to even" (banker's rounding).
    """
    return n.apply(lambda x: np.floor(x + 0.5) if x >= 0 else np.ceil(x - 0.5))


def resolve_uid_component(
    data: pd.DataFrame,
    field_name: str,
    column_name: str,
    config: ValidateHaulUID,
    _is_can: pd.Series,
    _dataset_type: str | None = None,
):
    """
    Determine the source value for a specific UID component.

    Resolves the hierarchy between manual configuration overrides,
    DataFrame columns, and model defaults. For nested models (ship/survey),
    it maps values based on the regional _is_can mask.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    field_name : str
        The attribute name in the Pydantic config.
    column_name : str
        The corresponding column name in the DataFrame.
    config : ValidateHaulUID
        The initialized configuration model.
    _is_can : pd.Series
        Boolean mask where True indicates a Canadian region record.
    single_country : bool
        Boolean flag dictating whether only a single survey is being included for a particular
        survey. This avoids issues where only a single region is provided, but the function may
        produce the opposite region due to the handling of the haul offset.
    _dataset_type : str, optional
        Reference label for warnings.

    Returns
    -------
    pd.Series or Any
        A series of resolved values (for nested/mixed sources) or a
        scalar default.
    """
    # Get the model from the configuration instance
    config_entry = getattr(config, field_name)

    # Handle nested model
    if isinstance(config_entry, BaseDictionary):
        # ---- Initialize column series
        output = pd.Series(index=data.index, dtype=object)
        # ---- Define explicit keys
        key_names = config_entry.model_fields_set
        # ---- Single-country handling
        if getattr(config, "single_country", False) and len(key_names) == 1:
            target_key = list(key_names)[0]
            # ---- Force _is_can to be True if CAN, all False if US
            _is_can = pd.Series(target_key == "CAN", index=data.index)
        # ---- Iterate through each model field
        for key in type(config_entry).model_fields.keys():
            # ---- Apply CAN mask where appropriate
            mask = _is_can if key == "CAN" else ~_is_can
            # ---- Skip if the mask is empty
            if not mask.any():
                continue
            # ---- Override and raise UserWarning alerting overwritten entries
            if key in key_names or column_name not in data.columns:
                if column_name in data.columns:
                    warnings.warn(
                        f"Override: Using input '{field_name}' instead of '{column_name} from "
                        f"'{_dataset_type}['{key}']' DataFrame input.",
                        UserWarning,
                        stacklevel=2,
                    )
                output.loc[mask] = getattr(config_entry, key)
            # ---- Use supplied column values from DataFrame
            else:
                output.loc[mask] = data.loc[mask, column_name]
        return output

    # Handle scalar model
    if field_name in config.model_fields_set:
        # ---- Override and raise UserWarning alerting overwritten entries
        if column_name in data.columns:
            warnings.warn(
                f"Override: Using input '{field_name}' instead of '{column_name}' from "
                f"'{_dataset_type}' DataFrame input.",
                UserWarning,
                stacklevel=2,
            )
        return config_entry

    # Return the DataFrame column if it exists, otherwise the model default
    return data[column_name] if column_name in data.columns else config_entry


def construct_haul_uids(
    data: pd.DataFrame, _dataset_type: str | None = None, **params
) -> pd.Series:
    """
    Assembles the component parts of the Haul UID into a single string series.

    This function manages the conditional math for CAN vs. US hauls based on
    the provided haul_offset.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    _dataset_type : str, optional
        Reference label for warnings.
    **params : dict
        Keyword arguments used to initialize the ValidateHaulUID model.

    Returns
    -------
    pd.Series
        A series of formatted UID strings ('ship-survey-species-haul').
    """
    # Validate configuration
    config = ValidateHaulUID(**params)

    # Adjust the haul numbers by removing any defined offsets
    offset_haul = data["haul_num"] - config.haul_offset

    # Determine the country mask based on the haul number and the respective offset, if supplied
    is_can = offset_haul >= 0

    # Resolve the final haul numbers
    haul = np.where(is_can, offset_haul, data["haul_num"])

    # Resolve components independently
    ship = resolve_uid_component(data, "ship_id", "ship", config, is_can, _dataset_type)
    survey = resolve_uid_component(data, "survey_id", "survey", config, is_can, _dataset_type)
    species = resolve_uid_component(
        data, "species_id", "species_code", config, is_can, _dataset_type
    )

    # Return the constructed UIDs
    return (
        pd.Series(ship, index=data.index).astype(str)
        + "-"
        + pd.Series(survey, index=data.index).astype(str)
        + "-"
        + pd.Series(species, index=data.index).astype(str)
        + "-"
        + pd.Series(haul, index=data.index).astype(str)
    )


def add_haul_uids(data: pd.DataFrame, _dataset_type: str | None = None, **params) -> None:
    """
    Add a ``'uid'`` column to the input DataFrame via in-place mutation.

    The UID is constructed as '{ship}-{survey}-{species}-{haul}'. The logic
    resolves values based on the following priority:

    1. Explicit manual overrides provided in `**params`.

    2. Existing DataFrame columns ('ship', 'survey', 'species_code').

    3. Standard default values.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame. Must contain at least the 'haul_num' column.
    _dataset_type : str, optional
        A label used to identify the dataset in warning messages (e.g., 'NASC', 'biometry').
    **params : dict
        Optional keyword arguments to override defaults or DataFrame values:

        - ship_id (dict): Region-specific IDs, e.g., {'US': 10, 'CAN': 20}.

        - survey_id (dict): Region-specific IDs, e.g., {'US': 1, 'CAN': 2}.

        - species_id (int/str): A global species code override.

        - haul_offset (int/float): A value subtracted from 'haul_num' for
          records identified as 'CAN' (where haul_num - offset >= 0).

        - single_country :(bool): A boolean flag dictating whether only a single survey is being
          included for a particular survey. This avoids issues where only a single region is
          provided, but the function may produce the opposite region due to the handling of the
          haul offset.

    Raises
    ------
    KeyError
        If 'haul_num' is missing from the input DataFrame.
    """
    # Validate that the input DataFrame contains the column 'haul_num'
    if "haul_num" not in data.columns:
        raise KeyError("Required column 'haul_num' missing from input DataFrame 'data'.")

    # Construct the haul-based UIDs
    uids = construct_haul_uids(data, _dataset_type, **params)

    # Assign directly to the DataFrame via inplace mutation
    data["uid"] = uids

    # Replace None
    return None
