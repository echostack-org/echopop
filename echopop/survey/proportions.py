"""Functions for computing number, weight, and NASC proportions from survey data."""

import copy
import functools
import operator
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from .. import utils


def compute_binned_counts(
    data: pd.DataFrame,
    groupby_cols: list[str],
    count_col: str,
    agg_func: str = "size",
) -> xr.DataArray:
    """
    Compute binned counts with grouping.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    groupby_cols : list
        Columns to group by
    count_col : str
        Column to aggregate
    agg_func : str, default "size"
        Aggregation function to apply: "size", "sum", "count", etc.
        Column-value pairs to exclude. Format: {column: value_to_exclude}

    Returns
    -------
    xr.DataArray
        Grouped counts with coordinates based on the column names provided in 'groupby_cols'.
    """
    # Copy the dataset
    df = data.copy()

    # Apply aggregation
    aggregate = df.groupby(groupby_cols, observed=False)[count_col].agg(agg_func).to_frame("count")

    # Convert to xarray.DataArray
    return aggregate.to_xarray()["count"]


def number_proportions(
    data: xr.DataArray | xr.Dataset,
    stratum_dim: str | None = None,
    exclude_filters: dict | None = None,
) -> xr.DataArray | xr.Dataset:
    """
    Calculate number proportions from one or more xarray DataArrays or Datasets.

    This function computes proportions across all provided xarray objects. It can handle different
    xarray structures, as long as they all have a ``'count'`` variable and share the grouping
    columns.

    Parameters
    ----------
    data : Union[xr.DataArray, xr.Dataset]
        Either a single xarray.DataArray or an xarray.Dataset containing multiple named
        xarray.DataArray objects.
    stratum_dim : str, optional
        Stratification dimension name to group by for calculating totals (e.g., ``"stratum_ks"``).
    exclude_filters : dict, default {}
        Filters to exclude coordinates from xarray objects. For a single DataArray, this should be
        a dict like {"sex": "unsexed"}; for a Dataset, a dict of dicts keyed by variable name.

    Returns
    -------
    Union[xr.DataArray, Dict[xr.Dataset]]
        If only one DataArray is provided, returns that xarray.DataArray with added proportion
        coordinates. If a Dataset is provided, returns a dictionary of xarray.Datasets for each
        variable within ``data``.

    Notes
    -----
    The function calculates a single proportion variable:

    - "proportion": Overall proportion (count divided by the sum of counts across all DataArrays
      within the original Dataset)

    Examples
    --------
    >>> # Single DataArray
    >>> result = number_proportions(aged_counts)
    >>>
    >>> # Dataset
    >>> dataset_input = Dataset({"aged": aged_counts, "unaged": unaged_counts})
    >>> result = number_proportions(dataset_input)
    """
    # Apply mutable default guards
    group_columns = [stratum_dim] if stratum_dim is not None else []
    exclude_filters = exclude_filters or {}

    # Handle xr.DataArray
    if isinstance(data, xr.DataArray):
        data = xr.Dataset({"data": data})
        # ---- Ensure exclude_filters matches the key
        exclude_filters = {"data": exclude_filters}
    elif isinstance(data, xr.Dataset):
        # ---- Check that all keys in exclude_filters exist in the dataset
        missing_keys = [k for k in exclude_filters if k not in data]
        if missing_keys:
            missing_str = "', '".join(missing_keys)
            raise KeyError(f"Variables in 'exclude_filters' not found in 'data': '{missing_str}'.")

    # Apply exclude_filters to each variable, transform to DataFrame, filter, then back to DataArray
    filtered_vars = {}
    for varname, da in data.data_vars.items():
        filt = exclude_filters.get(varname, {})
        # ---- Subset filter with overlapping coordinates
        filt_pruned = {k: v for k, v in filt.items() if k in data.coords}
        if filt:
            da = da.drop_sel(filt_pruned, errors="ignore")
        # ---- Convert back to DataArray
        if da.size == 0:
            # ---- If all filtered out, create empty DataArray with same dims
            filtered_vars[varname] = da.isel({d: slice(0) for d in da.dims})
        else:
            filtered_vars[varname] = da

    # Compute group totals and overall total for each variable
    group_totals = {}
    for varname, da in filtered_vars.items():
        if not group_columns:
            group_total = da.copy()
        else:
            sum_dims = [d for d in da.dims if d not in group_columns]
            if sum_dims:
                group_total = da.sum(dim=sum_dims, skipna=True)
            elif group_columns:
                group_total = da.groupby(group_columns).sum()
            else:
                group_total = da.sum(skipna=True)
        group_totals[varname] = group_total

    # Calculate grand totals
    overall_total = xr.concat(list(group_totals.values()), dim="source").sum(dim="source")

    # Compute proportions for each variable
    output = {}
    for varname, da in filtered_vars.items():
        # Store only overall proportions; within-group values can be derived downstream by
        # re-normalizing within the requested groups.
        proportion = da / overall_total
        # ---- Attach as new DataArrays with appropriate names
        output[varname] = xr.Dataset(
            {
                "count": da,
                "proportion": proportion,
            }
        )

    # Return either the dictionary (if multiple Datasets)
    # ---- or just the single Dataset (if only one)
    return list(output.values())[0] if len(output) == 1 else output


def apply_weight_interpolation(
    target_df: pd.DataFrame,
    interpolators: dict,
    dependent_var: str,
    independent_var: str,
    count_col: str,
    contrast_vars: str | list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply weight interpolation to a DataFrame based on interpolators.

    Uses the provided interpolators to estimate weights for each row in the target DataFrame,
    optionally multiplying by a count column (e.g., for expanding interpolated weights to a length
    frequency count).

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame to apply interpolation to
    interpolators : Dict
        Dictionary of interpolator functions from group_interpolator_creator
    dependent_var : str
        Column name of the dependent variable to be created (e.g., 'weight')
    independent_var : str
        Column name of the independent variable (e.g., 'length')
    count_col : str
        Column containing count values to multiply interpolated values by
    contrast_vars : str, List[str], or None, default None
        Column(s) used as contrast variables in the interpolators. If None or empty list, uses the
        global interpolator.

    Returns
    -------
    pd.DataFrame
        DataFrame with added dependent_var column containing interpolated values (multiplied by
        count_col values if present)

    Notes
    -----
    When using contrast variables, rows with contrast values not found in the interpolators will
    receive NaN values for the dependent variable.
    """
    # Create copy to avoid modifying original
    result_df = target_df.copy()

    # Check if we're using a global interpolator
    if "_global_" in interpolators:
        # Apply global interpolation
        global_interp = interpolators["_global_"]
        if global_interp is not None:
            result_df[dependent_var] = result_df[independent_var].apply(
                lambda x: global_interp(x) if not pd.isna(x) else np.nan
            )
        else:
            result_df[dependent_var] = np.nan
    else:
        # Convert single string to list for consistent handling
        if isinstance(contrast_vars, str):
            contrast_vars = [contrast_vars]
        elif contrast_vars is None:
            contrast_vars = []

        # Define interpolation function using contrast variables
        def get_interpolated_value(row):
            # Create key tuple from contrast variables
            if len(contrast_vars) == 1:
                key = row[contrast_vars[0]]
            else:
                key = tuple(row[var] for var in contrast_vars)

            # Get independent variable value
            ind_val = row[independent_var]

            # Apply interpolator if available for this key
            if key in interpolators and interpolators[key] is not None and not pd.isna(ind_val):
                return interpolators[key](ind_val)

            return np.nan

        # Apply interpolation
        result_df[dependent_var] = result_df.apply(get_interpolated_value, axis=1)

    # Multiply by count if count_col exists and is in the dataframe
    if count_col in result_df.columns:
        result_df[dependent_var] *= result_df[count_col]

    return result_df


def binned_weights(
    length_data: pd.DataFrame,
    interpolate_regression: bool = True,
    group_columns: list | None = None,
    length_weight_data: xr.DataArray | None = None,
    include_filter: dict[str, Any] | None = None,
) -> xr.DataArray:
    """
    Process length-weight data and return an xarray.DataArray of weights binned by group_columns.

    This function creates a DataArray of weights by length bins and optionally other grouping
    variables. If interpolate_regression is True, weights are interpolated using a regression
    model provided as a DataArray. If False, weights are taken directly from the length_dataset.
    Filtering is applied using include_filter, which keeps only rows matching the filter.

    Parameters
    ----------
    length_dataset : pd.DataFrame
        Dataset with length measurements. Must contain 'length_bin' and 'length_count' columns when
        using interpolation, or direct 'weight' values when not interpolating.
    interpolate_regression : bool, default True
        Whether to use interpolation for weights. If True, interpolates weights based on
        length-weight relationships from length_weight_dataset. If False, uses existing weight
        values in length_dataset.
    group_columns : list, default ["length_bin"]
        Columns to use as coordinates in the output DataArray.
    length_weight_dataset : xr.DataArray, optional
        DataArray with length-weight relationships. Required when interpolate_regression=True. Must
        contain 'length_bin' as a coordinate.
    include_filter : Dict[str, Any], optional
        Filter to apply to both datasets (e.g., to include only certain sexes). Rows are kept if
        they match the filter.

    Returns
    -------
    xr.DataArray
        DataArray of weights binned by group_columns.

    Raises
    ------
    ValueError
        If interpolate_regression=True but length_weight_dataset is None or missing 'length_bin'.

    Notes
    -----
    - The output DataArray will have coordinates as specified by group_columns.

    - Filtering is performed before binning/interpolation.

    - If interpolate_regression is True, interpolation is performed using the regression model
      in length_weight_dataset, which is first converted to a DataFrame for compatibility.

    - If interpolate_regression is False, weights are taken directly from length_dataset.

    Examples
    --------
    >>> # Using interpolation with regression model
    >>> weights = binned_weights(
    ...     length_dataset=length_freq_df,
    ...     length_weight_dataset=length_weight_da,
    ...     interpolate_regression=True,
    ...     group_columns=["length_bin", "sex"],
    ...     include_filter={"sex": ["female", "male"]}
    ... )
    >>> # Using direct weights (no interpolation)
    >>> weights = binned_weights(
    ...     length_dataset=specimen_df,
    ...     interpolate_regression=False,
    ...     group_columns=["length_bin", "sex"],
    ...     include_filter={"sex": ["female", "male"]}
    ... )
    """
    # Apply mutable default guard
    group_columns = group_columns if group_columns is not None else ["length_bin"]

    # Apply include_filter if provided
    if include_filter:
        length_data = utils.apply_filters(length_data, include_filter=include_filter)

    # Check that all group_columns are present in length_dataset
    missing_cols = [col for col in group_columns if col not in length_data.columns]
    if missing_cols:
        missing_str = "', '".join(missing_cols)
        raise KeyError(f"Columns in 'group_columns' not found in 'length_data': '{missing_str}'.")

    # Ensure 'length_bin' is always included in group_columns
    if "length_bin" not in group_columns:
        group_columns = ["length_bin"] + list(group_columns)

    # Straightforward case without interpolation
    if not interpolate_regression:
        return (
            length_data.groupby(group_columns, observed=False)["weight"]
            .sum()
            .to_xarray()
            .fillna(0.0)
        )

    # Verify compatible arguments
    if length_weight_data is None:
        raise ValueError(
            "'length_weight_data' must be provided when 'interpolate_regression'=True."
        )

    # Validate coordinates
    if "length_bin" not in length_weight_data.coords:
        raise ValueError("'length_weight_data' must have 'length_bin' as a coordinate.")

    # Match the inclusion filter, if needed
    if include_filter:
        length_weight_data = length_weight_data.sel(include_filter)

    # Convert to a DataFrame for interpolation
    length_weight_data_cnv = length_weight_data.to_dataframe().reset_index()

    # Extract the defining lengths for each interval category of length_bin
    length_weight_data_cnv["length"] = (
        length_weight_data_cnv["length_bin"].map(lambda x: x.mid).astype(float)
    )

    # Define contrast variables
    contrast_vars = list(set(list(length_weight_data.coords.keys())) - set(["length_bin"]))

    # Create interpolation generators
    interpolators = utils.group_interpolator_creator(
        grouped_data=length_weight_data_cnv,
        independent_var="length",
        dependent_var="weight_fitted",
        contrast_vars=contrast_vars,
    )

    # Apply the interpolation
    result = apply_weight_interpolation(
        target_df=length_data.copy(),
        interpolators=interpolators,
        dependent_var="weight",
        independent_var="length",
        count_col="length_count" if "length_count" in length_data.columns else None,
        contrast_vars=contrast_vars,
    )

    # Return the grouped sums
    return result.groupby(group_columns, observed=False)["weight"].sum().to_xarray().fillna(0.0)


def calculate_adjusted_proportions(
    group_keys: list[str],
    aggregate_proportions: xr.DataArray,
    group_proportions: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate adjusted proportions across multiple groups.

    Parameters
    ----------
    group_keys : List[str]
        List of group keys/names (e.g., ["aged", "unaged"])
    aggregate_table : xr.DataArray
        Table with aggregate proportions by group
    group_proportions_table : xr.DataArray
        Table with grouping-specific proportions by group

    Returns
    -------
    xr.DataArray
        DataArray with adjusted proportions

    Notes
    -----
    This function calculates adjusted proportions for multiple datasets. The first group is used as
    a reference, and other groups are adjusted based on their relative proportions.

    Examples
    --------
    >>> group_keys = ["aged", "unaged"]
    >>> adjusted = calculate_adjusted_proportions(
    ...     group_keys,
    ...     aggregate_table,
    ...     sex_proportions_table
    ... )
    """
    # Initialize the dictionary for the group-specific DataArrays
    adjusted_props = {}
    # ---- Use the first group as the reference
    first_group = group_keys[0]

    # For all other groups, calculate relative to the first group
    for group in group_keys[1:]:
        adjusted_props[group] = aggregate_proportions.sel(group=group) / (
            aggregate_proportions.sel(group=group) + group_proportions.sel(group=first_group)
        )

    # Calculate first group proportion using all other adjusted proportions
    if len(group_keys) > 1:
        other_adjusted = sum(adjusted_props[g] for g in group_keys[1:])
        adjusted_props[first_group] = group_proportions.sel(group=first_group) / (
            group_proportions.sel(group=first_group) + other_adjusted
        )
    else:
        # ---- If only one group, it gets 100% of the proportions
        adjusted_props[first_group] = group_proportions.sel(group=first_group)

    # Combine into a single DataArray with 'group' dimension
    return xr.concat(
        [adjusted_props[group].expand_dims({"group": [group]}) for group in group_keys], dim="group"
    )


def calculate_grouped_weights(
    length_weight_data: xr.DataArray,
    within_group_proportions: xr.DataArray,
    within_group_proportions_all: xr.DataArray,
    aggregate_proportions: xr.DataArray,
    adjusted_proportions: xr.DataArray,
    group_keys: list[str],
) -> xr.DataArray:
    """
    Calculate average weights by sex category across groups.

    Parameters
    ----------
    length_weight_data : xr.DataArray
        DataArray with weight values distributed across length bins and optionally other coordinates
    within_group_proportions : xr.DataArray
        Within-group proportions distributed over length and optionally other coordinates
    within_group_proportions_all : pd.DataFrame
        Across-group/overall proportions distributed over length and optionally other coordinates
    aggregate_proportions : xr.DataArray
        Table with aggregate proportions by group
    adjusted_proportions : xr.DataArray
        Adjusted proportions for each group and groupings
    group_keys : List[str]
        List of group keys/names (e.g., ["aged", "unaged"])

    Returns
    -------
    xr.DataArray
        DataArray with average weights by a grouping category defined by groupings

    Notes
    -----
    The function calculates weighted average weights for different grouping categories:

    - For "all", it uses aggregate proportions across all groups

    - For individual groupings, it uses sex-specific adjusted proportions

    Examples
    --------
    >>> weights = calculate_grouped_weights(
    ...     binned_weights,
    ...     length_proportions_group,
    ...     length_proportions_all,
    ...     aggregate_proportions,
    ...     adjusted_proportions,
    ...     ["aged", "unaged"]
    ... )
    """
    # Initialize the dictionary to contain the grouped weights
    weight_das = {}

    # Identify non-length dimensions (grouping dimensions)
    non_length_dims = [d for d in length_weight_data.dims if d != "length_bin"]

    # Get grouping categories from the first non-length dimension
    if not non_length_dims:
        raise ValueError("'length_weight_data' must have at least one dimension not 'length_bin'.")
    elif len(non_length_dims) > 1:
        # ---- Current dimensions
        all_dims = ", ".join(f"'{c}'" for c in non_length_dims)
        raise ValueError(
            f"'length_weight_data' must only have one additional dimension besides 'length_bin'. "
            f"Got: {all_dims}."
        )

    # Weight for 'all' category combined
    grouping_dim = non_length_dims[0]
    grouping_vals = length_weight_data[grouping_dim].values
    if "all" in grouping_vals:
        weight_all_values = length_weight_data.sel({grouping_dim: "all"})
        weighted_sum = (
            (aggregate_proportions * within_group_proportions_all) * weight_all_values
        ).sum(dim=["length_bin", "group"])
        weight_das["all"] = weighted_sum

    # Weight for each grouping category individually
    if grouping_dim in adjusted_proportions.dims:
        for grouping in grouping_vals:
            # ---- Skip if "all" to avoid double-processing
            if grouping == "all":
                continue
            if grouping in adjusted_proportions.coords[grouping_dim].values:
                # ---- Use 'all' weights if available, otherwise use category-specific weights
                if "all" in grouping_vals:
                    weight_base = length_weight_data.sel({grouping_dim: "all"})
                else:
                    weight_base = length_weight_data.sel({grouping_dim: grouping})
                # ---- Weighted sums
                weighted_sum = sum(
                    (
                        within_group_proportions.sel(group=g, **{grouping_dim: grouping})
                        * weight_base
                    ).sum(dim="length_bin")
                    * adjusted_proportions.sel(group=g, **{grouping_dim: grouping})
                    for g in group_keys
                    if grouping in within_group_proportions.sel(group=g).coords[grouping_dim].values
                )
                weight_das[grouping] = weighted_sum

    # Combine into final DataArray
    if "all" in grouping_vals:
        ordered_cats = ["all"] + [c for c in grouping_vals if c != "all"]
    else:
        ordered_cats = list(grouping_vals)

    # Format and return
    result = xr.concat(
        [weight_das[cat] for cat in ordered_cats if cat in weight_das],
        dim=xr.DataArray(
            [cat for cat in ordered_cats if cat in weight_das],
            dims=[grouping_dim],
            name=grouping_dim,
        ),
    ).fillna(0.0)
    return result


def calculate_within_group_proportions(
    proportions_dict: dict[str, pd.DataFrame],
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Calculate within-group proportions for each DataFrame.

    Parameters
    ----------
    proportions_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with proportion data, keyed by group name
    group_cols : List[str]
        Columns to group by for calculating within-group proportions (e.g., ["stratum_num", "sex"])

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with within-group proportions and original proportions

    Notes
    -----
    This function assumes each DataFrame has a 'proportion' column and a 'length_bin' column.
    The within-group proportion is calculated as the proportion divided by the sum of proportions
    within each grouping level.

    Examples
    --------
    >>> props_dict = {'aged': aged_df, 'unaged': unaged_df}
    >>> within_props = calculate_within_group_proportions(
    ...     props_dict,
    ...     ["stratum_num", "sex"]
    ... )
    """
    series = [
        (
            df.assign(
                within_group_proportion=df["proportion"]
                / df.groupby(group_cols)["proportion"].transform("sum")
            )
            .groupby(group_cols + ["length_bin"], observed=False)[
                ["within_group_proportion", "proportion"]
            ]
            .sum()
            .reset_index()
            .assign(group=key)
        )
        for key, df in proportions_dict.items()
    ]
    return pd.concat(series, axis=0)


def stratum_averaged_weight(
    number_proportions: dict[str, xr.Dataset] | xr.Dataset,
    length_weight_data: xr.DataArray,
    stratum_dim: str | None = None,
) -> xr.DataArray:
    """
    Calculate stratum- and group-specific average weights using xarray inputs.

    This function combines length-at-age or length-only distributions from multiple groups (e.g.,
    aged and unaged fish) to produce weighted average weights by group and stratum, using xarray
    objects for all calculations. It accounts for the proportions of each group and correctly
    weights their contributions to the final average.

    Parameters
    ----------
    number_proportions : Dict[str, xr.Dataset]
        Dictionary mapping group names to xarray Datasets containing proportion data. Each Dataset
        must have at least one variable representing proportions and coordinates for all relevant
        grouping and binning dimensions.

    length_weight_data : xr.DataArray
        xarray DataArray containing fitted weights, with coordinates matching the grouping and
        binning dimensions used in the proportion Datasets.

    stratum_dim : str, optional
        Stratification dimension name to use for grouping (e.g., ``"stratum_ks"``).
        If None, no stratification grouping is applied.

    Returns
    -------
    xr.DataArray
        DataArray with average weights for each stratum and group, indexed by stratum_dim and
        groupings.

    Notes
    -----
    - The function assumes that the length bins in number_proportions and length_weight_data match.

    - The function requires an 'all' group category in length_weight_data for calculating combined
      weights.

    - Missing strata or group categories will be excluded from the final results.

    - All calculations are performed using xarray and pandas interconversion for flexibility.

    Examples
    --------
    >>> # Prepare xarray Datasets for proportions (see proportions.number_proportions)
    >>> number_proportions = {'aged': ds_aged, 'unaged': ds_unaged}
    >>> # Prepare xarray DataArray for weights (see biology.length_binned_weights)
    >>> length_weight_data = da_binned_weights
    >>> # Calculate average weights
    >>> weights = stratum_averaged_weight(number_proportions, length_weight_data,
    stratum_dim='stratum_num')
    >>> print(weights)
    <xarray.DataArray ...>
    """
    # Apply mutable default guard
    group_columns = [stratum_dim] if stratum_dim is not None else []

    # Convert if needed
    if isinstance(number_proportions, xr.Dataset):
        number_proportions = {"data": number_proportions}

    # Get variable names for number proportions
    proportion_vars = list(number_proportions.keys())

    # Identify non-length dimensions from length_weight_data
    grps = [d for d in length_weight_data.dims if d != "length_bin"]

    # Compute the overall proportions across all animals
    # ---- Extract arrays
    das_list = [ds["proportion"] for ds in number_proportions.values()]
    # ---- Sum over the target dimensions
    das_list_sum = []
    for da in das_list:
        # ---- Find sum dimensions
        current_coords = [d for d in da.dims if d not in group_columns + grps + ["length_bin"]]
        # ---- Append summed
        das_list_sum.append(da.sum(dim=current_coords, skipna=True))
    # ---- Convert to DataArray
    within_grp_props_unnorm = xr.concat(
        das_list_sum, dim=xr.IndexVariable("group", list(number_proportions.keys())), join="outer"
    )
    # ---- Normalize
    within_grp_props = within_grp_props_unnorm / within_grp_props_unnorm.sum(dim=["length_bin"])
    # ---- Get for all
    within_grp_props_all = within_grp_props_unnorm.sum(dim=grps) / within_grp_props_unnorm.sum(
        dim=grps + ["length_bin"]
    )

    # Extract overall proportions and add group dimension
    proportion_arrays = []
    for group_name, ds in number_proportions.items():
        # ---- Get the grouped array
        grouped_array = ds["proportion"].sum(dim=[d for d in ds.dims if d not in group_columns])
        # ---- Expand the dimensions
        proportion_arrays.append(grouped_array.expand_dims({"group": [group_name]}))
    aggregate_proportions = xr.concat(proportion_arrays, dim="group")

    # Compute the re-weighted proportions from the mixture of the different groups
    adjusted_proportions = calculate_adjusted_proportions(
        proportion_vars, aggregate_proportions, within_grp_props_unnorm.sum(dim=["length_bin"])
    )

    # Adjust for case when only a single group is provided
    if len(proportion_vars) == 1:
        adjusted_proportions = xr.full_like(adjusted_proportions, 1)

    # Calculate final weights
    fitted_weights = calculate_grouped_weights(
        length_weight_data=length_weight_data,
        within_group_proportions=within_grp_props,
        within_group_proportions_all=within_grp_props_all,
        aggregate_proportions=aggregate_proportions,
        adjusted_proportions=adjusted_proportions,
        group_keys=proportion_vars,
    )

    # Drop group coordinate if retained
    if "group" in fitted_weights.coords:
        fitted_weights = fitted_weights.drop_vars("group", errors="ignore")

    return fitted_weights


def aggregate_stratum_weights(input_data, stratum_col="stratum_num"):
    """
    Aggregate weights by stratum across all groups in the input.

    This function sums weights by stratum for each group in the input data,
    handling both single DataFrame and dictionary of DataFrames cases.

    Parameters
    ----------
    input_data : Union[Dict[str, pd.DataFrame], pd.DataFrame]
        Either a DataFrame with multi-level columns including stratum_num or a dictionary of such
        DataFrames
    stratum_col : str, default "stratum_num"
        Column name for stratum identifier

    Returns
    -------
    pd.DataFrame
        DataFrame with stratum weights for each group, with stratum_num as index
        and group names as columns

    Examples
    --------
    >>> # With a dictionary of DataFrames
    >>> stratum_summary = aggregate_stratum_weights(dict_df_weight_distr)
    >>>
    >>> # With a single DataFrame
    >>> aged_summary = aggregate_stratum_weights(dict_df_weight_distr["aged"])
    """
    results = {}

    # Handle case where input is a single DataFrame
    if isinstance(input_data, pd.DataFrame):
        dict_df_weight_distr = {"data": input_data}
    else:
        dict_df_weight_distr = input_data

    for group_name, df in dict_df_weight_distr.items():
        # Find which level contains stratum_col
        stratum_level = None
        for i, name in enumerate(df.columns.names):
            if name == stratum_col:
                stratum_level = i
                break

        if stratum_level is None:
            print(f"Warning: No {stratum_col} level found in {group_name}")
            continue

        # Aggregate weights by stratum - this returns a Series indexed by stratum_col
        stratum_weights = df.T.groupby(level=stratum_level).sum().T.sum()

        # Store the series with appropriate name
        results[f"{group_name}"] = stratum_weights

    # Combine all results into a DataFrame
    if not results:
        return pd.DataFrame(index=pd.Index([], name=stratum_col))

    final_df = pd.DataFrame(results)

    # Fill NaN values with 0
    final_df = final_df.fillna(0)

    return final_df


def weight_proportions(
    weight_data: xr.Dataset,
    catch_data: pd.DataFrame | dict[str, pd.DataFrame],
    stratum_dim: str | None = None,
    proportion_reference: Literal["catch", "catch_plus_specimen"] = "catch",
    *,
    net_selectivity: bool = False,
) -> xr.Dataset:
    """
    Calculate stratified weight proportions using xarray and pandas inputs.

    This function computes the proportional weight distribution for each group relative to the
    total weights across all strata or groupings, using an xarray DataArray for biological weights
    and a pandas DataFrame for catch weights. All grouping is handled dynamically based on the
    provided ``stratum_dim``.

    Parameters
    ----------
    weight_data : xr.Dataset
        Summed weights by length and other grouping dimensions. Requires "aged"
        and "unaged". For selectivity-expanded data, also supply "aged_observed"
        and "unaged_observed" when net_selectivity=True. The corrected catch
        denominator: expanded aged + (catch - observed aged) *
        (expanded unaged / observed unaged), summed within each stratum.
        With net_selectivity=False, use the original denominator regardless of
        whether observed arrays are present.
    catch_data : pd.DataFrame
        DataFrame with catch data, including columns for stratum_dim and "weight".
    stratum_dim : str, optional
        Stratification dimension name to group by (e.g., ``"stratum_ks"``).
    proportion_reference : Literal["catch", "catch_plus_specimen"], default 'catch'
        Determines the denominator for the proportions calculation:

        - "catch": proportions are relative to the summed catch weights only (default).

        - "catch_plus_specimen": proportions are relative to the sum of catch weights and
          individual fish weights

    net_selectivity : bool, default False
        Expand the catch denominator when True. Requires "aged_observed",
        "unaged_observed", "unaged_sample_count", and proportion_reference="catch".
        When False, use the original denominator without these additional arrays.

        Within each stratum, divide expanded fitted unaged weight by observed
        fitted unaged weight, selecting samples as follows:

        - Use male and female fish whenever either was sampled
        - Use unsexed fish only when neither male nor female fish was sampled
        - When no unaged samples are available, use expansion ratio=1 without a warning

        Determine sample presence from "unaged_sample_count", which contains
        observed fish counts by stratum and sex. Recorded fish can still have
        unusable weights: missing lengths, for example, can prevent interpolation
        and leave both weight totals zero, producing 0 / 0. If the selected
        samples yield a NaN, infinite, zero, or negative ratio, issue a UserWarning
        identifying the strata and use 1. Do not switch to unsexed fish when
        sampled male or female fish have unusable weights. A ratio of 1 leaves
        the remaining unaged and tossed catch weight unexpanded.

        Observed and expanded aged weight totals use only male and female fish.
        Unsexed aged fish are excluded from the numerator and aged subtraction;
        their weight remains in the catch remainder. Unsexed unaged fish are
        retained only to supply the expansion ratio when no male or female
        unaged fish were sampled.

    Returns
    -------
    xr.Dataset
        Dataset containing the aged weight proportions in "proportion",
        indexed by all grouping dimensions present in the input data.

    Notes
    -----
    - The function assumes that grouping columns are consistent between weight_data
      and catch_data.

    - Missing or extra group/category combinations are handled automatically by xarray/pandas.

    Examples
    --------
    >>> weight_props = weight_proportions(
    ...     weight_data=ds_da_weight_dist,
    ...     catch_data=dict_df_bio["catch"],
    ...     stratum_dim="stratum_ks",
    ...     proportion_reference="catch"
    ... )
    >>> print(weight_props)
    <xarray.DataArray ...>
    """
    if not isinstance(weight_data, xr.Dataset):
        raise TypeError("weight_data must be a Dataset of summed aged/unaged weights.")
    required = {"aged", "unaged"}
    if net_selectivity:
        required.update({"aged_observed", "unaged_observed", "unaged_sample_count"})
    missing = required.difference(weight_data.data_vars)
    if missing:
        raise ValueError(f"weight_data is missing required variables: {sorted(missing)}")
    aged_weights = weight_data["aged"]
    expansion_inputs = (
        tuple(weight_data[key] for key in ("aged_observed", "unaged", "unaged_observed"))
        if net_selectivity
        else ()
    )
    if net_selectivity:
        if proportion_reference != "catch":
            raise ValueError("Weight expansion requires proportion_reference='catch'.")
        if stratum_dim is not None and any(
            stratum_dim not in value.dims for value in (aged_weights, *expansion_inputs)
        ):
            raise ValueError("All weight arrays must contain the stratum dimension.")

    # Convert stratum_dim to list for internal use
    stratum_dim_list = [stratum_dim] if stratum_dim is not None else []

    # Compute the total weights per stratum from the aged fish
    group_weights = aged_weights.sum(
        dim=[d for d in aged_weights.dims if d not in stratum_dim_list]
    )

    # Conditionally drop specimen-only hauls if required
    if proportion_reference == "catch_plus_specimen":
        # ---- Validation
        if (not isinstance(catch_data, dict)) | (set(catch_data) < {"length", "catch"}):
            raise KeyError(
                "'catch_data' must be a dictionary containing two dataframes assigned to the keys "
                "'length' and 'catch' when 'proportion_reference' is assigned "
                "'catch_plus_specimen'."
            )
        # ---- Create catch values copy
        catch_data = copy.deepcopy(catch_data)
        # ---- Get unique haul numbers
        haul_numbers = catch_data["length"]["haul_num"].unique()
        # ---- Find incompatible hauls
        catch_values = catch_data["catch"].loc[catch_data["catch"]["haul_num"].isin(haul_numbers)]
    else:
        catch_values = catch_data.copy()

    # Compute the grouped catch weights
    catch_weights = xr.DataArray(catch_values.groupby(stratum_dim_list)["weight"].sum())
    # ---- Align weights
    catch_weights_aligned, group_weights_aligned = xr.align(
        catch_weights, group_weights, join="outer", fill_value=0.0
    )
    # Make catch haul weights adjustment, if needed, to avoid double-counting weights
    if proportion_reference == "catch_plus_specimen":
        # ---- Sum together
        total_weights = catch_weights_aligned + group_weights_aligned
    elif proportion_reference == "catch":
        # ---- Use just the summed catch weights
        total_weights = catch_weights_aligned
    else:
        raise ValueError(
            f"Input for 'proportion_reference' ({proportion_reference}) is invalid. Valid options "
            f"are limited to 'catch' (default) and 'catch_plus_specimen'."
        )

    if net_selectivity:

        def stratum_total(data):
            return data.astype(float).sum(dim=[d for d in data.dims if d not in stratum_dim_list])

        observed_aged = stratum_total(weight_data["aged_observed"]).reindex_like(
            total_weights,
            fill_value=0.0,
        )
        remaining = total_weights - observed_aged
        if bool((remaining < 0).any()):
            raise ValueError("Observed aged weight exceeds catch weight.")

        def selected_total(data, sexes):
            selected = data.sel(sex=[sex for sex in sexes if sex in data.sex.values])
            return stratum_total(selected).reindex_like(total_weights, fill_value=0.0)

        observed_unaged_sexed = selected_total(weight_data["unaged_observed"], ["male", "female"])
        observed_unaged_unsexed = selected_total(weight_data["unaged_observed"], ["unsexed"])

        # Expansion ratio of sexed and unsexed fish
        ratio_unaged_sexed = (
            selected_total(weight_data["unaged"], ["male", "female"]) / observed_unaged_sexed
        )
        ratio_unaged_unsexed = (
            selected_total(weight_data["unaged"], ["unsexed"]) / observed_unaged_unsexed
        )

        # Use sample count to distinguish the case when there are sexed unaged fish without length,
        # or that there are no unaged fish at all
        counts = weight_data["unaged_sample_count"]
        has_sexed = selected_total(counts, ["male", "female"]) > 0
        has_unsexed = selected_total(counts, ["unsexed"]) > 0

        # Select which expansion ratio to use:
        # - Normally the expansion of unaged + toss fish
        #   will use the ratio computed from sexed fish ONLY
        # - If all fish are unsexed in the unaged sample,
        #   the expansion of unaged + toss fish will used the ratio computed from unsexed fish ONLY
        ratio = xr.where(
            has_sexed, ratio_unaged_sexed, xr.where(has_unsexed, ratio_unaged_unsexed, 1.0)
        )

        # Cases to use expansion ratio=1:
        # - If there is no unaged fish, use expansion ratio=1
        # - If there are unaged with but no length data are available to infer their weight,
        #   use expansion ratio=1 and issue a warning
        invalid = (~np.isfinite(ratio) | (ratio <= 0)) & (has_sexed | has_unsexed)
        if bool(invalid.any()):
            affected = invalid.to_series().loc[lambda values: values].index.tolist()
            warnings.warn(
                "Unaged samples have unusable weights; using expansion ratio 1 "
                f"in strata: {affected}",
                UserWarning,
                stacklevel=2,
            )
        ratio = ratio.where(~invalid, 1.0)
        total_weights = group_weights_aligned + remaining * ratio.where(remaining != 0, 1.0)

    # Compute the weight proportions for the array
    arr = aged_weights / total_weights
    # ---- Assign the array name
    arr.name = "proportion"
    return arr.to_dataset()


def _validate_fitted_weight_proportions(
    number_proportions: xr.DataArray | xr.Dataset,
    binned_weights: xr.DataArray,
    stratum_dim: str,
) -> None:
    """Validate inputs for ``fitted_weight_proportions``."""
    # Ensure number_proportions is a Dataset with the expected variables
    if not isinstance(number_proportions, xr.Dataset):
        raise TypeError(
            "'number_proportions' must be an xr.Dataset containing a 'proportion' variable."
        )
    if "proportion" not in number_proportions:
        raise KeyError("'number_proportions' must contain a 'proportion' variable.")

    # Ensure binned_weights is a DataArray with length_bin
    if not isinstance(binned_weights, xr.DataArray):
        raise TypeError("'binned_weights' must be an xr.DataArray.")
    if "length_bin" not in binned_weights.dims:
        raise KeyError("'binned_weights' must have 'length_bin' as a dimension.")

    # Ensure stratum_dim is a str
    if not isinstance(stratum_dim, str):
        raise TypeError(f"'stratum_dim' must be a str, got {type(stratum_dim).__name__}.")

    # Validate stratum_dim exists in number_proportions
    if stratum_dim not in number_proportions.coords:
        raise KeyError(
            f"Dimension '{stratum_dim}' in 'stratum_dim' not found in 'number_proportions'."
        )


def fitted_weight_proportions_combined(
    number_proportions: xr.Dataset,
    binned_weights: xr.DataArray,
    stratum_dim: str | None = None,
) -> xr.Dataset:
    """
    Calculate weight proportions based on number proportions.

    This function is used in workflows where the aged and unaged samples are pooled, such as when
    net selectivity is corrected based on length. The weight proportions are calculated by
    multiplying the number of proportions over sex, length, and aged with sex-specific mean weight
    at length, and then normalizing over all dimensions within a given stratum.

    Parameters
    ----------
    number_proportions : xr.Dataset
        Number proportions that must contain a ``proportion`` variable with dimensions including
        ``stratum_dim``, ``length_bin``, and ``sex``.
    binned_weights : xr.DataArray
        Mean fitted length-binned weights that must contain ``length_bin`` as a dimension and
        ``sex`` as a coordinate.
    stratum_dim : str, optional
        Stratification dimension name for normalizing proportions within each stratum
        (e.g., ``"stratum_ks"``).

    Returns
    -------
    xr.Dataset
        DataArray containing the calculated detailed weight proportions, indexed by all grouping
        and binning dimensions present in the input data.

    Examples
    --------
    >>> weight_props = fitted_weight_proportions_combined(
    ...     number_proportions=count_proportions,
    ...     binned_weights=da_binned_weights_all,
    ...     stratum_dim="stratum_ks",
    ... )
    """
    # Apply mutable default guard and convert to list for internal use
    stratum_dim_list = [stratum_dim] if stratum_dim is not None else []

    # Validate the inputs
    _validate_fitted_weight_proportions(
        number_proportions=number_proportions,
        binned_weights=binned_weights,
        stratum_dim=stratum_dim,
    )

    # Derive within-stratum proportions from overall proportions.
    number_proportion = number_proportions["proportion"]
    normalize_dims = [d for d in number_proportion.dims if d not in stratum_dim_list]
    within_group_proportion = number_proportion / number_proportion.sum(dim=normalize_dims)

    # Calculate mean weight in each bin of specific sex, length, and age
    mean_weight = within_group_proportion * binned_weights

    # Gather all dimensions that contribute to a single 'stratum_dim' total
    sum_dims = [d for d in mean_weight.dims if d not in stratum_dim_list]

    # Calculate the total 'stratum_dim' weight
    stratum_total = mean_weight.sum(dim=sum_dims)

    # Normalize to calculate the weight proportions (sum for each 'stratum_dim' = 1)
    weight_prop = (mean_weight / stratum_total.where(stratum_total > 0)).fillna(0.0)
    # ---- Assign array name
    weight_prop.name = "proportion"
    return weight_prop.to_dataset()


def fitted_weight_proportions(
    weight_data: xr.DataArray,
    aged_weight_proportions: xr.Dataset,
    number_proportions: xr.Dataset,
    binned_weights: xr.DataArray,
    stratum_dim: str | None = None,
) -> xr.Dataset:
    """
    Calculate weight proportions based on fitted weights with reference-value adjustments.

    This function computes comprehensive weight proportions for a group, accounting for reference
    proportions, length-binned proportions, and fitted weight tables. All grouping and dimension
    logic is handled dynamically based on the provided xarray objects and stratum_dim.

    Parameters
    ----------
    weight_data : xr.DataArray
        xarray DataArray of scaled weights for the group, with dimensions including stratum_dim.
    aged_weight_proportions : xr.Dataset
        xarray Dataset of reference aged weight proportions for comparison, with compatible
        dimensions.
    number_proportions : xr.Dataset
        xarray Dataset of number proportions by relevant grouping factors.
    binned_weights : xr.DataArray
        xarray DataArray of fitted weights by length bins (and possibly other groupings).
    stratum_dim : str, optional
        Stratification dimension name (e.g., ``"stratum_ks"``).

    Returns
    -------
    xr.Dataset
        Dataset containing the calculated detailed weight proportions, indexed by all grouping
        and binning dimensions present in the input data.

    Examples
    --------
    >>> props = fitted_weight_proportions(
    ...     weight_data=da_scaled_unaged_weights,
    ...     aged_weight_proportions=ds_da_weight_proportion["aged"],
    ...     number_proportions=dict_ds_number_proportion["unaged"],
    ...     binned_weights=da_binned_weights_all,
    ...     stratum_dim="stratum_ks"
    ... )
    >>> print(props)
    <xarray.DataArray ...>
    """
    # Apply mutable default guard and convert to list for internal use
    stratum_dim_list = [stratum_dim] if stratum_dim is not None else []

    # Preserve the original calculation that does not include "unsexed" fish
    # in unaged weight proportions
    weight_data = weight_data.sel(sex=["male", "female"])

    # Summed weight for each sex in each stratum
    # dim: [stratum x sex]
    weight_sex = (
        weight_data.groupby(stratum_dim_list)
        .sum(dim=[d for d in weight_data.dims if d not in stratum_dim_list + ["sex"]])
        .astype(float)
    )

    # Weight proportion for each sex in each stratum
    # dim: [stratum x sex]; sex=["female", "male"]
    weight_prop_sex = (weight_sex / weight_sex.sum(dim="sex")).fillna(0.0)

    # Number proportion for each length bin in each stratum
    # dim: [stratum x length_bin]
    number_overall = number_proportions["proportion"]
    number_prop_length = number_overall.sum(
        dim=[
            d
            for d in number_proportions["proportion"].dims
            if d not in stratum_dim_list + ["length_bin"]
        ]
    )
    number_prop_length = number_prop_length / number_prop_length.sum(dim="length_bin")

    # Average weight for each length bin and sex in each stratum
    # ---- binned_weights only has sex="all" over all length bins, so dropping this dimension
    # ---- dim: [stratum x length_bin]
    mean_weight_length = (number_prop_length * binned_weights.squeeze().drop_vars("sex")).fillna(
        0.0
    )

    # Weight proportion for each length bin in each stratum
    # ---- dim: [stratum x length_bin]
    weight_prop_length = (
        (mean_weight_length / mean_weight_length.sum(dim="length_bin")).fillna(0.0).squeeze()
    )
    # Weight proportion of aged samples within each stratum
    # Weight proportion of reference samples (e.g., aged) within each stratum
    # dim: [stratum]
    dim_to_sum = set(aged_weight_proportions.dims).difference(stratum_dim_list)
    weight_prop_aged = aged_weight_proportions.sum(dim=dim_to_sum)

    # Weight proportion of unaged samples in each stratum
    # dim: [stratum]
    weight_prop_unaged = 1 - weight_prop_aged

    # Compute overall weight proportions
    # dim: [stratum x length_bin x sex]; sex=["female", "male"]
    return weight_prop_unaged * weight_prop_length * weight_prop_sex


def get_nasc_proportions_slice(
    number_proportions: xr.Dataset,
    ts_length_regression_parameters: dict[str, float],
    include_filter: dict[str, Any] | None = None,
    exclude_filter: dict[str, Any] | None = None,
    stratum_dim: str | None = None,
) -> xr.DataArray:
    """
    Extract and aggregate NASC proportions for specified groups from an xarray.DataArray.

    This function selects and aggregates NASC proportions from an xarray.DataArray according to
    user-specified stratification dimension and filters, and returns the result as an
    xarray.DataArray. All grouping and filtering logic is handled dynamically based on the provided
    arguments.

    Parameters
    ----------
    number_proportions : xr.Dataset
        xarray.Dataset of number proportions by relevant grouping factors, with dimensions
        including stratum_dim and binning columns (e.g., 'sex').
    exclude_filter : Dict[str, Any], optional
        Dictionary specifying values to exclude for filtering.
    include_filter : Dict[str, Any], optional
        Dictionary specifying values to include for filtering.
    ts_length_regression_parameters : Dict[str, float]
        Target strength-length regression parameters:
        - slope: regression slope
        - intercept: regression intercept
    stratum_dim : str, optional
        Stratification dimension name to use for grouping (e.g., ``"stratum_ks"``).

    Returns
    -------
    xr.DataArray
        DataArray containing the aggregated NASC proportions for each group/stratum,
        indexed by stratum_dim.

    Notes
    -----
    - No dimension names are hard-coded; all grouping and filtering is dynamic.

    - The function assumes that grouping and binning dimensions are consistent with the DataArray
      structure.

    - Filtering is applied dynamically using include_filter and exclude_filter.

    - Output is an xarray.DataArray indexed by stratum_dim.

    Examples
    --------
    >>> props = get_nasc_proportions_slice(
    ...     number_proportions=aged_data,
    ...     stratum_dim="stratum_ks",
    ...     include_filter={"age_bin": [1]}
    ... )
    >>> print(props)
    <xarray.DataArray ...>
    """
    # Apply mutable default guards
    include_filter = include_filter or {}
    exclude_filter = exclude_filter or {}
    group_columns = [stratum_dim] if stratum_dim is not None else []

    # Get length values from length bins
    length_vals = number_proportions["length_bin"].to_series().map(lambda x: x.mid).astype(float)

    # Compute equivalent sigma_bs using target strength-length regression
    sigma_bs_equiv = xr.DataArray(
        10
        ** (
            (
                ts_length_regression_parameters["slope"] * np.log10(length_vals)
                + ts_length_regression_parameters["intercept"]
            )
            / 10.0
        ),
        dims=["length_bin"],
        coords={"length_bin": length_vals.index},
    )

    # Get length-binned proportions aggregated based on group_columns
    aggregate_array = number_proportions["proportion"].sum(
        dim=[d for d in number_proportions.coords if d not in [*group_columns, "length_bin"]]
    )

    # Calculate total weighted sigma_bs for all strata
    aggregate_weighted_sigma_bs = (aggregate_array * sigma_bs_equiv.T).sum(
        dim=[d for d in aggregate_array.coords if d not in group_columns]
    )

    # Apply filters, if any
    filtered_proportions = number_proportions["proportion"]
    if include_filter:
        filtered_proportions = filtered_proportions.sel(include_filter)
    if exclude_filter:
        filtered_proportions = filtered_proportions.drop_sel(exclude_filter)
    # ---- Drop any singleton coordinates
    grouped_props = filtered_proportions.squeeze(drop=True)

    # Weight sigma_bs
    weighted_sigma_bs = (grouped_props * sigma_bs_equiv.T).sum(
        dim=[d for d in grouped_props.coords if d not in group_columns]
    )

    # Calculate the proportions
    return weighted_sigma_bs / aggregate_weighted_sigma_bs


def get_number_proportions_slice(
    number_proportions: xr.DataArray,
    stratum_dim: list[str] | None = None,
    exclude_filter: dict[str, Any] | None = None,
    include_filter: dict[str, Any] | None = None,
) -> xr.DataArray:
    """
    Extract and aggregate number proportions for specified groups from an xarray.DataArray.

    This function selects and aggregates number proportions from an xarray.DataArray according to

    Extract and aggregate number proportions for specified groups from an xarray.DataArray.

    This function selects and aggregates number proportions from an xarray.DataArray according to
    user-specified grouping columns and filters, and returns the result as an xarray.DataArray.
    All grouping and filtering logic is handled dynamically based on the provided arguments.

    Parameters
    ----------
    number_proportions : xr.DataArray
        xarray.DataArray of number proportions by relevant grouping factors, with dimensions
        including stratum_dim and binning columns (e.g., 'length_bin').
    stratum_dim : str, optional
        Stratification dimension name (e.g., ``"stratum_ks"``).
    exclude_filter : Dict[str, Any], optional
        Dictionary specifying values to exclude for filtering.
    include_filter : Dict[str, Any], optional
        Dictionary specifying values to include for filtering.

    Returns
    -------
    xr.DataArray
        DataArray containing the aggregated number proportions for each group/stratum,
        indexed by stratum_dim.

    Notes
    -----
    - No dimension names are hard-coded; all grouping and filtering is dynamic.

    - The function assumes that grouping and binning dimensions are consistent with the DataArray
      structure.

    - Filtering is applied dynamically using include_filter and exclude_filter.

    - Output is an xarray.DataArray indexed by stratum_dim.

    Examples
    --------
    >>> props = get_number_proportions_slice(
    ...     number_proportions=dict_ds_number_proportion["aged"],
    ...     stratum_dim="stratum_ks",
    ...     include_filter={"age_bin": [1]}
    ... )
    >>> print(props)
    <xarray.DataArray ...>
    """
    # Apply mutable default guards
    stratum_dim_list = [stratum_dim] if stratum_dim is not None else []
    exclude_filter = exclude_filter or {}
    include_filter = include_filter or {}

    # Determine index columns from filter keys and length_bin
    filter_indices = set(list(exclude_filter.keys()) + list(include_filter.keys()) + ["length_bin"])

    # Intersect with actually available columns/indices
    index_set = set(number_proportions.coords).intersection(filter_indices)

    # Get length-binned proportions aggregated based on strata
    aggregate_array = number_proportions["proportion"].sum(
        dim=[d for d in number_proportions.coords if d not in [*stratum_dim_list, *index_set]]
    )

    # Normalize the proportions
    nonstrata = [d for d in aggregate_array.coords if d not in stratum_dim_list]
    strata_totals = aggregate_array.sum(dim=nonstrata)
    # ---- Re-weight
    aggregate_array_norm = aggregate_array / strata_totals

    # Apply filters, if any
    if include_filter:
        # ---- Parse existing labels
        to_keep = {k: v for k, v in include_filter.items() if k in aggregate_array_norm.coords}
        if to_keep:
            aggregate_array_norm = aggregate_array_norm.sel(to_keep)
    if exclude_filter:
        # ---- Parse existing labels
        to_drop = {k: v for k, v in exclude_filter.items() if k in aggregate_array_norm.coords}
        if to_drop:
            aggregate_array_norm = aggregate_array_norm.drop_sel(to_drop)
    # ---- Drop any singleton coordinates
    grouped_props = aggregate_array_norm.squeeze(drop=True)

    # Aggregate further over the defined strata
    return grouped_props.sum(dim=[d for d in grouped_props.coords if d not in stratum_dim_list])


def get_weight_proportions_slice(
    weight_proportions: xr.Dataset,
    stratum_dim: str | None = None,
    include_filter: dict[str, Any] | None = None,
    exclude_filter: dict[str, Any] | None = None,
    number_proportions: dict[str, xr.Dataset] | None = None,
    length_threshold_min: float | None = None,
    weight_proportion_threshold: float = 1e-10,
) -> xr.DataArray:
    """
    Calculate weight proportions for a specific population slice using xarray.

    This function computes weight proportions for a target population group from an
    xarray.Dataset, with optional thresholding based on number proportions. Thresholding helps
    identify and mask unreliable estimates due to low sample sizes or small proportions. Number
    proportions must be provided as a dictionary of xarray.Dataset objects, each containing a
    ``"proportion"`` variable.

    Parameters
    ----------
    weight_proportions : xr.Dataset
        Dataset with weight proportions and all relevant grouping dimensions.
    stratum_dim : str, optional
        Stratification dimension name (e.g., ``"stratum_ks"``).
    include_filter : Dict[str, Any], default {}
        Dictionary of filter criteria for the target group (e.g., {"age_bin": [1]}).
    exclude_filter : Dict[str, Any], default {}
        Dictionary of groups to exclude (e.g., {"length_bin": [small_lengths]}).
    number_proportions : Dict[str, xr.Dataset], default {}
        Dictionary of xarray.Dataset objects for thresholding. Each Dataset must contain a
        "proportion" variable and relevant grouping/binning dimensions.
    length_threshold_min : float or None, default None
        Minimum length for threshold calculations (e.g., 10.0 cm). Used only if number_proportions
        are provided. If None, no length-based thresholding is applied.
    weight_proportion_threshold : float, default 1e-10
        Threshold value for masking small proportions.

    Returns
    -------
    xr.DataArray
        Weight proportions by stratum_dim for the target group, with unreliable estimates masked
        to 0.

    Notes
    -----
    Thresholding Logic:

    1. Calculate target group weight proportions.

    2. If number_proportions are provided, calculate corresponding number thresholds.

    3. If length_threshold_min is set, exclude length bins below this threshold.

    4. Apply a mask where both weight and number proportions are very small.

    5. Set masked values to 0.0 to indicate unreliable estimates.

    Examples
    --------
    >>> weight_props = get_weight_proportions_slice(
    ...     weight_proportions,
    ...     stratum_dim="stratum_ks",
    ...     include_filter={"age_bin": [1]},
    ...     number_proportions={"aged": ds_aged, "unaged": ds_unaged},
    ...     length_threshold_min=10.0
    ... )
    >>> print(weight_props)
    <xarray.DataArray ...>
    """
    # Apply mutable default guards and convert stratum_dim to list for internal use
    stratum_dim_list = [stratum_dim] if stratum_dim is not None else []
    include_filter = include_filter or {}
    exclude_filter = exclude_filter or {}
    number_proportions = number_proportions or {}

    # Determine index columns from filter keys and length_bin
    index_set = set(list(exclude_filter.keys()) + list(include_filter.keys()) + ["length_bin"])

    # Get length-binned proportions aggregated based on strata
    aggregate_array = weight_proportions["proportion"].sum(
        dim=[d for d in weight_proportions.coords if d not in [*stratum_dim_list, *index_set]]
    )

    # Apply filters, if any
    if include_filter:
        aggregate_array = aggregate_array.sel(include_filter)
    if exclude_filter:
        aggregate_array = aggregate_array.drop_sel(exclude_filter)
    # ---- Drop any singleton coordinates
    grouped_props = aggregate_array.squeeze(drop=True)

    # Aggregate target stratum proportions
    target_group_weight_proportions = grouped_props.sum(
        dim=[d for d in grouped_props.dims if d not in stratum_dim_list], skipna=True
    )

    # Normalize by the unfiltered proportions
    total_weight_proportions = weight_proportions["proportion"].sum(
        dim=[d for d in weight_proportions.dims if d not in stratum_dim_list], skipna=True
    )
    proportions_weight = target_group_weight_proportions / total_weight_proportions

    # Apply thresholding if number proportions provided
    if isinstance(number_proportions, xr.Dataset):
        number_proportions = {"data": number_proportions}
    if len(number_proportions) > 0:
        # ---- Optional length-based threshold filter
        if length_threshold_min:
            # ---- Get all unique length values across datasets
            all_length_vals = np.unique(
                np.concatenate(
                    [
                        [bin.mid for bin in ds["length_bin"].values]
                        for ds in number_proportions.values()
                    ]
                )
            )
            # ---- Create length exclusion filter
            length_exclusion_filter = {
                "length_bin": all_length_vals[all_length_vals < length_threshold_min],
                **exclude_filter,
            }
            # ---- Get filtered number proportions for each dataset
            filtered_number_proportions_grp = {
                key: get_number_proportions_slice(
                    ds,
                    stratum_dim=stratum_dim,
                    exclude_filter=length_exclusion_filter,
                    include_filter=include_filter,
                )
                for key, ds in number_proportions.items()
            }
            # ---- Calculate element-wise product across all arrays
            filtered_number_proportions = functools.reduce(
                operator.mul, filtered_number_proportions_grp.values()
            ).sum(
                dim=[
                    d
                    for d in next(iter(filtered_number_proportions_grp.values())).dims
                    if d not in stratum_dim_list
                ]
            )
            # ---- Align/reindex
            proportions_weight = proportions_weight.reindex_like(
                filtered_number_proportions
            ).fillna(0.0)
            target_group_weight_proportions = target_group_weight_proportions.reindex_like(
                filtered_number_proportions
            ).fillna(0.0)
            # ---- Apply threshold mask
            threshold_mask_num = filtered_number_proportions <= weight_proportion_threshold
            # ---- Set masked values to 0
            threshold_mask_num, proportions_weight = xr.align(
                threshold_mask_num, proportions_weight, join="outer", fill_value=0.0
            )
            proportions_weight = xr.where((threshold_mask_num), 0, proportions_weight).fillna(0.0)

    # Apply weight thresholding
    threshold_mask_wgts = target_group_weight_proportions <= weight_proportion_threshold
    proportions_weight = xr.where((threshold_mask_wgts.values), 0, proportions_weight)

    # Output the masked proportions array
    return proportions_weight
