"""
FEAT apportionment functions for redistributing and standardizing population estimates.

Provides utilities for partitioning NASC, abundance, and biomass transect values across biological
groups, reallocating excluded age/size classes, and distributing unaged fish counts using aged
reference proportions.
"""

import functools
import operator
import warnings
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

warnings.simplefilter("always")


def remove_group_from_estimates(
    transect_data: pd.DataFrame,
    group_proportions: xr.Dataset,
) -> pd.DataFrame:
    """
    Partition along-transect population estimates.

    Partition NASC, abundance (and number density), and biomass (and biomass density) transect
    values across indexed groups.

    Parameters
    ----------
    transect_data : pd.DataFrame
        DataFrame containing transect data with number densities already computed. Must include:
        - A column that matches the index of the `group_proportions` DataArrays.
    group_proportions : xr.Dataset
        xarray.Dataset containing partitioning data for NASC, abundance, and biomass.
        Valid variable names are limited to 'abundance', 'biomass', and 'nasc'.
        Each variable must be a 1D DataArray with dimensions matching grouping columns in
        `transect_data`.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same structure as the input dataset, but with defined partitions applied
        to NASC and the other abundance/biomass columns.

    Notes
    -----
    The DataArrays in `group_proportions` must have dimensions that correspond to columns in the
    `transect_data`. Missing indices will result in NaN values for those rows. The function
    automatically identifies and uses the appropriate columns for merging.
    """
    # Create copy
    transect_data = transect_data.copy()

    # Empty Dataset not allowed
    if len(group_proportions.data_vars) == 0:
        raise ValueError("Input 'group_proportions' Dataset cannot be empty.")

    # Validate that there is a matching column with the Dataset coordinates
    # ---- Get coordinates
    grp_coords = list(group_proportions.coords.keys())
    if not set(grp_coords) <= set(transect_data.columns):
        # ---- Format error
        missing_coords = ", ".join(f"'{c}'" for c in grp_coords)
        raise KeyError(
            f"The following coordinates from 'group_proportions' could not be found in "
            f"'transect_data': {missing_coords}."
        )

    # Index the transect data based on the DataArray dimensions
    transect_data = transect_data.set_index(grp_coords)
    transect_data_cnv = transect_data.to_xarray()

    # Align coordinates
    group_proportions_aligned = group_proportions.reindex_like(transect_data_cnv).fillna(0.0)

    # Adjust NASC, if present; otherwise, drop to avoid partial evaluation
    if "nasc" in group_proportions_aligned:
        transect_data["nasc"] = transect_data_cnv["nasc"] * (1 - group_proportions_aligned["nasc"])
    else:
        transect_data = transect_data.drop(columns=["nasc"])

    # Adjust number density and abundance, if present; otherwise, drop to avoid partial evaluation
    # ---- Get variables
    abundance_columns = [
        v for v in transect_data_cnv.data_vars if ("abundance" in v or "number_density" in v)
    ]
    if "abundance" in group_proportions_aligned:
        transect_data[abundance_columns] = (
            transect_data_cnv[abundance_columns] * (1 - group_proportions_aligned["abundance"])
        ).to_pandas()
    else:
        transect_data = transect_data.drop(columns=abundance_columns)

    # Adjust biomass and biomass density, if present; otherwise, drop to avoid partial evaluation
    # ---- Get variables
    biomass_columns = [
        v for v in transect_data_cnv.data_vars if ("biomass" in v or "biomass_density" in v)
    ]
    if "biomass" in group_proportions_aligned:
        transect_data[biomass_columns] = (
            transect_data_cnv[biomass_columns] * (1 - group_proportions_aligned["biomass"])
        ).to_pandas()
    else:
        transect_data = transect_data.drop(columns=biomass_columns)

    # Restore original indexing
    transect_data = transect_data.reset_index()

    # Return the partitioned transect data
    return transect_data


def mesh_biomass_to_nasc(
    mesh_data: pd.DataFrame,
    biodata: pd.DataFrame | dict[str, pd.DataFrame],
    mesh_biodata_link: dict[str, str],
    stratum_weights: pd.DataFrame,
    stratum_sigma_bs: pd.DataFrame,
    group_columns: list[str] = None,
) -> None:
    """
    Convert kriged biomass estimates into abundance and NASC.

    Convert biomass estimates distributed across a grid or mesh into grouped (e.g. sex-specific)
    abundance and NASC values. This function takes kriged biomass density estimates from a
    mesh/grid and converts them to abundance and NASC (Nautical Area Scattering Coefficient) values
    by applying biological proportion data and acoustic backscattering coefficients. The function
    modifies the input mesh DataFrame in place.

    Parameters
    ----------
    mesh_data : pd.DataFrame
        DataFrame containing kriged biomass density estimates with spatial mesh information.
        Must contain columns for linking to biodata and either 'biomass', or 'biomass_density'
        and `'area'`, columns.
    biodata : pd.DataFrame or Dict[str, pd.DataFrame]
        Biological proportion data. Can be a single DataFrame or dictionary of DataFrames
        containing proportion data for different biological groups (e.g., aged/unaged).
    mesh_biodata_link : Dict[str, str]
        Dictionary mapping column names from ``mesh_data`` to biodata column names for joining.
        Example: {"geostratum_ks": "stratum_ks"}
    stratum_weights : pd.DataFrame
        DataFrame containing average weights per stratum for converting biomass to abundance.
        Index should match the linked columns from ``mesh_biodata_link``.
    stratum_sigma_bs : pd.DataFrame
        DataFrame containing sigma_bs (backscattering coefficient) values per stratum.
        Must have 'sigma_bs' column with index matching linked columns.
    group_by : List[str], default []
        List of column names to group biological data by (e.g., ["sex", "age_bin"]).

    Returns
    -------
    None
        Function modifies ``mesh_data`` in place, adding new columns for biomass, abundance,
        and NASC values.

    Raises
    ------
    KeyError
        If required link columns are missing from biodata or mesh DataFrames.

    Notes
    -----
    The function performs the following steps:

    1. Computes biomass from biomass_density x area if not present

    2. Links mesh data to biological proportions using mesh_biodata_link

    3. Applies proportions to distribute total biomass across biological groups

    4. Converts biomass to abundance using stratum weights

    5. Calculates NASC using abundance and sigma_bs values

    The NASC calculation uses the formula: NASC = abundance x sigma_bs x 4π
    """
    # Convert to a Dictionary if needed
    if group_columns is None:
        group_columns = []
    if isinstance(biodata, xr.Dataset):
        biodata = {"": biodata}

    # Get the coordinates that will be used for setting shared indices
    coord_names = set()
    for ds in biodata.values():
        coord_names.update(ds.coords.keys())

    # Check
    # ---- Setdiff: biodata
    missing_biodata_columns = set(list(mesh_biodata_link.values())).difference(coord_names)
    if len(missing_biodata_columns) > 0:
        raise KeyError(
            f"The following link columns were missing from the biological data: "
            f"{', '.join(col for col in missing_biodata_columns)}."
        )
    # ---- Setdiff: mesh
    missing_mesh_columns = set(list(mesh_biodata_link.keys())).difference(mesh_data.columns)
    if len(missing_mesh_columns) > 0:
        raise KeyError(
            f"The following link columns were missing from the mesh DataFrame: "
            f"{', '.join(col for col in missing_mesh_columns)}."
        )
    # ---- Biomass column
    if "biomass" not in mesh_data.columns:
        raise KeyError(
            "The input kriging mesh DataFrame does not have a column 'biomass'. Please compute the "
            "kriged biomass."
        )

    # Set the link column names
    mesh_data.columns = mesh_data.rename(columns=mesh_biodata_link).columns

    # Map the link columns
    mesh_index = list(mesh_biodata_link.values())

    # Index
    indexed_data = mesh_data.set_index(mesh_index)
    mesh_data.index = indexed_data.index
    for column in mesh_data.columns.difference(indexed_data.columns):
        del mesh_data[column]
    sorted_data = mesh_data.sort_index()
    for position in range(len(mesh_data.columns)):
        mesh_data.isetitem(position, sorted_data.iloc[:, position].array)
    mesh_data.index = sorted_data.index
    # ---- Convert to Dataset
    mesh_dataset = mesh_data.to_xarray()

    # Stack the proportions to be as a function of group
    biodata_grouped = {
        k: ds["proportion"].sum(dim=[v for v in ds.coords.keys() if v not in group_columns])
        for k, ds in biodata.items()
    }
    # ---- Get the coordinate values
    nonmesh_idx = list(
        set(
            coord
            for da in biodata_grouped.values()
            for coord in da.coords
            if coord not in mesh_index
        )
    )
    biodata_coords = sorted(
        set(
            coord
            for da in biodata_grouped.values()
            if isinstance(da, xr.DataArray)
            for coord in da.coords[nonmesh_idx[0]].values
        )
    )
    # ---- Fill in if empty
    if len(biodata_coords) == 0:
        biodata_coords = ["apportioned"]

    # Sum over all Datasets
    grouped_total_proportions = functools.reduce(operator.add, biodata_grouped.values())

    # Multiply the biomass by these combined proportions
    biomass_coords = [f"biomass_{c}" for c in biodata_coords]
    biomass_array = grouped_total_proportions.reindex_like(mesh_dataset) * mesh_dataset["biomass"]
    # ---- Add to original mesh Dataframe
    mesh_data[biomass_coords] = biomass_array

    # Convert to abundance
    abundance_coords = [f"abundance_{c}" for c in biodata_coords] + ["abundance"]
    biomass_array = mesh_data[biomass_coords + ["biomass"]].to_xarray().to_dataarray(dim="variable")
    abundance_array = biomass_array / stratum_weights.reindex_like(mesh_dataset)
    # ---- Add to original mesh DataFrame
    mesh_data[abundance_coords] = abundance_array.T

    # Convert to NASC
    mesh_data["nasc"] = (
        mesh_data["abundance"] * stratum_sigma_bs["sigma_bs"].reindex_like(mesh_data) * 4.0 * np.pi
    )

    # Reset the index
    reset_data = mesh_data.reset_index()
    mesh_data.index = reset_data.index
    for position, column in enumerate(reset_data.columns):
        if column not in mesh_data.columns:
            mesh_data.insert(position, column, reset_data[column].array)

    # Rename
    # ---- Create inverted link dictionary
    inverted_link = {v: k for k, v in mesh_biodata_link.items()}
    mesh_data.columns = mesh_data.rename(columns=inverted_link).columns


def impute_kriged_table(
    initial_table: xr.DataArray,
    reference_table: xr.DataArray,
    standardized_table: xr.DataArray,
    group_columns: list[str],
    subgroup_coords: list[str],
    impute_variable: list[str],
) -> xr.DataArray:
    """
    Impute missing slices in binned kriged population estimates.

    Impute missing or zero-valued slices in a standardized xarray DataArray using reference and
    initial tables. This function identifies zero-valued intervals in the reference table for
    grouped estimates, finds the nearest nonzero reference interval and imputes values in the
    standardized table using a ratio-based approach. The imputation is performed only where the
    initial table is nonzero and the reference table is zero, and is done for each group and
    interval independently.

    Parameters
    ----------
    initial_table : xr.DataArray
        The initial (unstandardized) table, typically representing population or abundance
        estimates.
    reference_table : xr.DataArray
        The reference table used to guide imputation, typically representing a more complete or
        trusted set of estimates.
    standardized_table : xr.DataArray
        The standardized table to be imputed, which will be updated and returned.
    group_columns : List[str]
        List of dimension names used to group the data (e.g., ["stratum"]).
    subgroup_coords : List[str]
        List of coordinate names that define subgroups (typically a subset of group_columns).
    impute_variable : List[str]
        List of dimension names along which imputation is performed (e.g., ["age_bin"]).

    Returns
    -------
    xr.DataArray
        The imputed standardized table, with missing or zero-valued slices replaced by imputed
        values.

    Raises
    ------
    ValueError
        If imputation fails for any group and interval combination.

    Notes
    -----
    The imputation is performed as follows:

    - For each group, identify intervals in the reference table that are zero but nonzero in the
        initial table.

    - For each such interval, find the nearest nonzero interval in the reference table.

    - Impute values in the standardized table using the formula:

        | ``imputed = initial[interval] * sum referenced[impute variable x interval] /``
        | ``summed reference[interval]``

        where the reference values are taken from the nearest nonzero interval.

    - The function updates the standardized table in-place and returns it.
    """
    # Extract dimensions
    subgroup_dim = subgroup_coords[0]

    # Get group and impute coordinate values
    group_vals = reference_table.coords[subgroup_dim].values
    interval_dim = [d for d in reference_table.dims if d not in group_columns + impute_variable][0]

    # Sum reference_table across all dims except group_columns + impute_variable
    reference_summed = reference_table.sum(
        dim=[c for c in reference_table.dims if c not in [subgroup_dim, interval_dim]]
    )

    # Sum reference_table across all group_columns
    reference_grouped_sum = reference_table.sum(dim=group_columns)

    # Mask for zeros/non-zeros along impute_variable
    ref_zero_mask = reference_summed == 0.0
    ref_nonzero_mask = reference_summed != 0.0

    # Gather indices for each grouping
    ref_zero_indices = {
        c: reference_summed.coords[interval_dim].values[ref_zero_mask.sel({subgroup_dim: c})]
        for c in group_vals
    }
    ref_nonzero_indices = {
        c: reference_summed.coords[interval_dim].values[ref_nonzero_mask.sel({subgroup_dim: c})]
        for c in group_vals
    }

    # Create translation for row numbers
    interval_to_numeric = {
        interval: i for i, interval in enumerate(reference_summed.coords[interval_dim].values)
    }

    # Convert to row numbers
    ref_zero_rows = {
        col: np.array([interval_to_numeric[ival] for ival in intervals])
        for col, intervals in ref_zero_indices.items()
    }
    ref_nonzero_rows = {
        col: np.array([interval_to_numeric[ival] for ival in intervals])
        for col, intervals in ref_nonzero_indices.items()
    }

    # Check dimensions
    extra_subgroup_coords = list(
        set(reference_summed.coords[subgroup_dim].values)
        ^ set(initial_table.coords[subgroup_dim].values)
    )
    if extra_subgroup_coords:
        extra_str = "', '".join(extra_subgroup_coords)
        # ---- Print warning
        warnings.warn(
            f"The following keys for coordinate '{subgroup_dim}' are missing from the reference "
            f"table and will therefore be skipped during imputation: '{extra_str}'. ",
            stacklevel=2,
        )

    # Apply the non-zero reference indices to each column
    table_nonzeros_mask = {
        v: initial_table.sel({subgroup_dim: v}).loc[ref_zero_indices[v]] != 0.0
        for v in ref_zero_indices
    }

    # Get actual indices of masked values
    nonzero_reference_to_table_indices = {
        v: np.array(ref_zero_indices[v])[table_nonzeros_mask[v]] for v in table_nonzeros_mask
    }

    # Get the nearest-neighbor rows and recompute the indices
    imputed_rows = {
        col: arr[
            np.argmin(
                np.abs(
                    ref_zero_rows[col][table_nonzeros_mask[col]][:, np.newaxis]
                    - ref_nonzero_rows[col]
                ),
                axis=1,
            )
        ]
        for col, arr in ref_nonzero_rows.items()
    }

    # Break and raise warning if no imputation occurs
    if all([len(v) == 0 for v in imputed_rows.values()]):
        warnings.warn(
            "No missing values detected. Imputation was skipped.",
            stacklevel=2,
        )
        return standardized_table

    # Impute to replace these values
    imputed_values = {
        v: (
            initial_table.sel({subgroup_dim: v})
            .loc[nonzero_reference_to_table_indices[v]]
            .to_numpy()[:, None]
            * reference_grouped_sum.sel({subgroup_dim: v}).isel({interval_dim: imputed_rows[v]})
            / reference_summed.sel({subgroup_dim: v}).isel({interval_dim: imputed_rows[v]})
        )
        for v in reference_summed.coords[subgroup_dim].values
    }

    # Prepare output
    target_table = standardized_table.copy()

    # Update the standardized values
    for group_key, imp_da in imputed_values.items():
        # ---- Get the coordinate mapping
        coords = {
            subgroup_dim: group_key,
            interval_dim: nonzero_reference_to_table_indices[group_key],
        }
        # ---- Apply imputed values
        target_table.loc[coords] = imp_da.data
        # ---- Validate that imputation correctly applied
        if target_table.loc[coords].size > 0 and target_table.loc[coords].equals(
            standardized_table.loc[coords]
        ):
            interval_str = "', '".join(
                str(x) for x in nonzero_reference_to_table_indices[group_key]
            )
            # ---- Format error keys
            raise ValueError(
                f"Imputation failed for group '{subgroup_dim}' = '{group_key}' at the following "
                f"'{interval_dim}' intervals: '{interval_str}'."
            )

    # Return the imputed table
    return target_table


def distribute_unaged_from_aged(
    population_table: xr.DataArray,
    reference_table: xr.DataArray,
    stratum_dim: str | None = None,
    impute: bool = True,
    impute_variable: list[str] | None = None,
) -> xr.DataArray:
    """
    Standardize and optionally impute population estimates using reference proportions.

    This function redistributes population estimates (e.g., unaged fish) to match the age/size
    structure observed in a reference table (e.g., aged fish), ensuring consistency across
    biological groups. Standardization is performed by summing over the stratification dimension
    in `stratum_dim`, then scaling by the corresponding reference proportions. Optionally,
    nearest-neighbor imputation is performed for missing or zero-valued intervals.

    Parameters
    ----------
    population_table : xr.DataArray
        The input population estimates to be standardized. Must have coordinates for
        ``stratum_dim`` and typically for "length_bin".
    reference_table : xr.DataArray
        The reference population data used for standardization. Should have matching or compatible
        coordinates/dimensions as `population_table`, with additional detail (e.g., age
        information).
    stratum_dim : str, optional
        Stratification dimension name to collapse (sum over) during standardization
        (e.g., ``"stratum_ks"``). If None, no collapsing is performed.
    impute : bool, default True
        If True, perform nearest-neighbor imputation for missing/zero values after standardization.
    impute_variable : List[str], optional
        List of dimension names along which imputation is performed (e.g., ["age_bin"]).
        Required if `impute=True`.

    Returns
    -------
    xr.DataArray
        Standardized (and optionally imputed) population estimates, with the same structure as
        `population_table` but adjusted according to reference proportions.

    Raises
    ------
    ValueError
        If required coordinates (e.g., "length_bin") are missing from either input table. If
        ``impute=True`` and ``impute_variable`` is not provided.

    Notes
    -----
    - ``stratum_dim`` is the dimension that will be summed over (collapsed) in both
      `population_table` and `reference_table`. For example, if ``stratum_dim="stratum_ks"``, the
      result will sum over the "stratum_ks" dimension, producing a table without it.

    - The function expects both `population_table` and `reference_table` to have compatible
      coordinates and dimensions.

    - If `impute=True`, missing or zero-valued slices are filled using nearest-neighbor imputation.

    - The function validates that required coordinates (e.g., "length_bin") are present and raises
      an error if not.

    Examples
    --------
    >>> result = distribute_unaged_from_aged(
    ...     population_table=da_unaged,
    ...     reference_table=da_aged,
    ...     stratum_dim="stratum_ks",
    ...     impute=True,
    ...     impute_variable=["age_bin"]
    ... )
    >>> print(result)
    <xarray.DataArray ...>
    """
    # Convert stratum_dim to list for internal use
    collapse_dims = [stratum_dim] if stratum_dim is not None else []
    # Coordinate validation for length bins
    if "length_bin" not in set(population_table.coords) & set(reference_table.coords):
        raise ValueError("Required coordinate 'length_bin' missing from the input tables.")

    # Coordinate validation for collapse dimensions
    if collapse_dims and not all(
        [
            d
            for d in collapse_dims
            if d in (set(population_table.coords) & set(reference_table.coords))
        ]
    ):
        collapse_str = "', '".join(collapse_dims)
        raise ValueError(
            f"Coordinate(s) for 'stratum_dim' ('{collapse_str}') missing from the input tables."
        )

    # Validate arguments for imputation
    if impute and not impute_variable:
        raise ValueError(
            "A string argument for 'impute_variable' must be provided if 'impute'=True"
        )
    elif impute and not set(impute_variable).issubset(reference_table.coords):
        raise ValueError(
            f"Variable for 'impute_variable' ('{impute_variable[0]}') missing from "
            f"'reference_table'."
        )

    # Get original table coordinates
    table_coords = list(population_table.coords.keys())
    # ---- Inherited grouping coordinate
    table_noncoords = list(set(table_coords) - set(collapse_dims + ["length_bin"]))

    # Get reference coordinates
    reference_coords = list(reference_table.coords.keys())

    # Standardize the apportioned table values
    standardized_table = (
        population_table.sum(dim=collapse_dims)
        * reference_table.sum(dim=collapse_dims)
        / reference_table.sum(
            dim=list(set(reference_coords).difference(table_coords)) + collapse_dims
        )
    ).fillna(0.0)

    # If imputation is requested, perform it
    if not impute:
        return standardized_table
    else:
        # ---- Validate grouping coordinates
        if len(table_noncoords) == 0:
            table_coords_str = "', '".join(table_coords)
            raise ValueError(
                f"No subgroup coordinate could be determined for imputation. Current coordinates "
                f"comprise '{table_coords_str}'. Please ensure that at least one additional "
                f"coordinate exists that does not overlap with 'length_bin' and the "
                f"'stratum_dim' coordinate."
            )
        elif len(table_noncoords) > 1:
            table_noncoords_str = "', '".join(table_noncoords)
            raise ValueError(
                f"Ambiguous subgroup coordinates for imputation: '{table_noncoords_str}'. "
                f"Only one grouping dimension is supported for imputation. "
                f"Please specify 'stratum_dim' to reduce ambiguity."
            )
        elif not all(coord in reference_table.coords for coord in table_noncoords):
            missing = [coord for coord in table_noncoords if coord not in reference_table.coords]
            missing_str = "', '".join(missing)
            raise ValueError(
                f"Inherited subgroup coordinate(s) missing from input tables: '{missing_str}'. "
                "Please ensure all subgroup coordinates are present in both tables."
            )
        # ---- Imputation
        standardized_table_imputed = impute_kriged_table(
            initial_table=population_table.sum(dim=collapse_dims),
            reference_table=reference_table,
            standardized_table=standardized_table,
            group_columns=collapse_dims,
            subgroup_coords=table_noncoords,
            impute_variable=impute_variable,
        )
        return standardized_table_imputed


def sum_population_tables(
    population_tables: dict[str, xr.DataArray],
) -> xr.DataArray:
    """
    Combine and sum population estimates from multiple input tables into a single result.

    This function takes a dictionary of population estimate tables and produces a single table by
    summing over any dimensions that are not shared by all input tables. Only dimensions present in
    every input table are retained in the result.

    Parameters
    ----------
    population_tables : Dict[str, xr.DataArray]
        Dictionary of population estimate tables to combine. Keys are table names and values are
        population tables with compatible dimensions.

    Returns
    -------
    xr.DataArray
        Combined population table with only the shared dimensions retained. Values are summed
        across all input tables for each combination of shared dimensions.

    Notes
    -----
    - Any dimension not present in all input tables is summed over before combining.

    - The result is aligned on shared dimensions and summed element-wise.

    - The function is robust to input tables with different sets of dimensions, as long as there is
      at least one shared dimension.

    Examples
    --------
    >>> combined = sum_population_tables({
    ...     "aged": table_aged,
    ...     "unaged": table_unaged
    ... })
    >>> print(combined)
    <xarray.DataArray ...>
    """
    # Validate typing
    if not population_tables:
        raise ValueError("Input 'population_tables' dictionary is empty.")
    else:
        for k, v in population_tables.items():
            if not isinstance(v, xr.DataArray):
                raise TypeError(f"Value for key '{k}' is not an xr.DataArray.")

    # Find all dimensions
    dims_sets = [set(da.dims) for da in population_tables.values()]

    # Check for shared dimensions
    shared_dims = set.intersection(*dims_sets)
    if not shared_dims:
        raise ValueError("No shared dimensions found across all input tables.")

    # Find mismatched dimensions
    unique_dims = set.union(*dims_sets) - set.intersection(*dims_sets)

    # Reduce the dimensions for alignment
    population_tables_reduced = {
        k: da.sum(dim=[d for d in da.coords if d in unique_dims])
        for k, da in population_tables.items()
    }

    # Format into list for alignment
    population_tables_list = [da for da in population_tables_reduced.values()]

    # Ensure alignment
    population_tables_aligned = xr.align(*population_tables_list)

    # Sum over the tables
    return sum(population_tables_aligned)


def reallocate_excluded_estimates(
    population_table: xr.DataArray,
    exclusion_filter: dict[str, Any],
    group_columns: list[str] = None,
) -> xr.DataArray:
    """
    Redistribute population estimates after excluding specified segments.

    Removes population segments matching the exclusion filter and proportionally reallocates their
    values across the remaining segments, preserving totals within each group defined by
    `group_columns`. This is useful for excluding specific categories (e.g., age-1 fish) while
    maintaining total population estimates within groups.

    Parameters
    ----------
    population_table : xr.DataArray
        Population estimates. Must have all dimensions referenced in `exclusion_filter` and
        `group_columns`.
    exclusion_filter : Dict[str, Any]
        Dictionary specifying which segments to exclude. Keys are dimension names, values are
        categories to exclude. Example: {"age_bin": [1]} excludes age-1 fish.
    group_columns : List[str], optional
        Dimensions to group by when redistributing excluded values (e.g., ["sex"]). If empty,
        redistribution is performed over all dimensions.

    Returns
    -------
    xr.DataArray
        Population table with excluded segments set to zero and their values redistributed across
        remaining segments.

    Raises
    ------
    ValueError
        If any exclusion_filter or group_columns dimension is missing from population_table.
    UserWarning
        If the sum of the redistributed table does not match the original within a small tolerance.

    Notes
    -----
    - Excluded segments are set to zero.
    - Their totals are redistributed proportionally across remaining segments within each group.
    - The function preserves the total population within each group defined by `group_columns`.

    Examples
    --------
    >>> result = reallocate_excluded_estimates(
    ...     population_table=da_population,
    ...     exclusion_filter={"age_bin": [1]},
    ...     group_columns=["sex"]
    ... )
    >>> print(result)
    <xarray.DataArray ...>
    """
    # If no appropriate filter is defined, then nothing is redistributed
    if group_columns is None:
        group_columns = []
    if not exclusion_filter:
        return population_table

    # Validate the exclusion filter
    missing_coords = [c for c in exclusion_filter if c not in population_table.dims]
    if missing_coords:
        missing_coords_str = "', '".join(missing_coords)
        raise ValueError(
            f"Dimensions in 'exclusion_filter' not found in 'population_table': "
            f"{missing_coords_str}."
        )

    # Validate grouping columns
    missing_groups = [g for g in group_columns if g not in population_table.dims]
    if missing_groups:
        missing_groups_str = "', '".join(missing_groups)
        raise ValueError(
            f"Dimensions in 'group_columns' not found in 'population_table': "
            f"{missing_groups_str}."
        )

    # Apply inverse of exclusion filter to get the values being excluded
    population_excluded = population_table.sel(exclusion_filter)

    # Replace the excluded values in the full table with 0
    population_masked = population_table.copy()
    population_masked.loc[exclusion_filter] = 0.0

    # Sum values over dimensions based on group_columns
    # ---- Values retained
    population_excluded_sum = population_excluded.sum(
        dim=[d for d in population_excluded.dims if d not in group_columns]
    )
    # ---- Values removed
    population_masked_sum = population_masked.sum(
        dim=[d for d in population_excluded.dims if d not in group_columns]
    )

    # Get the redistributed values that will be added to the filtered table
    population_adjusted = (
        population_masked * population_excluded_sum / population_masked_sum
    ).fillna(0.0)

    # Add the adjustments to the masked table
    population_masked += population_adjusted

    # Check for conservation of summed estimates
    orig_total = float(population_table.sum())
    masked_total = float(population_masked.sum())
    if not np.isclose(orig_total, masked_total):
        warnings.warn(
            f"Redistributed table sum ({masked_total}) does not match the original ({orig_total}) "
            f"within tolerance.",
            stacklevel=2,
        )

    # Return the adjusted population distribution
    return population_masked


def distribute_population_estimates(
    data: pd.DataFrame,
    proportions: dict[str, xr.Dataset] | xr.Dataset,
    variable: str,
    group_columns: list[str] = None,
    data_proportions_link: dict[str, str] | None = None,
) -> xr.DataArray | dict[str, xr.DataArray]:
    """
    Distribution population estimates across bins.

    Distribute population estimates (e.g. abundance, biomass) using proportions grouped by metrics
    such as age and length bins. This function takes population estimates and distributes them
    across different biological categories (e.g. sex, age, length) using calculated proportions.
    The distribution is performed by multiplying the population estimates with the corresponding
    proportions.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing population estimates. Must contain the columns specified by
        'variable' and 'group_columns'. If 'data_proportions_link' is defined, then the
        key value must also be present in the DataFrame.
    proportions : Union[Dict[str, xr.Dataset], xr.Dataset]
        Proportion data for distributing the population estimates. Can be a single xarray.Dataset
        or a dictionary of Datasets (e.g., {"aged": ds_aged, "unaged": ds_unaged}).
        Each Dataset must contain a variable named "proportion".
    variable : str
        Name of the column in data containing the values to distribute (e.g. "biomass",
        "abundance").
    group_columns : List[str]
        List of column names that define the biological groups for distribution and stratification
        (e.g. ["sex", "age_bin", "length_bin", "stratum_ks"]).
    data_proportions_link : Optional[Dict[str, str]], default None
        Dictionary mapping column names from 'data' to those in 'proportions'. For
        instance, the dictionary `{'stratum_A': 'stratum_B'}` links 'stratum_A' within
        'data' with 'stratum_B' in the 'proportions' Dataset(s).

    Returns
    -------
    Union[xr.DataArray, Dict[str, xr.DataArray]]
        Distributed estimates. If a single group is present, returns an xarray.DataArray.
        Otherwise, returns a dictionary of xarray.DataArray objects, each containing the original
        biological group structure with values distributed according to proportions.

    Notes
    -----
    The DataArrays in `proportions` must have dimensions that correspond to columns in the
    `data`. Missing indices will result in NaN values for those rows. The function
    automatically identifies and uses the appropriate columns for merging and grouping.
    """
    # Create copy
    if group_columns is None:
        group_columns = []
    data = data.copy()

    # Type-check and normalize proportions input
    if isinstance(proportions, xr.Dataset):
        proportions = {"data": proportions}

    # Verify that `variable` exists
    if variable not in data.columns:
        raise ValueError(f"Variable '{variable}' missing from `data`.")

    # Rename stratum naming, if defined
    if data_proportions_link is not None:
        data_stratum = next(iter(data_proportions_link))
        if data_stratum not in data.columns:
            raise KeyError(f"Stratum '{data_stratum}' missing from `data`.")
        data = data.rename(columns=data_proportions_link)

    # Validate group_columns overlapping with data and proportions
    # ---- proportions
    proportions_group_columns = set()
    for ds in proportions.values():
        proportions_group_columns.update(ds.coords.keys())
    # ---- Only require group_columns to be present in at least one of the sources
    missing = [
        col
        for col in group_columns
        if col not in data.columns and col not in proportions_group_columns
    ]
    if missing:
        missing_str = ", ".join(f"'{c}'" for c in missing)
        raise KeyError(
            f"The following columns from 'group_columns' could not be found in either "
            f"'data' or 'proportions': {missing_str}."
        )

    # Sum data variable over indices
    data_group_columns = [col for col in group_columns if col in data.columns]
    data_array = data.pivot_table(
        index=data_group_columns, values=variable, aggfunc="sum", observed=False
    ).to_xarray()

    # Parse the additional columns that are required for grouping
    proportions_group_columns = {
        k: [c for c in ds.coords.keys() if c in group_columns] for k, ds in proportions.items()
    }

    # Calculate the proportions
    # ---- Totals (for normalization)
    group_totals = sum(
        ds["proportion"].sum(
            dim=[c for c in proportions_group_columns[k] if c not in data_group_columns]
        )
        for k, ds in proportions.items()
    )
    # ---- Apply normalization
    proportions_norm = {
        k: (ds["proportion"] / group_totals).fillna(0.0) for k, ds in proportions.items()
    }

    # Distribute the variable over each table
    apportioned_groups = {
        k: proportions_norm[k].reindex_like(data_array[variable]) * data_array[variable]
        for k in proportions_norm.keys()
    }
    # ---- Update DataArray names
    for arr in apportioned_groups.values():
        arr.name = variable

    # Return
    if len(apportioned_groups) == 1:
        return next(iter(apportioned_groups.values()))
    else:
        return apportioned_groups
