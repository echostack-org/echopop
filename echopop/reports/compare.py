"""
FEAT comparison utilities for validating and visualizing echopop vs. EchoPro outputs.

Provides functions for loading, parsing, and comparing survey results across years and
methodologies, including spatial maps, difference plots, and tabular cross-year summaries.
"""

import hashlib
import os
import re
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import box

from echopop.graphics import transect_map as ptransect, utils as gtools


def extract_sex_from_sheet(df: pd.DataFrame) -> str:
    """
    Extract sex from any cell.

    Returns 'male', 'female', 'all', or None. Allows for optional parentheses and whitespace,
    e.g., '(male)', ' (ALL) ', etc. Does not match joint expressions like 'male+female',
    'female/male', etc.
    """
    # Get text
    text = " ".join(str(cell) for row in df.values for cell in row if isinstance(cell, str))

    # Find all matches that are surrounded by word boundaries or parentheses
    found = []
    for sex in ("all", "male", "female"):
        pattern = rf"(?<![+/])(?<![a-zA-Z0-9])\(?\s*{sex}\s*\)?(?![a-zA-Z0-9])(?!(\s*[+/]))"
        if re.search(pattern, text, re.IGNORECASE):
            found.append(sex)
            break

    # Avoid ambiguous sex identifiers
    if len(found) == 1:
        return found[0]
    elif len(found) > 1:
        raise ValueError(f"Multiple sex identifiers found in sheet: {', '.join(found)}.")
    return None


def extract_quantity_type(df: pd.DataFrame) -> str:
    """
    Extract the quantity-type (e.g., 'abundance', 'biomass', 'counts').

    Returns 'counts', 'abundance', 'biomass', or None. Allows for optional parentheses and
    whitespace, e.g., '(abundance)'.
    """
    # Get text
    text = " ".join(str(cell) for row in df.values for cell in row if isinstance(cell, str))

    # Find all matches that are surrounded by word boundaries or parentheses
    found = []
    for quantity in ("counts", "abundance", "biomass"):
        pattern = rf"(?<![+/])(?<![a-zA-Z0-9])\(?\s*{quantity}\s*\)?(?![a-zA-Z0-9])(?!(\s*[+/]))"
        if re.search(pattern, text, re.IGNORECASE):
            found.append(quantity)
    # ---- Get unique in case of duplicates
    found = list(set(found))

    # Avoid ambiguous quantity identifiers
    if len(found) == 1:
        return found[0]
    elif len(found) > 1:
        raise ValueError(f"Multiple quantity identifiers found in sheet: {', '.join(found)}.")
    return None


def translate_dataframe(df: pd.DataFrame) -> None:
    """
    Translate the Dataframe columns into a standardized naming scheme.

    This is done for easier tracking and developing downstream comparisons and visualization.
    This does not return an output since it is an inplace operation.
    """
    # Helper function for translating columns
    # ---- Key
    PATTERN_TRANSLATIONS = [
        (r"^Biomass$", "biomass"),
        (r"^Biomass density$", "biomass_density"),
        (r"^NASC$", "nasc"),
        (r"^Number density$", "number_density"),
        (r"nntk_total", "number_density"),
        (r"ntk_total", "abundance"),
        (r"nwgt_total", "biomass_density"),
        (r"wgt_total", "biomass"),
        (r"nntk", "number_density"),
        (r"nwgt", "biomass_density"),
        (r"ntk", "abundance"),
        (r"wgt", "biomass"),
        (r"^Lon", "longitude"),
        (r"^Lat", "latitude"),
        (r"^Transect$", "transect_num"),
    ]

    # ---- Function
    def translate_column(col: str) -> str:
        for pattern, replacement in PATTERN_TRANSLATIONS:
            if re.search(pattern, col, re.IGNORECASE):
                return re.sub(pattern, replacement, col, flags=re.IGNORECASE)
        # --- Fall back to original if no valid matches
        return col  # fallback: return original

    # Build a mapping from old column names to new names
    rename_map = {col: translate_column(str(col)) for col in df.columns}

    # Re-assign columns in the DataFrame
    df.columns = df.rename(columns=rename_map).columns


def align_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align indices of both pandas.DataFrame objects."""
    # Try alignment first
    df1, df2 = df1.align(df2, join="outer", fill_value=0, axis=1)

    # Reindex both DataFrames to the full set, in the chosen order
    if not df1[df1.index.duplicated(keep=False)].empty:
        cols = [c for c in df1.columns if c not in ["longitude", "latitude"]]
        df1_agg = df1.groupby(list(df1.index.names), sort=False)[cols].sum()
        for col in ["longitude", "latitude"]:
            df1_agg[col] = df1.groupby(list(df1.index.names), sort=False)[col].first()
        df1_agg = df1_agg[df1.columns]
    else:
        df1_agg = df1.copy()

    if not df2[df2.index.duplicated(keep=False)].empty:
        cols = [c for c in df1.columns if c not in ["longitude", "latitude"]]
        df2_agg = df2.groupby(list(df2.index.names), sort=False)[cols].sum()
        for col in ["longitude", "latitude"]:
            df2_agg[col] = df2.groupby(list(df2.index.names), sort=False)[col].first()
        df2_agg = df2_agg[df2.columns]
    else:
        df2_agg = df2.copy()

    # Return the aligned DataFrames
    return df1_agg, df2_agg


def read_pivot_table_report(filepath: Path) -> dict[str, pd.DataFrame]:
    """
    Read all sheets from the aged length haul counts report Excel file.

    This dynamically assigns sex to the sheet name and returns a dictionary {sex: DataFrame}.
    Drops last column and row (subtotal) for each column and row, respectively.
    """
    # Validate file typing
    if not isinstance(filepath, Path):
        raise TypeError(
            f"Argument 'filepath' must be type pathlib.Path. Got type {type(filepath)}."
        )

    # Validate file existence
    if not filepath.exists():
        raise FileExistsError(f"The input file could not be located at {filepath.as_posix()}.")

    # Inspect the Excel file
    with pd.ExcelFile(filepath) as xls:

        # Pre-allocate sheet-specific dictionary
        sheet_dfs = {}

        # Iterate through each
        for sheet_name in xls.sheet_names:
            # ---- Read in the file
            df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            # ---- Extract the sex-assigned sheet
            sex = extract_sex_from_sheet(df_raw)
            # ---- Row 1 (index 1): column indices - extract from column 2 to second-to-last
            # ---- (skip subtotal col)
            column_numbers = df_raw.iloc[1, 1:-1].dropna().values
            # ---- Row 2 onwards (index 2+): actual data starts here
            lengths = df_raw.iloc[2:, 0].values
            # ---- Find where "Subtotal" row is
            subtotal_idx = None
            for idx, val in enumerate(lengths):
                if isinstance(val, str) and "Subtotal" in str(val):
                    subtotal_idx = idx + 2  # +2 because we started at row 2
                    break
            # ---- Extract only the data rows (before Subtotal)
            if subtotal_idx:
                df_clean = df_raw.iloc[2:subtotal_idx, 1:-1].copy()
                df_clean.index = df_raw.iloc[2:subtotal_idx, 0].values
            else:
                df_clean = df_raw.iloc[2:, 1:-1].copy()
                df_clean.index = df_raw.iloc[2:, 0].values
            # ---- Set column names to their respective name types
            column_name_mask = np.array(
                [isinstance(vt, float) | isinstance(vt, int) for vt in column_numbers]
            )
            # ---- Coerce the numerics to ints
            column_numbers[column_name_mask] = column_numbers[column_name_mask].astype(int)
            # ---- Capitalize otherwise
            column_numbers[~column_name_mask] = np.char.capitalize(
                np.array(column_numbers[~column_name_mask], dtype=np.str_)
            )
            # ---- Set columns
            df_clean.columns = column_numbers
            # ---- Convert to numeric
            df_clean = df_clean.apply(pd.to_numeric, errors="coerce")
            # ---- Get quantity type
            quantity_type = extract_quantity_type(df_raw)
            # ---- Add quantity type
            df_clean.columns = pd.MultiIndex.from_product([[quantity_type], df_clean.columns])
            # ---- Store
            sheet_dfs[sex] = df_clean

    # Return the dictionary of sheets
    return sheet_dfs


def read_geodata(filepath: Path) -> gpd.GeoDataFrame:
    """
    Read in georeferenced population estimates from along-transect intervals of kriging mesh nodes.

    This outputs a `geopandas.GeoDataFrame` object with the geometry informed by columns associated
    with 'longitude' and 'latitude'. The EPSG:4326 projection is automatically applied to the
    coordinates.
    """
    # Validate file typing
    if not isinstance(filepath, Path):
        raise TypeError(
            f"Argument 'filepath' must be type pathlib.Path. Got type {type(filepath)}."
        )

    # Validate file existence
    if not filepath.exists():
        raise FileExistsError(f"The input file could not be located at {filepath.as_posix()}.")

    # Read in the file
    with pd.ExcelFile(filepath) as xls:
        # ---- Identify the sheetnames and take the first one
        sheet_name = xls.sheet_names[0]
        # ---- Import data
        df_raw = pd.read_excel(xls, sheet_name=sheet_name)
        # ---- Translate the columns to internal names
        translate_dataframe(df_raw)
        # ---- Check column integrity for kriging_input
        if df_raw.shape[1] == 5:
            df_raw.columns = ["latitude", "longitude", "biomass_density", "nasc", "number_density"]

    # Convert to a GeoDataFrame
    gdf = gtools.dataframe_to_geodataframe(df_raw, "epsg:4326", ("longitude", "latitude"))

    # Return the GeoDataFrame
    return gdf


def read_aged_geodata(filepath: Path) -> dict[str, gpd.GeoDataFrame]:
    """
    Read in georeferenced aged population estimates from transects and kriged mesh nodes.

    This outputs a ``geopandas.GeoDataFrame`` object with the geometry informed by columns
    associated with 'longitude' and 'latitude'. The EPSG:4326 projection is automatically applied
    to the coordinates.
    """
    # Validate file typing
    if not isinstance(filepath, Path):
        raise TypeError(
            f"Argument 'filepath' must be type pathlib.Path. Got type {type(filepath)}."
        )

    # Validate file existence
    if not filepath.exists():
        raise FileExistsError(f"The input file could not be located at {filepath.as_posix()}.")

    # Inspect the Excel file
    with pd.ExcelFile(filepath) as xls:

        # Pre-allocate sheet-specific dictionary
        sheet_gdfs = {}

        # Iterate through each
        for sheet_name in xls.sheet_names:
            # ---- Read in the file
            df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            # ---- Extract the sex-assigned sheet
            sex = extract_sex_from_sheet(df_raw)
            # ---- Get the column names to identify coordinates vs age-class bins
            column_names = df_raw.iloc[1, :].values
            # ---- Apply some massaging for the integer ages
            column_names = [col if isinstance(col, str) else str(int(col)) for col in column_names]
            # ---- Massage into DataFrame
            df_proc = df_raw.iloc[2:, :].copy()
            # ---- Assign the column names
            df_proc.columns = column_names
            # ---- Translate
            translate_dataframe(df_proc)
            # ---- Convert to a GeoDataFrame
            gdf_proc = gtools.dataframe_to_geodataframe(
                df_proc, "epsg:4326", ("longitude", "latitude")
            )
            # ---- Store
            sheet_gdfs[sex] = gdf_proc

    # Return the output dictionary
    return sheet_gdfs


def plot_haul_count_comparisons(
    echopro: dict[str, pd.DataFrame],
    echopop: dict[str, pd.DataFrame],
    save_filepath: Path | None = None,
    show_plot: bool = True,
):
    """
    Plot the differences in fish counts per haul.

    For each sex in the intersection of both dicts, plot:

    - EchoPro heatmap
    - EchoPop heatmap
    - Difference heatmap.

    Parameters
    ----------
    echopro : Dict[str, pd.DataFrame]
    echopop : Dict[str, pd.DataFrame]
    save_filepath : Optional[Path]
        If provided, the plot will be saved to this location.
    show_plot : bool
        If True, the plot will be rendered in the user's session.
    """
    # Get all sex keys
    all_sexes = sorted(set(echopro.keys()) | set(echopop.keys()))
    n = len(all_sexes)

    # Set up plotting area
    fig, axes = plt.subplots(n, 3, figsize=(24, 6 * n), squeeze=False)

    # Iterate through for plots
    for i, sex in enumerate(all_sexes):
        # ---- Extract
        df1 = echopro.get(sex)["counts"]
        df2 = echopop.get(sex)["counts"]
        if df1 is None or df2 is None:
            # ---- Blank axes if missing
            for j in range(3):
                axes[i, j].axis("off")
            continue  # skip if either is missing
        # ---- Ensure axes are sorted ascending
        df1, df2 = align_dataframes(df1, df2)
        df1 = df1.sort_index(axis=0).sort_index(axis=1)
        df2 = df2.sort_index(axis=0).sort_index(axis=1)
        diff = df1 - df2
        diff = diff.sort_index(axis=0).sort_index(axis=1)
        # ---- Compute the shared ranges
        vmin = min(np.nanmin(df1.values), np.nanmin(df2.values))
        vmax = max(np.nanmax(df1.values), np.nanmax(df2.values))
        # ---- EchoPro
        hm1 = sns.heatmap(
            df1,
            ax=axes[i, 0],
            cmap="viridis",
            linewidths=0.5,
            linecolor="black",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
        )
        hm1.collections[0].colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # ---- Echopop
        hm2 = sns.heatmap(
            df2,
            ax=axes[i, 1],
            cmap="viridis",
            linewidths=0.5,
            linecolor="black",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
        )
        hm2.collections[0].colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # ---- Net comparison
        if np.all(diff.values == 0):
            hm3 = sns.heatmap(
                diff,
                ax=axes[i, 2],
                cmap="coolwarm",
                center=0,
                linewidths=0.5,
                linecolor="black",
                cbar=True,
                vmin=-1,
                vmax=1,
            )
        else:
            hm3 = sns.heatmap(
                diff,
                ax=axes[i, 2],
                cmap="coolwarm",
                center=0,
                linewidths=0.5,
                linecolor="black",
                cbar=True,
            )
        hm3.collections[0].colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # ---- Remove y-axis ticks from other plots in the row
        axes[i, 0].set_yticklabels(axes[i, 0].get_yticklabels(), rotation=0, ha="right")
        axes[i, 1].set_yticks([])
        axes[i, 2].set_yticks([])
        # ---- Only show x-axis ticks for bottom row
        if i == n - 1:
            for j in range(3):
                axes[i, j].tick_params(axis="x", labelrotation=45, labelsize=10)
        else:
            for j in range(3):
                axes[i, j].set_xticks([])
                axes[i, j].set_xticklabels([])
        # Add sex label as annotation to the left of the row
        axes[i, 0].annotate(
            sex.capitalize(),
            xy=(0, 0.5),
            xycoords="axes fraction",
            fontsize=16,
            ha="right",
            va="center",
            rotation=90,
            xytext=(-axes[i, 0].yaxis.labelpad - 40, 0),
            textcoords="offset points",
        )

    # Set column titles at the top
    col_titles = ["EchoPro counts", "Echopop counts", r"Differences ($\mathregular{\Delta}$counts)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=18)

    # Set the outside/shared axis titles
    axes[1, 0].set_ylabel(r"Length ($\mathregular{\ell}$, cm)")
    axes[2, 1].set_xlabel("Haul number")

    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.12)

    # Save?
    if save_filepath is not None:
        # ---- Validate directory
        if not save_filepath.parent.exists():
            save_filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_filepath, bbox_inches="tight", dpi=300)
        print(f"Figure saved at {save_filepath.as_posix()}.")

    # Show?
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_population_table_comparisons(
    echopro: dict[str, pd.DataFrame],
    echopop: dict[str, pd.DataFrame],
    save_filepath: Path | None = None,
    show_plot: bool = True,
    log_transform: bool = False,
):
    """
    Plot the differences in population estimates.

    For each sex in the intersection of both dicts, plot:

    - EchoPro heatmap
    - EchoPop heatmap
    - Difference heatmap.

    Parameters
    ----------
    echopro : Dict[str, pd.DataFrame]
    echopop : Dict[str, pd.DataFrame]
    save_filepath : Optional[Path]
        If provided, the plot will be saved to this location.
    show_plot : bool
        If True, the plot will be rendered in the user's session.
    log_transform: bool
        If True, al values within the DataFrames are log-transformed (base-10).
    """
    # Get all sex keys
    all_sexes = sorted(set(echopro.keys()) | set(echopop.keys()))
    n = len(all_sexes)

    # Set up plotting area
    fig, axes = plt.subplots(n, 3, figsize=(24, 6 * n), squeeze=False)

    # Iterate through for plots
    for i, sex in enumerate(all_sexes):
        # ---- Extract
        df1 = echopro.get(sex)
        df2 = echopop.get(sex)
        # ---- Get the quantity-types
        df1_type = df1.columns.get_level_values(0).unique()[0]
        df2_type = df2.columns.get_level_values(0).unique()[0]
        # ---- Double-check
        if df1_type != df2_type:
            raise ValueError(
                f"Quantity types between the EchoPro ('{df1_type}' and Echopop ('{df2_type}').)"
            )
        if df1 is None or df2 is None:
            # ---- Blank axes if missing
            for j in range(3):
                axes[i, j].axis("off")
            continue  # skip if either is missing
        # ---- Align
        df1_aligned, df2_aligned = align_dataframes(df1[df1_type], df2[df2_type])
        # ---- Difference in population estimates
        diff = df1_aligned - df2_aligned
        # ---- Ensure axes are sorted ascending
        df1_aligned = df1_aligned.sort_index(axis=0)
        df2_aligned = df2_aligned.sort_index(axis=0)
        diff = diff.sort_index(axis=0)
        # ---- Log-transform?
        if log_transform:
            df1_aligned = np.log10(df1_aligned + 1)
            df2_aligned = np.log10(df2_aligned + 1)
            diff = df1_aligned - df2_aligned
        # ---- Compute the shared ranges
        vmin = min(np.nanmin(df1_aligned.values), np.nanmin(df2_aligned.values))
        vmax = max(np.nanmax(df1_aligned.values), np.nanmax(df2_aligned.values))
        # ---- EchoPro
        hm1 = sns.heatmap(
            df1_aligned,
            ax=axes[i, 0],
            cmap="viridis",
            linewidths=0.5,
            linecolor="black",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
        )
        hm1.collections[0].colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # ---- Echopop
        hm2 = sns.heatmap(
            df2_aligned,
            ax=axes[i, 1],
            cmap="viridis",
            linewidths=0.5,
            linecolor="black",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
        )
        hm2.collections[0].colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # ---- Net comparison
        if np.all(diff.values == 0):
            hm3 = sns.heatmap(
                diff,
                ax=axes[i, 2],
                cmap="coolwarm",
                center=0,
                linewidths=0.5,
                linecolor="black",
                cbar=True,
                vmin=-1,
                vmax=1,
            )
        else:
            hm3 = sns.heatmap(
                diff,
                ax=axes[i, 2],
                cmap="coolwarm",
                center=0,
                linewidths=0.5,
                linecolor="black",
                cbar=True,
            )
        hm3.collections[0].colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # ---- Remove y-axis ticks from other plots in the row
        axes[i, 0].set_yticklabels(axes[i, 0].get_yticklabels(), rotation=0, ha="right")
        axes[i, 1].set_yticks([])
        axes[i, 2].set_yticks([])
        # ---- Only show x-axis ticks for bottom row
        if i == n - 1:
            for j in range(3):
                axes[i, j].tick_params(axis="x", labelrotation=45, labelsize=10)
        else:
            for j in range(3):
                axes[i, j].set_xticks([])
                axes[i, j].set_xticklabels([])
        # ---- Add sex label as annotation to the left of the row
        axes[i, 0].annotate(
            sex.capitalize(),
            xy=(0, 0.5),
            xycoords="axes fraction",
            fontsize=16,
            ha="right",
            va="center",
            rotation=90,
            xytext=(-axes[i, 0].yaxis.labelpad - 40, 0),
            textcoords="offset points",
        )

    # Set column titles at the top
    if df1_type == "abundance" and df2_type == "abundance":
        if log_transform:
            col_titles = [
                "EchoPro log$_{10}$ abundance (N)",
                "Echopop log$_{10}$ abundance (N)",
                r"EchoPro - Echopop ($\mathregular{\Delta}$abundance)",
            ]
        else:
            col_titles = [
                "EchoPro abundance (N)",
                "Echopop abundance (N)",
                r"EchoPro - Echopop ($\mathregular{\Delta}$abundance)",
            ]
    else:
        if log_transform:
            col_titles = [
                "EchoPro log$_{10}$ biomass (mmt)",
                "Echopop log$_{10}$ biomass (mmt)",
                r"EchoPro - Echopop ($\mathregular{\Delta}$biomass)",
            ]
        else:
            col_titles = [
                "EchoPro biomass (mmt)",
                "Echopop biomass (mmt)",
                r"EchoPro - Echopop ($\mathregular{\Delta}$biomass)",
            ]

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=18)

    # Set the outside/shared axis titles
    axes[1, 0].set_ylabel(r"Length ($\mathregular{\ell}$, cm)")
    axes[2, 1].set_xlabel(r"Age ($\mathregular{\alpha}$, years)")

    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.12)

    # Save?
    if save_filepath is not None:
        # ---- Validate directory
        if not save_filepath.parent.exists():
            save_filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_filepath, bbox_inches="tight", dpi=300)
        print(f"Figure saved at {save_filepath.as_posix()}.")

    # Show?
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def prettify_varname(var: str) -> tuple[str, str, str, str]:
    """Prettify the variable label."""
    # Search over the keys
    matched_str = ""
    for key in VARIABLES_KEY.keys():
        if re.search(rf"^{key}$", var) or re.search(rf"^{key}_", var):
            matched_str = key
            break

    # If no match
    if matched_str == "":
        return (var, None, None, None)

    # Get label and units
    title = VARIABLES_KEY[matched_str]["title"]
    label = VARIABLES_KEY[matched_str]["label"]
    units = VARIABLES_KEY[matched_str]["units"]

    # Remove the matched key and underscore from the start, if present
    suffix = var[len(matched_str) :]
    if suffix.startswith("_"):
        suffix = suffix[1:]
        row_title = (suffix + " " + title).capitalize()
    else:
        row_title = title

    # Format
    return (row_title, title, label, units)


def get_mapped_axes(
    ax: Axes,
    gdf: gpd.GeoDataFrame,
    var: str,
    vrange: tuple[float, float],
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    cbar: bool = True,
):
    """Plot the data layer onto the defined plotting axes."""
    # Get coastline
    with warnings.catch_warnings():
        # ---- Spoof axis limits
        bbox_polygon = box(xrange[0], yrange[0], xrange[1], yrange[1])
        bbox_gdf = gpd.GeoDataFrame(
            {"name": ["overall_bbox"]}, geometry=[bbox_polygon], crs=gdf.crs
        )
        warnings.simplefilter("ignore")
        _, coastline, _ = gtools.get_coastline(bbox_gdf)

    # Plot the coastline
    coastline.plot(ax=ax, edgecolor="black", facecolor="#C3C7C3")

    # Optionally plot the transect lines if applicable
    if "transect_num" in gdf.columns:
        # ---- Get transect lines
        transect_lines = ptransect.get_transect_lines(gdf)
        # ---- Plot
        transect_lines.plot(ax=ax, zorder=2, color="#696B69", linewidth=0.5)

    # Create copy and get rid of matching values
    gdf_copy = gdf.loc[gdf[var] != 0]

    # Plot the data variable
    sc = ax.scatter(
        x=gdf_copy["longitude"],
        y=gdf_copy["latitude"],
        c=gdf_copy[var],
        s=gtools.scale_sizes(np.abs(gdf_copy[var]), 0, np.abs(gdf_copy[var]).max(), 0.001, 50),
        vmin=vrange[0],
        vmax=vrange[1],
        cmap="viridis",
        zorder=3,
    )

    # Get variable information
    var_info = prettify_varname(var)

    # Remove $ from label and units if present
    def strip_math(s):
        return s.replace("$", "") if s else ""

    # Add colorbar
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = ax.figure.colorbar(
            sc,
            cax=cax,
            orientation="vertical",
            label=f"${{{strip_math(var_info[2])}\\ ({strip_math(var_info[3])})}}$",
        )
        cb.formatter = ScalarFormatter(useMathText=True)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()

    # Set axis limits
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)

    # Set aspect ratio
    ax.set_aspect("auto")

    # Remove the margins
    ax.margins(0, 0)

    # Return the axes
    return sc


def get_mapped_delta_axes(
    ax: Axes,
    gdf: gpd.GeoDataFrame,
    var: str,
    vrange: tuple[float, float],
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    cbar: bool = True,
):
    """Plot the differences layer onto the defined plotting axes."""
    # Get coastline
    with warnings.catch_warnings():
        # ---- Spoof axis limits
        bbox_polygon = box(xrange[0], yrange[0], xrange[1], yrange[1])
        bbox_gdf = gpd.GeoDataFrame(
            {"name": ["overall_bbox"]}, geometry=[bbox_polygon], crs=gdf.crs
        )
        warnings.simplefilter("ignore")
        _, coastline, _ = gtools.get_coastline(bbox_gdf)

    # Plot the coastline
    coastline.plot(ax=ax, edgecolor="black", facecolor="#C3C7C3")

    # Optionally plot the transect lines if applicable
    if "transect_num" in gdf.columns:
        # ---- Get transect lines
        transect_lines = ptransect.get_transect_lines(gdf)
        # ---- Plot
        transect_lines.plot(ax=ax, zorder=2, color="#696B69", linewidth=0.5)

    # Create copy and get rid of matching values
    gdf_copy = gdf.loc[gdf[var] != 0]

    # Plot the data variable
    sc = ax.scatter(
        x=gdf_copy["longitude"],
        y=gdf_copy["latitude"],
        c=gdf_copy[var],
        s=gtools.scale_sizes(np.abs(gdf_copy[var]), 0, np.abs(gdf_copy[var]).max(), 0.001, 50),
        vmin=vrange[0],
        vmax=vrange[1],
        cmap="coolwarm",
        zorder=3,
    )

    # Get variable information
    var_info = prettify_varname(var)

    # Remove $ from label and units if present
    def strip_math(s):
        return s.replace("$", "") if s else ""

    # Add colorbar
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(
            sc,
            cax=cax,
            orientation="vertical",
            label=f"${{\\Delta\\ {strip_math(var_info[2])}\\ ({strip_math(var_info[3])})}}$",
        )
        cb.formatter = ScalarFormatter(useMathText=True)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()

    # Set axis limits
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)

    # Set aspect ratio
    ax.set_aspect("auto")

    # Remove the margins
    ax.margins(0, 0)

    # Return the axes
    return sc


VARIABLES_KEY = {
    "number_density": {
        "title": "Number density",
        "label": r"$\mathregular{\rho_\text{A}}",
        "units": r"$\mathregular{N~nmi^{-2}}$",
    },
    "biomass_density": {
        "title": "Biomass density",
        "label": r"$\mathregular{B}$",
        "units": r"$\mathregular{kg~nmi^{-2}}$",
    },
    "biomass": {
        "title": "Biomass",
        "label": r"$\mathregular{B}$",
        "units": r"$\mathregular{kg}$",
    },
    "abundance": {
        "title": "Abundance",
        "label": r"$\mathregular{N}$",
        "units": "count",
    },
    "nasc": {
        "title": r"$\mathregular{S_\text{A}}$",
        "label": r"$\mathregular{S_\text{A}}$",
        "units": r"$\mathregular{m^2~nmi^{-2}}$",
    },
}


def plot_geodata(
    echopro: gpd.GeoDataFrame,
    echopop: gpd.GeoDataFrame,
    save_filepath: Path | dict[Any, Path],
    show_plot: bool = True,
    log_transform: bool = False,
) -> None:
    """Plot georeferenced transect and kriging mesh data."""
    # Find all data variable coordinates
    # ---- Columns to exclude
    exclude_cols = {"longitude", "latitude", "transect_num", "geometry"}
    df1_cols = [col for col in echopro.columns if col not in exclude_cols]
    df2_cols = [col for col in echopop.columns if col not in exclude_cols]
    # ---- Find any differences
    only_in_df1 = set(df1_cols) - set(df2_cols)
    only_in_df2 = set(df2_cols) - set(df1_cols)
    if only_in_df1 or only_in_df2:
        msg = []
        if only_in_df1:
            msg.append(f"Columns only in echopro: {sorted(only_in_df1)}")
        if only_in_df2:
            msg.append(f"Columns only in echopop: {sorted(only_in_df2)}")
        warnings.warn(" | ".join(msg), stacklevel=2)
    # ---- Keep only columns present in both
    common_cols = sorted(set(df1_cols) & set(df2_cols))
    # ---- Sort
    common_cols = [v for v in common_cols if all(prettify_varname(v))]
    echopro_filtered = echopro[common_cols + list(exclude_cols & set(echopro.columns))]
    echopop_filtered = echopop[common_cols + list(exclude_cols & set(echopop.columns))]

    # If save_filepath is None, plot all common_cols in a single grid and show/save if desired
    if save_filepath is None or isinstance(save_filepath, Path):
        var_groups = [(tuple(common_cols), save_filepath)]
    else:
        var_groups = []
        for var_group, out_path in save_filepath.items():
            if isinstance(var_group, str):
                var_group = (var_group,)
            var_groups.append((var_group, out_path))

    # Get boundaries
    lon_min = min(echopro_filtered["longitude"].min(), echopop_filtered["longitude"].min()) * 1.005
    lon_max = max(echopro_filtered["longitude"].max(), echopop_filtered["longitude"].max()) * 0.995
    lat_min = min(echopro_filtered["latitude"].min(), echopop_filtered["latitude"].min()) * 0.995
    lat_max = max(echopro_filtered["latitude"].max(), echopop_filtered["latitude"].max()) * 1.005

    # Compute the aspect ratio
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    aspect = lon_range / lat_range if lat_range != 0 else 1

    # Define the base dimensions
    base_height = 4
    n_cols = 4  # | EchoPro | Echopop | Dummy | Delta |

    # Iterate through each data variable/grouped variables
    for var_group, out_path in var_groups:

        # Get dimensions of output plot
        n = len(var_group)
        fig_width = base_height * aspect * n_cols
        fig_height = base_height * n

        fig, axes = plt.subplots(
            n,
            n_cols,
            figsize=(fig_width, fig_height),
            gridspec_kw={"width_ratios": [1, 1, 0.15, 1]},
        )
        if n == 1:
            axes = axes.reshape(1, n_cols)

        # Validate that all variables in var_group are present in common_cols
        missing = [v for v in var_group if v not in common_cols]
        if missing:
            # ---- Format into string
            missing_str = ", ".join(f"'{v}'" for v in missing)
            raise ValueError(
                f"Variables defined in 'save_filepath' missing from input dataset:"
                f"{missing_str}."
            )

        # Parse column axes for the output plot
        for i, var in enumerate(var_group):
            # ---- Prepare data with optional log-transform
            def get_data(gdf, var):
                data = gdf[var].copy()
                if log_transform:
                    data = np.log10(data + 1)
                return data

            # --- EchoPro
            echopro_plot = echopro_filtered.copy().set_index(
                [c for c in echopro_filtered.columns if c in ["geometry", "transect_num"]]
            )
            echopro_plot[var] = get_data(echopro_plot, var)
            # ---- Echopop
            echopop_plot = echopop_filtered.copy().set_index(
                [c for c in echopop_filtered.columns if c in ["geometry", "transect_num"]]
            )
            echopop_plot[var] = get_data(echopop_plot, var)
            # ---- Align the dataframes
            ep1, ep2 = align_dataframes(echopop_plot, echopro_plot)
            # ---- Get the differences
            diff_plot = (ep1[var] - ep2[var]).to_frame()
            # ---- Reset indices
            echopro_plot = echopro_plot.reset_index()
            echopop_plot = echopop_plot.reset_index()
            diff_plot = diff_plot.reset_index()
            # ---- Regain the correct coordinates
            diff_plot["longitude"] = diff_plot["geometry"].apply(lambda p: p.x)
            diff_plot["latitude"] = diff_plot["geometry"].apply(lambda p: p.y)
            # ---- Compute the shared ranges
            vmin = min(np.nanmin(echopro_plot[var]), np.nanmin(echopop_plot[var]))
            vmax = max(np.nanmax(echopro_plot[var]), np.nanmax(echopop_plot[var]))
            dvmin = np.nanmin(diff_plot[var])
            dvmax = np.nanmax(diff_plot[var])
            # ---- EchoPro
            get_mapped_axes(
                axes[i, 0],
                echopro_plot,
                var,
                (vmin, vmax),
                (lon_min, lon_max),
                (lat_min, lat_max),
                False,
            )
            # ---- Echopop
            get_mapped_axes(
                axes[i, 1],
                echopop_plot,
                var,
                (vmin, vmax),
                (lon_min, lon_max),
                (lat_min, lat_max),
                True,
            )
            # ---- Dummy panel
            axes[i, 2].axis("off")
            # ---- Difference
            diff_geoplot = gtools.dataframe_to_geodataframe(
                diff_plot, "epsg:4326", ("longitude", "latitude")
            )
            if np.all(diff_plot[var] == 0):
                get_mapped_delta_axes(
                    axes[i, 3],
                    diff_geoplot,
                    var,
                    (-1, 1),
                    (lon_min, lon_max),
                    (lat_min, lat_max),
                    True,
                )
            else:
                get_mapped_delta_axes(
                    axes[i, 3],
                    diff_geoplot,
                    var,
                    (dvmin, dvmax),
                    (lon_min, lon_max),
                    (lat_min, lat_max),
                    True,
                )
            # ---- Remove y-axis ticks from other plots in the row
            axes[i, 0].tick_params(axis="y", labelrotation=0, labelright=False)
            axes[i, 0].set_yticks([35, 40, 45, 50, 55])
            axes[i, 1].set_yticks([])
            axes[i, 3].set_yticks([])
            # ---- Only show x-axis ticks for bottom row
            manual_x_xticks = [-132, -128, -124, -120]
            if i == n - 1:
                for j in range(n_cols):
                    axes[i, j].tick_params(axis="x", labelsize=10)
                    axes[i, j].set_xticks(manual_x_xticks)
            else:
                for j in range(n_cols):
                    axes[i, j].set_xticks([])
                    axes[i, j].set_xticklabels([])
            # ---- Add variable label as annotation to the left of the row
            axes[i, 0].annotate(
                prettify_varname(var)[0],
                xy=(0, 0.5),
                xycoords="axes fraction",
                fontsize=16,
                ha="right",
                va="center",
                rotation=90,
                xytext=(-axes[i, 0].yaxis.labelpad - 40, 0),
                textcoords="offset points",
            )

        # Set up column titles at top margin
        col_titles = ["EchoPro", "Echopop", "", "EchoPro - Echopop"]
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=18)

        # Set axis limits for all panels (if needed)
        for i in range(n):
            for j in range(n_cols):
                if j != 2:
                    axes[i, j].set_xlim(lon_min, lon_max)
                    axes[i, j].set_ylim(lat_min, lat_max)

        # Set the outside/shared axis titles
        axes[-1, 0].set_xlabel("Longitude (°E)")
        axes[-1, 0].set_ylabel("Latitude (°N)")

        # Adjust layout
        plt.tight_layout(pad=0.05)
        plt.subplots_adjust(wspace=0.05, hspace=0.10, top=0.95, bottom=0.05, left=0.1)

        if out_path is not None:
            plt.savefig(out_path, dpi=300)
            print(f"Figure saved at {out_path.as_posix()}.")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def _resolve_geodata_file(
    filepath: Path,
    dataset: Literal["echopop", "echopro"],
    type: Literal["transect", "kriging"],
    echopop_patterns: dict[str, str],
    echopro_patterns: dict[str, str],
) -> Path:
    """
    Resolve report filepath.

    Resolve the full path to the target Excel file from the report directory, dataset, and report
    type.

    Both Echopop and EchoPro filenames are matched via regex pattern. An exact filename
    string is a valid pattern and will match only that file.

    Parameters
    ----------
    filepath : Path
        Root report directory.
    dataset : Literal["echopop", "echopro"]
        The dataset source.
    type : Literal["transect", "kriging"]
        The report type.
    echopop_patterns : Dict[str, str]
        Regex filename patterns for Echopop reports, keyed by type.
    echopro_patterns : Dict[str, str]
        Regex filename patterns for EchoPro reports, keyed by type.

    Returns
    -------
    Path
        The resolved file path.
    """
    # Validate root directory
    if not filepath.exists():
        raise FileNotFoundError(f"Directory not found: '{filepath.as_posix()}'.")

    # Get regex matches depending on dataset source
    patterns = echopop_patterns if dataset == "echopop" else echopro_patterns
    pattern = re.compile(patterns[type])
    matched_files = [f for f in os.listdir(filepath) if pattern.search(f)]

    # Validate file existence
    if len(matched_files) == 1:
        return filepath / matched_files[0]
    elif len(matched_files) > 1:
        raise LookupError(
            f"Multiple files matching pattern for {dataset}-{type} in '{filepath.as_posix()}'."
        )
    else:
        raise FileNotFoundError(
            f"No file matching pattern for {dataset}-{type} found in '{filepath.as_posix()}'."
        )


def _get_cache_path(excel_file: Path, cache_dir: Path) -> Path:
    """
    Generate a parquet cache file path based on the Excel file's path and modification time.

    A change in mtime (i.e. the file was updated) produces a new cache key, forcing a re-read.

    Parameters
    ----------
    excel_file : Path
        The source Excel file.
    cache_dir : Path
        Directory in which cached parquet files are stored.

    Returns
    -------
    Path
        The cache file path.
    """
    # Get timestamp
    mtime = excel_file.stat().st_mtime

    # Generate md5 hash
    key = hashlib.md5(f"{excel_file.as_posix()}_{mtime}".encode()).hexdigest()

    # Format the cached parquet file
    return cache_dir / f"{excel_file.stem}_{key}.parquet"


def fetch_geodata(
    filepath: Path,
    dataset: Literal["echopop", "echopro"],
    type: Literal["transect", "kriging"],
    echopop_patterns: dict[str, str],
    echopro_patterns: dict[str, str],
    cache_dir: Path | None = None,
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    """
    Resolve, optionally cache, and return georeferenced population data from an Excel report.

    On the first call, the resolved file is read via ``read_geodata`` and cached as parquet.
    Subsequent calls return the cached version unless the source file has been modified.

    Parameters
    ----------
    filepath : Path
        Root report directory.
    dataset : Literal["echopop", "echopro"]
        The dataset source.
    type : Literal["transect", "kriging"]
        The report type.
    echopop_patterns : Dict[str, str]
        Regex filename patterns for Echopop reports, keyed by type.
    echopro_patterns : Dict[str, str]
        Regex filename patterns for EchoPro reports, keyed by type.
    cache_dir : Optional[Path]
        Directory for caching parquet files. If ``None``, caching is disabled.
    verbose : bool, default = False
        Boolean argument that will iteratively print out the loaded filepaths when set to True.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    # Resolve the actual Excel file
    excel_file = _resolve_geodata_file(filepath, dataset, type, echopop_patterns, echopro_patterns)

    # Check cache
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = _get_cache_path(excel_file, cache_dir)
        if cache_file.exists():
            if verbose:
                print(f"  [CACHE HIT]  {dataset}/{type} <- {excel_file.name}")
            return gpd.read_parquet(cache_file)

    # Read from Excel
    if verbose:
        print(f"  [READ EXCEL] {dataset}/{type} <- {excel_file.name}")
    gdf = read_geodata(excel_file)

    # Coerce any object-typed columns to numeric, replacing unconvertible values with NaN.
    # This handles cases where columns like 'Depth' contain placeholder strings (e.g. '.')
    # that would otherwise cause serialization failures (e.g. when writing to parquet).
    for col in gdf.select_dtypes(include="object").columns:
        if col != gdf.geometry.name:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    # Write cache
    if cache_dir is not None:
        gdf.to_parquet(cache_file)

    return gdf


def load_all_geodata_reports(
    years: list,
    echopro_root: Callable[[int], Path],
    echopop_root: Callable[[int], Path],
    echopop_patterns: dict[str, str],
    echopro_patterns: dict[str, str],
    cache_dir: Path | None = None,
    max_workers: int | None = None,
    verbose: bool = False,
) -> tuple[dict[str, dict[int, gpd.GeoDataFrame]], dict[str, dict[int, gpd.GeoDataFrame]]]:
    """
    Load all geodata reports across years and dataset types.

    By default, files are loaded sequentially. Parallel loading can be enabled by setting
    ``max_workers`` to an integer greater than 1, which uses a thread pool sized accordingly.

    Parameters
    ----------
    years : list
        Survey years to process.
    echopro_root : Callable[[int], Path]
        Function mapping year -> EchoPro report directory.
    echopop_root : Callable[[int], Path]
        Function mapping year -> Echopop report directory.
    echopop_patterns : Dict[str, str]
        Regex filename patterns for Echopop reports, keyed by type.
    echopro_patterns : Dict[str, str]
        Regex filename patterns for EchoPro reports, keyed by type.
    cache_dir : Optional[Path]
        Directory for caching parquet files. If ``None``, caching is disabled.
    max_workers : Optional[int]
        Number of threads for parallel loading. If ``None`` (default), files are loaded
        sequentially. Set to a positive integer (e.g. ``8``) to enable parallel loading.
    verbose : bool, default = False
        Boolean argument that will iteratively print out the loaded filepaths when set to True.

    Returns
    -------
    Dict[Tuple[str, str, int], geopandas.GeoDataFrame]
        Dictionary keyed by ``(dataset, type, year)``.
    """
    # Build the full task list
    tasks = [
        (year, dataset, rtype, root_fn(year))
        for year in years
        for dataset, rtype, root_fn in [
            ("echopro", "transect", echopro_root),
            ("echopro", "kriging", echopro_root),
            ("echopop", "transect", echopop_root),
            ("echopop", "kriging", echopop_root),
        ]
    ]

    # Pre-allocate the results --> Dictionary mapping year-dataset-report type to report
    results: dict[tuple[str, str, int], gpd.GeoDataFrame] = {}

    # Sequential or parallelized processing
    if max_workers is None:
        # ---- Sequential loading (default)
        for year, dataset, rtype, filepath in tasks:
            try:
                results[(dataset, rtype, year)] = fetch_geodata(
                    filepath, dataset, rtype, echopop_patterns, echopro_patterns, cache_dir, verbose
                )
            except Exception as e:
                print(f"  [ERROR] {dataset}/{rtype}/{year}: {e}")
    else:
        # ---- Parallel loading (opt-in)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    fetch_geodata,
                    filepath,
                    dataset,
                    rtype,
                    echopop_patterns,
                    echopro_patterns,
                    cache_dir,
                    verbose,
                ): (year, dataset, rtype)
                for year, dataset, rtype, filepath in tasks
            }
            for future in as_completed(futures):
                year, dataset, rtype = futures[future]
                try:
                    results[(dataset, rtype, year)] = future.result()
                except Exception as e:
                    print(f"  [ERROR] {dataset}/{rtype}/{year}: {e}")

    # Unpack flat results into nested {report_type: {year: GeoDataFrame}} containers
    report_types = ["transect", "kriging"]
    echopro_datasets = {
        rtype: {year: results[("echopro", rtype, year)] for year in years} for rtype in report_types
    }
    echopop_datasets = {
        rtype: {year: results[("echopop", rtype, year)] for year in years} for rtype in report_types
    }

    return echopro_datasets, echopop_datasets


def compute_dataset_differences(
    echopro_datasets: dict[str, dict[int, gpd.GeoDataFrame]],
    echopop_datasets: dict[str, dict[int, gpd.GeoDataFrame]],
    columns: list = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute magnitude and percent differences between EchoPro and Echopop reports.

    Compute magnitude and percent differences between EchoPro and Echopop report outputs across
    survey years and report types (transect, kriging).

    Parameters
    ----------
    echopro_datasets : Dict[str, Dict[int, geopandas.GeoDataFrame]]
        Nested dictionary of EchoPro data keyed by report type (``"transect"``, ``"kriging"``)
        and year.
    echopop_datasets : Dict[str, Dict[int, geopandas.GeoDataFrame]]
        Nested dictionary of Echopop data keyed by report type and year.
    columns : list
        Columns to sum and compare. Defaults to ``["abundance", "biomass", "nasc"]``.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]

        - ``differences``: Magnitude differences (EchoPro - Echopop), indexed by
          ``(report_type, year)``.

        - ``pct_diff``: Percent differences relative to the mean of both datasets,
          indexed by ``(report_type, year)``.
    """
    # Helper function for summations
    if columns is None:
        columns = ["abundance", "biomass", "nasc"]

    def _sum_columns(datasets: dict[str, dict[int, gpd.GeoDataFrame]]) -> pd.DataFrame:
        """
        Sum target columns across all years and report-types.

        Sum the target columns across all years and report types, returning a DataFrame indexed by
        (report_type, year).
        """
        sums = {
            rtype: {year: gdf[columns].sum() for year, gdf in year_dict.items()}
            for rtype, year_dict in datasets.items()
        }
        return pd.concat(
            {rtype: pd.DataFrame(year_sums).T for rtype, year_sums in sums.items()},
            names=["report_type", "year"],
        ).sort_index()

    # Summarize both datasets
    echopro = _sum_columns(echopro_datasets)
    echopop = _sum_columns(echopop_datasets)

    # Magnitude differences
    differences = echopro - echopop

    # Percent differences relative to the mean of both
    pct_diff = differences / echopro * 1e2

    return differences, pct_diff


def plot_dataset_differences(
    percent_differences: pd.DataFrame,
    save_filepath: Path | None = None,
    columns: list = None,
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """
    Plot the percent differences between EchoPro and Echopop population estimates.

    Plot percent differences between EchoPro and Echopop report outputs as a heatmap grid,
    with one panel per report type (transect, kriging).

    Parameters
    ----------
    percent_differences : pandas.DataFrame
        Percent differences indexed by ``(report_type, year)``, as returned by
        ``compute_dataset_differences``.
    save_filepath : Optional[Path]
        If provided, the figure is saved to this path at 300 dpi. If ``None``, the figure is
        only displayed.
    columns : list
        Column display labels for the x-axis. Defaults to ``["abundance", "biomass", "nasc"]``.
    figsize : Tuple[int, int]
        Figure size in inches. Defaults to ``(14, 6)``.
    """
    # Helper function that formats percent difference strings
    if columns is None:
        columns = ["abundance", "biomass", "nasc"]

    def _fmt_percent(df: pd.DataFrame) -> pd.DataFrame:
        """Format values as percentage strings for heatmap annotations."""
        return df.map(lambda x: f"{x:.1f}%" if not pd.isna(x) else "")

    # Annotation DataFrames per report type
    annot = {
        rtype: _fmt_percent(percent_differences.loc[rtype]) for rtype in ["transect", "kriging"]
    }

    # Consistent color scaling across both panels
    vmin = percent_differences.min().min()
    vmax = percent_differences.max().max()

    # Shared heatmap kwargs
    heatmap_kwargs = dict(
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        center=0,
        fmt="",
        linewidths=1,
        linecolor="black",
        cbar=False,
    )

    # Initialize plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Define panel configurations
    panel_configs = [
        ("transect", axes[0], "Transect", "Year", True),
        ("kriging", axes[1], "Kriging", "", False),
    ]

    # Create column mapping
    x_names = {"abundance": "Abundance", "biomass": "Biomass", "nasc": "NASC"}

    # Iterate through each data type
    for rtype, ax, title, ylabel, show_yticklabels in panel_configs:
        sns.heatmap(
            percent_differences.loc[rtype],
            ax=ax,
            annot=annot[rtype],
            **heatmap_kwargs,
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_xticklabels([x_names[c] for c in columns])
        if show_yticklabels:
            # Explicitly restore year labels in case sharey suppressed them
            ax.set_yticklabels(
                percent_differences.loc[rtype].index.astype(str),
                rotation=0,
            )
        else:
            ax.set_yticklabels([])

    # Add a single shared colorbar to the right of the right-most panel
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm, ax=axes[-1], fraction=0.046, pad=0.04, label="% Difference (EchoPro - Echopop)"
    )

    # Tighten layout
    plt.tight_layout()

    # If a save filepath is provided, write a *png image
    if save_filepath is not None:
        plt.savefig(save_filepath, dpi=300)

    # Render plot
    plt.show()
