"""
Sv (volume backscattering strength) data ingestion from Echoview CSV exports.

This module reads and preprocesses Sv data files produced by Echoview, aligning them with the
stratification and NASC data structures used downstream in the echopop analysis pipeline.
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from . import nasc


def read_echoview_sv(
    filename: Path,
    impute_coordinates: bool = True,
    transect_num: float | None = None,
    validator: Any | None = None,
    latlon_suffix: str = "s",
):
    """
    Read and process Echoview volume backscattering strength (Sv) export data.

    This function reads CSV files exported from Echoview containing acoustic backscatter
    measurements and performs basic data cleaning including coordinate imputation and transect
    numbering.

    Parameters
    ----------
    filename : pathlib.Path
        Path to the Echoview CSV export file containing Sv data
    impute_coordinates : bool, default=True
        Whether to impute missing or invalid latitude/longitude coordinates
    transect_num : float, optional
        Transect number to assign to all data in this file. If None, no transect number is added
    validator : Any, optional
        Validation object for data quality checks (currently unused)
    latlon_suffix : str, default "s"
        Coordinate suffix passed to :func:`echopop.ingest.nasc.read_echoview_export`

    Returns
    -------
    |pd.DataFrame| or None
        DataFrame containing processed Sv data with columns for acoustic measurements, coordinates,
        and metadata. Returns None if file is empty or contains no valid data

    Notes
    -----
    The function adds a `'filename'` column containing the full file path for data provenance
    tracking. Coordinate imputation uses interpolation methods from :mod:`echopop.ingest.nasc`.

    Examples
    --------
    >>> sv_data = read_echoview_sv(Path("sv_export.csv"), transect_num=1)
    >>> print(sv_data.columns)
    ['sv_mean', 'latitude', 'longitude', 'transect_num', 'filename', ...]
    """
    # Read in the defined CSV file
    sv_df = nasc.read_echoview_export(filename, validator, latlon_suffix=latlon_suffix)

    # Don't read in if the file contents are empty
    if sv_df.empty or sv_df.dropna(axis=1, how="all").empty:
        return None

    # Add transect number, if defined
    if transect_num is not None:
        sv_df["transect_num"] = transect_num

    # Append filename
    sv_df["filename"] = filename.as_posix()

    # Fix latitude and longitude
    # ---- Latitude
    if "latitude" in sv_df.columns and impute_coordinates:
        nasc.impute_bad_coordinates(sv_df, "latitude")
    # ---- Longitude
    if "longitude" in sv_df.columns and impute_coordinates:
        nasc.impute_bad_coordinates(sv_df, "longitude")

    # Return the cleaned DataFrame
    return sv_df


def apply_Sv_thresholds(data: pd.DataFrame, thresholds: dict[str, Any]):
    """
    Apply frequency-specific Sv thresholds to acoustic data.

    This function masks acoustic data outside specified minimum and maximum volume backscattering
    strength thresholds, setting invalid values to -999 dB which is the standard missing data
    indicator in acoustics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing acoustic data with 'frequency' and 'sv_mean' columns
    thresholds : Dict[str, Any]
        Dictionary mapping frequencies to threshold dictionaries containing
        'min' and 'max' values in dB re 1 m^-1

    Returns
    -------
    pd.DataFrame
        Copy of input data with Sv values outside thresholds set to -999.0

    Raises
    ------
    KeyError
        If 'sv_mean' column is not found in the input DataFrame

    Notes
    -----
    Thresholding removes noise and artifacts from acoustic data. Common threshold ranges for
    biological targets are typically -90 to -50 dB depending on frequency and expected target
    strength.

    Examples
    --------
    >>> thresholds = {18.0: {"min": -90., "max": -50.},
    ...               38.0: {"min": -85., "max": -45.}}
    >>> filtered_data = apply_Sv_thresholds(sv_data, thresholds)
    """
    # Create copy
    data = data.copy()

    # Create mapping series for minimum and maximum Sv thresholds
    freq_min_map = pd.Series({freq: vals["min"] for freq, vals in thresholds.items()})
    freq_max_map = pd.Series({freq: vals["max"] for freq, vals in thresholds.items()})

    # Apply frequency-specific thresholds
    min_thresh = data["frequency"].map(freq_min_map)
    max_thresh = data["frequency"].map(freq_max_map)

    # Create mask
    if "sv_mean" in data.columns:
        mask = (data["sv_mean"] < min_thresh) | (data["sv_mean"] > max_thresh)
    else:
        raise KeyError(
            "Could not apply Sv thresholds. Column 'sv_mean' could not be found within the "
            "ingested acoustic DataFrame."
        )

    # Apply the thresholds
    data.loc[mask, "sv_mean"] = np.nan

    return data


def sv_to_nasc(sv_linear, thickness_mean):
    r"""
    Convert volume backscattering coefficient (sv, linear) to NASC (:math:`S_A`).

    This function converts linear volume backscattering coefficient values to Nautical Area
    Scattering Coefficient using standard fisheries acoustics equations. NASC is the integrated
    backscatter over a layer expressed in units convenient for fisheries assessment.

    Parameters
    ----------
    sv_linear : array-like
        Volume backscattering coefficient in linear units (m^-1)
    thickness_mean : array-like
        Mean thickness of integration layer (m)

    Returns
    -------
    nasc : array-like
        Nautical area scattering coefficient (m^2 nmi^-2)

    Notes
    -----
    The conversion follows standard fisheries acoustics equations:

    .. math::
        s_A = 4\\pi (1852)^2 \\int s_v \\, dz

    where s_A is NASC, :math:`s_v` is volume backscattering coefficient, and the integral is
    approximated as :math:`s_v` × layer_thickness. The factor 4π accounts for spherical spreading,
    and (1852)² converts from m² to nmi².

    References
    ----------
    .. [1] Simmonds, J. and MacLennan, D. (2005). Fisheries Acoustics: Theory and Practice, 2nd ed.
           Oxford: Blackwell Science.
    """
    # From the equations in the image:
    # sa = ∫ sv dz  (area backscattering coefficient)
    # sA = 4π (1852)² sa  (NASC conversion)

    # Constants
    STERADIANS_SPHERE = 4 * np.pi  # 4π steradians
    METERS_PER_NMILE_SQUARED = 1852**2  # (1852 m/nmi)²

    # Calculate area backscattering coefficient (sa) by integrating sv over thickness
    sa = sv_linear * thickness_mean  # ∫ sv dz approximated as sv × thickness

    # Convert to NASC using sA = 4π(1852)² × sa
    nasc = STERADIANS_SPHERE * METERS_PER_NMILE_SQUARED * sa

    return nasc


def organize_cells(data: pd.DataFrame):
    """
    Organize acoustic data at the cell level by frequency.

    This function creates a pivot table with acoustic cells as rows and frequencies as columns,
    preserving the finest spatial resolution while organizing data for multi-frequency analysis.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing acoustic data with spatial indexing columns and frequency-dependent
        measurements

    Returns
    -------
    pd.DataFrame
        Pivot table with MultiIndex columns organized by measurement type and frequency. Missing
        values filled with appropriate defaults:

        - nasc: 0.0 (no scattering)

        - sv_mean: -999.0 (missing data indicator)

        - thickness_mean: 0.0 (no layer thickness)

    Notes
    -----
    The resulting DataFrame has a hierarchical column structure with measurement types as the first
    level and frequencies as the second level. This format is suitable for frequency-dependent
    analysis and multi-frequency target classification algorithms.

    Index columns include available spatial identifiers: 'transect_num', 'longitude', 'latitude',
    'interval', and 'layer'.
    """
    # Find the overlapping columns
    valid_idx_cols = [
        col
        for col in data.columns
        if col in ["transect_num", "longitude", "latitude", "interval", "layer"]
    ]

    # Compute pivot table
    return data.pivot_table(
        index=valid_idx_cols, columns=["frequency"], values=["sv_mean", "nasc", "thickness_mean"]
    ).fillna({"nasc": 0.0, "sv_mean": -999.0, "thickness_mean": 0.0})


def aggregate_intervals(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate acoustic integration over depth/range for each interval.

    This function integrates acoustic measurements over depth/range intervals along each transect
    by combining multiple cells within each interval while preserving frequency separation and
    spatial location information.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing acoustic data with 'interval' column for grouping and
        frequency-dependent measurements

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with intervals as primary spatial unit, organized by frequency with
        integrated Sv and summed NASC values

    Raises
    ------
    KeyError
        If 'interval' column is not present in input data

    Notes
    -----
    The aggregation process:

    1. Weights linear Sv by layer thickness: sv_weighted = sv_linear × thickness

    2. Sums weighted values within intervals: Σ(sv_weighted)

    3. Converts back to dB: Sv_interval = 10 × log₁₀(Σ(sv_weighted))

    4. Sums NASC and thickness values directly

    This approach properly accounts for varying layer thicknesses when integrating volume
    backscatter measurements.
    """
    # Create copy
    data = data.copy()

    # Find the overlapping columns
    valid_idx_cols = [
        col for col in data.columns if col in ["transect_num", "longitude", "latitude", "interval"]
    ]

    # 'Interval' must be present
    if "interval" not in valid_idx_cols:
        raise KeyError(
            "Integration over intervals requires column 'interval' to be present within the "
            "ingested acoustic DataFrame."
        )

    # Weight the 'sv' by thickness
    data["sv_t"] = data["sv_mean_linear"] * data["thickness_mean"]

    # Aggregate the values over each interval
    data_pvt = data.groupby(valid_idx_cols + ["frequency"]).agg(
        {"sv_t": lambda x: 10 * np.log10(x.sum()), "nasc": "sum", "thickness_mean": "sum"}
    )

    # Rename the Sv column
    data_pvt.rename(columns={"sv_t": "sv_mean"}, inplace=True)

    # Unstack for 'frequency'
    return data_pvt.unstack("frequency").fillna(
        {"nasc": 0.0, "sv_mean": -999.0, "thickness_mean": 0.0}
    )


def aggregate_transects(
    data: pd.DataFrame,
):
    """
    Aggregate acoustic measurements across intervals for each survey transect.

    This function integrates acoustic data across intervals for each transect, producing
    transect-level summaries with NASC-weighted coordinates and integrated volume backscatter
    estimates. This is the highest level of spatial aggregation.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing acoustic data with 'transect_num' column and spatial/acoustic
        measurements

    Returns
    -------
    pd.DataFrame
        Transect-level aggregated data with:

        - Integrated Sv values (line backscattering coefficient)

        - Summed NASC values

        - NASC-weighted average coordinates

        - Mean interval thickness values

    Raises
    ------
    KeyError
        If 'transect_num' column is not present in input data

    Notes
    -----
    The aggregation methodology:

    1. **Cell areas**: A = distance × thickness

    2. **Area weighting**: w_area = A_cell / A_total

    3. **NASC weighting**: w_nasc = NASC_cell / NASC_total

    4. **Line Sv integration**: Sv_L = 10 × log₁₀(Σ(sv_linear × w_area))

    5. **Coordinate weighting**: coord_weighted = coord × w_nasc

    This approach accounts for variable sampling density and properly weights spatial coordinates
    by acoustic backscatter intensity.
    """
    # Check for 'transect_num'
    if "transect_num" not in data.columns:
        raise KeyError(
            "Integration over transects requires column 'transect_num' to be present within the "
            "ingested acoustic DataFrame."
        )

    # Create copy
    data = data.copy()

    # Compute distance, if missing
    if "distance" not in data.columns:
        data["distance"] = data["distance_e"] - data["distance_s"]

    # Calculate 2D cell areas (needed for calculating the 'line backscattering coefficient', S_L)
    data["cell_area"] = data["distance"] * data["thickness_mean"]

    # Sum the total cell areas and NASC for each transect
    # ---- Cell area
    data["total_cell_area"] = data.groupby(["frequency", "transect_num"])["cell_area"].transform(
        "sum"
    )
    # ---- NASC
    data["total_nasc"] = data.groupby(["frequency", "transect_num"])["nasc"].transform("sum")

    # Calculate weights
    # ---- Cell areas
    data["cell_area_weight"] = data["cell_area"] / data["total_cell_area"]
    # ---- NASC
    data["nasc_weight"] = data["nasc"] / data["total_nasc"]

    # Calculate sv(L)
    data["sv_L"] = data["sv_mean_linear"] * data["cell_area_weight"]

    # Weight the coordinates
    # ---- Longitude
    data["longitude_weight"] = data["longitude"] * data["nasc_weight"]
    # ---- Latitude
    data["latitude_weight"] = data["latitude"] * data["nasc_weight"]

    # Sum the thicknesses per interval
    data["thickness_interval"] = data.groupby(["frequency", "transect_num", "interval"])[
        "thickness_mean"
    ].transform("sum")

    # Aggregate the values over each interval
    data_pvt = data.groupby(["frequency", "transect_num"]).agg(
        {
            "longitude_weight": "sum",
            "latitude_weight": "sum",
            "sv_L": lambda x: 10 * np.log10(x.sum()),
            "nasc": "sum",
            "thickness_interval": "mean",
        }
    )

    # Rename the columns
    data_pvt.rename(
        columns={
            "sv_L": "sv_mean",
            "longitude_weight": "longitude_weighted",
            "latitude_weight": "latitude_weighted",
            "thickness_interval": "thickness_mean",
        },
        inplace=True,
    )

    # Add the coordinates to the index
    return data_pvt.unstack("frequency").fillna(
        {"nasc": 0.0, "sv_mean": -999.0, "thickness_mean": 0.0}
    )


def integrate_measurements(
    data: pd.DataFrame,
    method: Literal["cells", "interval", "transect"],
    sv_thresholds: dict[str, float],
) -> pd.DataFrame:
    """
    Integrate acoustic measurements using specified spatial aggregation method.

    This is the main integration function that processes raw acoustic data through thresholding,
    unit conversion, and spatial aggregation to produce analysis-ready datasets at the desired
    spatial resolution.

    Parameters
    ----------
    data : pd.DataFrame
        Raw acoustic data DataFrame with Sv measurements and spatial coordinates
    method : Literal["cells", "interval", "transect"]
        Spatial aggregation method:

        - "cells": Preserve individual acoustic cells (finest resolution)

        - "interval": Aggregate by depth/range intervals

        - "transect": Aggregate by survey transect lines (coarsest resolution)

    sv_thresholds : Dict[str, float]
        Frequency-specific Sv threshold dictionaries with 'min' and 'max' values

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame or None]

        - Integrated acoustic data organized by frequency

        - Coordinate data (if longitude/latitude available) or None

    Raises
    ------
    ValueError
        If aggregation method is not one of the supported options

    Notes
    -----
    Processing pipeline:

    1. **Thresholding**: Remove noise and artifacts using Sv thresholds
    2. **Unit conversion**: Convert Sv from dB to linear units for integration
    3. **NASC calculation**: Compute Nautical Area Scattering Coefficient if missing
    4. **Frequency unit conversion**: Convert frequency units to Hz
    5. **Spatial aggregation**: Apply specified integration method
    6. **Coordinate processing**: Extract spatial reference data when available

    The function handles missing NASC values by calculating them from Sv
    and layer thickness using standard fisheries acoustics equations.
    """
    # Create copy
    data = data.copy()

    # Apply minimum and maximum Sv thresholds
    sv_thresholded = apply_Sv_thresholds(data, sv_thresholds)

    # Compute the linear volumetric scattering coefficient (sv)
    sv_thresholded["sv_mean_linear"] = 10.0 ** (sv_thresholded["sv_mean"] / 10.0)

    # Drop empty cells
    sv_reduced = sv_thresholded.loc[~np.isnan(sv_thresholded["sv_mean"])]

    # Convert frequency units
    sv_reduced.loc[:, "frequency"] = sv_reduced.loc[:, "frequency"] * 1e3

    # If NASC is not a column
    if "nasc" not in sv_reduced.columns:
        # ---- Calculate NASC
        sv_reduced.loc[:, "nasc"] = sv_to_nasc(
            sv_reduced["sv_mean_linear"], sv_reduced["thickness_mean"]
        )

    # Aggregation methods
    if method == "cells":
        sv_indexed = organize_cells(sv_reduced)
    elif method == "interval":
        sv_indexed = aggregate_intervals(sv_reduced)
    elif method == "transect":
        sv_indexed = aggregate_transects(sv_reduced)
    else:
        raise ValueError(
            f"The defined aggregation method ('{method}') is invalid. This method must either be "
            f"one of the following: 'cells', 'interval', 'transect'."
        )

    # Check for longitude/latitude
    if "longitude" in sv_reduced.columns and "latitude" in sv_reduced.columns:
        sv_coordinates = (
            (
                sv_reduced.groupby(
                    ["frequency"] + sv_indexed.index.names + ["longitude", "latitude"]
                )["nasc"].sum()
            )
            .unstack("frequency")
            .fillna(0.0)
        )
    else:
        sv_coordinates = None

    # Return the dataset
    return sv_indexed, sv_coordinates


def ingest_echoview_sv(
    sv_path: Path,
    center_frequencies: dict[str, float] | None = None,
    transect_pattern: str | None = None,
    aggregate_method: Literal["cells", "interval", "transect"] = "cells",
    impute_coordinates: bool = True,
    latlon_suffix: str = "s",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Complete ingestion pipeline for Echoview Sv export data.

    This is the main entry point for processing Echoview volume backscattering strength exports. It
    handles file discovery, data loading, coordinate imputation, frequency filtering, and spatial
    aggregation to produce analysis-ready acoustic datasets.

    Parameters
    ----------
    sv_path : pathlib.Path
        Directory path containing Echoview CSV export files
    center_frequencies : Dict[str, float], optional
        Dictionary mapping target frequencies (Hz) to threshold dictionaries with 'min' and 'max'
        Sv values in dB. If None, uses all frequencies with permissive thresholds.
    transect_pattern : str, optional
        Regular expression pattern to extract transect numbers from filenames.
        Should contain a capture group for the transect number
    aggregate_method : Literal["cells", "interval", "transect"], default="cells"
        Spatial aggregation method for acoustic data integration
    impute_coordinates : bool, default=True
        Whether to interpolate missing latitude/longitude coordinates
    latlon_suffix : str, default "s"
        Coordinate suffix passed to :func:`read_echoview_sv`

    Returns
    -------
    tuple[|pd.DataFrame|, |pd.DataFrame| or None]

        - ``sv_integrated``: Spatially aggregated acoustic data with :class:`pandas.MultiIndex`
          columns organized by measurement type and frequency.

        - ``sv_coordinates``: Coordinate reference data for spatial analysis, or None if
          coordinates unavailable.

    Raises
    ------
    FileNotFoundError
        If sv_path does not exist or contains no files

    Notes
    -----
    Complete processing workflow:

    1. **File Discovery**: Recursively find all CSV files in directory

    2. **Transect Mapping**: Extract transect numbers using regex pattern

    3. **Data Loading**: Read and concatenate all Sv export files

    4. **Quality Control**: Sort data and validate structure

    5. **Frequency Filtering**: Select target frequencies if specified

    6. **Integration**: Apply spatial aggregation with thresholding

    7. **Output Formatting**: Structure data for downstream analysis

    The function automatically converts frequency units from Hz to kHz to match Echoview export
    conventions.

    Examples
    --------
    >>> sv_path = Path("/data/acoustic_exports/")
    >>> frequencies = {18000: {"min": -90, "max": -50},
    ...                38000: {"min": -85, "max": -45}}
    >>> sv_data, coords = ingest_echoview_sv(
    ...     sv_path, frequencies, r"transect_(\d+)", "interval"
    ... )
    """
    # Validate directory existence
    if not sv_path.exists():
        raise FileNotFoundError(f"The export file directory ({sv_path.as_posix()}) not found!")

    # Validate files existence
    if not sv_path.iterdir():
        raise FileNotFoundError(
            f"The export file directory ({sv_path.as_posix()}) contains no files!"
        )

    # Update the units for `center_frequencies` to match expected values from Echoview
    # ---- Hz -> kHz
    if center_frequencies is not None:
        center_frequencies = {freq * 1e-3: value for freq, value in center_frequencies.items()}

    # Get the target Sv files
    sv_filepaths = {"cells": [p for p in sv_path.rglob("*.csv") if p.is_file()]}

    # Get the trasnect numbers
    if transect_pattern:
        transect_num_df = nasc.map_transect_num(sv_filepaths, transect_pattern)
    else:
        # ---- Fill in default values in the DataFrame
        transect_num_df = pd.DataFrame(
            {
                "file_type": ["cells"] * len(sv_filepaths["cells"]),
                "file_path": sv_filepaths["cells"],
                "transect_num": None,
            }
        )

    # Concatenate the files
    sv = pd.concat(
        [
            read_echoview_sv(
                row["file_path"],
                impute_coordinates,
                row["transect_num"],
                latlon_suffix=latlon_suffix,
            )
            for _, row in transect_num_df.iterrows()
        ]
    )

    # Sort
    nasc.sort_echoview_export_df(sv, inplace=True)

    # Set min/max threshold for each frequency
    if center_frequencies:
        sv_subset = sv[sv["frequency"].isin(center_frequencies)]
    else:
        sv_subset = sv.copy()
        center_frequencies = {
            freq: {"min": -999.0, "max": 999.0} for freq in sv_subset["frequency"].unique()
        }

    # Integrate the backscatter based on the defined aggregation method
    sv_integrated, sv_coordinates = integrate_measurements(
        data=sv_subset, method=aggregate_method, sv_thresholds=center_frequencies
    )

    # Return
    return sv_integrated, sv_coordinates
