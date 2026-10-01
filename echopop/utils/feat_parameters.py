"""
FEAT survey-specific spatial parameter definitions.

Provides year- and region-specific transect boundary functions that encode the spatial
configuration of NWFSC hake survey regions, used for mesh cropping and transect classification
across survey years.
"""

import numpy as np


def transect_mesh_region_2025(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2025 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2019 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 92
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 92
        # ---- Eastern-most transect
        transect_end = 95
        # ---- Southern boundary
        transect_lower_bound = [i + 0.6 for i in range(transect_start, transect_end + 1)]
        # ---- Northern boundary
        transect_upper_bound = [i + 0.9 for i in range(transect_start, transect_end + 1)]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 96
        # ---- Northern-most transect
        transect_end = 103
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2023(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2023 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2019 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 110
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 111
        # ---- Eastern-most transect
        transect_end = 113
        # ---- Southern boundary
        transect_lower_bound = [i + 0.6 for i in range(transect_start, transect_end + 1)]
        # ---- Northern boundary
        transect_upper_bound = [i + 0.9 for i in range(transect_start, transect_end + 1)]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 114
        # ---- Northern-most transect
        transect_end = 122
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2021(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2019 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2019 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 110
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 111
        # ---- Eastern-most transect
        transect_end = 113
        # ---- Southern boundary
        transect_lower_bound = [i + 0.6 for i in range(transect_start, transect_end + 1)]
        # ---- Northern boundary
        transect_upper_bound = [i + 0.9 for i in range(transect_start, transect_end + 1)]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 115
        # ---- Northern-most transect
        transect_end = 121
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2019(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2019 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2019 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 119
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 121
        # ---- Eastern-most transect
        transect_end = 127
        # ---- Southern boundary
        transect_lower_bound = [i + 0.6 for i in range(transect_start, transect_end + 1)]
        # ---- Northern boundary
        transect_upper_bound = [i + 0.9 for i in range(transect_start, transect_end + 1)]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 129
        # ---- Northern-most transect
        transect_end = 145
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2017(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2017 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2017 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 119
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 120
        # ---- Eastern-most transect
        transect_end = 126
        # ---- Southern boundary
        transect_lower_bound = [i + 0.6 for i in range(transect_start, transect_end + 1)]
        # ---- Northern boundary
        transect_upper_bound = [i + 0.9 for i in range(transect_start, transect_end + 1)]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 127
        # ---- Northern-most transect
        transect_end = 144
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2015(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2015 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2015 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 90
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        # transect_start = 90
        transect_start = 92
        # ---- Eastern-most transect
        # transect_end = 102
        transect_end = 97
        # ---- Southern boundary
        # transect_lower_bound = [90.1, 92.6, 102.4]
        # ---- Northern boundary
        # transect_upper_bound = [90.4, 92.9, 102.1]
        # ---- Southern boundary
        transect_lower_bound = [i + 0.6 for i in range(transect_start, transect_end + 1)]
        # ---- Northern boundary
        transect_upper_bound = [i + 0.9 for i in range(transect_start, transect_end + 1)]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        # transect_start = 75
        transect_start = 98
        # ---- Northern-most transect
        # transect_end = 102
        transect_end = 116
        # ---- Western boundary
        # transect_lower_bound = [75.1, 102.1, 104.1, 108.1, 110.1, 112.1, 114.1, 116.1]
        # ---- Eastern boundary
        # transect_upper_bound = [75.4, 102.4,   .4, 108.4, 110.4, 112.4, 114.4, 116.4]
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2013(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2013 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2013 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 117
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 117
        # ---- Eastern-most transect
        transect_end = 123
        # # ---- Southern boundary
        # transect_lower_bound = [114.4, 114.1, 115.1, 118.6, 119.6, 120.6, 122.6, 124.6]
        # # ---- Northern boundary
        # transect_upper_bound = [114.4, 117.4, 118.9, 119.9, 120.9, 122.9, 124.9]
        # ---- Southern boundary
        transect_lower_bound = [i + 0.6 for i in range(transect_start, transect_end + 1)]
        # ---- Northern boundary
        transect_upper_bound = [i + 0.9 for i in range(transect_start, transect_end + 1)]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 101
        # ---- Northern-most transect
        transect_end = 125
        # # ---- Western boundary
        # transect_lower_bound = list(np.array([101, 137, 135, 133, 130, 127, 126, 125]) + 0.1)
        # # ---- Eastern boundary
        # transect_upper_bound = [101.1, 102.1, 103.1, 138.4, 135.4, 133.4, 130.4, 123.6, 123.9]
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2012(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2012 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2012 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 107
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 107
        # ---- Eastern-most transect
        transect_end = 116
        # ---- Southern boundary
        transect_lower_bound = [105.4, 105.1, 111.6, 113.6, 115.6, 117.6]
        # ---- Northern boundary
        transect_upper_bound = [105.4, 111.9, 113.9, 115.9, 117.9]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 97
        # ---- Northern-most transect
        transect_end = 120.1
        # ---- Western boundary
        transect_lower_bound = list(np.array([97, 135, 131, 129, 127, 125, 123, 121, 120]) + 0.1)
        # ---- Eastern boundary
        transect_upper_bound = [97.1, 99.1, 133.4, 131.4, 129.4, 128.4, 126.4, 117.6, 117.9, 119.4]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2011(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2011 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2011 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 117
        # ---- Western boundary
        transect_lower_bound = list(
            np.concatenate((np.arange(transect_start, 78), np.arange(83, transect_end + 1))) + 0.1
        )
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 117
        # ---- Eastern-most transect
        transect_end = 124
        # ---- Southern boundary
        transect_lower_bound = [114.4, 114.1, 117.1, 120.6, 122.6, 124.6]
        # ---- Northern boundary
        transect_upper_bound = [114.4, 117.4, 120.9, 122.9, 124.9]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 106
        # ---- Northern-most transect
        transect_end = 128
        # ---- Western boundary
        transect_lower_bound = [105.1, 138.1, 136.1, 132.1, 130.1, 128.1]
        # ---- Eastern boundary
        transect_upper_bound = [105.1, 144.4, 138.4, 136.4, 134.4, 132.4, 124.6, 122.6, 128.4]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2009(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2009 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2009 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 112
        # ---- Western boundary
        transect_lower_bound = list(
            np.concatenate((np.arange(transect_start, 97), np.arange(98, 111))) + 0.1
        ) + [113.9]
        # ---- Eastern boundary
        transect_upper_bound = list(np.arange(transect_start, 105) + 0.4) + [112.9]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 111
        # ---- Eastern-most transect
        transect_end = 124
        # ---- Southern boundary
        transect_lower_bound = [111.9, 109.4, 113.6, 114.6, 115.6, 116.6, 117.6, 127.4, 124.1]
        # ---- Northern boundary
        transect_upper_bound = [111.9, 112.6] + list(np.arange(112, 118) + 0.9) + [124.4]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 96
        # ---- Northern-most transect
        transect_end = 124
        # ---- Western boundary
        transect_lower_bound = list(np.concatenate(([124], np.arange(127, 135), [139, 98])) + 0.1)
        # ---- Eastern boundary
        transect_upper_bound = list(np.concatenate(([124, 117], np.arange(128, 134))) + 0.4) + [
            96.1
        ]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2007(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2007 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2007 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 111
        # ---- Western boundary
        transect_lower_bound = [i + 0.1 for i in range(transect_start, transect_end + 1)]
        # ---- Eastern boundary
        transect_upper_bound = [i + 0.4 for i in range(transect_start, transect_end + 1)]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 111
        # ---- Eastern-most transect
        transect_end = 117
        # ---- Southern boundary
        transect_lower_bound = [111.1] + [i + 0.6 for i in range(transect_start, transect_end + 1)]
        # ---- Northern boundary
        transect_upper_bound = [111.4] + [i + 0.9 for i in range(transect_start, transect_end + 1)]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 97
        # ---- Northern-most transect
        transect_end = 119
        # ---- Western boundary
        transect_lower_bound = list(
            np.concatenate(
                (np.arange(119, 116, -1), np.arange(126, 136), np.arange(137, 142), [97])
            )
            + 0.1
        )
        # ---- Eastern boundary
        transect_upper_bound = (
            [119.4, 118.4, 116.9, 116.6]
            + list(np.concatenate((np.arange(128, 136), np.arange(137, 142))) + 0.4)
            + [98.1]
        )

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2005(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2005 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2005 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 108
        # ---- Western boundary
        transect_lower_bound = list(
            np.concatenate((np.arange(transect_start, 107), [transect_end])) + 0.1
        )
        # ---- Eastern boundary
        transect_upper_bound = list(
            np.concatenate(
                (np.arange(transect_start, 94), [107], np.arange(97, 107), [transect_end])
            )
            + 0.4
        )
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 108
        # ---- Eastern-most transect
        transect_end = 111
        # ---- Southern boundary
        transect_lower_bound = [108.1] + list(np.array([109, 111]) + 0.6)
        # ---- Northern boundary
        transect_upper_bound = [108.4] + list(np.array([109, 111]) + 0.9)
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 97
        # ---- Northern-most transect
        transect_end = 113
        # ---- Western boundary
        transect_lower_bound = list(np.concatenate((np.arange(113, 124, 2), [97])) + 0.1)
        # ---- Eastern boundary
        transect_upper_bound = [113.4, 111.9, 111.6] + list(np.arange(115, 124, 2) + 0.4) + [97.4]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2003(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2003 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2003 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 3
        # ---- Northern-most transect
        transect_end = 105
        # ---- Western boundary
        transect_lower_bound = list(
            np.concatenate((np.arange(transect_start, 95), np.arange(117, 120), np.arange(98, 106)))
            + 0.1
        )
        # ---- Eastern boundary
        transect_upper_bound = list(
            np.concatenate(
                (
                    np.arange(transect_start, 97),
                    [117],
                    np.arange(98, 106),
                )
            )
            + 0.4
        )
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 105
        # ---- Eastern-most transect
        transect_end = 110
        # ---- Southern boundary
        transect_lower_bound = [105.1] + list(np.arange(106, 109) + 0.6) + [110.4]
        # ---- Northern boundary
        transect_upper_bound = [105.4] + list(np.arange(106, 109) + 0.9) + [109.4]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 117
        # ---- Northern-most transect
        transect_end = 109
        # ---- Western boundary
        transect_lower_bound = list(np.arange(109, 118) + 0.1)
        # ---- Eastern boundary
        transect_upper_bound = list(np.arange(109, 117) + 0.4) + [98.1, 117.4]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_2001(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 2001 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    2001 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 139
        # ---- Western boundary
        transect_lower_bound = list(
            np.concatenate(
                (np.arange(transect_start, 86), np.arange(151, 146, -1), [144, 141, 140, 139])
            )
            + 0.1
        )
        # ---- Eastern boundary
        transect_upper_bound = list(
            np.concatenate(
                (
                    np.arange(transect_start, 86),
                    np.arange(151, 146, -1),
                    [144, 141, 140, 139],
                )
            )
            + 0.4
        )
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 139
        # ---- Eastern-most transect
        transect_end = 134
        # ---- Southern boundary
        transect_lower_bound = [139.1] + list(np.arange(138, 135, -1) + 0.6) + [134.4]
        # ---- Northern boundary
        transect_upper_bound = [139.4] + list(np.arange(138, 135, -1) + 0.9) + [135.4]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 83
        # ---- Northern-most transect
        transect_end = 135
        # ---- Western boundary
        transect_lower_bound = list(
            np.concatenate(
                (
                    np.arange(135, 127, -1),
                    [147],
                    np.arange(126, 122, -1),
                    np.arange(121, 118, -1),
                )
            )
            + 0.1
        )
        # ---- Eastern boundary
        transect_upper_bound = list(
            np.concatenate(
                (np.arange(135, 127, -1), np.arange(126, 122, -1), np.arange(121, 118, -1), [83])
            )
            + 0.4
        )

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_1998(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 1998 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    1998 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 222
        # ---- Western boundary
        transect_lower_bound = (
            list(
                np.concatenate((np.arange(transect_start, 71), np.arange(81, 104), [249, 252]))
                + 0.1
            )
            + [253.6, 253.9]
            + list(np.array([254, 248, 245, 239, 238, 236, 233, 231, 227]) + 0.1)
            + [226.6, 211.6]
        )
        # ---- Eastern boundary
        transect_upper_bound = list(
            np.concatenate(
                (
                    np.arange(transect_start, 61),
                    np.arange(70, 81),
                    [99, 265, 262, 260, 271, 274, 255, 246, 238, 236, 233, 231, 229],
                )
            )
            + 0.4
        ) + [226.9, 223.9, 222.4, 211.9]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 224
        # ---- Eastern-most transect
        transect_end = 319
        # ---- Southern boundary
        transect_lower_bound = list(np.array([226, 211, 209, 207, 205, 202]) + 0.6) + [320.4]
        # ---- Northern boundary
        transect_upper_bound = [224.6] + list(np.array([211, 217, 215, 213]) + 0.9) + [212.1, 325.4]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 103
        # ---- Northern-most transect
        transect_end = 368
        # ---- Western boundary
        transect_lower_bound = (
            [368.6, 367.4]
            + list(np.array([365.6, 363, 360, 358, 356, 353]) + 0.6)
            + list(np.array([351, 347]) + 0.1)
            + [343.6]
            + list(np.array([340, 338, 331]) + 0.1)
            + [330.6, 328.6]
            + list(
                np.concatenate(
                    ([325, 322, 320, 318, 316], np.arange(313, 301, -2), [300, 298, 294])
                )
                + 0.1
            )
            + [293.6]
            + list(np.array([286, 284, 282, 103]) + 0.1)
        )
        # ---- Eastern boundary
        transect_upper_bound = (
            [368.9, 366.4, 364.9]
            + list(np.array([360, 359, 357, 354, 352, 347, 345, 344, 342, 340, 336, 334]) + 0.4)
            + [330.9]
            + list(np.array([329, 327, 324]) + 0.4)
            + [212.1, 322.4, 321.6]
            + list(np.concatenate(([318, 316], np.arange(313, 306, -2), [304])) + 0.4)
            + [302.9, 300.4, 298.4, 297.9]
            + list(np.concatenate(([295], np.arange(292, 281, -2))) + 0.4)
        )

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound


def transect_mesh_region_1995(
    region: np.number,
) -> tuple[np.number, np.number, list[np.number], list[np.number]]:
    """
    Generate region-specific transect boundaries for the 1995 NWFSC survey.

    This function defines the spatial boundaries for three distinct survey regions used in the
    1995 Northwest Fisheries Science Center (NWFSC) survey. Each region has specific transect
    numbering schemes and boundary definitions.

    Parameters
    ----------
    region : np.number
        Region identifier (1, 2, or 3):

        - Region 1: Parallel transects to latitudes from south of SCB to west of Haida Gwaii
        - Region 2: Transects parallel to longitudes north of Haida Gwaii
        - Region 3: Parallel transects to latitudes west of Haida Gwaii

    Returns
    -------
    Tuple[np.number, np.number, List[np.number], List[np.number]]
        A tuple containing:

        - transect_start (np.number): Starting transect number for the region
        - transect_end (np.number): Ending transect number for the region
        - transect_lower_bound (List[np.number]): Lower boundary values for each transect
        - transect_upper_bound (List[np.number]): Upper boundary values for each transect

    Examples
    --------
    >>> start, end, lower, upper = transect_mesh_region_2019(1)
    >>> print(f"Region 1: transects {start} to {end}")
    Region 1: transects 1 to 119
    >>> print(f"Lower bounds: {lower[:3]}")  # First 3 values
    Lower bounds: [1.1, 2.1, 3.1]

    Notes
    -----
    The boundary values are encoded with decimal places that indicate spatial positions:

    - .1 indicates western boundary (for regions 1 and 3)
    - .4 indicates eastern boundary (for regions 1 and 3)
    - .6 indicates southern boundary (for region 2)
    - .9 indicates northern boundary (for region 2)
    """
    # Initialize variables
    transect_start = None
    transect_end = None
    transect_lower_bound = []  # W/S
    transect_upper_bound = []  # E/N

    # Region 1: parallel transects to latitudes from south of SCB to west of Haida Gwaii
    if region == 1:
        # ---- Southern-most transect
        transect_start = 1
        # ---- Northern-most transect
        transect_end = 152
        # ---- Western boundary
        transect_lower_bound = (
            [transect_start, 2]
            + list(np.arange(4, 79))
            + list(np.arange(80, 5, -2))
            + list(np.arange(89, 100))
            + list(np.arange(101, 106))
            + [120, 121]
            + list(np.arange(124, 128))
            + list(np.arange(129, 134, 2))
        )
        transect_lower_bound = [x + 0.1 for x in transect_lower_bound]
        # ---- Eastern boundary
        transect_upper_bound = (
            [transect_start, 2]
            + list(np.arange(4, 89))
            + list(np.arange(91, 106))
            + [152, 149, 145, 143, 142, 140, 135, 134, 133]
        )
        transect_upper_bound = [x + 0.4 for x in transect_upper_bound]
    # Region 2: transects parallel to longitudes north of Haida Gwaii
    elif region == 2:
        # ---- Western-most transect
        transect_start = 116
        # ---- Eastern-most transect
        transect_end = 119
        # ---- Southern boundary
        transect_lower_bound = [118.6, 117.6, 119.6, 116.6]
        # ---- Northern boundary
        transect_upper_bound = [118.9, 117.9, 119.9, 116.9]
    # Region 3: parallel transects to latitudes west of Haida Gwaii
    else:
        # ---- Southern-most transect
        transect_start = 106
        # ---- Northern-most transect
        transect_end = 128
        # ---- Western boundary
        transect_lower_bound = [116.6, 119.6] + [x + 0.1 for x in range(115, 105, -1)] + [126.1]
        # ---- Eastern boundary
        transect_upper_bound = [119.6] + [x + 0.4 for x in range(115, 105, -1)] + [128.1]

    return transect_start, transect_end, transect_lower_bound, transect_upper_bound
