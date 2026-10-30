"""
Ordinary kriging implementation and search-strategy utilities for spatial interpolation.

Implements isotropic and anisotropic ordinary kriging, local-search neighbourhood
selection, and the assembly of kriging weight systems from fitted variogram models.
"""

import inspect
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from .. import validators as val
from . import cropping, variogram_models as vgm
from .variogram import lag_distance_matrix

# Set warnings filter
warnings.simplefilter("always")


def uniform_search_strategy(
    sparse_radii: np.ndarray[int],
    valid_distances: np.ndarray[int],
    local_points: np.ndarray[float],
    k_min: int,
    k_max: int,
    search_radius: float,
    wr_indices: np.ndarray[int],
    oos_indices: np.ndarray[np.number],
    oos_weights: np.ndarray[float],
    **kwargs,
) -> tuple[np.ndarray[np.number], np.ndarray[np.number], np.ndarray[np.number]]:
    """
    Uniform nearest neighbors search strategy for ordinary kriging.

    Uniform extrapolation search strategy for finding (and weighting) k-th nearest points
    (relative to a reference coordinate) required for computing the lagged semivariogram in an
    adaptive approach.

    Parameters
    ----------
    sparse_radii : |np.ndarray[int]|
        Indices where there are fewer than ``k_min`` nearest neighbors.
    valid_distances : |np.ndarray[int]|
        The number of masked distance matrix values where extrapolation is required.
    local_points : |np.ndarray[float]|
        An array with the sorted distances (from nearest to furthest) relative to each point.
    k_min : int
        The minimum number of nearest neighbors required for including values for kriging within
        the search radius.
    k_max : int
        The maximum number of nearest neighbors required for including values for kriging detected
        within the search radius.
    search_radius : float
        The adaptive search radius that identifies the :math:`k`-nearest neighbors around each
        georeferenced value that are subsequently kriged.
    wr_indices : |np.ndarray[int]|
        Indices of within-radius (WR) (i.e. < ``k_max``) points.
    oos_indices : |np.ndarray[np.number]|
        Template array based on the size of the data input and ``k_min`` that will contain indices
        where extrapolation is required where there are fewer than ``k_min`` nearest neighbors.
    oos_weights : |np.ndarray[float]|
        Weights applied to extraplolated values.

    Returns
    -------
    Tuple[|np.ndarray[np.number]|, |np.ndarray[np.number]|, |np.ndarray[np.number]|]
        A tuple with updated values for ``wr_indices``, ``oos_indices``, and ``oos_weights`` via a
        search strategy that applies unconstrained and uniform extrapolation to out-of-sample (OOS)
        points.
    """
    # Index for areas with some valid points but fewer than k_min
    partial_indices = sparse_radii[valid_distances[sparse_radii] > 0]

    # Create boolean mask for within-range/sample points
    wr_mask = local_points[partial_indices, :k_max] < search_radius

    # Create temporary matrix for within-range samples
    wr_tmp = wr_indices[partial_indices].copy()
    # ---- Update temporary matrix by applying `wr_mask` for wr points
    wr_tmp[~wr_mask] = np.nan

    # Create temporary matrix for oos samples
    oos_tmp = wr_indices[partial_indices].copy()
    # ---- Update temporary matrix by applying `wr_mask` for oos points
    oos_tmp[wr_mask] = np.nan

    # Assign the OOS values to `oos_indices`
    oos_indices[partial_indices] = np.sort(oos_tmp[:, :k_min])

    # Apply the mask to the remaining `wr_indices` values
    wr_indices[partial_indices] = np.sort(wr_tmp[:, :k_max])

    # Get areas with no valid points within the search radius
    full_extrap_indices = sparse_radii[valid_distances[sparse_radii] == 0]
    if len(full_extrap_indices) > 0:
        # ---- Use all `k_min`-nearest neighbors for extrapolation
        oos_indices[full_extrap_indices] = wr_indices[full_extrap_indices, :k_min]
        wr_indices[full_extrap_indices] = np.nan
        # ---- Compute the OOS kriging weights
        oos_mean = np.apply_along_axis(np.nanmean, 1, local_points[full_extrap_indices, :k_min])
        # ---- Exponentiate the OOS mean
        oos_exp = np.exp(-oos_mean / search_radius)
        # ---- Update the OOS kriging weights
        oos_weights[full_extrap_indices] = oos_exp

    # Return Tuple
    return wr_indices, oos_indices, oos_weights


def search_radius_mask(
    distance_matrix: np.ndarray[float], search_radius: float
) -> np.ndarray[float]:
    """
    Generate a mask for the search radius to identify points within the search distance.

    Parameters
    ----------
    distance_matrix : |np.ndarray[float]|
        Distance matrix between points.
    search_radius : float
        Maximum search distance for identifying neighbors.

    Returns
    -------
    |np.ndarray[float]|
        Masked distance matrix where distances beyond search_radius are set to NaN.
    """
    # Create a copy to avoid modifying the original
    masked_matrix = distance_matrix.copy()
    # Set distances beyond search radius to NaN
    masked_matrix[masked_matrix > search_radius] = np.nan
    return masked_matrix


def count_within_radius(distance_matrix_masked: np.ndarray[float]) -> np.ndarray[int]:
    """
    Count the number of valid (non-NaN) distances for each point in the masked distance matrix.

    Parameters
    ----------
    distance_matrix_masked : |np.ndarray[float]|
        Masked distance matrix where invalid distances are NaN.

    Returns
    -------
    |np.ndarray[int]|
        Array containing the count of valid neighbors for each point.
    """
    # Count non-NaN values along each row
    return np.sum(~np.isnan(distance_matrix_masked), axis=1)


def adaptive_search_radius(
    distance_matrix: np.ndarray[float],
    k_min: int,
    k_max: int,
    search_radius: float,
    search_strategy: Callable = uniform_search_strategy,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray[np.number], np.ndarray[np.number], np.ndarray[float]]:
    r"""
    Find the indices of the k-th nearest points.

    Find the indices of the k-th nearest points (relative to a reference coordinate) required
    for computing the lagged semivariogram.

    Parameters
    ----------
    distance_matrix : |np.ndarray[float]|
        An array/matrix that includes the distances of each mesh points from every
        georeferenced along-transect interval.
    k_min : int
        The minimum number of nearest neighbors required for including values for kriging within
        the search radius.
    k_max : int
        The maximum number of nearest neighbors required for including values for kriging detected
        within the search radius.
    search_radius : float
        The adaptive search radius that identifies the *k*-nearest neighbors around each
        georeferenced value that are subsequently kriged.
    search_strategy : Callable, default=:func:`echopop.nwfsc_feat.spatial.uniform_search_strategy`
        A `Callable` function that defaults to using a uniform search strategy where out-of-sample
        points are extrapolated using equal weights. User-defined search strategies can be defined
        by parsing any of the internally computed variables:

        - ``sparse_radii (|np.ndarray[int]|)`` \n
        Indices where there are fewer than ``k_min`` nearest neighbors.

        - ``valid_distances (|np.ndarray[int]|)`` \n
        The number of masked distance matrix values where extrapolation is required.

        - ``local_points (|np.ndarray[float]|)``\n
        An array with the sorted distances (from nearest to furthest) relative to each point.

        - ``distance_matrix_masked (|np.ndarray[float]|)`` \n
        An array with the search-radius-masked nearest neighbor distances.

        - ``nearby_indices (|np.ndarray[int]|)`` \n
        Indices of points that require extrapolation.

        - ``k_min (int)`` \n
        The minimum number of nearest neighbors required for including values for kriging within
        the search radius.

        - ``k_max (int)`` \n
        The maximum number of nearest neighbors required for including values for kriging detected
        within the search radius.

        - ``search_radius (float)`` \n
        The adaptive search radius that identifies the *k*-nearest neighbors around each
        georeferenced value that are subsequently kriged.

        - ``wr_indices (|np.ndarray[int]|)`` \n
        Indices of within-radius (WR) (i.e. < ``k_max``) points.

        - ``oos_indices (|np.ndarray[np.number]|)`` \n
        Template array based on the size of the data input and ``k_min`` that will contain indices
        where extrapolation is required where there are fewer than ``k_min`` nearest neighbors.

        - ``oos_weights (|np.ndarray[float]|)``
        Weights applied to extraplolated values.

    Returns
    -------
    Tuple[|np.ndarray[float]|, |np.ndarray[int]|, |np.ndarray[int]|, |np.ndarray[float]|]
        A tuple with arrays representing the k-nearest distances, within-radius (WR) indices,
        out-of-sample (OOS) indices that indicate which points will be extrapolated, and the
        OOS weights applied to values that are extrapolated.
    """
    # Generate the search radius mask
    distance_matrix_masked = search_radius_mask(distance_matrix, search_radius)

    # Identify mesh points that require extrapolation
    # ---- Count the number of values within the search radius
    valid_distances = count_within_radius(distance_matrix_masked)
    # ---- Identify rows where the number of valid points are less than `k_min`
    sparse_radii = np.hstack(np.where(valid_distances < k_min))

    # Calculate the closest grid points and their indices
    # ---- Closest indices
    local_indices = distance_matrix.argsort(axis=1)
    # ---- Map the distance matrix to the local indices
    local_points = np.take_along_axis(distance_matrix, local_indices, axis=1)

    # Initialize matrices
    # ---- Within-radius (WR) samples
    wr_indices = local_indices[:, :k_max].astype(float)
    # ---- Out-of-sample (OOS) indices
    oos_indices = np.full((len(valid_distances), k_min), np.nan)
    # ---- OOS weights
    oos_weights = np.ones(len(valid_distances))

    # For points where there are fewer than `k_min` points within the search radius, extrapolate
    if np.size(sparse_radii) > 0:
        # ---- Extract the indices requiring extrapolation
        nearby_indices = local_indices[sparse_radii][:, :k_min]
        # ---- Update local points
        # -------- Fill NaN values
        local_points[sparse_radii, k_min:] = np.nan
        wr_indices[sparse_radii, k_min:] = np.nan
        # ---- Collect the available parameters for defining the search strategy
        search_params = {
            "sparse_radii": sparse_radii,
            "valid_distances": valid_distances,
            "local_points": local_points,
            "distance_matrix_masked": distance_matrix_masked,
            "nearby_indices": nearby_indices,
            "k_min": k_min,
            "k_max": k_max,
            "search_radius": search_radius,
            "wr_indices": wr_indices,
            "oos_indices": oos_indices,
            "oos_weights": oos_weights,
        }

        # ---- Apply search strategy to update the WR/OOS indices, and OOS weights
        if isinstance(search_strategy, Callable):
            wr_indices, oos_indices, oos_weights = search_strategy(**search_params)

    # Exclude surplus neighbors once the minimum in-radius count is satisfied.
    range_grid = local_points[:, :k_max].copy()
    exclude = (valid_distances >= k_min)[:, None] & (range_grid > search_radius)
    wr_indices[exclude] = np.nan
    range_grid[exclude] = np.nan

    # Return output
    return range_grid, wr_indices, oos_indices, oos_weights


def ordinary_kriging_matrix(
    local_distance_matrix: np.ndarray[float], variogram_parameters: dict[str, Any]
) -> np.ndarray[float]:
    r"""
    Calculate the kriging covariance matrix.

    Calculate thr kriging covariance matrix used for ordinary kriging.

    Parameters
    ----------
    local_distance_matrix : |np.ndarray[float]|
        A 2D array with the numeric distances between points.
    variogram_parameters : Dict[str, Any]
        Dictionary describing the variogram model and parameters, such as:
        - 'model' (str or list of str): variogram model names (e.g., 'bessel', 'exponential'),
        - 'nugget' (float): nugget effect,
        - 'sill' (float): sill parameter,
        - 'correlation_range' (float): range parameter,
        - 'hole_effect_range' (float): hole effect range,
        - 'decay_power' (float): power parameter for power variogram.

    Returns
    -------
    |np.ndarray[float]|
        A 2D array representing the (n+1, n+1) kriging covariance matrix including the unbiasedness
        constraint.

    Notes
    -----
    Ordinary Kriging assumes a constant but unknown mean, requiring the
    kriging weights to sum to 1:

    .. math::

        \\sum_{i=1}^n \\lambda_i = 1

    This leads to the augmented linear system:

    .. math::

        \\begin{bmatrix}
        K & \\mathbf{1} \\\\
        \\mathbf{1}^\\top & 0
        \\end{bmatrix}
        \\begin{bmatrix}
        \\boldsymbol{\\lambda} \\\\
        \\mu
        \\end{bmatrix}
        =
        \\begin{bmatrix}
        \\mathbf{k} \\\\
        1
        \\end{bmatrix}

    Where:

    - :math:`K` is the :math:`n \\times n` covariance or variogram matrix between known points
    - :math:`\\mathbf{k}` is the vector of covariances (or variograms) between the known points and
      the interpolation location
    - :math:`\\boldsymbol{\\lambda}` is the vector of kriging weights
    - :math:`\\mu` is the Lagrange multiplier enforcing the unbiasedness constraint
    - :math:`\\mathbf{1}` is a column vector of ones

    The solution yields the kriging weights :math:`\\lambda_i` and the multiplier :math:`\\mu`.
    """
    # Calculate the covariance/kriging matrix (without the constant term)
    kriging_matrix_initial = vgm.compute_variogram(
        distance_lags=local_distance_matrix, variogram_parameters=variogram_parameters
    )

    # Expand the covariance/kriging matrix with a constant
    # ---- In Ordinary Kriging, this should be '1'
    kriging_matrix = np.concatenate(
        [kriging_matrix_initial, np.ones((local_distance_matrix.shape[0], 1))], axis=1
    )

    # Add column and row of ones for Ordinary Kriging
    kriging_matrix = np.concatenate(
        [kriging_matrix, np.ones((1, local_distance_matrix.shape[0] + 1))], axis=0
    )

    # Diagonal fill (0.0)
    np.fill_diagonal(kriging_matrix, 0.0)

    return kriging_matrix


def kriging_lambda(
    aspect_ratio: float,
    lagged_semivariogram: np.ndarray,
    kriging_covariance_matrix: np.ndarray,
):
    r"""
    Solve the kriging system.

    Solve the Kriging system (set of linear equations) using a truncated singular value
    decomposition (SVD) of the kriging covariance matrix.

    Parameters
    ----------
    aspect_ratio : float
        Ratio of the correlation range along the minor axis to major axis in the principal
        direction of anisotropy. This defines the strength of the directional correlation in the
        variable being kriged. Values close to 1 indicate nearly isotropic correlation, while
        smaller values indicate stronger elongation along the major axis.

        .. math::

            \text{aspect_ratio} = \frac{r_\text{minor}}{r_\text{major}}

        This serves as the directional threshold for retaining singular values in the
        decomposition, given as a fraction of the largest singular value. Singular values below
        this threshold are discarded to regularize the solution.

    lagged_semivariogram: np.array[float]
        Right-hand side vector of the kriging system comprising the local lagged semivariogram.
        This corresponds to the semivariogram estimates between prediction location and observed
        sample points, with an additional `1` appended for the unbiasedness constrain when using
        ordinary kriging.

    kriging_covariance_matrix : np.array[float]
        A 2D array representing the full kriging covariance matrix, which is typically of size
        (n+1, n+1) when using ordinary kriging. This includes the variogram matrix between sample
        points. When ordinary kriging is being used, an additional column and row of `1` and a
        bottom-right `0` are also included to enforce the unbiasedness constraint:

        .. math::

            \\begin{bmatrix}
            \\gamma_{ij} & 1 \\\\
            1^T & 0
            \\end{bmatrix}

    Returns
    -------
    |np.ndarray[float]|
        A 1D array of length :math:`n + 1`, where the first :math:`n` values are the kriging
        weights :math:`\\lambda_i`, and the final element is the Lagrange multiplier :math:`\\mu`
        enforcing the unbiasedness constraint.

    Notes
    -----
    This function performs a numerically stable inversion of the kriging system using a truncated
    SVD:

    .. math::

        K^+ = V \\Sigma^{-1} U^T

    where only the singular values satisfying:

    .. math::

        \\frac{\\sigma_i}{\\sigma_1} > \\text{aspect_ratio}

    are retained.

    The final weights are given by:

    .. math::

        \\boldsymbol{\\lambda} = K^+ \\cdot \\mathbf{k}

    where :math:`\\mathbf{k}` is the lagged semivariogram vector to the target location.
    """
    # Singular value decomposition (SVD)
    # ---- U: left singular vectors (directions of maximum variance)
    # ---- Sigma: singular values (amount of variance captured by each singular vector, U)
    # ---- VH: conjugate transpose of the right singular vectors
    U, Sigma, VH = np.linalg.svd(kriging_covariance_matrix, full_matrices=True)

    # Create Sigma mask informed by the ratio-threshold
    # ---- The ratio between all singular values and their respective
    # ---- maximum is used to apply a mask that informs which values
    # ---- are used to tabulate the kriging weights (aka lambda)
    Sigma_mask = np.abs(Sigma / Sigma[0]) > aspect_ratio

    # Inverse masked semivariogram (K)
    K_inv = np.matmul(
        np.matmul(VH.T[:, Sigma_mask], np.diag(1.0 / Sigma[Sigma_mask])), U[:, Sigma_mask].T
    )

    # Calculate kriging weights (lambda)
    return np.dot(K_inv, lagged_semivariogram)


def parse_stacked_kriging_array(
    stacked_array: np.ndarray[np.number], k_min: int, k_max: int, **kwargs
) -> tuple[np.ndarray[float], np.ndarray[int], np.ndarray[float], np.ndarray[float]]:
    """
    Parse the horizontally stacked kriged arrays.

    Helper function for parsing the horizontally stacked array when interpolating points via
    kriging.

    Parameters
    ----------
    stacked_array : |np.ndarray[np.number]|
        A 1D array that is encoded with various parameters including the out-of-sample weights,
        extrapolation and interpolation indices, local lagged semivariogram, and range distance
        estimates.
    k_min : int
        The minimum number of nearest neighbors required for including values for kriging within
        the search radius.
    k_max : int
        The maximum number of nearest neighbors required for including values for kriging detected
        within the search radius.

    Returns
    -------
    Tuple[|np.ndarray[float]|, |np.ndarray[int]|, |np.ndarray[float]|, |np.ndarray[float]|]
        A tuple comprising the out-of-sample weights, composite indices for values that will be
        either interpolated or extrapolated, the local lagged semivariogram, and the range distance
        values.
    """
    # Break up the stacked index
    # ---- Extrapolation/out-of-sample (OOS) weight
    outside_weight = stacked_array[0]
    # ---- Get the Within- and out-of-sample indices
    interp_indices, extrap_indices = (
        stacked_array[s:e][~np.isnan(stacked_array[s:e])].astype(int)
        for s, e in [(1, k_max + 1), (k_max + 1, k_max + 1 + k_min)]
    )
    # ---- Combine the interpolated and extrapolated indices into a composite index
    composite = np.concatenate([interp_indices, extrap_indices])
    # ---- Extract the local lagged semivariogram and range distance values
    M2_vario, range_vals = (
        stacked_array[s:e][~np.isnan(stacked_array[s:e])]
        for s, e in [(-(2 * k_max) - 1, -k_max), (-k_max, None)]
    )

    # Return the values
    return outside_weight, composite, M2_vario, range_vals


def kriging_point_estimator(
    kriging_array: np.ndarray[np.number],
    data_array: np.ndarray[float],
    kriging_parameters: dict[str, Any],
    variogram_parameters: dict[str, Any],
) -> np.ndarray[float]:
    r"""
    Interpolate kriged estimates.

    Interpolate value at a specified point via kriging.

    Parameters
    ----------
    kriging_array: |np.ndarray[np.number]|
        Horizontally stacked 1D array containing kriging indices and local semivariogram values.
        Expected to include:

        - Outside weight scalar

        - Within- and out-of-sample indices

        - Lagged semivariogram values (M2)

        - Local range estimates

    data_array: |np.ndarray[float]|
        A 2D array of shape (n, 3), where columns represent [x-coordinates, y-coordinate,
        variable].
    kriging_parameters: dict
        Dictionary of kriging parameters, must contain keys:

        - 'k_min' (int): minimum number of neighbors

        - 'k_max' (int): maximum number of neighbors

        - 'search_radius' (float): radius to consider neighbors

        - 'aspect_ratio' (float): ratio of the correlation range along the minor axis to major axis
          in the principal direction of anisotropy

    variogram_parameters : Dict[str, Any]
        Dictionary describing the variogram model and parameters, such as:

        - 'model' (str or list of str): variogram model names (e.g., 'bessel', 'exponential')

        - 'nugget' (float): nugget effect

        - 'sill' (float): sill parameter

        - 'correlation_range' (float): range parameter

        - 'hole_effect_range' (float): hole effect range

        - 'decay_power' (float): power parameter for power variogram

    Returns
    -------
    |np.ndarray[float]|
        A 1D array of length 3 containing:

        - point estimate (float): the kriging predicted value at the target location

        - kriged variance (float): variance estimate associated with the kriging prediction

        - sample variance (float): coefficient of variation based variance estimate

          set to NaN if the point estimate is effectively zero to avoid division by zero

    Notes
    -----
    - The function assumes ordinary kriging with an unbiasedness constraint.

    - The sample variance is computed as:

      .. math::

        \text{sample_variance} = \frac{
            \\sqrt{\\text{kriged_variance} \\times \\mathrm{var}(Z)}
        }{
            |\\hat{Z}|
        }

      where :math:`\\hat{Z}` is the kriging estimate and :math:`Z` the observed variable values.

    - If the point estimate is zero or extremely close to zero, the sample variance is
      undefined and returned as NaN.
    """
    # Break up the stacked index
    outside_weight, composite_index, M2_vario, range_vals = parse_stacked_kriging_array(
        kriging_array, **kriging_parameters
    )

    # Index the data array
    data_array_idx = data_array[composite_index]

    # Set extrapolated variable values to 0.0
    # Only apply the mask if range_vals has the same length as data_array_idx
    if len(range_vals) == len(data_array_idx):
        data_array_idx[range_vals > kriging_parameters["search_radius"], 2] = 0.0

    # Calculate the local distance matrix
    local_distance_matrix, _ = lag_distance_matrix(
        coordinates_1=data_array_idx[:, 0], coordinates_2=data_array_idx[:, 1], self=True
    )

    # Compute the covariance matrix using ordinary kriging
    kriging_covariance_matrix = ordinary_kriging_matrix(local_distance_matrix, variogram_parameters)

    # Calculate the kriging weights (lambda)
    kriging_weights = kriging_lambda(
        kriging_parameters["aspect_ratio"], M2_vario, kriging_covariance_matrix
    )

    # Calculate the point estimate
    point_estimate = (
        kriging_weights[: len(composite_index)] * data_array_idx[:, 2]
    ).sum() * outside_weight

    # Calculate the kriged variance
    kriged_variance = kriging_weights.dot(M2_vario)

    # Calculate the sample variance and CV
    if abs(point_estimate) < np.finfo(float).eps:
        sample_variance = np.nan
    else:
        sample_variance = np.sqrt(kriged_variance * np.var(data_array_idx[:, 2], ddof=1)) / np.abs(
            point_estimate
        )

    # Return the point estimates, kriged/sample variances
    return np.array([point_estimate, kriged_variance, sample_variance])


def ordinary_kriging(
    transects: pd.DataFrame,
    kriging_mesh: pd.DataFrame,
    coordinate_names: tuple[str, str],
    variable: str,
    kriging_parameters: dict[str, Any],
    variogram_parameters: dict[str, Any],
    adaptive_search_strategy: Callable = uniform_search_strategy,
) -> np.ndarray[float]:
    r"""
    Apply ordinary kriging for spatial interpolation.

    Use ordinary kriging to interpolate georeferenced data onto a grid.

    Parameters
    ----------
    transects : pandas.DataFrame
        Dataframe including georeferenced data
    kriging_mesh : pandas.DataFrame
        Grid data that has been transformed
    coordinate_names : Tuple[str, str]
        Names of the coordinate columns when using DataFrames. Expected format: ``(x_col, y_col)``.
    variable : str
        The variable used for computing the empirical variogram (e.g. 'biomass_density'), which
        must exist as a column in 'transects'.
    kriging_parameters: dict
        Dictionary of kriging parameters, must contain keys:

        - ``k_min (int)``: minimum number of neighbors

        - ``k_max (int)``: maximum number of neighbors

        - ``search_radius (float)``: radius to consider neighbors

        - ``anisotropy (float)``: truncation threshold for singular values in SVD solver.

    variogram_parameters : Dict[str, Any]
        Dictionary describing the variogram model and parameters, such as:

        - ``model`` (str or list of str): variogram model names (e.g., 'bessel', 'exponential')

        - ``nugget`` (float): nugget effect

        - ``sill`` (float): sill parameter

        - 'correlation_range' (float): range parameter

        - 'hole_effect_range' (float): hole effect range

        - 'decay_power' (float): power parameter for power variogram

    adaptive_search_strategy : Callable, \
        default = :func:`echopop.geostatistics.uniform_search_strategy`
        A `Callable` function that defaults to using a uniform search strategy where out-of-sample
        points are extrapolated using equal weights. User-defined search strategies can be defined
        by parsing any of the internally computed variables:

        - ``sparse_radii (|np.ndarray[int]|)`` \n
          Indices where there are fewer than ``k_min`` nearest neighbors.

        - ``valid_distances (|np.ndarray[int]|)`` \n
          The number of masked distance matrix values where extrapolation is required.

        - ``local_points (|np.ndarray[float]|)``\n
          An array with the sorted distances (from nearest to furthest) relative to each point.

        - ``distance_matrix_masked (|np.ndarray[float]|)`` \n
          An array with the search-radius-masked nearest neighbor distances.

        - ``nearby_indices (|np.ndarray[int]|)`` \n
          Indices of points that require extrapolation.

        - ``k_min (int)`` \n
          The minimum number of nearest neighbors required for including values for kriging within
          the search radius.

        - ``k_max (int)`` \n
        The maximum number of nearest neighbors required for including values for kriging detected
        within the search radius.

        - ``search_radius (float)`` \n
          Maximum distance (in coordinate units) from the target location to consider neighbors for
          adaptive kriging. Only points within this radius are eligible for inclusion in the
          *k*-nearest neighbor search. The adaptive search radius that identifies the *k*-nearest
          neighbors around each georeferenced value that are subsequently kriged.

        - ``wr_indices (|np.ndarray[int]|)`` \n
          Indices of within-radius (WR) (i.e. < ``k_max``) points.

        - ``oos_indices (|np.ndarray[np.number]|)`` \n
          Template array based on the size of the data input and ``k_min`` that will contain indices
          where extrapolation is required where there are fewer than ``k_min`` nearest neighbors.

        - ``oos_weights (|np.ndarray[float]|)``
          Weights applied to extraplolated values.

    Returns
    -------
    A 1D array of the same length as ``transect_df`` containing:

        - point estimate (``float``): the kriging predicted value at the target location

        - kriged variance (``float``): variance estimate associated with the kriging prediction

        - sample variance (``float``): coefficient of variation based variance estimate,
          set to NaN if the point estimate is effectively zero to avoid division by zero.

    Notes
    -----
    The search radius (``search_radius``) should be large enough to incluade at least the ``k_min``
    points; otherwise, the local covariance matrix ay be underdetermined. If it is too large, local
    neighborhoods may approach global kriging, reducing local resolution.
    """
    # Generate the distance matrix for all mesh points relative to all transect coordinates
    distance_matrix, _ = lag_distance_matrix(
        coordinates_1=kriging_mesh, coordinates_2=transects, coordinate_names=coordinate_names
    )

    # Run the adaptive search window to identify which points have to be re-weighted to account
    # for extrapolation
    range_grid, inside_indices, outside_indices, outside_weights = adaptive_search_radius(
        distance_matrix,
        search_strategy=adaptive_search_strategy,
        **kriging_parameters,
    )

    # Calculate the lagged semivariogram (M20)
    local_variogram = np.apply_along_axis(
        vgm.compute_variogram, 1, range_grid, variogram_parameters
    )

    # Preserve padding for models that return finite values at NaN lags.
    local_variogram[np.isnan(range_grid)] = np.nan

    # Append 1.0 for the ordinary kriging assumptions (M2)
    local_variogram_M2 = np.sort(
        np.hstack((local_variogram, np.ones((local_variogram.shape[0], 1))))
    )

    # Stack all of the kriging parameters
    full_stack = np.hstack(
        (
            outside_weights.reshape(-1, 1).tolist(),
            inside_indices.tolist(),
            outside_indices.tolist(),
            local_variogram_M2.tolist(),
            range_grid.tolist(),
        )
    )

    # Extract x- and y-coordinates, and variable_data
    data_array = transects[list(coordinate_names) + [variable]].to_numpy()

    # Kriging point estimator (currently assumes ordinary kriging)
    kriged_values = np.apply_along_axis(
        kriging_point_estimator, 1, full_stack, data_array, kriging_parameters, variogram_parameters
    )

    # Return the 2D array
    return kriged_values


def project_kriging_results(
    kriged_estimates: np.ndarray[float],
    kriging_mesh: pd.DataFrame,
    transects: pd.DataFrame,
    variable: str,
    default_mesh_cell_area: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Project kriged results.

    Project the kriged results over the defined mesh.

    Parameters
    ----------
    kriged_estimates : |np.ndarray[float]|, shape (n, 3)
        Kriged values with columns: [estimate, kriging variance, sample variance]
    kriging_mesh : |pd.DataFrame|
        Grid data that has been transformed
    transects : |pd.DataFrame|
        Dataframe including georeferenced data
    coordinate_names : Tuple[str, str]
        Names of the coordinate columns when using DataFrames. Expected format: ``(x_col, y_col)``.
    variable : str
        The variable used for computing the empirical variogram (e.g. 'biomass_density'), which
        must exist as a column in ``'transects'``.
    default_mesh_cell_area: Optional[float], default=None
        Default cell area for kriging mesh nodes when no area is provided within the kriging
        mesh |pd.DataFrame|.

    Returns
    -------
    Tuple[|pd.DataFrame|, float]
        A tuple containing the output mesh |pd.DataFrame| with columns including the kriged
        estimates and variance, sample variance, and cell coefficient of variation (CV). The other
        value is the overall CV computed for the entire kriging mesh.
    """
    # Create copy
    mesh_results = kriging_mesh.copy()

    # Get the adjusted areas
    if "fraction" in kriging_mesh.columns and default_mesh_cell_area is not None:
        mesh_results["area"] = mesh_results["fraction"] * default_mesh_cell_area
    elif "area" not in mesh_results.columns:
        # If no area column exists, create a default area of 1.0 for each cell
        if default_mesh_cell_area is not None:
            mesh_results["area"] = default_mesh_cell_area
        else:
            mesh_results["area"] = 1.0

    # Calculate the (global) survey variance
    survey_variance = transects[variable].var()

    # Distribute estimates over an area
    survey_estimate = np.nansum(kriged_estimates[:, 0] * mesh_results["area"])

    # Add the kriged array values
    # ---- Apply kriged estimateadjustment if any values are negative
    if any(kriged_estimates[:, 0] < 0.0):
        # -------- Get count
        N0 = np.sum(kriged_estimates[:, 0] < 0.0)
        # -------- Truncate
        kriged_estimates[kriged_estimates[:, 0] < 0.0, 0] = 0.0
        # -------- Print warning
        warnings.warn(
            f"{N0} invalid kriged estimates (< 0.0) found. These have been replaced with '0.0'; "
            f"however, this truncation may distort error distributions. If this is problematic, "
            f"adjust the choice of variogram and kriging algorithm parameterization.",
            stacklevel=2,
        )
    # ---- Kriged estimates
    mesh_results["estimate"] = kriged_estimates[:, 0]
    # ---- Kriged variance
    mesh_results["kriged_variance"] = kriged_estimates[:, 1]
    # ---- Sample variance
    mesh_results["sample_variance"] = kriged_estimates[:, 2]
    # ---- Mesh node CV
    mesh_results["cell_cv"] = (
        mesh_results["area"].mean()
        * np.sqrt(kriged_estimates[:, 1].dot(survey_variance))
        / survey_estimate
        * np.sqrt(len(kriged_estimates[:, 1]))
    )

    # Calculate the global CV
    global_cv = (
        np.sqrt(np.nansum(kriged_estimates[:, 1] * mesh_results["area"] ** 2) * survey_variance)
        / survey_estimate
    )

    # Return the DataFrame and global CV estimate
    return mesh_results, global_cv


class Kriging:
    r"""
    Class for kriging spatially distributed data.

    Class for performing ordinary kriging to predict population values and other metrics at
    un-sampled locations defined by a mesh grid.

    Kriging is a geostatistical interpolation technique that provides the Best Linear Unbiased
    Predictor (BLUP) for spatial data. The method uses the spatial correlation structure,
    characterized by a semivariogram model, to optimally weight nearby observations when predicting
    values at unsampled locations.

    For ordinary kriging, the prediction at location x₀ is given by:

    .. math::
        \\mathbf{z}^*(\\mathbf{u}) = \\sum_{b=1}^n \\lambda_b(\\mathbf{u}) z(\\mathbf{u}_b)

    where :math:`n` is the number of observed locations, :math:`z(\\mathbf{u}_b)` are the known
    values of a spatially varying variable at locations :math:`\\mathbf{u}_b`, and
    :math:`\\lambda_b(\\mathbf{u})` are the kriging weights assigned to all known locations
    :math:`\\mathbf{u}`. This kriging approach is subject to the unbiasedness constraint where:

    .. math::
        \\sum\\limits_{b=1}^n \\lambda_b = 1

    The kriging weights :math:`\\lambda_b` are obtained by solving the kriging system:

    .. math::
        \\mathbf{\\Gamma} \\hat{\\mathbf{\\lambda}} = \\hat{\\mathbf{\\gamma}}_\\mathbf{u}

    where:

    .. math::

        \\underbrace{
            \\begin{bmatrix}
                \\gamma_{1,1} & \\cdots & \\gamma_{1,n} & 1 \\\\
                \\vdots & \\ddots & \\vdots & \\vdots \\\\
                \\gamma_{n,1} & \\cdots & \\gamma_{n,n} & 1 \\\\
                1 & \\cdots & 1 & 0
            \\end{bmatrix}
        }_{\\mathbf{\\Gamma}}
        \\underbrace{
            \\begin{bmatrix}
                \\lambda_1 \\\\
                \\vdots \\\\
                \\lambda_n \\\\
                \\mu
            \\end{bmatrix}
        }_{\\hat{\\mathbf{\\lambda}}} =
        \\underbrace{
            \\begin{bmatrix}
                \\gamma(\\mathbf{u}_1 - \\mathbf{u}) \\\\
                \\vdots \\\\
                \\gamma(\\mathbf{u}_n - \\mathbf{u}) \\\\
                1
            \\end{bmatrix}
        }_{\\mathbf{\\hat{\\gamma}}_\\mathbf{u}}\\\\

    Here, :math:`\\gamma_{i,j}` is the semivariance between locations :math:`i` and :math:`j`,
    :math:`\\gamma_{i,\\mathbf{u}} = \\gamma(\\mathbf{u}_i - \\mathbf{u})` is the semivariance
    between data point :math:`i` and target location :math:`\\mathbf{u}`, and
    :math:`\\mu` is the Lagrange multiplier enforcing the unbiasedness constraint [1]_, [2]_.

    Parameters
    ----------
    mesh : |pd.DataFrame|
        DataFrame containing the mesh grid used for interpolating the values from ``data``. This
        DataFrame must contain coordinate columns as specified in ``coordinate_names``.

        Optional columns include:

        - ``area`` (float): Cell areas in square nautical miles for projection calculations

        - ``fraction`` (float): Fraction of cell area if using default cell areas

    coordinate_names : Tuple[str, str], default=("x", "y")
        Names of the coordinate columns that are shared between input data and mesh. Format:
        (horizontal_coordinate, vertical_coordinate).

    kriging_params : Dict[str, Any]
        Dictionary containing kriging parameters required for interpolation:

        - ``aspect_ratio (float)``: Ratio of minor to major axis correlation ranges for anisotropy
          handling. Values near 1 indicate isotropy, smaller values indicate directional elongation.

        - ``k_min (int)``: Minimum number of nearest neighbors for kriging (typically 3-8).

        - ``k_max (int)``: Maximum number of nearest neighbors for kriging (typically 8-20).

        - ``search_radius (float)``: Maximum distance for neighbor search in coordinate units.

    variogram_params : Dict[str, Any]
        Dictionary containing variogram model parameters:

        - ``model`` (``str`` or ``List[str]``): Variogram model (e.g., ``'exponential'``,
          ``'gaussian'``, ``'spherical'``, or composite models like ``['bessel', 'exponential']``).

        - ``nugget`` (``float``): Nugget effect representing micro-scale variability.

        - ``sill`` (``float``): Total variance (nugget + partial sill).

        - ``correlation_range`` (``float``): Correlation length scale.

        - Additional parameters depending on model choice (e.g., ``hole_effect_range``,
          ``decay_power``).

    Attributes
    ----------
    mesh : |pd.DataFrame|
        Original mesh grid for interpolation.
    coordinate_names : Tuple[str, str]
        Coordinate column names.
    kriging_params : Dict[str, Any]
        Kriging parameters dictionary.
    variogram_params : Dict[str, Any]
        Variogram model parameters dictionary.
    mesh_cropped : |pd.DataFrame| or None
        Cropped mesh grid to prevent extrapolation beyond survey boundaries.
    survey_cv : float or None
        Overall survey coefficient of variation after kriging.

    Methods
    -------
    crop_mesh
        Crop the mesh grid to prevent extrapolation beyond survey boundaries.
    krige
        Perform ordinary kriging interpolation and project results onto the mesh grid.
    register_search_strategy
        Register a custom adaptive search strategy function.
    list_search_strategies
        List all available adaptive search strategy names.

    Examples
    --------
    Basic kriging workflow for fisheries acoustic data:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from echopop.nwfsc_feat.kriging import Kriging
    >>>
    >>> # Create sample transect data
    >>> transects = pd.DataFrame({
    ...     'longitude': np.random.uniform(-125, -120, 100),
    ...     'latitude': np.random.uniform(40, 45, 100),
    ...     'biomass_density': np.random.exponential(50, 100)
    ... })
    >>>
    >>> # Create kriging mesh
    >>> lon_grid, lat_grid = np.meshgrid(
    ...     np.linspace(-125, -120, 20),
    ...     np.linspace(40, 45, 20)
    ... )
    >>> mesh = pd.DataFrame({
    ...     'longitude': lon_grid.flatten(),
    ...     'latitude': lat_grid.flatten(),
    ...     'area': np.full(400, 1.0)  # 1 nmi² cells
    ... })
    >>>
    >>> # Define kriging parameters
    >>> kriging_params = {
    ...     'k_min': 4,
    ...     'k_max': 12,
    ...     'search_radius': 10.0,
    ...     'aspect_ratio': 0.8
    ... }
    >>>
    >>> # Define variogram parameters (from fitted model)
    >>> variogram_params = {
    ...     'model': 'exponential',
    ...     'nugget': 0.1,
    ...     'sill': 2.5,
    ...     'correlation_range': 8.0
    ... }
    >>>
    >>> # Initialize kriging object
    >>> krig = Kriging(
    ...     mesh=mesh,
    ...     coordinate_names=('longitude', 'latitude'),
    ...     kriging_params=kriging_params,
    ...     variogram_params=variogram_params
    ... )
    >>>
    >>> # Perform kriging interpolation
    >>> results = krig.krige(
    ...     transects=transects,
    ...     variable='biomass_density',
    ...     extrapolate=False  # Requires mesh cropping first
    ... )

    Notes
    -----
    **Key Features:**

    1. **Adaptive Search Strategy**: Uses k-nearest neighbor search with fallback extrapolation for
    sparse data regions.

    2. **Numerical Stability**: Employs truncated SVD for matrix inversion to handle
    ill-conditioned covariance matrices.

    3. **Anisotropy Support**: Handles directional correlation through aspect ratio
    parameterization.

    4. **Mesh Cropping**: Prevents extrapolation beyond survey boundaries using convex hull or
    custom boundary functions.

    **Typical Parameter Ranges:**

    - ``k_min``: 3-8 (minimum for stable estimates)
    - ``k_max``: 8-20 (balance between locality and stability)
    - ``search_radius``: 2-5× correlation range
    - ``aspect_ratio``: 0.1-1.0 (lower values = stronger anisotropy)

    **Performance Considerations:**

    - Computational complexity: :math:`\\mathcal{O}(n \\times m \\times k^3)` where :math:`n` = mesh
      points, :math:`m` = data points, :math:`k` = neighbors
    - Memory usage scales with mesh size and neighbor count
    - Large search radii increase computational cost but improve spatial continuity

    References
    ----------
    .. [1] Cressie, N.A.C. (1993). *Statistics for Spatial Data*. John Wiley & Sons.
    .. [2] Chilès, J.P., and Delfiner, P. (2012). *Geostatistics: Modeling Spatial
       Uncertainty*. 2nd ed. John Wiley & Sons.
    """

    def __new__(
        cls,
        mesh: pd.DataFrame,
        kriging_params: dict[str, Any],
        variogram_params: dict[str, Any],
        coordinate_names: tuple[str, str] = ("x", "y"),
    ):
        """Create class object. If input validation fails, then the object will not be created."""
        # Validate
        try:
            # ---- Check
            valid_args = val.ValidateKrigingClass.create(
                **dict(
                    mesh=mesh,
                    kriging_params=kriging_params,
                    variogram_params=variogram_params,
                    coordinate_names=coordinate_names,
                )
            )
        # Break creation
        except (ValidationError, Exception) as e:
            raise e from None

        # Create instance
        self = super().__new__(cls)

        # Update attributes
        self.__dict__.update(valid_args)

        # Initialize survey CV
        self.survey_cv = None

        # Generate
        return self

    def __init__(
        self,
        mesh: pd.DataFrame,
        kriging_params: dict[str, Any],
        variogram_params: dict[str, Any],
        coordinate_names: tuple[str, str],
    ):

        # Initialize variables
        self.mesh_cropped = None

    # Search strategy registry
    _search_strategies = {
        "uniform": uniform_search_strategy,
    }

    # Built-in strategies (immutable)
    _builtin_strategies = ["uniform"]

    @classmethod
    def register_search_strategy(cls, name: str, strategy: Callable) -> None:
        """Register a custom search strategy function.

        Parameters
        ----------
        name : str
            Name identifier for the strategy. Cannot overwrite built-in strategies.
        strategy : Callable
            Function implementing the search strategy

        Raises
        ------
        ValueError
            If attempting to overwrite a built-in strategy
        """
        if name in cls._builtin_strategies:
            raise ValueError(
                f"Cannot overwrite built-in strategy '{name}'. "
                f"Built-in strategies: {cls._builtin_strategies}"
            )
        cls._search_strategies[name] = strategy

    @classmethod
    def list_search_strategies(cls) -> list[str]:
        """List all available search strategies."""
        return list(cls._search_strategies.keys())

    def crop_mesh(
        self,
        crop_function: Callable = cropping.hull_crop,
        coordinate_names: tuple[str, str] = ("longitude", "latitude"),
        **kwargs,
    ) -> None:
        """
        Crop the mesh grid to prevent extrapolation beyond survey boundaries.

        Mesh cropping is essential for avoiding unreliable kriging predictions in regions with
        sparse or no data coverage. The default method uses convex hull boundaries around survey
        transects, but custom boundary functions can be specified for more complex survey
        geometries.

        Parameters
        ----------
        crop_function : Callable, default=hull_crop
            Function that defines the survey boundary for mesh subsetting. The default ``hull_crop``
            creates convex hull polygons around transect data with optional buffering. Custom
            functions must accept ``mesh`` as a keyword argument. See
            :func:`echopop.geostatistics.hull_crop` for more details.

        coordinate_names : Tuple[str, str], default=("longitude", "latitude")
            Column names for spatial coordinates in the cropping function. Uses geographic
            coordinates by default for compatibility with ``hull_crop``.

        **kwargs
            Additional arguments passed to the cropping function. For ``hull_crop``:

            - ``'transects' (pandas.DataFrame)``: Survey transect data for boundary definition

            - ``'num_nearest_transects'(int)``: Number of neighbors for local hull creation

            - ``'mesh_buffer_distance' (float)``: Buffer distance in nautical miles

            - ``'projection' (str)``: EPSG code for coordinate system (e.g., ``'epsg:4326'``)

        Returns
        -------
        None
            Updates the ``mesh_cropped`` attribute with the subsetted mesh.

        Notes
        -----
        The cropping process typically involves:

        1. **Boundary Definition**: Creating polygons around survey transects

        2. **Buffer Application**: Expanding boundaries to include nearby areas

        3. **Mesh Intersection**: Selecting mesh points within boundaries

        4. **Area Adjustment**: Updating cell areas for boundary cells if needed

        **Boundary Methods:**

        - **Convex Hull**: Simple, conservative boundary (default)

        - **Alpha Shapes**: More flexible boundaries for complex geometries

        - **Custom Polygons**: User-defined survey strata or management areas

        Examples
        --------
        Crop mesh using convex hull with 2.5 nmi buffer:

        >>> krig.crop_mesh(
        ...     transect_df=survey_data,
        ...     num_nearest_transects=5,
        ...     mesh_buffer_distance=2.5,
        ...     projection='epsg:4326'
        ... )

        Using a custom cropping function:

        >>> def custom_crop(mesh, boundary_polygon, **kwargs):
        ...     # Custom boundary logic
        ...     return mesh[mesh_within_polygon(mesh, boundary_polygon)]
        >>>
        >>> krig.crop_mesh(
        ...     crop_function=custom_crop,
        ...     boundary_polygon=survey_stratum
        ... )
        """
        # Get correct arguments
        args = inspect.signature(crop_function).parameters

        # Inject coordinate names
        if "coordinate_names" in args and "coordinate_names" not in kwargs:
            kwargs["coordinate_names"] = coordinate_names

        # Call the cropping function
        result = crop_function(mesh=self.mesh, **kwargs)

        # Update the cropped mesh DataFrame
        self.mesh_cropped = result[0] if isinstance(result, tuple) else result

    def krige(
        self,
        transects: pd.DataFrame,
        variable: str,
        extrapolate: bool = True,
        default_mesh_cell_area: float | None = None,
        adaptive_search_strategy: str = "uniform",
        custom_search_kwargs: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Perform ordinary kriging interpolation and project results onto the mesh grid.

        This method implements the complete kriging workflow: neighbor selection, covariance matrix
        construction, weight calculation, prediction, and variance estimation. The results are
        projected onto the mesh grid with proper area weighting for survey-level biomass estimates.

        Parameters
        ----------
        transects : |pd.DataFrame|
            Georeferenced survey data containing coordinates and the target variable. Must include
            columns specified in ``coordinate_names`` and ``variable``.

        variable : str
            Column name of the variable to interpolate (e.g., ``'biomass_density'``).

        extrapolate : bool, default=True
            If True, uses the full mesh grid (may extrapolate beyond data coverage). If False, uses
            the cropped mesh (requires prior call to
            :meth:`echopop.geostatistics.Kriging.crop_mesh`).

        default_mesh_cell_area : float, optional
            Default area (nmi²) for mesh cells when ``'area'`` column is missing from mesh.
            Required if mesh lacks area information and no ``'fraction'`` column exists.

        adaptive_search_strategy : str, default='uniform'
            Name of the search strategy for handling sparse data regions. Built-in strategies:

            - ``'uniform'``: Applies uniform weights to extrapolated points (default).

            Use :meth:`echopop.geostatistics.Kriging.register_search_strategy` to add custom
            strategies.

        custom_search_kwargs : Dict[str, Any], default={}
            Additional keyword arguments passed to the adaptive search strategy function. Available
            parameters depend on the selected strategy but may include custom weighting schemes,
            distance thresholds, or algorithm-specific parameters. If the custom function
            incorporates ``coordinate_names`` or ``kriging_mesh`` as arguments, they will be
            inherited from the class instance. See
            :func:`echopop.geostatistics.uniform_strategy` for more details on internal argument
            names that can be added to the custom function call.

        Returns
        -------
        |pd.DataFrame|
            Kriged results with columns:

            - Original mesh columns (coordinates, area, etc.)

            - ``{variable}``: Column name associated ``variable``

            - ``'kriged_variance'``: Prediction variance from kriging equations

            - ``'sample_variance'``: Coefficient of variation based variance

            - ``'cell_cv'``: Cell-level coefficient of variation

        Raises
        ------
        KeyError
            If required columns are missing from ``transects`` or mesh lacks area info.
        AttributeError
            If ``extrapolate=False`` but ``mesh_cropped`` is None.
        ValueError
            If  ``adaptive_search_strategy`` is not a registered strategy name.

        Notes
        -----
        **Kriging Process:**

        1. **Neighbor Search**: Find :math:`k`-nearest neighbors within search radius

        2. **Covariance Matrix**: Build spatial covariance structure using variogram

        3. **Weight Calculation**: Solve kriging system with singular value decomposition (SVD)
           for stability

        4. **Prediction**: Compute weighted estimates and prediction variance

        5. **Projection**: Scale results by cell areas for survey totals

        **Variance Components:**

        - **Kriged Variance**: From kriging equations, measures prediction uncertainty

        - **Sample Variance**: CV-based measure incorporating data variability

        - **Survey CV**: Overall coefficient of variation for the entire survey

        **Quality Indicators:**

        - Negative predictions are truncated to zero with warnings

        - High kriged variance indicates uncertain predictions

        - Large survey CV suggests high spatial variability or poor model fit

        **Search Strategy Options:**

        The adaptive search handles regions with insufficient neighbors:

        - **Interpolation**: ``k_min`` ≤ neighbors ≤ ``k_max`` within search radius

        - **Extrapolation**: < ``k_min`` neighbors, uses distance-weighted nearest points

        - **Full Extrapolation**: No neighbors within radius, uses ``k_min`` nearest

        Examples
        --------
        Standard kriging with extrapolation:

        >>> results = krig.krige(
        ...     transects=survey_data,
        ...     variable='biomass_density',
        ...     extrapolate=True,
        ...     default_mesh_cell_area=1.0
        ... )
        >>> print(f"Survey CV: {krig.survey_cv:.3f}")

        Conservative kriging without extrapolation:

        >>> krig.crop_mesh(transect_df=survey_data, mesh_buffer_distance=2.0)
        >>> results = krig.krige(
        ...     transects=survey_data,
        ...     variable='biomass_density',
        ...     extrapolate=False
        ... )

        Registering and using a custom strategy:

        >>> def conservative_search(**params):
        ...     # Custom logic for sparse regions
        ...     return modified_indices, weights
        >>>
        >>> Kriging.register_search_strategy('conservative', conservative_search)
        >>> results = krig.krige(
        ...     transects=survey_data,
        ...     variable='biomass_density',
        ...     adaptive_search_strategy='conservative',
        ...     custom_search_kwargs={'threshold': 0.5}
        ... )
        """
        # Apply mutable default guard
        custom_search_kwargs = custom_search_kwargs or {}

        # Validate the required columns in ``transects``
        if not set(list(self.coordinate_names) + [variable]) <= set(transects.columns):
            # ---- Find the missing columns
            missing_columns = list(
                set(list(self.coordinate_names) + [variable]).difference(transects.columns)
            )
            # ---- Format the names
            missing_columns = [f"'{mc}'" for mc in missing_columns]
            # ---- Raise error
            raise KeyError(
                f"The following columns are not present in the `transects` DataFrame:"
                f"{', '.join(missing_columns)}."
            )

        # Check for extrapolation
        if not extrapolate:
            # ---- Check for the cropped mesh attribute
            if self.mesh_cropped is None:
                raise AttributeError(
                    "The attribute `mesh_cropped` is None. The `Kriging.crop_mesh(...)` is "
                    "required to initialize the cropped mesh."
                )
            else:
                kriging_mesh = self.mesh_cropped.copy()
        else:
            kriging_mesh = self.mesh.copy()

        # Resolve search strategy
        if adaptive_search_strategy not in self._search_strategies:
            available = ", ".join(self.list_search_strategies())
            raise ValueError(
                f"Unknown search strategy '{adaptive_search_strategy}'. "
                f"Available strategies: {available}"
            )

        # Get the strategy function
        strategy_callable = self._search_strategies[adaptive_search_strategy]

        # Inspect the parameters
        strategy_params = inspect.signature(strategy_callable).parameters

        # Inherit kwargs from instance
        # ---- Mesh
        if "kriging_mesh" in strategy_params:
            custom_search_kwargs["kriging_mesh"] = kriging_mesh
        # ---- Coordinates
        if "coordinate_names" in strategy_params:
            custom_search_kwargs["coordinate_names"] = self.coordinate_names

        # Create partial function
        strategy_func = partial(strategy_callable, **custom_search_kwargs)

        # Apply ordinary kriging for interpolation
        kriged_estimates = ordinary_kriging(
            transects=transects,
            kriging_mesh=kriging_mesh,
            coordinate_names=self.coordinate_names,
            variable=variable,
            kriging_parameters=self.kriging_params,
            variogram_parameters=self.variogram_params,
            adaptive_search_strategy=strategy_func,
        )

        # Project the interpolated results
        if default_mesh_cell_area or "area" in kriging_mesh.columns:
            kriged_results, self.survey_cv = project_kriging_results(
                kriged_estimates=kriged_estimates,
                kriging_mesh=kriging_mesh,
                transects=transects,
                variable=variable,
                default_mesh_cell_area=default_mesh_cell_area,
            )
        else:
            raise KeyError(
                "Missing information required for grid area estimates. The input 'mesh' DataFrame"
                "must contain the column 'area'. Otherwise, a value for 'default_mesh_cell_area' "
                "is required."
            )

        # Rename the variable and return
        return kriged_results.rename(columns={"estimate": variable})
