"""
Variogram model definitions, fitting routines, and empirical variogram computation.

Defines parametric variogram families (spherical, exponential, Gaussian, etc.), provides weighted
least-squares fitting against empirical lag-distance bins, and exposes the initialisation and
bounds helpers consumed by the variogram optimisation workflow.
"""

import inspect
import warnings
from typing import Any

import numpy as np
from lmfit import Minimizer, Parameters
from scipy import special

# Set warnings filter
warnings.simplefilter("always")


# Single family models
# ---- Circular
def circular(distance_lags: np.ndarray, correlation_range: float, sill: float, nugget: float):
    r"""
    Circular variogram model with a smooth, finite range and plateau.

    Parameters
    ----------
    distance_lags: np.ndarray
        Array of spatial lag distances.
    correlation_range : float
        Range parameter where the model reaches the sill.
    sill : float
        Total variance (plateau value).
    nugget : float, optional
        Nugget effect representing measurement error and micro-scale variation.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The circular variogram model is defined as:

    .. math::
        \gamma(h) =
        \begin{cases}
            N + S \left[1 - \frac{2}{\pi} \arccos\left(\frac{h}{a}\right) + \frac{2h}{\pi a}
            \sqrt{1 - \left(\frac{h}{a}\right)^2}\right] & \text{if } h < a \\
            N + S & \text{if } h \geq a
        \end{cases}

    where:

    - \gamma(h) is the variogram at lag distance h

    - N is the nugget effect

    - S is the sill (plateau value)

    - a is the range parameter

    This model is suitable for processes with a smooth increase in variance up to a finite range,
    after which the semivariance remains constant.

    References
    ----------
    .. [1] Cressie, N. (1993). Statistics for Spatial Data. Wiley.
    .. [2] Journel, A.G. & Huijbregts, C.J. (1978). Mining Geostatistics. Academic Press.
    """
    hr = distance_lags / correlation_range
    result = np.where(
        distance_lags < correlation_range,
        nugget + sill * (1 - (2 / np.pi) * np.arccos(hr) + (2 * hr / np.pi) * np.sqrt(1 - hr**2)),
        nugget + sill,
    )
    return result


# ---- Cubic
def cubic(distance_lags: np.ndarray, sill: float, nugget: float, correlation_range: float):
    r"""
    Cubic variogram model with smooth transitions and finite range.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Effective range where correlation becomes zero.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The cubic variogram model is defined as:

    .. math::
        γ(h) = \\begin{cases}
        C_0 + C_1 \\left(7\\left(\\frac{h}{a}\\right)^2 \\right. \\\\
        \\qquad\\qquad \\left. - \\frac{35}{4}\\left(\\frac{h}{a}\\right)^3
        + \\frac{7}{2}\\left(\\frac{h}{a}\\right)^5 \\right. \\\\
        \\qquad\\qquad \\left. - \\frac{3}{4}\\left(\\frac{h}{a}\\right)^7\\right) & h ≤ a \\\\
        C_0 + C_1 & h > a
        \\end{cases}

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a is the effective range

    The cubic model provides very smooth transitions with continuous first and second derivatives
    at the origin and at the range. It's suitable for highly regular spatial processes requiring
    smooth interpolation.

    References
    ----------
    .. [1] Journel, A.G. & Huijbregts, C.J. (1978). Mining Geostatistics. Academic Press.
    .. [2] Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty. Wiley.
    """
    # Calculate normalized distances
    h_norm = distance_lags / correlation_range

    # Calculate cubic correlation for h <= a
    correlation = np.where(
        h_norm <= 1,
        1 - (7 * h_norm**2 - 35 / 4 * h_norm**3 + 7 / 2 * h_norm**5 - 3 / 4 * h_norm**7),
        0,
    )

    # Calculate variogram
    return nugget + (sill - nugget) * (1 - correlation)


# ---- Stable/Ex(ponential)class
def exclass(distance_lags: np.ndarray, correlation_range: float, sill: float, alpha: float):
    r"""
    Generalized (stable) exponential variogram model.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    correlation_range : float
        Range parameter controlling the scale of spatial correlation.
    sill : float
        Total variance (plateau value).
    alpha : float
        Shape parameter (controls the rate of decay, 0 < alpha <= 2).

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The generalized exponential (stable) variogram model is defined as:

    .. math::
        \gamma(h) = S \left[1 - \exp\left(-\left(\frac{h}{a}\right)^{\alpha}\right)\right]

    where:
    - \gamma(h) is the variogram at lag distance h
    - S is the sill (plateau value)
    - a is the correlation range
    - \alpha is the shape parameter

    This model generalizes the exponential (alpha=1) and Gaussian (alpha=2) models.

    References
    ----------
    .. [1] Wackernagel, H. (2003). Multivariate Geostatistics. Springer.
    """
    return sill * (1 - np.exp(-((distance_lags / correlation_range) ** alpha)))


# ---- Exponential
def exponential(distance_lags: np.ndarray, sill: float, nugget: float, correlation_range: float):
    r"""
    Exponential variogram model for spatial correlation analysis.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Effective range parameter controlling spatial correlation decay.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The exponential variogram model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\exp\\left(-\\frac{h}{a}\\right)\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a is the correlation range parameter

    The exponential model exhibits monotonic decay and reaches 95% of the sill at distance 3a. It
    represents processes with continuous but non-differentiable spatial correlation.

    References
    ----------
    .. [1] Cressie, N. (1993). Statistics for Spatial Data. Wiley.
    .. [2] Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty. Wiley.
    """
    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Calculate the spatial decay term
    decay = 1.0 - np.exp(-(distance_lags / correlation_range))

    # Compute the exponential semivariogram
    return partial_sill * decay + nugget


# ---- Gaussian
def gaussian(distance_lags: np.ndarray, sill: float, nugget: float, correlation_range: float):
    r"""
    Gaussian (squared exponential) variogram model for spatial correlation analysis.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Effective range parameter controlling spatial correlation decay.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The Gaussian variogram model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\exp\\left(-\\frac{h^2}{a^2}\\right)\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a is the correlation range parameter

    The Gaussian model exhibits very smooth spatial transitions with infinitely differentiable
    correlation functions. It approaches the sill more gradually than exponential models and is
    suitable for highly regular spatial processes.

    References
    ----------
    .. [1] Cressie, N. (1993). Statistics for Spatial Data. Wiley.
    .. [2] Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for Machine Learning. MIT
           Press.
    """
    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Calculate the spatial decay term
    decay = 1.0 - np.exp(-(distance_lags**2 / correlation_range**2.0))

    # Compute the Gaussian semivariogram
    return partial_sill * decay + nugget


# ---- J-Bessel
def jbessel(distance_lags: np.ndarray, sill: float, nugget: float, hole_effect_range: float):
    r"""
    J-Bessel variogram model exhibiting hole-effect (periodic) behavior.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    hole_effect_range : float
        Range parameter controlling the periodicity of oscillations.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The J-Bessel variogram model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\frac{J_1(h/a)}{h/(2a)}\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - J₁ is the first-order Bessel function of the first kind

    - a is the hole-effect range parameter

    This model exhibits damped oscillatory behavior, creating "hole effects" where the variogram
    exceeds the sill before returning. It's suitable for processes with regular spatial patterns or
    cyclical structures.

    References
    ----------
    .. [1] Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty. Wiley.
    .. [2] Cressie, N. (1993). Statistics for Spatial Data. Wiley.
    """
    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Calculate the spatial decay term
    decay = 1.0 - special.j0(hole_effect_range * distance_lags)

    # Compute the J-Bessel semivariogram
    return partial_sill * decay + nugget


# ---- K-Bessel
def kbessel(distance_lags: np.ndarray, sill: float, nugget: float, hole_effect_range: float):
    r"""
    K-Bessel variogram model with modified Bessel function of the second kind.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    hole_effect_range : float
        Correlation length in the same units as the lag distances.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The K-Bessel variogram model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\frac{2^{1-ν}}{Γ(ν)}
               \\left(\\frac{h}{a}\\right)^ν K_ν\\left(\\frac{h}{a}\\right)\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a is the correlation range parameter

    - K_ν is the modified Bessel function of the second kind

    - ν is fixed at 1

    - Γ(ν) is the gamma function

    References
    ----------
    .. [1] Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty. Wiley.
    .. [2] Stein, M.L. (1999). Interpolation of Spatial Data. Springer.
    """
    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Avoid the case where `hole_effect_range` = 0.0
    if hole_effect_range == 0.0:
        return np.where(distance_lags == 0.0, nugget, sill)

    # Compute the cyclical term
    cycle = np.where(
        distance_lags / hole_effect_range < 1e-4,
        0.0,
        special.kv(1.0, (distance_lags / hole_effect_range)),
    )

    # Calculate the spatial decay term
    decay = np.where(
        distance_lags / hole_effect_range < 1e-4,
        0.0,
        1.0 - (distance_lags / hole_effect_range) * cycle,
    )

    # Compute the K-Bessel semivariogram
    return partial_sill * decay + nugget


# ---- Linear
def linear(distance_lags: np.ndarray, sill: float, nugget: float):
    r"""
    Linear variogram model for unbounded spatial processes.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Linear slope parameter (variance increases linearly with distance).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The linear variogram model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\cdot h

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the slope parameter (equivalent to sill in this context)

    The linear model represents unbounded spatial processes where variance increases indefinitely
    with distance. It violates the assumption of a finite sill and should be used cautiously,
    typically for trend analysis or detrended residuals.

    References
    ----------
    .. [1] Cressie, N. (1993). Statistics for Spatial Data. Wiley.
    .. [2] Wackernagel, H. (2003). Multivariate Geostatistics. Springer.
    """
    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Compute the linear semivariogram
    return partial_sill * distance_lags + nugget


# --- Linear plateau
def linear_plateau(distance_lags: np.ndarray, sill: float, correlation_range: float):
    r"""
    Linear plateau variogram model with bounded linear growth up to a finite range.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Plateau value (maximum semivariance).
    correlation_range : float
        Effective range where the plateau is reached.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The linear plateau variogram model is defined as:

    .. math::
        \gamma(h) =
        \begin{cases}
            S \cdot \frac{h}{a} & \text{if } h < a \\
            S & \text{if } h \geq a
        \end{cases}

    where:

    - \gamma(h) is the variogram at lag distance h

    - S is the sill (plateau value)

    - a is the correlation range

    This model exhibits linear growth up to the specified range, after which the semivariance
    remains constant. It is suitable for processes with a linear increase in variance up to a
    threshold, followed by a stable plateau.

    References
    ----------
    .. [1] Cressie, N. (1993). Statistics for Spatial Data. Wiley.
    .. [2] Journel, A.G. & Huijbregts, C.J. (1978). Mining Geostatistics. Academic Press.
    """
    return np.where(
        distance_lags < correlation_range, sill * distance_lags / correlation_range, sill
    )


# ---- Logarithmic
def logarithmic(distance_lags: np.ndarray, correlation_range: float, sill: float, nugget: float):
    r"""
    Logarithmic variogram model with monotonic growth.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    correlation_range : float
        Range parameter added to the lag in the logarithm.
    sill : float
        Total variance (scaling factor).
    nugget : float, optional
        Nugget effect representing measurement error and micro-scale variation.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The logarithmic variogram model is defined as:

    .. math::
        \gamma(h) = N + S \log(h + a)

    where:

    - \gamma(h) is the variogram at lag distance h

    - N is the nugget effect

    - S is the sill (scaling factor)

    - a is the correlation range

    For h = 0, the value is set to the nugget.

    This model is suitable for processes with slow, monotonic increase in variance.

    References
    ----------
    .. [1] Wackernagel, H. (2003). Multivariate Geostatistics. Springer.
    """
    result = np.zeros_like(distance_lags)
    mask = distance_lags != 0
    result[mask] = nugget + sill * np.log(distance_lags[mask] + correlation_range)
    result[~mask] = nugget  # h == 0
    return result


# ---- Matern
def matern(distance_lags, sill, nugget, correlation_range, smoothness_parameter):
    r"""
    Matérn variogram model with flexible smoothness control.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Effective range parameter controlling spatial correlation decay.
    smoothness_parameter : float
        Smoothness parameter (ν) controlling field differentiability (ν > 0).

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The Matérn variogram model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\frac{2^{1-ν}}{Γ(ν)}
               \\left(\\frac{h}{a}\\right)^ν K_ν\\left(\\frac{h}{a}\\right)\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a is the correlation range parameter

    - ν is the smoothness parameter

    - K_ν is the modified Bessel function of the second kind

    - Γ(ν) is the gamma function

    **Common smoothness parameter interpretations:**

    - ν = 0.5: Exponential variogram (non-differentiable)

    - ν = 1.5: Once differentiable random fields

    - ν = 2.5: Twice differentiable random fields

    - ν → ∞: Approaches Gaussian variogram (infinitely differentiable)

    References
    ----------
    .. [1] Matérn, B. (1986). Spatial Variation. Springer-Verlag.
    .. [2] Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for Machine Learning. MIT
           Press.
    """
    # Calculate the argument for the Bessel function
    h_scaled = distance_lags / correlation_range

    # Handle zero distances
    h_scaled = np.where(h_scaled == 0, 1e-10, h_scaled)

    # Calculate Matérn correlation
    bessel_arg = np.sqrt(2 * smoothness_parameter) * h_scaled
    correlation = (
        (2 ** (1 - smoothness_parameter) / special.gamma(smoothness_parameter))
        * (bessel_arg**smoothness_parameter)
        * special.kv(smoothness_parameter, bessel_arg)
    )

    # Handle numerical issues at origin
    correlation = np.where(distance_lags == 0, 1.0, correlation)

    # Calculate variogram
    return nugget + (sill - nugget) * (1 - correlation)


# ---- Nugget
def nugget(distance_lags: np.ndarray, sill: float, nugget: float):
    r"""
    Pure nugget variogram model representing uncorrelated spatial noise.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (equivalent to nugget in this model).
    nugget : float
        Retained for the common model interface. The total sill determines positive-lag values.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The pure nugget variogram model is defined as:

    .. math::
        γ(h) = \\begin{cases}
        0 & \\text{if } h = 0 \\\\
        C & \\text{if } h > 0
        \\end{cases}

    where:

    - γ(h) is the variogram at lag distance h

    - C is the total sill, entirely attributable to uncorrelated variation

    This model represents completely uncorrelated spatial data where:

    - No spatial correlation exists beyond measurement locations

    - All variance is attributed to measurement error or micro-scale variation

    - The spatial process appears as white noise

    The pure nugget model is useful for:

    - Modeling measurement error components

    - Representing spatially uncorrelated residuals

    - Baseline comparison for other variogram models

    - Data with extremely short correlation ranges

    References
    ----------
    .. [1] Cressie, N. (1993). Statistics for Spatial Data. Wiley.
    .. [2] Journel, A.G. & Huijbregts, C.J. (1978). Mining Geostatistics. Academic Press.
    """
    # All variance is uncorrelated, with a zero semivariance at zero lag.
    return np.where(distance_lags == 0.0, 0.0, sill)


# --- Pentaspherical
def pentaspherical(distance_lags: np.ndarray, sill: float, nugget: float, correlation_range: float):
    r"""
    Pentaspherical variogram model with quintic polynomial behavior and finite range.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Effective range where correlation becomes zero.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The pentaspherical variogram model is defined as:

    .. math::
        γ(h) = \\begin{cases}
        C_0 + C_1 \\left(\\frac{15h}{8a} - \\frac{5h^3}{4a^3} \\right. \\\\
        \\qquad\\qquad \\left. + \\frac{3h^5}{8a^5}\\right) & \\text{if } h ≤ a \\\\
        C_0 + C_1 & \\text{if } h > a
        \\end{cases}

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a is the effective range

    The pentaspherical model exhibits smoother behavior than the spherical model with continuous
    derivatives up to second order. It provides a good balance between flexibility and
    computational efficiency for bounded spatial processes.

    References
    ----------
    .. [1] Journel, A.G. & Huijbregts, C.J. (1978). Mining Geostatistics. Academic Press.
    .. [2] Wackernagel, H. (2003). Multivariate Geostatistics. Springer.
    """
    # Calculate normalized distances
    h_norm = distance_lags / correlation_range

    # Calculate pentaspherical correlation for h <= a
    correlation = np.where(
        h_norm <= 1, 1 - (15 * h_norm / 8 - 5 * h_norm**3 / 4 + 3 * h_norm**5 / 8), 0
    )

    # Calculate variogram
    return nugget + (sill - nugget) * (1 - correlation)


# ---- Periodic
def periodic(distance_lags: np.ndarray, correlation_range: float, sill: float, nugget: float):
    r"""
    Periodic variogram model with regular oscillations.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    correlation_range : float
        Period of the oscillation (wavelength).
    sill : float
        Total variance (scaling factor).
    nugget : float, optional
        Nugget effect representing measurement error and micro-scale variation.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The periodic variogram model is defined as:

    .. math::
        \gamma(h) = N + S \left[1 - \cos\left(\frac{2\pi h}{a}\right)\right]

    where:

    - \gamma(h) is the variogram at lag distance h

    - N is the nugget effect

    - S is the sill (scaling factor)

    - a is the correlation range (period)

    This model is suitable for processes with regular, repeating spatial patterns.

    References
    ----------
    .. [1] Wackernagel, H. (2003). Multivariate Geostatistics. Springer.
    """
    return nugget + sill * (1 - np.cos(2 * np.pi * distance_lags / correlation_range))


# ---- Power law
def power(distance_lags: np.ndarray, sill: float, nugget: float, power_exponent: float):
    r"""
    Power law variogram model for fractal spatial processes.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Not used in power law model (set to 0 or ignore).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    power_exponent : float
        Power exponent parameter (0 < β < 2) controlling scaling behavior.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The power law variogram model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\cdot h^β

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the scaling coefficient

    - β is the power exponent (0 < β < 2)

    **Power exponent interpretations:**

    - β = 1: Linear variogram

    - β → 0: Approaches nugget effect only

    - β → 2: Approaches parabolic behavior (Brownian motion)

    This model represents unbounded, self-similar (fractal) spatial processes
    without a finite sill. It's commonly used for modeling phenomena with
    scale-invariant properties.

    References
    ----------
    .. [1] Cressie, N. (1993). Statistics for Spatial Data. Wiley.
    .. [2] Goovaerts, P. (1997). Geostatistics for Natural Resources Evaluation. Oxford University
           Press.
    """
    # Handle zero distances
    h_nonzero = np.where(distance_lags == 0, 0, distance_lags**power_exponent)

    # Calculate variogram (sill parameter acts as scaling coefficient)
    return nugget + sill * h_nonzero


# ---- (Rational) quadratic
def quadratic(
    distance_lags: np.ndarray,
    sill: float,
    nugget: float,
    correlation_range: float,
    shape_parameter: float,
):
    r"""
    Rational quadratic variogram model with polynomial decay behavior.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Effective range parameter controlling spatial correlation decay.
    shape_parameter : float
        Shape parameter (α) controlling decay rate (α > 0).

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The rational quadratic variogram model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\left(1 + \\frac{h^2}{2αa^2}\\right)^{-α}\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a is the correlation range parameter

    - α is the shape parameter

    The rational quadratic model exhibits polynomial decay and can be viewed as a scale mixture of
    Gaussian processes. It provides intermediate behavior between exponential and Gaussian models.

    References
    ----------
    .. [1] Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for Machine Learning. MIT
    Press.
    .. [2] Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty. Wiley.
    """
    # Calculate scaled distances
    h_scaled = distance_lags / correlation_range

    # Calculate correlation
    correlation = (1 + h_scaled**2 / (2 * shape_parameter)) ** (-shape_parameter)

    # Calculate variogram
    return nugget + (sill - nugget) * (1 - correlation)


# ---- Sinc
def sinc(distance_lags: np.ndarray, sill: float, nugget: float, hole_effect_range: float):
    r"""
    Sinc (cardinal sine) variogram model exhibiting oscillatory hole-effect behavior.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    hole_effect_range : float
        Range parameter controlling the wavelength of oscillations.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The sinc variogram model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\frac{\\sin(\\pi h/a)}{\\pi h/a}\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a is the hole-effect range parameter

    - The limit as h→0 gives sinc(0) = 1

    The sinc function creates regular oscillatory patterns with:
    - First zero crossing at h = a
    - Subsequent zeros at integer multiples of a
    - Damped oscillations that decay as 1/h

    This model is suitable for processes with:
    - Regular periodic structures
    - Wave-like spatial patterns
    - Alternating zones of positive and negative correlation
    - Spectral characteristics in spatial data

    References
    ----------
    .. [1] Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty. Wiley.
    .. [2] Wackernagel, H. (2003). Multivariate Geostatistics. Springer.
    """
    # Get machine epsilon
    eps = np.finfo(float).eps

    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Calculate the spatial decay term
    decay = 1.0 - np.sin(hole_effect_range * (distance_lags + eps)) / (distance_lags + eps)

    # Compute the sinc semivariogram
    return partial_sill * decay + nugget


# ---- Spherical
def spherical(distance_lags: np.ndarray, sill: float, nugget: float, correlation_range: float):
    r"""
    Spherical variogram model with finite effective range.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Effective range where correlation becomes zero.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The spherical variogram model is defined as:

    .. math::
        γ(h) = \\begin{cases}
        C_0 + C_1 \\left(\\frac{3h}{2a} - \\frac{h^3}{2a^3}\\right) & \\text{if } h \\leq a \\\\
        C_0 + C_1 & \\text{if } h > a
        \\end{cases}

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a is the effective range (correlation = 0 beyond this distance)

    The spherical model has a finite range and linear near-origin behavior,
    making it suitable for processes with clear spatial boundaries.

    References
    ----------
    .. [1] Matheron, G. (1963). Principles of geostatistics. Economic Geology, 58(8), 1246-1266.
    .. [2] Journel, A.G. & Huijbregts, C.J. (1978). Mining Geostatistics. Academic Press.
    """
    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Calculate the spatial decay term
    decay = (3.0 * distance_lags) / (2.0 * correlation_range) - distance_lags**3.0 / (
        2.0 * correlation_range**3.0
    )

    # Compute the spherical semivariogram
    return np.where(
        distance_lags < correlation_range,
        partial_sill * decay + nugget,
        sill,
    )


def spline(distance_lags: np.ndarray, correlation_range: float, sill: float):
    r"""
    Spline variogram model with quadratic-logarithmic growth and plateau.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    correlation_range : float
        Range parameter where the plateau is reached.
    sill : float
        Plateau value (maximum semivariance).

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The spline variogram model is defined as:

    .. math::
        \gamma(h) =
        \begin{cases}
            h^2 \log(h) & \text{if } h < a \\
            S & \text{if } h \geq a
        \end{cases}

    where:

    - \gamma(h) is the variogram at lag distance h

    - a is the correlation range

    - S is the sill (plateau value)

    For h = 0, the value is conventionally set to 0.

    This model is mainly used for theoretical purposes and can produce negative values for h < 1.

    References
    ----------
    .. [1] Wackernagel, H. (2003). Multivariate Geostatistics. Springer.
    """
    h_safe = np.where(distance_lags == 0, 1e-10, distance_lags)
    result = h_safe**2 * np.log(h_safe)
    result = np.where(distance_lags >= correlation_range, sill, result)


def stein(distance_lags: np.ndarray, correlation_range: float, smoothness: float):
    r"""
    Stein (Matérn) variogram model for flexible smoothness and spatial correlation.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    correlation_range : float
        Range parameter controlling the scale of spatial correlation.
    smoothness : float
        Smoothness parameter (Matérn index).

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The Stein (Matérn) variogram model is defined as:

    .. math::
        \gamma(h) = 1 - \frac{2^{1-\nu}}{\Gamma(\nu)} \left(2 \sqrt{\nu} \frac{h}{a}\right)^{\nu}
        K_{\nu}\left(2 \sqrt{\nu} \frac{h}{a}\right)

    where:

    - \gamma(h) is the variogram at lag distance h

    - a is the correlation range

    - \nu is the smoothness parameter

    - K_{\nu} is the modified Bessel function of the second kind

    - \Gamma(\nu) is the gamma function

    This model generalizes the exponential and Gaussian models and is widely used for its
    flexibility in controlling smoothness.

    References
    ----------
    .. [1] Stein, M.L. (1999). Statistical Interpolation of Spatial Data: Some Theory for Kriging.
    Springer.
    .. [2] Guttorp, P. & Gneiting, T. (2006). Studies in the Matérn Model. Bernoulli.
    """
    distance_lags = np.maximum(distance_lags, 1e-10)
    arg = 2 * np.sqrt(smoothness) * distance_lags / correlation_range
    part1 = (2 ** (1 - smoothness)) / special.gamma(smoothness)
    part2 = arg**smoothness
    part3 = special.kv(smoothness, arg)
    return 1 - part1 * part2 * part3


# ---- Tetraspherical
def tetraspherical(distance_lags: np.ndarray, correlation_range: float, sill: float, nugget: float):
    r"""
    Tetraspherical variogram model with smooth transition to the sill.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    correlation_range : float
        Range parameter where the sill is reached.
    sill : float
        Total variance (plateau value).
    nugget : float, optional
        Nugget effect representing measurement error and micro-scale variation.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The tetraspherical variogram model is defined as:

    .. math::
        \gamma(h) =
        \begin{cases}
            N + S \left[ \arcsin\left(\frac{h}{a}\right) + \frac{h}{a} \sqrt{1 -
            \left(\frac{h}{a}\right)^2} + \frac{2}{3} \frac{h}{a} \left(1 -
            \left(\frac{h}{a}\right)^2\right)^{3/2} \right], & 0 \leq h \leq a \\
            N + S, & h > a
        \end{cases}

    References
    ----------
    .. [1] Journel, A.G. & Huijbregts, C.J. (1978). Mining Geostatistics. Academic Press.
    """
    hr = distance_lags / correlation_range
    inside = hr < 1
    result = np.full_like(distance_lags, nugget + sill)
    if np.any(inside):
        term1 = np.arcsin(hr[inside])
        term2 = hr[inside] * np.sqrt(1 - hr[inside] ** 2)
        term3 = (2 / 3) * hr[inside] * (1 - hr[inside] ** 2) ** (3 / 2)
        result[inside] = nugget + sill * (term1 + term2 + term3)
    return result


def wave(distance_lags: np.ndarray, correlation_range: float, sill: float, nugget: float):
    r"""
    Wave (hole-effect) variogram model with oscillatory behavior.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    correlation_range : float
        Range parameter controlling the wavelength of oscillations.
    sill : float
        Total variance (plateau value).
    nugget : float, optional
        Nugget effect representing measurement error and micro-scale variation.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The wave variogram model is defined as:

    .. math::
        \gamma(h) =
        N + S \left[1 - \frac{a \sin\left(\pi h / a\right)}{\pi h}\right]

    where:

    - \gamma(h) is the variogram at lag distance h

    - N is the nugget effect

    - S is the sill (plateau value)

    - a is the correlation range

    For h = 0, the value is set to the nugget.

    This model is suitable for processes with regular oscillatory spatial patterns and
    hole-effect behavior.

    References
    ----------
    .. [1] Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty. Wiley.
    .. [2] Wackernagel, H. (2003). Multivariate Geostatistics. Springer.
    """
    result = np.zeros_like(distance_lags)
    mask = distance_lags != 0
    result[mask] = nugget + sill * (
        1
        - (correlation_range * np.sin(np.pi * distance_lags[mask] / correlation_range))
        / (np.pi * distance_lags[mask])
    )
    result[~mask] = nugget  # h == 0
    return result


# ---- Whittle's Elementary Correlation
def whittle(distance_lags: np.ndarray, sill: float, nugget: float, correlation_range: float):
    r"""
    Whittle variogram model using the modified Bessel function of the second kind.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (scaling factor).
    nugget : float, optional
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Range parameter controlling the scale of spatial correlation.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The Whittle variogram model is defined as:

    .. math::
        \gamma(h) = N + S \left[1 - K_1\left(\frac{h}{a}\right)\right]

    where:

    - \gamma(h) is the variogram at lag distance h

    - N is the nugget effect

    - S is the sill (scaling factor)

    - a is the correlation range

    - K_1 is the modified Bessel function of the second kind (order 1)

    For h = 0, the value is conventionally set to the nugget.

    References
    ----------
    .. [1] Whittle, P. (1954). On stationary processes in the plane. Biometrika.
    .. [2] Wackernagel, H. (2003). Multivariate Geostatistics. Springer.
    """
    distance_lags = np.maximum(distance_lags, 1e-10)
    val = special.kv(1, distance_lags / correlation_range)
    return nugget + sill * (1 - val)


# Composite family models (i.e. hole-effects)
# ---- J-Bessel and Gaussian
def bessel_gaussian(
    distance_lags: np.ndarray,
    sill: float,
    nugget: float,
    correlation_range: float,
    hole_effect_range: float,
):
    r"""
    Composite Bessel-Gaussian variogram model combining periodic and smooth decay.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Range parameter for Gaussian decay component.
    hole_effect_range : float
        Angular frequency of Bessel oscillations, in inverse lag-distance units.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The Bessel-Gaussian composite model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\exp\\left(-\\frac{h^2}{a_1^2}\\right)
        \\cdot J_0(a_2 h)\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a₁ is the correlation range (Gaussian component)

    - a₂ is the angular frequency (Bessel component)

    - J₀ is the zeroth-order Bessel function

    This model combines smooth Gaussian decay with oscillatory behavior, creating damped periodic
    patterns suitable for highly regular processes with cyclical spatial structures.

    References
    ----------
    .. [1] Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty. Wiley.
    .. [2] Cressie, N. (1993). Statistics for Spatial Data. Wiley.
    """
    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Calculate the spatial decay term
    decay = np.exp(-((distance_lags / correlation_range) ** 2))

    # Calculate the hole effect
    hole_effect = special.j0(hole_effect_range * distance_lags)

    # Compute the composite J-Bessel and Gaussian semivariogram
    return partial_sill * (1.0 - decay * hole_effect) + nugget


# ---- J-Bessel and exponential
def bessel_exponential(
    distance_lags: np.ndarray,
    nugget: float,
    sill: float,
    correlation_range: float,
    decay_power: float,
    hole_effect_range: float,
):
    r"""
    Composite Bessel-exponential variogram model combining periodic and decay behavior.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Range parameter for exponential decay component.
    hole_effect_range : float
        Angular frequency of Bessel oscillations, in inverse lag-distance units.
    decay_power : float
        Decay exponent for the exponential component (0 < α ≤ 2).

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The Bessel-exponential composite model combines periodic and exponential decay:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\exp\\left(-\\left(\\frac{h}{a_1}\\right)^α\\right)
               \\cdot J_0(a_2 h)\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a₁ is the correlation range (exponential component)

    - a₂ is the angular frequency (Bessel component)

    - α is the decay power

    - J₀ is the zeroth-order Bessel function

    This model captures both oscillatory patterns and long-range decay, suitable for processes with
    periodic structures that diminish with distance.

    **Decay power interpretations:**
    - α = 1.0: Linear exponential decay with Bessel oscillations
    - α = 2.0: Gaussian decay with Bessel oscillations
    - α ∈ (0,2): Intermediate decay behaviors

    References
    ----------
    .. [1] Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty. Wiley.
    .. [2] Matérn, B. (1986). Spatial Variation. Springer-Verlag.
    """
    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Combine the powered-exponential and Bessel correlations before converting to semivariance.
    decay = np.exp(-((distance_lags / correlation_range) ** decay_power))
    hole_effect = special.j0(hole_effect_range * distance_lags)
    return partial_sill * (1.0 - decay * hole_effect) + nugget


# ---- cosine and exponential
def cosine_exponential(
    distance_lags: np.ndarray,
    sill: float,
    nugget: float,
    correlation_range: float,
    hole_effect_range: float,
    enhance_semivariance: bool,
):
    r"""
    Composite cosine-exponential variogram model with trigonometric modulation.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Range parameter for exponential decay component.
    hole_effect_range : float
        Range parameter controlling cosine oscillation wavelength.
    decay_power : float
        Decay exponent for the exponential component (0 < α ≤ 2).

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The cosine-exponential composite model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\exp\\left(-\\left(\\frac{h}{a_1}\\right)^α\\right)
               \\cdot \\cos\\left(\\frac{\\pi h}{a_2}\\right)\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a₁ is the correlation range (exponential component)

    - a₂ is the hole-effect range (cosine wavelength)

    - α is the decay power

    This model creates sinusoidal modulation of exponential decay, producing regular oscillatory
    patterns. The cosine component creates predictable periodicity, making it suitable for
    processes with known cyclical behavior.

    **Decay power interpretations:**

    - α = 1.0: Linear exponential decay with cosine modulation

    - α = 2.0: Gaussian decay with cosine modulation

    - α ∈ (0,2): Intermediate decay behaviors

    References
    ----------
    .. [1] Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty. Wiley.
    .. [2] Cressie, N. (1993). Statistics for Spatial Data. Wiley.
    """
    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Calculate the spatial decay term
    decay_modifier = -1.0 if enhance_semivariance is True else 1.0
    decay = decay_modifier * np.exp(-(distance_lags / correlation_range))

    # Calculate the hole effect
    hole_effect = np.cos(hole_effect_range * distance_lags)

    # Compute the composite cosine and exponential semivariogram
    return partial_sill * (1.0 - decay * hole_effect) + nugget


# ---- cosine and Gaussian
def cosine_gaussian(
    distance_lags: np.ndarray,
    sill: float,
    nugget: float,
    correlation_range: float,
    hole_effect_range: float,
):
    r"""
    Composite cosine-Gaussian variogram model with trigonometric modulation and smooth decay.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Range parameter for Gaussian decay component.
    hole_effect_range : float
        Angular frequency of cosine oscillations, in inverse lag-distance units.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The cosine-Gaussian composite model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\exp\\left(-\\frac{h^2}{a_1^2}\\right)
               \\cdot \\cos(a_2 h)\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (sill - nugget)

    - a₁ is the correlation range (Gaussian component)

    - a₂ is the angular frequency (cosine component)

    This model combines:

    - **Gaussian smoothness**: Infinitely differentiable correlation structure

    - **Cosine periodicity**: Regular oscillatory patterns

    - **Smooth transitions**: Gradual approach to the sill

    The Gaussian component provides very smooth spatial correlation while the cosine component
    introduces predictable periodic behavior. This combination is ideal for highly regular spatial
    processes with known cyclical patterns.

    References
    ----------
    .. [1] Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty. Wiley.
    .. [2] Stein, M.L. (1999). Interpolation of Spatial Data. Springer.
    """
    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Calculate the spatial decay term
    decay = np.exp(-((distance_lags / correlation_range) ** 2))

    # Calculate the hole effect
    hole_effect = np.cos(hole_effect_range * distance_lags)

    # Compute the composite cosine and Gaussian semivariogram
    return partial_sill * (1.0 - decay * hole_effect) + nugget


# ---- exponential and linear
def exponential_linear(
    distance_lags: np.ndarray,
    sill: float,
    nugget: float,
    correlation_range: float,
    hole_effect_range: float,
    decay_power: float,
):
    r"""
    Composite exponential-linear variogram model with decay and polynomial modulation.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Range parameter for exponential decay component.
    hole_effect_range : float
        Polynomial coefficient in inverse lag-distance units raised to decay_power.
    decay_power : float
        Decay exponent for the exponential component (0 < α ≤ 2).

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The exponential-linear composite model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\exp\\left(-\\left(\\frac{h}{a_1}\\right)^α\\right)
               \\left(1 - a_2 h^α\\right)\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (exponential component)

    - a₁ is the correlation range (exponential component)

    - a₂ is the polynomial coefficient

    - α is the decay power

    **Decay power interpretations:**

    - α = 1.0: Linear exponential decay with linear modulation

    - α = 2.0: Gaussian decay with quadratic modulation

    - α ∈ (0,2): Intermediate decay behaviors

    References
    ----------
    .. [1] Cressie, N. (1993). Statistics for Spatial Data. Wiley.
    .. [2] Wackernagel, H. (2003). Multivariate Geostatistics. Springer.
    """
    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Calculate the spatial decay term
    decay = np.exp(-((distance_lags / correlation_range) ** decay_power))

    # Calculate the hole effect
    hole_effect = 1.0 - hole_effect_range * distance_lags**decay_power

    # Compute the composite exponential and linear semivariogram
    return partial_sill * (1.0 - decay * hole_effect) + nugget


# ---- Gaussian and linear
def gaussian_linear(
    distance_lags: np.ndarray,
    sill: float,
    nugget: float,
    correlation_range: float,
    hole_effect_range: float,
):
    r"""
    Composite Gaussian-linear variogram model with smooth decay and quadratic modulation.

    Parameters
    ----------
    distance_lags : np.ndarray
        Array of spatial lag distances.
    sill : float
        Total variance (nugget + partial sill).
    nugget : float
        Nugget effect representing measurement error and micro-scale variation.
    correlation_range : float
        Range parameter for Gaussian decay component.
    hole_effect_range : float
        Polynomial coefficient in inverse squared lag-distance units.

    Returns
    -------
    np.ndarray
        Variogram values at specified lag distances.

    Notes
    -----
    The Gaussian-linear composite model is defined as:

    .. math::
        γ(h) = C_0 + C_1 \\left(1 - \\exp\\left(-\\frac{h^2}{a_1^2}\\right)
               \\left(1 - a_2 h^2\\right)\\right)

    where:

    - γ(h) is the variogram at lag distance h

    - C₀ is the nugget effect

    - C₁ is the partial sill (Gaussian component)

    - a₁ is the correlation range (Gaussian component)

    - a₂ is the polynomial coefficient

    This model combines smooth Gaussian decay with quadratic modulation for spatial processes
    with hole effects.

    References
    ----------
    .. [1] Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty. Wiley.
    .. [2] Stein, M.L. (1999). Interpolation of Spatial Data. Springer.
    """
    # Calculate the partial sill (or the sill minus the nugget)
    partial_sill = sill - nugget

    # Calculate the spatial decay term
    decay = np.exp(-((distance_lags / correlation_range) ** 2))

    # Calculate the hole effect
    hole_effect = 1.0 - hole_effect_range * distance_lags**2

    # Compute the composite Gaussian and linear semivariogram
    return partial_sill * (1.0 - decay * hole_effect) + nugget


# Variogram function API
VARIOGRAM_MODELS = {
    "single": {
        "cubic": cubic,
        "exponential": exponential,
        "gaussian": gaussian,
        "jbessel": jbessel,
        "kbessel": kbessel,
        "linear": linear,
        "matern": matern,
        "nugget": nugget,
        "pentaspherical": pentaspherical,
        "power": power,
        "quadratic": quadratic,
        "sinc": sinc,
        "spherical": spherical,
    },
    "composite": {
        ("bessel", "exponential"): bessel_exponential,
        ("bessel", "gaussian"): bessel_gaussian,
        ("cosine", "exponential"): cosine_exponential,
        ("cosine", "gaussian"): cosine_gaussian,
        ("exponential", "linear"): exponential_linear,
        ("gaussian", "linear"): gaussian_linear,
    },
}


# Variogram wrapper function
def compute_variogram(
    distance_lags: np.ndarray[float],
    variogram_parameters: dict[str, float] | None = None,
    model: str | list[str] | None = None,
    **kwargs,
):
    """
    Compute the theoretical semivariogram.

    Parameters
    ----------
    distance_lags : |np.ndarray[float]|
        An array of lag distances.
    variogram_parameters : Optional[Dict[str, float]]
        An optional dictionary that can contain values for variogram model parameters (see the
        below table associated with the argument ``model``). Alternatively, these parameters can be
        entered directly and are contained within ``kwargs``. Possible parameters include:

        - ``sill`` (Sill): The asymptotic value as lags approach infinity.

        - ``nugget`` (Nugget): The semivariogram y-intercept that corresponds to variability
          at lag distances shorter than the lag resolution.

        - ``correlation_range`` (Correlation length scale/range): The ascending rate for the
          semivariogram.

        - ``hole_effect_range`` (Hole effect range): The (normalized) length scale/range that
          'holes' are observed, which represent 'null' (or very small) points compared to their
          neighboring lags.

        - ``decay_power`` (Decay term exponent): An exponential term that is used in certain
          generalized exponential (or related) semivariogram models that modulates the ascending
          rate for a semivariogram.

        - ``enhance_semivariance`` (Semivariance enhancement): A boolean term that determines
          whether the correlation decay in certain cosine-related variogram models are enhanced
          (or not) at further lag distances.

    model : Optional[Union[str, list]]
        A string or list of model names. A single name represents a single family model. Two inputs
        represent the desired composite model (e.g. the composite J-Bessel and exponential model).
        Available variogram models and their respective arguments include (alongside
        ``distance_lags``):


        +----------------------------+-------------------------------+---------------------------+
        | :func:`variogram`          | Input                         | Parameters                |
        | model                      |                               |                           |
        +============================+===============================+===========================+
        | :py:func:`cubic`           | ``'cubic'``                   | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`exponential`        | ``'exponential'``             | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`gaussian`           | ``'gaussian'``                | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`jbessel`            | ``'jbessel'``                 | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``hole_effect_range``   |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`kbessel`            | ``'kbessel'``                 | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``hole_effect_range``   |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`linear`             | ``'linear'``                  | - ``nugget``              |
        |                            |                               | - ``sill``                |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`matern`             | ``'matern'``                  | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        |                            |                               | - ``smoothness_parameter``|
        +----------------------------+-------------------------------+---------------------------+
        | :func:`nugget`             | ``'nugget'``                  | - ``nugget``              |
        |                            |                               | - ``sill``                |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`pentaspherical`     | ``'pentaspherical'``          | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`power`              | ``'power'``                   | - ``nugget``              |
        |                            |                               | - ``sill``                |
        |                            |                               | - ``power_exponent``      |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`quadratic`          | ``'quadratic'``               | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        |                            |                               | - ``shape_parameter``     |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`sinc`               | ``'sinc'``                    | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``hole_effect_range``   |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`spherical`          | ``'spherical'``               | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`bessel_exponential` | ``['bessel', 'exponential']`` | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        |                            |                               | - ``decay_power``         |
        |                            |                               | - ``hole_effect_range``   |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`bessel_gaussian`    | ``['bessel', 'gaussian']``    | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        |                            |                               | - ``decay_power``         |
        |                            |                               | - ``hole_effect_range``   |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`cosine_exponential` | ``['cosine', 'exponential']`` | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        |                            |                               | - ``hole_effect_range``   |
        |                            |                               | - ``enhance_semivariance``|
        +----------------------------+-------------------------------+---------------------------+
        | :func:`cosine_gaussian`    | ``['cosine', 'gaussian']``    | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        |                            |                               | - ``hole_effect_range``   |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`exponential_linear` | ``['exponential', 'linear']`` | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        |                            |                               | - ``hole_effect_range``   |
        |                            |                               | - ``decay_power``         |
        +----------------------------+-------------------------------+---------------------------+
        | :func:`gaussian_linear`    | ``['gaussian', 'linear']``    | - ``sill``                |
        |                            |                               | - ``nugget``              |
        |                            |                               | - ``correlation_range``   |
        |                            |                               | - ``hole_effect_range``   |
        +----------------------------+-------------------------------+---------------------------+

    Returns
    -------
    variogram : np.ndarray
        An array containing the (normalized) semivariance for each lag bin.
    """
    # Determine model source
    if model is not None:
        # ---- Get the variogram arguments and function from `model`
        model_source = model
    elif variogram_parameters is not None:
        # ---- Get the variogram arguments and function from `variogram_parameters`
        model_source = variogram_parameters["model"]
    else:
        raise ValueError("Argument `model` is missing.")

    # Extract function signatures
    variogram_args, variogram_function = get_variogram_arguments(model_source)

    # Evaluate whether required function parameters are present
    if variogram_parameters is not None:
        # ---- Get input arguments
        input_args = variogram_parameters
        # ---- Use `variogram_parameters` as source
        arg_diff = set(list(variogram_args)) - set(input_args) - set(["distance_lags"])
    else:
        # ---- Get input arguments
        input_args = kwargs
        # ---- Use `kwargs` as source
        arg_diff = set(list(variogram_args)) - set(input_args) - set(["distance_lags"])

    # Raise error if any are missing
    if len(arg_diff) > 0:
        raise ValueError(
            f"The following variogram parameters are missing: {', '.join(list(arg_diff))}"
        )

    # Filter out only the variogram parameters required for the model
    required_args = dict((k, input_args[k]) for k in input_args if k in list(variogram_args))

    # Pipe the parameters into the appropriate variogram function
    return variogram_function["model_function"](distance_lags, **required_args)


def get_variogram_arguments(model_name: str | list[str]):
    """
    Get the variogram function arguments and model function for a given model.

    Parameters
    ----------
    model_name : Union[str, List[str]]
        A string or list of model names. A single name represents a single family model.
        Two inputs represent the desired composite model (e.g. the composite J-Bessel and
        exponential model).

    Returns
    -------
    Tuple[inspect.Signature.parameters, Dict[str, Any]]
        A tuple containing the function signature parameters and a dictionary with the
        model function.
    """
    # Convert to lowercase to match reference model dictionary
    if isinstance(model_name, str):
        model_input = model_name.lower()
    elif isinstance(model_name, list) & len(model_name) == 1:
        model_input = "".join(model_name).lower()
    else:
        model_input = [name.lower() for name in model_name]
        # ---- Alphabetic sort
        model_input.sort()

    # Parse user input from reference model dictionary
    # ---- Check against VARIOGRAM_MODELS API to ensure model exists
    if isinstance(model_input, list) and (tuple(model_input) in VARIOGRAM_MODELS["composite"]):
        # if (len(model_input) > 1) & (tuple(model_input) in VARIOGRAM_MODELS["composite"]):
        # ---- Parse model function
        model_function = VARIOGRAM_MODELS["composite"][tuple(model_input)]
    # elif (len([model_input]) == 1) & (model_input in VARIOGRAM_MODELS["single"]):
    elif not isinstance(model_input, list) and (model_input in VARIOGRAM_MODELS["single"]):
        # ---- Parse model function
        model_function = VARIOGRAM_MODELS["single"][model_input]
    else:
        raise LookupError(
            f"The model input ({model_name}) could not be matched to an"
            f" existing variogram method."
        )

    # Check input parameters against the required function arguments
    # ---- Get the function signature
    function_signature = inspect.signature(model_function)
    # ---- Create ordered dictionary of required arguments
    return function_signature.parameters, {"model_function": model_function}


def fit_variogram(
    lags: np.ndarray[float],
    lag_counts: np.ndarray[int],
    gamma: np.ndarray[float],
    model_parameters: Parameters,
    model: str | list[str] | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float, float]:
    r"""
    Fit theoretical variogram models to empirical semivariogram data using weighted least squares.

    This function performs non-linear optimization to find the best-fitting parameters for
    theoretical variogram models. The optimization uses weighted least squares where weights
    are proportional to the lag counts, giving more influence to lags with more data pairs.

    Parameters
    ----------
    lags : |np.ndarray[float]|
        Array of lag distances from empirical variogram computation.
    lag_counts : |np.ndarray[int]|
        Number of data point pairs contributing to each lag estimate. Used as optimization weights.
    gamma : |np.ndarray[float]|
        Empirical semivariogram values (standardized semivariance) at each lag.
    model : str or List[str]
        Theoretical variogram model specification. Single string for basic models
        (e.g., ``'exponential'``, ``'gaussian'``, ``'spherical'``, ``'jbessel'``, ``'linear'``).
        List of two strings for composite models (e.g., ``['bessel', 'exponential']``,
        ``['bessel', 'gaussian']``, ``['cosine', 'exponential']``).
    model_parameters : lmfit..parameters.Parameters
        Parameter object containing initial values, bounds, and constraints for optimization.
        Required parameters depend on the selected model.
    optimizer_kwargs : dict, default={}
        Additional keyword arguments passed to :func:`lmfit.minimizer.minimize()`. Common options
        include ``'max_nfev'`` for maximum function evaluations and solver-specific parameters.

    Returns
    -------
    Tuple[dict, dict, float]

        - Optimized parameter values as dictionary

        - Initial parameter values as dictionary

        - Mean absolute deviation of optimized fit

    Notes
    -----
    The optimization minimizes the weighted residual sum of squares:

    .. math::
        \\min_{\\theta} \\sum_{b=1}^{n} \\hat{w}_b \\left[ \\gamma_\\text{empirical}(h_b) -
        \\gamma_\\text{model}(h_b; \\theta) \\right]^2

    where :math:`n` is the number of lag bins and :math:`b` indexes those lag bins
    (:math:`b=1, \\dots, n`). The weights :math:`\\hat{w}_b` are associated with each lag bins
    and are normalized where:

    .. math::
        \\hat{w}_b = \\frac{N(h_b)}{\\sum\\limits_{b=1}^{n} N(h_b)}

    The function uses Trust Region Reflective algorithm (default in `lmfit.minimizer.minimize()`)
    which handles parameter bounds robustly and is suitable for the non-linear nature of variogram
    models.

    References
    ----------
    .. [1] Cressie, N. (1993). Statistics for Spatial Data. Wiley.
    .. [2] Newville, M., et al. (2014). LMFIT: Non-Linear Least-Square Minimization and
           Curve-Fitting for Python. Zenodo.
    """
    # Apply mutable default guards
    model = model if model is not None else ["bessel", "exponential"]
    optimizer_kwargs = optimizer_kwargs or {}

    # Normalize the lag counts to get the lag weights
    lag_weights = lag_counts / lag_counts.sum()

    # Vertically stack the lags, semivariance, and weights
    data_stack = np.vstack((lags, gamma, lag_weights))

    # Recover the lag resolution
    delta_lag = np.diff(lags).mean()

    # Compute the range
    range = lags.max() + delta_lag

    # Index lag distances that are within the parameterized range
    within_range = np.where(lags <= range)[0]

    # Truncate the data stack
    truncated_stack = data_stack[:, within_range]

    # Create helper cost-function that is weighted using the kriging weights (`w`), lag
    # distances (`x`), and empirical semivariance (`y`)
    def cost_function(parameters, x, y, w, model):
        yr = compute_variogram(x, {**parameters, **{"model": model}})
        return (yr - y) * w

    # Compute the initial fit based on the pre-optimized parameter values
    initial_fit = cost_function(
        model_parameters,
        x=truncated_stack[0],
        y=truncated_stack[1],
        w=truncated_stack[2],
        model=model,
    )

    # Compute the initial mean absolute deviation (MAD)
    mad_initial = np.mean(np.abs(initial_fit))

    # Generate `Minimizer` function class required for bounded optimization
    minimizer = Minimizer(
        cost_function,
        model_parameters,
        fcn_args=(truncated_stack[0], truncated_stack[1], truncated_stack[2], model),
    )

    # Minimize the cost-function to compute the best-fit/optimized variogram parameters
    parameters_optimized = minimizer.minimize(method="least_squares", **optimizer_kwargs)

    # Calculate the optimized MAD
    mad_optimized = np.mean(np.abs(parameters_optimized.residual))

    # Extract the best-fit parameter values
    best_fit_params = parameters_optimized.params.valuesdict()

    # Return the final tuple
    return best_fit_params, mad_initial, mad_optimized
