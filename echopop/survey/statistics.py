"""Bootstrap and confidence interval statistics for stratified survey estimates."""

from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats as st


def bc(
    samples: np.ndarray[float],
    interval: np.ndarray[float],
    population_statistic: float,
    estimator_name: str,
    **kwargs,
) -> np.ndarray[float]:
    """
    Compute bias-corrected (BC) bootstrap confidence interval.

    The BC method adjusts for bias in the bootstrap distribution by using the
    empirical cumulative distribution function (ECDF) to estimate the bias-correction
    factor z0.

    Parameters
    ----------
    samples : np.ndarray
        Bootstrap sample replicates.
    interval : np.ndarray
        Array of percentile values (between 0 and 1) for confidence interval bounds.
    population_statistic : float
        The original (non-bootstrap) population statistic for bias correction.
    estimator_name : str, optional
        Name of the estimator for error reporting.
    **kwargs
        Additional keyword arguments (unused but kept for compatibility).

    Returns
    -------
    np.ndarray
        Bias-corrected confidence interval bounds.

    Raises
    ------
    ValueError
        If the bootstrap distribution is too skewed (ECDF is 0.0 or 1.0).

    References
    ----------
    Efron, B. (1981). Nonparametric standard errors and confidence intervals.
    *Canadian Journal of Statistics*, *9*(2), 139-158.
    """
    # Check for NaN or infinite values in samples
    if not np.all(np.isfinite(samples)) or not np.isfinite(population_statistic):
        return np.full(len(interval), np.nan)

    # Compute the ECDF and proportion of values below the population estimate
    ecdf = np.mean(samples < population_statistic)

    # Edge case where `ecdf == 0.0` or `ecdf == 1.0`
    if ecdf in [0.0, 1.0]:
        error_msg = (
            f"Significant skewness detected in '{estimator_name}' when attempting to "
            f"calculate the bias-corrected (BC) confidence interval. The estimated "
            f"Z-statistic was either degenerate or too skewed to produce a reasonable "
            f"interval estimate. Please use a different method for 'ci_method'."
        )
        raise ValueError(error_msg)

    # Calculate the Z-statistic
    z0 = st.norm.ppf(ecdf)

    # Estimate the bias-corrected percentiles using the Normal CDF
    percentiles_cdf = st.norm.cdf(2 * z0 + st.norm.ppf(interval))

    # Return the confidence interval
    return np.quantile(samples, percentiles_cdf)


def bca(
    samples: np.ndarray[float],
    interval: np.ndarray[float],
    population_statistic: float,
    estimator_name: str,
    **kwargs,
):
    """
    Compute bias-corrected and accelerated (BCa) bootstrap confidence interval.

    The BCa method extends the BC method by also correcting for skewness in the bootstrap
    distribution using an acceleration parameter estimated via finite-sample jackknife
    resampling^[2].

    Parameters
    ----------
    samples : np.ndarray
        Bootstrap sample replicates.
    interval : np.ndarray
        Array of percentile values (between 0 and 1) for confidence interval bounds.
    population_statistic : float
        The original (non-bootstrap) population statistic for bias correction.
    estimator_name : str
        Name of the estimator for error reporting.
    **kwargs
        Additional keyword arguments (unused but kept for compatibility).

    Returns
    -------
    np.ndarray
        Bias-corrected and accelerated confidence interval bounds.

    Raises
    ------
    ValueError
        If the bootstrap distribution is too skewed (ECDF is 0.0 or 1.0).

    Notes
    -----
    The acceleration parameter (a-hat) is computed using leave-one-out jackknife resampling to
    estimate the influence function. For each jackknife replicate, the influence values are
    calculated as I_i = (n-1)(θ̂ - θ̂_(i)), where θ̂_(i) is the statistic computed from the sample
    with the i-th observation removed^[2]. The acceleration constant is then estimated as:
    a = Σ(I_i)³ / (6 x (Σ(I_i)²)^1.5)

    References
    ----------
    .. [1] Efron, B., and Tibshirani, R.J. (1993). *An introduction to the Bootstrap*.
       Springer US. https://doi.org/10.1007/978-1-4899-4541-9

    .. [2] Frangos, C.C., and Schucany, W.R. (1990). Jackknife estimation of the bootstrap
        acceleration constant. *Computational Statistics & Data Analysis*, *9*(3), 271-281.
        https://doi.org/10.1016/0167-9473(90)90109-U
    """
    # Check for NaN or infinite values in samples
    if not np.all(np.isfinite(samples)) or not np.isfinite(population_statistic):
        return np.full(len(interval), np.nan)

    # Compute the ECDF and proportion of values below the population estimate
    ecdf = np.mean(samples < population_statistic)

    # Edge case where `ecdf == 0.0` or `ecdf == 1.0`
    if ecdf in [0.0, 1.0]:
        error_msg = (
            f"Significant skewness detected in '{estimator_name}' when attempting to "
            f"calculate the bias-corrected (BC) confidence interval. The estimated "
            f"Z-statistic was either degenerate or too skewed to produce a reasonable "
            f"interval estimate. Please use a different method for 'ci_method'."
        )
        raise ValueError(error_msg)

    # Calculate the Z-statistic
    z0 = st.norm.ppf(ecdf)

    # Create helper function to compute the finite-sample jackknife estimator
    def calc_theta(values):
        return values.var(ddof=1) / values.mean() ** 2 - 1 / values.mean()

    # Calculate the global theta
    theta_global = calc_theta(samples)

    # Calculate theta for each jackknife replicate (ie LOO resampling)
    theta_jackknife = np.array([calc_theta(np.delete(samples, i)) for i in range(len(samples))])

    # Compute the influences
    influence = (len(samples) - 1) * (theta_global - theta_jackknife)

    # Calculate the acceleration term (a_hat)
    a_hat = (influence**3).sum() / (6.0 * (influence**2).sum() ** 1.5)

    # Calculate the bias-corrected percentiles from the Normal CDF with acceleration term
    percentiles_cdf = st.norm.cdf(
        z0 + (z0 + st.norm.ppf(interval)) / (1 - a_hat * (z0 + st.norm.ppf(interval)))
    )

    # Return the bias-corrected and accelerated confidence interval
    return np.quantile(samples, percentiles_cdf)


def empirical(
    samples: np.ndarray[float],
    interval: np.ndarray[float],
    population_statistic: float,
    **kwargs,
):
    """
    Compute empirical bootstrap confidence interval.

    The empirical method uses the distribution of bootstrap deviations from the
    population statistic to construct confidence intervals.

    Parameters
    ----------
    samples : np.ndarray
        Bootstrap sample replicates.
    interval : np.ndarray
        Array of percentile values (between 0 and 1) for confidence interval bounds.
    population_statistic : float
        The original (non-bootstrap) population statistic.
    **kwargs
        Additional keyword arguments (unused but kept for compatibility).

    Returns
    -------
    np.ndarray
        Empirical confidence interval bounds.

    References
    ----------
    Efron, B. (1981). Nonparametric standard errors and confidence intervals. *Canadian Journal
    of Statistics*, *9*(2), 139-158. https://doi.org/10.2307/3314608
    """
    # Check for NaN or infinite values in samples
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(population_statistic)):
        return np.full(len(interval), np.nan)

    # Adjust shape of 'population_statistic' if needed
    if len(population_statistic) == 1 and isinstance(
        population_statistic, pd.Series | pd.DataFrame
    ):
        population_statistic = population_statistic.to_numpy()

    # Compute the delta deviation term
    delta = samples - population_statistic

    # Get the quantiles
    quantiles = np.quantile(delta, interval)

    # Add the deviations
    return samples.mean() + quantiles


def normal(
    samples: np.ndarray[float],
    interval: np.ndarray[float],
    **kwargs,
):
    """
    Compute confidence interval assuming a Normal distribution.

    This method constructs confidence intervals using the normal approximation
    with the sample mean and standard deviation.

    Parameters
    ----------
    samples : np.ndarray
        Bootstrap sample replicates.
    interval : np.ndarray
        Array of percentile values (between 0 and 1) for confidence interval bounds.
    **kwargs
        Additional keyword arguments (unused but kept for compatibility).

    Returns
    -------
    np.ndarray
        Normal-based confidence interval bounds.

    Notes
    -----
    This method assumes the bootstrap distribution is approximately normal.
    Use with caution for highly skewed distributions.

    References
    ----------
    DiCiccio, T.J., and Efron, B. (1996). Bootstrap confidence intervals. *Statistical Science*,
    *11*(3). https://doi.org/10.1214/ss/1032280214
    """
    # Check for NaN or infinite values in samples
    if not np.all(np.isfinite(samples)):
        return np.full(len(interval), np.nan)

    # Return the Normal confidence interval
    return samples.mean() + st.norm.ppf(interval) * samples.std(ddof=1)


def percentile(
    samples: np.ndarray[float],
    interval: np.ndarray[float],
    **kwargs,
):
    """
    Compute percentile bootstrap confidence interval.

    The percentile method directly uses the quantiles of the bootstrap distribution
    as confidence interval bounds.

    Parameters
    ----------
    samples : np.ndarray
        Bootstrap sample replicates.
    interval : np.ndarray
        Array of percentile values (between 0 and 1) for confidence interval bounds.
    **kwargs
        Additional keyword arguments (unused but kept for compatibility).

    Returns
    -------
    np.ndarray
        Percentile confidence interval bounds.

    Notes
    -----
    This is the simplest bootstrap confidence interval method but may not
    perform well for biased estimators.

    References
    ----------
    Efron, B. (1981). Nonparametric standard errors and confidence intervals. *Canadian Journal of
    Statistics*, *9*(2), 139-158. https://doi.org/10.2307/3314608
    """
    # Check for NaN or infinite values in samples
    if not np.all(np.isfinite(samples)):
        return np.full(len(interval), np.nan)

    # Return the exact percentiles of the bootstrapped distribution
    return np.quantile(samples, interval)


def student_jackknife(
    samples: np.ndarray[float],
    interval: list[float],
    **kwargs,
):
    """
    Compute studentized confidence interval using jackknife resampling.

    This method uses jackknife (leave-one-out) resampling to estimate the
    variance for studentizing the bootstrap distribution, providing improved
    performance for small samples.

    Parameters
    ----------
    samples : np.ndarray
        Bootstrap sample replicates.
    interval : np.ndarray
        Array of percentile values (between 0 and 1) for confidence interval bounds.
    **kwargs
        Additional keyword arguments (unused but kept for compatibility).

    Returns
    -------
    np.ndarray
        Studentized jackknife confidence interval bounds.

    Notes
    -----
    This method is computationally intensive as it requires leave-one-out
    calculations for each bootstrap sample but provides better coverage
    properties than simple t-based methods.

    References
    ----------
    DiCiccio, T.J., and Efron, B. (1996). Bootstrap confidence intervals.
    *Statistical Science*, *11*(3), 189-228.
    """
    # Check for NaN or infinite values in samples
    if not np.all(np.isfinite(samples)):
        return np.full(len(interval), np.nan)

    # Get the sample size
    n = samples.size

    # Handle edge case of insufficient data
    if n < 3:
        return np.full(len(interval), np.nan)

    # Sum the values
    sample_sum = samples.sum()

    # Take the squared-sum of the values
    sample_sq_sum = (samples**2).sum()

    # Leave-one-out (LOO) means
    pseudo_means = (sample_sum - samples) / (n - 1)

    # LOO variance
    pseudo_vars = ((sample_sq_sum - samples**2) - (n - 1) * pseudo_means**2) / (n - 2)

    # Handle edge case where variance is negative or zero
    pseudo_vars = np.maximum(pseudo_vars, 1e-12)

    # LOO standard deviation
    pseudo_stds = np.sqrt(pseudo_vars)

    # Calculate the t-statistic
    t_statistic = (samples - pseudo_means) / pseudo_stds

    # Compute the t-distribution intervals
    t_interval = np.quantile(t_statistic, interval)

    # Apply to the original mean to get the studentized confidence interval
    return samples.mean() + t_interval * samples.std(ddof=1)


def student_standard(
    samples: np.ndarray[float],
    interval: np.ndarray[float],
    **kwargs,
):
    """
    Compute studentized confidence interval assuming a t-distribution.

    This method uses the t-distribution with degrees of freedom equal to
    the sample size minus one to construct confidence intervals.

    Parameters
    ----------
    samples : np.ndarray
        Bootstrap sample replicates.
    interval : np.ndarray
        Array of percentile values (between 0 and 1) for confidence interval bounds.
    **kwargs
        Additional keyword arguments (unused but kept for compatibility).

    Returns
    -------
    np.ndarray
        t-distribution based confidence interval bounds.

    Notes
    -----
    This method is appropriate when the bootstrap distribution is approximately
    t-distributed. It's simpler than jackknife studentization but may have
    worse coverage properties for small samples.

    References
    ----------
    DiCiccio, T.J., and Efron, B. (1996). Bootstrap confidence intervals. *Statistical Science*,
    *11*(3). https://doi.org/10.1214/ss/1032280214
    """
    # Check for NaN or infinite values in samples
    if not np.all(np.isfinite(samples)):
        return np.full(len(interval), np.nan)

    return samples.mean() + st.t.ppf(interval, len(samples) - 1) * samples.std(ddof=1)


# Bootstrap confidence interval methods reference dictionary
BOOTSTRAP_CI_METHODS = {
    "bc": bc,
    "bca": bca,
    "empirical": empirical,
    "normal": normal,
    "percentile": percentile,
    "t": student_standard,
    "t-jackknife": student_jackknife,
}


def confidence_interval(
    bootstrap_samples: pd.Series | pd.DataFrame,
    population_values: pd.Series | pd.DataFrame,
    ci_method: Literal[
        "bc",
        "bca",
        "empirical",
        "normal",
        "percentile",
        "t",
        "t-jackknife",
    ],
    ci_percentile: float = 0.95,
):
    """
    Calculate confidence intervals for bootstrap samples using various methods.

    This function provides multiple methods for computing confidence intervals
    from bootstrap replicates, including bias-corrected methods and studentized
    approaches.

    Parameters
    ----------
    bootstrap_samples : Union[pd.Series, pd.DataFrame]
        DataFrame or Series containing bootstrap replicates. Each column represents a
        different variable, and each row represents a bootstrap replicate.
    population_values : Union[pd.Series, pd.DataFrame]
        DataFrame or Series containing the original (non-bootstrap) population statistics
        for each variable. Must have the same column structure as bootstrap_samples.
    ci_method : {"bc", "bca", "empirical", "normal", "percentile", "t", "t-jackknife"}
        Method for computing the bootstrap confidence interval:

        - "bc": Bias-corrected method[1]_

        - "bca": Bias-corrected and accelerated method[2]_

        - "empirical": Empirical method using bootstrap deviations[1]_

        - "normal": Normal approximation[3]_

        - "percentile": Simple percentile method[1]_

        - "t": t-distribution based[4]_

        - "t-jackknife": Jackknife studentized method[4]_

    ci_percentile : float, default 0.95
        Confidence level (between 0 and 1). For example, 0.95 gives a 95% CI.

    Returns
    -------
    pd.DataFrame

        DataFrame containing confidence interval results with the following structure:

        - If population_values has additional column index levels: Returns a DataFrame
          with confidence interval bounds and statistics stacked appropriately

        - If population_values has simple columns: Returns a transposed DataFrame
          with metrics as columns

    Raises
    ------
    TypeError
        If population_values is not a pandas DataFrame or Series.
    ValueError
        If ci_method is not one of the supported methods.
    KeyError
        If bootstrap_samples and population_values don't have matching columns.

    Notes
    -----
    The function automatically handles different DataFrame structures and
    provides appropriate output formatting. Bias calculations are included
    in the output for all methods.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> bootstrap_data = pd.DataFrame({
    ...     'biomass': np.random.normal(100, 10, 1000),
    ...     'density': np.random.normal(50, 5, 1000)
    ... })
    >>> population_data = pd.DataFrame({
    ...     'biomass': [95],
    ...     'density': [48]
    ... })
    >>> ci_results = confidence_interval(
    ...     bootstrap_data, population_data,
    ...     ci_method='percentile', ci_percentile=0.95
    ... )

    References
    ----------
    .. [1] Efron, B. (1981). Nonparametric standard errors and confidence intervals.
        *Canadian Journal of Statistics*, 9(2), 139-158.

    .. [2] Efron, B., and Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*.
        Springer US. https://doi.org/10.1007/978-1-4899-4541-9

    .. [3] Efron, B., and Tibshirani, R.J. (1986). Bootstrap methods for standard errors,
        confidence intervals, and other measures of statistical accuracy.
        *Statistical Science*, 1(1), 54-75.

    .. [4] DiCiccio, T.J., and Efron, B. (1996). Bootstrap confidence intervals.
        *Statistical Science*, 11(3), 189-228.
    """
    # Validate inputs
    if not isinstance(population_values, pd.DataFrame | pd.Series):
        raise TypeError("Input for population_values must be a pandas.Series or pandas.DataFrame.")

    if ci_method not in BOOTSTRAP_CI_METHODS:
        raise ValueError(
            f"ci_method '{ci_method}' not supported. "
            f"Available methods: {list(BOOTSTRAP_CI_METHODS.keys())}"
        )

    # Compute the interval percentiles
    interval = np.array([(1 - ci_percentile) / 2, (1 + ci_percentile) / 2])

    # Compute the bootstrap means
    bs_mean = bootstrap_samples.mean()

    # Get population mean
    if isinstance(population_values, pd.DataFrame | pd.Series):
        pop_mean = population_values.mean()
    else:
        raise TypeError(
            "Input for population data for population-level values must either be a "
            "`pandas.Series` or an indexed `pandas.DataFrame`."
        )

    # Compute confidence intervals
    ci_est = bootstrap_samples.apply(
        lambda col: BOOTSTRAP_CI_METHODS[ci_method](
            samples=col,
            interval=interval,
            population_statistic=population_values[col.name],
            estimator_name=col.name,
        ),
        axis=0,
    )
    # ---- Rename the index
    ci_est.index = ["low", "high"]

    # Pivot the means and bias estimates
    ci_stats = pd.DataFrame({"mean": bs_mean}).T
    # ---- Add bias
    ci_stats = pd.concat(
        [
            ci_stats,
            pd.DataFrame(
                {"bias": bs_mean.to_numpy() - pop_mean.to_numpy()}, index=ci_stats.columns
            ).T,
        ]
    )

    # Add mean value to the ci estimates and reorder
    ci_output = pd.concat([ci_est, ci_stats], axis=0).reindex(index=["low", "mean", "high", "bias"])
    # ---- Rename the index
    ci_output.index = ci_output.index.set_names(["metric"])

    # Handle output formatting based on population_values structure
    if hasattr(population_values, "columns") and hasattr(population_values.columns, "names"):
        col_idx_name = [
            name for name in population_values.columns.names if name not in [None, "variable"]
        ]

        # Stack and unstack to create appropriate MultiIndex structure
        if len(col_idx_name) > 0:
            return ci_output.stack(col_idx_name, future_stack=True).unstack(["metric"])

    # For simple case, transpose to get metrics as columns
    return ci_output.unstack().to_frame().T
