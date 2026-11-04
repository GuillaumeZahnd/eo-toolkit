from typing import Any, Dict
import pandas as pd
import numpy as np
import pymannkendall as mk
from numpy.typing import NDArray

from .utils import slice_tensor_with_time_interval


def compute_one_sided_cusum(
    residuals: NDArray[np.float32],
    slack: float = 0.0
) -> NDArray[np.float32]:
    """Computes the one-sided upper CuSum chart for a 4D residual tensor
    to track persistent negative shifts (declines) in radar backscatter/indices.

    Formula: S_t = max(0, S_{t-1} + residual_t - slack)

    Parameters:
    - residuals: 4D array of shape (time, bands, height, width) representing (baseline - observed).
    - slack: Allowance parameter (k) to filter out minor high-frequency noise fluctuations.

    Returns:
    - NDArray: 4D CuSum statistic tensor of shape (time, bands, height, width).
    """
    n_time, n_bands, height, width = residuals.shape
    cusum_tensor = np.empty_like(residuals, dtype=np.float32)

    # Initialize the CuSum accumulator at time t=0 to zero for all pixels/bands
    current_cusum = np.zeros((n_bands, height, width), dtype=np.float32)

    for t in range(n_time):
        # Accumulate positive deviations, pulling back by the slack factor,
        # and clamping at 0 to prevent negative accumulation.
        current_cusum = np.maximum(0.0, current_cusum + residuals[t] - slack)
        cusum_tensor[t] = current_cusum

    return cusum_tensor


def bootstrap_cusum_paths(
    data_tensor: NDArray[np.float32],
    dates: NDArray[np.datetime64] | list[str],
    coefficients: NDArray[np.float64],
    start_date: str,
    end_date: str,
    nb_harmonics: int = 2,
    nb_bootstraps: int = 1000,
    block_size: int = 12,
    slack: float = 0.1,
    seed: int = 0
) -> NDArray[np.float32]:
    """Generates synthetic bootstrap replicates of CuSum trajectories from baseline residuals
    using moving block bootstrap to preserve temporal autocorrelation.

    Parameters:
    - data_tensor: 4D array of shape (time, bands, height, width) for the full dataset.
    - dates: Array or list of timestamps matching the time dimension.
    - coefficients: Fitted harmonic coefficients of shape (bands, height, width, n_coeffs).
    - start_date: Start date string for the baseline period.
    - end_date: End date string for the baseline period.
    - nb_harmonics: Number of harmonic components used in the baseline model.
    - nb_bootstraps: Number of bootstrap sample paths to generate.
    - block_size: Length of temporal blocks for resampling (in number of observations).
    - slack: Allowance parameter (k) for the CuSum accumulator.
    - seed: Random seed for reproducibility.

    Returns:
    - NDArray: Resampled bootstrap CuSum tensor of shape (nb_bootstraps, time, bands, height, width).
    """
    np.random.seed(seed)

    # Extract baseline residuals from the reference healthy period
    _, baseline_residuals, _, _ = compute_expected_baseline_and_residuals(
        data_tensor=data_tensor,
        dates=dates,
        coefficients=coefficients,
        start_date=start_date,
        end_date=end_date,
        nb_harmonics=nb_harmonics
    )

    nb_timesteps, nb_bands, height, width = baseline_residuals.shape

    # Storage for bootstrap CuSum paths
    bootstrap_cusum_samples = np.empty(
        (nb_bootstraps, nb_timesteps, nb_bands, height, width),
        dtype=np.float32
    )

    # Number of blocks required to cover the time dimension
    nb_blocks = int(np.ceil(nb_timesteps / block_size))

    for b in range(nb_bootstraps):
        # Randomly select starting indices for the blocks
        block_starts = np.random.randint(0, nb_timesteps - block_size + 1, size=nb_blocks)

        # Construct synthetic time index path by stitching blocks together
        resampled_indices = []
        for start_idx in block_starts:
            resampled_indices.extend(range(start_idx, start_idx + block_size))

        # Trim to exact original time length
        resampled_indices = resampled_indices[:nb_timesteps]

        # Extract resampled raw residuals for this bootstrap replicate
        resampled_residuals = baseline_residuals[resampled_indices, :, :, :]

        # Run CuSum accumulation on the synthetic resampled path
        bootstrap_cusum_samples[b] = compute_one_sided_cusum(resampled_residuals, slack=slack)

    return bootstrap_cusum_samples


def compute_expected_baseline_and_residuals(
    data_tensor: NDArray[np.float32],
    dates: NDArray[np.datetime64] | list[str],
    coefficients: NDArray[np.float64],
    start_date: str,
    end_date: str,
    nb_harmonics: int = 2
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float32], NDArray[np.datetime64]]:
    """Computes expected baseline values using fitted harmonic coefficients, along with residuals.

    Parameters:
    - data_tensor: 4D array of shape (time, bands, height, width) for the full dataset.
    - dates: Array or list of timestamps matching the time dimension.
    - coefficients: Fitted harmonic coefficients of shape (bands, height, width, n_coeffs).
    - start_date: Start date string for the monitoring/evaluation period.
    - end_date: End date string for the monitoring/evaluation period.
    - nb_harmonics: Number of harmonic components used in the baseline model.

    Returns:
    - tuple containing:
        - baseline_expected: Expected baseline tensor of shape (time, bands, height, width).
        - residuals: Deviation tensor (baseline - observed) of shape (time, bands, height, width).
        - monitoring_data_tensor: Sliced data tensor for the evaluation period.
        - monitoring_dates: Sliced datetime array.
    """
    # Slice the temporal window (monitoring period)
    monitoring_data_tensor, monitoring_dates, monitoring_t = slice_tensor_with_time_interval(
        data_tensor, dates, start_date, end_date
    )

    # Construct the design matrix X for the target dates
    X_cols = [np.ones_like(monitoring_t)]
    for k in range(1, nb_harmonics + 1):
        X_cols.append(np.cos(2.0 * np.pi * k * monitoring_t))
        X_cols.append(np.sin(2.0 * np.pi * k * monitoring_t))
    X = np.column_stack(X_cols)  # Shape: (n_target_time, n_coeffs)

    # Compute expected baseline values (across all pixels and bands simultaneously).
    # X has shape (T, C), coefficients has shape (B, H, W, C), outcome has shape (T, B, H, W).
    baseline_expected = np.einsum('tc,bhwc->tbhw', X, coefficients)

    # Calculate residuals against the sliced monitoring tensor (fixing shape mismatch)
    residuals = baseline_expected - monitoring_data_tensor

    return baseline_expected, residuals, monitoring_data_tensor, monitoring_dates


def compute_harmonic_baselines(
    data_tensor: NDArray[np.float32],
    dates: NDArray[np.datetime64] | list[str],
    start_date: str,
    end_date: str,
    nb_harmonics: int = 2
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.datetime64], NDArray[np.float64]]:
    """Computes vectorized harmonic baseline coefficients and returns the sliced
    baseline tensor, coefficients, and time metadata.

    Parameters:
    - data_tensor: 4D array of shape (time, bands, height, width).
    - dates: Array or list of timestamps matching the time dimension.
    - start_date: Start date string for the baseline period.
    - end_date: End date string for the baseline period.
    - nb_harmonics: Number of harmonic components to fit per pixel.

    Returns:
    - tuple containing:
        - baseline_data_tensor: Sliced 4D tensor for the baseline window.
        - baseline_coefficients: 4D array of shape (bands, height, width, nb_coefficients).
        - baseline_dates: Sliced datetime array.
        - baseline_t: 1D array of fractional years.
    """
    # Slice the temporal window (baseline period) internally
    baseline_data_tensor, baseline_dates, baseline_t = slice_tensor_with_time_interval(
        data_tensor, dates, start_date, end_date
    )

    nb_timesteps, nb_bands, height, width = baseline_data_tensor.shape

    X_cols = [np.ones_like(baseline_t)]
    for k in range(1, nb_harmonics + 1):
        X_cols.append(np.cos(2.0 * np.pi * k * baseline_t))
        X_cols.append(np.sin(2.0 * np.pi * k * baseline_t))

    X = np.column_stack(X_cols)
    nb_coefficients = X.shape[1]

    reshaped_tensor = baseline_data_tensor.reshape(nb_timesteps, nb_bands * height * width)
    valid_time_mask = ~np.isnan(reshaped_tensor).any(axis=1)

    X_clean = X[valid_time_mask]
    Y_clean = reshaped_tensor[valid_time_mask]

    beta_matrix, _, _, _ = np.linalg.lstsq(X_clean, Y_clean, rcond=None)
    baseline_coefficients = beta_matrix.T.reshape(nb_bands, height, width, nb_coefficients)

    return baseline_data_tensor, baseline_coefficients, baseline_dates, baseline_t


def correlated_seasonal_mann_kendall(s: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Performs a Correlated Seasonal Mann-Kendall trend test on a monthly time series.

    Parameters:
    -----------
    s : pd.Series
        A monthly time series with a DatetimeIndex and monthly start ('MS') frequency.
    alpha : float, optional
        The significance level for hypothesis testing (default is 0.05).

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing:
        - 'trend': Qualitative trend classification ('increasing', 'decreasing', or 'no trend').
        - 'p': The p-value of the test.
        - 'sen_slope_per_year': The magnitude of the trend per year (Sen's slope).
        - 'n_months': The total count of valid, non-NaN months analyzed.
    """
    s_freq = s.asfreq("MS")
    r = mk.correlated_seasonal_test(s_freq.to_numpy(), period=12, alpha=alpha)

    return {
        "trend": r.trend,
        "p": r.p,
        "sen_slope_per_year": r.slope,
        "n_months": int(s_freq.notna().sum())
    }


def compute_residuals_and_baseline(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    De-seasons and de-trends a monthly time series to extract residual anomalies
    and the reconstructed baseline model.

    Parameters:
    -----------
    s : pd.Series
        Monthly time series with a DatetimeIndex ('MS' frequency).

    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        - residuals: Residual anomalies with seasonality and linear trend removed.
        - baseline: Expected baseline signal (seasonal multi-year median + linear trend).
    """
    # De-seasoning: Subtract the multi-year monthly median for each calendar month
    monthly_baselines = s.groupby(s.index.month).transform("median")
    deseasoned = s - monthly_baselines

    # De-trending: Fit a linear trend line via least squares and remove it
    time_idx = np.arange(len(deseasoned))
    valid_mask = deseasoned.notna()

    slope, intercept = np.polyfit(time_idx[valid_mask], deseasoned[valid_mask], 1)
    trend_component = pd.Series(slope * time_idx + intercept, index=deseasoned.index)

    # Residuals and reconstructed baseline
    residuals = deseasoned - trend_component
    baseline = s - residuals  # Recombines seasonal baseline + linear trend

    return residuals, baseline
