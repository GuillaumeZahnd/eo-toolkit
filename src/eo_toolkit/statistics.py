from typing import Any, Dict
import pandas as pd
import numpy as np
import pymannkendall as mk
from numpy.typing import NDArray

from .utils import slice_tensor_with_time_interval


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

    n_time, n_bands, height, width = baseline_data_tensor.shape

    X_cols = [np.ones_like(baseline_t)]
    for k in range(1, nb_harmonics + 1):
        X_cols.append(np.cos(2.0 * np.pi * k * baseline_t))
        X_cols.append(np.sin(2.0 * np.pi * k * baseline_t))

    X = np.column_stack(X_cols)
    nb_coefficients = X.shape[1]

    reshaped_tensor = baseline_data_tensor.reshape(n_time, n_bands * height * width)
    valid_time_mask = ~np.isnan(reshaped_tensor).any(axis=1)

    X_clean = X[valid_time_mask]
    Y_clean = reshaped_tensor[valid_time_mask]

    beta_matrix, _, _, _ = np.linalg.lstsq(X_clean, Y_clean, rcond=None)
    baseline_coefficients = beta_matrix.T.reshape(n_bands, height, width, nb_coefficients)

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
