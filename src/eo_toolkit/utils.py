import pandas as pd
import numpy as np


def compute_monthly_median(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Resamples and aggregates a DataFrame column into a monthly time series using the median.

    Computes one value per calendar month (e.g., 2014-01, 2014-02, ...) as the
    median of the available data points for each month, dropping missing values.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing a 'date' column and the target metric column.
    col : str
        The name of the column to resample and aggregate.

    Returns:
    --------
    pd.Series
        A monthly time series indexed by month-start timestamps ('MS') containing the median values.
    """
    s = df.set_index("date")[col]
    return s.resample("MS").median().dropna()


def compute_patch_series(
    arr: np.ndarray,
    dates: np.ndarray,
) -> pd.DataFrame:
    """
    Computes spatial mean time series from a 4D tensor patch.

    Averaging is performed in linear power space and then converted back to dB
    (since dB values must not be averaged directly). NaN pixels are ignored.

    Formulas:
        - Linear conversion: linear = 10^(dB / 10)
        - RVI (Radar Vegetation Index): RVI = (4 * VH) / (VV + VH)

    Parameters:
    -----------
    arr : np.ndarray
        4D NumPy tensor of shape (T, 2, H, W) containing VV (band 0) and VH (band 1) in dB.
    dates : np.ndarray or list
        1D array of timestamps corresponding to each time step.

    Returns:
    --------
    pd.DataFrame
        A DataFrame sorted by date containing columns:
        - 'date': Timestamp of acquisition
        - 'vv_db': Mean VV backscatter in dB
        - 'vh_db': Mean VH backscatter in dB
        - 'ratio_db': Cross-polarization difference/ratio in dB (vh_db - vv_db)
        - 'rvi': Radar Vegetation Index (dimensionless, between 0 and ~1)
    """
    # Convert dB to linear power: lin = 10^(dB / 10)
    lin = 10.0 ** (np.asarray(arr, dtype=np.float64) / 10.0)  # Shape: (T, 2, H, W)

    # Compute spatial mean ignoring NaNs over height and width (axes 2 and 3)
    mean_lin = np.nanmean(lin, axis=(2, 3))  # Shape: (T, 2)
    vv_lin = mean_lin[:, 0]
    vh_lin = mean_lin[:, 1]

    # Safeguard against zero or negative values before log/division operations
    eps = 1e-12
    vv_lin = np.maximum(vv_lin, eps)
    vh_lin = np.maximum(vh_lin, eps)

    # Construct DataFrame with derived metrics
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "vv_db": 10.0 * np.log10(vv_lin),
        "vh_db": 10.0 * np.log10(vh_lin),
        "ratio_db": 10.0 * np.log10(vh_lin / vv_lin),
        "rvi": 4.0 * vh_lin / (vv_lin + vh_lin)
    })

    return df.sort_values("date").reset_index(drop=True)
    
    
def softclip(x, lo=None, hi=None, lo_pct=2, hi_pct=98, smoothness=3.0):
    """Soft clip values using a robust percentile anchor
    and a smooth sigmoidal transition to preserve highlight/shadow detail.
    """
    if lo is None or hi is None:
        lo, hi = np.percentile(x, (lo_pct, hi_pct))

    # Avoid division by zero
    span = (hi - lo) if (hi - lo) > 1e-9 else 1e-9

    # Normalize center around 0 and scale by smoothness
    # We map the [lo, hi] range approximately to [-0.5, 0.5]
    y = (x - lo) / span

    # Apply a smooth sigmoidal transfer function (e.g., scaled tanh or logistic)
    # This smoothly asymptotes towards 0 and 1 without harsh flattening
    rescaled = 0.5 * (np.tanh(smoothness * (y - 0.5)) / np.tanh(0.5 * smoothness) + 1)

    # Fallback clip to strictly keep output in [0, 1] range for matplotlib
    return np.clip(rescaled, 0, 1)    
