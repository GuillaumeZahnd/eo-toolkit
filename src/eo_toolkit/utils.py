import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from numpy.typing import NDArray


def bbox_around(lon: float, lat: float, size_m: float) -> tuple[float, float, float, float]:
    """Calculates a lon/lat bounding box of `size_m` metres centered on a point."""
    if size_m <= 0:
        raise ValueError(f"size_m must be positive, got {size_m}.")
    if not -90.0 < lat < 90.0:
        raise ValueError(f"lat must be strictly between -90 and 90, got {lat}.")
    dlat = size_m / 111_320 / 2
    dlon = size_m / (111_320 * math.cos(math.radians(lat))) / 2
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    

def interpolate_tensor_to_regular_grid(
    data_tensor: NDArray[np.float32],
    dates: NDArray[np.datetime64],
    freq: str = "6D"
) -> tuple[NDArray[np.float32], NDArray[np.datetime64]]:
    """
    Interpolates a 4D Sentinel-1 data tensor (time, bands, height, width) 
    from irregular observation dates onto a regular temporal grid.
    
    Parameters:
    - data_tensor: 4D numpy array of shape (time, bands, height, width).
    - dates: Array-like of datetimes corresponding to the time axis.
    - freq: Target frequency string compatible with pandas (e.g., '6D' or '12D').
    
    Returns:
    - interpolated_tensor: 4D numpy array resampled to the regular grid.
    - regular_dates: 1D array of regular datetime64 timestamps.
    """
    # Convert input dates to numeric timestamps for interpolation
    dt_dates = pd.to_datetime(dates)
    x_original = dt_dates.astype(np.int64).values.astype(float)
    
    # Create a regular target date range covering the exact same temporal span
    regular_dt_dates = pd.date_range(start=dt_dates[0], end=dt_dates[-1], freq=freq)
    x_target = regular_dt_dates.astype(np.int64).values.astype(float)
    
    # Use scipy's interp1d along the time axis (axis=0) for multi-dimensional arrays
    interpolator = interp1d(
        x_original, 
        data_tensor, 
        kind="linear", 
        axis=0, 
        bounds_error=False, 
        fill_value="extrapolate"
    )
    
    interpolated_tensor = interpolator(x_target)
    
    return interpolated_tensor.astype(np.float32), regular_dt_dates.values.astype("datetime64[ns]")


def compute_sentinel1_radar_index(
    data_tensor: NDArray[np.float32],
    vv_index: int,
    vh_index: int,
) -> NDArray[np.float32]:
    """Computes the radar polarization index (VH - VV in decibel space),
    preserving the band dimension (reduced to 1).

    Parameters:
    - data_tensor: 4D array of shape (time, bands, height, width).
    - vv_index: Index of the VV band along the band axis.
    - vh_index: Index of the VH band along the band axis.

    Returns:
    - NDArray: 4D array of shape (time, 1, height, width) representing the VH - VV index.
    """
    vv = data_tensor[:, vv_index:vv_index+1, :, :]  # this indexing preserve the band dimension
    vh = data_tensor[:, vh_index:vh_index+1, :, :]  # this indexing preserve the band dimension

    radar_index_tensor = vh - vv
    return radar_index_tensor


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


def slice_tensor_with_time_interval(
    data_tensor: NDArray[np.float32],
    dates: NDArray[np.datetime64] | list[str],
    start_date: str,
    end_date: str
) -> tuple[NDArray[np.float32], NDArray[np.datetime64], NDArray[np.float64]]:
    """Slices a 4D spatial-temporal tensor and its corresponding dates based on a date range.

    Parameters:
    - data_tensor: 4D array of shape (time, bands, height, width).
    - dates: Array or list of timestamps corresponding to the time axis.
    - start_date: Start date string (YYYY-MM-DD).
    - end_date: End date string (YYYY-MM-DD).

    Returns:
    - tuple containing:
        - Sliced data tensor subset.
        - Sliced datetime array.
        - 1D array of fractional years.
    """
    dt_dates = pd.to_datetime(dates)
    time_mask = (dt_dates >= start_date) & (dt_dates <= end_date)
    indices = np.where(time_mask)[0]

    sliced_tensor = data_tensor[indices, :, :, :]
    sliced_dates = dt_dates[indices]
    t_fractional = sliced_dates.year + (sliced_dates.dayofyear - 1) / 365.25

    return sliced_tensor, sliced_dates, t_fractional
