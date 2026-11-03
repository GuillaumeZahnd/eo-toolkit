from typing import Any, Dict
import pandas as pd
import numpy as np
import pymannkendall as mk


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
    
    
def compute_residuals(s: pd.Series) -> pd.Series:
    """
    De-seasons and de-trends a monthly time series to extract residual anomalies.
    
    Parameters:
    -----------
    s : pd.Series
        Monthly time series with a DatetimeIndex ('MS' frequency).
        
    Returns:
    --------
    pd.Series
        The residual anomalies with seasonality and linear trend removed.
    """
    # De-seasoning: Subtract the multi-year monthly median for each calendar month
    monthly_baselines = s.groupby(s.index.month).transform("median")
    deseasoned = s - monthly_baselines
    
    # De-trending: Fit a linear trend line via least squares and remove it
    time_idx = np.arange(len(deseasoned))
    valid_mask = deseasoned.notna()
    
    # Fit linear slope and intercept on valid points
    slope, intercept = np.polyfit(time_idx[valid_mask], deseasoned[valid_mask], 1)
    trend_component = pd.Series(slope * time_idx + intercept, index=deseasoned.index)
    
    # Residuals
    residuals = deseasoned - trend_component
    return residuals    
