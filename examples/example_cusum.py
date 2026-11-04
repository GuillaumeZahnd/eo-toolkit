import hydra
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from eo_toolkit import (
    load_cached_sentinel1_as_numpy_tensor,
    compute_harmonic_baselines,
    compute_sentinel1_radar_index,
    compute_expected_baseline_and_residuals,
    compute_one_sided_cusum,
    bootstrap_cusum_paths,
    interpolate_tensor_to_regular_grid,
)

from viz import (
    plot_pixelwise_harmonic_fit_with_residuals,
    plot_bootstrap_cusum_envelope,
    plot_alarm_thresholds,
    plot_anomaly_summary,
    plot_pixelwise_cusum_trajectories,
    plot_spatial_first_alarms,
)


@hydra.main(version_base=None, config_path="config", config_name="config")
def example_cusum(cfg: DictConfig) -> None:

    darney = "sentinel1_lat_48.008_lon_6.0816_size_3000"
    path_to_data = Path(cfg.path_to_data) / darney

    sentinel1_data = load_cached_sentinel1_as_numpy_tensor(path_to_data)

    baseline_start_date = "2014-10-20"  # First available s1 data
    baseline_end_date = "2017-10-20"    # 3 years baseline period (scolyte infestation started in 2018)

    # Forward marching
    observation_start_date = baseline_end_date
    observation_end_date = "2026-12-31"

    nb_harmonics = 2
    cusum_slack = 1.0  # Higher slack aloows the path to reset to zero during healthy periods
    cusum_percentile = 99
    ewma_span = 30

    # Standard sample pixels for localized inspections
    pixel_coordinates = [(44, 67), (47, 9), (64, 83), (67, 21)]

    # Compute the radar index and build the raw data tensor
    radar_index_tensor = compute_sentinel1_radar_index(sentinel1_data["bands"], vv_index=0, vh_index=1)
    data_tensor_raw = np.concatenate([sentinel1_data["bands"], radar_index_tensor], axis=1)

    # Fit harmonic baseline on original irregular dates (capturing baseline_data)
    _, baseline_coefficients, baseline_dates, _ = compute_harmonic_baselines(
        data_tensor=data_tensor_raw,
        dates=sentinel1_data["dates"],
        start_date=baseline_start_date,
        end_date=baseline_end_date,
        nb_harmonics=nb_harmonics
    )

    # Compute expected baseline and residuals (for the baseline period)
    baseline_expected, baseline_residuals, baseline_obs_data, baseline_obs_dates = compute_expected_baseline_and_residuals(
        data_tensor=data_tensor_raw,
        dates=sentinel1_data["dates"],
        coefficients=baseline_coefficients,
        start_date=baseline_start_date,
        end_date=baseline_end_date,
        nb_harmonics=nb_harmonics
    )

    # Plot baseline fit and baseline residuals side-by-side
    plot_pixelwise_harmonic_fit_with_residuals(
        data=baseline_obs_data,
        expected=baseline_expected,
        residuals=baseline_residuals,
        dates=baseline_obs_dates,
        pixel_coordinates=pixel_coordinates,
        band_idx=2,
        band_name="VH - VV Radar Index",
        period_type="baseline",
        ewma_span=ewma_span,        
    )

    # Generate bootstrap CUSUM paths for threshold calibration
    bootstrap_cusums = bootstrap_cusum_paths(
        data_tensor=data_tensor_raw,
        dates=sentinel1_data["dates"],
        coefficients=baseline_coefficients,
        start_date=baseline_start_date,
        end_date=baseline_end_date,
        nb_harmonics=nb_harmonics,
        nb_bootstraps=1000,
        block_size=6,
        slack=cusum_slack,
    )

    plot_bootstrap_cusum_envelope(
        bootstrap_cusums=bootstrap_cusums,
        dates=baseline_dates,
        pixel_coordinates=pixel_coordinates,
        band_idx=2,
        band_name="VH - VV Radar Index"
    )

    max_bootstrap_peaks = np.max(bootstrap_cusums, axis=1)
    alarm_threshold = np.percentile(max_bootstrap_peaks, cusum_percentile, axis=0)

    plot_alarm_thresholds(
        alarm_threshold=alarm_threshold,
        quantile=cusum_percentile,
        band_titles=["VV Band", "VH Band", "VH - VV Radar Index"]
    )

    # Compute monitoring residuals on original dates
    raw_baseline_expected, raw_monitoring_residuals, raw_monitoring_data, raw_monitoring_dates = compute_expected_baseline_and_residuals(
        data_tensor=data_tensor_raw,
        dates=sentinel1_data["dates"],
        coefficients=baseline_coefficients,
        start_date=observation_start_date,
        end_date=observation_end_date,
        nb_harmonics=nb_harmonics
    )

    plot_pixelwise_harmonic_fit_with_residuals(
        data=raw_monitoring_data,
        expected=raw_baseline_expected,
        residuals=raw_monitoring_residuals,
        dates=raw_monitoring_dates,
        pixel_coordinates=pixel_coordinates,
        band_idx=2,
        band_name="VH - VV Radar Index",
        period_type="monitoring",
        ewma_span=ewma_span,
    )

    # Regularly interpolate the monitoring residuals to a uniform 6-day grid
    monitoring_residuals_interpolated, monitoring_dates_interpolated = interpolate_tensor_to_regular_grid(
        data_tensor=raw_monitoring_residuals,
        dates=raw_monitoring_dates,
        freq="6D"
    )

    # Run continuous one-sided CUSUM accumulation on the regularized residuals
    monitoring_cusum = compute_one_sided_cusum(monitoring_residuals_interpolated, slack=cusum_slack)

    # Anomaly detection & tracking
    infestation_alerts = monitoring_cusum > alarm_threshold

    plot_anomaly_summary(
        alerts=infestation_alerts,
        dates=monitoring_dates_interpolated,
        band_titles=["VV Band", "VH Band", "VH - VV Radar Index"]
    )

    plot_pixelwise_cusum_trajectories(
        monitoring_cusum=monitoring_cusum,
        alarm_threshold=alarm_threshold,
        dates=monitoring_dates_interpolated,
        pixel_coordinates=pixel_coordinates,
        band_idx=2,
        band_name="VH - VV Radar Index"
    )

    n_bands, height, width = alarm_threshold.shape
    first_alarm_indices = np.full((n_bands, height, width), -1, dtype=np.int32)

    for t_idx in range(len(monitoring_dates_interpolated)):
        is_alarm = infestation_alerts[t_idx]
        new_alarms = is_alarm & (first_alarm_indices == -1)
        first_alarm_indices[new_alarms] = t_idx

    first_alarm_dates = np.array([
        monitoring_dates_interpolated[idx] if idx != -1 else pd.NaT
        for idx in first_alarm_indices.reshape(-1)
    ]).reshape(n_bands, height, width)

    plot_spatial_first_alarms(
        first_alarm_dates=first_alarm_dates,
        monitoring_dates=monitoring_dates_interpolated,
        band_titles=["VV Band", "VH Band", "VH - VV Radar Index"]
    )

    print("Pipeline executed successfully. All figures saved in 'images/'.")


if __name__ == "__main__":
    example_cusum()
