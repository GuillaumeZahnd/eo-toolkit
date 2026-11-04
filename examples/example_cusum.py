import hydra
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from eo_toolkit import (
    load_cached_sentinel1_as_numpy_tensor,
    compute_harmonic_baselines,
    compute_sentinel1_radar_index,
    compute_expected_baseline_and_residuals,
    compute_one_sided_cusum,
    bootstrap_cusum_paths,
)


@hydra.main(version_base=None, config_path="config", config_name="config")
def example_cusum(cfg: DictConfig) -> None:
    darney = "sentinel1_lat_48.008_lon_6.0816_size_3000"
    path_to_data = Path(cfg.path_to_data) / darney

    sentinel1_data = load_cached_sentinel1_as_numpy_tensor(path_to_data)

    baseline_start_date = "2016-01-01"
    baseline_end_date = "2017-12-31"  # Scolyte infestation started in 2018

    observation_start_date = "2018-01-01"
    observation_end_date = "2021-12-31"

    nb_harmonics = 2
    cusum_slack = 0.2
    cusum_percentile = 99

    # Compute the radar index, and concatenate to the data tensor as an additional band
    radar_index_tensor = compute_sentinel1_radar_index(sentinel1_data["bands"], vv_index=0, vh_index=1)
    data_tensor = np.concatenate([sentinel1_data["bands"], radar_index_tensor], axis=1)

    baseline_data_tensor, baseline_coefficients, _, _ = compute_harmonic_baselines(
        data_tensor=data_tensor,
        dates=sentinel1_data["dates"],
        start_date=baseline_start_date,
        end_date=baseline_end_date,
        nb_harmonics=nb_harmonics
    )

    # Generate bootstrap cumsum, for calibration
    bootstrap_cusums = bootstrap_cusum_paths(
        data_tensor=data_tensor,
        dates=sentinel1_data["dates"],
        coefficients=baseline_coefficients,
        start_date=baseline_start_date,
        end_date=baseline_end_date,
        nb_harmonics=nb_harmonics,
        nb_bootstraps=1000,
        block_size=12,
        slack=cusum_slack,
    )

    # Alarm threshold, per band (quantile-based)
    max_bootstrap_peaks = np.max(bootstrap_cusums, axis=1)  # Shape: (nb_bootstraps, bands, height, width)
    alarm_threshold = np.percentile(max_bootstrap_peaks, cusum_percentile, axis=0)  # Shape: (bands, height, width)


    # Evaluate on the monitoring period
    _, monitoring_residuals, monitoring_tensor, monitoring_dates = compute_expected_baseline_and_residuals(
        data_tensor=data_tensor,
        dates=sentinel1_data["dates"],
        coefficients=baseline_coefficients,
        start_date=observation_start_date,
        end_date=observation_end_date,
        nb_harmonics=nb_harmonics
    )

    # Run CUSUM on the monitoring observations
    monitoring_cusum = compute_one_sided_cusum(monitoring_residuals, slack=cusum_slack)

    # Anomaly detection: If the CUSUM surpasses the bootstrap-derived threshold
    infestation_alerts = monitoring_cusum > alarm_threshold

if __name__ == "__main__":
    example_cusum()
