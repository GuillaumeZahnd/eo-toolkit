import hydra
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from eo_toolkit import (
    load_cached_sentinel1_as_numpy_tensor,
    compute_harmonic_baselines,
    compute_sentinel1_radar_index,
    bootstrap_baseline_residuals,
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

    # Generate bootstrap residual replicates for threshold calibration
    bootstrapped_residuals = bootstrap_baseline_residuals(
        data_tensor=data_tensor,
        dates=sentinel1_data["dates"],
        coefficients=baseline_coefficients,
        start_date=baseline_start_date,
        end_date=baseline_end_date,
        nb_harmonics=nb_harmonics,
        nb_bootstraps=500,
        block_size=12
    )

    print(bootstrapped_residuals.shape)

if __name__ == "__main__":
    example_cusum()
