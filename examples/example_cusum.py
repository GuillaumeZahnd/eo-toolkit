import hydra
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from eo_toolkit import (
    load_cached_sentinel1_as_numpy_tensor,
    compute_harmonic_baselines,
    slice_tensor_with_time_interval,
    compute_sentinel1_radar_index,
)


def plot_random_pixel_baselines(
    sub_tensor: NDArray[np.float32],
    coefficients: NDArray[np.float64],
    t_baseline: NDArray[np.float64],
    dates: NDArray[np.datetime64],
    band_idx: int = 1,  # e.g., 1 for VH band
    n_samples: int = 9
) -> None:
    """Plots a grid of randomly selected pixels comparing raw baseline time series
    against the fitted harmonic model.
    """
    _, n_bands, height, width = sub_tensor.shape

    # Randomly select 9 distinct pixel coordinates (h, w)
    np.random.seed(42)
    rand_h = np.random.randint(0, height, size=n_samples)
    rand_w = np.random.randint(0, width, size=n_samples)

    # Reconstruct design matrix for the baseline period to compute fitted curves
    n_coeffs = coefficients.shape[-1]
    num_harmonics = (n_coeffs - 1) // 2

    X_cols = [np.ones_like(t_baseline)]
    for k in range(1, num_harmonics + 1):
        X_cols.append(np.cos(2.0 * np.pi * k * t_baseline))
        X_cols.append(np.sin(2.0 * np.pi * k * t_baseline))
    X = np.column_stack(X_cols)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    axes = axes.flatten()

    for idx, (h, w) in enumerate(zip(rand_h, rand_w)):
        ax = axes[idx]

        # Extract raw time series and fitted coefficients for this pixel/band
        y_raw = sub_tensor[:, band_idx, h, w]
        beta = coefficients[band_idx, h, w, :]

        # Compute harmonic fit
        y_fit = X @ beta

        # Plot
        ax.plot(dates, y_raw, 'ko', markersize=3, alpha=0.5, label='Raw Data')
        ax.plot(dates, y_fit, 'r-', linewidth=2, label='Harmonic Fit')

        ax.set_title(f"Pixel [{h}, {w}]", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f"Baseline Harmonic Fits (Band {band_idx}) - Random 3x3 Sample", fontsize=14)
    plt.tight_layout()
    plt.savefig("harmonic.png", dpi=300, bbox_inches='tight')
    plt.close(fig)



@hydra.main(version_base=None, config_path="config", config_name="config")
def example_cusum(cfg: DictConfig) -> None:
    darney = "sentinel1_lat_48.008_lon_6.0816_size_3000"
    path_to_data = Path(cfg.path_to_data) / darney

    sentinel1_data = load_cached_sentinel1_as_numpy_tensor(path_to_data)

    baseline_start_date = "2016-01-01"
    baseline_end_date = "2017-12-31"  # Scolyte infestation started in 2018

    # Compute the radar index, and concatenate to the data tensor as an additional band
    radar_index_tensor = compute_sentinel1_radar_index(sentinel1_data["bands"], vv_index=0, vh_index=1)
    data_tensor = np.concatenate([sentinel1_data["bands"], radar_index_tensor], axis=1)

    # Slice tensor to retain baseline subset and dates (FOR PLOTTING ONLY)
    sub_tensor, baseline_dates, baseline_t = slice_tensor_with_time_interval(
        data_tensor=data_tensor,
        dates=sentinel1_data["dates"],
        start_date=baseline_start_date,
        end_date=baseline_end_date
    )

    baseline_coefficients, _, _ = compute_harmonic_baselines(
        data_tensor=data_tensor,
        dates=sentinel1_data["dates"],
        start_date=baseline_start_date,
        end_date=baseline_end_date,
        num_harmonics=2
    )

    print(f"Baseline coefficients shape: {baseline_coefficients.shape}")

    # Visualize 3x3 random pixel fits
    plot_random_pixel_baselines(
        sub_tensor=sub_tensor,
        coefficients=baseline_coefficients,
        t_baseline=baseline_t,
        dates=baseline_dates,
        band_idx=2,  # VH band
        n_samples=9
    )


    print(f"Baseline coefficients shape: {baseline_coefficients.shape}")
    # Shape: (bands, height, width, 5) for 2 harmonics (intercept + 2*cos + 2*sin)


if __name__ == "__main__":
    example_cusum()
