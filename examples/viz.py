import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

# Ensure the output directory exists
Path("images").mkdir(parents=True, exist_ok=True)


def plot_random_pixel_baselines(
    data_tensor: NDArray[np.float32],
    coefficients: NDArray[np.float64],
    t_baseline: NDArray[np.float64],
    dates: NDArray[np.datetime64],
    pixel_coordinates: list[tuple[int, int]],
    band_idx: int = 0  # [vv, vh, ratio]
) -> None:
    """Plots a 2x2 grid of 4 specified pixel coordinates comparing raw baseline time series
    against the fitted harmonic model.
    """
    n_coeffs = coefficients.shape[-1]
    num_harmonics = (n_coeffs - 1) // 2

    X_cols = [np.ones_like(t_baseline)]
    for k in range(1, num_harmonics + 1):
        X_cols.append(np.cos(2.0 * np.pi * k * t_baseline))
        X_cols.append(np.sin(2.0 * np.pi * k * t_baseline))
    X = np.column_stack(X_cols)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    axes = axes.flatten()

    for idx, (h, w) in enumerate(pixel_coordinates):
        ax = axes[idx]

        y_raw = data_tensor[:, band_idx, h, w]
        beta = coefficients[band_idx, h, w, :]
        y_fit = X @ beta

        ax.plot(dates, y_raw, 'ko', markersize=3, alpha=0.5, label='Raw Data')
        ax.plot(dates, y_fit, 'r-', linewidth=2, label='Harmonic Fit')

        ax.set_title(f"Pixel [{h}, {w}]", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f"Baseline Harmonic Fits (Band {band_idx}) - 4-Pixel Sample", fontsize=14)
    plt.tight_layout()
    
    output_path = Path("images") / f"{plot_random_pixel_baselines.__name__}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')



def plot_monitoring_and_residuals(
    monitoring_data: NDArray[np.float32],
    baseline_expected: NDArray[np.float32],
    residuals: NDArray[np.float32],
    dates: NDArray[np.datetime64],
    pixel_coordinates: list[tuple[int, int]],
    band_idx: int = 2,
    band_name: str = "VH - VV Radar Index"
) -> None:
    """Plots a grid of specified pixel coordinates comparing the monitoring observations 
    against the expected baseline, alongside their resulting residuals.
    """
    n_samples = len(pixel_coordinates)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3 * n_samples), sharex=True)
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for idx, (h, w) in enumerate(pixel_coordinates):
        obs = monitoring_data[:, band_idx, h, w]
        pred = baseline_expected[:, band_idx, h, w]
        res = residuals[:, band_idx, h, w]
        
        ax_ts = axes[idx, 0]
        ax_ts.plot(dates, obs, 'ko-', markersize=3, alpha=0.5, label='Observed Data')
        ax_ts.plot(dates, pred, 'r-', linewidth=2, label='Expected Baseline')
        ax_ts.set_ylabel("Backscatter / Index", fontsize=9)
        ax_ts.set_title(f"Pixel [{h}, {w}] - {band_name}", fontsize=10)
        ax_ts.grid(True, linestyle='--', alpha=0.5)
        if idx == 0:
            ax_ts.legend(loc='upper right', fontsize=8)
            
        ax_res = axes[idx, 1]
        ax_res.plot(dates, res, 'b-', linewidth=1.5, label='Residual (Base - Obs)')
        ax_res.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax_res.fill_between(dates, res, 0, where=(res > 0), color='salmon', alpha=0.3, label='Positive Deviation (Stress)')
        ax_res.set_ylabel("Residual Value", fontsize=9)
        ax_res.set_title(f"Pixel [{h}, {w}] - Residual Deviation", fontsize=10)
        ax_res.grid(True, linestyle='--', alpha=0.5)
        if idx == 0:
            ax_res.legend(loc='upper right', fontsize=8)

    plt.xlabel("Monitoring Date", fontsize=10)
    plt.tight_layout()
    
    output_path = Path("images") / f"{plot_monitoring_and_residuals.__name__}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')



def plot_bootstrap_cusum_envelope(
    bootstrap_cusums: NDArray[np.float32],
    dates: NDArray[np.datetime64],
    pixel_coordinates: list[tuple[int, int]],
    band_idx: int = 2,
    band_name: str = "VH - VV Radar Index"
) -> None:
    """Plots the median and confidence envelopes (percentile bands) of the 
    bootstrap CUSUM paths for the specified pixels.
    """
    n_bootstraps, n_time, n_bands, height, width = bootstrap_cusums.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    axes = axes.flatten()
    
    for idx, (h, w) in enumerate(pixel_coordinates):
        ax = axes[idx]
        
        pixel_paths = bootstrap_cusums[:, :, band_idx, h, w]
        
        p50 = np.percentile(pixel_paths, 50, axis=0)
        p95 = np.percentile(pixel_paths, 95, axis=0)
        p99 = np.percentile(pixel_paths, 99, axis=0)
        
        ax.fill_between(dates, p50, p99, color='skyblue', alpha=0.2, label='50% - 99% Noise Envelope')
        ax.fill_between(dates, p50, p95, color='dodgerblue', alpha=0.3, label='50% - 95% Noise Envelope')
        ax.plot(dates, p50, 'b-', linewidth=2, label='Median Bootstrap CUSUM')
        
        ax.set_title(f"Pixel [{h}, {w}] - {band_name}", fontsize=10)
        ax.set_ylabel("CUSUM Accumulation", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)

    plt.xlabel("Baseline Time Axis", fontsize=10)
    plt.suptitle("Bootstrap CUSUM Noise Envelopes (Calibration Check)", fontsize=14)
    plt.tight_layout()
    
    output_path = Path("images") / f"{plot_bootstrap_cusum_envelope.__name__}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')



def plot_alarm_thresholds(
    alarm_threshold: NDArray[np.float32],
    quantile: int = 99,
    band_titles: list[str] = ["VV Band", "VH Band", "VH - VV Radar Index"]
) -> None:
    """Plots a 1x3 grid displaying the spatial CUSUM alarm threshold maps for each band.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for b_idx, ax in enumerate(axes):    
        threshold_map = alarm_threshold[b_idx, :, :]    
        im = ax.imshow(threshold_map, cmap='viridis', origin='upper')
        ax.set_title(f"Alarm Threshold ({quantile}th percentile) - {band_titles[b_idx]}", fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.suptitle("CUSUM alarm thresholds", fontsize=14)
    plt.tight_layout()
    
    output_path = Path("images") / f"{plot_alarm_thresholds.__name__}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')



def plot_anomaly_summary(
    alerts: NDArray[np.bool_],
    dates: NDArray[np.datetime64],
    band_titles: list[str] = ["VV Band", "VH Band", "VH - VV Radar Index"]
) -> None:
    """Plots a 2x3 summary grid of CUSUM anomaly detection results.
    """
    n_time, n_bands, height, width = alerts.shape

    spatial_counts = np.sum(alerts, axis=0)
    temporal_counts = np.sum(alerts, axis=(2, 3))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for b_idx in range(n_bands):
        ax_spatial = axes[0, b_idx]
        im = ax_spatial.imshow(spatial_counts[b_idx], cmap='YlOrRd', origin='upper')
        ax_spatial.set_title(f"Spatial Persistence - {band_titles[b_idx]}", fontsize=10)
        ax_spatial.axis('off')
        fig.colorbar(im, ax=ax_spatial, fraction=0.046, pad=0.04)

        ax_temporal = axes[1, b_idx]
        ax_temporal.plot(dates, temporal_counts[:, b_idx], color='crimson', linewidth=1.5, label='Active Anomalies')
        ax_temporal.set_title(f"Global Timeline - {band_titles[b_idx]}", fontsize=10)
        ax_temporal.set_ylabel("Global Pixel Count", fontsize=9)
        ax_temporal.set_xlabel("Monitoring Date", fontsize=9)
        ax_temporal.grid(True, linestyle='--', alpha=0.5)
        ax_temporal.tick_params(axis='x', rotation=15)

    plt.suptitle("CUSUM Forest Health Monitoring - Rolling Window Anomaly Summary", fontsize=14)
    plt.tight_layout()
    
    output_path = Path("images") / f"{plot_anomaly_summary.__name__}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')



def plot_pixel_cusum_trajectories(
    monitoring_cusum: NDArray[np.float32],
    alarm_threshold: NDArray[np.float32],
    dates: NDArray[np.datetime64],
    pixel_coordinates: list[tuple[int, int]],
    band_idx: int = 2,
    band_name: str = "VH - VV Radar Index"
) -> None:
    """Plots the continuous forward-marching CUSUM accumulation path against the pixel-specific 
    alarm threshold for selected pixel coordinates.
    """
    n_pixels = len(pixel_coordinates)
    n_cols = 2 if n_pixels > 1 else 1
    n_rows = int(np.ceil(n_pixels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    
    if n_pixels == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for idx, (h, w) in enumerate(pixel_coordinates):
        if idx >= len(axes):
            break
        ax = axes[idx]

        cusum_path = monitoring_cusum[:, band_idx, h, w]
        threshold_val = alarm_threshold[band_idx, h, w]
        pixel_alerts = cusum_path > threshold_val

        ax.plot(dates, cusum_path, color='dodgerblue', linewidth=1.5, label='CUSUM Path')
        ax.axhline(threshold_val, color='crimson', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold_val:.2f})')

        if np.any(pixel_alerts):
            ax.plot(dates[pixel_alerts], cusum_path[pixel_alerts], 'ro', markersize=5, label='Alarm Triggered')

        ax.set_title(f"Pixel [{h}, {w}] - {band_name}", fontsize=10)
        ax.set_ylabel("CUSUM Accumulation", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)

    for idx in range(n_pixels, len(axes)):
        axes[idx].axis('off')

    plt.xlabel("Monitoring Date", fontsize=10)
    plt.suptitle(f"Pixel-Level Continuous CUSUM Trajectories & Alarm Triggers ({band_name})", fontsize=14)
    plt.tight_layout()
    
    output_path = Path("images") / f"{plot_pixel_cusum_trajectories.__name__}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

