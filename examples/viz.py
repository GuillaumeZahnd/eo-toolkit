import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Ensure the output directory exists
Path("images").mkdir(parents=True, exist_ok=True)


def plot_pixelwise_harmonic_fit_with_residuals(
    data: NDArray[np.float32],
    expected: NDArray[np.float32],
    residuals: NDArray[np.float32],
    dates: NDArray[np.datetime64],
    pixel_coordinates: list[tuple[int, int]],
    band_idx: int = 2,
    band_name: str = "VH - VV Radar Index",
    period_type: str = "Baseline"  # Baseline vs Monitoring
) -> None:
    """Plots observations with harmonic fit alongside residuals to verify noise characteristics."""
    n_samples = len(pixel_coordinates)
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3 * n_samples), sharex=True)
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for idx, (h, w) in enumerate(pixel_coordinates):
        obs = data[:, band_idx, h, w]
        pred = expected[:, band_idx, h, w]
        res = residuals[:, band_idx, h, w]
        
        # --- Left: Raw Data vs Harmonic Fit ---
        ax_fit = axes[idx, 0]
        ax_fit.plot(dates, obs, 'ko', markersize=3, alpha=0.5, label=f'{period_type} data')
        ax_fit.plot(dates, pred, 'r-', linewidth=2, label='Harmonic fit')
        ax_fit.set_ylabel("Index [a.u.]", fontsize=9)
        ax_fit.set_xlabel("Time", fontsize=9)        
        ax_fit.set_title(f"Pixel [{h}, {w}] - {period_type} and harmonic fit", fontsize=10)
        ax_fit.grid(True, linestyle='--', alpha=0.5)
        ax_fit.tick_params(axis='x', rotation=90)
        if idx == 0:
            ax_fit.legend(loc='upper right', fontsize=8)
            
        # --- Right: Residuals with Percentile Lines ---
        ax_res = axes[idx, 1]
        ax_res.plot(dates, res, 'b-', linewidth=1.5, label='Residual')
        ax_res.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # Compute residual percentiles for this specific pixel
        p50 = np.percentile(res, 50)
        p95 = np.percentile(res, 95)
        p99 = np.percentile(res, 99)
        
        # Plot percentile reference lines
        ax_res.axhline(p50, color='forestgreen', linestyle='--', linewidth=1, label='median' if idx == 0 else "")
        ax_res.axhline(p95, color='darkorange', linestyle='--', linewidth=1, label='95th percentile' if idx == 0 else "")
        ax_res.axhline(p99, color='crimson', linestyle='--', linewidth=1, label='99th percentile' if idx == 0 else "")
        
        ax_res.set_ylabel("Residual", fontsize=9)
        ax_res.set_xlabel("Time", fontsize=9)
        ax_res.set_title(f"Pixel [{h}, {w}] - Residuals", fontsize=10)
        ax_res.grid(True, linestyle='--', alpha=0.5)
        ax_res.tick_params(axis='x', rotation=90)
        if idx == 0:
            ax_res.legend(loc='upper right', fontsize=8)

    plt.suptitle(f"{period_type} data, Harmonic fit, and Residuals (Band {band_idx})", fontsize=14)
    plt.tight_layout()
    
    output_path = Path("images") / f"plot_pixelwise_harmonic_fit_with_residuals_{period_type.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


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
        ax.set_xlabel("Time", fontsize=9)        
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
            
        ax.tick_params(axis='x', rotation=90)            


    plt.suptitle("Bootstrap CUSUM Noise Envelopes (Calibration Check)", fontsize=14)
    plt.tight_layout()
    
    output_path = Path("images") / "plot_bootstrap_cusum_envelope.png"
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
    
    output_path = Path("images") / f"plot_alarm_thresholds.png"
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
        ax_temporal.set_xlabel("Time", fontsize=9)
        ax_temporal.grid(True, linestyle='--', alpha=0.5)
        ax_temporal.tick_params(axis='x', rotation=15)

    plt.suptitle("CUSUM anomaly summary", fontsize=14)
    plt.tight_layout()
    
    output_path = Path("images") / f"{plot_anomaly_summary.__name__}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')



def plot_pixelwise_cusum_trajectories(
    monitoring_cusum: NDArray[np.float32],
    alarm_threshold: NDArray[np.float32],
    dates: NDArray[np.datetime64],
    pixel_coordinates: list[tuple[int, int]],
    band_idx: int = 2,
    band_name: str = "VH - VV Radar Index"
) -> None:
    """Plots the continuous forward-marching CUSUM accumulation path against the pixel-specific 
    alarm threshold, highlighting active alarm windows and coloring points above the threshold distinctly.
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

        # Plot base path line
        ax.plot(dates, cusum_path, color='dodgerblue', linewidth=1.5, label='CUSUM Path')
        
        # Split markers into normal (below threshold) and alert (above threshold)
        normal_mask = ~pixel_alerts
        if np.any(normal_mask):
            ax.plot(dates[normal_mask], cusum_path[normal_mask], 'o', color='dodgerblue', markersize=3, alpha=0.6, label='Normal' if idx == 0 else "")
        if np.any(pixel_alerts):
            ax.plot(dates[pixel_alerts], cusum_path[pixel_alerts], 'o', color='crimson', markersize=5, label='Above Threshold' if idx == 0 else "")

        ax.axhline(threshold_val, color='crimson', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold_val:.2f})' if idx == 0 else "")

        # Highlight active alarm periods with a soft span and first-trigger annotation
        if np.any(pixel_alerts):
            alert_indices = np.where(pixel_alerts)[0]
            first_idx = alert_indices[0]
            
            ax.axvspan(dates[alert_indices[0]], dates[alert_indices[-1]], color='salmon', alpha=0.25, label='Active Alarm Window' if idx == 0 else "")
            
            ax.annotate(
                f"{pd.to_datetime(dates[first_idx]).strftime('%Y-%m-%d')}",
                xy=(dates[first_idx], cusum_path[first_idx]),
                xytext=(0, 15), textcoords='offset points',
                ha='center', fontsize=8, fontweight='bold', color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=1)
            )

        ax.set_title(f"Pixel [{h}, {w}] - {band_name}", fontsize=10)
        ax.set_ylabel("CUSUM Accumulation", fontsize=9)
        ax.set_xlabel("Time", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', rotation=90)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)

    for idx in range(n_pixels, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"Pixel-Level Continuous CUSUM Trajectories & Alarm Triggers ({band_name})", fontsize=14)
    plt.tight_layout()
    
    output_path = Path("images") / "plot_pixel_cusum_trajectories.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
