import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from pathlib import Path

from eo_toolkit import (
    get_s2_manifest,
    fetch_s2_image,
)

@hydra.main(version_base=None, config_path="config", config_name="config")
def download_sentinel2_patches(cfg: DictConfig) -> None:

    YEARS = range(2015, 2027)  # Sentinel-2A was launched on June 23, 2015.
    MAX_CLOUD_COVER = 40.0     # Strict threshold to avoid heavy clouds

    DATA_COLLECTION_NAME = f"sentinel2_lat_{cfg.region.latitude}_lon_{cfg.region.longitude}_size_{cfg.region.size_m}"
    patch_dir = Path(cfg.path_to_data) / DATA_COLLECTION_NAME
    patch_dir.mkdir(parents=True, exist_ok=True)

    # Fetch the complete multi-year manifest
    print(f"Querying Sentinel-2 STAC for region ({cfg.region.longitude}, {cfg.region.latitude})...")
    raw_manifest = get_s2_manifest(
        cfg.region.longitude,
        cfg.region.latitude,
        cfg.region.size_m,
        start=f"{YEARS[0]}-01-01",
        end=f"{YEARS[-1]}-12-31",
        max_cloud=MAX_CLOUD_COVER
    )
    print(f"Found {len(raw_manifest)} raw scenes passing initial cloud filter.")

    # Process month by month to pick the absolute cleanest scene per month (or keep all)
    # Strategy: Group by Year-Month and pick the item with minimum cloud cover
    df_manifest = pd.DataFrame(raw_manifest)
    if df_manifest.empty:
        print("No Sentinel-2 scenes found.")
        return

    df_manifest["datetime"] = pd.to_datetime(df_manifest["datetime"]).dt.tz_localize(None)
    df_manifest["year_month"] = df_manifest["datetime"].dt.to_period("M")

    saved_count = 0
    for _, group in df_manifest.groupby("year_month"):
        # Select the single least cloudy image for this specific month
        best_scene = group.loc[group["cloud_cover"].idxmin()]

        exact_date = best_scene["date"]  # "YYYY-MM-DD"
        file_path = patch_dir / f"{exact_date}.npz"

        # Caching check: skip if already downloaded locally
        if file_path.exists():
            continue

        # Fetch raster data & compute NDRE (or requested optical bands)
        try:
            arr, band_names, exact_date = fetch_s2_image(
                best_scene["_item_obj"],
                cfg.region.longitude, cfg.region.latitude, cfg.region.size_m,
                bands=("rededge1", "nir08")
            )

            np.savez_compressed(
                file_path,
                data=arr,  # Shape (C, H, W)
                band_names=np.array(band_names, dtype=str),
                cloud_cover=np.array([best_scene["cloud_cover"]]),
                date=np.array([exact_date], dtype=str)
            )
            saved_count += 1
            print(f"Downloaded & Cached S2 for {exact_date} (Cloud Cover: {best_scene['cloud_cover']:.1f}%)")

        except Exception as e:
            print(f"Failed to download scene for {exact_date}: {e}")

    print(f"Sentinel-2 download complete. {saved_count} new patches cached to {patch_dir}.")


if __name__ == "__main__":
    download_sentinel2_patches()
