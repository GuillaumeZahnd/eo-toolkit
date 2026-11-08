from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np
import numpy.typing as npt
import pandas as pd
from pystac import Item
import logging

from .stac import (
    EARTH_SEARCH_URL,
    utm_epsg,
    item_time,
    item_date,
    search_stac,
    load_stac_raster,
)

# Sentinel-2 Scene Classification Layer (SCL) invalid/cloud mask values:
# 0: No Data, 1: Saturated/Defective, 3: Cloud Shadows, 8: Cloud Medium Probability, 9: Cloud High Probability, 10: Thin Cirrus
BAD_SCL = [0, 1, 3, 8, 9, 10]
CELL_S2 = 30  # Default working resolution in meters

# Suppress informational messages from rasterio (like the boto3 fallback notice)
logging.getLogger("rasterio").setLevel(logging.WARNING)
logging.getLogger("rasterio.session").setLevel(logging.WARNING)


def download_sentinel2_patches(
    unique_id: str,
    bbox: list[float],
    manifest: List[Dict[str, Any]],
    path_to_data: Union[str, Path],
    required_bands: tuple[str, ...],
    max_cloud_cover: float,
) -> None:
    """
    Downloads Sentinel-2 rectangular patches for a given bounding box [xmin, ymin, xmax, ymax]
    filtered by a maximum cloud cover threshold.
    """

    data_collection_name = f"sentinel2_patches_{unique_id}"
    patch_dir = Path(path_to_data) / data_collection_name
    patch_dir.mkdir(parents=True, exist_ok=True)

    if not manifest:
        print("Provided manifest is empty. No Sentinel-2 scenes to download.")
        return

    df_manifest = pd.DataFrame(manifest)
    if df_manifest.empty:
        print("Manifest DataFrame is empty.")
        return

    # Filter manifest based on max_cloud_cover threshold
    df_manifest = df_manifest[df_manifest["cloud_cover"] <= max_cloud_cover].copy()
    if df_manifest.empty:
        print(f"No Sentinel-2 scenes found meeting the cloud cover threshold of <= {max_cloud_cover}%.")
        return

    df_manifest["datetime"] = pd.to_datetime(df_manifest["datetime"]).dt.tz_localize(None)

    # Create a 15-day period grouping column
    df_manifest["fortnight"] = df_manifest["datetime"].apply(
        lambda dt: f"{dt.year}-{dt.month:02d}-H1" if dt.day <= 15 else f"{dt.year}-{dt.month:02d}-H2"
    )

    saved_count = 0

    for _, group in tqdm(df_manifest.groupby("fortnight"), desc="Downloading S2 patches"):
        # Select the single least cloudy image for this 15-day window
        best_scene = group.loc[group["cloud_cover"].idxmin()]

        exact_date = best_scene["date"]  # "YYYY-MM-DD"
        file_path = patch_dir / f"{exact_date}.npz"

        # Caching check: skip if already downloaded locally
        if file_path.exists():
            continue

        try:
            arr, exact_date = fetch_s2_image(
                item=best_scene["_item_obj"],
                bbox=bbox,
                required_bands=required_bands,
            )

            np.savez_compressed(
                file_path,
                data=arr,  # Shape (C, H, W)
                band_names=np.array(required_bands, dtype=str),
                cloud_cover=np.array([best_scene["cloud_cover"]]),
                date=np.array([exact_date], dtype=str)
            )
            saved_count += 1

        except Exception as e:
            print(f"Failed to download scene for {exact_date}: {e}")

    print(f"Sentinel-2 download complete. {saved_count} new patches cached to {patch_dir}.")


def get_s2_manifest(
    bbox: list[float],
    start: str,
    end: str,
    required_bands: tuple[str, ...],
    path_to_logs: Union[str, Path],
    unique_id: str,
    stac_url: str = EARTH_SEARCH_URL,
) -> List[Dict[str, Any]]:
    """Queries the STAC catalog once and returns a clean manifest of all available
    Sentinel-2 acquisitions within the given spatial bounding box and date range that
    contain all required bands, and writes a manifest audit log to a text file.
    """
    try:
        items = search_stac(stac_url, "sentinel-2-l2a", bbox, start, end, {})
    except Exception:
        return []

    # Sort items chronologically
    items = sorted(items, key=item_time)

    # Define the full list of bands that must be present in item.assets
    target_bands = set(list(required_bands) + ["scl"])

    # Filter items during manifest creation
    manifest = [
        {
            "date": item_date(item),
            "datetime": item.datetime,
            "cloud_cover": item.properties.get("eo:cloud_cover", 100.0),
            "item_id": item.id,
            "_item_obj": item,
        }
        for item in items
        if target_bands.issubset(item.assets.keys())
    ]

    # Write log file
    path_to_log_file = Path(path_to_logs) / f"sentinel2_cloud_cover_{unique_id}.txt"
    path_to_log_file.parent.mkdir(parents=True, exist_ok=True)

    log_lines = [
        f"Sentinel-2 Manifest Audit Report (Total valid passes: {len(manifest)})\n",
        "-" * 65 + "\n"
    ]
    for entry in manifest:
        log_lines.append(
            f"Date: {entry['date']} | Cloud Cover: {entry['cloud_cover']:5.2f}% | ID: {entry['item_id']}\n"
        )

    path_to_log_file.write_text("".join(log_lines), encoding="utf-8")
    print(f"Sentinel-2 manifest successfully logged to {path_to_log_file} ({len(manifest)} items).")

    return manifest


def fetch_s2_image(
    item: Item,
    bbox: list[float],
    required_bands: tuple[str, ...],
    resolution: float = CELL_S2,
) -> tuple[npt.NDArray[np.float32], str]:
    """Loads a single pre-selected Sentinel-2 STAC item into a raster array,
    applying SCL cloud/shadow masking, and returns the cleaned bands array and acquisition date.

    Parameters
    ----------
    item : pystac.Item
        The pre-selected Sentinel-2 STAC item to load.
    bbox : list[float]
        Bounding box of the area in format [xmin, ymin, xmax, ymax] (WGS84).
    required_bands : tuple[str, ...]
        Any arbitrary tuple of Sentinel-2 asset names to load.
    resolution : float
        Output spatial resolution in meters (default is 30.0).

    Returns
    -------
    tuple[np.ndarray, str]
        - Float32 array of shape ``(C, H, W)`` with invalid/cloudy pixels set as NaN.
        - The exact acquisition date string ``'YYYY-MM-DD'``.
    """
    xmin, ymin, xmax, ymax = bbox
    # Center point calculation for UTM zone determination
    lon = (xmin + xmax) / 2.0
    lat = (ymin + ymax) / 2.0

    observation_date = item_date(item)

    # Always request the Scene Classification Layer (scl) alongside bands for masking
    raster_bands = list(required_bands) + ["scl"]

    data, _ = load_stac_raster(
        [item], raster_bands, bbox, utm_epsg(lon, lat), resolution,
        {"*": "average", "scl": "nearest"}
    )

    *band_arrays, scl = data[0]

    # Build strict validity mask (removes clouds, cloud shadows, nodata, and invalid values)
    ok = ~np.isin(scl, BAD_SCL)
    for arr in band_arrays:
        ok &= np.isfinite(arr) & (arr > 0)

    raw_bands = np.array([
        np.where(ok, arr, np.float32(np.nan)) for arr in band_arrays
    ], dtype=np.float32)

    return raw_bands, observation_date


def select_least_cloudy_manifest_item(manifest: list[dict], target_date: str, window_days: int = 7) -> dict | None:
    """Selects the least cloudy manifest entry within a temporal search window around a target date."""
    if not manifest:
        return None

    t_target = pd.Timestamp(target_date)
    candidates = [
        entry for entry in manifest
        if abs((pd.Timestamp(entry["date"]) - t_target).days) <= window_days
    ]

    if not candidates:
        return None

    best_entry = min(candidates, key=lambda entry: entry["cloud_cover"])
    return best_entry


def compute_ndre_from_patch(
    data: np.ndarray,
    required_bands: list[str] | np.ndarray,
    nir_band: str = "nir08",
    re_band: str = "rededge1"
) -> np.ndarray:
    """
    Computes NDRE dynamically from a patch tensor using its band names.

    Parameters:
    -----------
    data : np.ndarray
        Raster array of shape (C, H, W).
    required_bands : list[str] or np.ndarray
        List or array of corresponding band names matching the channels.
    nir_band : str, optional
        The name of the Near-Infrared band (default "nir08").
    re_band : str, optional
        The name of the Red Edge band (default "rededge1").

    Returns:
    --------
    np.ndarray
        Float32 NDRE array of shape (H, W) with invalid pixels set to NaN.
    """
    band_list = list(required_bands)

    if nir_band not in band_list or re_band not in band_list:
        raise ValueError(
            f"Required bands ('{nir_band}', '{re_band}') not found in available bands: {band_list}"
        )

    nir_idx = band_list.index(nir_band)
    re_idx = band_list.index(re_band)

    nir = data[nir_idx]
    re = data[re_idx]

    with np.errstate(invalid="ignore", divide="ignore"):
        ndre = np.where(
            np.isfinite(nir) & np.isfinite(re) & ((nir + re) != 0),
            (nir - re) / (nir + re),
            np.nan
        ).astype(np.float32)

    return ndre
