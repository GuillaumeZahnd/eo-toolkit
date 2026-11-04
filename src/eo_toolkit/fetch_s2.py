import numpy as np
import numpy.typing as npt
import pandas as pd
from pystac import Item

from .stac import (
    EARTH_SEARCH_URL,
    bbox_around,
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


def get_s2_manifest(
    lon: float,
    lat: float,
    size_m: float,
    start: str,
    end: str,
    max_cloud: float = 60.0,
    stac_url: str = EARTH_SEARCH_URL,
) -> list[dict]:
    """Queries the STAC catalog once and returns a clean manifest of available
    Sentinel-2 acquisitions filtered by spatial bounding box and cloud cover.
    """
    bbox = bbox_around(lon, lat, size_m)
    q = {}
    if max_cloud is not None:
        q["eo:cloud_cover"] = {"lt": max_cloud}

    try:
        items = search_stac(stac_url, "sentinel-2-l2a", bbox, start, end, q)
    except Exception:
        return []

    items = sorted(items, key=item_time)
    manifest = []
    for item in items:
        manifest.append({
            "date": item_date(item),
            "datetime": item.datetime,
            "cloud_cover": item.properties.get("eo:cloud_cover", 100.0),
            "item_id": item.id,
            "_item_obj": item
        })
    return manifest


def fetch_s2_image(
    item: Item,
    lon: float,
    lat: float,
    size_m: float,
    bands: tuple[str, ...],
    resolution: float = CELL_S2,
) -> tuple[npt.NDArray[np.float32], list[str], str]:
    """Loads a single pre-selected Sentinel-2 STAC item into a raster array,
    applying SCL cloud/shadow masking, and returns the raw bands along with their names.

    Parameters
    ----------
    item : pystac.Item
        The pre-selected Sentinel-2 STAC item to load.
    lon, lat, size_m : float
        Spatial definition of the ROI.
    bands : tuple[str, ...]
        Any arbitrary tuple of Sentinel-2 asset names to load.
    resolution : float
        Output spatial resolution in meters (default is 10.0).

    Returns
    -------
    tuple[np.ndarray, list[str], str]
        - Float32 array of shape ``(C, H, W)`` with invalid/cloudy pixels set as NaN.
        - List of corresponding band name strings.
        - The exact acquisition date string ``'YYYY-MM-DD'``.
    """
    bbox = bbox_around(lon, lat, size_m)
    exact_date = item_date(item)

    # Always request the Scene Classification Layer (scl) alongside bands for masking
    raster_bands = list(bands) + ["scl"]

    data, _ = load_stac_raster(
        [item], raster_bands, bbox, utm_epsg(lon, lat), resolution,
        {"*": "average", "scl": "nearest"}
    )

    *band_arrays, scl = data[0]

    # Build strict validity mask (removes clouds, cloud shadows, nodata, and invalid values)
    ok = ~np.isin(scl, BAD_SCL)
    for arr in band_arrays:
        ok &= np.isfinite(arr) & (arr > 0)

    # Store raw bands dynamically without hardcoding positions or indices
    cleaned_bands = np.array([
        np.where(ok, arr, np.float32(np.nan)) for arr in band_arrays
    ], dtype=np.float32)

    return cleaned_bands, list(bands), exact_date


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
    band_names: list[str] | np.ndarray,
    nir_band: str = "nir08",
    re_band: str = "rededge1"
) -> np.ndarray:
    """
    Computes NDRE dynamically from a patch tensor using its band names.

    Parameters:
    -----------
    data : np.ndarray
        Raster array of shape (C, H, W).
    band_names : list[str] or np.ndarray
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
    band_list = list(band_names)

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
