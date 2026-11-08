from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pystac import Item

from .stac import (
    load_stac_raster,
    item_date,
    utm_epsg,
    CELL,
    search_stac,
    item_time,
    PLANETARY_COMPUTER_URL,
)


def fetch_s1_image(
    item: Item,
    bbox: list[float],
    required_bands: tuple[str, ...],
    resolution: float = CELL,
    to_db: bool = True,
) -> tuple[np.ndarray, str]:
    """Loads a single pre-selected Sentinel-1 STAC item into a (2, H, W) array.

    Parameters
    ----------
    item : pystac.Item
        The pre-selected Sentinel-1 STAC item to load.
    bbox : list[float]
        Bounding box of the area in format [xmin, ymin, xmax, ymax] (WGS84).
    required_bands (tuple[str, ...]): A tuple of string identifiers representing the
        mandatory bands that each item asset must possess.
    to_db : bool, optional
        If True, converts linear power backscatter to decibels (10 * log10(x)).
        Default is True.
    resolution : float, optional
        Output spatial resolution in meters (default is :data:`CELL`).

    Returns
    -------
    tuple[numpy.ndarray, str]
        A tuple containing:
        - Float32 array of shape ``(2, H, W)`` for VV and VH polarizations, with invalid pixels as NaN.
        - The exact acquisition date string ``'YYYY-MM-DD'``.
    """
    xmin, ymin, xmax, ymax = bbox
    # Center point calculation for UTM zone determination
    lon = (xmin + xmax) / 2.0
    lat = (ymin + ymax) / 2.0

    exact_date = item_date(item)

    data, _ = load_stac_raster([item], required_bands, bbox, utm_epsg(lon, lat), resolution, "nearest")
    arr = data[0]

    # Mask invalid/non-finite values
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, np.float32(np.nan))
    if to_db:
        arr = 10.0 * np.log10(arr)

    return arr.astype(np.float32, copy=False), exact_date


def download_sentinel1_patches(
    manifest: List[Dict[str, Any]],
    bbox: List[float],
    unique_id: str,
    required_bands: tuple[str, ...],
    path_to_data: Union[str, Path],
) -> None:
    """Downloads Sentinel-1 rectangular raster patches for a specified bounding box
    using items from a filtered manifest.

    Args:
        manifest (List[Dict[str, Any]]): A list of filtered metadata dictionaries,
            where each entry must contain a `"date"` string, a `"relative_orbit"` value,
            and a `"_item_obj"` reference object.
        bbox (List[float]): A bounding box defined as `[xmin, ymin, xmax, ymax]` in
            the appropriate coordinate reference system.
        unique_id (str): A unique identifier suffix used to name the output directory
            (e.g., `sentinel1_patches_{unique_id}`).
        required_bands (tuple[str, ...]): A tuple of string identifiers representing the
            mandatory bands that each item asset must possess.
        path_to_data (Union[str, Path]): The root directory path where the patch collection
            directory will be created and saved.

    Returns:
        None
    """
    data_collection_name = f"sentinel1_patches_{unique_id}"
    patch_dir = Path(path_to_data) / data_collection_name
    patch_dir.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(manifest, desc="Downloading Sentinel-1 patches"):
        file_path = patch_dir / f"{entry['date']}.npz"

        if file_path.exists():
            continue  # Skip downloading if already cached locally

        item_obj = entry["_item_obj"]
        best_orbit = entry.get("relative_orbit")

        arr, exact_date = fetch_s1_image(item_obj, bbox, required_bands)

        np.savez_compressed(
            file_path,
            data=arr,
            band_names=np.array(required_bands, dtype=str),
            orbit_number=np.array([best_orbit] if best_orbit is not None else [-1]),
            date=np.array([exact_date], dtype=str)
        )

    print(f"Download complete. Raster patches saved to {patch_dir}.")


def get_s1_manifest(
    bbox: list[float],
    start: str,
    end: str,
    orbit_state: str = "descending",
    stac_url: str = PLANETARY_COMPUTER_URL,
) -> list[dict]:
    """Queries the STAC catalog once and returns a clean manifest of all available
    Sentinel-1 acquisitions in the given spatiotemporal range without orbit pre-filtering.

    Parameters
    ----------
    bbox : list[float]
        Bounding box of the area in format [xmin, ymin, xmax, ymax] (WGS84).
    start : str
        Start date in ISO format ("YYYY-MM-DD").
    end : str
        End date in ISO format ("YYYY-MM-DD").
    orbit_state : str, optional
        Orbit direction, either "descending" or "ascending" (default is "descending").
    stac_url : str, optional
        The STAC API endpoint URL.

    Returns
    -------
    list[dict]
        A chronologically sorted list of dictionaries containing metadata and
        the raw STAC item reference (`_item_obj`) for each acquisition.
    """
    q = {}
    if orbit_state:
        q["sat:orbit_state"] = {"eq": orbit_state}

    try:
        items = search_stac(stac_url, "sentinel-1-rtc", bbox, start, end, q)
    except Exception:
        return []

    # Sort chronologically and build a lightweight manifest plan
    items = sorted(items, key=item_time)
    manifest = []
    for item in items:
        manifest.append({
            "date": item_date(item),
            "datetime": item.datetime,
            "relative_orbit": item.properties.get("sat:relative_orbit"),
            "orbit_source": item.properties.get("s1:orbit_source"),
            "item_id": item.id,
            "_item_obj": item  # Hold reference for direct loading
        })
    return manifest


def filter_manifest_by_bands(
    manifest: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    required_bands: tuple[str, ...],
    path_to_logs: Union[str, Path],
    unique_id: str,
) -> Dict[int, Dict[str, Any]]:
    """Filters manifest items by year and required bands, counts valid orbits per year,
    logs the text audit breakdown, and returns the audited data structure.

    Args:
        manifest (List[Dict[str, Any]]): A list of dictionaries containing metadata entries,
            where each entry may include date information, asset objects, and orbit numbers.
        start_date (str): The start date string (compatible with `pd.Timestamp`) defining
            the beginning of the filtering range (inclusive).
        end_date (str): The end date string (compatible with `pd.Timestamp`) defining
            the end of the filtering range (inclusive).
        required_bands (tuple[str, ...]): A tuple of string identifiers representing the
            mandatory bands that each item asset must possess.
        path_to_logs (Union[str, Path]): Directory path where the textual orbit frequency
            audit report will be saved.
        unique_id (str): A unique identifier used to construct the specific log file name
            (`sentinel1_orbitals_{unique_id}.txt`).

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary mapping each year (as an integer) to its
            corresponding audit dictionary, which contains:
            - `"manifest_with_required_bands"` (List[Dict[str, Any]]): Filtered manifest entries.
            - `"sorted_orbit_counts"` (List[tuple[Any, int]]): A frequency-sorted list of
              `(orbit, count)` tuples.
            - `"most_frequent_orbit"` (Any): The orbit number with the highest frequency,
              or `None` if no orbits are present.
    """
    path_to_log_file = Path(path_to_logs) / f"sentinel1_orbitals_{unique_id}.txt"
    path_to_log_file.parent.mkdir(parents=True, exist_ok=True)
    if path_to_log_file.exists():
        path_to_log_file.unlink()  # Clear old logs for a clean run

    start_year = int(pd.Timestamp(start_date).year)
    end_year = int(pd.Timestamp(end_date).year)
    years = range(start_year, end_year + 1)

    manifest_filtered_by_bands = {}

    for year in years:
        # Filter items by calendar year
        year_items = [item for item in manifest if pd.Timestamp(item["date"]).year == year]
        if not year_items:
            continue

        # Filter out items missing the required bands
        manifest_with_required_bands = []
        for entry in year_items:
            item_obj = entry.get("_item_obj")
            if item_obj and hasattr(item_obj, "assets"):
                asset_keys = item_obj.assets.keys()
                if all(b in asset_keys for b in required_bands):
                    manifest_with_required_bands.append(entry)

        if not manifest_with_required_bands:
            continue

        # Aggregate orbit frequencies for items possessing the required bands
        orbits = [
            entry["relative_orbit"]
            for entry in manifest_with_required_bands
            if entry.get("relative_orbit") is not None
        ]

        counts = Counter(orbits)
        sorted_orbit_counts = counts.most_common()  # e.g., [(66, 97), (139, 3)]
        most_frequent_orbit = sorted_orbit_counts[0][0] if sorted_orbit_counts else None

        # Store audit metadata per year
        manifest_filtered_by_bands[year] = {
            "manifest_with_required_bands": manifest_with_required_bands,
            "sorted_orbit_counts": sorted_orbit_counts,
            "most_frequent_orbit": most_frequent_orbit
        }

        # Log results to text file
        with open(path_to_log_file, "a", encoding="utf-8") as f:
            f.write(f"Year {year} (Most frequent orbit: {most_frequent_orbit}):\n")
            for orb, cnt in sorted_orbit_counts:
                f.write(f"  - Orbit {orb}: {cnt} passes\n")
            f.write("\n\n")

    return manifest_filtered_by_bands


def filter_manifest_by_orbit_numbers(
    manifest_filtered_by_bands: Dict[int, Dict[str, Any]],
    target_orbits: Optional[Union[int, Dict[int, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Step 2: Trims the audited manifest data for each year, keeping only the items
    corresponding to the specified target orbit.

    Parameters
    ----------
    manifest_filtered_by_bands : Dict[int, Dict[str, Any]]
        The audited data dictionary returned by audit_s1_manifest.
    target_orbits : int or Dict[int, int], optional
        - If an int (scalar), it broadcasts and forces that same orbit for all years.
        - If a dict, maps specific years to target orbits (e.g., {2020: 66, 2021: 139}).
        - If None, defaults to the dominant orbit (argmax) computed per year during the audit.
    """
    # Normalize target_orbits: if a scalar integer is provided, broadcast it to all years in audit data
    if isinstance(target_orbits, int):
        target_orbits = {year: target_orbits for year in manifest_filtered_by_bands.keys()}
    elif target_orbits is None:
        target_orbits = {}

    manifest_filtered_by_orbit_numbers = []

    for year, data in manifest_filtered_by_bands.items():
        manifest_with_required_bands = data["manifest_with_required_bands"]
        most_frequent_orbit = data["most_frequent_orbit"]

        # Resolve target orbit (dictionary lookup fallback to dominant argmax)
        chosen_orbit = target_orbits.get(year, most_frequent_orbit)
        if chosen_orbit is None:
            continue

        # Filter valid items to keep only entries matching the chosen orbit
        year_manifest = [
            entry for entry in manifest_with_required_bands
            if entry.get("relative_orbit") == chosen_orbit
        ]
        manifest_filtered_by_orbit_numbers.extend(year_manifest)

    return manifest_filtered_by_orbit_numbers
