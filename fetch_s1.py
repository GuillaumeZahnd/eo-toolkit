import numpy as np
import numpy.typing as npt
import pandas as pd
from pystac import Item
from collections import Counter


from stac import (
    PLANETARY_COMPUTER_URL,
    NoSceneFoundError,
    bbox_around,
    utm_epsg,
    item_time,
    item_date,
    search_stac,
    load_stac_raster,
)

from fetch import CELL


def get_s1_manifest(
    lon: float,
    lat: float,
    size_m: float,
    start: str,
    end: str,
    orbit_state: str = "descending",
    stac_url: str = PLANETARY_COMPUTER_URL,
) -> list[dict]:
    """Queries the STAC catalog once and returns a clean manifest of all available
    Sentinel-1 acquisitions in the given spatiotemporal range without orbit pre-filtering.

    Parameters
    ----------
    lon : float
        Longitude of the area's center point (WGS84).
    lat : float
        Latitude of the area's center point (WGS84).
    size_m : float
        Spatial window size in meters around the center point.
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
    bbox = bbox_around(lon, lat, size_m)
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


def fetch_s1_image(
    item: Item,
    lon: float,
    lat: float,
    size_m: float,
    to_db: bool = True,
    resolution: float = CELL,
) -> tuple[npt.NDArray[np.float32], str]:
    """Loads a single pre-selected Sentinel-1 STAC item into a (2, H, W) array.

    Parameters
    ----------
    item : pystac.Item
        The pre-selected Sentinel-1 STAC item to load.
    lon : float
        Longitude of the area's center point (WGS84).
    lat : float
        Latitude of the area's center point (WGS84).
    size_m : float
        Spatial window size in meters around the center point.
    to_db : bool, optional
        If True, converts linear power backscatter to decibels ($10 \cdot \log_{10}(x)$).
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
    bbox = bbox_around(lon, lat, size_m)
    exact_date = item_date(item)

    data, _ = load_stac_raster([item], ("vv", "vh"), bbox, utm_epsg(lon, lat), resolution, "nearest")
    arr = data[0]

    # Mask invalid/non-finite values
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, np.float32(np.nan))
    if to_db:
        arr = 10.0 * np.log10(arr)

    return arr.astype(np.float32, copy=False), exact_date


def select_closest_manifest_item(manifest: list[dict], target_date: str) -> dict | None:
    """Selects the manifest entry closest to a given target date.

    Parameters
    ----------
    manifest : list[dict]
        The list of acquisition dicts returned by :func:`get_s1_manifest`.
    target_date : str
        The target date to match in ISO format ('YYYY-MM-DD').

    Returns
    -------
    dict | None
        The dictionary of the closest manifest entry, or None if the manifest is empty.
    """
    if not manifest:
        return None

    t_target = pd.Timestamp(target_date)

    # Find the entry with the minimum absolute time difference
    closest_entry = min(
        manifest,
        key=lambda entry: abs(pd.Timestamp(entry["date"]) - t_target)
    )

    return closest_entry


def filter_manifest_by_best_orbit(
    manifest: list[dict]
) -> tuple[list[dict], int | None, list[tuple[int, int]]]:
    """Analyzes a Sentinel-1 manifest, returns frequency counts of all relative
    orbits for auditing, determines the dominant orbit by frequency, and filters
    the manifest to retain only items from that track.

    Parameters
    ----------
    manifest : list[dict]
        A list of dictionary records returned by :func:`get_s1_manifest`, where
        each entry contains metadata including ``"relative_orbit"``.

    Returns
    -------
    tuple[list[dict], int | None, list[tuple[int, int]]]
        A 3-element tuple containing:
        - filtered_manifest: The subset of manifest items matching the chosen track.
        - orbit_number: The integer ID of the selected relative orbit, or ``None``.
        - sorted_counts: A list of ``(orbit_number, count)`` tuples sorted in
          descending order of frequency.

    Notes
    -----
    [ROOM FOR IMPROVEMENT] The current selection strategy uses a simple maximum
    frequency count. In close tie scenarios or boundary overlaps, this can cause
    year-on-year track flipping. Future iterations can introduce stateful track
    persistence or incidence angle checks to smooth transitions.
    """
    if not manifest:
        return [], None, []

    orbits = [entry["relative_orbit"] for entry in manifest if entry.get("relative_orbit") is not None]

    if not orbits:
        return manifest, None, []

    # Count frequencies and sort descending
    counts = Counter(orbits)
    sorted_counts = counts.most_common()  # e.g., [(66, 97), (139, 3)]

    # The 1st item in most_common is the most frequent (dominant) orbit
    orbit_number = sorted_counts[0][0]

    # Filter manifest to keep only the entries matching the dominant orbit
    filtered_manifest = [entry for entry in manifest if entry.get("relative_orbit") == orbit_number]

    return filtered_manifest, orbit_number, sorted_counts
