import math
from typing import Any
import numpy as np
import odc.stac
from pystac_client import Client


def fetch_stac(
    lon: float,
    lat: float,
    size_m: float,
    start: str,
    end: str,
    bands: list[str],
    stac_url: str,
    collection: str = "sentinel-2-l2a",
    query: dict[str, Any] | None = None,
    scale_factor: float = 10_000.0,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], str]:
    """Fetch from a STAC catalog.

    Parameters
    ----------
    lon : float
        Longitude of the target point in decimal degrees.
    lat : float
        Latitude of the target point in decimal degrees.
    size_m : float
        Size of the square spatial window around the point in meters.
    start : str
        Start date for the STAC search in 'YYYY-MM-DD' format.
    end : str
        End date for the STAC search in 'YYYY-MM-DD' format.
    bands : list[str]
        List of band asset keys to fetch.
    stac_url : str
        URL of the STAC endpoint/catalog.

    Returns
    -------
    tuple[np.ndarray[Any, np.dtype[np.float32]], str]
        A tuple containing:
        - reflectance: A float32 NumPy array of shape (len(bands), H, W) normalized to [0, 1.5].
        - date_str: The calendar date string of the least cloudy scene found[cite: 2].
    """
    # Square lon/lat box of about size_m around the point
    dlat = size_m / 111_320 / 2
    dlon = size_m / (111_320 * math.cos(math.radians(lat))) / 2
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

    # Search with dynamic collection and query parameters (metadata only, nothing heavy yet)
    search_args = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": f"{start}/{end}",
    }
    if query:
        search_args["query"] = query

    items = Client.open(stac_url).search(**search_args).item_collection()

    if not items:
        raise SystemExit(f"No scene found for collection {collection}.")

    # Sort or select best item (if optical, sort by cloud cover; if radar, pick most recent or specific orbit)
    if "cloud_cover" in str(query):
        best = min(items, key=lambda it: it.properties.get("eo:cloud_cover", 0))
    else:
        best = max(items, key=lambda it: it.datetime or "")

    # Read only the specified window, all bands resampled to a 10 m grid (UTM zone)
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    utm = f"EPSG:{32600 + int((lon + 180) // 6) + 1}"
    ds = odc.stac.load([best], bands=bands, bbox=bbox, crs=utm, resolution=10).compute()

    raw = np.stack([ds[b].values[0] for b in bands]).astype(np.float32)
    processed = np.nan_to_num(raw / scale_factor)

    item_date = best.datetime.date() if best.datetime else best.properties.get("datetime", "unknown")

    return processed, str(item_date)
