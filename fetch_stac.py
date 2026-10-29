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
    max_cloud: float,
    bands: list[str],
    stac_url: str,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], str]:
    """Fetch a small Sentinel-2 L2A window around a point from a STAC catalog.

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
    max_cloud : float
        Maximum allowable cloud cover percentage.
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

    # Search the catalogue (metadata only, nothing heavy yet)
    items = Client.open(stac_url).search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}}).item_collection()

    if not items:
        raise SystemExit("No scene found: widen the dates or raise max_cloud.")

    best = min(items, key=lambda it: it.properties["eo:cloud_cover"])

    # Read only the specified window, all bands resampled to a 10 m grid (UTM zone)
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    utm = f"EPSG:{32600 + int((lon + 180) // 6) + 1}"
    ds = odc.stac.load([best], bands=bands, bbox=bbox, crs=utm, resolution=10).compute()

    raw = np.stack([ds[b].values[0] for b in bands]).astype(np.float32)
    reflectance = np.clip(raw / 10_000.0, 0.0, 1.5).astype(np.float32)

    # Fallback/safeguard for datetime attribute
    item_date = best.datetime.date() if best.datetime else best.properties.get("datetime", "unknown")

    return reflectance, str(item_date)
