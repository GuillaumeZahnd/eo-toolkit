import math
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np
import numpy.typing as npt
import odc.stac
from pystac import Item
from pystac_client import Client

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

Bbox = tuple[float, float, float, float]
_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


class NoSceneFoundError(LookupError):
    """Raised when a STAC search returns no matching scene."""


def bbox_around(lon: float, lat: float, size_m: float) -> Bbox:
    """Calculates a lon/lat bounding box of approximately `size_m` metres centered on a point."""
    if size_m <= 0:
        raise ValueError(f"size_m must be positive, got {size_m}.")
    if not -90.0 < lat < 90.0:
        raise ValueError(f"lat must be strictly between -90 and 90, got {lat}.")
    dlat = size_m / 111_320 / 2
    dlon = size_m / (111_320 * math.cos(math.radians(lat))) / 2
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def utm_epsg(lon: float, lat: float) -> str:
    """Returns the EPSG code of the UTM zone containing the point (north: 326xx, south: 327xx)."""
    zone = int((lon + 180) // 6) % 60 + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def item_time(item: Item) -> datetime:
    return item.datetime or _EPOCH


def item_date(item: Item) -> str:
    if item.datetime:
        return item.datetime.date().isoformat()
    return str(item.properties.get("datetime", "unknown"))[:10]


def _modifier(stac_url: str, modifier: Callable[[Any], Any] | None) -> Callable[[Any], Any] | None:
    if modifier is not None or "planetarycomputer.microsoft.com" not in stac_url:
        return modifier
    import planetary_computer
    return planetary_computer.sign_inplace


def search_stac(
    stac_url: str,
    collection: str,
    bbox: Bbox,
    start: str,
    end: str,
    query: dict[str, Any] | None = None,
    modifier: Callable[[Any], Any] | None = None,
) -> list[Item]:
    """Queries a STAC API endpoint and returns matching items."""
    client = Client.open(stac_url, modifier=_modifier(stac_url, modifier))
    items = list(
        client.search(
            collections=[collection], bbox=bbox, datetime=f"{start}/{end}", query=query or None
        ).item_collection()
    )
    if not items:
        raise NoSceneFoundError(f"No scene found for collection {collection!r} (query={query}).")
    return items


def load_stac_raster(
    items: Sequence[Item],
    bands: Sequence[str],
    bbox: tuple[float, float, float, float],
    crs: str,
    resolution: float,
    resampling: str,
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """Loads windowed raster data from validated STAC items into a float32 array
    and corresponding date list.

    Parameters
    ----------
    items : Sequence[Item]
        A sequence of PySTAC Item objects (pre-filtered to guarantee the presence of the requested bands).
    bands : Sequence[str]
        The band names to load (e.g., ``("vv", "vh")``).
    bbox : tuple[float, float, float, float]
        Spatial bounding box coordinates ``(xmin, ymin, xmax, ymax)`` in the target CRS.
    crs : str
        Target Coordinate Reference System string (e.g., a UTM EPSG string like ``"EPSG:32632"``).
    resolution : float
        Target spatial resolution in meters.
    resampling : str
        Resampling method to apply during loading (e.g., ``"nearest"``, ``"bilinear"``).

    Returns
    -------
    tuple[npt.NDArray[np.float32], list[str]]
        A tuple containing:
        - data: A float32 NumPy array stacked across the requested bands.
        - dates: A list of ISO date strings representing each temporal slice.
    """
    # Load and compute spatial/temporal subset via odc.stac
    ds = odc.stac.load(
        list(items),
        bands=list(bands),
        bbox=bbox,
        crs=crs,
        resolution=resolution,
        groupby="solar_day",
        resampling=resampling,
    ).compute()

    # Extract band values and stack into a clean float32 array
    data = np.stack([ds[b].values for b in bands], axis=1).astype(np.float32)

    # Extract unique date strings per acquisition slice
    dates = [str(np.datetime_as_string(t, unit="D")) for t in ds["time"].values]

    return data, dates
