import math
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, Literal

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
    """Raised when a STAC search returns no scene."""


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _bbox_around(lon: float, lat: float, size_m: float) -> Bbox:
    """Lon/lat box of about ``size_m`` metres centred on a point."""
    if size_m <= 0:
        raise ValueError(f"size_m must be positive, got {size_m}.")
    if not -90.0 < lat < 90.0:
        raise ValueError(f"lat must be strictly between -90 and 90, got {lat}.")
    dlat = size_m / 111_320 / 2
    dlon = size_m / (111_320 * math.cos(math.radians(lat))) / 2
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _utm_epsg(lon: float, lat: float) -> str:
    """EPSG code of the WGS 84 / UTM zone containing the point (north: 326xx, south: 327xx)."""
    zone = int((lon + 180) // 6) % 60 + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _item_time(item: Item) -> datetime:
    return item.datetime or _EPOCH


def _item_date(item: Item) -> str:
    if item.datetime:
        return item.datetime.date().isoformat()
    return str(item.properties.get("datetime", "unknown"))[:10]


def _modifier(stac_url: str, modifier: Callable[[Any], Any] | None) -> Callable[[Any], Any] | None:
    """Explicit modifier wins; Planetary Computer URLs get automatic asset signing."""
    if modifier is not None or "planetarycomputer.microsoft.com" not in stac_url:
        return modifier
    import planetary_computer

    return planetary_computer.sign_inplace


def _search(
    stac_url: str,
    collection: str,
    bbox: Bbox,
    start: str,
    end: str,
    query: dict[str, Any] | None,
    modifier: Callable[[Any], Any] | None = None,
) -> list[Item]:
    client = Client.open(stac_url, modifier=_modifier(stac_url, modifier))
    items = list(
        client.search(
            collections=[collection], bbox=bbox, datetime=f"{start}/{end}", query=query or None
        ).item_collection()
    )
    if not items:
        raise NoSceneFoundError(f"No scene found for collection {collection!r} (query={query}).")
    return items


def _load(
    items: Sequence[Item],
    bands: Sequence[str],
    bbox: Bbox,
    crs: str,
    resolution: float,
    resampling: str,
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """Read only the window. Returns float32 ``(T, C, H, W)`` and one date per time step."""
    ds = odc.stac.load(
        list(items),
        bands=list(bands),
        bbox=bbox,
        crs=crs,
        resolution=resolution,
        groupby="solar_day",
        resampling=resampling,
    ).compute()
    data = np.stack([ds[b].values for b in bands], axis=1).astype(np.float32)
    dates = [str(np.datetime_as_string(t, unit="D")) for t in ds["time"].values]
    return data, dates


# --------------------------------------------------------------------------- #
# Sentinel-2
# --------------------------------------------------------------------------- #
def fetch_s2(
    lon: float,
    lat: float,
    size_m: float,
    start: str,
    end: str,
    bands: Sequence[str],
    stac_url: str,
    *,
    collection: str = "sentinel-2-l2a",
    max_cloud: float | None = 20.0,
    query: dict[str, Any] | None = None,
    scale_factor: float = 10_000.0,
    resolution: float = 10.0,
    resampling: str = "nearest",
    modifier: Callable[[Any], Any] | None = None,
) -> tuple[npt.NDArray[np.float32], str]:
    """Fetch the least cloudy Sentinel-2 scene as a windowed reflectance array.

    Parameters
    ----------
    lon, lat : float
        Target point in decimal degrees.
    size_m : float
        Side length of the square window around the point, in metres.
    start, end : str
        Search interval, ``'YYYY-MM-DD'``.
    bands : Sequence[str]
        Asset keys to fetch (e.g. ``["red", "nir"]`` on Earth Search, ``["B04", "B08"]``
        on Planetary Computer). Output channel order follows this sequence.
    stac_url : str
        STAC API endpoint, e.g. :data:`EARTH_SEARCH_URL`.
    collection : str, optional
        Collection ID. Default ``"sentinel-2-l2a"``.
    max_cloud : float | None, optional
        Only consider scenes with ``eo:cloud_cover < max_cloud``. ``None`` disables
        the filter. An ``eo:cloud_cover`` entry in ``query`` takes precedence.
    query : dict[str, Any] | None, optional
        Extra STAC query-extension filters.
    scale_factor : float, optional
        Divisor converting digital numbers to reflectance. Default ``10_000``.
    resolution : float, optional
        Output pixel size in metres. Default ``10``.
    resampling : str, optional
        Resampling for bands with a different native grid. Default ``"nearest"``.
    modifier : Callable | None, optional
        Item modifier that signs assets. Applied automatically for Planetary Computer.

    Returns
    -------
    reflectance : numpy.ndarray
        Float32 array ``(len(bands), H, W)``: ``raw / scale_factor``, NaN -> 0, not clipped.
    date : str
        Acquisition date of the selected scene, ``'YYYY-MM-DD'`` (or ``"unknown"``).

    Raises
    ------
    NoSceneFoundError
        If no scene matches.
    """
    bbox = _bbox_around(lon, lat, size_m)

    q: dict[str, Any] = dict(query or {})
    if max_cloud is not None:
        q.setdefault("eo:cloud_cover", {"lt": max_cloud})

    items = _search(stac_url, collection, bbox, start, end, q, modifier)

    # Lowest cloud cover; ties -> most recent. Missing cloud cover counts as 100 %.
    best = min(
        items,
        key=lambda it: (
            float(it.properties.get("eo:cloud_cover", 100.0)),
            -_item_time(it).timestamp(),
        ),
    )

    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    data, _ = _load([best], bands, bbox, _utm_epsg(lon, lat), resolution, resampling)
    return np.nan_to_num(data[0] / np.float32(scale_factor)), _item_date(best)


# --------------------------------------------------------------------------- #
# Sentinel-1
# --------------------------------------------------------------------------- #
def fetch_s1(
    lon: float,
    lat: float,
    size_m: float,
    start: str,
    end: str,
    stac_url: str,
    *,
    bands: Sequence[str] = ("vv", "vh"),
    collection: str = "sentinel-1-rtc",
    orbit_state: Literal["ascending", "descending"] | None = None,
    relative_orbit: int | None = None,
    query: dict[str, Any] | None = None,
    reducer: Literal["latest", "median"] = "latest",
    max_scenes: int = 20,
    to_db: bool = False,
    resolution: float = 10.0,
    resampling: str = "nearest",
    modifier: Callable[[Any], Any] | None = None,
) -> tuple[npt.NDArray[np.float32], str]:
    """Fetch Sentinel-1 backscatter for a window around a point.

    Meant for analysis-ready RTC products (linear power), e.g. Planetary Computer's
    ``sentinel-1-rtc``. For comparable scenes, fix ``orbit_state`` / ``relative_orbit``.

    Parameters
    ----------
    lon, lat : float
        Target point in decimal degrees.
    size_m : float
        Side length of the square window around the point, in metres.
    start, end : str
        Search interval, ``'YYYY-MM-DD'``.
    stac_url : str
        STAC API endpoint, e.g. :data:`PLANETARY_COMPUTER_URL`.
    bands : Sequence[str], optional
        Polarisation asset keys. Default ``("vv", "vh")``.
    collection : str, optional
        Collection ID. Default ``"sentinel-1-rtc"``.
    orbit_state : {"ascending", "descending"} | None, optional
        Filter on ``sat:orbit_state``.
    relative_orbit : int | None, optional
        Filter on ``sat:relative_orbit``.
    query : dict[str, Any] | None, optional
        Extra STAC query-extension filters (e.g. ``{"sar:instrument_mode": {"eq": "IW"}}``).
    reducer : {"latest", "median"}, optional
        ``"latest"``: newest acquisition day (slices from that day are mosaicked).
        ``"median"``: per-pixel NaN-aware median over the newest ``max_scenes`` items.
    max_scenes : int, optional
        Max number of items used for ``reducer="median"``. Default ``20``.
    to_db : bool, optional
        Convert to decibels, ``10 * log10(x)``. Default ``False``.
    resolution : float, optional
        Output pixel size in metres. Default ``10``.
    resampling : str, optional
        Resampling method. Default ``"nearest"``.
    modifier : Callable | None, optional
        Item modifier that signs assets. Applied automatically for Planetary Computer.

    Returns
    -------
    backscatter : numpy.ndarray
        Float32 array ``(len(bands), H, W)``, linear power or dB. Invalid pixels
        (non-finite or ``<= 0``) are NaN, not 0.
    date : str
        ``'YYYY-MM-DD'``; for a median over several dates, ``'first/last'``.

    Raises
    ------
    NoSceneFoundError
        If no scene matches.
    """
    if reducer not in ("latest", "median"):
        raise ValueError(f"reducer must be 'latest' or 'median', got {reducer!r}.")
    if max_scenes < 1:
        raise ValueError(f"max_scenes must be >= 1, got {max_scenes}.")

    bbox = _bbox_around(lon, lat, size_m)

    q: dict[str, Any] = dict(query or {})
    if orbit_state is not None:
        q.setdefault("sat:orbit_state", {"eq": orbit_state})
    if relative_orbit is not None:
        q.setdefault("sat:relative_orbit", {"eq": relative_orbit})

    items = sorted(
        _search(stac_url, collection, bbox, start, end, q, modifier), key=_item_time, reverse=True
    )
    if reducer == "latest":
        newest = _item_date(items[0])
        selected = [it for it in items if _item_date(it) == newest]
    else:
        selected = items[:max_scenes]

    odc.stac.configure_rio(cloud_defaults=True)
    data, dates = _load(selected, bands, bbox, _utm_epsg(lon, lat), resolution, resampling)

    data = np.where(np.isfinite(data) & (data > 0), data, np.float32(np.nan))
    if data.shape[0] > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
            out = np.nanmedian(data, axis=0)
    else:
        out = data[0]
    if to_db:
        out = 10.0 * np.log10(out)

    date = dates[0] if dates[0] == dates[-1] else f"{dates[0]}/{dates[-1]}"
    return out.astype(np.float32, copy=False), date
