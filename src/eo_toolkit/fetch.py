import numpy as np
import pandas as pd
from typing import Literal
from odc.geo import AnchorEnum, geom
from odc.geo.geobox import GeoBox
from datetime import datetime


from .stac import (
    EARTH_SEARCH_URL,
    PLANETARY_COMPUTER_URL,
    NoSceneFoundError,
    bbox_around,
    utm_epsg,
    item_time,
    item_date,
    search_stac,
    load_stac_raster,
)


def grid_of(lon: float, lat: float, size_m: float) -> tuple[int, int, tuple[int, int]]:
    """Calculates the (ix, iy, shape) grid alignment for 30m cell tables."""
    gb = GeoBox.from_geopolygon(
        geom.box(*bbox_around(lon, lat, size_m), "EPSG:4326"),
        resolution=CELL, crs=utm_epsg(lon, lat), anchor=AnchorEnum.EDGE
    )
    return round(gb.affine.c / CELL), round(gb.affine.f / CELL) - 1, tuple(gb.shape)


def windows(start: str, end: str, step_days: int) -> list[tuple[str, str]]:
    """Generates consecutive date window tuples (start, end)."""
    starts = pd.date_range(start, end, freq=f"{step_days}D")
    return [
        (s.date().isoformat(), min(s + pd.Timedelta(days=step_days - 1), pd.Timestamp(end)).date().isoformat())
        for s in starts
    ]


def parse_date_midpoint(date_str: str) -> pd.Timestamp:
    """Computes the midpoint timestamp from a single date or date range string ('YYYY-MM-DD' or 'first/last')."""
    d = [pd.Timestamp(s) for s in date_str.split("/")]
    return d[0] + (d[-1] - d[0]) / 2


def rows_from_array(values: dict[str, np.ndarray], grid: tuple[int, int, tuple[int, int]], when: pd.Timestamp) -> pd.DataFrame:
    """Converts a dictionary of (H, W) arrays into a long-format DataFrame of valid cells."""
    ix0, iy_top, shape = grid
    first = next(iter(values.values()))
    if first.shape != shape:
        raise ValueError(f"Array shape {first.shape} does not match expected grid shape {shape}.")
    H, W = shape
    df = pd.DataFrame({
        "ix": np.tile(ix0 + np.arange(W), H).astype("int32"),
        "iy": np.repeat(iy_top - np.arange(H), W).astype("int32"),
        "date": when.normalize(),
        **{k: v.ravel() for k, v in values.items()}
    })
    return df.dropna(subset=list(values), how="all")


def fetch_s2_ndre_timeseries(
    lon: float, lat: float, size_m: float, start: str, end: str,
    step_days: int = 14, max_cloud: float = 60.0, stac_url: str = EARTH_SEARCH_URL
) -> pd.DataFrame:
    """Generates a tabular time series of NDRE using temporal windows."""
    grid = grid_of(lon, lat, size_m)
    out = []
    for a, b in windows(start, end, step_days):
        try:
            arr, d = fetch_s2_image(lon, lat, size_m, a, b, stac_url=stac_url, max_cloud=max_cloud)
            out.append(rows_from_array({"ndre": arr[0]}, grid, parse_date_midpoint(d)))
        except NoSceneFoundError:
            continue
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["ix", "iy", "date", "ndre"])
