import numpy as np
import pandas as pd
from odc.geo import AnchorEnum, geom
from odc.geo.geobox import GeoBox
from fetch_stac import _bbox_around, _utm_epsg

CELL = 30  # 30m working resolution

def grid_of(lon: float, lat: float, size_m: float) -> tuple[int, int, tuple[int, int]]:
    """Calculate the (ix, iy, shape) grid alignment for the 30m cell tables."""
    gb = GeoBox.from_geopolygon(
        geom.box(*_bbox_around(lon, lat, size_m), "EPSG:4326"),
        resolution=CELL, crs=_utm_epsg(lon, lat), anchor=AnchorEnum.EDGE
    )
    return round(gb.affine.c / CELL), round(gb.affine.f / CELL) - 1, tuple(gb.shape)

def windows(start: str, end: str, step_days: int) -> list[tuple[str, str]]:
    """Generate consecutive date window tuples (start, end)."""
    starts = pd.date_range(start, end, freq=f"{step_days}D")
    return [
        (s.date().isoformat(), min(s + pd.Timedelta(days=step_days - 1), pd.Timestamp(end)).date().isoformat())
        for s in starts
    ]

def rows_from_array(
    values: dict[str, np.ndarray],
    grid: tuple[int, int, tuple[int, int]],
    when: pd.Timestamp
) -> pd.DataFrame:
    """Convert a dictionary of (H, W) arrays into a long-format DataFrame of valid cells."""
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
