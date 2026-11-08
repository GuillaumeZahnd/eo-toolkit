from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

from .utils import _to_naive_timestamp


def _empty_s2_result() -> Dict[str, Any]:
    return {
        "bands": np.empty((0, 0, 0, 0), dtype=np.float32),
        "dates": np.array([], dtype="datetime64[ns]"),
        "cloud_cover": np.array([], dtype=np.float32),
        "band_names": [],
    }


def _to_naive_timestamp(value) -> pd.Timestamp:
    """Parse a stored date and drop any timezone (converted to UTC first), so datetime64[ns] is safe."""
    ts = pd.to_datetime(str(value))
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def load_cached_sentinel2_as_numpy_tensor(patch_dir: Path) -> Dict[str, Any]:
    """
    Loads cached Sentinel-2 .npz patches into a NumPy tensor, along with metadata, cloud cover and band names.

    Tensor structure: (time, bands, height, width), sorted by date. Masked pixels (clouds, invalid) are NaN.

    Safety checks (each raises instead of loading silently wrong data):
      - every file must carry 'band_names', identical (same names, same order) in all files
      - bands in 'data' must equal len(band_names); all files must share (bands, height, width)
      - no two files may share the same date

    Returns:
    --------
    dict
        - 'bands': float32 tensor (N, C, H, W)
        - 'dates': datetime64[ns] (N,), ascending
        - 'cloud_cover': float32 (N,), NaN where the file has no cloud_cover
        - 'band_names': list of str, band_names[i] describes tensor[:, i]
    """
    if not patch_dir.exists():
        return _empty_s2_result()

    arrays, dates, clouds = [], [], []
    band_names = None

    for file_path in sorted(patch_dir.glob("*.npz")):
        with np.load(file_path) as opt:  # no pickle needed: numbers and strings only
            arr = opt["data"]
            if arr.ndim == 4:
                arr = arr[0]  # (bands, height, width)

            names = [str(n) for n in opt["band_names"]]  # KeyError if missing: no guessing
            if band_names is None:
                band_names = names
            elif names != band_names:
                raise ValueError(f"{file_path.name}: bands {names} != {band_names}")
            if arr.shape[0] != len(names):
                raise ValueError(f"{file_path.name}: data has {arr.shape[0]} bands but band_names has {len(names)}")

            arrays.append(arr.astype(np.float32))
            dates.append(_to_naive_timestamp(opt["date"][0]))
            clouds.append(float(opt["cloud_cover"][0]) if "cloud_cover" in opt else np.nan)

    if not arrays:
        return _empty_s2_result()

    tensor_data = np.stack(arrays)  # raises if bands/height/width differ between files
    dates_arr = np.array(dates, dtype="datetime64[ns]")
    clouds_arr = np.array(clouds, dtype=np.float32)

    order = np.argsort(dates_arr, kind="stable")
    tensor_data, dates_arr, clouds_arr = tensor_data[order], dates_arr[order], clouds_arr[order]

    if np.unique(dates_arr).size < dates_arr.size:
        dup = dates_arr[:-1][dates_arr[1:] == dates_arr[:-1]]
        raise ValueError(f"Duplicate dates (remove files): {np.unique(dup)[:5]}")

    return {
        "bands": tensor_data,
        "dates": dates_arr,
        "cloud_cover": clouds_arr,
        "band_names": band_names,
    }
