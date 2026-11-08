from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

from .utils import _to_naive_timestamp


def _empty_s1_result() -> Dict[str, np.ndarray]:
    return {
        "bands": np.empty((0, 0, 0, 0), dtype=np.float32),
        "dates": np.array([], dtype="datetime64[ns]"),
        "orbits": np.array([], dtype=np.int64),
        "band_names": np.array([], dtype=str),
    }


def load_cached_sentinel1_as_numpy_tensor(patch_dir: Path) -> Dict[str, np.ndarray]:
    """
    Loads cached Sentinel-1 .npz patches into a NumPy tensor, with associated metadata.

    Tensor structure: (time, bands, height, width), sorted by date.

    Safety checks (each raises ValueError / KeyError instead of loading silently wrong data):
      - every file must carry 'band_names', and they must be identical (same names, same order) in all files
      - every file must have the same (bands, height, width) shape, and bands must match len(band_names)
      - no two files may share the same date

    Parameters:
    -----------
    patch_dir : Path
        The directory path containing the cached .npz raster patches.

    Returns:
    --------
    dict of np.ndarray
        - 'bands': 4D float32 tensor of shape (N, C, H, W) containing band backscatter.
        - 'dates': 1D array of datetime64[ns] timestamps for each acquisition, ascending.
        - 'orbits': 1D array of int64 relative orbit numbers.
        - 'band_names': 1D array of str, band_names[i] describes tensor[:, i].
    """
    if not patch_dir.exists():
        return _empty_s1_result()

    arrays, dates, orbits = [], [], []
    band_names = None

    for file_path in sorted(patch_dir.glob("*.npz")):
        with np.load(file_path) as sar:
            arr = sar["data"]
            if arr.ndim == 4:
                arr = arr[0]  # squeeze batch dimension -> (bands, height, width)

            names = [str(n) for n in sar["band_names"]]  # KeyError if missing: no guessing
            if band_names is None:
                band_names = names
            elif names != band_names:
                raise ValueError(f"{file_path.name}: bands {names} != {band_names}")
            if arr.shape[0] != len(names):
                raise ValueError(f"{file_path.name}: data has {arr.shape[0]} bands but band_names has {len(names)}")

            arrays.append(arr.astype(np.float32))
            dates.append(_to_naive_timestamp(sar["date"][0]))
            orbits.append(int(sar["orbit_number"][0]))

    if not arrays:
        return _empty_s1_result()

    tensor_data = np.stack(arrays)  # raises if bands/height/width differ between files
    dates_arr = np.array(dates, dtype="datetime64[ns]")
    orbits_arr = np.array(orbits, dtype=np.int64)

    order = np.argsort(dates_arr, kind="stable")
    tensor_data, dates_arr, orbits_arr = tensor_data[order], dates_arr[order], orbits_arr[order]

    if np.unique(dates_arr).size < dates_arr.size:
        dup = dates_arr[:-1][dates_arr[1:] == dates_arr[:-1]]
        raise ValueError(f"Duplicate dates (filter by orbit or remove files): {np.unique(dup)[:5]}")

    return {
        "bands": tensor_data,
        "dates": dates_arr,
        "orbits": orbits_arr,
        "band_names": np.array(band_names, dtype=str),
    }
