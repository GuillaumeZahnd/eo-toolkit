from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict


def load_cached_sentinel1_as_numpy_tensor(patch_dir: Path) -> Dict[str, np.ndarray]:
    """
    Loads cached Sentinel-1 .npz patches into a NumPy tensor, with associated metadata.

    Tensor structure: (time, bands, height, width)
      - Band 0: VV polarization backscatter
      - Band 1: VH polarization backscatter

    Parameters:
    -----------
    patch_dir : Path
        The directory path containing the cached .npz raster patches.

    Returns:
    --------
    dict of np.ndarray
        A dictionary containing:
        - 'bands': 4D float32 tensor of shape (N, 2, H, W) containing VV and VH bands.
        - 'dates': 1D array of datetime64[ns] timestamps for each acquisition.
        - 'orbits': 1D array of int64 relative orbit numbers.
    """
    arrays = []
    dates = []
    orbits = []

    if not patch_dir.exists():
        return {
            "bands": np.empty((0, 2, 0, 0), dtype=np.float32),
            "dates": np.array([], dtype="datetime64[ns]"),
            "orbits": np.array([], dtype=np.int64)
        }

    for file_path in sorted(patch_dir.glob("*.npz")):
        with np.load(file_path, allow_pickle=True) as sar:
            arr = sar["data"]
            if arr.ndim == 4:
                arr = arr[0]  # Squeeze batch dimension -> shape: (bands, height, width)

            # Ensure consistent float32 typing for machine learning / numerical operations
            arrays.append(arr.astype(np.float32))
            dates.append(pd.to_datetime(str(sar["date"][0])))
            orbits.append(int(sar["orbit_number"][0]))

    if not arrays:
        return {
            "bands": np.empty((0, 2, 0, 0), dtype=np.float32),
            "dates": np.array([], dtype="datetime64[ns]"),
            "orbits": np.array([], dtype=np.int64)
        }

    # Stack into a single 4D tensor: shape (time, bands=2, height, width)
    tensor_data = np.stack(arrays)

    return {
        "bands": tensor_data,
        "dates": np.array(dates, dtype="datetime64[ns]"),
        "orbits": np.array(orbits, dtype=np.int64)
    }
