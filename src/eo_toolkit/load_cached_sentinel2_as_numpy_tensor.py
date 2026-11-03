from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict

def load_cached_sentinel2_as_numpy_tensor(patch_dir: Path) -> Dict[str, any]:
    """
    Loads cached Sentinel-2 .npz patches into a NumPy tensor, along with
    metadata, cloud cover, and memorized band names.
    """
    arrays = []
    dates = []
    clouds = []
    band_names_list = None

    if not patch_dir.exists():
        return {
            "bands": np.empty((0, 0, 0, 0), dtype=np.float32),
            "dates": np.array([], dtype="datetime64[ns]"),
            "cloud_cover": np.array([], dtype=np.float32),
            "band_names": []
        }

    for file_path in sorted(patch_dir.glob("*.npz")):
        with np.load(file_path, allow_pickle=True) as opt:
            arr = opt["data"]
            if arr.ndim == 4:
                arr = arr[0]  # Shape: (bands, height, width)

            arrays.append(arr.astype(np.float32))
            dates.append(pd.to_datetime(str(opt["date"][0])))
            clouds.append(float(opt["cloud_cover"][0]) if "cloud_cover" in opt else np.nan)

            # Capture band names from the patch file
            if band_names_list is None and "band_names" in opt:
                band_names_list = list(opt["band_names"])

    tensor_data = np.stack(arrays) if arrays else np.empty((0, 0, 0, 0), dtype=np.float32)

    return {
        "bands": tensor_data,
        "dates": np.array(dates, dtype="datetime64[ns]"),
        "cloud_cover": np.array(clouds, dtype=np.float32),
        "band_names": band_names_list or []
    }
