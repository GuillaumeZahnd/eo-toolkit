import calendar
import datetime
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform, transform_bounds
import torch
import hydra
from omegaconf import DictConfig

# Ensure local src/ folder is first so project modules take precedence
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Append any-sat root so hubconf is accessible without shadowing local files
ANYSAT_ROOT = Path("/home/guillaume/PROJECTS/GitHub/any-sat")
if str(ANYSAT_ROOT) not in sys.path:
    sys.path.append(str(ANYSAT_ROOT))

from hubconf import AnySat
from fetch_stac import EARTH_SEARCH_URL, PLANETARY_COMPUTER_URL, fetch_s1, fetch_s2, NoSceneFoundError
from build_sample import fetch_weather_monthly


S2_BANDS = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "swir16", "swir22"]
S1_BANDS = ["vv", "vh"]
MONTHS = [4, 5, 6, 7, 8, 9, 10]  # Leaf-on season


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def process_efda_to_patches(cfg: DictConfig):
    raw_dir = cfg.preprocess.raw_dir
    output_dir = cfg.preprocess.output_dir
    patch_size = cfg.preprocess.patch_size
    years = cfg.preprocess.years
    bbox = cfg.preprocess.bbox_latlon
    max_patches = cfg.preprocess.get("max_patches", None)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    annual_dist_path = Path(raw_dir) / "annual_disturbance_france_v30.tif"
    mask_path = Path(raw_dir) / "forestlanduse_mask_france_v30.tif"

    if not annual_dist_path.exists():
        raise FileNotFoundError(f"Could not find {annual_dist_path}. Check your dataset directory.")

    # Initialize local AnySat model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading pretrained AnySat from local codebase...")
    anysat_model = AnySat.from_pretrained('base', flash_attn=False).to(device)
    anysat_model.eval()

    with rasterio.open(annual_dist_path) as src_dist, rasterio.open(mask_path) as src_mask:
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src_dist.crs, bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat
        )
        window_box = rasterio.windows.from_bounds(left, bottom, right, top, src_dist.transform)

        col_off = max(0, int(window_box.col_off))
        row_off = max(0, int(window_box.row_off))
        width = min(src_dist.width - col_off, int(window_box.width))
        height = min(src_dist.height - row_off, int(window_box.height))

        patch_id = 0
        print(f"Extracting & embedding patches from Vosges sub-region...")

        for y in range(row_off, row_off + height - patch_size, patch_size):
            for x in range(col_off, col_off + width - patch_size, patch_size):
                window = Window(x, y, patch_size, patch_size)

                mask_data = src_mask.read(1, window=window)
                if mask_data.size == 0 or (mask_data == 0).mean() > 0.7:
                    continue

                dist_data = src_dist.read(1, window=window)
                patch_dir = out_path / f"patch_{patch_id:06d}"
                patch_dir.mkdir(exist_ok=True)

                # Compute geographic center (lon, lat) of the window in EPSG:4326
                center_x = x + patch_size / 2
                center_y = y + patch_size / 2
                lon_arr, lat_arr = transform(src_dist.crs, "EPSG:4326", [center_x], [center_y])
                lon, lat = lon_arr[0], lat_arr[0]
                resolution = 10.0
                size_m = patch_size * resolution

                embeddings = {}
                for yr in years:
                    periods = [f"{yr}-{m:02d}" for m in MONTHS]
                    s2_list, s2_doy_list = [], []
                    s1_list, s1_doy_list = [], []

                    for p in periods:
                        y_val, m_val = int(p[:4]), int(p[5:])
                        start = f"{p}-01"
                        end = f"{p}-{calendar.monthrange(y_val, m_val)[1]:02d}"

                        # Fetch Sentinel-2
                        try:
                            s2_arr, s2_date_str = fetch_s2(
                                lon, lat, size_m, start, end, S2_BANDS, EARTH_SEARCH_URL, resolution=resolution
                            )
                            s2_list.append(s2_arr)
                            doy = datetime.date.fromisoformat(s2_date_str).timetuple().tm_yday
                            s2_doy_list.append(doy)
                        except NoSceneFoundError:
                            s2_list.append(np.zeros((len(S2_BANDS), patch_size, patch_size), dtype=np.float32))
                            s2_doy_list.append(0)

                        # Fetch Sentinel-1
                        try:
                            s1_arr, s1_date_str = fetch_s1(
                                lon, lat, size_m, start, end, PLANETARY_COMPUTER_URL,
                                bands=S1_BANDS, orbit_state="descending", to_db=True, resolution=resolution
                            )
                            # s1_arr shape: (2, H, W) -> [vv, vh] in dB
                            # Compute 3rd channel (Ratio) in linear scale
                            lin_vv = 10.0 ** (0.1 * s1_arr[0])
                            lin_vh = 10.0 ** (0.1 * s1_arr[1])
                            ratio = lin_vv / (lin_vh + 1e-6)

                            # Stack into 3 channels: [VV, VH, Ratio] matching AnySat's 's1' expectation
                            s1_3ch = np.stack([s1_arr[0], s1_arr[1], ratio], axis=0) # Shape: (3, H, W)

                            s1_list.append(s1_3ch)
                            doy = datetime.date.fromisoformat(s1_date_str.split("/")[-1]).timetuple().tm_yday
                            s1_doy_list.append(doy)
                        except NoSceneFoundError:
                            s1_list.append(np.zeros((3, patch_size, patch_size), dtype=np.float32))
                            s1_doy_list.append(0)

                    # Fetch weather aggregates
                    weather_arr = fetch_weather_monthly(lon, lat, periods)
                    np.save(patch_dir / f"weather_{yr}.npy", weather_arr)

                    # Prepare tensors for AnySat
                    s2_tensor = torch.from_numpy(np.stack(s2_list)).unsqueeze(0).float()
                    s2_dates = torch.tensor([s2_doy_list], dtype=torch.long)
                    s1_tensor = torch.from_numpy(np.stack(s1_list)).unsqueeze(0).float()
                    s1_dates = torch.tensor([s1_doy_list], dtype=torch.long)

                    batch_data = {
                        "s2": s2_tensor.to(device),
                        "s2_dates": s2_dates.to(device),
                        "s1": s1_tensor.to(device),
                        "s1_dates": s1_dates.to(device),
                    }

                    with torch.no_grad():
                        feat = anysat_model(batch_data, patch_size=patch_size, output='patch')
                        feat_np = feat.squeeze(0).cpu().numpy()

                    embeddings[yr] = feat_np
                    np.save(patch_dir / f"embedding_{yr}.npy", feat_np)

                # Compute temporal drift / change vector between consecutive years
                for i in range(1, len(years)):
                    yr = years[i]
                    prev_yr = years[i-1]
                    change_vec = embeddings[yr] - embeddings[prev_yr]
                    np.save(patch_dir / f"embedding_change_{yr}_{prev_yr}.npy", change_vec)

                # Save target disturbance masks
                for yr in years[1:]:
                    label_map = (dist_data == yr).astype(np.float32)
                    np.save(patch_dir / f"disturbance_{yr}.npy", label_map)

                patch_id += 1
                if max_patches and patch_id >= max_patches:
                    break
            if max_patches and patch_id >= max_patches:
                break

    print(f"Successfully processed and embedded {patch_id} patches with AnySat into {output_dir}/")


if __name__ == "__main__":
    process_efda_to_patches()
