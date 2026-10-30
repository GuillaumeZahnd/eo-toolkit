import calendar
from pathlib import Path
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import hydra
from omegaconf import DictConfig

S2_BANDS = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "swir16", "swir22"]

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def process_efda_to_patches(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths & Config
    raw_dir = Path(cfg.preprocess.raw_dir)
    output_dir = Path(cfg.preprocess.output_dir)
    regional_dir = output_dir / "regional_rasters"
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    patch_size = cfg.preprocess.patch_size  # e.g., 64
    max_patches = getattr(cfg.preprocess, "max_patches", 800)
    years = cfg.preprocess.years

    # EFDA source rasters
    mask_path = raw_dir / "forestlanduse_mask_france_v30.tif"
    disturbance_path = raw_dir / "annual_disturbance_france_v30.tif"

    print(f"Loading pretrained AnySat model...")
    anysat_model = torch.hub.load("gastruc/anysat", "anysat", pretrained=True, flash_attn=False, trust_repo=True)
    anysat_model.to(device)
    anysat_model.eval()

    bbox = cfg.preprocess.bbox_latlon
    region_bbox_lonlat = (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)

    # Open EFDA mask to find valid forest pixels in our bounding box
    with rasterio.open(mask_path) as src_mask, rasterio.open(disturbance_path) as src_dist:
        # Transform lat/lon bbox to EFDA raster CRS
        transformer_to_efda = Transformer.from_crs("EPSG:4326", src_mask.crs, always_xy=True)
        min_x, min_y = transformer_to_efda.transform(region_bbox_lonlat[0], region_bbox_lonlat[1])
        max_x, max_y = transformer_to_efda.transform(region_bbox_lonlat[2], region_bbox_lonlat[3])

        # Get window bounds in the EFDA raster
        r_start, c_start = src_mask.index(min_x, max_y)
        r_end, c_end = src_mask.index(max_x, min_y)

        # Ensure bounds are valid and within raster limits
        r_start, r_end = max(0, min(r_start, r_end)), min(src_mask.height, max(r_start, r_end))
        c_start, c_end = max(0, min(c_start, c_end)), min(src_mask.width, max(c_start, c_end))

        print(f"Scanning EFDA forest mask within bounding box window...")
        # Read sub-window of the forest mask
        window = Window(c_start, r_start, c_end - c_start, r_end - r_start)
        mask_data = src_mask.read(1, window=window)
        dist_data = src_dist.read(1, window=window)

        # Find coordinates where forest mask is true/valid (> 0)
        forest_rows, forest_cols = np.where(mask_data > 0)
        
        if len(forest_rows) == 0:
            print("Warning: No forest pixels found in this bounding box!")
            return

        # Subsample if we have too many patches
        if len(forest_rows) > max_patches:
            indices = np.random.choice(len(forest_rows), max_patches, replace=False)
            forest_rows = forest_rows[indices]
            forest_cols = forest_cols[indices]

        print(f"Extracted {len(forest_rows)} valid forest patch locations. Beginning embedding extraction...")

        for idx, (r_offset, c_offset) in enumerate(zip(forest_rows, forest_cols)):
            abs_r = r_start + r_offset
            abs_c = c_start + c_offset

            # Get geographic coordinates of the pixel center
            x_coord, y_coord = src_mask.xy(abs_r, abs_c)
            transformer_to_latlon = Transformer.from_crs(src_mask.crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer_to_latlon.transform(x_coord, y_coord)

            # Get label from disturbance raster
            label = int(dist_data[r_offset, c_offset])

            # Pick a representative year and month (e.g., peak summer of the latest year)
            year = years[-1]
            month = 7
            period_str = f"{year}-{month:02d}"

            s2_path = regional_dir / f"s2_{period_str}.tif"
            s1_path = regional_dir / f"s1_{period_str}.tif"

            s2_patch = np.zeros((len(S2_BANDS), patch_size, patch_size), dtype=np.float32)
            s1_patch = np.zeros((3, patch_size, patch_size), dtype=np.float32)

            # Extract S2 window from cached regional raster
            if s2_path.exists():
                with rasterio.open(s2_path) as src_s2:
                    t_s2 = Transformer.from_crs("EPSG:4326", src_s2.crs, always_xy=True)
                    cx, cy = t_s2.transform(lon, lat)
                    try:
                        row_idx, col_idx = src_s2.index(cx, cy)
                        w_s2 = Window(col_idx - patch_size // 2, row_idx - patch_size // 2, patch_size, patch_size)
                        raw_s2 = src_s2.read(window=w_s2).astype(np.float32)
                        if raw_s2.shape[1] == patch_size and raw_s2.shape[2] == patch_size:
                            s2_patch = raw_s2
                    except Exception:
                        pass

            # Extract S1 window from cached regional raster
            if s1_path.exists():
                with rasterio.open(s1_path) as src_s1:
                    t_s1 = Transformer.from_crs("EPSG:4326", src_s1.crs, always_xy=True)
                    cx, cy = t_s1.transform(lon, lat)
                    try:
                        row_idx, col_idx = src_s1.index(cx, cy)
                        w_s1 = Window(col_idx - patch_size // 2, row_idx - patch_size // 2, patch_size, patch_size)
                        raw_s1 = src_s1.read(window=w_s1).astype(np.float32)
                        if raw_s1.shape[1] == patch_size and raw_s1.shape[2] == patch_size:
                            lin_vv = 10.0 ** (0.1 * raw_s1[0])
                            lin_vh = 10.0 ** (0.1 * raw_s1[1])
                            ratio = lin_vv / (lin_vh + 1e-6)
                            s1_patch = np.stack([raw_s1[0], raw_s1[1], ratio], axis=0)
                    except Exception:
                        pass

            doy = 196  # Mid-July day of year

            batch_data = {
                "s2": torch.tensor(s2_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device),
                "s2_dates": torch.tensor([[doy]], dtype=torch.long).to(device),
                "s1": torch.tensor(s1_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device),
                "s1_dates": torch.tensor([[doy]], dtype=torch.long).to(device),
            }

            with torch.no_grad():
                try:
                    feat = anysat_model(batch_data, patch_size=patch_size, output="patch")
                    embedding = feat.cpu().numpy()

                    out_file = embeddings_dir / f"sample_{idx:05d}.pt"
                    torch.save({
                        "embedding": embedding,
                        "label": label,
                        "lon": lon,
                        "lat": lat,
                        "year": year
                    }, out_file)
                except Exception as e:
                    print(f"Error embedding sample {idx}: {e}")

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(forest_rows)} patches...")

    print("Raster-based patch embedding extraction complete!")


if __name__ == "__main__":
    process_efda_to_patches()
