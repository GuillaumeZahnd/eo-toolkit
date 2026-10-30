import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from pathlib import Path
import hydra
from omegaconf import DictConfig


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

    with rasterio.open(annual_dist_path) as src_dist, rasterio.open(mask_path) as src_mask:
        
        # Transform Lat/Lon bounding box to the raster's native CRS (EPSG:3035)
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", 
            src_dist.crs, 
            bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat
        )
        
        # Convert geographic window bounds to pixel window
        window_box = rasterio.windows.from_bounds(left, bottom, right, top, src_dist.transform)
        
        col_off = max(0, int(window_box.col_off))
        row_off = max(0, int(window_box.row_off))
        width = min(src_dist.width - col_off, int(window_box.width))
        height = min(src_dist.height - row_off, int(window_box.height))
        
        patch_id = 0
        print(f"Extracting patches from Vosges sub-region (Window size: {width}x{height} pixels)...")
        
        for y in range(row_off, row_off + height - patch_size, patch_size):
            for x in range(col_off, col_off + width - patch_size, patch_size):
                window = Window(x, y, patch_size, patch_size)
                
                # Check forest mask: ensure patch has substantial forest cover (>30%)
                mask_data = src_mask.read(1, window=window)
                if mask_data.size == 0 or (mask_data == 0).mean() > 0.7:
                    continue
                
                dist_data = src_dist.read(1, window=window)
                
                patch_dir = out_path / f"patch_{patch_id:06d}"
                patch_dir.mkdir(exist_ok=True)
                
                # Save dummy feature tensors & weather (ready for AnySat replacement)
                for yr in years:
                    np.save(patch_dir / f"embedding_{yr}.npy", np.random.randn(768, patch_size, patch_size).astype(np.float32))
                    if yr > years[0]:
                        np.save(patch_dir / f"embedding_change_{yr}_{yr-1}.npy", np.random.randn(768, patch_size, patch_size).astype(np.float32))
                    np.save(patch_dir / f"weather_{yr-1}.npy", np.random.randn(12).astype(np.float32))
                
                # Save target disturbance label maps for each target year
                for yr in years[1:]:
                    label_map = (dist_data == yr).astype(np.float32)
                    np.save(patch_dir / f"disturbance_{yr}.npy", label_map)
                
                patch_id += 1
                if max_patches and patch_id >= max_patches:
                    break
            if max_patches and patch_id >= max_patches:
                break

    print(f"Successfully generated {patch_id} focused forest patches in {output_dir}/")


if __name__ == "__main__":
    process_efda_to_patches()
