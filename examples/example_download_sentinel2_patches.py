import hydra
from omegaconf import DictConfig
from pathlib import Path
import geopandas as gpd

from eo_toolkit import (
    get_s2_manifest,
    download_sentinel2_patches,
)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    # Load GeoJSON data
    path_to_geojson = Path(cfg.get("path_to_geojson", "./geojson"))
    region_name = "lisle_healthy_control.geojson"  # "darney_1_region", "lisle_healthy_control.geojson"
    path_to_geojson_file = path_to_geojson / region_name
    gdf = gpd.read_file(path_to_geojson_file)

    # Get total bounds [xmin, ymin, xmax, ymax]
    bbox = gdf.total_bounds.tolist()

    path_to_logs = Path(cfg.get("path_to_logs", "./logs"))
    path_to_data = cfg.get("path_to_data", "./data")

    start_date = cfg.get("start_date", "2014-01-01")
    end_date = cfg.get("end_date", "2026-12-31")

    max_cloud_cover = 20.0
    required_bands = (
        "coastal",    # Band 1 (Coastal aerosol)
        "blue",       # Band 2 (Blue)
        "green",      # Band 3 (Green)
        "red",        # Band 4 (Red)
        "rededge1",   # Band 5 (Vegetation red edge)
        "rededge2",   # Band 6 (Vegetation red edge)
        "rededge3",   # Band 7 (Vegetation red edge)
        "nir",        # Band 8 (NIR)
        "nir08",      # Band 8A (Narrow NIR)
        "nir09",      # Band 9 (Water vapour)
        "swir16",     # Band 11 (SWIR)
        "swir22",     # Band 12 (SWIR)
    )

    # Build the unique identifier string
    data_nickname = "lisle"  # "darney", "lisle"
    xmin, ymin, xmax, ymax = bbox
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    unique_id = f"{data_nickname}_{start_date}_{end_date}_{center_x:.4f}_{center_y:.4f}"

    # Fetch the raw manifest using bounding box and time range
    print(f"Querying Sentinel-2 STAC manifest...")
    raw_manifest = get_s2_manifest(
        bbox=bbox,
        start=start_date,
        end=end_date,
        path_to_logs=path_to_logs,
        unique_id=unique_id,
    )

    # Download patches
    download_sentinel2_patches(
        unique_id=unique_id,
        bbox=bbox,
        manifest=raw_manifest,
        path_to_data=path_to_data,
        required_bands=required_bands,
        max_cloud_cover=max_cloud_cover,
    )

if __name__ == "__main__":
    main()
