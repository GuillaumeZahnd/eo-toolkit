import hydra
from omegaconf import DictConfig
from pathlib import Path
import geopandas as gpd

from eo_toolkit import (
    get_s1_manifest,
    download_sentinel1_patches,
    filter_manifest_by_bands,
    filter_manifest_by_orbit_numbers,
)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    # Load GeoJSON data
    path_to_geojson = Path(cfg.get("path_to_geojson", "./geojson"))
    region_name = "darney_1_region.geojson"
    path_to_geojson_file = path_to_geojson / region_name
    gdf = gpd.read_file(path_to_geojson_file)

    # Get total bounds [xmin, ymin, xmax, ymax]
    bbox = gdf.total_bounds.tolist()

    path_to_logs = Path(cfg.get("path_to_logs", "./logs"))
    path_to_data = cfg.get("path_to_data", "./data")

    start_date = cfg.get("start_date", "2014-01-01")
    end_date = cfg.get("end_date", "2026-12-31")

    required_bands = ("vv", "vh")

    # Unpack bounding box and calculate center coordinates
    xmin, ymin, xmax, ymax = bbox
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0

    data_nickname = "darney"

    # Build the unique identifier string
    unique_id = f"{data_nickname}_{start_date}_{end_date}_{center_x:.4f}_{center_y:.4f}"

    # Fetch the complete raw manifest using the bounding box and full date range
    raw_manifest = get_s1_manifest(
        bbox,
        start_date,
        end_date,
    )
    print(f"Total raw manifest items fetched: {len(raw_manifest)}")

    # Filter the raw manifest year-by-year for band availability, and count yearly orbit passes
    manifest_filtered_by_bands = filter_manifest_by_bands(
        manifest=raw_manifest,
        start_date=start_date,
        end_date=end_date,
        required_bands=required_bands,
        path_to_logs=path_to_logs,
        unique_id=unique_id,
    )

    # Filter the manifest by orbitals (please inspect the logs firts to decide which orbits to select)
    manifest_filtered_by_orbit_number = filter_manifest_by_orbit_numbers(
        manifest_filtered_by_bands=manifest_filtered_by_bands,
        target_orbits=139
    )
    print(f"Filtered manifest contains {len(manifest_filtered_by_orbit_number)} observations.")

    # Download patches using the filtered manifest
    download_sentinel1_patches(
        manifest=manifest_filtered_by_orbit_number,
        unique_id=unique_id,
        bbox=bbox,
        path_to_data=path_to_data,
    )


if __name__ == "__main__":
    main()
