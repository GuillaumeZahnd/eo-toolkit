import calendar
import datetime
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import odc.stac
import hydra
from omegaconf import DictConfig

from fetch_stac import (
    EARTH_SEARCH_URL,
    PLANETARY_COMPUTER_URL,
    _search,
    _utm_epsg,
    NoSceneFoundError,
)

S2_BANDS = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "swir16", "swir22"]
S1_BANDS = ["vv", "vh"]
MONTHS = [4, 5, 6, 7, 8, 9, 10]  # Leaf-on season


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def prefetch_region(cfg: DictConfig):
    output_dir = Path(cfg.preprocess.output_dir) / "regional_rasters"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bbox = cfg.preprocess.bbox_latlon
    years = cfg.preprocess.years
    
    # Define regional bounding box (min_lon, min_lat, max_lon, max_lat)
    region_bbox = (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
    center_lon = (bbox.min_lon + bbox.max_lon) / 2
    center_lat = (bbox.min_lat + bbox.max_lat) / 2
    crs = _utm_epsg(center_lon, center_lat)
    
    resolution = 10.0
    print(f"Prefetching regional GeoTIFFs into {output_dir} using CRS {crs}...")

    for yr in years:
        for m in MONTHS:
            period_str = f"{yr}-{m:02d}"
            start = f"{period_str}-01"
            end = f"{period_str}-{calendar.monthrange(yr, m)[1]:02d}"
            
            s2_path = output_dir / f"s2_{period_str}.tif"
            s1_path = output_dir / f"s1_{period_str}.tif"
            
            # Prefetch Sentinel-2 regional mosaic if missing
            if not s2_path.exists():
                try:
                    items = _search(EARTH_SEARCH_URL, "sentinel-2-l2a", region_bbox, start, end, {"eo:cloud_cover": {"lt": 30.0}})
                    best_item = min(items, key=lambda it: float(it.properties.get("eo:cloud_cover", 100.0)))
                    
                    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
                    ds = odc.stac.load(
                        [best_item],
                        bands=S2_BANDS,
                        bbox=region_bbox,
                        crs=crs,
                        resolution=resolution,
                        groupby="solar_day",
                    ).compute()
                    
                    data = np.stack([ds[b].values[0] for b in S2_BANDS], axis=0).astype(np.float32) / 10000.0
                    transform_affine = ds.odc.geobox.transform
                    
                    with rasterio.open(
                        s2_path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
                        count=data.shape[0], dtype=rasterio.float32, crs=crs, transform=transform_affine,
                        compress="lzw", predictor=2  # Adds lossless compression
                    ) as dst:
                        dst.write(data)
                    print(f"Saved regional S2: {s2_path.name}")
                except NoSceneFoundError:
                    print(f"No S2 scene found for {period_str}")

            # Prefetch Sentinel-1 regional mosaic if missing
            if not s1_path.exists():
                try:
                    items = _search(PLANETARY_COMPUTER_URL, "sentinel-1-rtc", region_bbox, start, end, {"sat:orbit_state": {"eq": "descending"}})
                    odc.stac.configure_rio(cloud_defaults=True)
                    ds = odc.stac.load(
                        items[:10],
                        bands=S1_BANDS,
                        bbox=region_bbox,
                        crs=crs,
                        resolution=resolution,
                        groupby="solar_day",
                    ).compute()
                    
                    # Compute median over time axis: resulting shape (bands, height, width)
                    data = np.nanmedian(np.stack([ds[b].values for b in S1_BANDS], axis=1), axis=0).astype(np.float32)
                    data = np.where(np.isfinite(data) & (data > 0), data, np.float32(np.nan))
                    transform_affine = ds.odc.geobox.transform
                    
                    with rasterio.open(
                        s1_path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
                        count=data.shape[0], dtype=rasterio.float32, crs=crs, transform=transform_affine,
                        compress="lzw", predictor=2
                    ) as dst:
                        dst.write(data)
                    print(f"Saved regional S1: {s1_path.name}")
                except NoSceneFoundError:
                    print(f"No S1 scene found for {period_str}")

    print("Regional prefetching complete!")

if __name__ == "__main__":
    prefetch_region()
