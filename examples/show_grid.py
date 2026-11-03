import hydra
from omegaconf import DictConfig
from pathlib import Path
import math
import folium
import geopandas as gpd
from shapely.geometry import MultiPolygon, box


@hydra.main(version_base=None, config_path="config", config_name="config")
def show_grid(cfg: DictConfig) -> None:

    utm = gpd.GeoSeries(gpd.points_from_xy([cfg.region.longitude], [cfg.region.latitude]), crs="EPSG:4326").estimate_utm_crs()
    cx, cy = gpd.GeoSeries(gpd.points_from_xy([cfg.region.longitude], [cfg.region.latitude]), crs="EPSG:4326").to_crs(utm).iloc[0].coords[0]
    n = round(cfg.region.size_m / cfg.region.cell)
    ix0, iy0 = math.floor(cx / cfg.region.cell - n / 2), math.floor(cy / cfg.region.cell - n / 2)

    outline = box(ix0 * cfg.region.cell, iy0 * cfg.region.cell, (ix0 + n) * cfg.region.cell, (iy0 + n) * cfg.region.cell)
    black = MultiPolygon(
        [box((ix0 + i) * cfg.region.cell, (iy0 + j) * cfg.region.cell, (ix0 + i + 1) * cfg.region.cell, (iy0 + j + 1) * cfg.region.cell)
        for i in range(n) for j in range(n) if (ix0 + i + iy0 + j) % 2 == 0]
    )
    outline, black = (gpd.GeoDataFrame(geometry=[g], crs=utm).to_crs("EPSG:4326") for g in (outline, black))

    x0, y0, x1, y1 = outline.total_bounds
    m = folium.Map(tiles="Esri.WorldImagery", prefer_canvas=True)
    m.fit_bounds([[y0, x0], [y1, x1]])
    folium.GeoJson(black, style_function=lambda _: {"stroke": False, "fillColor": "white", "fillOpacity": 0.2}).add_to(m)
    folium.GeoJson(
        outline,
        style_function=lambda _: {"fill": False, "color": "#ff4b4b", "weight": 3, "dashArray": "6, 6"},
        tooltip=f"{cfg.region.size_m}x{cfg.region.size_m} m2 patch: {n}x{n} cells {cfg.region.cell}x{cfg.region.cell} m2"
    ).add_to(m)

    output_dir = Path(cfg.path_to_maps)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "grid_map.html"
    m.save(str(output_path))

    print(f"Grid generated: {output_path}.")


if __name__ =="__main__":
    show_grid()
