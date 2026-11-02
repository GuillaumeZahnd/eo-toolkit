import geopandas as gpd
from shapely.geometry import box


def build_spatiotemporal_index(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    years: list[int],
    weeks: list[int],
    resolution: int = 30,
    crs: str = "EPSG:32631",
) -> gpd.GeoDataFrame:
    """Generate a deterministic GeoDataFrame index mapping patch IDs to spatial bounding boxes and temporal bins.

    Parameters
    ----------
    xmin : float
        Minimum X coordinate (e.g., Easting in meters) of the bounding box.
    ymin : float
        Minimum Y coordinate (e.g., Northing in meters) of the bounding box.
    xmax : float
        Maximum X coordinate of the bounding box.
    ymax : float
        Maximum Y coordinate of the bounding box.
    years : list[int]
        List of years to include in the index (e.g., [2023, 2024]).
    weeks : list[int]
        List of ISO week numbers to include (e.g., list(range(1, 53))).
    resolution : int, optional
        Spatial resolution of each patch in CRS units (default is 30).
    crs : str, optional
        Coordinate reference system string (default is "EPSG:32631").

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing columns: patch_index, column, row, year, week, and geometry.
    """
    patches = []

    columns = int((xmax - xmin) / resolution)
    rows = int((ymax - ymin) / resolution)

    for i in range(columns):
        for j in range(rows):
            p_xmin = xmin + i * resolution
            p_ymin = ymin + j * resolution
            p_xmax = p_xmin + resolution
            p_ymax = p_ymin + resolution

            geometry = box(p_xmin, p_ymin, p_xmax, p_ymax)

            # Example: p_10_15_2024_W31 refers to
            # column index 10, row index 15, for the year 2024, during ISO week 31.

            for year in years:
                for week in weeks:
                    patch_index = f"p_{i}_{j}_{year}_W{week:02d}"
                    patches.append({
                        "patch_index": patch_index,
                        "column": i,
                        "row": j,
                        "year": year,
                        "week": week,
                        "geometry": geometry,
                    })

    gdf = gpd.GeoDataFrame(patches, crs=crs)

    return gdf.astype({
        "patch_index": "string",
        "column": "int32",
        "row": "int32",
        "year": "int32",
        "week": "int32",
    })
