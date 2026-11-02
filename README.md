# Earth-observation toolkit

## Stack

#### Geospatial & remote sensing

- `odc-stac`: Loading STAC catalogs into Open Data Cube / xarray datasets.
- `rasterio`: Low-level reading and writing of geospatial raster data via GDAL.
- `rioxarray`: Geospatial extension for xarray powered by rasterio.
- `pystac-client`: Searching and accessing SpatioTemporal Asset Catalogs.
- `planetary-computer`: Utilities for accessing Microsoft Planetary Computer data and APIs.
- `earthengine-api`: Client library for accessing Google Earth Engine.
- `pyproj`: Cartographic projections and coordinate transformations.
- `geemap` & `ipyleaflet`: Interactive mapping tools and visualization widgets for notebooks.
- `folium`: Visualize data on an interactive leaflet map. 
- `geopandas`: Open source project to facilitate working with geospatial data.
- `geoparquet`: Facilitate fast input/output of GIS data in the open source Parquet file format.

#### Machine learning & deep learning

- `torch`: Core deep learning and tensor computation framework (driving AnySat).
- `lightning`: PyTorch Lightning for structured, scalable model training loops.

#### MLOps & configuration

- `hydra-core`: Hierarchical configuration management for pipeline execution and overrides.
- `mlflow`: Experiment tracking, model registry, and metrics logging.

#### Core data science & basic python-esque utilities

- `numpy`: Fundamental multi-dimensional array and numerical computing operations.
- `pandas`: Tabular data structures and feature dataset handling (e.g., EFDA logs).
- `matplotlib`: Plotting and visualization.
- `ipywidgets`: Interactive HTML control elements for Jupyter notebooks.

### Acronyms

- **CRS:** Coordinate Reference System
- **GIS:** Geographic Information System
- **NDMI:** Normalized Difference Moisture Index, describes the crop’s water stress level (`NDMI := NIR-SWIR)/(NIR+SWIR)`)
- **NDRE:** Normalized Difference Red Edge Index, describes the amount of chlorophyll in a plant (`NDRE = (NIR-RE)/(NIR+RE)`)
- **NDVI:** Normalized Difference Vegetation Index, describes the vigour level of the crop (`NDVI := (NIR-Red)/(NIR+Red)`)

