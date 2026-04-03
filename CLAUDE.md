# CLAUDE.md — PNW Avalanche Danger Prediction

## Project Overview

This project builds a machine learning system to predict backcountry avalanche
danger in the Pacific Northwest, focused on NWAC (Northwest Avalanche Center)
forecast zones. It is a personal research/portfolio project developed
iteratively, starting with zone-level danger classification and progressively
moving toward slope-level spatial prediction.

The core research contribution over prior work (Open Avalanche Project) is the
fusion of three data modalities:
- Meteorological/snowpack time series (SNOTEL stations)
- Static terrain features derived from DEM (slope, aspect, elevation, curvature)
- Satellite snow cover imagery (MODIS, eventually Sentinel-2)

## Goals

1. Reproduce and extend Open Avalanche Project baseline accuracy using NWAC
   zone-level labels (2015–2021)
2. Demonstrate lift from adding terrain features over weather-only baseline
3. Integrate MODIS satellite snow cover as a third modality
4. Push toward grid-cell (slope-level) spatial prediction as the research
   novelty — this is the primary differentiator from existing work
5. Build clean, modular, reproducible code suitable as a portfolio project

## Project Structure
```
avalanche-danger/
├── data/
│   ├── raw/           # Downloaded source data — never modify
│   ├── processed/     # Cleaned, aligned, feature-engineered outputs
│   └── labels/        # OAP label files + NWAC forecast archives
├── notebooks/
│   ├── 01_snotel_exploration.ipynb
│   ├── 02_terrain_features.ipynb
│   ├── 03_modis_snow_cover.ipynb
│   ├── 04_feature_fusion.ipynb
│   └── 05_baseline_model.ipynb
├── src/
│   ├── data/
│   │   ├── snotel.py       # SNOTEL ingestion and cleaning
│   │   ├── terrain.py      # DEM download and feature extraction
│   │   ├── satellite.py    # MODIS ingestion and aggregation
│   │   └── labels.py       # Label loading and zone alignment
│   ├── features/
│   │   └── fusion.py       # Aligning modalities to common spatiotemporal index
│   └── models/
│       └── baseline.py     # Training, evaluation, cross-validation
├── configs/
│   └── zones.geojson       # NWAC zone boundaries (sourced from OAP)
├── CLAUDE.md
├── requirements.txt
└── README.md
```

## Data Sources

### Weather & Snowpack
- **SNOTEL** (USDA NRCS): snow depth, SWE, temperature, precipitation
  - Free, no auth, REST API
  - Target stations: PNW stations within or adjacent to NWAC zones
  - Date range: 2015-10-01 through 2021-04-30 (matching label window first)
  - API base: https://wcc.sc.egov.usda.gov/awdbRestApi/
- **NOAA NCEI**: supplemental gridded climate data if needed

### Training Labels
- **Open Avalanche Project labels**: pre-cleaned NWAC danger ratings 2015–2021
  - Download from: https://github.com/scottcha/OpenAvalancheProject
  - File: `Data/CleanedForecastsNWAC_CAIC_UAC_CAC.V1.2013-2021.zip`
  - Use NWAC rows only for this project
  - Schema: date, zone, danger_rating (1–5), avalanche_problems
- **OAP GeoJSON zone boundaries**: `configs/zones.geojson`
  - Source: OpenAvalancheProject repo Data/ folder

### Terrain (Static, computed once)
- **USGS 3DEP**: 1m or 10m DEM for PNW
  - Access via `py3dep` Python library
  - Derived features: slope angle (degrees), aspect (degrees + cardinal bins),
    curvature, elevation, terrain ruggedness index (TRI)
  - Store as GeoTIFF per zone, then extract summary stats per zone or grid cell
- Terrain features do not change — compute once, cache, reuse

### Satellite Snow Cover
- **MODIS MOD10A1 / MYD10A1**: daily snow cover, 500m resolution, since 2000
  - Access via NASA Earthdata — requires free account
  - Python library: `earthaccess`
  - Known issue: PNW cloud cover causes frequent missing data days — handle
    with forward-fill, interpolation, or explicit missing flags
  - Aggregate to zone-level snow cover fraction per day
- **Sentinel-2**: 10m resolution — Phase 2 only, defer for now
- **Sentinel-1 SAR**: cloud-penetrating — defer, significant processing overhead

## Tech Stack
```
python = "3.11+"
pandas          # tabular data manipulation
numpy           # numerical ops
xarray          # gridded/raster data (MODIS, GFS)
geopandas       # spatial data, zone boundaries
rasterio        # GeoTIFF read/write
shapely         # geometry operations
py3dep          # USGS 3DEP DEM access
earthaccess     # NASA Earthdata access
scikit-learn    # preprocessing, cross-validation, metrics
xgboost         # baseline classifier (Phase 1)
lightgbm        # alternative baseline
matplotlib      # plotting
folium          # interactive maps for spatial QA
pyarrow         # parquet read/write for processed data
```

Graduate to PyTorch only if Phase 3 spatial CNN architecture is warranted.
No database needed — use parquet for tabular, GeoTIFF for raster.

## Development Phases

### Phase 1 — Weather + Terrain Baseline (current focus)
- [x] Project structure and stubs initialized
- [x] SNOTEL ingestion script for PNW stations, 2015–2021
  (71 PNW stations identified; test pull confirmed clean for 2018-19 season; full pull pending)
- [ ] Load and inspect OAP label dataset, understand schema
- [ ] Spatial join: map SNOTEL stations to NWAC zones via GeoJSON
- [ ] Compute terrain features from 3DEP DEM per zone
- [ ] Feature fusion: align weather time series + static terrain per zone-day
- [ ] Train XGBoost baseline, evaluate with stratified k-fold by season
- [ ] Benchmark: match or exceed OAP reported accuracy

### Phase 2 — Satellite Integration
- [ ] MODIS snow cover ingestion for NWAC zone bounding boxes
- [ ] Missing data handling strategy (cloud cover)
- [ ] Aggregate to zone-level daily snow cover fraction
- [ ] Add to feature matrix, retrain, evaluate lift

### Phase 3 — Spatial Resolution (research contribution)
- [ ] Define 500m or 1km prediction grid over NWAC zones
- [ ] Extract per-cell terrain features from DEM
- [ ] Spatially interpolate SNOTEL weather to grid (kriging or IDW)
- [ ] Train per-cell model, generate danger maps
- [ ] Validate against known high-hazard terrain features

## Key Design Decisions

**Zone-first, then grid**: Start with zone-level prediction to validate the
pipeline and labels before attempting spatial disaggregation. Don't jump to
Phase 3 before Phase 1 is solid.

**Season-aware cross-validation**: Always split train/test by full seasons,
never by random rows. Avalanche conditions within a season are highly
autocorrelated. Held-out test set = most recent 2 seasons (2019-20, 2020-21).

**Class imbalance**: Danger levels 4 and 5 are rare. Use stratified sampling,
class weights, and report per-class F1 — not just overall accuracy. OAP and
human forecasters both struggle at high danger levels.

**Missing satellite data**: Cloud cover in the Cascades is frequent and not
random — it correlates with weather patterns. Do not simply drop cloudy days.
Use explicit missing flags as features or temporal interpolation.

**Modality independence**: Keep each data source in its own ingestion module
(`snotel.py`, `terrain.py`, `satellite.py`). The fusion step happens in
`fusion.py` only. This makes it easy to add/remove modalities and debug
independently.

## Known Challenges

- SNOTEL station coverage is sparse at high elevations — where avalanche danger
  is highest. Spatial interpolation will introduce uncertainty; quantify it.
- OAP labels end at 2020-21. Extending to present requires scraping NWAC
  forecast archives directly — defer until baseline is working.
- The target variable (danger rating) is a human judgment call with inherent
  subjectivity and inter-forecaster variability. Treat model as decision support,
  not ground truth.
- PNW coastal snowpack behaves differently from continental — models trained
  on Colorado (CAIC) data from OAP will not transfer directly.
- numpy is pinned at 2.0.x (breaking changes vs 1.x). If obscure type errors
  appear in rasterio, geopandas, or scipy, fall back to numpy==1.26.4.

## Coding Conventions

- All data loading functions return clean pandas DataFrames or xarray Datasets
  with explicit dtypes and datetime indices
- Raster data stored as GeoTIFF with CRS explicitly set (EPSG:4326 or UTM zone
  10N for PNW)
- No hardcoded file paths — use pathlib.Path and a config dict or constants file
- Each notebook should be independently runnable given processed data exists
- Write docstrings for all src/ functions; notebooks are exploratory and do not
  need them
- Use `logging` not `print` in src/ modules

## Immediate Next Steps

1. Initialize repo, set up virtual environment, install requirements
2. Write `src/data/snotel.py` — query SNOTEL API for PNW stations, pull
   2015–2021 weather variables, save to `data/raw/snotel/`
3. Download OAP label zip, inspect schema in `notebooks/01_snotel_exploration.ipynb`
4. Load NWAC zone GeoJSON, plot stations vs zone boundaries to verify alignment
5. Sanity check: can we predict danger rating from SNOTEL features alone at
   above-chance accuracy? This sets the baseline before adding terrain/satellite.