#!/usr/bin/env python3
"""Smoke test for src/data/terrain.py against the live USGS 3DEP service.

Uses a small synthetic polygon near Snoqualmie Pass (~15 × 12 km, ~180 km²)
to keep the DEM download fast.  No zones.geojson needed.

Run from the project root with:
    .venv/bin/python scripts/test_terrain.py
"""

import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.constants import ZONE_NAME_COL
from src.data.terrain import (
    compute_aspect,
    compute_curvature,
    compute_slope,
    compute_terrain_features,
    compute_tri,
    download_dem,
    extract_zone_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_terrain")

# Snoqualmie Pass area — good 3DEP coverage, compact, steep Cascade terrain
# ~15 km E–W × 12 km N–S ≈ 180 km² → ~1800×1200 px at 10m → fast download
TEST_BBOX = box(-121.42, 47.38, -121.26, 47.49)
TEST_ZONE_NAME = "snoqualmie_pass_test"
OUTPUT_DIR = Path("data/raw/terrain")


def main() -> None:
    # ------------------------------------------------------------------
    # Step 0: build one-zone GeoDataFrame
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 0: Build test zone GeoDataFrame")
    print("=" * 60)

    zones_gdf = gpd.GeoDataFrame(
        {ZONE_NAME_COL: [TEST_ZONE_NAME]},
        geometry=[TEST_BBOX],
        crs="EPSG:4326",
    )
    zone_area_km2 = zones_gdf.to_crs("EPSG:32610").geometry.area.iloc[0] / 1e6
    print(f"Zone:   {TEST_ZONE_NAME}")
    print(f"Bounds: {TEST_BBOX.bounds}")
    print(f"Area:   {zone_area_km2:.1f} km²")

    # ------------------------------------------------------------------
    # Step 1: DEM download (with cache behaviour check)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Download DEM (expect ~30s first run, instant on re-run)")
    print("=" * 60)

    raster_dir = OUTPUT_DIR / "rasters"
    dem_path = download_dem(
        TEST_BBOX, TEST_ZONE_NAME, resolution=10, output_dir=raster_dir
    )
    print(f"DEM path: {dem_path}")
    print(f"File size: {dem_path.stat().st_size / 1e6:.1f} MB")

    import rasterio
    with rasterio.open(dem_path) as src:
        print(f"CRS: {src.crs}")
        print(f"Shape: {src.height} × {src.width} px")
        print(f"Cell size: {src.transform.a:.2f} m")
        print(f"Nodata: {src.nodata}")

    # ------------------------------------------------------------------
    # Step 2: Compute derivative rasters
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Compute slope, aspect, curvature, TRI")
    print("=" * 60)

    slope_path  = compute_slope(dem_path,     raster_dir)
    aspect_path = compute_aspect(dem_path,    raster_dir)
    curv_path   = compute_curvature(dem_path, raster_dir)
    tri_path    = compute_tri(dem_path,       raster_dir)

    import numpy as np
    for label, path in [("slope", slope_path), ("aspect", aspect_path),
                        ("curvature", curv_path), ("tri", tri_path)]:
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float32")
        valid = arr[~np.isnan(arr)]
        print(f"\n{label}:")
        print(f"  shape={arr.shape}  valid_px={len(valid):,}")
        print(f"  min={valid.min():.3f}  mean={valid.mean():.3f}  "
              f"max={valid.max():.3f}  p90={np.percentile(valid, 90):.3f}")

    # ------------------------------------------------------------------
    # Step 3: extract_zone_stats directly (inspect schema)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: extract_zone_stats — schema and sample values")
    print("=" * 60)

    terrain_paths = {
        "elevation": dem_path,
        "slope":     slope_path,
        "aspect":    aspect_path,
        "curvature": curv_path,
        "tri":       tri_path,
    }
    stats_df = extract_zone_stats(terrain_paths, zones_gdf)

    print(f"\nshape: {stats_df.shape}")
    print(f"\nColumns ({len(stats_df.columns)}):")
    for col in sorted(stats_df.columns):
        print(f"  {col}")
    print(f"\nDtypes:\n{stats_df.dtypes.to_string()}")
    print(f"\nAll values (transposed for readability):")
    print(stats_df.T.to_string())

    # ------------------------------------------------------------------
    # Step 4: full pipeline via compute_terrain_features
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: compute_terrain_features (should use all cached rasters)")
    print("=" * 60)

    result_df = compute_terrain_features(zones_gdf, OUTPUT_DIR, resolution=10)
    print(f"\nResult shape: {result_df.shape}")

    parquet_path = OUTPUT_DIR / "terrain_features.parquet"
    loaded = pd.read_parquet(parquet_path)
    print(f"Parquet round-trip shape: {loaded.shape} ✓")
    print(f"Index name: '{loaded.index.name}'")
    assert loaded.index.name == ZONE_NAME_COL, \
        f"Expected index '{ZONE_NAME_COL}', got '{loaded.index.name}'"
    assert TEST_ZONE_NAME in loaded.index, \
        f"Zone '{TEST_ZONE_NAME}' not found in parquet index"

    print(f"\nAll assertions passed.")
    print(f"\nAll done.")


if __name__ == "__main__":
    main()
