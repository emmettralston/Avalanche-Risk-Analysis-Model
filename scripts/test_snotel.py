#!/usr/bin/env python3
"""Smoke test for src/data/snotel.py against the live AWDB API.

Uses a synthetic bounding box covering the WA/OR Cascades (no zones.geojson
needed) to exercise the full ingestion pipeline:

  get_pnw_stations → fetch_all_stations (3 stations, 1 season) → clean_station_data

Run from the project root with:
    .venv/bin/python scripts/test_snotel.py
"""

import logging
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.snotel import clean_station_data, fetch_all_stations, get_pnw_stations

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_snotel")

# Bounding box covering core WA + northern OR Cascades (NWAC region)
NWAC_BBOX = box(-122.5, 45.5, -119.0, 49.1)

TEST_START = "2018-10-01"
TEST_END = "2019-04-30"
N_TEST_STATIONS = 3
OUTPUT_DIR = Path("data/raw/snotel")


def main() -> None:
    # ------------------------------------------------------------------
    # Step 1: station discovery
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Fetching PNW SNOTEL stations near NWAC bounding box")
    print("=" * 60)

    zones_gdf = gpd.GeoDataFrame(
        {"zone_name": ["nwac_bbox"]},
        geometry=[NWAC_BBOX],
        crs="EPSG:4326",
    )

    stations_gdf = get_pnw_stations(zones_gdf, buffer_km=20.0)

    print(f"\nTotal stations found: {len(stations_gdf)}")
    print("\nFirst 10 stations:")
    preview_cols = ["station_id", "station_name", "latitude", "longitude", "elevation_ft", "state"]
    print(stations_gdf[preview_cols].head(10).to_string(index=False))

    if len(stations_gdf) < N_TEST_STATIONS:
        logger.error("Fewer than %d stations found — aborting", N_TEST_STATIONS)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: fetch raw data for first N stations, one season
    # ------------------------------------------------------------------
    test_ids = stations_gdf["station_id"].tolist()[:N_TEST_STATIONS]

    print("\n" + "=" * 60)
    print(f"STEP 2: Fetching {TEST_START} → {TEST_END} for {N_TEST_STATIONS} stations")
    print(f"  Stations: {test_ids}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = fetch_all_stations(
        station_ids=test_ids,
        start_date=TEST_START,
        end_date=TEST_END,
        output_dir=OUTPUT_DIR,
    )

    print(f"\nRaw DataFrame shape: {raw_df.shape}")
    print(f"\nRaw dtypes:\n{raw_df.dtypes.to_string()}")
    print(f"\nRaw NaN counts per column:")
    nan_counts = raw_df.isna().sum()
    for col, n in nan_counts.items():
        pct = 100 * n / len(raw_df)
        print(f"  {col:<35} {n:>5} ({pct:5.1f}%)")

    print(f"\nSample rows (first station):")
    sample = raw_df[raw_df["station_id"] == test_ids[0]].head(5)
    print(sample.to_string())

    # ------------------------------------------------------------------
    # Step 3: clean
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Cleaning data")
    print("=" * 60)

    clean_df = clean_station_data(raw_df)

    weather_cols = [c for c in clean_df.columns
                    if not c.endswith("_missing") and c != "station_id"]
    missing_flag_cols = [c for c in clean_df.columns if c.endswith("_missing")]

    print(f"\nCleaned DataFrame shape: {clean_df.shape}")
    print(f"\nCleaned dtypes:\n{clean_df.dtypes.to_string()}")

    print(f"\nResidual NaN counts after cleaning (values still missing after gap-fill):")
    for col in weather_cols:
        n = int(clean_df[col].isna().sum())
        pct = 100 * n / len(clean_df)
        print(f"  {col:<35} {n:>5} ({pct:5.1f}%)")

    print(f"\n_missing flag totals (should match residual NaN counts above):")
    for col in missing_flag_cols:
        n = int(clean_df[col].sum())
        print(f"  {col:<43} {n:>5}")

    print(f"\nDate range: {clean_df.index.min().date()} → {clean_df.index.max().date()}")
    print(f"Stations present: {sorted(clean_df['station_id'].unique())}")

    # ------------------------------------------------------------------
    # Step 4: save clean file
    # ------------------------------------------------------------------
    clean_path = OUTPUT_DIR / "snotel_clean.parquet"
    clean_df.to_parquet(clean_path)
    print(f"\nSaved cleaned data → {clean_path}")
    print(f"\nAll done.")


if __name__ == "__main__":
    main()
