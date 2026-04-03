"""Aligning meteorological, terrain, and satellite modalities to a common index.

This module is the sole place where data from different sources is joined.
Each modality (SNOTEL weather, terrain stats, MODIS satellite) is ingested
independently in src/data/.  fusion.py aligns them to a shared (date, zone)
spatiotemporal index and assembles the design matrix (X) and target vector (y)
ready for model training.

Key design principles:
- The join index is (date, zone_name), matching the label schema.
- Terrain features are static — broadcast across all dates for each zone.
- Missing satellite data is handled upstream in satellite.py; this module
  expects explicit NaN values or imputation flags.
- Season-aware train/test splits are handled here to prevent data leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

# Held-out test seasons (most recent two, per design decision)
DEFAULT_TEST_SEASONS = ["2019-20", "2020-21"]


def align_snotel_to_zones(
    snotel_df: pd.DataFrame,
    zones_gdf: gpd.GeoDataFrame,
    station_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Spatially join SNOTEL stations to zones and aggregate to zone-day level.

    Each station is assigned to the zone whose boundary it falls within (or, if
    outside all zones, to the nearest zone by centroid distance).  Multiple
    stations per zone are aggregated with the mean for continuous variables.

    Parameters
    ----------
    snotel_df:
        Tidy DataFrame with columns: date, station_id, plus weather variables.
    zones_gdf:
        GeoDataFrame of NWAC zone polygons with a ``zone_name`` column.
    station_gdf:
        GeoDataFrame of SNOTEL station locations with a ``station_id`` column
        and point geometry.

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, zone_name).  Columns: mean of each weather variable
        across stations in the zone, plus ``station_count`` (number of stations
        contributing to the zone average).
    """
    raise NotImplementedError


def merge_terrain_features(
    weather_df: pd.DataFrame,
    terrain_df: pd.DataFrame,
) -> pd.DataFrame:
    """Broadcast static terrain features onto the weather zone-day DataFrame.

    Terrain features do not vary by date, so they are joined on ``zone_name``
    alone and replicated across all dates for that zone.

    Parameters
    ----------
    weather_df:
        Zone-day DataFrame with a (date, zone_name) MultiIndex.
    terrain_df:
        Zone-level terrain feature table (index: zone_name).

    Returns
    -------
    pd.DataFrame
        ``weather_df`` with terrain feature columns appended.  All rows for a
        zone share identical terrain values.
    """
    raise NotImplementedError


def merge_satellite_features(
    weather_terrain_df: pd.DataFrame,
    satellite_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join daily MODIS snow cover features onto the weather+terrain DataFrame.

    Parameters
    ----------
    weather_terrain_df:
        Zone-day DataFrame with (date, zone_name) MultiIndex and weather +
        terrain columns.
    satellite_df:
        Zone-day DataFrame with (date, zone_name) index containing
        snow_cover_fraction, cloud_fraction, and snow_cover_imputed columns.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.  Rows present in ``weather_terrain_df`` but missing
        from ``satellite_df`` receive NaN satellite columns (indicating no data
        was available for that day/zone).
    """
    raise NotImplementedError


def build_feature_matrix(
    labels_df: pd.DataFrame,
    snotel_df: pd.DataFrame,
    terrain_df: pd.DataFrame,
    zones_gdf: gpd.GeoDataFrame,
    station_gdf: gpd.GeoDataFrame,
    satellite_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble the full feature matrix and target vector.

    Orchestrates the full fusion pipeline:
    1. Align SNOTEL to zones.
    2. Merge terrain features.
    3. Optionally merge satellite features.
    4. Inner-join with labels on (date, zone_name).

    Only rows present in ``labels_df`` are kept — this ensures the feature
    matrix is aligned to labelled zone-days.

    Parameters
    ----------
    labels_df:
        DataFrame with columns: date, zone_name, danger_rating, season.
    snotel_df:
        Tidy SNOTEL DataFrame as returned by ``src.data.snotel.load_snotel``.
    terrain_df:
        Zone-level terrain feature table.
    zones_gdf:
        GeoDataFrame of NWAC zone polygons.
    station_gdf:
        GeoDataFrame of SNOTEL station locations.
    satellite_df:
        Optional zone-day MODIS snow cover DataFrame.  If None, satellite
        features are omitted from the matrix.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X: feature DataFrame indexed by (date, zone_name).
        y: Series of danger_rating (int8), same index as X.
    """
    raise NotImplementedError


def get_season_splits(
    labels_df: pd.DataFrame,
    test_seasons: list[str] = DEFAULT_TEST_SEASONS,
) -> tuple[pd.Index, pd.Index]:
    """Return row indices for season-aware train/test split.

    Training set: all rows whose ``season`` is NOT in ``test_seasons``.
    Test set: all rows whose ``season`` IS in ``test_seasons``.

    Never split by random row selection — avalanche conditions within a season
    are highly autocorrelated and random splits leak temporal information.

    Parameters
    ----------
    labels_df:
        DataFrame with a ``season`` column (as added by
        ``src.data.labels.add_season_column``).
    test_seasons:
        Seasons to hold out as the test set.  Defaults to the two most recent
        seasons (2019-20, 2020-21) per the project design decision.

    Returns
    -------
    tuple[pd.Index, pd.Index]
        (train_index, test_index) — integer positional indices suitable for
        use with ``iloc``.
    """
    raise NotImplementedError
