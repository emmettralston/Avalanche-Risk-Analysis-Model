"""MODIS snow cover ingestion and zone-level aggregation.

Downloads MODIS MOD10A1 (Terra) and MYD10A1 (Aqua) daily snow cover products
at 500m resolution via the NASA Earthdata API (earthaccess library).  Requires
a free NASA Earthdata account; credentials are read from ~/.netrc or environment
variables EARTHDATA_USERNAME / EARTHDATA_PASSWORD.

The primary output is a daily snow cover fraction per NWAC zone, saved as
data/processed/modis_snow_cover.parquet.

Known challenge: PNW cloud cover is frequent and correlated with weather
patterns.  Missing days are NOT dropped; instead explicit missing-data flags
are added as additional features so the model can learn cloud/weather
associations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

MODIS_PRODUCTS = ("MOD10A1", "MYD10A1")

# MODIS NDSI snow cover layer name and valid range
SNOW_COVER_LAYER = "NDSI_Snow_Cover"
SNOW_COVER_VALID_MIN = 0
SNOW_COVER_VALID_MAX = 100
CLOUD_VALUE = 50  # MODIS fill value indicating cloud-obscured pixel


def search_modis(
    zone_gdf: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    product: str = "MOD10A1",
    version: str = "061",
) -> list:
    """Search NASA Earthdata for MODIS snow cover granules covering NWAC zones.

    Uses the earthaccess library to query the CMR API.  Returns granule metadata
    objects that can be passed to ``download_modis``.

    Parameters
    ----------
    zone_gdf:
        GeoDataFrame of NWAC zone polygons (EPSG:4326).  The bounding box of
        all zones is used as the spatial filter.
    start_date:
        Inclusive start date as "YYYY-MM-DD".
    end_date:
        Inclusive end date as "YYYY-MM-DD".
    product:
        MODIS short name, either "MOD10A1" (Terra) or "MYD10A1" (Aqua).
    version:
        Collection version string.  Use "061" for Collection 6.1.

    Returns
    -------
    list
        List of earthaccess granule objects.
    """
    raise NotImplementedError


def download_modis(
    granules: list,
    output_dir: Path,
) -> list[Path]:
    """Download MODIS granules to local storage.

    Parameters
    ----------
    granules:
        Granule objects returned by ``search_modis``.
    output_dir:
        Directory where HDF files will be written.

    Returns
    -------
    list[Path]
        Paths to downloaded HDF files.
    """
    raise NotImplementedError


def load_modis_granule(hdf_path: Path) -> xr.Dataset:
    """Open a single MODIS MOD10A1/MYD10A1 HDF file as an xarray Dataset.

    Extracts the NDSI_Snow_Cover layer and associated QA flags.  Applies scale
    factors and masks fill values.

    Parameters
    ----------
    hdf_path:
        Path to the MODIS HDF4 file.

    Returns
    -------
    xr.Dataset
        Variables: ``snow_cover`` (0–100 NDSI), ``qa``, ``cloud_mask``
        (boolean, True where cloudy).  Coordinates: x, y (sinusoidal
        projection), time (single timestamp).
    """
    raise NotImplementedError


def compute_snow_cover_fraction(
    granule_ds: xr.Dataset,
    zone_gdf: gpd.GeoDataFrame,
    cloud_strategy: str = "flag",
) -> pd.DataFrame:
    """Compute per-zone snow cover fraction from a single MODIS granule.

    For each zone, counts valid (non-cloud) pixels with NDSI >= 40 (considered
    snow-covered) and divides by total valid pixels.  Cloud pixels are handled
    according to ``cloud_strategy``.

    Parameters
    ----------
    granule_ds:
        xarray Dataset from ``load_modis_granule``.
    zone_gdf:
        GeoDataFrame of NWAC zone polygons.
    cloud_strategy:
        How to handle cloud-obscured pixels:
        - "flag": set snow_cover_fraction to NaN and set cloud_fraction column.
        - "exclude": exclude cloudy pixels from denominator (can over-estimate).

    Returns
    -------
    pd.DataFrame
        Columns: date, zone_name, snow_cover_fraction, cloud_fraction,
        valid_pixel_count.
    """
    raise NotImplementedError


def handle_cloud_cover(
    df: pd.DataFrame,
    strategy: str = "forward_fill",
    max_fill_days: int = 3,
) -> pd.DataFrame:
    """Impute or flag NaN snow cover values caused by cloud cover.

    Parameters
    ----------
    df:
        DataFrame with columns: date, zone_name, snow_cover_fraction,
        cloud_fraction.  NaN in snow_cover_fraction indicates a cloudy day.
    strategy:
        Imputation strategy:
        - "forward_fill": carry forward last known valid value (up to
          ``max_fill_days`` consecutive days).
        - "interpolate": linear interpolation between bracketing valid values.
        - "flag_only": leave NaNs but ensure ``cloud_fraction`` column is set.
    max_fill_days:
        Maximum number of consecutive missing days to impute.  Days beyond
        this threshold remain NaN.

    Returns
    -------
    pd.DataFrame
        Same structure with NaNs filled according to ``strategy``.  Adds a
        boolean column ``snow_cover_imputed`` indicating imputed rows.
    """
    raise NotImplementedError


def aggregate_modis_to_zones(
    granule_paths: list[Path],
    zones_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    cloud_strategy: str = "forward_fill",
) -> pd.DataFrame:
    """Full pipeline: load granules, compute zone fractions, handle cloud cover.

    Processes all granules in ``granule_paths`` (typically covering the full
    2015–2021 date range for both Terra and Aqua), merges Terra and Aqua
    observations for each day (preferring Terra), applies cloud handling, and
    saves the result to ``output_dir / "modis_snow_cover.parquet"``.

    Parameters
    ----------
    granule_paths:
        Paths to downloaded MODIS HDF files.
    zones_gdf:
        GeoDataFrame of NWAC zone polygons.
    output_dir:
        Directory where ``modis_snow_cover.parquet`` will be written.
    cloud_strategy:
        Passed to ``handle_cloud_cover``.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame indexed by (date, zone_name) with snow cover features.
    """
    raise NotImplementedError
