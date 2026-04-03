"""DEM download and terrain feature extraction for NWAC forecast zones.

Uses the py3dep library to access USGS 3DEP elevation data at 10m resolution.
Derived features — slope, aspect, curvature, elevation statistics, and terrain
ruggedness index (TRI) — are computed once per zone and cached as GeoTIFFs in
data/raw/terrain/rasters/.  Zone-level summary statistics are written to
data/raw/terrain/terrain_features.parquet.

CRS flow
--------
1. Input zone geometries arrive in EPSG:4326.
2. ``download_dem``: py3dep.get_dem returns a DataArray in EPSG:4326 (the
   ``crs`` parameter to get_dem is the *input* geometry CRS, not the output).
   The DataArray is immediately reprojected to UTM zone 10N (EPSG:32610) and
   written to GeoTIFF.  All subsequent rasters are in EPSG:32610.
3. ``compute_slope/aspect/curvature/tri``: read UTM GeoTIFFs, compute in
   projected coordinates where cell size is in metres.
4. ``extract_zone_stats``: reprojects zone polygons to EPSG:32610 for masking.

Multi-tile handling
-------------------
py3dep.static_3dep_dem reads from a seamless USGS S3 VRT that pre-mosaics all
3DEP tiles.  A single py3dep call covers any zone regardless of how many
underlying 3DEP tiles it spans — no manual stitching needed.

Terrain features are static (do not change over time) and should be joined
onto the zone-day feature matrix as a constant per zone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import py3dep
import rasterio
import rasterio.mask
import rioxarray  # registers .rio accessor on xarray DataArrays  # noqa: F401
import scipy.ndimage
from rasterio.enums import Resampling

from src.constants import ZONE_NAME_COL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DEFAULT_RESOLUTION = 10  # metres
OUTPUT_CRS = "EPSG:32610"  # UTM zone 10N — all raster outputs use this CRS
INPUT_CRS = "EPSG:4326"   # CRS of zone geometries supplied by caller

# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------

_STEEP_30_DEG = 30.0   # primary avalanche release slope threshold
_STEEP_45_DEG = 45.0   # near-cliff threshold

# Aspect quadrant boundaries (clockwise from north, degrees)
#   North: [315, 360) ∪ [0, 45)
#   East:  [45,  135)
#   South: [135, 225)
#   West:  [225, 315)
_ASP_NORTH_LO, _ASP_NORTH_HI = 315.0, 45.0   # wraps around 0
_ASP_EAST_LO,  _ASP_EAST_HI  =  45.0, 135.0
_ASP_SOUTH_LO, _ASP_SOUTH_HI = 135.0, 225.0
_ASP_WEST_LO,  _ASP_WEST_HI  = 225.0, 315.0

_RASTER_PROFILE_BASE: dict[str, Any] = {
    "driver": "GTiff",
    "dtype": "float32",
    "count": 1,
    "nodata": float("nan"),
    "compress": "deflate",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _write_geotiff(path: Path, array: np.ndarray, profile: dict) -> None:
    """Write a 2-D float32 array to a GeoTIFF, inheriting CRS and transform."""
    p = {**_RASTER_PROFILE_BASE, **{
        "crs": profile["crs"],
        "transform": profile["transform"],
        "width": array.shape[1],
        "height": array.shape[0],
    }}
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array.astype("float32"), 1)


def _read_dem_array(
    dem_path: Path,
) -> tuple[np.ndarray, Any, dict]:
    """Open a UTM 10N GeoTIFF and return (float32 array, transform, profile).

    Nodata pixels are set to NaN in the returned array.
    """
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata
        arr = src.read(1).astype("float32")
    if nodata is not None and not np.isnan(nodata):
        arr[arr == nodata] = np.nan
    return arr, transform, profile


def _compute_gradients(
    dem_path: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute Horn (1981) horizontal gradients from a UTM GeoTIFF DEM.

    Returns ``(dz_dx, dz_dy, cell_size_m)`` where ``dz_dx`` is the rate of
    elevation change in the east direction, ``dz_dy`` in the north direction,
    both in m/m.  ``cell_size_m`` is the DEM x-cell size in metres.

    Both gradient arrays have NaN at the 1-pixel border (convolution edge
    effect) and at any source nodata location.  The DEM must be in a projected
    CRS with metres as units (EPSG:32610).
    """
    arr, transform, _ = _read_dem_array(dem_path)
    cell_size = abs(float(transform.a))  # x-cell size in metres

    # Horn (1981) kernels — standard 3×3 finite-difference gradient
    # dz/dx: ((NE + 2E + SE) − (NW + 2W + SW)) / (8 × cs)
    kern_x = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], dtype="float32"
    ) / (8.0 * cell_size)

    # dz/dy: ((NW + 2N + NE) − (SW + 2S + SE)) / (8 × cs)
    # Row 0 = north in raster convention, so kernel rows go north→south.
    kern_y = np.array(
        [[ 1,  2,  1],
         [ 0,  0,  0],
         [-1, -2, -1]], dtype="float32"
    ) / (8.0 * cell_size)

    # Replace NaN with 0 before convolution; restore NaN mask afterward.
    arr_filled = np.nan_to_num(arr, nan=0.0)
    dz_dx = scipy.ndimage.convolve(arr_filled, kern_x, mode="nearest")
    dz_dy = scipy.ndimage.convolve(arr_filled, kern_y, mode="nearest")

    # Invalidate border pixels (Horn kernel uses 3×3 window) and source NaN
    invalid = np.zeros(arr.shape, dtype=bool)
    invalid[:1, :]  = True
    invalid[-1:, :] = True
    invalid[:, :1]  = True
    invalid[:, -1:] = True
    invalid |= np.isnan(arr)

    dz_dx[invalid] = np.nan
    dz_dy[invalid] = np.nan

    return dz_dx, dz_dy, cell_size


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def download_dem(
    zone_geom,
    zone_name: str,
    resolution: int = DEFAULT_RESOLUTION,
    output_dir: Path | None = None,
) -> Path:
    """Download a Digital Elevation Model for a single zone via py3dep.

    The returned GeoTIFF is in UTM zone 10N (EPSG:32610) with cell size
    approximately ``resolution`` metres.  py3dep fetches from a seamless USGS
    S3 VRT, so multi-tile zones are handled transparently.

    Parameters
    ----------
    zone_geom:
        Shapely geometry (polygon or multipolygon) in EPSG:4326.
    zone_name:
        Short identifier used in the output filename, e.g. "olympics".
    resolution:
        DEM resolution in metres.  3DEP supports 1, 3, 5, 10, 30 m.
    output_dir:
        Directory where the GeoTIFF will be saved.

    Returns
    -------
    Path
        Absolute path to the written GeoTIFF file.
    """
    if output_dir is None:
        raise ValueError("output_dir must be specified")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{zone_name}_dem_{resolution}m.tif"

    if out_path.exists():
        logger.info("DEM cache hit: %s", out_path.name)
        return out_path

    logger.info("Downloading %dm DEM for zone '%s'…", resolution, zone_name)

    # py3dep.get_dem: crs= is the CRS of the INPUT geometry; output is always
    # returned in EPSG:4326 regardless of this parameter.
    dem_4326 = py3dep.get_dem(zone_geom, resolution, crs=4326)

    # Reproject to UTM 10N so cell size is in metres — required by all
    # subsequent gradient/curvature/TRI computations.
    dem_utm = dem_4326.rio.reproject(OUTPUT_CRS, resampling=Resampling.bilinear)

    height, width = dem_utm.shape[-2], dem_utm.shape[-1]
    profile = {**_RASTER_PROFILE_BASE, **{
        "crs": OUTPUT_CRS,
        "transform": dem_utm.rio.transform(),
        "width": width,
        "height": height,
    }}
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dem_utm.values.squeeze().astype("float32"), 1)

    logger.info("DEM saved: %s  (%d × %d px)", out_path.name, height, width)
    return out_path


def compute_slope(dem_path: Path, output_dir: Path | None = None) -> Path:
    """Compute slope in degrees from a UTM GeoTIFF DEM.

    Uses the ``_compute_gradients`` helper (Horn 1981 kernel) shared with
    ``compute_aspect`` so the DEM is read and the 3×3 convolutions are run
    only once per call to either function.

    Parameters
    ----------
    dem_path:
        Path to input DEM GeoTIFF in a projected CRS with metres as units.
    output_dir:
        Directory for output file.  Writes alongside ``dem_path`` if None.

    Returns
    -------
    Path
        Path to slope GeoTIFF (float32, degrees, 0–90).
    """
    out_dir = Path(output_dir) if output_dir else dem_path.parent
    out_path = out_dir / f"{dem_path.stem}_slope.tif"

    if out_path.exists():
        logger.debug("Slope cache hit: %s", out_path.name)
        return out_path

    _, _, profile = _read_dem_array(dem_path)
    dz_dx, dz_dy, _ = _compute_gradients(dem_path)

    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype("float32")
    _write_geotiff(out_path, slope, profile)
    logger.debug("Slope saved: %s", out_path.name)
    return out_path


def compute_aspect(dem_path: Path, output_dir: Path | None = None) -> Path:
    """Compute aspect in degrees from a UTM GeoTIFF DEM.

    Uses the shared ``_compute_gradients`` helper (Horn 1981 kernel).
    Aspect is the downslope-facing direction, measured clockwise from north
    (0 = north-facing, 90 = east-facing, 180 = south-facing, 270 = west-facing).
    Flat pixels (gradient magnitude < 1e-6 m/m) are set to -1.

    Parameters
    ----------
    dem_path:
        Path to input DEM GeoTIFF.
    output_dir:
        Directory for output file.  Writes alongside ``dem_path`` if None.

    Returns
    -------
    Path
        Path to aspect GeoTIFF (float32, degrees 0–360, or -1 for flat).
    """
    out_dir = Path(output_dir) if output_dir else dem_path.parent
    out_path = out_dir / f"{dem_path.stem}_aspect.tif"

    if out_path.exists():
        logger.debug("Aspect cache hit: %s", out_path.name)
        return out_path

    _, _, profile = _read_dem_array(dem_path)
    dz_dx, dz_dy, _ = _compute_gradients(dem_path)

    # Downslope direction (clockwise from north):
    #   east component of downhill vector = -dz_dx
    #   north component of downhill vector = -dz_dy
    #   atan2(east, north) gives clockwise-from-north angle
    # Verification:
    #   slope up to north  (dz_dy>0, dz_dx≈0): atan2(0, -q) = 180° (south-facing) ✓
    #   slope up to east   (dz_dx>0, dz_dy≈0): atan2(-p, 0) = -90° → 270° (west-facing) ✓
    #   slope up to south  (dz_dy<0, dz_dx≈0): atan2(0, +|q|) = 0° (north-facing) ✓
    #   slope up to west   (dz_dx<0, dz_dy≈0): atan2(+|p|, 0) = 90° (east-facing) ✓
    aspect = (np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360.0).astype("float32")
    flat = np.hypot(dz_dx, dz_dy) < 1e-6
    aspect[flat] = -1.0

    _write_geotiff(out_path, aspect, profile)
    logger.debug("Aspect saved: %s", out_path.name)
    return out_path


def compute_curvature(dem_path: Path, output_dir: Path | None = None) -> Path:
    """Compute plan curvature from a UTM GeoTIFF DEM.

    Uses a discrete Laplacian (second-order central difference).  Units are
    1/m (rate of slope change per metre of horizontal distance).  Positive
    values indicate convex terrain (ridges); negative values indicate concave
    terrain (bowls and gullies) that tends to accumulate wind-deposited snow.

    Parameters
    ----------
    dem_path:
        Path to input DEM GeoTIFF.
    output_dir:
        Directory for output file.

    Returns
    -------
    Path
        Path to curvature GeoTIFF (float32, 1/m).
    """
    out_dir = Path(output_dir) if output_dir else dem_path.parent
    out_path = out_dir / f"{dem_path.stem}_curvature.tif"

    if out_path.exists():
        logger.debug("Curvature cache hit: %s", out_path.name)
        return out_path

    arr, transform, profile = _read_dem_array(dem_path)
    cell_size = abs(float(transform.a))

    # Discrete Laplacian: ∇²z ≈ (N + S + E + W − 4·center) / cs²
    kernel = np.array(
        [[0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]], dtype="float32"
    )
    arr_filled = np.nan_to_num(arr, nan=0.0)
    curv = (
        scipy.ndimage.convolve(arr_filled, kernel, mode="nearest")
        / (cell_size ** 2)
    ).astype("float32")

    # Mask border pixels and source nodata
    curv[:1, :]  = np.nan
    curv[-1:, :] = np.nan
    curv[:, :1]  = np.nan
    curv[:, -1:] = np.nan
    curv[np.isnan(arr)] = np.nan

    _write_geotiff(out_path, curv, profile)
    logger.debug("Curvature saved: %s", out_path.name)
    return out_path


def compute_tri(dem_path: Path, output_dir: Path | None = None) -> Path:
    """Compute Terrain Ruggedness Index (TRI) from a UTM GeoTIFF DEM.

    TRI = √(Σ(neighbor − center)²) for the 8 surrounding cells (Riley et al.
    1999).  Higher values indicate rough, wind-exposed terrain with variable
    snow redistribution potential.  Units: metres.

    Parameters
    ----------
    dem_path:
        Path to input DEM GeoTIFF.
    output_dir:
        Directory for output file.

    Returns
    -------
    Path
        Path to TRI GeoTIFF (float32, metres).
    """
    out_dir = Path(output_dir) if output_dir else dem_path.parent
    out_path = out_dir / f"{dem_path.stem}_tri.tif"

    if out_path.exists():
        logger.debug("TRI cache hit: %s", out_path.name)
        return out_path

    arr, _, profile = _read_dem_array(dem_path)
    rows, cols = arr.shape
    interior = arr[1:-1, 1:-1]

    sq_diff_sum = np.zeros_like(interior, dtype="float32")
    for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        r0 = 1 + dr; r1 = rows - 1 + dr
        c0 = 1 + dc; c1 = cols - 1 + dc
        neighbor = arr[r0:r1, c0:c1]
        # NOTE: NaN neighbors are treated as equal to the center cell (diff=0),
        # which underestimates TRI at zone edges where some neighbours are
        # outside the 3DEP data extent.  Revisit if edge zones show anomalously
        # low TRI values in model feature importance.
        diff = np.nan_to_num(interior - neighbor, nan=0.0)
        sq_diff_sum += diff.astype("float32") ** 2

    tri_interior = np.sqrt(sq_diff_sum)
    tri_interior[np.isnan(interior)] = np.nan  # restore source nodata

    tri = np.full(arr.shape, np.nan, dtype="float32")
    tri[1:-1, 1:-1] = tri_interior

    _write_geotiff(out_path, tri, profile)
    logger.debug("TRI saved: %s", out_path.name)
    return out_path


def extract_zone_stats(
    terrain_paths: dict[str, Path],
    zone_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Compute per-zone summary statistics from terrain rasters.

    Masks each raster to the zone polygon and computes descriptive statistics.
    Zone geometries are reprojected to EPSG:32610 to match the raster CRS.

    Parameters
    ----------
    terrain_paths:
        Mapping from variable name (``"elevation"``, ``"slope"``, ``"aspect"``,
        ``"curvature"``, ``"tri"``) to GeoTIFF path (EPSG:32610).
    zone_gdf:
        GeoDataFrame with one row per zone.  Must have a ``ZONE_NAME_COL``
        column and geometry in any CRS (reprojected internally).

    Returns
    -------
    pd.DataFrame
        Index: ``ZONE_NAME_COL``.

        Per-variable columns (for each var in terrain_paths):
        ``{var}_mean``, ``{var}_median``, ``{var}_std``,
        ``{var}_p10``, ``{var}_p25``, ``{var}_p75``, ``{var}_p90``,
        ``{var}_valid_pixel_frac``  ← fraction of zone bbox with valid data.

        Elevation extras: ``elevation_min``, ``elevation_max``.
        Slope extras: ``frac_steep_30deg``, ``frac_steep_45deg``.
        Aspect extras: ``aspect_north_frac``, ``aspect_east_frac``,
        ``aspect_south_frac``, ``aspect_west_frac``  (from valid non-flat pixels).
        ``pixel_count`` (int32): valid elevation pixels — overall coverage metric.
    """
    zones_utm = zone_gdf[[ZONE_NAME_COL, "geometry"]].to_crs(OUTPUT_CRS)

    records: list[dict] = []
    for _, row in zones_utm.iterrows():
        zone_name = row[ZONE_NAME_COL]
        zone_geom = row.geometry
        rec: dict = {ZONE_NAME_COL: zone_name}

        for var_name, raster_path in terrain_paths.items():
            with rasterio.open(raster_path) as src:
                try:
                    masked, _ = rasterio.mask.mask(
                        src, [zone_geom], crop=True,
                        nodata=float("nan"), filled=True,
                        all_touched=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "mask failed for zone '%s' var '%s': %s — skipping",
                        zone_name, var_name, exc,
                    )
                    continue

            arr = masked[0].astype("float64")   # shape (H, W); float64 for percentile accuracy
            total_px = arr.size
            valid_mask = ~np.isnan(arr)
            pixels = arr[valid_mask]
            n_valid = len(pixels)

            rec[f"{var_name}_valid_pixel_frac"] = (
                float(n_valid / total_px) if total_px > 0 else 0.0
            )

            if n_valid == 0:
                logger.warning(
                    "No valid pixels for zone '%s' var '%s'", zone_name, var_name
                )
                for stat in ("mean", "median", "std", "p10", "p25", "p75", "p90"):
                    rec[f"{var_name}_{stat}"] = float("nan")
                continue

            rec[f"{var_name}_mean"]   = float(np.mean(pixels))
            rec[f"{var_name}_median"] = float(np.median(pixels))
            rec[f"{var_name}_std"]    = float(np.std(pixels))
            rec[f"{var_name}_p10"]    = float(np.percentile(pixels, 10))
            rec[f"{var_name}_p25"]    = float(np.percentile(pixels, 25))
            rec[f"{var_name}_p75"]    = float(np.percentile(pixels, 75))
            rec[f"{var_name}_p90"]    = float(np.percentile(pixels, 90))

            # Variable-specific extras
            if var_name == "elevation":
                rec["elevation_min"] = float(np.min(pixels))
                rec["elevation_max"] = float(np.max(pixels))
                rec["pixel_count"]   = int(n_valid)

            elif var_name == "slope":
                rec["frac_steep_30deg"] = float((pixels > _STEEP_30_DEG).mean())
                rec["frac_steep_45deg"] = float((pixels > _STEEP_45_DEG).mean())

            elif var_name == "aspect":
                # Exclude flat pixels (aspect == -1) from directional fractions
                valid_asp = pixels[pixels >= 0.0]
                n_asp = len(valid_asp)
                if n_asp > 0:
                    rec["aspect_north_frac"] = float(
                        ((valid_asp >= _ASP_NORTH_LO) | (valid_asp < _ASP_NORTH_HI)).sum()
                        / n_asp
                    )
                    rec["aspect_east_frac"]  = float(
                        ((valid_asp >= _ASP_EAST_LO) & (valid_asp < _ASP_EAST_HI)).sum()
                        / n_asp
                    )
                    rec["aspect_south_frac"] = float(
                        ((valid_asp >= _ASP_SOUTH_LO) & (valid_asp < _ASP_SOUTH_HI)).sum()
                        / n_asp
                    )
                    rec["aspect_west_frac"]  = float(
                        ((valid_asp >= _ASP_WEST_LO) & (valid_asp < _ASP_WEST_HI)).sum()
                        / n_asp
                    )
                else:
                    for k in ("aspect_north_frac", "aspect_east_frac",
                              "aspect_south_frac", "aspect_west_frac"):
                        rec[k] = float("nan")

        records.append(rec)

    df = pd.DataFrame(records).set_index(ZONE_NAME_COL)
    if df.empty:
        return df

    float_cols = [c for c in df.columns if c != "pixel_count"]
    df[float_cols] = df[float_cols].astype("float32")
    if "pixel_count" in df.columns:
        df["pixel_count"] = df["pixel_count"].astype("int32")

    return df


def compute_terrain_features(
    zones_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    resolution: int = DEFAULT_RESOLUTION,
) -> pd.DataFrame:
    """Full pipeline: download DEMs and extract terrain features for all zones.

    For each zone in ``zones_gdf``:
    1. Download DEM (cached if GeoTIFF already exists).
    2. Compute slope, aspect, curvature, TRI (each cached independently).
    3. Extract zone-level summary statistics via ``extract_zone_stats``.

    Rasters are saved to ``output_dir / "rasters"``.  The final feature table
    is saved to ``output_dir / "terrain_features.parquet"``.

    Parameters
    ----------
    zones_gdf:
        GeoDataFrame of NWAC zone polygons.  Must have a ``ZONE_NAME_COL``
        column and geometry in EPSG:4326.
    output_dir:
        Root output directory.
    resolution:
        DEM resolution in metres.

    Returns
    -------
    pd.DataFrame
        Zone-level terrain feature table (one row per zone, indexed by
        ``ZONE_NAME_COL``).
    """
    output_dir = Path(output_dir)
    raster_dir = output_dir / "rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[pd.DataFrame] = []

    for _, row in zones_gdf.iterrows():
        zone_name = row[ZONE_NAME_COL]
        zone_geom = row.geometry

        logger.info("--- Processing zone: %s ---", zone_name)
        try:
            dem_path = download_dem(zone_geom, zone_name, resolution, raster_dir)
            slope_path    = compute_slope(dem_path,     raster_dir)
            aspect_path   = compute_aspect(dem_path,    raster_dir)
            curv_path     = compute_curvature(dem_path, raster_dir)
            tri_path      = compute_tri(dem_path,       raster_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Terrain pipeline failed for zone '%s': %s — skipping",
                zone_name, exc,
            )
            continue

        terrain_paths = {
            "elevation": dem_path,
            "slope":     slope_path,
            "aspect":    aspect_path,
            "curvature": curv_path,
            "tri":       tri_path,
        }

        # Pass a single-row GeoDataFrame for this zone
        zone_row_gdf = zones_gdf[zones_gdf[ZONE_NAME_COL] == zone_name].copy()
        stats = extract_zone_stats(terrain_paths, zone_row_gdf)
        all_stats.append(stats)

    if not all_stats:
        raise RuntimeError("No zones processed successfully")

    result = pd.concat(all_stats)
    result.index.name = ZONE_NAME_COL

    out_path = output_dir / "terrain_features.parquet"
    result.to_parquet(out_path)
    logger.info("Saved terrain features → %s  shape=%s", out_path, result.shape)

    return result
