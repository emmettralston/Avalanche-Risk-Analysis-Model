"""SNOTEL ingestion and cleaning for PNW avalanche danger prediction.

Queries the USDA NRCS AWDB REST API to retrieve snow depth, SWE, temperature,
and precipitation time series for SNOTEL stations within or adjacent to NWAC
forecast zones. Outputs clean pandas DataFrames saved as parquet files in
data/raw/snotel/.

API base: https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1
Date range target: 2015-10-01 through 2021-04-30 (matching OAP label window).

Unit conversion is applied at parse time: raw API values are imperial (inches,
°F); all DataFrames returned by this module use metric units matching the
column names in SNOTEL_VARIABLES.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point  # noqa: F401 — re-exported for notebook convenience
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

AWDB_BASE = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
PNW_STATES = ["WA", "OR", "ID"]
DATE_START = "2015-10-01"
DATE_END = "2021-04-30"

# Single source of truth: AWDB element code → internal metric column name.
# All column names used across this module are derived from this dict.
SNOTEL_VARIABLES: dict[str, str] = {
    "WTEQ": "snow_water_equivalent_cm",
    "SNWD": "snow_depth_cm",
    "TOBS": "air_temp_observed_degC",
    "PREC": "precipitation_accum_cm",
    "TAVG": "air_temp_avg_degC",
    "TMAX": "air_temp_max_degC",
    "TMIN": "air_temp_min_degC",
}

# Wildcard station triplet patterns for the stations endpoint.
# The /services/v1/stations endpoint filters by stationTriplets wildcard,
# not by stateCode/networkCd params (confirmed from OpenAPI spec).
_PNW_SNTL_TRIPLETS = ",".join(f"*:{s}:SNTL" for s in PNW_STATES)

# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------

# Element codes grouped by unit conversion rule
_INCH_TO_CM_CODES: frozenset[str] = frozenset({"WTEQ", "SNWD", "PREC"})
_FAHRENHEIT_CODES: frozenset[str] = frozenset({"TOBS", "TAVG", "TMAX", "TMIN"})

# Legacy AWDB sentinel for "no data" — treat as NaN
_MISSING_SENTINEL = -9999.0

# Physical plausibility bounds (metric units).
# Keys are derived from SNOTEL_VARIABLES to avoid hardcoded strings.
_BOUNDS: dict[str, tuple[float, float]] = {
    SNOTEL_VARIABLES["WTEQ"]: (0.0, 762.0),    # 0–300 inches SWE
    SNOTEL_VARIABLES["SNWD"]: (0.0, 1524.0),   # 0–600 inches snow depth
    SNOTEL_VARIABLES["PREC"]: (0.0, 3000.0),   # seasonal accum, very generous
    SNOTEL_VARIABLES["TOBS"]: (-60.0, 50.0),   # °C
    SNOTEL_VARIABLES["TAVG"]: (-60.0, 50.0),
    SNOTEL_VARIABLES["TMAX"]: (-60.0, 50.0),
    SNOTEL_VARIABLES["TMIN"]: (-60.0, 50.0),
}

_UTM10N = "EPSG:32610"          # projected CRS for PNW spatial ops
_REQUEST_DELAY_S = 0.25         # polite delay between API calls
_DATA_REQUEST_TIMEOUT_S = 120   # generous timeout for large batch data requests
_META_REQUEST_TIMEOUT_S = 60    # station list can be large; allow 60s


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _convert_value(element_code: str, raw: float) -> float:
    """Convert a raw API value from imperial units to metric.

    Parameters
    ----------
    element_code:
        AWDB element code, e.g. "WTEQ".
    raw:
        Raw numeric value from the API (assumed non-null, non-sentinel).

    Returns
    -------
    float
        Value in metric units matching the corresponding SNOTEL_VARIABLES name.
    """
    if element_code in _INCH_TO_CM_CODES:
        return raw * 2.54
    if element_code in _FAHRENHEIT_CODES:
        return (raw - 32.0) * 5.0 / 9.0
    return raw


def _parse_value(element_code: str, raw: object) -> float | None:
    """Parse and convert a raw API value, returning None if missing or invalid.

    Handles null, -9999 sentinel, and non-numeric strings defensively.
    """
    if raw is None:
        return None
    try:
        fval = float(raw)
    except (TypeError, ValueError):
        return None
    if abs(fval - _MISSING_SENTINEL) < 0.001:
        return None
    return _convert_value(element_code, fval)


def _get_json(url: str, params: dict, timeout: int = _META_REQUEST_TIMEOUT_S) -> object:
    """GET a URL, raise on HTTP error, return parsed JSON."""
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _get_json_with_retry(
    url: str,
    params: dict,
    timeout: int = _DATA_REQUEST_TIMEOUT_S,
) -> object:
    """GET a URL with up to 3 retries and exponential backoff (1s, 2s, 4s).

    Used for batch data requests that may hit transient server 5xx errors.
    Raises the last exception if all attempts fail.
    """
    return _get_json(url, params=params, timeout=timeout)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_pnw_stations(
    zones_gdf: gpd.GeoDataFrame,
    buffer_km: float = 10.0,
) -> gpd.GeoDataFrame:
    """Return SNOTEL stations located within or near NWAC forecast zones.

    Queries the AWDB REST API for active stations in WA, OR, and ID, then
    spatially filters to those within ``buffer_km`` of any zone boundary.

    Parameters
    ----------
    zones_gdf:
        GeoDataFrame of NWAC zone polygons (EPSG:4326).
    buffer_km:
        Distance in kilometres to buffer zone boundaries when filtering
        stations. Stations beyond this distance are excluded.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: station_id, station_name, latitude, longitude, elevation_ft,
        state, geometry.  CRS is EPSG:4326.
    """
    # Step 1: fetch full station objects for all PNW SNTL stations.
    # The /stations endpoint filters by stationTriplets wildcard patterns;
    # there are no stateCode/networkCd params (per OpenAPI spec).
    logger.info("Fetching SNOTEL station metadata for %s (this may take ~30s)…", PNW_STATES)
    raw_stations = _get_json(
        f"{AWDB_BASE}/stations",
        params={
            "stationTriplets": _PNW_SNTL_TRIPLETS,
            "activeOnly": "false",
        },
        timeout=_META_REQUEST_TIMEOUT_S,
    )
    if not isinstance(raw_stations, list) or not raw_stations:
        raise RuntimeError(
            f"Unexpected station list response type: {type(raw_stations)}"
        )
    logger.info("Received %d station records from API", len(raw_stations))

    # Step 2: parse StationDTO objects.
    # Field names per OpenAPI spec: stationTriplet, name, stateCode, latitude,
    # longitude, elevation, beginDate, endDate.
    records: list[dict] = []
    for st in raw_stations:
        if not isinstance(st, dict):
            continue
        triplet = st.get("stationTriplet")
        if not triplet:
            continue
        try:
            records.append({
                "station_id":   triplet,
                "station_name": st.get("name", ""),
                "latitude":     float(st["latitude"]),
                "longitude":    float(st["longitude"]),
                "elevation_ft": float(st.get("elevation") or 0),
                "state":        st.get("stateCode", ""),   # field is stateCode, not state
                "begin_date":   pd.to_datetime(st.get("beginDate"), errors="coerce"),
                "end_date":     pd.to_datetime(
                    st.get("endDate") or "2100-01-01", errors="coerce"
                ),
            })
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed station record %s: %s", triplet, exc)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No valid station metadata parsed from API response")

    # Step 3: filter to stations active during our target window
    t_start = pd.Timestamp(DATE_START)
    t_end = pd.Timestamp(DATE_END)
    active_mask = (df["begin_date"] <= t_end) & (df["end_date"] >= t_start)
    df = df[active_mask].copy()
    logger.info("%d stations active during %s – %s", len(df), DATE_START, DATE_END)

    # Step 4: spatial filter — keep stations within buffer_km of any zone
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    zones_proj = zones_gdf.to_crs(_UTM10N)
    gdf_proj = gdf.to_crs(_UTM10N)
    buffered_union = zones_proj.geometry.union_all().buffer(buffer_km * 1_000.0)
    in_buffer = gdf_proj.geometry.within(buffered_union)

    result = (
        gdf[in_buffer]
        .drop(columns=["begin_date", "end_date"])
        .reset_index(drop=True)
    )
    logger.info("%d stations within %.1f km of zone boundaries", len(result), buffer_km)
    return result


def fetch_station_data(
    station_id: str,
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch daily time series for a single SNOTEL station from the AWDB API.

    Issues one HTTP request per element code. Unit conversion (imperial →
    metric) is applied inline; returned column names match SNOTEL_VARIABLES
    values.

    Parameters
    ----------
    station_id:
        AWDB station triplet, e.g. "679:WA:SNTL".
    start_date:
        Inclusive start date as "YYYY-MM-DD".
    end_date:
        Inclusive end date as "YYYY-MM-DD".
    variables:
        List of AWDB element codes to retrieve (see ``SNOTEL_VARIABLES``).
        Defaults to all keys in ``SNOTEL_VARIABLES``.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex named "date" (daily), columns named by
        ``SNOTEL_VARIABLES`` values (metric units).  Every day in
        [start_date, end_date] is present; missing values are NaN.
    """
    if variables is None:
        variables = list(SNOTEL_VARIABLES.keys())

    date_index = pd.date_range(start_date, end_date, freq="D", name="date")
    col_names = [SNOTEL_VARIABLES[v] for v in variables]
    result = pd.DataFrame(
        np.nan,
        index=date_index,
        columns=col_names,
        dtype="float32",
    )

    for elem_code in variables:
        col = SNOTEL_VARIABLES[elem_code]
        try:
            response = _get_json(
                f"{AWDB_BASE}/data",
                params={
                    "stationTriplets": station_id,
                    "elements": elem_code,       # API param is "elements" not "elementCd"
                    "beginDate": start_date,
                    "endDate": end_date,
                    "duration": "DAILY",
                    "returnFlags": "false",
                },
                timeout=_DATA_REQUEST_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            logger.warning("Request failed for %s / %s: %s", station_id, elem_code, exc)
            time.sleep(_REQUEST_DELAY_S)
            continue

        if not isinstance(response, list) or not response:
            logger.debug("Empty response for %s / %s", station_id, elem_code)
            time.sleep(_REQUEST_DELAY_S)
            continue

        # Response schema: list of StationDataDTO.
        # Each StationDataDTO: { stationTriplet, data: [ { stationElement, values: [...] } ] }
        station_dto = next(
            (e for e in response if isinstance(e, dict)
             and e.get("stationTriplet") == station_id),
            None,
        )
        if station_dto is None:
            logger.warning("Station %s absent from %s response", station_id, elem_code)
            time.sleep(_REQUEST_DELAY_S)
            continue

        data_list = station_dto.get("data") or []
        if not data_list:
            time.sleep(_REQUEST_DELAY_S)
            continue

        # data[0] is DataDTO; values are in data[0]["values"]
        values = data_list[0].get("values") or []
        for v in values:
            converted = _parse_value(elem_code, v.get("value"))
            if converted is None:
                continue
            date_str = str(v.get("date", ""))[:10]
            try:
                ts = pd.Timestamp(date_str)
                if ts in result.index:
                    result.at[ts, col] = converted
            except Exception:  # noqa: BLE001
                continue

        time.sleep(_REQUEST_DELAY_S)

    return result


def fetch_all_stations(
    station_ids: list[str],
    start_date: str = DATE_START,
    end_date: str = DATE_END,
    variables: list[str] | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch and concatenate daily data for a list of SNOTEL stations.

    Issues one HTTP request per element code with all station IDs batched in a
    single call (7 requests total for the default variable set).  Per-station
    errors in a batch response are caught and logged without dropping the rest
    of the batch.

    Parameters
    ----------
    station_ids:
        List of AWDB station triplets.
    start_date:
        Inclusive start date as "YYYY-MM-DD".
    end_date:
        Inclusive end date as "YYYY-MM-DD".
    variables:
        Element codes to retrieve. Defaults to all ``SNOTEL_VARIABLES`` keys.
    output_dir:
        Directory where ``snotel_raw.parquet`` will be written.  If None,
        the DataFrame is returned only in memory.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex named "date" (daily), columns: station_id (str), plus
        one column per variable in ``SNOTEL_VARIABLES`` (metric units, float32).
    """
    if variables is None:
        variables = list(SNOTEL_VARIABLES.keys())

    invalid = [v for v in variables if v not in SNOTEL_VARIABLES]
    if invalid:
        raise ValueError(f"Unknown element codes: {invalid}")

    date_index = pd.date_range(start_date, end_date, freq="D", name="date")

    # Accumulate parsed series: station_id → {elem_code → pd.Series(date→value)}
    accumulated: dict[str, dict[str, pd.Series]] = {sid: {} for sid in station_ids}

    for elem_code in variables:
        col = SNOTEL_VARIABLES[elem_code]
        logger.info("Fetching %-4s (%s) for %d stations…",
                    elem_code, col, len(station_ids))
        try:
            response = _get_json_with_retry(
                f"{AWDB_BASE}/data",
                params={
                    "stationTriplets": ",".join(station_ids),
                    "elements": elem_code,       # API param is "elements" not "elementCd"
                    "beginDate": start_date,
                    "endDate": end_date,
                    "duration": "DAILY",
                    "returnFlags": "false",
                },
            )
        except requests.RequestException as exc:
            # All 3 retry attempts exhausted — log and leave column all-NaN.
            logger.warning(
                "Batch request for %s failed after retries: %s — column will be NaN",
                elem_code, exc,
            )
            time.sleep(_REQUEST_DELAY_S)
            continue

        if not isinstance(response, list):
            logger.error("Unexpected response type for %s: %s", elem_code, type(response))
            time.sleep(_REQUEST_DELAY_S)
            continue

        # Response: list of StationDataDTO.
        # Each: { stationTriplet, data: [ { stationElement, values: [...] } ] }
        stations_seen: set[str] = set()
        for station_dto in response:
            if not isinstance(station_dto, dict):
                continue
            sid = station_dto.get("stationTriplet")
            if sid is None:
                continue
            if sid not in accumulated:
                logger.debug("Unexpected station in response: %s", sid)
                continue

            stations_seen.add(sid)
            data_list = station_dto.get("data") or []
            if not data_list:
                continue

            # Per-station error isolation: catch any parse failure without
            # dropping other stations in this batch.
            values = data_list[0].get("values") or []
            parsed: dict[pd.Timestamp, float] = {}
            for v in values:
                converted = _parse_value(elem_code, v.get("value"))
                if converted is None:
                    continue
                date_str = str(v.get("date", ""))[:10]
                try:
                    parsed[pd.Timestamp(date_str)] = converted
                except Exception:  # noqa: BLE001
                    continue

            if parsed:
                accumulated[sid][elem_code] = pd.Series(parsed, dtype="float32")

        missing = set(station_ids) - stations_seen
        if missing:
            logger.warning(
                "%d/%d stations absent from %s response: %s%s",
                len(missing), len(station_ids), elem_code,
                list(missing)[:3],
                " …" if len(missing) > 3 else "",
            )

        time.sleep(_REQUEST_DELAY_S)

    # Assemble tidy DataFrame (date × station rows)
    frames: list[pd.DataFrame] = []
    skipped = 0
    for sid in station_ids:
        elem_map = accumulated[sid]
        if not elem_map:
            logger.warning("Station %s returned no data — skipping", sid)
            skipped += 1
            continue

        sdf = pd.DataFrame(index=date_index)
        for elem_code in variables:
            col = SNOTEL_VARIABLES[elem_code]
            series = elem_map.get(elem_code)
            if series is not None:
                sdf[col] = series.reindex(date_index).values
            else:
                sdf[col] = np.nan
        sdf = sdf.astype("float32")
        sdf["station_id"] = sid
        frames.append(sdf)

    if not frames:
        raise RuntimeError("No data retrieved for any of the requested stations")

    logger.info(
        "Assembled data for %d/%d stations (%d skipped)",
        len(frames), len(station_ids), skipped,
    )

    combined = pd.concat(frames)
    combined.index.name = "date"

    if output_dir is not None:
        out_path = Path(output_dir) / "snotel_raw.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path)
        logger.info("Saved raw data → %s  shape=%s", out_path, combined.shape)

    return combined


def clean_station_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality-control rules to raw SNOTEL station data.

    Steps applied (in order):
    1. Clip physically implausible values to NaN (bounds in ``_BOUNDS``).
    2. Re-index each station to a gapless daily DatetimeIndex.
    3. Fill short gaps (≤ 3 consecutive days) by linear time interpolation,
       per station, per variable.
    4. Add ``*_missing`` boolean columns (True where NaN remains after fill).
    5. Drop duplicate (date, station_id) rows, keeping first.

    Parameters
    ----------
    df:
        Raw DataFrame as returned by ``fetch_all_stations``.  Must have a
        DatetimeIndex named "date" and a "station_id" column.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with same structure plus ``*_missing`` boolean
        columns for each weather variable.
    """
    metric_cols = list(SNOTEL_VARIABLES.values())
    present_cols = [c for c in metric_cols if c in df.columns]

    if "station_id" not in df.columns:
        raise ValueError("Input DataFrame must have a 'station_id' column")

    out = df.copy()

    # Step 1: bounds clipping — out-of-range → NaN
    for col in present_cols:
        lo, hi = _BOUNDS.get(col, (None, None))
        if lo is None:
            continue
        bad = (out[col] < lo) | (out[col] > hi)
        n_bad = int(bad.sum())
        if n_bad:
            logger.warning("Replaced %d out-of-range values in %s with NaN", n_bad, col)
        out.loc[bad, col] = np.nan

    # Steps 2–3: per-station reindex + interpolation
    groups: list[pd.DataFrame] = []
    for sid, grp in out.groupby("station_id", sort=False):
        grp = grp.copy().sort_index()

        # Reindex to complete daily range for this station's span
        full_idx = pd.date_range(
            grp.index.min(), grp.index.max(), freq="D", name="date"
        )
        grp = grp.reindex(full_idx)
        grp["station_id"] = sid  # reindex sets station_id to NaN for new rows

        for col in present_cols:
            grp[col] = grp[col].interpolate(
                method="time", limit=3, limit_direction="forward"
            )

        groups.append(grp)

    out = pd.concat(groups)
    out.index.name = "date"

    # Step 4: missing flags (after interpolation, so gaps > 3 days are True)
    for col in present_cols:
        out[f"{col}_missing"] = out[col].isna()

    # Step 5: deduplicate on (date, station_id)
    n_before = len(out)
    out_reset = out.reset_index()
    out_reset = out_reset.drop_duplicates(subset=["date", "station_id"], keep="first")
    n_duped = n_before - len(out_reset)
    if n_duped:
        logger.warning("Dropped %d duplicate (date, station_id) rows", n_duped)
    out = out_reset.set_index("date")

    return out


def load_snotel(data_dir: Path) -> pd.DataFrame:
    """Load SNOTEL data from parquet.

    Prefers ``snotel_clean.parquet`` if present; falls back to
    ``snotel_raw.parquet``.

    Parameters
    ----------
    data_dir:
        Directory containing the parquet file(s).

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex named "date" and a "station_id" column.
    """
    data_dir = Path(data_dir)
    clean_path = data_dir / "snotel_clean.parquet"
    raw_path = data_dir / "snotel_raw.parquet"

    if clean_path.exists():
        path = clean_path
    elif raw_path.exists():
        path = raw_path
    else:
        raise FileNotFoundError(
            f"No SNOTEL parquet file found in {data_dir}. "
            "Run fetch_all_stations() first."
        )

    df = pd.read_parquet(path)

    # Ensure DatetimeIndex
    if "date" in df.columns:
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    logger.info("Loaded %s  shape=%s", path.name, df.shape)
    return df
