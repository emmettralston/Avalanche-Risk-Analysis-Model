"""Label loading and zone alignment for OAP / NWAC danger ratings.

Loads pre-cleaned avalanche danger ratings from the Open Avalanche Project
(OAP) label archive and aligns them to NWAC forecast zone names used in the
zones GeoJSON.  Provides utilities for season-aware labelling needed for
temporally correct cross-validation.

OAP label source:
  https://github.com/scottcha/OpenAvalancheProject
  File: Data/CleanedForecastsNWAC_CAIC_UAC_CAC.V1.2013-2021.zip

Schema (relevant columns):
  date          YYYY-MM-DD
  center        Avalanche center (filter to "NWAC")
  zone          Forecast zone name (may need normalisation)
  danger_rating Overall danger level 1–5
  avalanche_problems  Comma-separated avalanche problem types (optional)
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

# Danger level mapping following North American Avalanche Danger Scale
DANGER_LEVELS = {1: "Low", 2: "Moderate", 3: "Considerable", 4: "High", 5: "Extreme"}

# Avalanche season runs approximately November through April
SEASON_START_MONTH = 11  # November
SEASON_END_MONTH = 4  # April


def load_oap_labels(
    zip_path: Path,
    center: str = "NWAC",
) -> pd.DataFrame:
    """Load and filter the OAP label CSV from the distribution zip archive.

    Parameters
    ----------
    zip_path:
        Path to the OAP zip file
        (``CleanedForecastsNWAC_CAIC_UAC_CAC.V1.2013-2021.zip``).
    center:
        Avalanche center abbreviation to keep.  Use "NWAC" for this project.

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime64[ns]), zone (str), danger_rating (int8),
        avalanche_problems (str, nullable).  Sorted by date, zone.
        Only rows where danger_rating is in [1, 5] are kept.
    """
    raise NotImplementedError


def normalize_zone_names(
    labels_df: pd.DataFrame,
    zones_gdf: gpd.GeoDataFrame,
    zone_col: str = "zone_name",
) -> pd.DataFrame:
    """Align OAP zone name strings to canonical names in the zones GeoJSON.

    OAP zone names may differ in capitalisation, spacing, or abbreviation from
    the GeoJSON ``zone_col`` values.  This function applies a best-effort
    normalisation (lowercasing, stripping punctuation) and warns on any
    unmatched names.

    Parameters
    ----------
    labels_df:
        DataFrame as returned by ``load_oap_labels``.
    zones_gdf:
        GeoDataFrame with authoritative zone names in column ``zone_col``.
    zone_col:
        Column in ``zones_gdf`` containing canonical zone names.

    Returns
    -------
    pd.DataFrame
        ``labels_df`` with the ``zone`` column replaced by the canonical name.
        Rows with unresolvable zone names are dropped with a warning.
    """
    raise NotImplementedError


def validate_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Check label integrity and report anomalies.

    Checks performed:
    - No duplicate (date, zone) pairs.
    - danger_rating values are in [1, 5].
    - Date range covers at least 2015-10-01 through 2021-04-30.
    - No zones with fewer than 30 labelled days (likely incomplete data).

    Parameters
    ----------
    labels_df:
        DataFrame as returned by ``load_oap_labels`` (after zone normalisation).

    Returns
    -------
    pd.DataFrame
        Validated DataFrame.  Raises ValueError on critical integrity failures.
        Logs warnings for non-critical anomalies.
    """
    raise NotImplementedError


def get_season(date: pd.Timestamp) -> str:
    """Return the avalanche season string for a given date.

    The season is defined as November through April.  A date in November or
    December belongs to the season that starts that calendar year; a date in
    January through April belongs to the season that started the previous year.

    Examples
    --------
    >>> get_season(pd.Timestamp("2019-01-15"))
    '2018-19'
    >>> get_season(pd.Timestamp("2019-11-20"))
    '2019-20'

    Parameters
    ----------
    date:
        Timestamp to classify.

    Returns
    -------
    str
        Season identifier, e.g. "2018-19".
    """
    raise NotImplementedError


def add_season_column(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``season`` string column to the labels DataFrame.

    Applies ``get_season`` to the ``date`` column.  This column is used as the
    grouping key for season-aware cross-validation splits.

    Parameters
    ----------
    labels_df:
        DataFrame with a ``date`` column of dtype datetime64.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional ``season`` column (str).
    """
    raise NotImplementedError
