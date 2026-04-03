"""Project-wide constants shared across all src/ modules.

Centralising these prevents silent join failures caused by mismatched string
literals spread across multiple files.
"""

# The canonical column name used to identify NWAC forecast zones.
# Every DataFrame and GeoDataFrame in this project that carries zone
# identifiers must use exactly this name.  Modules that touch zone names —
# terrain.py, labels.py, fusion.py — import ZONE_NAME_COL from here rather
# than hardcoding the string.
ZONE_NAME_COL: str = "zone_name"
