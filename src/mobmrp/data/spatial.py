"""
Spatial nearest-neighbour join between CDR home locations and geographic units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from mobmrp.config import MRPConfig

# Approximate degrees-to-km conversion for mid-latitudes (~33 S for Chile).
_DEG_TO_KM = 111.0


def nearest_neighbor_join(
    cdr: pd.DataFrame,
    geo_lookup: pd.DataFrame,
    config: MRPConfig,
    group_by_region: bool = True,
) -> pd.DataFrame:
    """Assign geographic identifiers to CDR rows via nearest-neighbour matching.

    For each CDR row the closest point in *geo_lookup* is found using a
    KD-tree.  Matches beyond ``config.max_join_distance_km`` are dropped.

    Parameters
    ----------
    cdr : DataFrame
        Must contain ``config.columns.lat`` and ``config.columns.lon``.
    geo_lookup : DataFrame
        Must contain latitude/longitude columns (same names as in *cdr*)
        plus at least ``config.columns.fine_geographic_unit`` and
        ``config.columns.geographic_unit``.
    config : MRPConfig
    group_by_region : bool
        If *True* and ``config.columns.region`` is set, the join is
        performed separately per region (more accurate when regions are
        far apart).

    Returns
    -------
    DataFrame
        *cdr* with geographic columns from *geo_lookup* appended and a
        ``join_distance_km`` column.
    """
    cols = config.columns
    lat, lon = cols.lat, cols.lon

    # Columns to transfer from geo_lookup
    geo_id_cols = [cols.fine_geographic_unit, cols.geographic_unit]
    transfer_cols = [
        c for c in geo_lookup.columns if c not in {lat, lon, cols.region}
    ]

    do_per_region = (
        group_by_region
        and cols.region is not None
        and cols.region in cdr.columns
        and cols.region in geo_lookup.columns
    )

    parts: list[pd.DataFrame] = []

    if do_per_region:
        regions = cdr[cols.region].unique()
        for region in regions:
            cdr_r = cdr[cdr[cols.region] == region].copy()
            geo_r = geo_lookup[geo_lookup[cols.region] == region]
            if geo_r.empty:
                continue
            matched = _join_one(cdr_r, geo_r, lat, lon, transfer_cols)
            parts.append(matched)
    else:
        parts.append(_join_one(cdr.copy(), geo_lookup, lat, lon, transfer_cols))

    result = pd.concat(parts, ignore_index=True)

    # Filter by max distance
    far = result["join_distance_km"] > config.max_join_distance_km
    result = result[~far].copy()

    return result.reset_index(drop=True)


def _join_one(
    cdr: pd.DataFrame,
    geo: pd.DataFrame,
    lat: str,
    lon: str,
    transfer_cols: list[str],
) -> pd.DataFrame:
    """KD-tree nearest-neighbour join for a single partition."""
    tree = cKDTree(geo[[lat, lon]].values)
    dists, idxs = tree.query(cdr[[lat, lon]].values)

    geo_info = geo.iloc[idxs][transfer_cols].reset_index(drop=True)
    cdr = cdr.reset_index(drop=True)
    cdr = pd.concat([cdr, geo_info], axis=1)
    cdr["join_distance_km"] = dists * _DEG_TO_KM
    return cdr
