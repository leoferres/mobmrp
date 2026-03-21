"""
Data preparation: filtering and cell-level aggregation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mobmrp.config import MRPConfig


def load_and_filter(
    df: pd.DataFrame,
    config: MRPConfig,
    filters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Validate required columns and apply user-specified filters.

    Parameters
    ----------
    df : DataFrame
        Raw CDR data (loaded by the user).
    config : MRPConfig
        Column mappings and settings.
    filters : dict, optional
        ``{column_name: value_or_list}`` filters.  Rows that do not match
        are dropped.

    Returns
    -------
    DataFrame
        Filtered copy of *df*.
    """
    cols = config.columns
    required = [cols.outcome, cols.socioeconomic_group, cols.gender]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Drop rows with missing outcome or group columns
    df = df.dropna(subset=[cols.outcome, cols.socioeconomic_group, cols.gender])

    # Apply user filters
    if filters:
        for col, val in filters.items():
            if col not in df.columns:
                raise ValueError(f"Filter column '{col}' not found in data")
            if isinstance(val, (list, set, tuple)):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]

    # Coerce socioeconomic group to ordered categorical if levels provided
    if config.socioeco_levels:
        cat_dtype = pd.CategoricalDtype(
            categories=config.socioeco_levels, ordered=True
        )
        df[cols.socioeconomic_group] = df[cols.socioeconomic_group].astype(cat_dtype)

    return df.reset_index(drop=True)


def aggregate_to_cells(
    df: pd.DataFrame,
    config: MRPConfig,
) -> pd.DataFrame:
    """Aggregate individual-level CDR data to cell-level summaries.

    Groups by ``(fine_geographic_unit, geographic_unit, socioeconomic_group,
    gender, [month], [weekday])`` and computes the mean of the outcome
    variable plus a user count per cell.

    Parameters
    ----------
    df : DataFrame
        Individual-level CDR data (output of :func:`load_and_filter` or
        :func:`~mobmrp.data.spatial.nearest_neighbor_join`).
    config : MRPConfig

    Returns
    -------
    DataFrame
        One row per cell with the outcome mean, ``n_users``, and optionally
        ``log_{outcome}`` if ``config.log_transform`` is True.
    """
    cols = config.columns

    group_keys: list[str] = [
        cols.fine_geographic_unit,
        cols.geographic_unit,
        cols.socioeconomic_group,
        cols.gender,
    ]
    if cols.region and cols.region in df.columns:
        group_keys.append(cols.region)
    if cols.month and cols.month in df.columns:
        group_keys.append(cols.month)
    if cols.weekday and cols.weekday in df.columns:
        group_keys.append(cols.weekday)

    # De-duplicate keys while preserving order
    group_keys = list(dict.fromkeys(group_keys))

    agg = (
        df.groupby(group_keys, observed=True)
        .agg(
            **{
                cols.outcome: (cols.outcome, "mean"),
                "n_users": (cols.outcome, "count"),
            }
        )
        .reset_index()
    )

    if config.log_transform:
        log_col = f"log_{cols.outcome}"
        agg[log_col] = np.log(agg[cols.outcome])

    return agg
