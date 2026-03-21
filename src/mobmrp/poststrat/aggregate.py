"""
Core poststratification math: census-weighted aggregation of predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mobmrp.config import MRPConfig


def poststratify(
    pred_frame: pd.DataFrame,
    config: MRPConfig,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute MRP-corrected weighted estimates.

    For a domain *D* defined by *group_cols*:

    .. math::

        \\hat\\theta_D = \\frac{\\sum_{j \\in D} N_j \\, \\hat y_j}
                              {\\sum_{j \\in D} N_j}

    This is computed for every posterior draw to propagate uncertainty.

    Parameters
    ----------
    pred_frame : DataFrame
        Output of :func:`~mobmrp.model.predict.generate_predictions`.
        Must contain a ``census_pop`` column (name from
        ``config.columns.census_pop``) and draw columns ``draw_0``, ...
    config : MRPConfig
    group_cols : list[str] or None
        Columns defining the aggregation domain.  If *None*, computes
        the overall population estimate.

    Returns
    -------
    DataFrame
        Weighted estimates with columns for each draw plus ``census_pop``.
    """
    pop_col = config.columns.census_pop
    draw_cols = [c for c in pred_frame.columns if c.startswith("draw_")]
    if not draw_cols:
        # Fallback to pred column
        pred_col = f"pred_{config.columns.outcome}"
        if pred_col in pred_frame.columns:
            draw_cols = [pred_col]
        else:
            raise ValueError("No draw or prediction columns found in pred_frame")

    if group_cols is None:
        total_pop = pred_frame[pop_col].sum()
        result: dict = {}
        result[pop_col] = total_pop
        for c in draw_cols:
            result[c] = (pred_frame[c] * pred_frame[pop_col]).sum() / total_pop
        return pd.DataFrame([result])

    grouped = pred_frame.groupby(group_cols, observed=True)
    rows: list[dict] = []
    for name, grp in grouped:
        total_pop = grp[pop_col].sum()
        if total_pop == 0:
            continue
        if isinstance(name, tuple):
            row = dict(zip(group_cols, name))
        else:
            row = {group_cols[0]: name}
        row[pop_col] = total_pop
        for c in draw_cols:
            row[c] = (grp[c] * grp[pop_col]).sum() / total_pop
        rows.append(row)

    return pd.DataFrame(rows)


def naive_estimate(
    cell_data: pd.DataFrame,
    config: MRPConfig,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute unweighted CDR average (the naive/biased estimate).

    Parameters
    ----------
    cell_data : DataFrame
        Cell-level data with the outcome column.
    config : MRPConfig
    group_cols : list[str] or None
        If *None*, returns a single-row DataFrame with the overall mean.

    Returns
    -------
    DataFrame
        With columns ``group_cols`` + ``naive_mean``.
    """
    outcome = config.columns.outcome
    if group_cols is None:
        return pd.DataFrame([{"naive_mean": cell_data[outcome].mean()}])

    result = (
        cell_data.groupby(group_cols, observed=True)[outcome]
        .mean()
        .reset_index()
        .rename(columns={outcome: "naive_mean"})
    )
    return result


def summarize_draws(
    df: pd.DataFrame,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Add ``mrp_mean``, ``mrp_lo``, ``mrp_hi`` from draw columns.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``draw_0``, ``draw_1``, ...
    hdi_prob : float
        Probability mass for the interval (default 0.94).

    Returns
    -------
    DataFrame
        *df* with three new columns appended.
    """
    draw_cols = [c for c in df.columns if c.startswith("draw_")]
    if not draw_cols:
        raise ValueError("No draw columns found")

    alpha = (1 - hdi_prob) / 2
    lo_q = alpha
    hi_q = 1 - alpha

    draws = df[draw_cols].values
    df = df.copy()
    df["mrp_mean"] = draws.mean(axis=1)
    df["mrp_lo"] = np.percentile(draws, lo_q * 100, axis=1)
    df["mrp_hi"] = np.percentile(draws, hi_q * 100, axis=1)

    return df
