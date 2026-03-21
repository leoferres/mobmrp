"""
Compare naive CDR averages with MRP-corrected estimates.
"""

from __future__ import annotations

import pandas as pd

from mobmrp.config import MRPConfig
from mobmrp.poststrat.aggregate import naive_estimate, poststratify, summarize_draws


def compare_estimates(
    pred_frame: pd.DataFrame,
    cell_data: pd.DataFrame,
    config: MRPConfig,
    group_cols_list: list[list[str] | None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Produce naive-vs-MRP comparison tables at multiple aggregation levels.

    Parameters
    ----------
    pred_frame : DataFrame
        Output of :func:`~mobmrp.model.predict.generate_predictions`.
    cell_data : DataFrame
        Cell-level CDR data (for naive estimates).
    config : MRPConfig
    group_cols_list : list, optional
        Each element defines one aggregation level.  *None* entries mean
        "overall".  Default levels:

        * overall
        * by socioeconomic group
        * by gender
        * by socioeconomic group x gender
        * by geographic unit

    Returns
    -------
    dict[str, DataFrame]
        Mapping from level name (e.g. ``"by_gse"``) to a comparison
        DataFrame with columns: group cols, ``naive_mean``,
        ``mrp_mean``, ``mrp_lo``, ``mrp_hi``, ``shift``.
    """
    cols = config.columns

    if group_cols_list is None:
        group_cols_list = [
            None,
            [cols.socioeconomic_group],
            [cols.gender],
            [cols.socioeconomic_group, cols.gender],
            [cols.geographic_unit],
        ]

    level_names = {
        None: "overall",
    }

    results: dict[str, pd.DataFrame] = {}

    for gc in group_cols_list:
        # Name the level
        if gc is None:
            name = "overall"
        else:
            name = "by_" + "_x_".join(gc)

        # MRP-corrected
        mrp = poststratify(pred_frame, config, group_cols=gc)
        mrp = summarize_draws(mrp, hdi_prob=config.hdi_prob)

        # Naive
        naive = naive_estimate(cell_data, config, group_cols=gc)

        # Merge
        if gc is None:
            merged = pd.concat([naive, mrp[["mrp_mean", "mrp_lo", "mrp_hi"]]], axis=1)
        else:
            merged = mrp.merge(naive, on=gc, how="left")

        merged["shift"] = merged["mrp_mean"] - merged["naive_mean"]

        # Keep only the useful columns
        keep = []
        if gc:
            keep.extend(gc)
        keep.extend(["naive_mean", "mrp_mean", "mrp_lo", "mrp_hi", "shift"])
        keep = [c for c in keep if c in merged.columns]
        results[name] = merged[keep].reset_index(drop=True)

    return results
