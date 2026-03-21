"""
Generate posterior predictions on the poststratification frame.
"""

from __future__ import annotations

import warnings

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from packaging.version import Version

from mobmrp.config import MRPConfig


def generate_predictions(
    model: bmb.Model,
    idata: az.InferenceData,
    poststrat_frame: pd.DataFrame,
    config: MRPConfig,
) -> pd.DataFrame:
    """Predict on the poststratification frame and attach posterior draws.

    Steps:

    1. Augment *poststrat_frame* with reference temporal conditions from
       ``config.reference_conditions``.
    2. Filter to geographic units observed during training (Bambi cannot
       predict for unseen random-effect groups without
       ``sample_new_groups``).
    3. Call ``model.predict()`` to obtain posterior predictions.
    4. Back-transform from log scale if ``config.log_transform``.
    5. Attach ``config.n_posterior_draws`` draws as columns ``draw_0`` ...
       ``draw_N`` and a point-estimate column ``pred_{outcome}``.

    Parameters
    ----------
    model : bmb.Model
    idata : az.InferenceData
    poststrat_frame : DataFrame
        Output of :func:`~mobmrp.data.poststrat_frame.build_poststrat_frame`.
    config : MRPConfig

    Returns
    -------
    DataFrame
        *poststrat_frame* with prediction columns appended.
    """
    cols = config.columns
    pred_frame = poststrat_frame.copy()

    # Add reference temporal conditions
    for col, val in config.reference_conditions.items():
        pred_frame[col] = val

    # Ensure categorical types match training data
    if config.socioeco_levels:
        cat_dtype = pd.CategoricalDtype(
            categories=config.socioeco_levels, ordered=True
        )
        pred_frame[cols.socioeconomic_group] = pred_frame[
            cols.socioeconomic_group
        ].astype(cat_dtype)

    # Filter to observed geographic units
    ri_col = config.random_intercept or cols.geographic_unit
    trained_groups = set(model.data[ri_col].unique())
    pred_frame = pred_frame[
        pred_frame[ri_col].astype(str).isin({str(g) for g in trained_groups})
    ].copy()

    # Call model.predict
    predict_kwargs: dict = dict(data=pred_frame, inplace=True)
    try:
        bambi_version = Version(bmb.__version__)
    except Exception:
        bambi_version = Version("0.0.0")

    if bambi_version >= Version("0.18"):
        predict_kwargs["kind"] = "response_params"
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            predict_kwargs["kind"] = "mean"

    model.predict(idata, **predict_kwargs)

    # Extract log-scale predictions
    # Variable name varies across Bambi versions: "mu" or "{outcome}_mean"
    outcome_name = f"log_{cols.outcome}" if config.log_transform else cols.outcome
    for candidate in ("mu", f"{outcome_name}_mean", outcome_name):
        if candidate in idata.posterior:
            pred_var = candidate
            break
    else:
        raise RuntimeError(
            f"Could not find prediction variable in posterior. "
            f"Available: {list(idata.posterior.data_vars)}"
        )

    log_preds = idata.posterior[pred_var].values  # (chains, draws, cells)

    # Back-transform and compute point estimate
    if config.log_transform:
        preds_original = np.exp(log_preds)
    else:
        preds_original = log_preds

    pred_frame[f"pred_{cols.outcome}"] = preds_original.mean(axis=(0, 1))

    # Keep a subset of posterior draws for uncertainty propagation
    n_keep = config.n_posterior_draws
    flat_draws = preds_original.reshape(-1, preds_original.shape[-1])
    rng = np.random.default_rng(config.random_seed)
    idx = rng.choice(flat_draws.shape[0], min(n_keep, flat_draws.shape[0]), replace=False)
    draw_subset = flat_draws[idx]

    draw_df = pd.DataFrame(
        draw_subset.T,
        columns=[f"draw_{i}" for i in range(draw_subset.shape[0])],
        index=pred_frame.index,
    )
    pred_frame = pd.concat([pred_frame, draw_df], axis=1)

    return pred_frame
