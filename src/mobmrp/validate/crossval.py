"""
Grouped cross-validation by geographic unit.
"""

from __future__ import annotations

import warnings

import bambi as bmb
import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn.model_selection import KFold

from mobmrp.config import MRPConfig
from mobmrp.model.fit import build_formula


def grouped_cross_validate(
    cell_data: pd.DataFrame,
    config: MRPConfig,
    formula: str | None = None,
) -> pd.DataFrame:
    """K-fold cross-validation grouped by geographic unit.

    For each fold, 1/K of the geographic units are held out.  A model is
    fit on the remaining data and used to predict the held-out units
    (``sample_new_groups=True``).

    Parameters
    ----------
    cell_data : DataFrame
        Cell-level aggregated data.
    config : MRPConfig
    formula : str, optional
        Override formula.  Defaults to ``config.formula`` or auto-generated.

    Returns
    -------
    DataFrame
        Per-fold metrics: ``fold``, ``n_test_units``, ``rmse_log``,
        ``coverage``.
    """
    cols = config.columns
    ri_col = config.random_intercept or cols.geographic_unit
    outcome_col = f"log_{cols.outcome}" if config.log_transform else cols.outcome

    if formula is None:
        formula = config.formula or build_formula(config)

    units = cell_data[ri_col].unique()
    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed)

    # Detect inference method
    method = config.inference_method
    if method == "nuts_numpyro":
        try:
            import numpyro  # noqa: F401
        except ImportError:
            method = "mcmc"

    try:
        bambi_version = Version(bmb.__version__)
    except Exception:
        bambi_version = Version("0.0.0")

    results: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(units)):
        train_units = set(units[train_idx])
        test_units = set(units[test_idx])

        df_train = cell_data[cell_data[ri_col].isin(train_units)].copy()
        df_test = cell_data[cell_data[ri_col].isin(test_units)].copy()

        model_cv = bmb.Model(formula, df_train)
        idata_cv = model_cv.fit(
            draws=config.cv_draws,
            chains=config.cv_chains,
            random_seed=fold_i,
            inference_method=method,
        )

        # Predict on held-out units
        predict_kwargs: dict = dict(
            data=df_test,
            inplace=True,
            sample_new_groups=True,
        )
        if bambi_version >= Version("0.18"):
            predict_kwargs["kind"] = "response_params"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                predict_kwargs["kind"] = "mean"

        model_cv.predict(idata_cv, **predict_kwargs)

        # Find prediction variable
        for candidate in ("mu", f"{outcome_col}_mean", outcome_col):
            if candidate in idata_cv.posterior:
                pred_var = candidate
                break
        else:
            raise RuntimeError("Could not find prediction variable in posterior")

        log_preds = idata_cv.posterior[pred_var].values
        pred_mean = log_preds.mean(axis=(0, 1))

        # RMSE on log scale
        rmse = float(np.sqrt(np.mean((df_test[outcome_col].values - pred_mean) ** 2)))

        # Coverage
        alpha = (1 - config.hdi_prob) / 2
        pred_lo = np.percentile(log_preds, alpha * 100, axis=(0, 1))
        pred_hi = np.percentile(log_preds, (1 - alpha) * 100, axis=(0, 1))
        observed = df_test[outcome_col].values
        coverage = float(np.mean((observed >= pred_lo) & (observed <= pred_hi)))

        results.append(
            {
                "fold": fold_i + 1,
                "n_test_units": len(test_units),
                "rmse_log": rmse,
                "coverage": coverage,
            }
        )

    return pd.DataFrame(results)
