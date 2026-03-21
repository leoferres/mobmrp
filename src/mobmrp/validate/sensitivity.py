"""
Model sensitivity analysis via LOO-CV / WAIC comparison.
"""

from __future__ import annotations

import warnings

import arviz as az
import bambi as bmb
import pandas as pd

from mobmrp.config import MRPConfig
from mobmrp.model.fit import build_formula


def compare_specifications(
    cell_data: pd.DataFrame,
    config: MRPConfig,
    specifications: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compare model specifications using LOO-CV and WAIC.

    Parameters
    ----------
    cell_data : DataFrame
        Cell-level aggregated data.
    config : MRPConfig
    specifications : dict[str, str], optional
        ``{name: bambi_formula}`` pairs.  If *None*, three default
        specifications are compared:

        * ``"no_interaction"`` -- drops the GSE x gender interaction
        * ``"with_interaction"`` -- the default model
        * ``"no_random_intercept"`` -- drops the random effect

    Returns
    -------
    DataFrame
        Columns: ``model``, ``loo_elpd``, ``loo_se``, ``p_loo``,
        ``waic``, ``waic_se``.
    """
    cols = config.columns
    ri_col = config.random_intercept or cols.geographic_unit
    outcome = f"log_{cols.outcome}" if config.log_transform else cols.outcome

    if specifications is None:
        base_terms = [f"C({cols.socioeconomic_group})", f"C({cols.gender})"]
        temporal = []
        if cols.weekday:
            temporal.append(f"C({cols.weekday})")
        if cols.month:
            temporal.append(f"C({cols.month})")
        temporal_str = (" + " + " + ".join(temporal)) if temporal else ""

        interaction = f"C({cols.socioeconomic_group}):C({cols.gender})"

        specifications = {
            "no_interaction": (
                f"{outcome} ~ {' + '.join(base_terms)}"
                f"{temporal_str} + (1 | {ri_col})"
            ),
            "with_interaction": (
                f"{outcome} ~ {' + '.join(base_terms)} + {interaction}"
                f"{temporal_str} + (1 | {ri_col})"
            ),
            "no_random_intercept": (
                f"{outcome} ~ {' + '.join(base_terms)} + {interaction}"
                f"{temporal_str}"
            ),
        }

    # Detect inference method
    method = config.inference_method
    if method == "nuts_numpyro":
        try:
            import numpyro  # noqa: F401
        except ImportError:
            method = "mcmc"

    results: list[dict] = []

    for name, formula in specifications.items():
        model = bmb.Model(formula, cell_data)
        idata = model.fit(
            draws=config.cv_draws,
            chains=config.cv_chains,
            random_seed=config.random_seed,
            inference_method=method,
            idata_kwargs={"log_likelihood": True},
        )

        loo = az.loo(idata, pointwise=True)
        waic = az.waic(idata, pointwise=True)

        results.append(
            {
                "model": name,
                "loo_elpd": loo.elpd_loo,
                "loo_se": loo.se,
                "p_loo": loo.p_loo,
                "waic": waic.elpd_waic,
                "waic_se": waic.se,
            }
        )

    return pd.DataFrame(results)
