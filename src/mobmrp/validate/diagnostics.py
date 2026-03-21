"""
Convergence diagnostics and posterior predictive checks.
"""

from __future__ import annotations

import arviz as az
import bambi as bmb


def check_convergence(idata: az.InferenceData) -> dict:
    """Check standard MCMC convergence diagnostics.

    Returns
    -------
    dict
        Keys: ``max_rhat``, ``min_ess_bulk``, ``min_ess_tail``,
        ``n_divergences``, ``converged`` (bool).
    """
    summary = az.summary(idata)

    max_rhat = float(summary["r_hat"].max())
    min_ess_bulk = float(summary["ess_bulk"].min())
    min_ess_tail = float(summary["ess_tail"].min())

    n_divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_divergences = int(idata.sample_stats["diverging"].sum().values)

    converged = max_rhat <= 1.01 and min_ess_bulk >= 400 and n_divergences == 0

    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "n_divergences": n_divergences,
        "converged": converged,
    }


def posterior_predictive_check(
    model: bmb.Model,
    idata: az.InferenceData,
) -> az.InferenceData:
    """Generate posterior predictive samples (modifies *idata* in-place).

    Returns
    -------
    az.InferenceData
        The same *idata* object with ``posterior_predictive`` group added.
    """
    model.predict(idata, kind="pps", inplace=True)
    return idata
