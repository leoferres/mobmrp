"""
Bayesian multilevel model fitting via Bambi / PyMC.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import arviz as az
import bambi as bmb
import pandas as pd

from mobmrp.config import MRPConfig


def build_formula(config: MRPConfig) -> str:
    """Auto-generate a Bambi formula from the config.

    The default formula includes main effects for socioeconomic group and
    gender, their interaction, optional temporal controls, and a random
    intercept by geographic unit::

        log_outcome ~ C(socioeco) + C(gender) + C(socioeco):C(gender)
                    + C(weekday) + C(month)
                    + (1 | geographic_unit)

    Temporal terms are included only if the corresponding columns are set
    in ``config.columns``.
    """
    cols = config.columns

    outcome = f"log_{cols.outcome}" if config.log_transform else cols.outcome

    terms = [
        f"C({cols.socioeconomic_group})",
        f"C({cols.gender})",
        f"C({cols.socioeconomic_group}):C({cols.gender})",
    ]
    if cols.weekday:
        terms.append(f"C({cols.weekday})")
    if cols.month:
        terms.append(f"C({cols.month})")

    ri = config.random_intercept or cols.geographic_unit
    terms.append(f"(1 | {ri})")

    return f"{outcome} ~ {' + '.join(terms)}"


def fit_mrp_model(
    cell_data: pd.DataFrame,
    config: MRPConfig,
    formula: str | None = None,
    log_likelihood: bool = False,
) -> tuple[bmb.Model, az.InferenceData]:
    """Fit the multilevel regression model.

    Parameters
    ----------
    cell_data : DataFrame
        Cell-level aggregated data (output of
        :func:`~mobmrp.data.prep.aggregate_to_cells`).
    config : MRPConfig
    formula : str, optional
        Override formula.  If *None*, uses ``config.formula`` or
        auto-generates one via :func:`build_formula`.
    log_likelihood : bool
        If *True*, store pointwise log-likelihood in the trace (needed
        for LOO-CV / WAIC).

    Returns
    -------
    (model, idata)
        The fitted Bambi model and ArviZ InferenceData.
    """
    if formula is None:
        formula = config.formula or build_formula(config)

    model = bmb.Model(formula, cell_data)

    # Choose inference method: prefer numpyro if available
    method = config.inference_method
    if method == "nuts_numpyro":
        try:
            import numpyro  # noqa: F401
        except ImportError:
            warnings.warn(
                "numpyro not installed; falling back to default PyMC sampler. "
                "Install with: pip install numpyro jax",
                stacklevel=2,
            )
            method = "mcmc"

    fit_kwargs: dict = dict(
        draws=config.draws,
        chains=config.chains,
        random_seed=config.random_seed,
        inference_method=method,
    )

    if log_likelihood:
        fit_kwargs["idata_kwargs"] = {"log_likelihood": True}

    idata = model.fit(**fit_kwargs)
    return model, idata


def save_trace(idata: az.InferenceData, path: str | Path) -> None:
    """Save InferenceData to NetCDF."""
    idata.to_netcdf(str(path))


def load_trace(path: str | Path) -> az.InferenceData:
    """Load InferenceData from NetCDF."""
    return az.from_netcdf(str(path))
