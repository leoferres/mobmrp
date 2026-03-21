"""
mobmrp -- Correct socioeconomic bias in mobile phone mobility data
using Multilevel Regression and Poststratification (MRP).
"""

from mobmrp.config import ColumnMap, MRPConfig
from mobmrp.data.prep import aggregate_to_cells, load_and_filter
from mobmrp.data.spatial import nearest_neighbor_join
from mobmrp.data.poststrat_frame import (
    build_poststrat_frame,
    build_poststrat_frame_from_microdata,
)
from mobmrp.model.fit import (
    build_formula,
    fit_mrp_model,
    load_trace,
    save_trace,
)
from mobmrp.model.predict import generate_predictions
from mobmrp.poststrat.aggregate import naive_estimate, poststratify, summarize_draws
from mobmrp.poststrat.compare import compare_estimates
from mobmrp.validate.crossval import grouped_cross_validate
from mobmrp.validate.diagnostics import check_convergence, posterior_predictive_check
from mobmrp.validate.sensitivity import compare_specifications

__version__ = "0.1.0"

__all__ = [
    "ColumnMap",
    "MRPConfig",
    "aggregate_to_cells",
    "build_formula",
    "build_poststrat_frame",
    "build_poststrat_frame_from_microdata",
    "check_convergence",
    "compare_estimates",
    "compare_specifications",
    "fit_mrp_model",
    "generate_predictions",
    "grouped_cross_validate",
    "load_and_filter",
    "load_trace",
    "naive_estimate",
    "nearest_neighbor_join",
    "posterior_predictive_check",
    "poststratify",
    "save_trace",
    "summarize_draws",
]
