"""
Configuration dataclasses for the mobmrp pipeline.

Users map their column names to internal roles via ColumnMap, then wrap
everything in MRPConfig to drive the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColumnMap:
    """Maps user column names to the roles the library expects.

    Parameters
    ----------
    outcome : str
        Name of the mobility metric column (e.g. ``"avgROG"``).
    socioeconomic_group : str
        Socioeconomic classification column (e.g. ``"gse"``).
    gender : str
        Gender column (e.g. ``"genero"``).
    geographic_unit : str
        The geographic unit for the random intercept (e.g. ``"comuna"``).
    fine_geographic_unit : str
        Finest geographic unit for the poststratification frame
        (e.g. ``"redcode"``).
    lat, lon : str
        Latitude / longitude columns in the CDR data.  Only required when
        using :func:`~mobmrp.data.spatial.nearest_neighbor_join`.
    region : str or None
        Optional higher-level region column.  When set, the spatial join is
        performed per region.
    month : str or None
        Optional month column.
    weekday : str or None
        Optional weekday indicator column.
    census_pop : str
        Column with census population counts in the poststratification frame.
    """

    outcome: str
    socioeconomic_group: str
    gender: str
    geographic_unit: str
    fine_geographic_unit: str
    lat: str = "lat"
    lon: str = "lon"
    region: str | None = None
    month: str | None = None
    weekday: str | None = None
    census_pop: str = "census_pop"


@dataclass
class MRPConfig:
    """Full pipeline configuration.

    Parameters
    ----------
    columns : ColumnMap
        Column-name mapping.
    socioeco_levels : list[str]
        Ordered list of socioeconomic group labels (e.g.
        ``["ABC1", "C2", "C3", "D", "E"]``).
    gender_levels : list[str]
        Gender labels (default ``["M", "F"]``).
    formula : str or None
        Bambi formula.  If *None*, one is auto-generated from the column map.
    log_transform : bool
        Whether to log-transform the outcome before modelling.
    random_intercept : str or None
        Column for the random intercept.  Defaults to
        ``columns.geographic_unit``.
    chains, draws : int
        MCMC sampling parameters.
    inference_method : str
        Bambi/PyMC inference method (e.g. ``"nuts_numpyro"``).
    random_seed : int
        Seed for reproducibility.
    max_join_distance_km : float
        Maximum allowable distance for the spatial nearest-neighbour join.
    reference_conditions : dict
        Temporal reference values used when predicting on the poststratification
        frame (e.g. ``{"month": 5, "is_weekday": True}``).
    n_posterior_draws : int
        Number of posterior draws to keep for uncertainty propagation.
    hdi_prob : float
        Probability mass for the highest-density interval (default 0.94).
    cv_folds : int
        Number of cross-validation folds.
    cv_chains, cv_draws : int
        Lighter sampling parameters used during cross-validation.
    """

    columns: ColumnMap

    # Categorical levels
    socioeco_levels: list[str] = field(default_factory=list)
    gender_levels: list[str] = field(default_factory=lambda: ["M", "F"])

    # Model specification
    formula: str | None = None
    log_transform: bool = True
    random_intercept: str | None = None

    # Sampling
    chains: int = 4
    draws: int = 2000
    inference_method: str = "nuts_numpyro"
    random_seed: int = 42

    # Spatial join
    max_join_distance_km: float = 5.0

    # Poststratification
    reference_conditions: dict = field(default_factory=dict)
    n_posterior_draws: int = 200
    hdi_prob: float = 0.94

    # Validation
    cv_folds: int = 5
    cv_chains: int = 2
    cv_draws: int = 1000

    def __post_init__(self) -> None:
        if self.random_intercept is None:
            self.random_intercept = self.columns.geographic_unit
