# mobmrp

Correct socioeconomic bias in mobile phone mobility data using Multilevel Regression and Poststratification (MRP).

Mobile phone data (Call Detail Records) from a single carrier are not representative of the general population. Different socioeconomic groups have different probabilities of appearing in a given carrier's data, which biases aggregate mobility estimates. `mobmrp` implements a principled correction: fit a Bayesian multilevel model on the biased CDR data, then reweight predictions to match the true census population distribution. The method is described in:

> Ferres, L. "Correcting Socioeconomic Bias in Mobile Phone Mobility Estimates Using Multilevel Regression and Poststratification." Working paper, 2026.

## Installation

```bash
pip install mobmrp
```

For JAX-accelerated sampling (recommended for large datasets):

```bash
pip install mobmrp[numpyro]
```

## Quick start

```python
import pandas as pd
from mobmrp import (
    MRPConfig, ColumnMap,
    load_and_filter, aggregate_to_cells,
    build_poststrat_frame,
    fit_mrp_model, generate_predictions,
    compare_estimates, check_convergence,
)

# 1. Configure: map your column names
config = MRPConfig(
    columns=ColumnMap(
        outcome="avg_radius_of_gyration",
        socioeconomic_group="income_group",
        gender="sex",
        geographic_unit="district",
        fine_geographic_unit="census_tract",
        month="month",
        weekday="is_weekday",
    ),
    socioeco_levels=["high", "mid_high", "mid", "mid_low", "low"],
    gender_levels=["M", "F"],
    reference_conditions={"month": 5, "is_weekday": True},
)

# 2. Prepare data
cdr = load_and_filter(pd.read_csv("cdr.csv"), config)
cells = aggregate_to_cells(cdr, config)

# 3. Build poststratification frame from census
ps_frame = build_poststrat_frame(geo_proportions, gender_proportions, config)

# 4. Fit model
model, idata = fit_mrp_model(cells, config)
print(check_convergence(idata))

# 5. Predict and compare
pred_frame = generate_predictions(model, idata, ps_frame, config)
results = compare_estimates(pred_frame, cells, config)

for level, df in results.items():
    print(f"\n{level}:")
    print(df)
```

See `examples/synthetic_example.py` for a fully self-contained demo.

## Required data

### CDR data

A DataFrame with one row per user (or user-period). Required columns:

| Column | Description | Example |
|--------|-------------|---------|
| outcome | Mobility metric (e.g. radius of gyration) | `12.5` |
| socioeconomic_group | Socioeconomic classification | `"ABC1"` |
| gender | Gender | `"M"` |
| geographic_unit | Geographic unit for random intercept | `"13101"` |
| fine_geographic_unit | Finest geographic unit | `"13101011001"` |
| month *(optional)* | Month indicator | `5` |
| weekday *(optional)* | Weekday/weekend indicator | `True` |
| lat, lon *(optional)* | Home location, if spatial join is needed | `-33.45, -70.66` |

Column names are arbitrary; you map them in `ColumnMap`.

### Census / population data

Two DataFrames for building the poststratification frame:

**Geographic proportions** (one row per fine geographic unit):

| Column | Description |
|--------|-------------|
| fine_geographic_unit | Area identifier |
| geographic_unit | Parent area identifier |
| total_pop | Total population (persons or households) |
| one column per socioeco level | Proportion of that level in the area |

**Gender proportions** (one row per geographic unit):

| Column | Description |
|--------|-------------|
| geographic_unit | Area identifier |
| one column per gender level | Proportion of that gender in the area |

### Geographic lookup (optional)

Only needed if your CDR data has raw lat/lon and needs a spatial join. A DataFrame with lat/lon plus geographic identifiers. `nearest_neighbor_join()` uses a KD-tree to match CDR locations to the nearest point.

## Pipeline overview

1. **Data preparation** -- Filter CDR, optionally spatial-join to geographic units, aggregate to cells
2. **Model fitting** -- Bayesian multilevel regression via Bambi/PyMC with random intercepts by geographic unit
3. **Prediction** -- Posterior predictions for every cell in the poststratification frame
4. **Poststratification** -- Census-weighted aggregation with full uncertainty propagation
5. **Validation** *(optional)* -- Grouped cross-validation and model comparison via LOO-CV

## Configuration

All column names and pipeline parameters are set via `MRPConfig`:

```python
config = MRPConfig(
    columns=ColumnMap(
        outcome="avgROG",              # your mobility metric
        socioeconomic_group="gse",     # socioeconomic classification
        gender="genero",               # gender column
        geographic_unit="comuna",      # random intercept level
        fine_geographic_unit="redcode",# finest census unit
    ),
    socioeco_levels=["ABC1", "C2", "C3", "D", "E"],
    log_transform=True,         # log-transform outcome before modelling
    chains=4, draws=2000,       # MCMC sampling
    inference_method="nuts_numpyro",  # JAX-accelerated NUTS
    hdi_prob=0.94,              # credible interval width
)
```

## API reference

### Data preparation

- `load_and_filter(df, config, filters=None)` -- Validate columns and apply filters
- `aggregate_to_cells(df, config)` -- Aggregate to cell-level summaries
- `nearest_neighbor_join(cdr, geo_lookup, config)` -- Spatial join via KD-tree
- `build_poststrat_frame(geo_proportions, gender_proportions, config)` -- Build census frame
- `build_poststrat_frame_from_microdata(census_microdata, geo_proportions, config, ...)` -- Build from person-level census

### Model

- `build_formula(config)` -- Auto-generate Bambi formula
- `fit_mrp_model(cell_data, config)` -- Fit multilevel model, returns `(model, idata)`
- `generate_predictions(model, idata, poststrat_frame, config)` -- Posterior predictions
- `save_trace(idata, path)` / `load_trace(path)` -- NetCDF I/O

### Poststratification

- `poststratify(pred_frame, config, group_cols=None)` -- Census-weighted estimates
- `naive_estimate(cell_data, config, group_cols=None)` -- Unweighted CDR averages
- `summarize_draws(df, hdi_prob=0.94)` -- Mean + HDI from posterior draws
- `compare_estimates(pred_frame, cell_data, config)` -- Naive vs MRP at multiple levels

### Validation

- `check_convergence(idata)` -- R-hat, ESS, divergences
- `posterior_predictive_check(model, idata)` -- Generate PPC samples
- `grouped_cross_validate(cell_data, config)` -- K-fold CV by geographic unit
- `compare_specifications(cell_data, config, specifications=None)` -- LOO-CV model comparison

## Citation

```bibtex
@unpublished{ferres2026mobmrp,
  title={Correcting Socioeconomic Bias in Mobile Phone Mobility Estimates
         Using Multilevel Regression and Poststratification},
  author={Ferres, Leo},
  year={2026},
  note={Working paper}
}
```

## License

MIT
