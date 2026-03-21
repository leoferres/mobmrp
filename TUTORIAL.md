# mobmrp tutorial

This tutorial walks through the full pipeline for correcting socioeconomic bias in mobile phone mobility data using MRP.

## The problem

A single mobile carrier's user base is not representative of the general population. Some socioeconomic groups are over- or under-represented, which distorts aggregate mobility statistics. `mobmrp` corrects this by fitting a Bayesian multilevel model on the biased data, then reweighting predictions to match the true census population.

## What you need

Three datasets, loaded as pandas DataFrames:

### 1. CDR data

Individual-level mobility records from the carrier. One row per user (or per user-month, user-day, etc.). You need at minimum:

- A **mobility metric** (e.g. radius of gyration, entropy, number of unique locations)
- A **socioeconomic group** label per user (e.g. income quintile, assigned from the area where they live)
- **Gender**
- A **geographic unit** identifier (e.g. municipality, district) for the multilevel model's random intercept
- A **fine geographic unit** (e.g. census tract, block group) for the poststratification frame

Optional columns: month, weekday/weekend indicator, lat/lon (if you need a spatial join).

Example:

```
| redcode      | comuna | gse  | genero | month | is_weekday | avgROG |
|--------------|--------|------|--------|-------|------------|--------|
| 13101011001  | 13101  | ABC1 | M      | 5     | True       | 12.3   |
| 13101011001  | 13101  | C2   | F      | 5     | False      |  8.7   |
| 13402041002  | 13402  | D    | M      | 6     | True       | 22.1   |
```

### 2. Geographic proportions

One row per fine geographic unit. Contains the **proportion** of each socioeconomic group in that area, derived from census or survey data.

```
| redcode      | comuna | total_pop | ABC1  | C2    | C3    | D     | E     |
|--------------|--------|-----------|-------|-------|-------|-------|-------|
| 13101011001  | 13101  | 1200      | 0.31  | 0.15  | 0.25  | 0.24  | 0.05  |
| 13402041002  | 13402  | 800       | 0.08  | 0.16  | 0.49  | 0.26  | 0.01  |
```

- `total_pop`: total population (households or persons) in the area
- The socioeconomic columns must sum to ~1.0 per row

### 3. Gender proportions

One row per geographic unit (the parent unit). Contains the proportion of each gender.

```
| comuna | M    | F    |
|--------|------|------|
| 13101  | 0.51 | 0.49 |
| 13402  | 0.49 | 0.51 |
```

These two proportion tables are combined under a conditional independence assumption to build the full poststratification frame:

```
N(socioeco, gender, area) = total_pop(area) * P(socioeco | area) * P(gender | parent_area)
```

## Step-by-step walkthrough

### Step 0: Install

```bash
pip install -e .                # from the mobmrp directory
pip install numpyro jax         # optional, for faster sampling
```

### Step 1: Configure

Tell the library what your columns are called:

```python
from mobmrp import MRPConfig, ColumnMap

config = MRPConfig(
    columns=ColumnMap(
        outcome="avgROG",               # your mobility metric
        socioeconomic_group="gse",       # socioeconomic classification
        gender="genero",                 # gender column
        geographic_unit="comuna",        # for the random intercept
        fine_geographic_unit="redcode",  # finest census geography
        month="month",                   # optional
        weekday="is_weekday",            # optional
    ),
    socioeco_levels=["ABC1", "C2", "C3", "D", "E"],
    gender_levels=["M", "F"],
    log_transform=True,                  # log-transform the outcome
    reference_conditions={               # temporal reference for predictions
        "month": 5,
        "is_weekday": True,
    },
    chains=4,
    draws=2000,
    inference_method="nuts_numpyro",     # or "mcmc" without JAX
)
```

### Step 2: Load and prepare CDR data

```python
import pandas as pd
from mobmrp import load_and_filter, aggregate_to_cells

cdr_raw = pd.read_csv("my_cdr_data.csv")
cdr = load_and_filter(cdr_raw, config, filters={"nationality": "CHILEAN"})
```

`load_and_filter` validates that the required columns exist, drops rows with missing values in key fields, and applies any filters you pass.

Then aggregate to cell level (one row per unique combination of area, socioeconomic group, gender, and temporal indicators):

```python
cells = aggregate_to_cells(cdr, config)
# Result: ~42,000 cells with columns:
#   redcode, comuna, gse, genero, month, is_weekday, avgROG, n_users, log_avgROG
```

### Step 2b (optional): Spatial join

If your CDR data has raw lat/lon coordinates but no geographic identifiers yet:

```python
from mobmrp import nearest_neighbor_join

geo_lookup = pd.read_csv("antenna_locations.csv")  # with lat, lon, redcode, comuna
cdr = nearest_neighbor_join(cdr, geo_lookup, config)
```

This uses a KD-tree to match each CDR record to the nearest point in the lookup table, dropping matches beyond 5 km (configurable via `config.max_join_distance_km`).

### Step 3: Build the poststratification frame

```python
from mobmrp import build_poststrat_frame

geo_proportions = pd.read_csv("gse_proportions_by_tract.csv")
gender_proportions = pd.read_csv("gender_proportions_by_district.csv")

ps_frame = build_poststrat_frame(geo_proportions, gender_proportions, config)
# Result: ~7,000 cells with columns:
#   redcode, comuna, gse, genero, census_pop
```

Each row represents a demographic cell in the census population. The `census_pop` column is the estimated number of people in that cell.

If you have person-level census microdata instead of pre-computed proportions, use `build_poststrat_frame_from_microdata()`.

### Step 4: Fit the model

```python
from mobmrp import fit_mrp_model, check_convergence

model, idata = fit_mrp_model(cells, config)

conv = check_convergence(idata)
print(conv)
# {'max_rhat': 1.001, 'min_ess_bulk': 1029, 'min_ess_tail': 1670,
#  'n_divergences': 0, 'converged': True}
```

The model fitted is:

```
log(avgROG) ~ C(gse) + C(genero) + C(gse):C(genero)
            + C(is_weekday) + C(month)
            + (1 | comuna)
```

You can override the formula via `config.formula` or the `formula` argument.

To save/reload the trace:

```python
from mobmrp import save_trace, load_trace

save_trace(idata, "my_trace.nc")
idata = load_trace("my_trace.nc")
```

### Step 5: Predict and poststratify

```python
from mobmrp import generate_predictions, compare_estimates

pred_frame = generate_predictions(model, idata, ps_frame, config)
```

This produces posterior predictions for every cell in the poststratification frame, back-transforms from log scale, and attaches 200 posterior draws (as columns `draw_0` ... `draw_199`) for uncertainty propagation.

Now compare naive CDR averages with MRP-corrected estimates:

```python
results = compare_estimates(pred_frame, cells, config)

for level, df in results.items():
    print(f"\n--- {level} ---")
    print(df.to_string(index=False, float_format="%.2f"))
```

## Output

`compare_estimates` returns a dictionary with five tables:

**`overall`**: single-row population-level estimate

```
 naive_mean  mrp_mean  mrp_lo  mrp_hi   shift
      25.51     18.81   18.52   19.14   -6.70
```

**`by_gse`**: one row per socioeconomic group

```
  gse   naive_mean  mrp_mean  mrp_lo  mrp_hi   shift
  ABC1       28.59     18.46   18.03   18.87  -10.13
  C2         26.73     20.76   20.33   21.20   -5.96
  C3         25.33     19.97   19.50   20.45   -5.37
  D          24.58     18.39   18.02   18.79   -6.19
  E          22.17     11.97   11.66   12.26  -10.20
```

**`by_genero`**: one row per gender

**`by_gse_x_genero`**: one row per socioeconomic group x gender combination

**`by_comuna`**: one row per geographic unit

In every table:
- `naive_mean` is the raw, unweighted CDR average (biased)
- `mrp_mean` is the census-corrected estimate
- `mrp_lo` / `mrp_hi` is the 94% credible interval
- `shift` = `mrp_mean - naive_mean` (how much the correction changes the estimate)

## Optional: validation

### Cross-validation

```python
from mobmrp import grouped_cross_validate

cv = grouped_cross_validate(cells, config)
print(cv)
#   fold  n_test_units  rmse_log  coverage
#      1            10     0.877     0.335
#      2            10     0.881     0.285
#      ...
```

Holds out 20% of geographic units per fold, fits on the rest, predicts the held-out units.

### Model comparison

```python
from mobmrp import compare_specifications

sens = compare_specifications(cells, config)
print(sens)
#                 model   loo_elpd   loo_se  p_loo
#        no_interaction  -52869.5    180.9   51.9
#      with_interaction  -52869.8    180.8   55.4
#   no_random_intercept  -53537.3    178.4   14.6
```

Compares three model specifications via LOO-CV. The random intercept is clearly essential (removing it costs ~668 ELPD points).

## Using the results in your own work

Once you have the output from `compare_estimates` or `poststratify`, there are several ways to put it to use.

### Report corrected estimates instead of raw averages

Whenever you would normally report "the average radius of gyration in district X is Y km", use `mrp_mean` instead of the naive CDR average, and report the credible interval (`mrp_lo`, `mrp_hi`). This is the main point of the library: replacing biased aggregates with census-corrected ones.

### Feed corrected values into downstream analyses

If your CDR-derived mobility metric is an input to another model (e.g. predicting commute flows, estimating exposure to pollution, calibrating an epidemic simulation), use `pred_frame` as your data source. Each row is a demographic-geographic cell with a corrected prediction and full posterior uncertainty. You can aggregate to whatever spatial or demographic resolution your downstream model requires:

```python
from mobmrp import poststratify

# Custom aggregation: by district and socioeconomic group
custom = poststratify(pred_frame, config, group_cols=["comuna", "gse"])
custom = summarize_draws(custom, hdi_prob=0.94)
```

### Propagate uncertainty

The 200 posterior draws attached to `pred_frame` (columns `draw_0` ... `draw_199`) are not just for computing credible intervals. You can propagate them through any downstream calculation. For example, if you are computing a ratio between two groups:

```python
draws = [f"draw_{i}" for i in range(200)]

mrp_male = poststratify(pred_frame, config, group_cols=["genero"])
mrp_male = mrp_male[mrp_male["genero"] == "M"][draws].values[0]

mrp_female = poststratify(pred_frame, config, group_cols=["genero"])
mrp_female = mrp_female[mrp_female["genero"] == "F"][draws].values[0]

ratio_draws = mrp_male / mrp_female
print(f"Male/Female mobility ratio: {ratio_draws.mean():.2f} "
      f"[{np.percentile(ratio_draws, 3):.2f}, {np.percentile(ratio_draws, 97):.2f}]")
```

This gives you a proper posterior distribution over the ratio, not just a point estimate.

### Quantify the bias

The `shift` column in the comparison tables tells you how far off the naive estimate is. Large shifts indicate strong compositional bias. You can use this to:

- Argue that raw CDR averages should not be taken at face value in a given study area
- Identify which subgroups or areas are most affected by the carrier's selection bias
- Decide whether correction is worth the effort for your application (if shifts are negligible, maybe the naive estimate is fine)

### Apply to other mobility metrics

The library is not specific to radius of gyration. You can run the same pipeline on entropy, number of unique locations, or any other metric derived from CDR. Just change `config.columns.outcome`:

```python
config_entropy = MRPConfig(
    columns=ColumnMap(outcome="avgS", ...),  # everything else the same
    ...
)
```

Fit a separate model for each metric and compare the corrections.

## Summary

```
CDR data + Census proportions
        |
        v
  load_and_filter()          -- clean and validate
  aggregate_to_cells()       -- group to demographic cells
  build_poststrat_frame()    -- census population by cell
        |
        v
  fit_mrp_model()            -- Bayesian multilevel regression
  generate_predictions()     -- posterior predictions on census frame
        |
        v
  compare_estimates()        -- naive vs corrected at every level
```
