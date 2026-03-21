"""
Synthetic example: run the full mobmrp pipeline on generated data.

No external data required. This script creates fake CDR + census data
and demonstrates every step from data preparation to poststratification.
"""

import numpy as np
import pandas as pd

from mobmrp import (
    MRPConfig,
    ColumnMap,
    load_and_filter,
    aggregate_to_cells,
    build_poststrat_frame,
    fit_mrp_model,
    generate_predictions,
    compare_estimates,
    check_convergence,
)


def make_synthetic_data(
    n_users: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate fake CDR, geographic proportions, and gender proportions."""
    rng = np.random.default_rng(seed)

    gse_levels = ["high", "mid_high", "mid", "mid_low", "low"]
    genders = ["M", "F"]
    comunas = [f"comuna_{i:02d}" for i in range(10)]
    redcodes = [f"rc_{i:03d}" for i in range(30)]

    # Map redcodes to comunas (3 redcodes per comuna)
    rc_to_com = {rc: comunas[i // 3] for i, rc in enumerate(redcodes)}

    # True mobility by GSE (high-income travels more)
    gse_effect = {"high": 2.5, "mid_high": 2.8, "mid": 3.0, "mid_low": 2.7, "low": 2.0}
    gender_effect = {"M": 0.2, "F": -0.2}
    comuna_effect = {c: rng.normal(0, 0.3) for c in comunas}

    # CDR has biased sampling: over-represents mid_high, under-represents high
    cdr_gse_weights = {"high": 0.10, "mid_high": 0.30, "mid": 0.25, "mid_low": 0.25, "low": 0.10}

    rows = []
    for _ in range(n_users):
        gse = rng.choice(gse_levels, p=[cdr_gse_weights[g] for g in gse_levels])
        gender = rng.choice(genders)
        redcode = rng.choice(redcodes)
        comuna = rc_to_com[redcode]

        log_rog = (
            gse_effect[gse]
            + gender_effect[gender]
            + comuna_effect[comuna]
            + rng.normal(0, 0.8)
        )
        rows.append({
            "redcode": redcode,
            "comuna": comuna,
            "gse": gse,
            "gender": gender,
            "month": rng.choice([5, 6, 7]),
            "is_weekday": bool(rng.choice([True, False])),
            "rog": np.exp(log_rog),
        })

    cdr = pd.DataFrame(rows)

    # Census proportions: true population has more "high" and "low" than CDR
    geo_rows = []
    for rc in redcodes:
        com = rc_to_com[rc]
        total = rng.integers(500, 2000)
        props = rng.dirichlet([3, 2, 2, 2, 1])  # different from CDR weights
        geo_rows.append({
            "redcode": rc,
            "comuna": com,
            "total_pop": total,
            "high": props[0],
            "mid_high": props[1],
            "mid": props[2],
            "mid_low": props[3],
            "low": props[4],
        })
    geo_proportions = pd.DataFrame(geo_rows)

    # Gender proportions by comuna
    gender_rows = []
    for com in comunas:
        pf = rng.uniform(0.48, 0.52)
        gender_rows.append({"comuna": com, "M": 1 - pf, "F": pf})
    gender_proportions = pd.DataFrame(gender_rows)

    return cdr, geo_proportions, gender_proportions


def main():
    # --- Configuration ---
    config = MRPConfig(
        columns=ColumnMap(
            outcome="rog",
            socioeconomic_group="gse",
            gender="gender",
            geographic_unit="comuna",
            fine_geographic_unit="redcode",
            month="month",
            weekday="is_weekday",
        ),
        socioeco_levels=["high", "mid_high", "mid", "mid_low", "low"],
        gender_levels=["M", "F"],
        log_transform=True,
        reference_conditions={"month": 5, "is_weekday": True},
        # Use lighter sampling for the demo
        chains=2,
        draws=500,
        inference_method="mcmc",  # avoid numpyro dependency for demo
        random_seed=42,
    )

    # --- Generate data ---
    print("Generating synthetic data...")
    cdr_raw, geo_proportions, gender_proportions = make_synthetic_data()
    print(f"  CDR: {len(cdr_raw)} rows")

    # --- Step 1: Filter and aggregate ---
    cdr = load_and_filter(cdr_raw, config)
    cells = aggregate_to_cells(cdr, config)
    print(f"  Cells: {len(cells)}")

    # --- Step 2: Build poststratification frame ---
    ps_frame = build_poststrat_frame(geo_proportions, gender_proportions, config)
    print(f"  Poststrat frame: {len(ps_frame)} cells, "
          f"total pop: {ps_frame['census_pop'].sum():,.0f}")

    # --- Step 3: Fit model ---
    print("\nFitting MRP model...")
    model, idata = fit_mrp_model(cells, config)

    conv = check_convergence(idata)
    print(f"  Converged: {conv['converged']} "
          f"(R-hat: {conv['max_rhat']:.3f}, "
          f"ESS: {conv['min_ess_bulk']:.0f}, "
          f"divergences: {conv['n_divergences']})")

    # --- Step 4: Predict and poststratify ---
    print("\nGenerating predictions...")
    pred_frame = generate_predictions(model, idata, ps_frame, config)
    print(f"  Predictions for {len(pred_frame)} cells")

    # --- Step 5: Compare ---
    print("\nComparing naive vs MRP-corrected estimates:\n")
    results = compare_estimates(pred_frame, cells, config)

    for level, df in results.items():
        print(f"--- {level} ---")
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
