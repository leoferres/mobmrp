"""
Chile replication: reproduce the paper results using mobmrp.

This script replicates the analysis from:

    "Correcting Socioeconomic Bias in Mobile Phone Mobility Estimates
     Using Multilevel Regression and Poststratification"

Requirements:
    - CDR data file (not included; provided by Telefonica Chile)
    - TEF-census geographic join file
    - 2017 Chilean census microdata (censo2017_personas.tar.gz)

Adjust DATA_DIR below to point to your data location.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import tarfile

from mobmrp import (
    MRPConfig,
    ColumnMap,
    load_and_filter,
    aggregate_to_cells,
    nearest_neighbor_join,
    build_poststrat_frame,
    build_poststrat_frame_from_microdata,
    fit_mrp_model,
    generate_predictions,
    compare_estimates,
    check_convergence,
    save_trace,
)


# ---------------------------------------------------------------------------
# Paths -- adjust these to your data location
# ---------------------------------------------------------------------------
DATA_DIR = Path("../../data")

CDR_PATH = DATA_DIR / "cdr2016_results_by_gse_gender_foreign_avg1callperday.csv"
TEF_CENSUS_PATH = DATA_DIR / "tef_census_join.csv"
CENSUS_TAR_PATH = DATA_DIR / "census" / "censo2017_personas.tar.gz"
OUTPUT_DIR = DATA_DIR

GSE_LEVELS = ["ABC1", "C2", "C3", "D", "E"]
REGION_MAP = {
    "REGION METROPOLITANA": "Santiago",
    "QUINTA REGION": "Valparaiso",
    "OCTAVA REGION": "Biobio",
}


def main():
    # --- Configuration ---
    config = MRPConfig(
        columns=ColumnMap(
            outcome="avgROG",
            socioeconomic_group="gse",
            gender="genero",
            geographic_unit="comuna",
            fine_geographic_unit="redcode",
            lat="home_lat",
            lon="home_lon",
            region="home_region",
            month="month",
            weekday="is_weekday",
        ),
        socioeco_levels=GSE_LEVELS,
        gender_levels=["M", "F"],
        log_transform=True,
        reference_conditions={"month": 5, "is_weekday": True},
        chains=4,
        draws=2000,
        inference_method="nuts_numpyro",
        random_seed=42,
    )

    # -----------------------------------------------------------------------
    # Step 1: Load and filter CDR
    # -----------------------------------------------------------------------
    print("Loading CDR data...")
    cdr_raw = pd.read_csv(CDR_PATH)
    cdr = load_and_filter(
        cdr_raw, config,
        filters={"nacionalidad": "CHILENO"},
    )
    cdr["region_short"] = cdr["home_region"].map(REGION_MAP)
    print(f"  CDR after filtering: {len(cdr):,} rows")

    # -----------------------------------------------------------------------
    # Step 2: Spatial join to get geographic identifiers
    # -----------------------------------------------------------------------
    print("Spatial join with geographic lookup...")
    tef = pd.read_csv(TEF_CENSUS_PATH)
    tef["redcode"] = tef["redcode"].astype(str).str.replace(r"\.0$", "", regex=True)
    tef["comuna"] = tef["comuna"].astype(str).str.replace(r"\.0$", "", regex=True)

    # Deduplicate to one row per antenna location
    geo_cols = [
        "home_lat", "home_lon", "home_region",
        "region", "comuna", "redcode",
        "ABC1", "C2", "C3", "D", "E", "NSE",
    ]
    geo_lookup = tef[geo_cols].drop_duplicates(
        subset=["home_lat", "home_lon", "home_region"]
    )

    cdr = nearest_neighbor_join(cdr, geo_lookup, config)
    print(f"  CDR after spatial join: {len(cdr):,} rows")

    # -----------------------------------------------------------------------
    # Step 3: Aggregate to cells
    # -----------------------------------------------------------------------
    cells = aggregate_to_cells(cdr, config)
    cells["comuna"] = cells["comuna"].astype(str).str.replace(r"\.0$", "", regex=True)
    cells["redcode"] = cells["redcode"].astype(str).str.replace(r"\.0$", "", regex=True)
    print(f"  Cells: {len(cells):,}")

    # -----------------------------------------------------------------------
    # Step 4: Build poststratification frame
    # -----------------------------------------------------------------------
    print("Building poststratification frame...")

    # GSE proportions per redcode from ISMT
    redcode_info = (
        tef[["redcode", "comuna", "home_region"] + GSE_LEVELS + ["NSE"]]
        .drop_duplicates(subset=["redcode"])
        .copy()
    )
    redcode_info["comuna"] = redcode_info["comuna"].astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    redcode_info["redcode"] = redcode_info["redcode"].astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    for g in GSE_LEVELS:
        redcode_info[g] = redcode_info[g].astype(float) / redcode_info["NSE"].astype(float)
    redcode_info = redcode_info.rename(columns={"NSE": "total_pop"})

    # Gender proportions from census microdata
    print("  Reading census microdata...")
    regions_of_interest = {5, 8, 13}
    chunks = []
    with tarfile.open(CENSUS_TAR_PATH, "r:gz") as tar:
        f = tar.extractfile("censo2017_personas.csv")
        for chunk in pd.read_csv(f, sep=";", usecols=["REGION", "COMUNA", "P08"], chunksize=500_000):
            chunk = chunk[chunk["REGION"].isin(regions_of_interest)]
            chunks.append(chunk)
    census = pd.concat(chunks, ignore_index=True)
    census["COMUNA"] = census["COMUNA"].astype(str)
    census["genero"] = census["P08"].map({1: "M", 2: "F"})

    gender_by_comuna = (
        census.groupby(["COMUNA", "genero"]).size().unstack(fill_value=0)
    )
    total = gender_by_comuna.sum(axis=1)
    gender_props = (gender_by_comuna.div(total, axis=0)).reset_index()
    gender_props = gender_props.rename(columns={"COMUNA": "comuna"})

    ps_frame = build_poststrat_frame(redcode_info, gender_props, config)
    print(f"  Poststrat frame: {len(ps_frame):,} cells, "
          f"total pop: {ps_frame['census_pop'].sum():,.0f}")

    # -----------------------------------------------------------------------
    # Step 5: Fit the model
    # -----------------------------------------------------------------------
    print("\nFitting MRP model (this may take a few minutes)...")
    model, idata = fit_mrp_model(cells, config)

    conv = check_convergence(idata)
    print(f"  Converged: {conv['converged']} "
          f"(R-hat: {conv['max_rhat']:.3f}, "
          f"ESS bulk: {conv['min_ess_bulk']:.0f}, "
          f"divergences: {conv['n_divergences']})")

    save_trace(idata, OUTPUT_DIR / "mrp_trace_full.nc")

    # -----------------------------------------------------------------------
    # Step 6: Predict and poststratify
    # -----------------------------------------------------------------------
    print("\nGenerating predictions and poststratifying...")
    pred_frame = generate_predictions(model, idata, ps_frame, config)

    # Filter cell_data to reference conditions for naive estimates
    cells_ref = cells[
        (cells["month"] == 5) & (cells["is_weekday"] == True)
    ].copy()

    results = compare_estimates(pred_frame, cells_ref, config)

    print("\n" + "=" * 60)
    for level, df in results.items():
        print(f"\n--- {level} ---")
        print(df.to_string(index=False, float_format="%.2f"))

    # Save results
    for level, df in results.items():
        df.to_csv(OUTPUT_DIR / f"mrp_comparison_{level}.csv", index=False)
    pred_frame.to_csv(OUTPUT_DIR / "mrp_predictions.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
