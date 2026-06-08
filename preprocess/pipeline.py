"""
preprocess/pipeline.py — Full preprocessing pipeline for Song et al. RFR.

Replicates Table 3 preprocessing steps:
  Step 2  — TEOS-10 variable calculation
  Step 3  — Mixed layer depth calculation
  Step 4  — Feature engineering & seasonal encoding
  Step 5  — (model.py) RFR training

Input:  Raw CSV from Zenodo (bgcArgo, GO-SHIP, SOGOS float, Seaglider)
Output: RFR-ready feature matrix X and target y

Usage:
  uv run python preprocess/pipeline.py
  uv run python preprocess/pipeline.py --all       # process all datasets
  uv run python preprocess/pipeline.py --training  # training data only
  uv run python preprocess/pipeline.py --test      # test data only
  uv run python preprocess/pipeline.py --gliders   # glider application data
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.features import (
    RFR_FEATURES,
    RFR_TARGET,
    SCREENED_OUT,
    seasonal_encode,
    datetime_to_yearday,
    select_rfr_features,
    get_feature_info,
)
from utils.teos10 import compute_all_teos10
from utils.mld import (
    mixed_layer_depth,
    compute_mld_dataframe,
    mixed_layer_mean_nitrate,
    nitrate_jump_ml_base,
)

# ── Paths & Constants ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "zenodo_main"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper: "Upper 1000 m limit" (Table 3, Step 1)
PRES_MAX = 1000

# Paper: "SOGOS float (WMO 5906030) was deliberately withheld from
#  training" (p.12, l.256–258)
WMO_SOGOS = 5906030

# Zenodo file mapping — training/test split matches paper Table 1
DATASETS = {
    "training": {
        "argo": {
            "path": DATA_DIR / "bgcArgo_datasetB_10float_ASZ_2017-2021.csv",
            "filter": lambda df: df[df["wmoid"] != WMO_SOGOS],
        },
        "goship_i06": {
            "path": DATA_DIR / "goship_dataset_i06_i07_ASZ_2019.csv",
            "filter": lambda df: df[df["cruise_id"] == "i06"],
        },
        "description": "Training: 9 BGC-Argo floats (excl. 5906030) + GO-SHIP I06",
    },
    "test": {
        "sogos_float": {
            "path": DATA_DIR / "sogos_float5906030_RFRtest_setB_modG.csv",
            "filter": None,
        },
        "goship_i07": {
            "path": DATA_DIR / "goship_dataset_i06_i07_ASZ_2019.csv",
            "filter": lambda df: df[df["cruise_id"] == "i07"],
        },
        "description": "Test: SOGOS float 5906030 + GO-SHIP I07 (withheld)",
    },
    "gliders": {
        "sg659": {
            "path": DATA_DIR / "sogos_glider659_RFRpred_setB_modG.csv",
            "filter": None,
        },
        "sg660": {
            "path": DATA_DIR / "sogos_glider660_RFRpred_setB_modG.csv",
            "filter": None,
        },
        "description": "Application: Seagliders SG659 + SG660",
    },
}


# ── Step 1: Load & QC (paper Table 3, Step 1) ─────────────────────
def load_dataset(csv_path, label, row_filter=None):
    """
    Load Zenodo CSV and perform QC matching paper Table 3, Step 1:
      - Upper 1000 dbar limit
      - Remove NaN in critical fields
      - Apply dataset-specific filter (exclude SOGOS float, split I06/I07)
    """
    print(f"  Loading {label}: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Remove unnamed index columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    before = len(df)

    # Paper: "Upper 1000 m limit" (Table 3, Step 1)
    if "pressure" in df.columns:
        over = (df["pressure"] > PRES_MAX).sum()
        if over:
            df = df[df["pressure"] <= PRES_MAX]
            print(f"    QC: removed {over} obs > {PRES_MAX} dbar")

    # Dataset-specific filter (exclude SOGOS float, split I06/I07)
    if row_filter is not None:
        before_f = len(df)
        df = row_filter(df)
        removed = before_f - len(df)
        if removed:
            print(f"    Filter: removed {removed} obs")

    # QC: remove NaN in critical fields
    critical = ["pressure", "CT", "SA", "oxygen", "nitrate"]
    available_critical = [c for c in critical if c in df.columns]
    df = df.dropna(subset=available_critical)
    after = len(df)
    if before != after:
        print(f"    QC: {before - after} total removed ({after} remain)")

    print(f"    {len(df)} obs, {len(df.columns)} columns")
    return df


# ── Step 2: TEOS-10 (if raw T/S present) ───────────────────────────
def apply_teos10(df, label):
    """Compute TEOS-10 variables if not already present."""
    # Zenodo data already has CT, SA, sigma0, spice — skip if present
    if all(c in df.columns for c in ["CT", "SA", "sigma0", "spice"]):
        print(f"  {label}: TEOS-10 variables already present — skipping")
        return df

    # For raw data, compute from T/S/p
    t_col = "temp" if "temp" in df.columns else "T"
    sp_col = "psal" if "psal" in df.columns else "S"

    if t_col in df.columns and sp_col in df.columns:
        print(f"  {label}: Computing TEOS-10 from {t_col}/{sp_col}...")
        df = compute_all_teos10(df, t_col=t_col, sp_col=sp_col)
    else:
        print(f"  {label}: No raw T/S columns found, skipping TEOS-10")
    return df


# ── Step 3: MLD ────────────────────────────────────────────────────
def apply_mld(df, label):
    """Compute MLD and mixed-layer nitrate metrics."""
    profid_col = "profid" if "profid" in df.columns else None
    if profid_col is None or "sigma0" not in df.columns:
        print(f"  {label}: Cannot compute MLD (missing profid or sigma0)")
        return df, None

    print(f"  {label}: Computing MLD...")
    mld_df = compute_mld_dataframe(df, profid_col=profid_col)
    print(f"    {len(mld_df)} profiles, MLD range: "
          f"{mld_df['MLD'].min():.1f} – {mld_df['MLD'].max():.1f} dbar")

    if "nitrate" in df.columns:
        n_ml = mixed_layer_mean_nitrate(df, mld_df, profid_col=profid_col)
        delta_n = nitrate_jump_ml_base(df, n_ml, mld_df, profid_col=profid_col)
        mld_df["N_ML"] = n_ml
        mld_df["delta_N_ML"] = delta_n

    return df, mld_df


# ── Step 4: Feature Engineering ────────────────────────────────────
def apply_features(df, label):
    """Compute seasonal encoding and select RFR features."""
    # Ensure yearday exists (Zenodo data has "yearday")
    if "yearday" not in df.columns and "juld" in df.columns:
        print(f"  {label}: Converting juld → yearday...")
        df["yearday"] = datetime_to_yearday(df["juld"], ref_year=2019)

    if "yearday" not in df.columns:
        print(f"  {label}: WARNING — no yearday column, skipping seasonal encoding")
        return df, None, None

    # Compute seasonal encoding
    if "ydcos" not in df.columns or "ydsin" not in df.columns:
        print(f"  {label}: Computing seasonal encoding (sz₁, sz₂)...")
        df["ydcos"], df["ydsin"] = seasonal_encode(df["yearday"])

    # Select RFR features
    try:
        X, y = select_rfr_features(df, target_col=RFR_TARGET)
    except KeyError as e:
        # Target not present (e.g., glider prediction data)
        if "nitrate" in df.columns:
            X, y = select_rfr_features(df)
        elif "nitrate_G" in df.columns:
            # Glider data uses "nitrate_G" for RFR predictions
            X = df[RFR_FEATURES].copy()
            y = df["nitrate_G"].copy() if "nitrate_G" in df.columns else None
        else:
            X = df[[c for c in RFR_FEATURES if c in df.columns]].copy()
            y = None

    print(f"  {label}: X={X.shape}, y={'nitrate' if y is not None else 'None'}")

    return df, X, y


# ── Main Pipeline ──────────────────────────────────────────────────
def process_dataset(name, info):
    """Run full pipeline on one dataset group."""
    print(f"\n{'=' * 55}")
    print(f"  {info['description']}")
    print(f"{'=' * 55}")

    all_X = []
    all_y = []
    all_mld = []

    for key, spec in info.items():
        if key == "description":
            continue
        path = spec["path"]
        row_filter = spec.get("filter", None)

        if not path.exists():
            print(f"  ⚠ File not found: {path}")
            continue

        # Step 1: Load & QC (upper 1000 m, dataset-specific filter)
        df = load_dataset(path, key, row_filter=row_filter)

        # Step 2: TEOS-10
        df = apply_teos10(df, key)

        # Step 3: MLD
        df, mld_df = apply_mld(df, key)

        # Step 4: Features
        df, X, y = apply_features(df, key)

        if X is not None:
            all_X.append(X)
        if y is not None:
            all_y.append(y)
        if mld_df is not None:
            all_mld.append(mld_df)

    # Concatenate
    X_out = pd.concat(all_X, ignore_index=True) if all_X else None
    y_out = pd.concat(all_y, ignore_index=True) if all_y else None
    mld_out = pd.concat(all_mld, ignore_index=False) if all_mld else None

    if X_out is not None:
        print(f"\n  Combined: X={X_out.shape}, y={y_out.shape if y_out is not None else 'N/A'}")

    return X_out, y_out, mld_out


# ── Main ───────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]

    # Show feature info
    if "--info" in args:
        print("Feature roster for Song et al. RFR pipeline:\n")
        print(get_feature_info().to_string(index=False))
        return

    # Determine which datasets to process
    selected = [k for k in DATASETS if f"--{k}" in args]
    if not selected:
        selected = list(DATASETS.keys())

    print("=" * 55)
    print("  Song et al. RFR Preprocessing Pipeline")
    print(f"  Processing: {', '.join(selected)}")
    print("=" * 55)

    for name in selected:
        info = DATASETS[name]
        X, y, mld = process_dataset(name, info)

        if X is not None:
            X.to_csv(OUTPUT_DIR / f"{name}_X.csv", index=False)
            print(f"  → Saved: {OUTPUT_DIR / f'{name}_X.csv'}")
        if y is not None:
            y.to_csv(OUTPUT_DIR / f"{name}_y.csv", index=False, header=True)
            print(f"  → Saved: {OUTPUT_DIR / f'{name}_y.csv'}")
        if mld is not None:
            mld.to_csv(OUTPUT_DIR / f"{name}_MLD.csv")
            print(f"  → Saved: {OUTPUT_DIR / f'{name}_MLD.csv'}")

    print(f"\n✓ Pipeline complete. Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
