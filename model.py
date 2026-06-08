"""
model.py — Random Forest Regression for Southern Ocean nitrate inference.

Implements Song et al. (submitted) methodology
  Step 1  — RFR training (1000 trees, max_features=1/3, min_samples_leaf=5)
         — Triple validation: holdout 20%, k-fold (k=10), spatial LOO
         — Profiles kept intact during splitting
  Step 2  — Independent testing on SOGOS float + GO-SHIP I07
  Step 3  — Benchmark comparison (CANYON-B, ESPER-Mixed)
  Step 4  — Application to Seagliders SG659/SG660

Usage:
  uv run python model.py --train          # train & triple-validate
  uv run python model.py --evaluate       # independent test evaluation
  uv run python model.py --predict        # apply to gliders
  uv run python model.py --benchmark      # CANYON-B / ESPER comparison
  uv run python model.py --all            # full pipeline
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"
ZENODO = ROOT / "data" / "zenodo_main"
OUTPUT = ROOT / "data" / "model_output"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Paper hyperparameters ────────────────────────
RFR_PARAMS = dict(
    n_estimators=1000,
    max_features=1 / 3,          # "1/3 of total features at each node"
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    oob_score=True,
    verbose=0,
)

# Validation folds use fewer trees for speed
RFR_VAL_PARAMS = {**RFR_PARAMS, "n_estimators": 200, "oob_score": False, "verbose": 0}

FEATURE_COLS = [
    "CT", "SA", "pressure", "oxygen",
    "lat", "lon", "yearday", "ydcos", "ydsin",
]
TARGET_COL = "nitrate"
UNCERTAINTY = 0.5   # µmol kg⁻¹, float measurement uncertainty


# ── Data Loading ───────────────────────────────────────────────────
def load_processed_data():
    """Load preprocessed training & test data."""
    train_X = pd.read_csv(PROCESSED / "training_X.csv")
    train_y = pd.read_csv(PROCESSED / "training_y.csv").squeeze("columns")
    test_X = pd.read_csv(PROCESSED / "test_X.csv")
    test_y = pd.read_csv(PROCESSED / "test_y.csv").squeeze("columns")

    # Build profile-group arrays for profile-aware splitting
    argo = pd.read_csv(ZENODO / "bgcArgo_datasetB_10float_ASZ_2017-2021.csv")
    argo_train = argo[argo["wmoid"] != 5906030]
    goship = pd.read_csv(ZENODO / "goship_dataset_i06_i07_ASZ_2019.csv")
    goship_i06 = goship[goship["cruise_id"] == "i06"]

    # Paper: spatial LOO withholds one entire FLOAT → use wmoid
    n_argo = len(argo_train)
    train_groups = np.concatenate([
        argo_train["wmoid"].values[:n_argo],
        ["goship_i06"] * len(goship_i06),
    ])
    train_groups = train_groups[:len(train_X)]
    print(f"  Training groups: {len(np.unique(train_groups))} (9 floats + GO-SHIP I06)")

    print(f"  Training: X={train_X.shape}, y={train_y.shape}")
    print(f"  Test:     X={test_X.shape}, y={test_y.shape}")
    return train_X, train_y, test_X, test_y, train_groups


# ── Profile-aware splitting (paper: profiles kept intact) ──────────
def profile_holdout_split(X, y, groups, test_size=0.2, random_state=42):
    """Holdout 20% — entire profiles kept together."""
    unique = np.unique(groups)
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * test_size))
    test_set = set(unique[:n_test])
    mask = np.array([g in test_set for g in groups])
    return (X[~mask], X[mask], y[~mask], y[mask],
            groups[~mask], groups[mask])


def profile_kfold_cv(X, y, groups, n_splits=10):
    """k-fold (k=10) — GroupKFold keeps profiles intact."""
    return list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))


def spatial_leave_one_out(X, y, groups):
    """Spatial LOO — withhold one float at a time."""
    splits = []
    for g in np.unique(groups):
        val = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        splits.append((tr, val))
    return splits


# ── Evaluation Metrics (paper Table 4) ─────────────────────────────
def evaluate(y_true, y_pred, label=""):
    """MAE, IQR-AE, % within ±0.5, mean bias, 95% error bounds."""
    ae = np.abs(y_pred - y_true)
    err = y_pred - y_true
    mae = np.median(ae)
    iqr_ae = np.percentile(ae, 75) - np.percentile(ae, 25)
    within = np.mean(ae <= UNCERTAINTY) * 100
    bias = np.mean(err)
    lo, hi = np.percentile(err, 2.5), np.percentile(err, 97.5)

    print(f"\n  ── {label} ──")
    print(f"  MAE (median):            {mae:.4f} µmol kg⁻¹")
    print(f"  IQR-AE:                  {iqr_ae:.4f} µmol kg⁻¹")
    print(f"  % within ±{UNCERTAINTY}:          {within:.1f}%")
    print(f"  Mean bias:               {bias:+.4f} µmol kg⁻¹")
    print(f"  95% error bounds:        [{lo:.4f}, {hi:.4f}]")
    return dict(MAE=mae, IQR_AE=iqr_ae, pct_within=within,
                mean_bias=bias, err_lo=lo, err_hi=hi)


# ── RFR Training ───────────────────────────────────────────────────
def train_rfr(X, y, verbose=True):
    """Train RFR with paper hyperparameters, return model + OOB score."""
    if verbose:
        print(f"\n  Training RFR ({RFR_PARAMS['n_estimators']} trees, "
              f"max_features=1/3, min_samples_leaf=5)...")
        t0 = time.time()
    rf = RandomForestRegressor(**RFR_PARAMS)
    rf.fit(X, y)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s  |  OOB R² = {rf.oob_score_:.4f}")
    return rf


# ── Triple Validation (Table 3, Step 5) ────────────────────────────
def triple_validation(X, y, groups):
    """Holdout 20% + k-Fold (k=10) + Spatial leave-one-out."""
    print("\n" + "=" * 55)
    print("  Triple Validation (Table 3, Step 5)")
    print("=" * 55)
    Xa, ya = X.values, y.values
    results = {}

    # (1) Holdout 20%
    print("\n  ── (1) Holdout 20% (profiles intact) ──")
    X_tr, X_v, y_tr, y_v, g_tr, g_v = profile_holdout_split(Xa, ya, groups)
    rf = train_rfr(X_tr, y_tr, verbose=False)
    results["holdout"] = evaluate(y_v, rf.predict(X_v), "Holdout 20%")

    # (2) k-Fold
    print("\n  ── (2) k-Fold CV (k=10, profiles intact) ──")
    kf_splits = profile_kfold_cv(Xa, ya, groups)
    kf_ae = []
    for i, (tr_i, val_i) in enumerate(kf_splits):
        print(f"    fold {i+1}/10...", end=" ", flush=True)
        rf_k = RandomForestRegressor(**RFR_VAL_PARAMS)
        rf_k.fit(Xa[tr_i], ya[tr_i])
        kf_ae.extend(np.abs(rf_k.predict(Xa[val_i]) - ya[val_i]))
        print(f"MAE={np.median(np.abs(rf_k.predict(Xa[val_i]) - ya[val_i])):.3f}")
    kf_ae = np.array(kf_ae)
    print(f"\n  ── k-Fold (combined) ──")
    print(f"  MAE: {np.median(kf_ae):.4f}  IQR-AE: "
          f"{np.percentile(kf_ae,75)-np.percentile(kf_ae,25):.4f}  "
          f"% within ±0.5: {np.mean(kf_ae<=UNCERTAINTY)*100:.1f}%")
    results["kfold"] = {"MAE": np.median(kf_ae)}

    # (3) Spatial LOO
    print("\n  ── (3) Spatial Leave-One-Out (per float) ──")
    loo_splits = spatial_leave_one_out(Xa, ya, groups)
    loo_ae = []
    unique_g = np.unique(groups)
    for (tr_i, val_i), gname in zip(loo_splits, unique_g):
        if len(val_i) < 5:
            continue
        print(f"    {str(gname)[:25]:26s} n={len(val_i):5d}...", end=" ", flush=True)
        rf_l = RandomForestRegressor(**RFR_VAL_PARAMS)
        rf_l.fit(Xa[tr_i], ya[tr_i])
        err = np.abs(rf_l.predict(Xa[val_i]) - ya[val_i])
        loo_ae.extend(err)
        print(f"MAE={np.median(err):.3f}")
    loo_ae = np.array(loo_ae)
    print(f"\n  ── Spatial LOO (combined) ──")
    print(f"  MAE: {np.median(loo_ae):.4f}  IQR-AE: "
          f"{np.percentile(loo_ae,75)-np.percentile(loo_ae,25):.4f}  "
          f"% within ±0.5: {np.mean(loo_ae<=UNCERTAINTY)*100:.1f}%")
    results["spatial_loo"] = {"MAE": np.median(loo_ae)}

    return results


# ── Independent Testing (Table 3, Step 7) ──────────────────────────
def independent_test(rf, X_test, y_test):
    """Evaluate final model on withheld data."""
    print("\n" + "=" * 55)
    print("  Independent Testing (Table 3, Step 7)")
    print("=" * 55)
    Xa = X_test.values
    y_pred = rf.predict(Xa)
    evaluate(y_test, y_pred, "Combined (SOGOS float + I07)")

    # Per-subset
    sogos_n = len(pd.read_csv(ZENODO / "sogos_float5906030_RFRtest_setB_modG.csv"))
    sogos_n = min(sogos_n, len(X_test))  # safety
    if sogos_n > 0 and sogos_n < len(X_test):
        evaluate(y_test.iloc[:sogos_n], y_pred[:sogos_n], "SOGOS Float WMO 5906030")
        evaluate(y_test.iloc[sogos_n:], y_pred[sogos_n:], "GO-SHIP I07")


# ── Application to Seagliders (Table 3, Step 9) ────────────────────
def apply_to_gliders(rf):
    """Apply trained RFR to SG659 and SG660 high-resolution data."""
    print("\n" + "=" * 55)
    print("  Application to Seagliders (Table 3, Step 9)")
    print("=" * 55)

    for glider_tag, glider_name in [("sg659", "SG659"), ("sg660", "SG660")]:
        csv = ZENODO / f"sogos_glider{glider_tag[-3:]}_RFRpred_setB_modG.csv"
        if not csv.exists():
            print(f"  ⚠ {csv} not found")
            continue
        print(f"\n  ── {glider_name} ──")
        df = pd.read_csv(csv)
        X = df[FEATURE_COLS].copy()
        valid = ~X.isna().any(axis=1)
        Xc = X[valid]
        print(f"    {len(X)} obs → {len(Xc)} valid (filtered NaN features)")

        y_pred = rf.predict(Xc.values)
        print(f"    nitrate range: {y_pred.min():.2f} – {y_pred.max():.2f} µmol kg⁻¹")

        # Save
        out = pd.DataFrame({"nitrate_RFR": y_pred})
        out_path = OUTPUT / f"{glider_tag}_nitrate_RFR.csv"
        out.to_csv(out_path, index=False)
        print(f"    → {out_path}")


# ── Main ───────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]

    if not args or "--all" in args:
        do_train = do_eval = do_predict = do_bench = True
    else:
        do_train = "--train" in args
        do_eval = "--evaluate" in args
        do_predict = "--predict" in args
        do_bench = "--benchmark" in args

    print("=" * 55)
    print("  Song et al. RFR Model")
    print(f"  {RFR_PARAMS['n_estimators']} trees, max_features=1/3, min_samples_leaf=5")
    print("=" * 55)

    train_X, train_y, test_X, test_y, train_groups = load_processed_data()

    if do_train:
        triple_validation(train_X, train_y, train_groups)
        print("\n" + "=" * 55)
        print("  Training Final RFR on Full Training Set")
        print("=" * 55)
        rf = train_rfr(train_X.values, train_y.values)
        import joblib
        joblib.dump(rf, OUTPUT / "sogos_rfr_model.pkl")
        print(f"  → Saved: {OUTPUT / 'sogos_rfr_model.pkl'}")

    if do_eval:
        import joblib
        rf = joblib.load(OUTPUT / "sogos_rfr_model.pkl")
        independent_test(rf, test_X, test_y)

    if do_predict:
        import joblib
        rf = joblib.load(OUTPUT / "sogos_rfr_model.pkl")
        apply_to_gliders(rf)

    if do_bench:
        print("\n" + "=" * 55)
        print("  Benchmark Comparison (Table 3, Step 8)")
        print("=" * 55)
        print("  CANYON-B: https://github.com/HCBScienceProducts/CANYON-B")
        print("  ESPER:    https://github.com/BRCScienceProducts/ESPER")
        print("  (Install packages above to enable benchmark)")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()

