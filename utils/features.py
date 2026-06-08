"""
utils/features.py — Feature engineering for Song et al. RFR pipeline.

Feature engineering for the RFR pipeline:
  - Seasonal cyclical encoding: sz₁ = cos(2π·yd/365), sz₂ = sin(2π·yd/365)
  - Final 9-feature selection: Θ, SA, p, O₂, lat, lon, time, sz₁, sz₂
  - Features *screened out*: σ₀, τ, N², O₂,sat (redundant with core 9)

Also handles yearday conversion from datetime timestamps.
"""

import numpy as np
import pandas as pd

# ── Final RFR feature set ──────────────────────────────────────────
RFR_FEATURES = [
    "CT",       # Conservative Temperature Θ  [°C]
    "SA",       # Absolute Salinity            [g kg⁻¹]
    "pressure", # Pressure / depth             [dbar]
    "oxygen",   # Dissolved oxygen             [µmol kg⁻¹]
    "lat",      # Latitude                     [°N]
    "lon",      # Longitude                    [°E]
    "yearday",  # Time since Jan 1 of reference year [days]
    "ydcos",    # sz₁ = cos(2π·yd/365)
    "ydsin",    # sz₂ = sin(2π·yd/365)
]

RFR_TARGET = "nitrate"  # [NO₃⁻]  [µmol kg⁻¹]

# Features screened out as redundant:
SCREENED_OUT = ["sigma0", "spice", "N2", "O2sat"]


# ── Seasonal Encoding ──────────────────────────────────────────────
def seasonal_encode(yearday):
    """
    Cyclical encoding of day-of-year for seasonality.

    Seasonal encoding improves upper 100 m performance.

    Parameters
    ----------
    yearday : float or array-like
        Days elapsed since Jan 1 of reference year.
        Paper uses Jan 1, 2019 as reference (p.12).

    Returns
    -------
    sz1, sz2 : tuple of arrays
        sz₁ = cos(2π·yd/365), sz₂ = sin(2π·yd/365)
    """
    yd = np.asarray(yearday, dtype=float)
    sz1 = np.cos(2 * np.pi * yd / 365.0)
    sz2 = np.sin(2 * np.pi * yd / 365.0)
    return sz1, sz2


# ── Yearday Conversion ─────────────────────────────────────────────
def datetime_to_yearday(dt_series, ref_year=2019):
    """
    Convert datetime to yearday (fractional days since Jan 1 of ref_year).

    Parameters
    ----------
    dt_series : pd.Series of datetime-like
    ref_year  : int, reference year (paper uses 2019)

    Returns
    -------
    pd.Series of float (yearday)
    """
    ref = pd.Timestamp(f"{ref_year}-01-01")
    return (pd.to_datetime(dt_series) - ref).dt.total_seconds() / 86400.0


# ── Feature Selection ──────────────────────────────────────────────
def select_rfr_features(df, target_col="nitrate"):
    """
    Extract the 9 RFR feature columns + target from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain CT, SA, pressure, oxygen, lat, lon, yearday
        (or ydcos/ydsin will be computed).

    Returns
    -------
    X : pd.DataFrame  (9 columns)
    y : pd.Series or None
    """
    # Ensure seasonal encoding exists
    if "ydcos" not in df.columns or "ydsin" not in df.columns:
        if "yearday" in df.columns:
            df["ydcos"], df["ydsin"] = seasonal_encode(df["yearday"])
        else:
            raise KeyError("Need 'yearday' column to compute ydcos/ydsin")

    available = [c for c in RFR_FEATURES if c in df.columns]
    missing = set(RFR_FEATURES) - set(available)
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    X = df[RFR_FEATURES].copy()

    y = None
    if target_col in df.columns:
        y = df[target_col].copy()

    return X, y


# ── Feature Screening Info ─────────────────────────────────────────
def get_feature_info():
    """Return a DataFrame describing all features considered in the paper."""
    info = pd.DataFrame([
        {"feature": "CT",       "role": "selected",  "reason": "Core thermodynamic variable"},
        {"feature": "SA",       "role": "selected",  "reason": "Core salinity variable"},
        {"feature": "pressure", "role": "selected",  "reason": "Vertical coordinate"},
        {"feature": "oxygen",   "role": "selected",  "reason": "Biological proxy, ventilation tracer"},
        {"feature": "lat",      "role": "selected",  "reason": "Meridional gradients"},
        {"feature": "lon",      "role": "selected",  "reason": "Zonal gradients"},
        {"feature": "yearday",  "role": "selected",  "reason": "Temporal evolution"},
        {"feature": "ydcos",    "role": "selected",  "reason": "Seasonal encoding (cos)"},
        {"feature": "ydsin",    "role": "selected",  "reason": "Seasonal encoding (sin)"},
        {"feature": "sigma0",   "role": "excluded",  "reason": "Redundant with Θ, SA, p"},
        {"feature": "spice",    "role": "excluded",  "reason": "Redundant; used in CWT analysis only"},
        {"feature": "N2",       "role": "excluded",  "reason": "Redundant (log-transformed in screening)"},
        {"feature": "O2sat",    "role": "excluded",  "reason": "Redundant with Θ, SA, p, O₂"},
    ])
    return info


# ── Quick self-test ────────────────────────────────────────────────
if __name__ == "__main__":
    # Test seasonal encoding
    yd = np.array([0, 91, 182, 273, 365])
    sz1, sz2 = seasonal_encode(yd)
    print("Seasonal encoding test (yd=0,91,182,273,365):")
    for i in range(len(yd)):
        print(f"  yd={yd[i]:4.0f}  sz1={sz1[i]:+.4f}  sz2={sz2[i]:+.4f}")

    # Test feature selection
    df = pd.DataFrame({
        "CT": [2.0], "SA": [34.5], "pressure": [100],
        "oxygen": [300], "lat": [-55], "lon": [40],
        "yearday": [150], "nitrate": [25.0],
        "sigma0": [27.0], "spice": [0.1],  # screened out
    })
    X, y = select_rfr_features(df)
    print(f"\nRFR features ({X.shape[1]}): {list(X.columns)}")
    print(f"Target: {y.name if y is not None else 'None'}")
    print("\nFeature roster:")
    print(get_feature_info().to_string(index=False))
