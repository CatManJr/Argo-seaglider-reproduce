"""
utils/mld.py — Mixed Layer Depth (MLD) calculation.

Implements the density-threshold method from Dove et al. (2021),
as used in Song et al.:

  "shallowest depth where σ₀ exceeds the 10 m value by 0.05 kg m⁻³"

The MLD is used for:
  - Mixed-layer mean nitrate: N_ML = ∫₀^MLD nitrate dz / MLD
  - Nitrate jump at MLD base: ΔN_ML = N_ML − mean(nitrate, MLD+10 → MLD+30)
  - Horizontal nitrate variance: s²_H,NO₃ in 10-profile overlapping windows
"""

import numpy as np
import pandas as pd


def mixed_layer_depth(sigma0, pressure, threshold=0.05, ref_level=10.0):
    """
    Compute MLD from a single profile's σ₀ and pressure.

    Parameters
    ----------
    sigma0    : array-like, potential density anomaly [kg m⁻³]
    pressure  : array-like, pressure [dbar] (same length)
    threshold : float, density increase threshold [kg m⁻³] (default 0.05)
    ref_level : float, reference depth [dbar] (default 10)

    Returns
    -------
    mld : float, mixed layer depth [dbar] (or NaN if not found)
    """
    sigma0 = np.asarray(sigma0, dtype=float)
    pressure = np.asarray(pressure, dtype=float)

    # Sort by pressure
    order = np.argsort(pressure)
    p_sorted = pressure[order]
    s_sorted = sigma0[order]

    # Find reference density at ~10 dbar
    idx_ref = np.argmin(np.abs(p_sorted - ref_level))
    sigma_ref = s_sorted[idx_ref]

    # Find first depth where σ₀ exceeds ref + threshold
    exceed = np.where(s_sorted > sigma_ref + threshold)[0]
    if len(exceed) == 0:
        return np.nan

    return float(p_sorted[exceed[0]])


def compute_mld_dataframe(df, sigma0_col="sigma0", pressure_col="pressure",
                          profid_col="profid", threshold=0.05):
    """
    Compute MLD for each profile in a DataFrame.

    Parameters
    ----------
    df            : pd.DataFrame with sigma0, pressure, profid columns
    sigma0_col    : column name for potential density
    pressure_col  : column name for pressure
    profid_col    : column name for profile identifier
    threshold     : density threshold [kg m⁻³]

    Returns
    -------
    pd.DataFrame indexed by profid with column 'MLD' [dbar]
    """
    mlds = {}
    for pid, grp in df.groupby(profid_col):
        mld = mixed_layer_depth(
            grp[sigma0_col].values,
            grp[pressure_col].values,
            threshold=threshold,
        )
        mlds[pid] = mld

    result = pd.DataFrame({"MLD": pd.Series(mlds)})
    result.index.name = profid_col
    return result


def mixed_layer_mean_nitrate(df, mld_df, nitrate_col="nitrate",
                              pressure_col="pressure", profid_col="profid"):
    """
    Compute N_ML (mixed-layer mean nitrate) for each profile.

    N_ML = ∫₀^MLD nitrate(z) dz / MLD

    Parameters
    ----------
    df      : pd.DataFrame with nitrate, pressure, profid
    mld_df  : pd.DataFrame indexed by profid with 'MLD' column
    Returns
    -------
    pd.Series of N_ML [µmol kg⁻¹] indexed by profid
    """
    n_ml = {}
    for pid, grp in df.groupby(profid_col):
        mld = mld_df.loc[pid, "MLD"] if pid in mld_df.index else np.nan
        if np.isnan(mld):
            n_ml[pid] = np.nan
            continue
        in_ml = grp[grp[pressure_col] <= mld]
        if len(in_ml) == 0:
            n_ml[pid] = np.nan
        else:
            n_ml[pid] = in_ml[nitrate_col].mean()
    return pd.Series(n_ml, name="N_ML")


def nitrate_jump_ml_base(df, n_ml_series, mld_df,
                          nitrate_col="nitrate", pressure_col="pressure",
                          profid_col="profid", dz=20):
    """
    Compute ΔN_ML = N_ML − mean(nitrate, MLD+10 → MLD+30 dbar).

    Positive ΔN_ML → nitrate higher in mixed layer (injection).
    Negative ΔN_ML → nitrate higher below mixed layer (drawdown).

    Parameters
    ----------
    df          : pd.DataFrame with nitrate, pressure, profid
    n_ml_series : pd.Series of N_ML indexed by profid
    mld_df      : pd.DataFrame indexed by profid with 'MLD'
    dz          : thickness of sub-MLD averaging band [dbar] (default 20)

    Returns
    -------
    pd.Series of ΔN_ML [µmol kg⁻¹]
    """
    delta = {}
    for pid, grp in df.groupby(profid_col):
        mld = mld_df.loc[pid, "MLD"] if pid in mld_df.index else np.nan
        nml = n_ml_series.get(pid, np.nan)
        if np.isnan(mld) or np.isnan(nml):
            delta[pid] = np.nan
            continue
        below = grp[(grp[pressure_col] > mld + 10) &
                    (grp[pressure_col] <= mld + 10 + dz)]
        if len(below) == 0:
            delta[pid] = np.nan
        else:
            delta[pid] = nml - below[nitrate_col].mean()
    return pd.Series(delta, name="delta_N_ML")


def horizontal_nitrate_variance(n_ml_series, window=10):
    """
    Compute s²_H,NO₃ — rolling horizontal variance of N_ML.

    Uses overlapping windows of `window` consecutive profiles (~2 days
    for Seaglider at 2-3h resolution). Captures small-scale stirring.

    Parameters
    ----------
    n_ml_series : pd.Series of N_ML, indexed by profile order
    window      : int, number of profiles per window

    Returns
    -------
    pd.Series of s²_H,NO₃ [µmol² kg⁻²]
    """
    return n_ml_series.rolling(window=window, center=True, min_periods=3).var()


# ── Quick self-test ────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic profile: Southern Ocean late fall
    p = np.linspace(5, 200, 50)
    # Stratified: surface density ~27.0, deepens to ~27.5
    sigma0 = 27.0 + 0.05 * (p - 5) + np.random.randn(50) * 0.005
    mld = mixed_layer_depth(sigma0, p)
    print(f"MLD test: {mld:.1f} dbar (expected ~15 for threshold 0.05)")

    # Profile-level MLD
    df = pd.DataFrame({
        "profid": ["p1"] * 50 + ["p2"] * 50,
        "pressure": list(p) + list(p),
        "sigma0": list(sigma0) + list(27.2 + 0.08 * (p - 5)),
        "nitrate": list(25 + 0.02 * p) + list(24 + 0.03 * p),
    })
    mlds = compute_mld_dataframe(df)
    print(f"Profile MLDs:\n{mlds}")
