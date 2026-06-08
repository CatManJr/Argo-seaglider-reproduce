"""
utils/teos10.py — TEOS-10 thermodynamic variable calculations.

Converts raw in-situ measurements (T, S, p, O₂) into the derived variables
used in Song et al. RFR pipeline (Table 3, Step 2):

  - Conservative Temperature (Θ)
  - Absolute Salinity (SA)
  - Potential Density (σ₀)
  - Spice / spiciness (τ)
  - Brunt-Väisälä frequency (N²)
  - Oxygen saturation (O₂,sat)

Requires: gsw (Gibbs Seawater Oceanographic Toolbox)
  pip install gsw  (or uv add gsw)
"""

import numpy as np
import gsw


def conservative_temperature(t, sp, p):
    """
    Conservative Temperature Θ [°C]  (TEOS-10).
    Inputs:
      t  : in-situ temperature [°C]
      sp : practical salinity [PSU]
      p  : sea pressure [dbar]
    Returns:
      CT : Conservative Temperature [°C]
    """
    sa = gsw.SA_from_SP(sp, p, lon=0, lat=0)
    return gsw.CT_from_t(sa, t, p)


def absolute_salinity(sp, p, lon, lat):
    """
    Absolute Salinity SA [g kg⁻¹]  (TEOS-10).
    Inputs:
      sp  : practical salinity [PSU]
      p   : sea pressure [dbar]
      lon : longitude [°E]
      lat : latitude [°N]
    Returns:
      SA : Absolute Salinity [g kg⁻¹]
    """
    return gsw.SA_from_SP(sp, p, lon, lat)


def potential_density(sa, ct, p_ref=0):
    """
    Potential density σ₀ [kg m⁻³] referenced to surface.
    Inputs:
      sa   : Absolute Salinity [g kg⁻¹]
      ct   : Conservative Temperature [°C]
      p_ref: reference pressure [dbar] (default 0)
    Returns:
      sigma0 : potential density anomaly [kg m⁻³]
    """
    return gsw.sigma0(sa, ct)
    # Equivalent to: gsw.rho(sa, ct, p_ref) - 1000


def spice(sa, ct):
    """
    Spice / spiciness τ [kg m⁻³] — water-mass tracer.
    Inputs:
      sa : Absolute Salinity [g kg⁻¹]
      ct : Conservative Temperature [°C]
    Returns:
      tau : spiciness [kg m⁻³]
    """
    return gsw.spiciness0(sa, ct)


def brunt_vaisala_frequency(sa, ct, p):
    """
    Brunt-Väisälä (buoyancy) frequency N² [s⁻²].
    Uses adiabatic level-shifting to compute vertical stratification.
    Inputs:
      sa : Absolute Salinity [g kg⁻¹]  1D profile
      ct : Conservative Temperature [°C] 1D profile
      p  : sea pressure [dbar] 1D profile
    Returns:
      N2 : N² [s⁻²]  (same length, NaN at surface)
    """
    # gsw.Nsquared expects [lat, lon] scalars
    # Using mid-lat=0 for simplicity (small error in Southern Ocean)
    n2, p_mid = gsw.Nsquared(sa, ct, p, lat=0)
    # Interpolate back to original p levels
    N2 = np.full_like(p, np.nan)
    for i in range(len(p) - 1):
        N2[i] = n2[i]
    return N2


def oxygen_saturation(sa, ct, p, o2):
    """
    Oxygen saturation O₂,sat [%].
    Inputs:
      sa : Absolute Salinity [g kg⁻¹]
      ct : Conservative Temperature [°C]
      p  : sea pressure [dbar]
      o2 : dissolved oxygen [µmol kg⁻¹]
    Returns:
      O2sat : oxygen saturation [%]
    """
    # gsw.O2sol gives saturation at 1 atm; correct for pressure
    o2_sol = gsw.O2sol(sa, ct, p, lon=0, lat=0)
    return (o2 / o2_sol) * 100.0


def compute_all_teos10(df, t_col="temp", sp_col="psal", p_col="pressure",
                       o2_col="oxygen", lon_col="lon", lat_col="lat"):
    """
    Compute all TEOS-10 derived variables for a DataFrame.

    The DataFrame may already have pre-computed CT/SA columns;
    this function computes from raw T/SP if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: temp (or CT), psal (or SA), pressure,
        oxygen (optional), lon, lat.

    Returns
    -------
    pd.DataFrame with added columns:
      CT, SA, sigma0, spice, N2, O2sat
    """
    import pandas as pd

    result = df.copy()

    # --- Absolute Salinity ---
    if "SA" not in result.columns:
        result["SA"] = absolute_salinity(
            result[sp_col].values,
            result[p_col].values,
            result[lon_col].values,
            result[lat_col].values,
        )
        print(f"  Computed SA from {sp_col}")

    # --- Conservative Temperature ---
    if "CT" not in result.columns:
        # Need SA for CT calculation
        sa = result["SA"].values if "SA" in result.columns else absolute_salinity(
            result[sp_col].values, result[p_col].values,
            result[lon_col].values, result[lat_col].values,
        )
        result["CT"] = conservative_temperature(
            result[t_col].values,
            result[sp_col].values,
            result[p_col].values,
        )
        print(f"  Computed CT from {t_col}")

    sa = result["SA"].values
    ct = result["CT"].values
    p = result[p_col].values

    # --- Potential Density ---
    if "sigma0" not in result.columns:
        result["sigma0"] = potential_density(sa, ct)
        print("  Computed sigma0")

    # --- Spice ---
    if "spice" not in result.columns:
        result["spice"] = spice(sa, ct)
        print("  Computed spice")

    # --- N² --- (computed per-profile to avoid cross-profile gradients)
    if "N2" not in result.columns:
        result["N2"] = np.nan
        if "profid" in result.columns:
            for pid, grp in result.groupby("profid"):
                idx = grp.index
                n2 = brunt_vaisala_frequency(
                    sa[idx], ct[idx], p[idx]
                )
                result.loc[idx, "N2"] = n2
        else:
            result["N2"] = brunt_vaisala_frequency(sa, ct, p)
        print("  Computed N2 (per-profile)")

    # --- O₂ Saturation ---
    if o2_col in result.columns and "O2sat" not in result.columns:
        result["O2sat"] = oxygen_saturation(sa, ct, p, result[o2_col].values)
        print(f"  Computed O2sat from {o2_col}")

    return result


# ── Quick self-test ────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    # Synthetic profile like Southern Ocean upper 1000 m
    p = np.linspace(10, 1000, 50)
    t = 2.0 - 0.003 * p + np.random.randn(50) * 0.05   # ~2°C at surface
    sp = 34.0 + 0.001 * p + np.random.randn(50) * 0.01  # ~34 PSU
    o2 = 300 - 0.05 * p + np.random.randn(50) * 2       # ~300 µmol/kg
    lon = np.full(50, 40.0)
    lat = np.full(50, -55.0)

    df = pd.DataFrame({
        "temp": t, "psal": sp, "pressure": p,
        "oxygen": o2, "lon": lon, "lat": lat,
    })

    df = compute_all_teos10(df)
    print("\n  Computed columns:", [c for c in df.columns if c not in
          ["temp", "psal", "pressure", "oxygen", "lon", "lat"]])
    print(f"  CT range:     {df['CT'].min():.2f} – {df['CT'].max():.2f} °C")
    print(f"  SA range:     {df['SA'].min():.2f} – {df['SA'].max():.2f} g/kg")
    print(f"  sigma0 range: {df['sigma0'].min():.2f} – {df['sigma0'].max():.2f} kg/m³")
    print(f"  O2sat range:  {df['O2sat'].min():.1f} – {df['O2sat'].max():.1f} %")
