"""
fetch_argo.py — Download BGC-Argo float data for Song et al. RFR pipeline.

Source: Argo GDAC (https://doi.org/10.17882/42182)
  https://argo.ucsd.edu
  https://www.ocean-ops.org

Datasets for RFR:
  1. SOGOS float (WMO 5906030) — independent test (withheld from training)
  2. 9 standard BGC-Argo floats — training data
     Antarctic Southern Zone (5°E–65°E), May 2017 – Jul 2021, upper 1000 m

Uses argopy → Ifremer ERDDAP or GDAC.

Usage:
  uv run python fetch_argo.py
"""

from pathlib import Path
import time
import argopy

DATA_DIR = Path(__file__).parent / "argo"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Paper Data Availability Statement:
#   "Argo float data and metadata are from Global Data Assembly Centre
#    (Argo 2021; DOI 10.17882/42182)"
ARGO_DOI = "10.17882/42182"

# SOGOS float — deliberately withheld from training (p.12 l.256-258)
WMO_SOGOS = 5906030

# Antarctic Southern Zone domain
LON_MIN, LON_MAX = 5, 65
LAT_MIN, LAT_MAX = -65, -49
TIME_START = "2017-05-01"
TIME_END = "2021-07-31"
PRES_MAX = 1000


def _retry_load(fetcher, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Attempt {attempt}/{max_retries}...")
            ds = fetcher.load()
            n = ds.obs.shape[0]
            print(f"    ✓ {n} observations loaded")
            return ds
        except Exception as e:
            print(f"    ✗ {str(e)[:150]}")
            if attempt < max_retries:
                wait = attempt * 10
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
    return None


def _save(ds, label):
    nc = DATA_DIR / f"{label}.nc"
    ds.to_netcdf(nc)
    mb = nc.stat().st_size / 1e6
    print(f"  → {nc} ({mb:.1f} MB)")
    return nc


def fetch_float(wmo, label):
    """Fetch single BGC-Argo float by WMO."""
    print(f"\n── {label} (WMO {wmo}) ──")
    for src in ["erddap", "gdac"]:
        print(f"  Source: {src}")
        try:
            fetcher = argopy.DataFetcher(mode="standard", src=src)
            ds = _retry_load(fetcher.float(wmo))
            if ds is not None:
                return _save(ds, label)
        except Exception as e:
            print(f"    ✗ {src}: {e}")
    print(f"  ⚠ Failed to fetch {label}")
    return None


def fetch_region(label, t0, t1):
    """Fetch all BGC-Argo in Antarctic Southern Zone."""
    print(f"\n── {label} ──")
    print(f"  lon=[{LON_MIN},{LON_MAX}] lat=[{LAT_MIN},{LAT_MAX}] pres≤{PRES_MAX}")
    print(f"  {t0} → {t1}")
    for src in ["erddap", "gdac"]:
        print(f"  Source: {src}")
        try:
            fetcher = argopy.DataFetcher(mode="standard", src=src)
            ds = _retry_load(fetcher.region(
                [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, 0, PRES_MAX, t0, t1]
            ))
            if ds is not None:
                try:
                    wmos = sorted(set(str(w).strip() for w in ds["PLATFORM_NUMBER"].values))
                    print(f"  Floats ({len(wmos)}): {wmos}")
                except Exception:
                    pass
                return _save(ds, label)
        except Exception as e:
            print(f"    ✗ {src}: {e}")
    print(f"  ⚠ Failed to fetch {label}")
    return None


def main():
    print("=" * 55)
    print("  BGC-Argo Fetcher — Song et al. RFR Pipeline")
    print(f"  Source: Argo GDAC  {ARGO_DOI}")
    print("=" * 55)

    fetch_float(WMO_SOGOS, "sogos_float_5906030")
    fetch_region("training_bgc_argo", TIME_START, TIME_END)

    print("\n✓ Argo fetch complete.")


if __name__ == "__main__":
    main()
