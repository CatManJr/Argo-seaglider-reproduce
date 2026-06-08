"""
fetch_goship.py — Download GO-SHIP I06 and I07 cruise data from CCHDO.

Source: GO-SHIP program (http://www.go-ship.org/)
Data:   CCHDO (https://cchdo.ucsd.edu/)

From the paper's Data Availability Statement:
  I06 (2019): https://cchdo.ucsd.edu/cruise/325020190403  (training)
  I07 (2019): https://cchdo.ucsd.edu/cruise/49NZ20191229  (independent test)

Usage:
  uv run python fetch_goship.py
"""

from pathlib import Path
import requests
import time
import json

DATA_DIR = Path(__file__).parent / "goship"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Exact cruise IDs from the paper's Data Availability Statement
CRUISES = {
    "goship_I06_2019": {
        "cruise_id": "325020190403",
        "role": "Training data (16 stations, 246 bottle obs)",
        "url": "https://cchdo.ucsd.edu/cruise/325020190403",
    },
    "goship_I07_2019": {
        "cruise_id": "49NZ20191229",
        "role": "Independent test (8 stations, 96 bottle obs)",
        "url": "https://cchdo.ucsd.edu/cruise/49NZ20191229",
    },
}


def fetch_cruise(label, cruise_id, role, cruise_url):
    """Download bottle data for a GO-SHIP cruise from CCHDO."""
    print(f"\n── {label} ──")
    print(f"  Cruise ID: {cruise_id}")
    print(f"  Role:      {role}")
    print(f"  URL:       {cruise_url}")

    # CCHDO provides bottle data as CSV/NetCDF via direct download.
    # The file naming convention:
    #   https://cchdo.ucsd.edu/data/<first2>/<cruise_id>/<cruise_id>_hy1.csv
    prefix = cruise_id[:2]

    patterns = [
        # Bottle summary (CSV)
        f"https://cchdo.ucsd.edu/data/{prefix}/{cruise_id}/{cruise_id}_hy1.csv",
        # Bottle NetCDF
        f"https://cchdo.ucsd.edu/data/{prefix}/{cruise_id}/{cruise_id}_btl.nc",
        # CTD NetCDF
        f"https://cchdo.ucsd.edu/data/{prefix}/{cruise_id}/{cruise_id}_ct1.nc",
    ]

    downloaded = False
    for url in patterns:
        if downloaded:
            break
        fname = url.split("/")[-1]
        fpath = DATA_DIR / f"{label}_{fname}"
        if fpath.exists():
            print(f"  Already exists: {fname}")
            downloaded = True
            break
        try:
            print(f"  GET {url}")
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code == 200:
                with open(fpath, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                mb = fpath.stat().st_size / 1e6
                print(f"  ✓ {mb:.2f} MB → {fpath}")
                downloaded = True
            else:
                print(f"  → HTTP {resp.status_code}")
        except Exception as e:
            print(f"  ✗ {e}")
        time.sleep(0.5)

    if not downloaded:
        print(f"  ⚠ Manual download: {cruise_url}")


def main():
    print("=" * 55)
    print("  GO-SHIP Data Fetcher — Song et al. RFR Pipeline")
    print("  Source: http://www.go-ship.org/ | CCHDO")
    print("=" * 55)

    for label, info in CRUISES.items():
        fetch_cruise(label, info["cruise_id"], info["role"], info["url"])
        time.sleep(1)

    print("\n✓ GO-SHIP fetch complete.")
    print("  If downloads failed, visit:")
    for _, info in CRUISES.items():
        print(f"    {info['url']}")


if __name__ == "__main__":
    main()
