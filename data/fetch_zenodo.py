"""
fetch_zenodo.py — Download from Song et al. Zenodo data releases.

Paper's Data Availability Statement lists three Zenodo DOIs:

  1. Song 2025 (main data archive) ⭐ PRIMARY:
     https://doi.org/10.5281/zenodo.17508960
     Contents:
       - bgcArgo_datasetB_10float_ASZ_2017-2021.csv  (5.1 MB) — training data
       - goship_dataset_i06_i07_ASZ_2019.csv           (0.1 MB) — GO-SHIP
       - sogos_float5906030_RFRtest_setB_modG.csv      (0.8 MB) — test data
       - sogos_glider659_RFRpred_setB_modG.csv       (185.0 MB) — SG659 RFR output
       - sogos_glider660_RFRpred_setB_modG.csv       (203.1 MB) — SG660 RFR output

  2. Song 2025 (RFR nitrate estimates):
     https://doi.org/10.5281/zenodo.14510704

  3. Balwada 2023 (processed Seaglider datasets):
     https://doi.org/10.5281/zenodo.8361656

Usage:
  uv run python fetch_zenodo.py           # all 3 records
  uv run python fetch_zenodo.py --main    # main data only
  uv run python fetch_zenodo.py --nitrate # RFR nitrate only
  uv run python fetch_zenodo.py --glider  # glider data only
"""

import sys
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent
DATA_DIR.mkdir(parents=True, exist_ok=True)

RECORDS = {
    "main": {
        "doi": "10.5281/zenodo.17508960",
        "record_id": "17508960",
        "description": "Song 2025 — Main RFR data archive",
        "dir": DATA_DIR / "zenodo_main",
    },
    "nitrate": {
        "doi": "10.5281/zenodo.14510704",
        "record_id": "14510704",
        "description": "Song 2025 — SOGOS RFR nitrate estimates",
        "dir": DATA_DIR / "zenodo_nitrate",
    },
    "glider": {
        "doi": "10.5281/zenodo.8361656",
        "record_id": "8361656",
        "description": "Balwada 2023 — Processed Seaglider datasets",
        "dir": DATA_DIR / "zenodo_glider",
    },
}


def download_file(url, dest, fsize, max_retries=3):
    """Download a file with progress bar, resume, and retry."""
    if dest.exists() and dest.stat().st_size == fsize:
        print(f"    ✓ already complete ({fsize/1e6:.1f} MB)")
        return True

    # Resume from partial download
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if existing > 0 and existing < fsize:
        headers["Range"] = f"bytes={existing}-"
        print(f"    Resuming from {existing/1e6:.1f} MB...")

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=300, stream=True)
            resp.raise_for_status()

            mode = "ab" if "Range" in headers else "wb"
            total = fsize
            initial = existing if mode == "ab" else 0

            with open(dest, mode) as f:
                with tqdm(
                    total=total, initial=initial, unit="B",
                    unit_scale=True, unit_divisor=1024,
                    desc=f"    {dest.name}",
                ) as pbar:
                    for chunk in resp.iter_content(1024 * 1024):  # 1 MB chunks
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Verify
            if dest.stat().st_size == fsize:
                return True
            else:
                print(f"    ⚠ Size mismatch, retrying...")
                existing = dest.stat().st_size
                headers["Range"] = f"bytes={existing}-"

        except Exception as e:
            print(f"    ✗ Attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(5)

    return False


def fetch_record(key, info):
    """Download all files from a Zenodo record."""
    print(f"\n{'─' * 55}")
    print(f"  {info['description']}")
    print(f"  DOI: {info['doi']}")
    print(f"{'─' * 55}")

    info["dir"].mkdir(parents=True, exist_ok=True)

    api_url = f"https://zenodo.org/api/records/{info['record_id']}"

    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        record = resp.json()
    except Exception as e:
        print(f"  ✗ API: {e}")
        return

    files = record.get("files", [])
    print(f"  {len(files)} file(s)")

    # Sort: small files first
    files.sort(key=lambda f: f.get("size", 0))

    ok = 0
    for fmeta in files:
        fname = fmeta.get("key", "unknown")
        fsize = fmeta.get("size", 0)
        furl = fmeta.get("links", {}).get("self", "")
        if not furl:
            continue

        if download_file(furl, info["dir"] / fname, fsize):
            ok += 1

    print(f"  → {ok}/{len(files)} downloaded to {info['dir']}")


def main():
    args = sys.argv[1:]
    selected = [k for k in RECORDS if f"--{k}" in args]
    if not selected:
        selected = list(RECORDS.keys())

    print("=" * 55)
    print("  Zenodo Data Fetcher — Song et al. RFR Pipeline")
    print("=" * 55)

    for key in selected:
        fetch_record(key, RECORDS[key])

    print("\n✓ Zenodo fetch complete.")


if __name__ == "__main__":
    main()
