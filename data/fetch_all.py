"""
fetch_all.py — Master orchestrator for Song et al. RFR data pipeline.

Downloads in priority order:
  1. Zenodo    — Author's data release (DOI 10.5281/zenodo.17508960) ⭐ BEST
  2. Argo      — BGC-Argo floats via Argo GDAC (DOI 10.17882/42182)
  3. GO-SHIP   — I06/I07 bottle data via CCHDO
  4. Seaglider — SG659/SG660 via NOAA NCEI or Zenodo

All URLs/DOIs extracted from the paper's Data Availability Statement.

Usage:
  uv run python fetch_all.py              # run ALL
  uv run python fetch_all.py --zenodo     # Zenodo only (recommended first)
  uv run python fetch_all.py --argo       # Argo only
  uv run python fetch_all.py --goship     # GO-SHIP only
  uv run python fetch_all.py --seaglider  # Seaglider only
  uv run python fetch_all.py --list       # list datasets
"""

import sys
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

FETCHERS = {
    "zenodo": {
        "script": "fetch_zenodo.py",
        "desc": "Main Zenodo data archive (Song 2025 + Balwada 2023)",
        "doi": "10.5281/zenodo.17508960 / 10.5281/zenodo.14510704",
    },
    "argo": {
        "script": "fetch_argo.py",
        "desc": "BGC-Argo floats — SOGOS (test) + 9 training floats",
        "doi": "10.17882/42182 (Argo GDAC)",
    },
    "goship": {
        "script": "fetch_goship.py",
        "desc": "GO-SHIP I06 (training) + I07 (independent test)",
        "doi": "CCHDO: 325020190403 / 49NZ20191229",
    },
    "seaglider": {
        "script": "fetch_seaglider.py",
        "desc": "SOGOS Seagliders SG659 + SG660",
        "doi": "NOAA NCEI: 0228185 / 0228187",
    },
}


def run_fetcher(name):
    info = FETCHERS[name]
    script = SCRIPTS_DIR / info["script"]
    if not script.exists():
        print(f"  ✗ Missing: {script}")
        return False

    print(f"\n{'#' * 60}")
    print(f"#  [{name}] {info['desc']}")
    print(f"#  Source: {info['doi']}")
    print(f"{'#' * 60}")

    try:
        result = subprocess.run(
            ["uv", "run", "python", str(script)],
            cwd=str(SCRIPTS_DIR),
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  ✗ {e}")
        return False


def list_datasets():
    print("=" * 60)
    print("  Song et al. (submitted) — RFR Dataset Summary")
    print("=" * 60)
    for name, info in FETCHERS.items():
        print(f"\n  [{name}] {info['desc']}")
        print(f"        DOI: {info['doi']}")
    print(f"\n  Data dir: {SCRIPTS_DIR}")
    print(f"  Inventory: DATA_INVENTORY.md\n")


def main():
    args = sys.argv[1:]

    if "--list" in args or "-l" in args:
        list_datasets()
        return

    selected = [k for k in FETCHERS if f"--{k}" in args]
    if not selected:
        selected = list(FETCHERS.keys())  # run all

    print("=" * 60)
    print("  Song et al. RFR Data Orchestrator")
    print("=" * 60)
    print(f"  Datasets: {', '.join(selected)}")
    print(f"  Target:   {SCRIPTS_DIR}")
    print()

    results = {}
    for name in selected:
        results[name] = run_fetcher(name)

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name:12s} → {'✓' if ok else '✗ FAILED'}")
    n = sum(results.values())
    print(f"\n  {n}/{len(results)} succeeded.")
    if n < len(results):
        print("  See DATA_INVENTORY.md for manual download URLs.")
    print()


if __name__ == "__main__":
    main()

