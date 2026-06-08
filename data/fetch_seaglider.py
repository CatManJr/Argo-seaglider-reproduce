"""
fetch_seaglider.py — Download SOGOS Seaglider data (SG659, SG660).

From the paper's Data Availability Statement:
  "The SOGOS Seaglider data are archived at NOAA's National Centers for
   Environmental Information and can be accessed at
   https://www.ncei.noaa.gov/archive/accession/0228185  (SG659)
   https://www.ncei.noaa.gov/archive/accession/0228187  (SG660)
   Processed glider datasets are also available through
   Balwada 2023 (DOI 10.5281/zenodo.8361656)"

Usage:
  uv run python fetch_seaglider.py
"""

from pathlib import Path
import requests
import time

DATA_DIR = Path(__file__).parent / "seaglider"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Exact NOAA NCEI accession URLs from the paper
GLIDERS = {
    "SG659": {
        "accession": "0228185",
        "url": "https://www.ncei.noaa.gov/archive/accession/0228185",
    },
    "SG660": {
        "accession": "0228187",
        "url": "https://www.ncei.noaa.gov/archive/accession/0228187",
    },
}

# Alternative: Zenodo (Balwada 2023)
ZENODO_BALWADA = "https://doi.org/10.5281/zenodo.8361656"


def fetch_ncei_accession(glider, accession, url):
    """Download from NOAA NCEI archive accession."""
    print(f"\n── {glider} ──")
    print(f"  Accession: {accession}")
    print(f"  URL:       {url}")

    # NOAA NCEI archive access API
    # Archive download: https://www.ncei.noaa.gov/archive/accession/<id>
    # Direct file listing: https://www.ncei.noaa.gov/archive/accession/<id>/format

    # Try to get file listing
    api_url = f"https://www.ncei.noaa.gov/archive/accession/{accession}"

    try:
        # First, get the archive landing page to find files
        resp = requests.get(api_url, timeout=30)
        if resp.status_code == 200:
            print(f"  ✓ Archive page accessible")
        else:
            print(f"  → HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ✗ {e}")

    # NCEI provides downloads via:
    # https://www.ncei.noaa.gov/data/oceans/ncei/archive/
    # The exact path depends on the archive structure.
    # For automation, use the accession download endpoint:
    dl_url = (
        f"https://www.ncei.noaa.gov/archive/accession/"
        f"{accession}/download"
    )

    try:
        print(f"  Download: {dl_url}")
        resp = requests.get(dl_url, timeout=120, stream=True, allow_redirects=True)
        if resp.status_code == 200:
            fpath = DATA_DIR / f"{glider}_{accession}.tar.gz"
            with open(fpath, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            mb = fpath.stat().st_size / 1e6
            print(f"  ✓ {mb:.1f} MB → {fpath}")
            return fpath
        else:
            print(f"  → HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ✗ {e}")

    print(f"  ⚠ Auto-download failed. Visit: {url}")
    return None


def main():
    print("=" * 55)
    print("  Seaglider Data Fetcher — Song et al. RFR Pipeline")
    print(f"  NOAA NCEI archives")
    print(f"  Also: Balwada 2023  {ZENODO_BALWADA}")
    print("=" * 55)

    for glider, info in GLIDERS.items():
        fetch_ncei_accession(glider, info["accession"], info["url"])
        time.sleep(1)

    print("\n✓ Seaglider fetch complete.")
    print(f"  If NCEI downloads failed, try Zenodo: {ZENODO_BALWADA}")


if __name__ == "__main__":
    main()
