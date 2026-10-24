"""
download_gtfs.py
----------------
Downloads the official MBTA GTFS feed and extracts the files TransitWise AI
needs into data/gtfs/.

Official source (from the MBTA's own GTFS documentation):
  https://cdn.mbta.com/MBTA_GTFS.zip
  (see https://github.com/mbta/gtfs-documentation)

Standard library only — no third-party packages required.

Usage (from the project root, using the project's virtual environment):
  .\\.venv\\Scripts\\python.exe download_gtfs.py

This does NOT touch the GTFS parser or routing logic; it only fetches the raw
data files the app reads at runtime. The feed is large-ish (~a few MB zipped),
so it is intentionally kept out of version control (see .gitignore).
"""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile

GTFS_URL = "https://cdn.mbta.com/MBTA_GTFS.zip"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "gtfs")
ZIP_PATH = os.path.join(os.path.dirname(__file__), "data", "MBTA_GTFS.zip")

# The files the app actually reads (gtfs_loader.REQUIRED_FILES). We extract the
# whole archive, but verify these exist afterwards.
REQUIRED_FILES = [
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "transfers.txt",
]


def download() -> None:
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    print(f"Downloading MBTA GTFS feed from {GTFS_URL} ...")
    try:
        req = urllib.request.Request(GTFS_URL, headers={"User-Agent": "TransitWiseAI/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} from {GTFS_URL}")
            data = resp.read()
        with open(ZIP_PATH, "wb") as f:
            f.write(data)
        print(f"  Downloaded {len(data) / 1_000_000:.1f} MB to {ZIP_PATH}")
    except Exception as exc:
        print(
            "\nERROR: could not download the MBTA GTFS feed.\n"
            f"  URL:   {GTFS_URL}\n"
            f"  Cause: {exc}\n\n"
            "What to try:\n"
            "  - Check your internet connection / proxy / firewall.\n"
            "  - Open the URL in a browser to confirm it is reachable.\n"
            "  - If it works in the browser, download MBTA_GTFS.zip manually and\n"
            "    extract its .txt files into data/gtfs/.\n",
            file=sys.stderr,
        )
        sys.exit(1)


def extract() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Extracting to {OUT_DIR} ...")
    try:
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(OUT_DIR)
    except zipfile.BadZipFile as exc:
        print(f"ERROR: downloaded file is not a valid ZIP ({exc}).", file=sys.stderr)
        sys.exit(1)


def verify() -> None:
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(OUT_DIR, f))]
    if missing:
        print(
            "ERROR: the GTFS archive extracted, but these required files are "
            "missing: " + ", ".join(missing),
            file=sys.stderr,
        )
        sys.exit(1)
    print("All required GTFS files are present:")
    for f in REQUIRED_FILES:
        size_kb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f}: {size_kb:,.0f} KB")


def main() -> None:
    download()
    extract()
    verify()
    # Remove the downloaded ZIP; only the extracted .txt files are needed.
    try:
        os.remove(ZIP_PATH)
    except OSError:
        pass
    print("\nDone. TransitWise AI can now read the GTFS feed from data/gtfs/.")


if __name__ == "__main__":
    main()
