"""
download_gtfs.py
----------------
Downloads the official MBTA GTFS feed and extracts the files TransitWise AI
needs into data/gtfs/.

Official source (from the MBTA's own GTFS documentation):
  https://cdn.mbta.com/MBTA_GTFS.zip
  (see https://github.com/mbta/gtfs-documentation)

Standard library only — no third-party packages required. Works on Windows and
Linux (e.g. Streamlit Community Cloud).

Two ways to use it:
  1. Command line (local setup):
       .\\.venv\\Scripts\\python.exe download_gtfs.py
  2. Imported by the app: gtfs_loader calls ensure_gtfs() automatically the
     first time the GTFS files are missing (e.g. on a fresh Streamlit Cloud
     clone, where data/gtfs/ is git-ignored and therefore absent).

This does NOT touch the GTFS parser or routing logic; it only fetches the raw
data files the app reads at runtime. The feed is kept out of version control
(see .gitignore), so the hosted app downloads it on first startup.
"""

from __future__ import annotations

import os
import sys
import tempfile
import urllib.request
import zipfile

GTFS_URL = "https://cdn.mbta.com/MBTA_GTFS.zip"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "gtfs")

# The files the app actually reads (mirrors gtfs_loader.REQUIRED_FILES). We
# extract the whole archive, but verify these exist afterwards.
REQUIRED_FILES = [
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "transfers.txt",
]

DOWNLOAD_TIMEOUT_S = 120


class GTFSDownloadError(Exception):
    """Raised when the GTFS feed cannot be downloaded, extracted, or verified."""


def files_present(out_dir: str = OUT_DIR) -> bool:
    """True only if every required GTFS file is already on disk."""
    return all(os.path.isfile(os.path.join(out_dir, f)) for f in REQUIRED_FILES)


def _download_to(tmp_zip_path: str, log=print) -> None:
    log(f"Downloading MBTA GTFS feed from {GTFS_URL} ...")
    try:
        req = urllib.request.Request(
            GTFS_URL, headers={"User-Agent": "TransitWiseAI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp:
            status = getattr(resp, "status", 200)
            if status != 200:
                raise GTFSDownloadError(f"HTTP {status} from {GTFS_URL}")
            data = resp.read()
        with open(tmp_zip_path, "wb") as f:
            f.write(data)
        log(f"  Downloaded {len(data) / 1_000_000:.1f} MB")
    except GTFSDownloadError:
        raise
    except Exception as exc:
        raise GTFSDownloadError(
            f"Could not download the MBTA GTFS feed from {GTFS_URL}: {exc}"
        ) from exc


def _safe_extract(tmp_zip_path: str, out_dir: str, log=print) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log(f"Extracting to {out_dir} ...")
    try:
        with zipfile.ZipFile(tmp_zip_path) as zf:
            dest_root = os.path.abspath(out_dir)
            for member in zf.namelist():
                # Guard against path traversal (zip-slip): resolve the target
                # and confirm it stays inside out_dir.
                target = os.path.abspath(os.path.join(out_dir, member))
                if not (target == dest_root or target.startswith(dest_root + os.sep)):
                    raise GTFSDownloadError(
                        f"Refusing to extract unsafe path from archive: {member}"
                    )
            zf.extractall(out_dir)
    except zipfile.BadZipFile as exc:
        raise GTFSDownloadError(
            f"Downloaded file is not a valid ZIP: {exc}"
        ) from exc


def _verify(out_dir: str) -> None:
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(out_dir, f))]
    if missing:
        raise GTFSDownloadError(
            "The GTFS archive extracted, but these required files are missing: "
            + ", ".join(missing)
        )


def ensure_gtfs(out_dir: str = OUT_DIR, force: bool = False, log=print) -> bool:
    """
    Ensure the required GTFS files exist in `out_dir`, downloading + extracting
    the official feed if they are missing.

    - Returns True if a download was performed, False if files were already
      present (no-op).
    - Raises GTFSDownloadError on any failure (network, timeout, bad zip,
      missing files) so callers can show a clear message instead of crashing.
    - Uses a temporary ZIP file that is always cleaned up.
    - Safe to call on Windows and Linux.
    """
    if files_present(out_dir) and not force:
        return False

    os.makedirs(out_dir, exist_ok=True)
    # Temp file in the same parent dir as out_dir so it's on the same volume.
    tmp_parent = os.path.dirname(os.path.abspath(out_dir)) or "."
    os.makedirs(tmp_parent, exist_ok=True)
    fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip", prefix="mbta_gtfs_", dir=tmp_parent)
    os.close(fd)
    try:
        _download_to(tmp_zip_path, log=log)
        _safe_extract(tmp_zip_path, out_dir, log=log)
        _verify(out_dir)
    finally:
        try:
            os.remove(tmp_zip_path)
        except OSError:
            pass
    return True


def main() -> None:
    try:
        performed = ensure_gtfs(force=True)
    except GTFSDownloadError as exc:
        print(
            "\nERROR: " + str(exc) + "\n\n"
            "What to try:\n"
            "  - Check your internet connection / proxy / firewall.\n"
            f"  - Open {GTFS_URL} in a browser to confirm it is reachable.\n"
            "  - If it works in the browser, download MBTA_GTFS.zip manually and\n"
            "    extract its .txt files into data/gtfs/.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print("All required GTFS files are present:")
    for f in REQUIRED_FILES:
        size_kb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f}: {size_kb:,.0f} KB")
    print("\nDone. TransitWise AI can now read the GTFS feed from data/gtfs/.")


if __name__ == "__main__":
    main()
