"""
prepare_historical.py
----------------------
ONE-TIME data preparation for the Historical Reliability feature.

Downloads a small set of recent WEEKDAY daily files from the official MBTA
LAMP "Subway Performance Data" archive, keeps only Red and Orange Line
observed headways at Downtown Crossing (the journey's transfer station), and
writes a single small cleaned CSV to data/historical/.

Source (official, public, CC0):
  https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/
  <YYYY-MM-DD>-subway-on-time-performance-v1.parquet

Run with the project's virtual environment:
  .\\.venv\\Scripts\\python.exe prepare_historical.py

This keeps the Streamlit app light: the app reads only the small CSV and never
downloads parquet at runtime. We never fabricate observations; if a date is not
available in the archive it is simply skipped and reported.
"""

from __future__ import annotations

import os
import urllib.request

import pandas as pd

LAMP_BASE = (
    "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/"
)

# A short, recent run of WEEKDAY service dates (Mon-Fri). These are the most
# recent full weekdays available in the archive at build time.
SERVICE_DATES = [
    "2026-10-19",  # Monday
    "2026-10-20",  # Tuesday
    "2026-10-21",  # Wednesday
    "2026-10-22",  # Thursday
    "2026-10-23",  # Friday
]

TRANSFER_PARENT_STATION = "place-dwnxg"   # Downtown Crossing
ROUTES = ["Red", "Orange"]

RAW_DIR = os.path.join("data", "historical", "raw")
OUT_DIR = os.path.join("data", "historical")
OUT_CSV = os.path.join(OUT_DIR, "dtx_headways.csv")

# Keep only the columns we actually need for the reliability analysis.
KEEP_COLUMNS = [
    "service_date",
    "route_id",
    "direction_id",
    "stop_id",
    "parent_station",
    "start_time",               # seconds after midnight of the trip start
    "headway_trunk_seconds",    # the observed headway (what we analyze)
    "scheduled_headway_trunk",  # scheduled headway, for context
]


def download_day(date_str: str) -> str | None:
    """Download one daily parquet into RAW_DIR. Returns the local path or None."""
    os.makedirs(RAW_DIR, exist_ok=True)
    url = f"{LAMP_BASE}{date_str}-subway-on-time-performance-v1.parquet"
    local = os.path.join(RAW_DIR, f"{date_str}.parquet")
    if os.path.exists(local):
        return local
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TransitWiseAI/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        with open(local, "wb") as f:
            f.write(data)
        return local
    except Exception as exc:
        print(f"  SKIP {date_str}: could not download ({exc})")
        return None


def clean_day(local_path: str) -> pd.DataFrame:
    """Load one daily file and keep Red/Orange observed headways at DTX."""
    df = pd.read_parquet(local_path)
    mask = (
        (df["parent_station"] == TRANSFER_PARENT_STATION)
        & (df["route_id"].isin(ROUTES))
        & (df["headway_trunk_seconds"].notna())
    )
    subset = df.loc[mask, [c for c in KEEP_COLUMNS if c in df.columns]].copy()
    # direction_id is a bool in the source; store as 0/1 for clarity.
    if "direction_id" in subset.columns:
        subset["direction_id"] = subset["direction_id"].astype(int)
    return subset


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    used_dates = []
    for date_str in SERVICE_DATES:
        local = download_day(date_str)
        if not local:
            continue
        day = clean_day(local)
        if not day.empty:
            frames.append(day)
            used_dates.append(date_str)
            print(f"  OK   {date_str}: {len(day)} observed headways")

    if not frames:
        print("No data could be prepared. Nothing written.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    size_kb = os.path.getsize(OUT_CSV) / 1024
    print(f"\nWrote {OUT_CSV}: {len(combined)} rows, {size_kb:.1f} KB")
    print(f"Dates used: {', '.join(used_dates)}")
    print("Routes:", ", ".join(sorted(combined["route_id"].unique())))


if __name__ == "__main__":
    main()
