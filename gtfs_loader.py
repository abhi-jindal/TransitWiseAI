"""
gtfs_loader.py
--------------
Loads and queries the MBTA GTFS dataset for TransitWise AI.

This module is intentionally simple and beginner-friendly. It:
  - reads the GTFS text files with pandas
  - exposes real subway stations (no duplicate platform entries)
  - finds scheduled trips that run on a given date
  - handles GTFS times that go past 24:00:00

It does NOT invent any data. Everything comes from the GTFS files.
"""

from __future__ import annotations

import datetime
import os
from functools import lru_cache

import pandas as pd

# Folder that holds the extracted GTFS .txt files.
GTFS_DIR = os.path.join(os.path.dirname(__file__), "data", "gtfs")

# MBTA subway/metro is route_type 1 (Red, Orange, Blue). The Green and
# Mattapan lines are light rail, route_type 0. We now support the Green Line,
# so the station list includes both heavy and light rail rapid transit.
SUBWAY_ROUTE_TYPES = {"0", "1"}
# The specific route_ids we support (excludes Mattapan to match the four lines).
SUPPORTED_ROUTE_IDS = {"Red", "Orange", "Blue", "Green-B", "Green-C", "Green-D", "Green-E"}

# The GTFS files we rely on. Used to give a clear error if one is missing.
REQUIRED_FILES = [
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "transfers.txt",
]


class GTFSError(Exception):
    """Raised when the GTFS data cannot be loaded or a journey is impossible."""


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _path(filename: str) -> str:
    return os.path.join(GTFS_DIR, filename)


def check_files_exist() -> None:
    """Raise a clear error if the GTFS folder or any required file is missing."""
    if not os.path.isdir(GTFS_DIR):
        raise GTFSError(
            f"GTFS folder not found at: {GTFS_DIR}\n"
            "Make sure the MBTA GTFS files are extracted into data/gtfs/."
        )
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(_path(f))]
    if missing:
        raise GTFSError(
            "These required GTFS files are missing from data/gtfs/: "
            + ", ".join(missing)
        )


# Only the columns each file actually needs downstream. Reading a subset of
# columns (especially for the ~2.8M-row stop_times.txt) is a large cold-start
# win. If a listed column is missing in a feed, pandas is told to ignore it.
_USECOLS = {
    "stops": [
        "stop_id", "stop_name", "parent_station", "location_type",
        "stop_lat", "stop_lon", "platform_code", "platform_name",
    ],
    "routes": [
        "route_id", "route_short_name", "route_long_name", "route_type",
    ],
    "trips": [
        "route_id", "service_id", "trip_id", "trip_headsign", "direction_id",
    ],
    "stop_times": [
        "trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence",
    ],
    "calendar": [
        "service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday", "start_date", "end_date",
    ],
    "calendar_dates": ["service_id", "date", "exception_type"],
    "transfers": [
        "from_stop_id", "to_stop_id", "transfer_type", "min_transfer_time",
    ],
}


@lru_cache(maxsize=1)
def _read_all() -> dict[str, pd.DataFrame]:
    """
    Read the GTFS files once and cache them in memory.

    Everything is read as strings (dtype=str) so that IDs and times are never
    accidentally turned into numbers. We only read the columns we actually use
    (see _USECOLS) which greatly speeds up loading the large stop_times.txt.
    """
    check_files_exist()
    data = {}
    for name in REQUIRED_FILES:
        key = name.replace(".txt", "")
        wanted = _USECOLS.get(key)
        if wanted:
            # Intersect with the file's real header so a missing optional
            # column never raises.
            header = pd.read_csv(_path(name), nrows=0).columns.tolist()
            cols = [c for c in wanted if c in header]
            df = pd.read_csv(_path(name), dtype=str, usecols=cols)
        else:
            df = pd.read_csv(_path(name), dtype=str)
        data[key] = df.fillna("")
    return data


# ---------------------------------------------------------------------------
# Time handling (handles GTFS times >= 24:00:00)
# ---------------------------------------------------------------------------
def gtfs_time_to_seconds(value: str) -> int | None:
    """
    Convert a GTFS time string like '24:01:00' into seconds past midnight.

    GTFS allows hours of 24, 25, ... for trips that run after midnight but
    still belong to the previous service day. Treating the time as plain
    seconds-past-midnight keeps those values correct and comparable.
    """
    value = (value or "").strip()
    if not value:
        return None
    try:
        hours, minutes, seconds = (int(part) for part in value.split(":"))
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_clock(total_seconds: int) -> str:
    """
    Turn seconds-past-midnight into a friendly 12-hour clock string.

    Hours past 24 wrap around (so 24:01 shows as 12:01 AM) but we add a
    '(next day)' note so the timeline stays honest.
    """
    next_day = total_seconds >= 24 * 3600
    seconds_in_day = total_seconds % (24 * 3600)
    hours = seconds_in_day // 3600
    minutes = (seconds_in_day % 3600) // 60

    suffix = "AM" if hours < 12 else "PM"
    display_hour = hours % 12
    if display_hour == 0:
        display_hour = 12
    label = f"{display_hour}:{minutes:02d} {suffix}"
    if next_day:
        label += " (next day)"
    return label


def seconds_to_duration(total_seconds: int) -> str:
    """Turn a span of seconds into a friendly 'X min' / 'X hr Y min' string."""
    minutes = round(total_seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    rest = minutes % 60
    return f"{hours} hr {rest} min" if rest else f"{hours} hr"


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_subway_stations() -> pd.DataFrame:
    """
    Return the parent stations that are served by the subway, as a DataFrame
    with columns [stop_id, stop_name], sorted by name and de-duplicated.

    Using parent stations (location_type == '1') avoids showing each platform
    as a separate entry.
    """
    data = _read_all()
    stops = data["stops"]
    trips = data["trips"]
    stop_times = data["stop_times"]

    # 1. Trips on the supported routes (by explicit route_id).
    subway_trip_ids = set(
        trips.loc[trips["route_id"].isin(SUPPORTED_ROUTE_IDS), "trip_id"]
    )

    # 2. Which platform stop_ids do those trips actually visit?
    visited_platform_ids = set(
        stop_times.loc[stop_times["trip_id"].isin(subway_trip_ids), "stop_id"]
    )

    # 3. Map each visited platform up to its parent station (fallback to self).
    platforms = stops[stops["stop_id"].isin(visited_platform_ids)].copy()
    platforms["station_id"] = platforms["parent_station"].where(
        platforms["parent_station"] != "", platforms["stop_id"]
    )
    station_ids = set(platforms["station_id"])

    # 4. Keep parent stations, plus any surface stops that are their own station.
    stations = stops[stops["stop_id"].isin(station_ids)][
        ["stop_id", "stop_name"]
    ].drop_duplicates()

    return stations.sort_values("stop_name").reset_index(drop=True)


def station_name_to_id(name: str) -> str | None:
    """Look up a parent station's stop_id from its display name."""
    stations = get_subway_stations()
    match = stations[stations["stop_name"] == name]
    if match.empty:
        return None
    return match.iloc[0]["stop_id"]


def child_platform_ids(parent_station_id: str) -> set[str]:
    """Return all platform stop_ids that belong to a parent station."""
    stops = _read_all()["stops"]
    kids = stops[stops["parent_station"] == parent_station_id]["stop_id"]
    return set(kids)


# ---------------------------------------------------------------------------
# Service calendar (which service_ids run on a given date)
# ---------------------------------------------------------------------------
def active_service_ids(date: "datetime.date") -> set[str]:
    """
    Return the set of service_ids that operate on the given calendar date,
    using calendar.txt for the regular weekly pattern and calendar_dates.txt
    for holiday additions/removals.
    """
    data = _read_all()
    calendar = data["calendar"]
    calendar_dates = data["calendar_dates"]

    date_str = date.strftime("%Y%m%d")
    weekday_column = [
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ][date.weekday()]

    active: set[str] = set()

    # Regular weekly service: the weekday flag is '1' and the date is in range.
    for _, row in calendar.iterrows():
        runs_today = row[weekday_column] == "1"
        in_range = row["start_date"] <= date_str <= row["end_date"]
        if runs_today and in_range:
            active.add(row["service_id"])

    # Exceptions: type 1 adds service for that date, type 2 removes it.
    exceptions = calendar_dates[calendar_dates["date"] == date_str]
    for _, row in exceptions.iterrows():
        if row["exception_type"] == "1":
            active.add(row["service_id"])
        elif row["exception_type"] == "2":
            active.discard(row["service_id"])

    return active


# ---------------------------------------------------------------------------
# Transfer times (from transfers.txt)
# ---------------------------------------------------------------------------
# A sensible fallback if the GTFS file has no published time for a transfer.
# Clearly labeled as an assumption wherever it is used.
DEFAULT_TRANSFER_SECONDS = 180  # 3 minutes


def lookup_transfer_seconds(
    from_platform_ids: set[str], to_platform_ids: set[str]
) -> tuple[int, bool]:
    """
    Find the minimum published transfer time (in seconds) between any platform
    in `from_platform_ids` and any platform in `to_platform_ids`.

    Returns (seconds, is_published). If no published value exists, returns the
    default assumption with is_published=False so the UI can label it.
    """
    transfers = _read_all()["transfers"]
    relevant = transfers[
        transfers["from_stop_id"].isin(from_platform_ids)
        & transfers["to_stop_id"].isin(to_platform_ids)
    ]

    times = [
        gtfs_time_to_seconds_safe(v)
        for v in relevant["min_transfer_time"]
        if v not in ("", None)
    ]
    times = [t for t in times if t is not None]

    if times:
        return min(times), True
    return DEFAULT_TRANSFER_SECONDS, False


def gtfs_time_to_seconds_safe(value: str) -> int | None:
    """min_transfer_time is a plain number of seconds, not an HH:MM:SS clock."""
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None
