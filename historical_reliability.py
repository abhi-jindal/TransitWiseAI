"""
historical_reliability.py
-------------------------
Analyzes REAL observed MBTA headways from a small cleaned dataset.

Data source (official, public, CC0):
  MBTA LAMP "Subway Performance Data"
  https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/

The dataset under data/historical/dtx_headways.csv was produced by
prepare_historical.py from a short run of recent weekday daily files, filtered
to Red and Orange Line observed headways at Downtown Crossing (the journey's
transfer station).

Important honesty note:
  Headway variability describes how spread out the real gaps between trains
  were. It is NOT a probability of missing your connection. We never present
  it as such.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

DATA_CSV = os.path.join(
    os.path.dirname(__file__), "data", "historical", "dtx_headways.csv"
)

SOURCE_NAME = "MBTA LAMP Subway Performance Data"
SOURCE_URL = "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/"
STATION_NAME = "Downtown Crossing"


class HistoricalDataUnavailable(Exception):
    """Raised when the cleaned historical CSV is missing or empty."""


@dataclass
class HeadwayStats:
    """Summary statistics for observed headways (all in minutes)."""
    route_id: str
    station_name: str
    observations: int
    mean_min: float
    median_min: float
    p25_min: float
    p75_min: float
    p90_min: float
    date_min: str
    date_max: str
    scheduled_median_min: float | None
    # observed headway values in minutes, for charting
    headways_min: list = field(default_factory=list)
    # hourly breakdown: {hour: {"count", "median_min", "p90_min"}}
    hourly: dict = field(default_factory=dict)


def load_dataset() -> pd.DataFrame:
    """Load the cleaned historical CSV, or raise HistoricalDataUnavailable."""
    if not os.path.isfile(DATA_CSV):
        raise HistoricalDataUnavailable(
            f"Historical dataset not found at {DATA_CSV}. "
            "Run prepare_historical.py first."
        )
    df = pd.read_csv(DATA_CSV)
    if df.empty:
        raise HistoricalDataUnavailable("Historical dataset is empty.")
    return df


def available_routes() -> list[str]:
    """Routes present in the cleaned dataset (e.g. ['Orange', 'Red'])."""
    try:
        df = load_dataset()
    except HistoricalDataUnavailable:
        return []
    return sorted(df["route_id"].unique())


def available_pairs() -> set:
    """
    Set of (parent_station, route_id) pairs that actually have verified
    observations in the cleaned dataset. Used to decide, per transfer, whether
    journey-specific historical data genuinely exists (vs. applying the wrong
    line's stats).
    """
    try:
        df = load_dataset()
    except HistoricalDataUnavailable:
        return set()
    if "parent_station" not in df.columns:
        return set()
    sub = df[df["headway_trunk_seconds"].notna()]
    return set(zip(sub["parent_station"], sub["route_id"]))


def has_data_for(station_id: str, route_id: str) -> bool:
    """True only if the dataset has verified observations for this exact
    station + route. Never guesses or substitutes another line."""
    return (station_id, route_id) in available_pairs()


def _format_date(yyyymmdd: int) -> str:
    s = str(int(yyyymmdd))
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def compute_stats(route_id: str, station_id: str | None = None,
                  station_label: str | None = None) -> HeadwayStats:
    """
    Compute observed-headway statistics for one route, optionally restricted to
    a specific parent station. If `station_id` is given, only observations at
    that station are used; if none exist, raises HistoricalDataUnavailable so
    the caller can show an honest "unavailable" message (never substitute
    another station's or line's data).

    `station_label` is the human-facing station name for display; defaults to
    the module's example station when not restricting by station.
    """
    df = load_dataset()
    rows = df[df["route_id"] == route_id].copy()
    if station_id is not None and "parent_station" in rows.columns:
        rows = rows[rows["parent_station"] == station_id]
    rows = rows[rows["headway_trunk_seconds"].notna()]
    if rows.empty:
        where = station_label or (station_id or STATION_NAME)
        raise HistoricalDataUnavailable(
            f"No observed headways for the {route_id} Line at {where} "
            "in the loaded sample."
        )
    station_name = station_label or (STATION_NAME if station_id is None else station_id)

    # Convert to minutes once.
    rows["headway_min"] = rows["headway_trunk_seconds"] / 60.0

    headways = rows["headway_min"]
    scheduled_median = None
    if "scheduled_headway_trunk" in rows.columns:
        sched = rows["scheduled_headway_trunk"].dropna()
        if not sched.empty:
            scheduled_median = round(float(sched.median()) / 60.0, 1)

    # Hourly variation (by trip start hour), only if we can derive an hour.
    #
    # NOTE: start_time is "seconds after midnight" and can exceed 86400 for
    # after-midnight service (the same convention as GTFS times past 24:00:00).
    # Dividing directly would yield hours like 24 or 25. We wrap into a real
    # local clock hour (0-23) with % 24, so a 24:30 trip is counted in hour 0.
    # The underlying service_date is unchanged; only the display hour wraps.
    hourly: dict = {}
    if "start_time" in rows.columns:
        rows["hour"] = ((rows["start_time"] // 3600) % 24).astype(int)
        for hour, grp in rows.groupby("hour"):
            hw = grp["headway_min"]
            hourly[int(hour)] = {
                "count": int(len(hw)),
                "median_min": round(float(hw.median()), 1),
                "p90_min": round(float(hw.quantile(0.90)), 1),
            }

    return HeadwayStats(
        route_id=route_id,
        station_name=station_name,
        observations=int(len(headways)),
        mean_min=round(float(headways.mean()), 1),
        median_min=round(float(headways.median()), 1),
        p25_min=round(float(headways.quantile(0.25)), 1),
        p75_min=round(float(headways.quantile(0.75)), 1),
        p90_min=round(float(headways.quantile(0.90)), 1),
        date_min=_format_date(rows["service_date"].min()),
        date_max=_format_date(rows["service_date"].max()),
        scheduled_median_min=scheduled_median,
        headways_min=[round(float(x), 2) for x in headways.tolist()],
        hourly=dict(sorted(hourly.items())),
    )


def hourly_dataframe(stats: HeadwayStats) -> pd.DataFrame:
    """Turn the hourly dict into a tidy DataFrame for charting."""
    if not stats.hourly:
        return pd.DataFrame(columns=["hour", "median_min", "p90_min", "count"])
    records = [
        {"hour": h, "median_min": v["median_min"],
         "p90_min": v["p90_min"], "count": v["count"]}
        for h, v in stats.hourly.items()
    ]
    return pd.DataFrame(records)
