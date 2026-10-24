"""
network.py
----------
General MBTA subway network model + journey search for TransitWise AI.

Replaces the old hardcoded Red->Orange routing with a reusable, line-agnostic
search over the real GTFS schedule. Supports the Red, Orange, Blue and Green
(B/C/D/E) lines, direct trips and journeys with one or two transfers, in both
directions and across Red/Green branches.

Everything is derived from the official GTFS files via gtfs_loader. Nothing is
invented: stops, trips, times and transfers all come from the data. Published
GTFS transfer times are used where available; any fallback is clearly labeled.

Design notes:
  - Heavy GTFS processing is cached with functools.lru_cache so Streamlit does
    not rebuild the network on every rerun.
  - Times are "seconds past midnight" so GTFS values beyond 24:00:00 work.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

import gtfs_loader as gtfs

# The four lines we support in this iteration (real GTFS route_ids).
# Mattapan is intentionally excluded to keep scope to the four named lines.
SUPPORTED_ROUTE_IDS = ["Red", "Orange", "Blue", "Green-B", "Green-C", "Green-D", "Green-E"]

ROUTE_DISPLAY = {
    "Red": "Red Line",
    "Orange": "Orange Line",
    "Blue": "Blue Line",
    "Green-B": "Green Line B",
    "Green-C": "Green Line C",
    "Green-D": "Green Line D",
    "Green-E": "Green Line E",
}

ROUTE_EMOJI = {
    "Red": "🔴", "Orange": "🟠", "Blue": "🔵",
    "Green-B": "🟢", "Green-C": "🟢", "Green-D": "🟢", "Green-E": "🟢",
}

# The most transfers we will consider (Phase 1 requirement: up to two).
MAX_TRANSFERS = 2

# Minimum sensible transfer time if GTFS publishes none (clearly labeled).
FALLBACK_TRANSFER_SECONDS = gtfs.DEFAULT_TRANSFER_SECONDS


def route_display(route_id: str) -> str:
    return ROUTE_DISPLAY.get(route_id, route_id)


def route_emoji(route_id: str) -> str:
    return ROUTE_EMOJI.get(route_id, "🚇")


# ---------------------------------------------------------------------------
# Core GTFS tables, trimmed to the supported network and cached.
#
# Cold-start optimization: the trimmed, time-parsed stop_times table is the
# expensive thing to build. We persist it as a versioned parquet artifact keyed
# by the stop_times.txt modification time, so a warm cold-start just reads a
# small parquet instead of reparsing + rebuilding from the full GTFS feed.
# ---------------------------------------------------------------------------
import os as _os

_ARTIFACT_DIR = _os.path.join(_os.path.dirname(__file__), "data", "cache")


def _artifact_version() -> str:
    """Version token derived from the GTFS stop_times.txt mtime + size."""
    try:
        p = _os.path.join(gtfs.GTFS_DIR, "stop_times.txt")
        stt = _os.stat(p)
        return f"{int(stt.st_mtime)}_{stt.st_size}"
    except OSError:
        return "unknown"


def _build_supported_trip_stops() -> pd.DataFrame:
    """Build the trimmed, time-parsed supported stop_times table from GTFS."""
    data = gtfs._read_all()
    trips = data["trips"]
    stop_times = data["stop_times"]
    stops = data["stops"]

    route_of = dict(zip(trips["trip_id"], trips["route_id"]))
    service_of = dict(zip(trips["trip_id"], trips["service_id"]))
    direction_of = dict(zip(trips["trip_id"], trips["direction_id"]))
    parent_of = dict(zip(stops["stop_id"], stops["parent_station"]))

    supported_trip_ids = set(
        trips.loc[trips["route_id"].isin(SUPPORTED_ROUTE_IDS), "trip_id"]
    )
    st = stop_times[stop_times["trip_id"].isin(supported_trip_ids)].copy()

    st["route_id"] = st["trip_id"].map(route_of)
    st["service_id"] = st["trip_id"].map(service_of)
    st["direction_id"] = st["trip_id"].map(direction_of)
    st["parent_station"] = st["stop_id"].map(parent_of).fillna("")
    st.loc[st["parent_station"] == "", "parent_station"] = st["stop_id"]
    st["seq"] = st["stop_sequence"].astype(int)
    st["dep_s"] = st["departure_time"].map(gtfs.gtfs_time_to_seconds)
    st["arr_s"] = st["arrival_time"].map(gtfs.gtfs_time_to_seconds)

    return st.sort_values(["trip_id", "seq"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _supported_trip_stops() -> pd.DataFrame:
    """
    Trimmed stop_times for supported routes, with route/service/direction and
    parsed times. Loads a cached parquet artifact when it matches the current
    GTFS version; otherwise builds it and writes the artifact for next time.
    """
    version = _artifact_version()
    artifact = _os.path.join(_ARTIFACT_DIR, f"supported_trip_stops_{version}.parquet")

    if _os.path.isfile(artifact):
        try:
            return pd.read_parquet(artifact)
        except Exception:
            pass  # fall through to rebuild on any read problem

    df = _build_supported_trip_stops()

    try:
        _os.makedirs(_ARTIFACT_DIR, exist_ok=True)
        # Remove stale artifacts from older GTFS versions.
        for f in _os.listdir(_ARTIFACT_DIR):
            if f.startswith("supported_trip_stops_") and f != _os.path.basename(artifact):
                try:
                    _os.remove(_os.path.join(_ARTIFACT_DIR, f))
                except OSError:
                    pass
        df.to_parquet(artifact, index=False)
    except Exception:
        pass  # caching is best-effort; never break the app if write fails
    return df


@lru_cache(maxsize=1)
def supported_stations() -> pd.DataFrame:
    """Parent stations served by any supported route: [stop_id, stop_name]."""
    st = _supported_trip_stops()
    stops = gtfs._read_all()["stops"]
    parent_ids = set(st["parent_station"]) - {""}
    parents = stops[
        (stops["stop_id"].isin(parent_ids)) & (stops["location_type"] == "1")
    ][["stop_id", "stop_name"]].drop_duplicates()
    # Some light-rail surface stops are their own station (location_type 0 with
    # no parent). Include those names too so Green surface stops appear.
    own = stops[stops["stop_id"].isin(parent_ids) & (stops["location_type"] != "1")][
        ["stop_id", "stop_name"]
    ].drop_duplicates()
    combined = pd.concat([parents, own]).drop_duplicates("stop_id")
    return combined.sort_values("stop_name").reset_index(drop=True)


@lru_cache(maxsize=1)
def station_name_map() -> dict:
    """name -> stop_id for supported stations (first match wins)."""
    df = supported_stations()
    return dict(zip(df["stop_name"], df["stop_id"]))


def station_id(name: str) -> str | None:
    return station_name_map().get(name)


def station_name(stop_id: str) -> str:
    df = supported_stations()
    hit = df[df["stop_id"] == stop_id]
    return hit.iloc[0]["stop_name"] if not hit.empty else stop_id


@lru_cache(maxsize=1)
def _station_routes() -> dict:
    """parent_station -> set of route_ids serving it."""
    st = _supported_trip_stops()
    out: dict[str, set] = {}
    for station, grp in st.groupby("parent_station"):
        out[station] = set(grp["route_id"].unique())
    return out


@lru_cache(maxsize=1)
def transfer_stations() -> dict:
    """
    station -> set of OTHER stations reachable by a published GTFS transfer
    (including same-station line changes). Built from transfers.txt, mapped up
    to parent stations. Used to enumerate candidate transfer points.
    """
    data = gtfs._read_all()
    transfers = data["transfers"]
    stops = data["stops"]
    parent_of = dict(zip(stops["stop_id"], stops["parent_station"]))

    def to_parent(sid):
        p = parent_of.get(sid, "")
        return p if p else sid

    out: dict[str, set] = {}
    for _, row in transfers.iterrows():
        a = to_parent(row["from_stop_id"])
        b = to_parent(row["to_stop_id"])
        if not a or not b:
            continue
        out.setdefault(a, set()).add(b)
    return out


# ---------------------------------------------------------------------------
# Per-(route, service, station) departure boards, for fast search.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _trip_sequences() -> dict:
    """
    trip_id -> list of (seq, station, dep_s, arr_s), sorted by seq.

    Built via a single vectorized pass (no per-group pandas overhead) and
    persisted as a pickle keyed by the GTFS version, because this derivation
    over ~250k rows is the single largest cold-start cost. The source table is
    already sorted by (trip_id, seq), so we can group by scanning contiguous
    runs of trip_id.
    """
    import pickle

    version = _artifact_version()
    pkl = _os.path.join(_ARTIFACT_DIR, f"trip_sequences_{version}.pkl")
    if _os.path.isfile(pkl):
        try:
            with open(pkl, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass

    st = _supported_trip_stops()
    # Extract columns as plain Python lists once (fast), then single-pass group.
    trip_ids = st["trip_id"].tolist()
    seq_col = st["seq"].tolist()
    station_col = st["parent_station"].tolist()
    dep_col = st["dep_s"].tolist()
    arr_col = st["arr_s"].tolist()

    seqs: dict[str, list] = {}
    for i, trip_id in enumerate(trip_ids):
        row = (int(seq_col[i]), station_col[i], dep_col[i], arr_col[i])
        bucket = seqs.get(trip_id)
        if bucket is None:
            seqs[trip_id] = [row]
        else:
            bucket.append(row)
    # Ensure each trip's rows are sorted by sequence (source is already sorted,
    # but this guards against any ordering surprise).
    for trip_id, rows in seqs.items():
        rows.sort(key=lambda r: r[0])

    try:
        _os.makedirs(_ARTIFACT_DIR, exist_ok=True)
        for f in _os.listdir(_ARTIFACT_DIR):
            if f.startswith("trip_sequences_") and f != _os.path.basename(pkl):
                try:
                    _os.remove(_os.path.join(_ARTIFACT_DIR, f))
                except OSError:
                    pass
        with open(pkl, "wb") as fh:
            pickle.dump(seqs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return seqs


# ---------------------------------------------------------------------------
# Leg finding
# ---------------------------------------------------------------------------
@dataclass
class RawLeg:
    route_id: str
    trip_id: str
    from_station: str
    board_s: int
    to_station: str
    alight_s: int


@lru_cache(maxsize=4)
def _station_boardings(service_ids_key: tuple) -> dict:
    """
    For the active services, map each station to the trips that depart from it:
      station -> list of (dep_s, trip_id, from_seq), sorted by dep_s.
    One pass; cached per service-id set. Fast O(log n) boarding lookup via
    bisect on dep_s.
    """
    service_ids = set(service_ids_key)
    st = _supported_trip_stops()
    active = st[st["service_id"].isin(service_ids)]
    out: dict[str, list] = {}
    for row in active.itertuples(index=False):
        if row.dep_s is None:
            continue
        out.setdefault(row.parent_station, []).append(
            (int(row.dep_s), row.trip_id, int(row.seq))
        )
    for station in out:
        out[station].sort()
    return out


def earliest_leg(
    from_station: str,
    to_station: str,
    earliest_dep_s: int,
    service_ids_key: tuple,
) -> RawLeg | None:
    """
    Earliest single-vehicle ride from from_station to to_station departing at
    or after earliest_dep_s. Correct direction guaranteed by stop_sequence.
    Uses bisect to jump to the first eligible boarding.
    """
    import bisect

    boardings = _station_boardings(service_ids_key)
    seqs = _trip_sequences()
    station_boards = boardings.get(from_station, [])
    if not station_boards:
        return None

    # Jump to the first boarding at/after earliest_dep_s.
    i = bisect.bisect_left(station_boards, (earliest_dep_s, "", 0))
    best: RawLeg | None = None
    # Only scan a bounded window of upcoming departures (plenty for subway
    # frequencies); the first trip that reaches the destination wins because
    # boardings are sorted ascending by departure.
    for dep_s, trip_id, from_seq in station_boards[i:i + 60]:
        if best is not None and dep_s > best.board_s:
            break
        for seq, station, _d, arr in seqs.get(trip_id, []):
            if seq <= from_seq or arr is None:
                continue
            if station == to_station:
                if best is None or dep_s < best.board_s:
                    route_id = _trip_route().get(trip_id, "")
                    best = RawLeg(route_id, trip_id, from_station, dep_s,
                                  to_station, int(arr))
                break
    return best


@lru_cache(maxsize=1)
def _trip_route() -> dict:
    """trip_id -> route_id for supported trips."""
    st = _supported_trip_stops()
    return dict(zip(st["trip_id"], st["route_id"]))


@lru_cache(maxsize=1)
def _trip_reachable_stations() -> dict:
    """
    trip_id -> dict of {from_seq_station_index}. Precompute, for each trip, the
    list of (seq, station, arr_s) so onward reachability is a slice. Returned as
    trip_id -> list of (seq, station, arr_s).
    """
    return _trip_sequences()


# ---------------------------------------------------------------------------
# Transfer time between two stations (published GTFS or labeled fallback).
# ---------------------------------------------------------------------------
def transfer_time(from_station: str, to_station: str) -> tuple[int, bool]:
    """
    Transfer walking time in seconds between two parent stations.
    Returns (seconds, is_published). Same-station line changes and short
    station-to-station walks both come from transfers.txt where available.
    """
    from_platforms = gtfs.child_platform_ids(from_station) | {from_station}
    to_platforms = gtfs.child_platform_ids(to_station) | {to_station}
    return gtfs.lookup_transfer_seconds(from_platforms, to_platforms)


# ---------------------------------------------------------------------------
# General journey search
# ---------------------------------------------------------------------------
@dataclass
class Transfer:
    at_station: str
    walk_seconds: int
    is_published: bool

    def wait_seconds(self, prev_alight_s: int, next_board_s: int) -> int:
        return max(0, next_board_s - prev_alight_s - self.walk_seconds)


@dataclass
class GeneralJourney:
    """A journey as an ordered list of legs with transfers between them."""
    legs: list  # list[RawLeg]
    transfers: list = field(default_factory=list)  # list[Transfer], len = legs-1

    @property
    def depart_s(self) -> int:
        return self.legs[0].board_s

    @property
    def arrive_s(self) -> int:
        return self.legs[-1].alight_s

    @property
    def total_seconds(self) -> int:
        return self.arrive_s - self.depart_s

    @property
    def num_transfers(self) -> int:
        return len(self.legs) - 1

    def signature(self) -> tuple:
        return tuple((l.route_id, l.board_s, l.to_station, l.alight_s) for l in self.legs)


@lru_cache(maxsize=1)
def interchange_hubs() -> list[str]:
    """
    The true interchange stations: parent stations served by two or more
    distinct supported routes (Green-* count as Green for this purpose, but a
    station with any two different route_ids qualifies). On the MBTA this is a
    small set (~a dozen), which keeps the transfer search fast.
    """
    sr = _station_routes()
    hubs = []
    for station, routes in sr.items():
        # Treat all Green branches as interchangeable at a shared platform, but
        # a hub is any station where you can change to a different line family.
        families = {r.split("-")[0] for r in routes}
        if len(families) >= 2:
            hubs.append(station)
    return sorted(hubs)


def _reachable_hubs_from(station: str, service_ids_key: tuple, earliest_s: int,
                         limit_hubs: int = 20) -> list[tuple]:
    """
    From `station`, the earliest leg to each interchange hub (if any), as a list
    of (hub, RawLeg), sorted by hub arrival time. Bounded to keep it fast.
    """
    out = []
    for hub in interchange_hubs():
        if hub == station:
            continue
        leg = earliest_leg(station, hub, earliest_s, service_ids_key)
        if leg is not None:
            out.append((hub, leg))
    out.sort(key=lambda x: x[1].alight_s)
    return out[:limit_hubs]


def _direct_journeys(from_station, to_station, earliest_s, skey, limit):
    """Direct (0-transfer) journeys, earliest departures first."""
    results = []
    seen = set()
    dep_cursor = earliest_s
    for _ in range(limit + 2):
        leg = earliest_leg(from_station, to_station, dep_cursor, skey)
        if leg is None:
            break
        j = GeneralJourney(legs=[leg])
        if j.signature() not in seen:
            seen.add(j.signature())
            results.append(j)
            if len(results) >= limit:
                break
        dep_cursor = leg.board_s + 1
    return results


def _one_transfer_journeys(from_station, to_station, earliest_s, skey, limit):
    """Journeys with exactly one transfer via an interchange hub."""
    results = []
    seen = set()
    for hub, leg1 in _reachable_hubs_from(from_station, skey, earliest_s):
        walk_s, published = transfer_time(hub, hub)
        leg2 = earliest_leg(hub, to_station, leg1.alight_s + walk_s, skey)
        if leg2 is None or leg2.route_id == leg1.route_id:
            continue
        j = GeneralJourney(
            legs=[leg1, leg2],
            transfers=[Transfer(hub, walk_s, published)],
        )
        if j.signature() not in seen:
            seen.add(j.signature())
            results.append(j)
    results.sort(key=lambda j: j.arrive_s)
    return results[:limit]


def _two_transfer_journeys(from_station, to_station, earliest_s, skey, limit):
    """Journeys with exactly two transfers via two interchange hubs."""
    results = []
    seen = set()
    for hub1, leg1 in _reachable_hubs_from(from_station, skey, earliest_s):
        w1, p1 = transfer_time(hub1, hub1)
        for hub2, leg2 in _reachable_hubs_from(hub1, skey, leg1.alight_s + w1):
            if hub2 == hub1 or leg2.route_id == leg1.route_id:
                continue
            w2, p2 = transfer_time(hub2, hub2)
            leg3 = earliest_leg(hub2, to_station, leg2.alight_s + w2, skey)
            if leg3 is None or leg3.route_id == leg2.route_id:
                continue
            j = GeneralJourney(
                legs=[leg1, leg2, leg3],
                transfers=[Transfer(hub1, w1, p1), Transfer(hub2, w2, p2)],
            )
            if j.signature() not in seen:
                seen.add(j.signature())
                results.append(j)
    results.sort(key=lambda j: j.arrive_s)
    return results[:limit]


def search_journeys(
    from_station: str,
    to_station: str,
    travel_date: datetime.date,
    earliest_departure_s: int,
    max_results: int = 3,
) -> list[GeneralJourney]:
    """
    Find up to `max_results` journeys from from_station to to_station departing
    at or after earliest_departure_s on travel_date. Prefers fewer transfers
    and earlier arrival. Returns [] if none found (never fabricates).
    """
    if from_station == to_station:
        return []

    service_ids = gtfs.active_service_ids(travel_date)
    if not service_ids:
        return []
    skey = tuple(sorted(service_ids))

    # 1. Direct trips first.
    direct = _direct_journeys(from_station, to_station, earliest_departure_s, skey, max_results)
    if len(direct) >= max_results:
        return direct[:max_results]

    # 2. One-transfer journeys.
    one = _one_transfer_journeys(
        from_station, to_station, earliest_departure_s, skey, max_results
    )

    # 3. Two-transfer journeys only if we still need more options.
    results = _dedupe_merge(direct + one)
    if len(results) < max_results:
        two = _two_transfer_journeys(
            from_station, to_station, earliest_departure_s, skey, max_results
        )
        results = _dedupe_merge(results + two)

    # Rank: fewer transfers, then earlier arrival, then shorter total.
    results.sort(key=lambda j: (j.num_transfers, j.arrive_s, j.total_seconds))
    return results[:max_results]


def _dedupe_merge(journeys: list) -> list:
    seen = set()
    out = []
    for j in journeys:
        sig = j.signature()
        if sig not in seen:
            seen.add(sig)
            out.append(j)
    return out
