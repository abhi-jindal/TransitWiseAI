"""
journey_planner.py
------------------
General scheduled journey planning for the MBTA subway (Red, Orange, Blue,
Green B/C/D/E), built on top of network.py.

This module keeps a stable public API used by the rest of the app
(plan_journey, arrive_by, supported_journey, leg_trip_info, leg_segment_stops,
_best_leg, and the Leg / Journey / Transfer types) while delegating the actual
multi-line routing to network.py. Nothing is invented: every leg, time and
transfer comes from the GTFS schedule.

A Journey is now a general sequence of legs with transfers between them (0, 1
or 2 transfers). For backward compatibility it still exposes `leg1` / `leg2`
and the single-transfer helpers that earlier code relied on.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import gtfs_loader as gtfs
import network as net

# Backward-compatible constant: the classic Red->Orange transfer hub. Kept so
# older references keep working; general routing no longer depends on it.
TRANSFER_STATION_ID = "place-dwnxg"
TRANSFER_STATION_NAME = "Downtown Crossing"
LEG1_ROUTE_ID = "Red"
LEG2_ROUTE_ID = "Orange"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Leg:
    """One ride on a single line."""
    route_id: str
    route_name: str
    board_station: str          # parent station id
    board_time_s: int
    alight_station: str         # parent station id
    alight_time_s: int
    trip_id: str | None = None

    @property
    def travel_seconds(self) -> int:
        return self.alight_time_s - self.board_time_s

    @property
    def board_station_name(self) -> str:
        return net.station_name(self.board_station)

    @property
    def alight_station_name(self) -> str:
        return net.station_name(self.alight_station)


@dataclass
class Transfer:
    """A transfer between two consecutive legs, at a station."""
    at_station: str
    walk_seconds: int
    is_published: bool
    prev_alight_s: int
    next_board_s: int

    @property
    def at_station_name(self) -> str:
        return net.station_name(self.at_station)

    @property
    def total_gap_seconds(self) -> int:
        """Whole interval between trains (walk + any platform wait)."""
        return self.next_board_s - self.prev_alight_s

    @property
    def wait_after_walk_seconds(self) -> int:
        return max(0, self.total_gap_seconds - self.walk_seconds)

    @property
    def usable_buffer_seconds(self) -> int:
        """Spare time after walking before the next train leaves."""
        return self.next_board_s - self.prev_alight_s - self.walk_seconds


@dataclass
class Journey:
    """A journey: ordered legs with transfers between them (len transfers = legs-1)."""
    legs: list = field(default_factory=list)       # list[Leg]
    transfers: list = field(default_factory=list)  # list[Transfer]

    @property
    def depart_s(self) -> int:
        return self.legs[0].board_time_s

    @property
    def arrive_s(self) -> int:
        return self.legs[-1].alight_time_s

    @property
    def total_seconds(self) -> int:
        return self.arrive_s - self.depart_s

    @property
    def num_transfers(self) -> int:
        return len(self.legs) - 1

    # --- Backward-compatible single-transfer accessors -------------------
    @property
    def leg1(self) -> Leg:
        return self.legs[0]

    @property
    def leg2(self) -> Leg:
        # For multi-leg journeys this is the LAST leg, which is what the old
        # single-transfer UI treated as the "arriving" leg.
        return self.legs[-1]

    @property
    def transfer_seconds(self) -> int:
        return self.transfers[0].walk_seconds if self.transfers else 0

    @property
    def transfer_is_published(self) -> bool:
        return self.transfers[0].is_published if self.transfers else True

    @property
    def transfer_wait_seconds(self) -> int:
        return self.transfers[0].total_gap_seconds if self.transfers else 0


# ---------------------------------------------------------------------------
# Conversion from network.GeneralJourney
# ---------------------------------------------------------------------------
def _from_general(gj: "net.GeneralJourney") -> Journey:
    legs = [
        Leg(
            route_id=rl.route_id,
            route_name=net.route_display(rl.route_id),
            board_station=rl.from_station,
            board_time_s=rl.board_s,
            alight_station=rl.to_station,
            alight_time_s=rl.alight_s,
            trip_id=rl.trip_id,
        )
        for rl in gj.legs
    ]
    transfers = []
    for i, t in enumerate(gj.transfers):
        transfers.append(
            Transfer(
                at_station=t.at_station,
                walk_seconds=t.walk_seconds,
                is_published=t.is_published,
                prev_alight_s=legs[i].alight_time_s,
                next_board_s=legs[i + 1].board_time_s,
            )
        )
    return Journey(legs=legs, transfers=transfers)


# ---------------------------------------------------------------------------
# Public planning API
# ---------------------------------------------------------------------------
def supported_journey(origin_name: str, destination_name: str) -> bool:
    """True if both stations are in the supported network and are different."""
    oid = net.station_id(origin_name)
    did = net.station_id(destination_name)
    return bool(oid and did and oid != did)


def plan_journeys(
    origin_name: str,
    destination_name: str,
    travel_date: datetime.date,
    earliest_departure: datetime.time,
    max_results: int = 3,
) -> list[Journey]:
    """Up to `max_results` scheduled journeys departing at/after the time."""
    oid = net.station_id(origin_name)
    did = net.station_id(destination_name)
    if not oid or not did:
        raise gtfs.GTFSError("Could not find one of the selected stations in the data.")
    if oid == did:
        raise gtfs.GTFSError("Origin and destination are the same station.")

    earliest_s = (
        earliest_departure.hour * 3600
        + earliest_departure.minute * 60
        + earliest_departure.second
    )
    general = net.search_journeys(oid, did, travel_date, earliest_s, max_results)
    return [_from_general(g) for g in general]


def plan_journey(
    origin_name: str,
    destination_name: str,
    travel_date: datetime.date,
    earliest_departure: datetime.time,
) -> Journey:
    """
    Best single scheduled journey (fewest transfers, earliest arrival).
    Raises gtfs.GTFSError if none exists.
    """
    results = plan_journeys(origin_name, destination_name, travel_date, earliest_departure, 1)
    if not results:
        raise gtfs.GTFSError(
            f"No scheduled {origin_name} → {destination_name} journey found after "
            f"{earliest_departure:%I:%M %p} on {travel_date:%A, %B %d}."
        )
    return results[0]


def arrive_by(
    origin_name: str,
    destination_name: str,
    travel_date: datetime.date,
    deadline: datetime.time,
    safety_margin_minutes: int = 0,
    max_results: int = 3,
) -> list[Journey]:
    """
    Journeys that arrive on or before `deadline` minus an optional user margin,
    latest-departing first. Reuses the general search. Empty list if none.

    The safety margin is a USER CHOICE, not a statistical guarantee.
    """
    oid = net.station_id(origin_name)
    did = net.station_id(destination_name)
    if not oid or not did:
        raise gtfs.GTFSError("Could not find one of the selected stations in the data.")
    if oid == did:
        raise gtfs.GTFSError("Origin and destination are the same station.")

    service_ids = gtfs.active_service_ids(travel_date)
    if not service_ids:
        raise gtfs.GTFSError(
            f"No MBTA service is scheduled on {travel_date:%A, %B %d, %Y} in this dataset."
        )

    deadline_s = deadline.hour * 3600 + deadline.minute * 60 + deadline.second
    effective_deadline_s = deadline_s - max(0, safety_margin_minutes) * 60

    # Probe candidate departure windows up to the deadline and keep journeys
    # that arrive in time; then pick the latest departures. Reuses
    # search_journeys. We probe a ~2-hour window at 10-minute granularity,
    # which is plenty for frequent subway service and keeps this responsive.
    candidates: dict[tuple, Journey] = {}
    window_start = max(0, effective_deadline_s - 2 * 3600)
    step = 10 * 60
    t = window_start
    while t <= effective_deadline_s:
        general = net.search_journeys(oid, did, travel_date, t, max_results=3)
        for g in general:
            if g.arrive_s <= effective_deadline_s:
                j = _from_general(g)
                sig = (j.depart_s, j.arrive_s, j.num_transfers)
                candidates[sig] = j
        t += step

    if not candidates:
        return []

    journeys = sorted(candidates.values(), key=lambda j: j.depart_s, reverse=True)
    return journeys[:max_results]


# ---------------------------------------------------------------------------
# Backward-compatible helpers used by reliability.py and app.py
# ---------------------------------------------------------------------------
@dataclass
class LegTripInfo:
    trip_id: str | None
    direction_id: int | None
    board_stop_id: str | None


def _best_leg(
    route_id: str,
    route_name: str,
    from_station_id: str,
    to_station_id: str,
    earliest_departure_s: int,
    service_ids: set[str],
) -> Leg | None:
    """
    Earliest single-vehicle ride on `route_id` from from_station to to_station
    at/after earliest_departure_s, correct direction. Used by the delay
    simulator to find the next feasible service. Reuses network.earliest_leg
    but constrains to the given route_id.
    """
    skey = tuple(sorted(service_ids))
    # network.earliest_leg searches all routes; we filter to the requested one
    # by scanning boardings on that route via a small specialized loop.
    import bisect
    boardings = net._station_boardings(skey).get(from_station_id, [])
    seqs = net._trip_sequences()
    trip_route = net._trip_route()
    i = bisect.bisect_left(boardings, (earliest_departure_s, "", 0))
    best: Leg | None = None
    for dep_s, trip_id, from_seq in boardings[i:i + 120]:
        if best is not None and dep_s > best.board_time_s:
            break
        if trip_route.get(trip_id) != route_id:
            continue
        for seq, station, _d, arr in seqs.get(trip_id, []):
            if seq <= from_seq or arr is None:
                continue
            if station == to_station_id:
                if best is None or dep_s < best.board_time_s:
                    best = Leg(route_id, route_name, from_station_id, int(dep_s),
                               to_station_id, int(arr), trip_id=trip_id)
                break
    return best


def leg_trip_info(
    route_id: str,
    from_station_id: str,
    to_station_id: str,
    board_time_s: int,
    service_ids: set[str],
) -> LegTripInfo:
    """
    Recover the GTFS trip_id / direction_id / boarding platform for a leg that
    boards at `board_time_s`. Used to match live predictions.
    """
    data = gtfs._read_all()
    trips = data["trips"]
    stop_times = data["stop_times"]

    from_platforms = gtfs.child_platform_ids(from_station_id)
    to_platforms = gtfs.child_platform_ids(to_station_id)
    route_trips = trips[
        (trips["route_id"] == route_id) & (trips["service_id"].isin(service_ids))
    ]
    trip_direction = dict(zip(route_trips["trip_id"], route_trips["direction_id"]))
    valid = set(route_trips["trip_id"])
    if not valid:
        return LegTripInfo(None, None, None)

    rel = stop_times[
        stop_times["trip_id"].isin(valid)
        & (stop_times["stop_id"].isin(from_platforms) | stop_times["stop_id"].isin(to_platforms))
    ].copy()
    rel["seq"] = rel["stop_sequence"].astype(int)
    rel["dep_s"] = rel["departure_time"].map(gtfs.gtfs_time_to_seconds)

    for trip_id, group in rel.groupby("trip_id"):
        board = group[group["stop_id"].isin(from_platforms)]
        alight = group[group["stop_id"].isin(to_platforms)]
        if board.empty or alight.empty:
            continue
        b = board.iloc[0]
        a = alight.iloc[0]
        if b["seq"] >= a["seq"]:
            continue
        if b["dep_s"] == board_time_s:
            d = trip_direction.get(trip_id)
            return LegTripInfo(
                trip_id=trip_id,
                direction_id=int(d) if d not in (None, "") else None,
                board_stop_id=b["stop_id"],
            )
    return LegTripInfo(None, None, None)


def leg_segment_stops(trip_id, from_station_id, to_station_id) -> set:
    """Stops a leg actually passes (platforms + parents), for alert relevance."""
    data = gtfs._read_all()
    stops = data["stops"]
    stop_times = data["stop_times"]
    parent_of = dict(zip(stops["stop_id"], stops["parent_station"]))

    result: set[str] = set()
    for station in (from_station_id, to_station_id):
        result.add(station)
        result |= gtfs.child_platform_ids(station)

    if trip_id:
        from_platforms = gtfs.child_platform_ids(from_station_id)
        to_platforms = gtfs.child_platform_ids(to_station_id)
        trip_rows = stop_times[stop_times["trip_id"] == trip_id].copy()
        if not trip_rows.empty:
            trip_rows["seq"] = trip_rows["stop_sequence"].astype(int)
            b = trip_rows[trip_rows["stop_id"].isin(from_platforms)]
            a = trip_rows[trip_rows["stop_id"].isin(to_platforms)]
            if not b.empty and not a.empty:
                lo = int(b.iloc[0]["seq"])
                hi = int(a.iloc[0]["seq"])
                on_seg = trip_rows[(trip_rows["seq"] >= lo) & (trip_rows["seq"] <= hi)]
                for sid in on_seg["stop_id"]:
                    result.add(sid)
                    p = parent_of.get(sid)
                    if p:
                        result.add(p)
    return result
