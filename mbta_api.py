"""
mbta_api.py
-----------
Fetches LIVE data from the official MBTA V3 API (https://api-v3.mbta.com):
real-time predictions and service alerts.

Design goals for a beginner-friendly, robust prototype:
  - It NEVER crashes the app. Every network call is wrapped so that on any
    failure it returns empty data plus an error note. The GTFS journey planner
    and delay simulator keep working with no live data.
  - Short timeouts and light caching (so moving the delay slider does not spam
    the API).
  - Clear provenance: everything here is LIVE data, tagged as such. We never
    invent a prediction and never claim a statistical probability.

The MBTA V3 API works without an API key at a low rate limit, which is fine for
a prototype. If you have a key, set the environment variable MBTA_API_KEY and
it will be sent automatically (higher rate limit).

What we learned from inspecting real responses:
  prediction.attributes: arrival_time, departure_time (ISO 8601 w/ TZ or null),
      arrival_uncertainty, departure_uncertainty, direction_id,
      schedule_relationship (e.g. "CANCELLED"), status, stop_sequence,
      trip_headsign, update_type
  prediction.relationships: route.data.id, stop.data.id, trip.data.id, vehicle
  alert.attributes: header, short_header, description, effect, severity,
      lifecycle, active_period[], service_effect, url, informed_entity[]
  alert.informed_entity: {stop, direction_id, route, route_type, activities[]}
"""

from __future__ import annotations

import datetime
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

API_BASE = "https://api-v3.mbta.com"
REQUEST_TIMEOUT_S = 5          # keep the UI responsive even if the API is slow
CACHE_TTL_S = 20               # re-use responses for 20s (slider-friendly)

# The MBTA runs on America/New_York (US/Eastern). The API already returns times
# with the correct offset, so we read the offset from each timestamp. We also
# need "now" in Eastern to decide whether a journey is in the past / today.
MBTA_TIMEZONE_LABEL = "America/New_York"

# How close (in minutes) a live predicted departure must be to the scheduled
# departure to count as "the same journey" when we don't have an exact trip_id.
MATCH_WINDOW_MINUTES = 10


def eastern_now() -> datetime.datetime:
    """
    Current time in America/New_York.

    We avoid a hard dependency on the tz database: the API gives us timestamps
    that already carry the Eastern offset, and we only need 'now' to compare
    against them. We derive Eastern 'now' from the local system clock's UTC
    time plus the offset reported by a fresh API-style timestamp when possible;
    as a simple, dependency-free approach we use the standard library's
    zoneinfo if available, otherwise fall back to the system local time.
    """
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+, standard library
        return datetime.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        # Fallback: system local time (the user is expected to be in Boston for
        # this prototype). Clearly a fallback; still safe for comparisons.
        return datetime.datetime.now()


class MBTAAPIError(Exception):
    """Internal: raised inside this module and converted to a friendly note."""


# ---------------------------------------------------------------------------
# Simple time-based cache so repeated reruns (e.g. slider drags) don't hammer
# the API. Keyed by the full request URL.
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[float, dict]] = {}


def _request(path: str, params: dict) -> dict:
    """
    Make a GET request to the MBTA API and return the parsed JSON.
    Raises MBTAAPIError on any problem (network, timeout, bad status, bad JSON).
    """
    query = urllib.parse.urlencode(params)
    url = f"{API_BASE}{path}?{query}"

    # Serve from cache if fresh.
    now = time.time()
    cached = _cache.get(url)
    if cached and (now - cached[0]) < CACHE_TTL_S:
        return cached[1]

    headers = {"Accept": "application/vnd.api+json"}
    api_key = os.environ.get("MBTA_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            if resp.status != 200:
                raise MBTAAPIError(f"MBTA API returned HTTP {resp.status}")
            payload = json.loads(resp.read().decode("utf-8"))
    except MBTAAPIError:
        raise
    except Exception as exc:  # network error, timeout, JSON error, etc.
        raise MBTAAPIError(f"Could not reach the MBTA API: {exc}") from exc

    _cache[url] = (now, payload)
    return payload


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def iso_to_datetime(iso_string: str | None) -> datetime.datetime | None:
    """Parse an MBTA ISO timestamp (with offset) into a full datetime."""
    if not iso_string:
        return None
    try:
        return datetime.datetime.fromisoformat(iso_string)
    except ValueError:
        return None


def iso_to_seconds_of_day(iso_string: str | None) -> int | None:
    """
    Convert an MBTA ISO timestamp (e.g. '2026-10-24T14:02:24-04:00') into
    seconds past midnight in its own local timezone, so it lines up with the
    GTFS seconds-past-midnight model used elsewhere in the app.
    """
    dt = iso_to_datetime(iso_string)
    if dt is None:
        return None
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def now_timestamp_label() -> str:
    """A human-friendly 'data fetched at' label in Eastern time, with zone."""
    return eastern_now().strftime("%I:%M:%S %p America/New_York")


# ---------------------------------------------------------------------------
# Data classes for the parsed live values (all clearly LIVE).
# ---------------------------------------------------------------------------
@dataclass
class LivePrediction:
    """One live prediction matched to a route/stop/direction."""
    route_id: str
    stop_id: str
    trip_id: str | None
    direction_id: int | None
    arrival_iso: str | None
    departure_iso: str | None
    departure_seconds: int | None       # seconds past midnight, local
    arrival_seconds: int | None
    schedule_relationship: str | None    # e.g. "CANCELLED" or None (normal)
    status: str | None
    departure_date: datetime.date | None = None   # the Eastern calendar date


@dataclass
class LiveAlert:
    """One service alert relevant to our journey."""
    alert_id: str
    header: str
    effect: str
    severity: int | None
    lifecycle: str | None
    url: str | None
    routes: list[str] = field(default_factory=list)
    stops: list[str] = field(default_factory=list)
    active_periods: list[tuple] = field(default_factory=list)  # (start_dt, end_dt)
    journey_specific: bool = False   # touches one of our journey's stops
    # Raw informed entities: list of {"route","stop","direction_id"} dicts, kept
    # so we can decide journey relevance honestly from the actual API data.
    informed_entities: list[dict] = field(default_factory=list)
    relevance: str = "other"         # "segment" | "route" | "other"

    def is_active_on(self, when: datetime.datetime) -> bool:
        """
        True if the alert has an active period covering `when` (or has no
        active periods listed, which the MBTA treats as 'currently active').
        Comparison is timezone-aware when the data provides offsets.
        """
        if not self.active_periods:
            return True
        for start, end in self.active_periods:
            after_start = (start is None) or (when >= start)
            before_end = (end is None) or (when <= end)
            if after_start and before_end:
                return True
        return False


@dataclass
class LiveResult:
    """Everything the live layer returns, plus freshness and error info."""
    predictions: list[LivePrediction] = field(default_factory=list)
    alerts: list[LiveAlert] = field(default_factory=list)
    fetched_at: str | None = None
    ok: bool = True
    error_note: str | None = None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def get_predictions(
    route_id: str,
    stop_ids: list[str],
    direction_id: int | None = None,
) -> tuple[list[LivePrediction], str | None]:
    """
    Fetch live predictions for a route at one or more platform stop IDs.
    Returns (list_of_predictions, error_note). On failure, returns ([], note).
    """
    params = {
        "filter[route]": route_id,
        "filter[stop]": ",".join(stop_ids),
        "sort": "departure_time",
        "page[limit]": "10",
    }
    if direction_id is not None:
        params["filter[direction_id]"] = str(direction_id)

    try:
        payload = _request("/predictions", params)
    except MBTAAPIError as exc:
        return [], str(exc)

    predictions: list[LivePrediction] = []
    for item in payload.get("data", []):
        attrs = item.get("attributes", {})
        rels = item.get("relationships", {})

        def rel_id(name: str) -> str | None:
            node = rels.get(name, {}).get("data")
            return node.get("id") if node else None

        dep_dt = iso_to_datetime(attrs.get("departure_time")) or iso_to_datetime(
            attrs.get("arrival_time")
        )
        predictions.append(
            LivePrediction(
                route_id=rel_id("route") or route_id,
                stop_id=rel_id("stop") or "",
                trip_id=rel_id("trip"),
                direction_id=attrs.get("direction_id"),
                arrival_iso=attrs.get("arrival_time"),
                departure_iso=attrs.get("departure_time"),
                departure_seconds=iso_to_seconds_of_day(attrs.get("departure_time")),
                arrival_seconds=iso_to_seconds_of_day(attrs.get("arrival_time")),
                schedule_relationship=attrs.get("schedule_relationship"),
                status=attrs.get("status"),
                departure_date=dep_dt.date() if dep_dt else None,
            )
        )
    return predictions, None


def get_alerts(
    route_ids: list[str],
    stop_ids: list[str] | None = None,
) -> tuple[list[LiveAlert], str | None]:
    """
    Fetch active service alerts for the given routes (and optionally stops).
    Returns (list_of_alerts, error_note). On failure, returns ([], note).
    """
    params = {
        "filter[route]": ",".join(route_ids),
        "page[limit]": "20",
    }
    try:
        payload = _request("/alerts", params)
    except MBTAAPIError as exc:
        return [], str(exc)

    wanted_stops = set(stop_ids or [])
    alerts: list[LiveAlert] = []
    for item in payload.get("data", []):
        attrs = item.get("attributes", {})
        informed = attrs.get("informed_entity", []) or []

        entity_routes = sorted({e.get("route") for e in informed if e.get("route")})
        entity_stops = sorted({e.get("stop") for e in informed if e.get("stop")})

        touches_stop = bool(wanted_stops & set(entity_stops))
        touches_route = bool(set(route_ids) & set(entity_routes))

        # If specific stops were requested, keep only alerts that touch our
        # routes generally OR one of our stops specifically.
        if wanted_stops and not (touches_stop or touches_route):
            continue

        # Parse active periods into (start, end) datetime tuples.
        periods = []
        for ap in attrs.get("active_period", []) or []:
            periods.append(
                (iso_to_datetime(ap.get("start")), iso_to_datetime(ap.get("end")))
            )

        # Keep the raw informed entities so journey relevance can be judged
        # honestly from the real data (route + stop + direction).
        entities = [
            {
                "route": e.get("route"),
                "stop": e.get("stop"),
                "direction_id": e.get("direction_id"),
            }
            for e in informed
        ]

        alerts.append(
            LiveAlert(
                alert_id=item.get("id", ""),
                header=attrs.get("header") or attrs.get("short_header") or "(no title)",
                effect=attrs.get("effect") or "UNKNOWN",
                severity=attrs.get("severity"),
                lifecycle=attrs.get("lifecycle"),
                url=attrs.get("url"),
                routes=entity_routes,
                stops=entity_stops,
                active_periods=periods,
                journey_specific=touches_stop,
                informed_entities=entities,
            )
        )
    return alerts, None


def classify_alert_relevance(
    alert: LiveAlert,
    segment_stop_ids: set[str],
    leg_routes_directions: list[tuple],
) -> str:
    """
    Decide how a service alert relates to the SELECTED journey, using only the
    alert's actual informed_entity data. Returns one of:

      "segment" : the alert names a stop the rider actually passes (strongest,
                  journey-specific evidence), OR a route+direction that matches
                  one of the journey's legs.
      "route"   : the alert names one of the journey's routes but with no stop
                  we traverse and no matching direction -> line-level only.
      "other"   : the alert is for a different route/stop entirely.

    `leg_routes_directions` is a list of (route_id, direction_id) for the legs,
    e.g. [("Red", 0), ("Orange", 0)].

    We never claim segment relevance unless the data supports it. When only
    route-level evidence exists, we label it "route" so the UI can say so.
    """
    leg_routes = {r for r, _ in leg_routes_directions}
    route_dir_pairs = set(leg_routes_directions)

    saw_route = False
    for e in alert.informed_entities:
        route = e.get("route")
        stop = e.get("stop")
        direction = e.get("direction_id")

        # Strongest evidence: this entity names a stop we actually pass through.
        if stop and stop in segment_stop_ids:
            return "segment"

        # If this entity names a stop that is NOT on our path (e.g. an Oak Grove
        # or Braintree parking notice), it is NOT journey-specific — even if the
        # route and direction happen to match. Do not treat it as "segment".
        if stop and stop not in segment_stop_ids:
            if route in leg_routes:
                saw_route = True
            continue

        # No stop named on this entity -> it is line-wide. Count route+direction
        # on our line as journey-relevant (our direction of travel is affected),
        # otherwise it is route-level evidence only.
        if route in leg_routes:
            saw_route = True
            if direction is not None and (route, direction) in route_dir_pairs:
                return "segment"

    return "route" if saw_route else "other"


@dataclass
class MatchResult:
    """
    The outcome of trying to match live predictions to a SELECTED scheduled leg.

    - prediction: the matched LivePrediction, or None.
    - matched: True only when we are confident the prediction is the same trip.
    - reason: a short machine-readable reason, one of:
        "trip_exact"         -> exact GTFS trip_id match
        "time_window"        -> same service date and within the match window
        "no_predictions"     -> the API returned nothing for this stop/route
        "trip_in_past"       -> the selected trip is earlier today / a past date
        "future_date"        -> the selected date is not today (no live yet)
        "out_of_window"      -> predictions exist but none near the scheduled time
    """
    prediction: "LivePrediction | None"
    matched: bool
    reason: str


def match_prediction_to_leg(
    predictions: list[LivePrediction],
    stop_ids: set[str],
    direction_id: int | None,
    trip_id: str | None,
    scheduled_departure_seconds: int,
    service_date: datetime.date,
    now_eastern: datetime.datetime | None = None,
) -> MatchResult:
    """
    Decide whether any live prediction genuinely corresponds to the SELECTED
    scheduled leg. This is intentionally strict so we never show an unrelated
    "next train" as if it were the chosen 9 AM trip.

    A prediction matches only if:
      1. its trip_id equals the scheduled trip_id (the same train), OR
      2. it is for TODAY (Eastern) and its departure is within
         MATCH_WINDOW_MINUTES of the scheduled departure.

    If the selected trip is in the past (earlier today or a past date), or the
    date is in the future, we return matched=False with a clear reason instead
    of substituting the current next train.
    """
    now_eastern = now_eastern or eastern_now()
    today = now_eastern.date()

    # Predictions that concern this leg's stop + direction at all.
    def concerns_leg(p: LivePrediction) -> bool:
        if p.stop_id not in stop_ids:
            return False
        if direction_id is not None and p.direction_id not in (None, direction_id):
            return False
        return True

    leg_predictions = [p for p in predictions if concerns_leg(p)]

    # 1. Exact trip match always wins (and is always correct).
    if trip_id:
        for p in leg_predictions:
            if p.trip_id == trip_id:
                return MatchResult(p, True, "trip_exact")

    # 2. Time-window match, but only for TODAY.
    if service_date != today:
        return MatchResult(None, False, "future_date")

    # The selected trip's absolute moment today.
    now_seconds = (
        now_eastern.hour * 3600 + now_eastern.minute * 60 + now_eastern.second
    )
    # If the scheduled departure already passed by more than the window, the
    # live stream (which only looks forward) cannot represent that trip.
    if scheduled_departure_seconds < now_seconds - MATCH_WINDOW_MINUTES * 60:
        return MatchResult(None, False, "trip_in_past")

    if not leg_predictions:
        return MatchResult(None, False, "no_predictions")

    window = MATCH_WINDOW_MINUTES * 60
    best = None
    best_gap = None
    for p in leg_predictions:
        if p.departure_seconds is None:
            continue
        if p.departure_date is not None and p.departure_date != service_date:
            continue
        gap = abs(p.departure_seconds - scheduled_departure_seconds)
        if gap <= window and (best_gap is None or gap < best_gap):
            best, best_gap = p, gap

    if best is not None:
        return MatchResult(best, True, "time_window")
    return MatchResult(None, False, "out_of_window")


def next_trains_now(
    predictions: list[LivePrediction],
    stop_ids: set[str],
    direction_id: int | None,
    limit: int = 3,
) -> list[LivePrediction]:
    """
    The upcoming live departures at a stop RIGHT NOW (not tied to the selected
    journey). Used only for the clearly-labeled 'Next trains right now' section.
    """
    def concerns(p: LivePrediction) -> bool:
        if p.stop_id not in stop_ids:
            return False
        if direction_id is not None and p.direction_id not in (None, direction_id):
            return False
        return p.departure_seconds is not None and p.schedule_relationship != "CANCELLED"

    rows = [p for p in predictions if concerns(p)]
    rows.sort(key=lambda p: p.departure_seconds)
    return rows[:limit]


def get_live_for_journey(
    red_board_stop_ids: list[str],
    red_direction_id: int | None,
    red_trip_id: str | None,
    orange_board_stop_ids: list[str],
    orange_direction_id: int | None,
    orange_trip_id: str | None,
    transfer_stop_ids: list[str],
) -> LiveResult:
    """
    One call that gathers everything the UI needs for the current journey:
    Red predictions at the origin, Orange predictions at the transfer, and
    alerts for both lines plus the transfer station.

    Always returns a LiveResult. If the API fails, ok=False and error_note is
    set, but the object is still safe to render.
    """
    result = LiveResult()

    red_preds, red_err = get_predictions("Red", red_board_stop_ids, red_direction_id)
    orange_preds, orange_err = get_predictions(
        "Orange", orange_board_stop_ids, orange_direction_id
    )
    alerts, alert_err = get_alerts(["Red", "Orange"], transfer_stop_ids)

    result.predictions = red_preds + orange_preds
    result.alerts = alerts
    result.fetched_at = now_timestamp_label()

    errors = [e for e in (red_err, orange_err, alert_err) if e]
    if errors and not (red_preds or orange_preds or alerts):
        # Everything failed -> live data is unavailable.
        result.ok = False
        result.error_note = errors[0]
    elif errors:
        # Partial failure -> keep what we have but note it.
        result.ok = True
        result.error_note = "Some live data could not be loaded: " + errors[0]

    return result
