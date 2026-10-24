"""
reliability.py
--------------
The TransitWise AI "Connection Confidence" and delay simulator logic.

This module answers one question: *if your first train is delayed, will you
still make your connection, and when will you actually arrive?*

Everything here is deterministic and schedule-based:
  - It uses the real GTFS schedule for the next Orange Line train.
  - It does NOT use statistics, history, or probability.
  - The "risk" labels are simple prototype categories based on how much
    connection buffer is left, not a measured chance of success.

Key terms (all in seconds unless noted):
  - walk_seconds        : minimum walking time between platforms (transfers.txt)
  - scheduled buffer    : orange_departure - red_arrival - walk_seconds
                          (spare time after walking, before the train leaves)
  - delay_seconds       : hypothetical delay applied to the first train
  - remaining buffer    : scheduled buffer - delay_seconds
                          (if this goes negative, the connection is missed)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import gtfs_loader as gtfs
import journey_planner as jp


# ---------------------------------------------------------------------------
# Risk thresholds (documented PROTOTYPE values, in seconds).
# These are deterministic, timing-based buffer categories, NOT probabilities.
#
# Exact boundary rules (buffer measured in whole minutes of spare time):
#   Risky           : buffer < 3 minutes
#   Medium          : buffer from 3 through 6 minutes (both ends inclusive)
#   High confidence : buffer > 6 minutes
#
# So exactly 3 min -> Medium, and exactly 6 min -> Medium.
# ---------------------------------------------------------------------------
RISKY_BELOW_S = 3 * 60      # strictly below 3 min -> "Risky"
MEDIUM_MAX_S = 6 * 60       # up to and including 6 min -> "Medium"
# strictly above 6 min                             -> "High confidence"


@dataclass
class ConnectionResult:
    """The full result of simulating a delay on the first train."""
    # Scheduled (no delay) facts
    scheduled_red_arrival_s: int
    walk_seconds: int
    walk_is_published: bool
    scheduled_orange_departure_s: int
    scheduled_buffer_s: int
    scheduled_final_arrival_s: int

    # Simulated (with delay) facts
    delay_seconds: int
    delayed_red_arrival_s: int
    ready_to_board_s: int          # when you reach the Orange platform
    connection_made: bool

    # Which Orange train you actually catch (scheduled one, or a later real one)
    actual_orange_departure_s: int | None
    actual_final_arrival_s: int | None
    caught_later_train: bool
    no_onward_service: bool

    # Buffer left against the train you are trying to catch
    remaining_buffer_s: int

    @property
    def added_delay_to_arrival_s(self) -> int:
        """How much later you arrive compared to the scheduled plan."""
        if self.actual_final_arrival_s is None:
            return 0
        return self.actual_final_arrival_s - self.scheduled_final_arrival_s


def risk_label(buffer_s: int) -> tuple[str, str]:
    """
    Turn a connection-buffer value (in seconds) into a prototype confidence
    label. This is the SINGLE source of truth for risk classification, used by
    both the scheduled-buffer display and the delay simulator.

    Boundaries (inclusive as noted):
        buffer < 3 min            -> "Risky"
        3 min <= buffer <= 6 min  -> "Medium"   (exactly 3 and exactly 6 = Medium)
        buffer > 6 min            -> "High confidence"

    These are deterministic, timing-based categories only — never a measured
    probability of making the connection.

    Returns (label, emoji).
    """
    if buffer_s < RISKY_BELOW_S:
        return "Risky", "🔴"
    if buffer_s <= MEDIUM_MAX_S:
        return "Medium", "🟡"
    return "High confidence", "🟢"


def _next_leg_after(
    route_id: str,
    route_name: str,
    from_station_id: str,
    to_station_id: str,
    ready_s: int,
    service_ids: set[str],
) -> jp.Leg | None:
    """
    Next real trip on `route_id` (correct direction/date) departing
    from_station at or after ready_s and reaching to_station. Reuses the
    planner's verified leg search, so there is no invented waiting penalty.
    """
    return jp._best_leg(
        route_id, route_name, from_station_id, to_station_id, ready_s, service_ids
    )


# ---------------------------------------------------------------------------
# General multi-transfer delay propagation (Phase 3).
# ---------------------------------------------------------------------------
@dataclass
class TransferOutcome:
    """What happens at one transfer once an upstream delay is applied."""
    at_station: str
    walk_seconds: int
    walk_is_published: bool
    scheduled_buffer_s: int      # scheduled: next_board - prev_alight - walk
    ready_to_board_s: int        # when you reach the platform (with delay)
    scheduled_next_board_s: int  # the originally-planned next train
    actual_next_board_s: int | None
    connection_made: bool
    caught_later_train: bool
    remaining_buffer_s: int      # against the train you actually catch


@dataclass
class MultiResult:
    """Delay simulation across the whole (possibly multi-transfer) journey."""
    delay_seconds: int
    scheduled_final_arrival_s: int
    actual_final_arrival_s: int | None
    transfers: list              # list[TransferOutcome]
    no_onward_service: bool

    @property
    def added_delay_to_arrival_s(self) -> int:
        if self.actual_final_arrival_s is None:
            return 0
        return self.actual_final_arrival_s - self.scheduled_final_arrival_s

    @property
    def any_connection_missed(self) -> bool:
        return any(not t.connection_made for t in self.transfers)


def simulate_delay_multi(
    journey: jp.Journey,
    travel_date: datetime.date,
    delay_minutes: int,
) -> MultiResult:
    """
    Apply a delay to the FIRST train and propagate it through every transfer.
    At each transfer, if the delayed arrival (plus walk) misses the scheduled
    train, roll forward to the next real scheduled service on that leg's route.
    The knock-on delay carries into subsequent transfers.

    Works for 0, 1 or 2 transfers. Never invents a waiting penalty.
    """
    delay_seconds = max(0, int(delay_minutes) * 60)
    service_ids = gtfs.active_service_ids(travel_date)

    scheduled_final = journey.arrive_s
    outcomes: list[TransferOutcome] = []

    # Running alight time of the current (possibly delayed) leg.
    current_alight_s = journey.legs[0].alight_time_s + delay_seconds
    no_onward = False

    for i, transfer in enumerate(journey.transfers):
        next_leg_sched = journey.legs[i + 1]
        walk_s = transfer.walk_seconds
        scheduled_buffer_s = (
            next_leg_sched.board_time_s
            - journey.legs[i].alight_time_s
            - walk_s
        )
        ready_s = current_alight_s + walk_s
        scheduled_board = next_leg_sched.board_time_s
        made = ready_s <= scheduled_board

        if made:
            actual_board = scheduled_board
            actual_alight = next_leg_sched.alight_time_s
            caught_later = False
            remaining_buffer = scheduled_board - ready_s
        else:
            replacement = _next_leg_after(
                next_leg_sched.route_id,
                next_leg_sched.route_name,
                next_leg_sched.board_station,
                next_leg_sched.alight_station,
                ready_s,
                service_ids,
            )
            if replacement is None:
                no_onward = True
                outcomes.append(TransferOutcome(
                    at_station=transfer.at_station, walk_seconds=walk_s,
                    walk_is_published=transfer.is_published,
                    scheduled_buffer_s=scheduled_buffer_s,
                    ready_to_board_s=ready_s,
                    scheduled_next_board_s=scheduled_board,
                    actual_next_board_s=None, connection_made=False,
                    caught_later_train=False, remaining_buffer_s=0,
                ))
                return MultiResult(
                    delay_seconds=delay_seconds,
                    scheduled_final_arrival_s=scheduled_final,
                    actual_final_arrival_s=None,
                    transfers=outcomes, no_onward_service=True,
                )
            actual_board = replacement.board_time_s
            actual_alight = replacement.alight_time_s
            caught_later = True
            remaining_buffer = actual_board - ready_s

        outcomes.append(TransferOutcome(
            at_station=transfer.at_station, walk_seconds=walk_s,
            walk_is_published=transfer.is_published,
            scheduled_buffer_s=scheduled_buffer_s,
            ready_to_board_s=ready_s,
            scheduled_next_board_s=scheduled_board,
            actual_next_board_s=actual_board,
            connection_made=made, caught_later_train=caught_later,
            remaining_buffer_s=remaining_buffer,
        ))
        current_alight_s = actual_alight  # propagate to next transfer

    actual_final = current_alight_s if journey.transfers else (
        journey.legs[-1].alight_time_s + delay_seconds
    )
    return MultiResult(
        delay_seconds=delay_seconds,
        scheduled_final_arrival_s=scheduled_final,
        actual_final_arrival_s=actual_final,
        transfers=outcomes, no_onward_service=no_onward,
    )


def simulate_delay(
    journey: jp.Journey,
    destination_name: str,
    travel_date: datetime.date,
    delay_minutes: int,
) -> ConnectionResult:
    """
    Apply a hypothetical delay (0-20 min) to the first train and work out:
      - the new arrival at the transfer station
      - whether the scheduled Orange connection is still feasible
      - if missed, the NEXT real Orange trip from the schedule
      - the resulting final arrival time

    All times stay in GTFS seconds-past-midnight, so values beyond 24:00:00
    are handled naturally, and we never produce negative waiting times.
    """
    # If the journey has no transfer (direct trip), the delay simply shifts
    # arrival; there is no connection to make.
    if not journey.transfers:
        delay_seconds = max(0, int(delay_minutes) * 60)
        sched_arr = journey.arrive_s
        return ConnectionResult(
            scheduled_red_arrival_s=journey.legs[0].alight_time_s,
            walk_seconds=0, walk_is_published=True,
            scheduled_orange_departure_s=journey.legs[0].alight_time_s,
            scheduled_buffer_s=0, scheduled_final_arrival_s=sched_arr,
            delay_seconds=delay_seconds,
            delayed_red_arrival_s=journey.legs[0].alight_time_s + delay_seconds,
            ready_to_board_s=journey.legs[0].alight_time_s + delay_seconds,
            connection_made=True,
            actual_orange_departure_s=journey.legs[0].alight_time_s + delay_seconds,
            actual_final_arrival_s=sched_arr + delay_seconds,
            caught_later_train=False, no_onward_service=False,
            remaining_buffer_s=0,
        )

    # General propagation across all transfers, then expose the FIRST transfer
    # in the legacy single-transfer shape the current UI expects. The final
    # arrival reflects knock-on delays through every transfer.
    multi = simulate_delay_multi(journey, travel_date, delay_minutes)
    first = multi.transfers[0]

    return ConnectionResult(
        scheduled_red_arrival_s=journey.legs[0].alight_time_s,
        walk_seconds=first.walk_seconds,
        walk_is_published=first.walk_is_published,
        scheduled_orange_departure_s=first.scheduled_next_board_s,
        scheduled_buffer_s=first.scheduled_buffer_s,
        scheduled_final_arrival_s=multi.scheduled_final_arrival_s,
        delay_seconds=multi.delay_seconds,
        delayed_red_arrival_s=journey.legs[0].alight_time_s + multi.delay_seconds,
        ready_to_board_s=first.ready_to_board_s,
        connection_made=first.connection_made,
        actual_orange_departure_s=first.actual_next_board_s,
        actual_final_arrival_s=multi.actual_final_arrival_s,
        caught_later_train=first.caught_later_train,
        no_onward_service=multi.no_onward_service,
        remaining_buffer_s=first.remaining_buffer_s,
    )
