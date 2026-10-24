"""
TransitWise AI - reliability-aware MBTA subway journey planner
-------------------------------------------------------------
Plans real scheduled journeys across all four MBTA subway lines from the
official GTFS dataset: the Red, Orange, Blue and Green lines (including the
Green Line B, C, D and E branches). It supports direct trips and journeys with
one or two transfers, in both directions, via "Depart at" or "Arrive by".

Key features:
  - Connection Confidence: the usable buffer at each transfer, with a
    deterministic timing-based label (Risky / Medium / High).
  - Delay simulator: a manual what-if that propagates a first-train delay
    through every transfer and rolls to the next real scheduled service.
  - Live MBTA V3 predictions and service alerts, matched to the selected trip.
  - Historical reliability from official MBTA LAMP observations.

Data provenance is always kept distinct:
  - Scheduled  = the published GTFS timetable.
  - Live       = the MBTA V3 API right now.
  - Simulated  = your manual what-if delay (not a prediction).
  - Historical = observed MBTA LAMP data for stations with verified records.
Nothing is invented and no probability of making a connection is claimed.
"""

import datetime

import pandas as pd
import streamlit as st

import gtfs_loader as gtfs
import network as net
import journey_planner as jp
import reliability as rel
import mbta_api as api
import historical_reliability as hist
try:
    import ml_model as ml
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TransitWise AI",
    page_icon="🚇",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Header: title + tagline  (unchanged interface)
# ---------------------------------------------------------------------------
st.title("🚇 TransitWise AI")
st.subheader("Know when you'll actually arrive, not just when you're scheduled to.")

st.caption(
    "TransitWise AI combines official MBTA schedules, live transit updates, and "
    "historical performance data to help you plan more reliable journeys across "
    "Boston's Red, Orange, Green, and Blue subway lines."
)
st.caption(
    "Plans direct, one-transfer and two-transfer journeys (Green Line B/C/D/E "
    "included). Scheduled, live, simulated, historical, and experimental figures "
    "are always labeled separately. Historical coverage and live predictions are "
    "available only where verified data exists — never fabricated."
)

st.divider()


# ---------------------------------------------------------------------------
# Load the real station list. If the GTFS files are missing, show a clear
# message instead of crashing.
# ---------------------------------------------------------------------------
try:
    stations_df = gtfs.get_subway_stations()
    station_names = stations_df["stop_name"].tolist()
    data_ok = True
except gtfs.GTFSError as error:
    station_names = []
    data_ok = False
    st.error("Could not load the MBTA data.")
    st.code(str(error))
    st.info(
        "Make sure the GTFS files are extracted into the folder: data/gtfs/\n"
        "(stops.txt, routes.txt, trips.txt, stop_times.txt, calendar.txt, "
        "calendar_dates.txt, transfers.txt)"
    )


def default_index(names: list[str], preferred: str, fallback: int) -> int:
    """Pick a sensible default dropdown index by station name."""
    return names.index(preferred) if preferred in names else fallback


# ---------------------------------------------------------------------------
# Plain-English helpers (presentation only — no new calculations).
# ---------------------------------------------------------------------------
def buffer_plain_english(buffer_seconds: int) -> str:
    """Turn a usable-buffer value into a commuter-friendly sentence.

    Uses the SAME thresholds as rel.risk_label (Risky < 3 min, Medium 3-6 min
    inclusive, High > 6 min) so wording and label never disagree.
    """
    minutes = max(0, round(buffer_seconds / 60))
    spare = "no time" if minutes == 0 else (
        "about 1 minute" if minutes == 1 else f"about {minutes} minutes")
    if buffer_seconds < rel.RISKY_BELOW_S:
        return (f"You'd have {spare} to spare after walking to the next platform. "
                "This is a **tight** connection.")
    if buffer_seconds <= rel.MEDIUM_MAX_S:
        return (f"You'd have {spare} to spare after walking to the next platform. "
                "This is a **moderate** connection.")
    return (f"You'd have {spare} to spare after walking to the next platform. "
            "This is a **comfortable** connection.")


def journey_risk_summary(journey) -> tuple[str, str]:
    """A one-line transfer-risk summary for the recommended-trip card.

    Returns (emoji, text). Uses the tightest transfer's existing risk label.
    """
    if journey.num_transfers == 0:
        return "🟢", "Direct trip — no transfers, no connection to miss."
    tightest = min(journey.transfers, key=lambda t: t.usable_buffer_seconds)
    label, emoji = rel.risk_label(tightest.usable_buffer_seconds)
    mins = max(0, round(tightest.usable_buffer_seconds / 60))
    n = journey.num_transfers
    plural = "transfer" if n == 1 else "transfers"
    word = {"Risky": "tight", "Medium": "moderate",
            "High confidence": "comfortable"}.get(label, label.lower())
    return emoji, (f"{n} {plural}; tightest is **{word}** "
                   f"(~{mins} min to spare at {tightest.at_station_name}).")


# ---------------------------------------------------------------------------
# 2. Your recommended trip — the headline card commuters see first.
# ---------------------------------------------------------------------------
def show_recommended_trip(journey, origin_name, destination_name, travel_date,
                          arrive_info=None):
    st.markdown("## 🧭 Your recommended trip")

    lines_desc = " → ".join(l.route_name for l in journey.legs)
    st.markdown(f"**{origin_name} → {destination_name}**  ·  {lines_desc}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Leave at", gtfs.seconds_to_clock(journey.depart_s))
    m2.metric("Arrive (scheduled)", gtfs.seconds_to_clock(journey.arrive_s))
    m3.metric("Total time", gtfs.seconds_to_duration(journey.total_seconds))

    # Arrive-by: show deadline + available margin (uses existing values only).
    if arrive_info:
        deadline = arrive_info["deadline"]
        margin = arrive_info["margin"]
        deadline_s = deadline.hour * 3600 + deadline.minute * 60
        minutes_early = (deadline_s - journey.arrive_s) // 60
        line = (f"🎯 Deadline **{deadline:%I:%M %p}** — this trip arrives "
                f"**{minutes_early} min early**.")
        if margin:
            line += (f" (Includes your **{margin}-min** personal safety margin — "
                     "your own cushion, not a statistical guarantee.)")
        st.markdown(line)

    # Transfers + simple risk summary.
    emoji, summary = journey_risk_summary(journey)
    st.markdown(f"{emoji} {summary}")

    # Scenario arrival window — ONLY if the existing code can calculate one
    # (a transfer with verified historical data). Never invented.
    window = _scenario_window(journey)
    if window is not None:
        low_s, high_s, station = window
        st.markdown(
            f"🔬 **Scenario arrival window:** "
            f"**{gtfs.seconds_to_clock(low_s)} – {gtfs.seconds_to_clock(high_s)}** "
            f"(a **schedule-based scenario range** from the observed train-gap "
            f"spread at {station} — **not** a statistical confidence interval). "
            "See the historical section for details."
        )
        # Deadline status for the DISPLAYED scenarios (Arrive by only). We
        # compare the latest scenario arrival to the deadline. No numeric
        # confidence level is shown because no journey-level interval has been
        # calibrated/validated on held-out arrival outcomes.
        if arrive_info:
            deadline = arrive_info["deadline"]
            deadline_s = deadline.hour * 3600 + deadline.minute * 60
            if high_s <= deadline_s:
                st.markdown(
                    f"✅ **Within deadline in the displayed scenarios** "
                    f"(latest shown scenario {gtfs.seconds_to_clock(high_s)} ≤ "
                    f"{deadline:%I:%M %p})."
                )
            else:
                st.markdown(
                    f"⚠️ **Some scenarios miss your deadline** "
                    f"(latest shown scenario {gtfs.seconds_to_clock(high_s)} is "
                    f"after {deadline:%I:%M %p})."
                )
            st.caption(
                "Statistical arrival confidence is not yet available — these are "
                "schedule-based scenarios, not a calibrated probability."
            )

    st.caption(f"Scheduled for {travel_date:%A, %B %d, %Y}. Times are from the "
               "official MBTA GTFS timetable.")
    st.divider()


def _scenario_window(journey):
    """
    If a transfer on the journey has VERIFIED historical data, return a coarse
    (low_s, high_s, station_name) scenario arrival window using the SAME
    p25-p90 logic already used in the experimental section. Returns None
    otherwise. Reads only existing historical stats — no new model, no
    fabrication, and never borrows another station's data.
    """
    if journey.num_transfers == 0:
        return None
    try:
        for i, transfer in enumerate(journey.transfers):
            conn = journey.legs[i + 1]
            if not hist.has_data_for(transfer.at_station, conn.route_id):
                continue
            stats = hist.compute_stats(
                conn.route_id, transfer.at_station, transfer.at_station_name
            )
            buffer_min = transfer.usable_buffer_seconds / 60.0
            extra_low = max(0.0, stats.p25_min - buffer_min)
            extra_high = max(0.0, stats.p90_min - buffer_min)
            low_s = journey.arrive_s + int(round(extra_low * 60))
            high_s = journey.arrive_s + int(round(extra_high * 60))
            return (low_s, high_s, transfer.at_station_name)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Render the SCHEDULED journey as a clear timeline (no delay applied).
# ---------------------------------------------------------------------------
def show_scheduled_journey(journey, origin_name, destination_name, travel_date):
    st.success("Here is your scheduled journey, straight from the MBTA timetable:")

    m1, m2, m3 = st.columns(3)
    m1.metric("Depart", gtfs.seconds_to_clock(journey.depart_s))
    m2.metric("Arrive", gtfs.seconds_to_clock(journey.arrive_s))
    m3.metric("Total time", gtfs.seconds_to_duration(journey.total_seconds))

    st.caption(f"Scheduled for {travel_date:%A, %B %d, %Y}")
    st.divider()

    n = journey.num_transfers
    label = "Direct — no transfers" if n == 0 else (
        "1 transfer" if n == 1 else f"{n} transfers")
    st.markdown(f"#### Scheduled journey timeline ({label})")

    # Render every leg, with each transfer shown between consecutive legs.
    for i, leg in enumerate(journey.legs):
        emoji = net.route_emoji(leg.route_id)
        st.markdown(
            f"{emoji} **{leg.route_name}**  \n"
            f"&nbsp;&nbsp;Board at **{leg.board_station_name}** — "
            f"{gtfs.seconds_to_clock(leg.board_time_s)}  \n"
            f"&nbsp;&nbsp;Ride to **{leg.alight_station_name}** — "
            f"{gtfs.seconds_to_clock(leg.alight_time_s)}  \n"
            f"&nbsp;&nbsp;*Travel time: {gtfs.seconds_to_duration(leg.travel_seconds)}*"
        )
        if i < len(journey.transfers):
            t = journey.transfers[i]
            walk_source = (
                "published MBTA transfer time"
                if t.is_published
                else "estimated — no published value in the data"
            )
            st.markdown(
                f"🔁 **Transfer at {t.at_station_name}**  \n"
                f"&nbsp;&nbsp;Minimum walk between platforms: "
                f"**{gtfs.seconds_to_duration(t.walk_seconds)}** ({walk_source})  \n"
                f"&nbsp;&nbsp;Wait on the platform after walking: "
                f"**{gtfs.seconds_to_duration(t.wait_after_walk_seconds)}**  \n"
                f"&nbsp;&nbsp;Total time at the station: "
                f"**{gtfs.seconds_to_duration(t.total_gap_seconds)}**"
            )

    st.divider()
    st.markdown(
        f"🏁 **Arrive at {destination_name}** — "
        f"**{gtfs.seconds_to_clock(journey.arrive_s)}**"
    )


# ---------------------------------------------------------------------------
# 3. Transfer risks — plain-English transfer assessment (timing-based).
#    (Heading renamed from "Connection Confidence" to "Transfer Risk"; the
#     underlying rel.risk_label logic is unchanged.)
# ---------------------------------------------------------------------------
def show_transfer_risk(journey, origin_name, destination_name, travel_date):
    st.divider()
    st.markdown("## 🔁 Transfer Risk")

    if journey.num_transfers == 0:
        st.info(
            "This is a **direct** trip with no transfer, so there's no connection "
            "to miss."
        )
        return

    st.caption(
        "How much spare time you have at each transfer after walking to the next "
        "platform. Buffer = next train's departure − your arrival − walking time."
    )
    for t in journey.transfers:
        buf = t.usable_buffer_seconds
        lab, emo = rel.risk_label(buf)
        b1, b2 = st.columns([1, 2])
        b1.metric(f"Spare time @ {t.at_station_name}",
                  gtfs.seconds_to_duration(max(0, buf)))
        b2.metric("Transfer risk", f"{emo} {lab}")
        st.markdown(f"&nbsp;&nbsp;{buffer_plain_english(buf)}")

    st.caption(
        "ℹ️ **Risky / Medium / High** are simple timing categories based only on "
        "spare time (Risky < 3 min, Medium 3–6 min, High > 6 min). They are "
        "**not** a measured probability of making the connection."
    )


# ---------------------------------------------------------------------------
# 4. What if your train is late? — the delay simulator (calculations preserved).
# ---------------------------------------------------------------------------
def show_delay_simulator(journey, origin_name, destination_name, travel_date,
                         arrive_info=None):
    st.divider()
    st.markdown("## ⏱️ What if your train is late?")
    first_line = journey.legs[0].route_name
    st.caption(
        f"Drag the slider to imagine your first train (**{first_line}**) running "
        "late. We recompute each connection and your final arrival using only the "
        "real schedule, rolling to the next scheduled train after a missed "
        "connection."
    )
    delay_minutes = st.slider(
        "What if my first train is delayed? (minutes)",
        min_value=0, max_value=20, value=0, step=1,
    )

    multi = rel.simulate_delay_multi(journey, travel_date, delay_minutes)

    # --- Commuter-friendly headline sentence.
    if delay_minutes == 0:
        st.info("No delay applied yet — drag the slider to see the effect.")
    elif multi.no_onward_service:
        st.error(
            f"⛔ With a **{delay_minutes}-min** delay you miss a connection and "
            f"there's no later scheduled train to finish the trip to "
            f"{destination_name} today."
        )
    else:
        missed = [t for t in multi.transfers if not t.connection_made]
        extra = gtfs.seconds_to_duration(multi.added_delay_to_arrival_s)
        revised = gtfs.seconds_to_clock(multi.actual_final_arrival_s)
        if missed:
            where = ", ".join(net.station_name(t.at_station) for t in missed)
            headline = (
                f"🚨 A **{delay_minutes}-min** delay makes you **miss** your "
                f"connection at **{where}**. You'd catch the next scheduled train "
                f"and now arrive **{revised}** — about **{extra} later** than planned."
            )
        else:
            headline = (
                f"👍 Even with a **{delay_minutes}-min** delay you still make every "
                f"connection. You'd arrive **{revised}**"
                + (f" — about {extra} later than planned."
                   if multi.added_delay_to_arrival_s > 0 else ", right on schedule.")
            )
        # Deadline check (Arrive by only).
        if arrive_info:
            deadline = arrive_info["deadline"]
            deadline_s = deadline.hour * 3600 + deadline.minute * 60
            if multi.actual_final_arrival_s <= deadline_s:
                mins = (deadline_s - multi.actual_final_arrival_s) // 60
                headline += (f" ✅ You'd still meet your **{deadline:%I:%M %p}** "
                             f"deadline (by {mins} min).")
            else:
                mins = (multi.actual_final_arrival_s - deadline_s) // 60
                headline += (f" ❌ You'd **miss** your **{deadline:%I:%M %p}** "
                             f"deadline by {mins} min.")
        st.markdown(headline)

    # --- Per-transfer detail (unchanged logic).
    if not multi.no_onward_service and journey.num_transfers > 0:
        for t in multi.transfers:
            name = net.station_name(t.at_station)
            lab, emo = rel.risk_label(t.remaining_buffer_s)
            if t.connection_made:
                st.success(
                    f"✅ At **{name}**: you still make the scheduled train "
                    f"(departs {gtfs.seconds_to_clock(t.scheduled_next_board_s)})."
                )
            else:
                st.warning(
                    f"⚠️ At **{name}**: you **miss** the scheduled train. Next real "
                    f"train departs **{gtfs.seconds_to_clock(t.actual_next_board_s)}**."
                )
            r1, r2 = st.columns([1, 2])
            r1.metric("Spare time for the train you catch",
                      gtfs.seconds_to_duration(t.remaining_buffer_s))
            r2.metric("Transfer risk", f"{emo} {lab}")

    # --- Arrival comparison.
    if not multi.no_onward_service:
        st.markdown("#### Scheduled vs. simulated arrival")
        c1, c2, c3 = st.columns(3)
        c1.metric("Scheduled arrival",
                  gtfs.seconds_to_clock(multi.scheduled_final_arrival_s))
        c2.metric("Simulated arrival",
                  gtfs.seconds_to_clock(multi.actual_final_arrival_s))
        c3.metric("Extra delay",
                  gtfs.seconds_to_duration(multi.added_delay_to_arrival_s))

    st.caption(
        "These are scheduled times with a hypothetical delay **you** control "
        "(simulated, not a prediction). No probabilities are implied."
    )


# ---------------------------------------------------------------------------
# Live MBTA data section (predictions + alerts). Fully additive: if the API
# is unavailable, everything above still works.
# ---------------------------------------------------------------------------
def show_live_data(journey, origin_name, destination_name, travel_date):
    st.divider()
    st.markdown("## 🔴 Live MBTA data")

    # A short legend so scheduled / live / assumed / simulated are never confused.
    st.caption(
        "**Scheduled** = GTFS timetable · **Live** = MBTA V3 API right now · "
        "**Assumed** = labeled estimate (e.g. transfer walk) · "
        "**Simulated** = your what-if delay above. No historical or probability data."
    )

    service_ids = gtfs.active_service_ids(travel_date)
    now_eastern = api.eastern_now()

    # Work out the GTFS trip identity + platforms for EVERY leg of the journey.
    def subway_platforms(station_id):
        return [s for s in gtfs.child_platform_ids(station_id) if s.startswith("700")]

    leg_infos = []   # per leg: dict with route_id, info, board_stops, board_s
    all_predictions = []
    all_route_ids = set()
    all_transfer_stops = set()
    any_error = None
    with st.spinner("Fetching live MBTA predictions and alerts…"):
        for leg in journey.legs:
            info = jp.leg_trip_info(
                leg.route_id, leg.board_station, leg.alight_station,
                leg.board_time_s, service_ids,
            )
            board_stops = subway_platforms(leg.board_station) or [leg.board_station]
            preds, err = api.get_predictions(leg.route_id, board_stops, info.direction_id)
            if err:
                any_error = err
            all_predictions.extend(preds)
            all_route_ids.add(leg.route_id)
            all_transfer_stops.add(leg.board_station)
            all_transfer_stops.add(leg.alight_station)
            leg_infos.append({
                "leg": leg, "info": info, "board_stops": board_stops,
            })
        alerts, alert_err = api.get_alerts(
            sorted(all_route_ids), list(all_transfer_stops)
        )
        if alert_err and not all_predictions and not alerts:
            any_error = any_error or alert_err

    fetched_at = api.now_timestamp_label()
    st.caption(
        f"🕓 Live data fetched at **{fetched_at}**. MBTA predictions only cover "
        "trips in roughly the next hour; trips outside that horizon correctly "
        "show as unavailable and fall back to the scheduled GTFS plan."
    )
    if any_error and not all_predictions and not alerts:
        st.warning(
            "⚠️ **Live MBTA data is currently unavailable.** Showing the scheduled "
            "plan only. Reason: " + str(any_error)
        )
        return

    # --- Match predictions to the SELECTED journey (date + time aware).
    st.markdown("#### Live status for your selected journey")

    reason_text = {
        "future_date": "this journey is on a future date, so live predictions "
                       "aren't published yet",
        "trip_in_past": "this scheduled train has already departed (live "
                        "predictions only look forward)",
        "no_predictions": "the MBTA isn't reporting predictions for this stop "
                          "right now",
        "out_of_window": "no live train is predicted near your scheduled time",
    }

    any_matched = False
    for li in leg_infos:
        leg = li["leg"]; info = li["info"]
        emoji = net.route_emoji(leg.route_id)
        match = api.match_prediction_to_leg(
            all_predictions, set(li["board_stops"]), info.direction_id,
            info.trip_id, leg.board_time_s, travel_date, now_eastern,
        )
        header = (
            f"{emoji} **{leg.route_name}** at **{leg.board_station_name}**  \n"
            f"&nbsp;&nbsp;Scheduled departure: "
            f"**{gtfs.seconds_to_clock(leg.board_time_s)}**  \n"
        )
        if not match.matched or match.prediction is None:
            why = reason_text.get(match.reason, "no matching live train")
            st.markdown(header + f"&nbsp;&nbsp;ℹ️ **Live prediction unavailable "
                        f"for this selected journey** — {why}.")
            continue
        p = match.prediction
        if p.schedule_relationship == "CANCELLED":
            st.markdown(header + "&nbsp;&nbsp;🚫 **Live: this trip is CANCELLED.**")
            continue
        any_matched = True
        live_txt = (gtfs.seconds_to_clock(p.departure_seconds)
                    if p.departure_seconds is not None else "—")
        note = {
            "trip_exact": "same train — exact trip match",
            "time_window": "matched by route, stop, direction and time",
        }.get(match.reason, "matched")
        st.markdown(header + f"&nbsp;&nbsp;🔴 Live predicted departure: "
                    f"**{live_txt}** ({note})")

    if not any_matched:
        st.caption(
            "No live prediction matched your selected trip, so the scheduled plan "
            "and the delay simulator above are based on the GTFS timetable only. "
            "We do not apply unrelated live times to your journey."
        )

    # --- Live-aware connection analysis (Task 3). Scheduled buffer is always
    # shown; a live-aware buffer is computed ONLY when both the arriving leg's
    # live arrival at the transfer and the connecting leg's live departure can
    # be matched reliably. We never assume a first-leg delay propagates.
    if journey.num_transfers > 0:
        st.markdown("#### Live-aware connection analysis")
        st.caption(
            "Scheduled buffer comes from the GTFS timetable. The live-aware "
            "buffer is only computed when both the incoming train's live arrival "
            "and the connecting train's live departure are reliably matched. "
            "This is independent of the manual delay simulator above."
        )
        for i, transfer in enumerate(journey.transfers):
            arriving_leg = journey.legs[i]
            connecting_leg = journey.legs[i + 1]
            arriving_info = leg_infos[i]["info"]
            connecting_info = leg_infos[i + 1]["info"]
            station_label = transfer.at_station_name

            scheduled_buffer_s = transfer.usable_buffer_seconds
            lab, emo = rel.risk_label(scheduled_buffer_s)

            st.markdown(f"**Transfer at {station_label}**")
            cols = st.columns(2)
            cols[0].metric(
                "Scheduled buffer", gtfs.seconds_to_duration(max(0, scheduled_buffer_s)),
                help=f"{emo} {lab} (timing-based category)",
            )

            # Live arrival of the ARRIVING train at the transfer station.
            transfer_platforms = subway_platforms(transfer.at_station) or [transfer.at_station]
            arr_preds, _e1 = api.get_predictions(
                arriving_leg.route_id, transfer_platforms, arriving_info.direction_id
            )
            arr_match = api.match_prediction_to_leg(
                arr_preds, set(transfer_platforms), arriving_info.direction_id,
                arriving_info.trip_id, arriving_leg.alight_time_s, travel_date, now_eastern,
            )
            # Live departure of the CONNECTING train at the transfer station.
            dep_match = api.match_prediction_to_leg(
                all_predictions, set(leg_infos[i + 1]["board_stops"]),
                connecting_info.direction_id, connecting_info.trip_id,
                connecting_leg.board_time_s, travel_date, now_eastern,
            )

            live_arr_s = (
                arr_match.prediction.arrival_seconds
                if arr_match.matched and arr_match.prediction else None
            )
            live_dep_s = (
                dep_match.prediction.departure_seconds
                if dep_match.matched and dep_match.prediction else None
            )

            if live_arr_s is not None and live_dep_s is not None:
                live_buffer_s = live_dep_s - live_arr_s - transfer.walk_seconds
                llab, lemo = rel.risk_label(live_buffer_s)
                cols[1].metric(
                    "Live-aware buffer",
                    gtfs.seconds_to_duration(max(0, live_buffer_s)),
                    help=f"{lemo} {llab} (timing-based category)",
                )
                if live_buffer_s < 0:
                    # Live data indicates the scheduled connection is missed.
                    ready_s = live_arr_s + transfer.walk_seconds
                    service_ids_local = gtfs.active_service_ids(travel_date)
                    nxt = jp._best_leg(
                        connecting_leg.route_id, connecting_leg.route_name,
                        connecting_leg.board_station, connecting_leg.alight_station,
                        ready_s, service_ids_local,
                    )
                    if nxt is not None:
                        st.warning(
                            f"⚠️ **Live data suggests you miss this connection.** "
                            f"Next feasible **{connecting_leg.route_name}** (from the "
                            f"schedule) departs **{gtfs.seconds_to_clock(nxt.board_time_s)}** "
                            f"→ arrives **{gtfs.seconds_to_clock(nxt.alight_time_s)}**."
                        )
                    else:
                        st.warning(
                            "⚠️ **Live data suggests you miss this connection**, and "
                            "no later scheduled service was found for this leg."
                        )
                else:
                    st.success(
                        f"✅ Live-aware buffer is "
                        f"{gtfs.seconds_to_duration(max(0, live_buffer_s))} "
                        f"({lemo} {llab}) — both trains are live-matched."
                    )
            else:
                cols[1].metric("Live-aware buffer", "—")
                missing = []
                if live_arr_s is None:
                    missing.append("incoming train's live arrival")
                if live_dep_s is None:
                    missing.append("connecting train's live departure")
                st.info(
                    "**Live transfer risk unavailable** — we couldn't reliably "
                    "match the " + " and the ".join(missing) + ". The scheduled "
                    "buffer above still applies. We do **not** assume a first-leg "
                    "delay carries unchanged through the transfer."
                )

    # --- 'Next trains right now' per boarding leg (context only).
    st.markdown("#### 🔎 Next trains right now (not tied to your selected trip)")
    st.caption(
        "The next live departures at your boarding stops at this moment, shown "
        "for context only — not used to adjust your journey's times or risk."
    )
    def fmt_now(rows):
        if not rows:
            return "_none reported right now_"
        return ", ".join(gtfs.seconds_to_clock(p.departure_seconds) for p in rows)
    for li in leg_infos:
        leg = li["leg"]; info = li["info"]
        rows = api.next_trains_now(all_predictions, set(li["board_stops"]), info.direction_id)
        st.markdown(f"{net.route_emoji(leg.route_id)} **{leg.route_name}** at "
                    f"{leg.board_station_name}: {fmt_now(rows)}")

    # --- Service alerts, classified by real journey relevance (Step 8).
    segment_stops = set()
    leg_routes_directions = []
    for li in leg_infos:
        leg = li["leg"]; info = li["info"]
        segment_stops |= jp.leg_segment_stops(
            info.trip_id, leg.board_station, leg.alight_station
        )
        leg_routes_directions.append((leg.route_id, info.direction_id))

    # `alerts` is the list fetched above via api.get_alerts(...). It is always a
    # list (empty if the API returned none or failed), so this is safe even when
    # live data is unavailable.
    active_alerts = [a for a in (alerts or []) if a.is_active_on(now_eastern)]
    for a in active_alerts:
        a.relevance = api.classify_alert_relevance(
            a, segment_stops, leg_routes_directions
        )

    journey_alerts = [a for a in active_alerts if a.relevance == "segment"]
    route_level = [a for a in active_alerts if a.relevance == "route"]
    other = [a for a in active_alerts if a.relevance == "other"]
    # 'Other route alerts' expander holds both route-level and unrelated.
    other_bucket = route_level + other

    def alert_body(a, note=None):
        routes = ", ".join(a.routes) if a.routes else "—"
        lines = [f"**Routes:** {routes}"]
        if a.stops:
            lines.append(f"**Affected stops (API):** {', '.join(a.stops[:6])}"
                         + (" …" if len(a.stops) > 6 else ""))
        if a.severity is not None:
            lines.append(f"**Severity:** {a.severity}")
        if a.lifecycle:
            lines.append(f"**Lifecycle:** {a.lifecycle}")
        if note:
            lines.append(f"_{note}_")
        lines.append(a.header)
        if a.url:
            lines.append(f"[More info]({a.url})")
        st.markdown("  \n".join(lines))

    # Journey-specific alerts: each in its own expander (not nested).
    st.markdown("#### Alerts affecting your journey")
    st.caption(
        "Alerts the MBTA data ties to a stop you actually pass through, or to "
        "your line in your direction of travel. Parking and station notices for "
        "stops you don't pass (e.g. Oak Grove, Braintree) are moved to "
        "*Other route alerts* below."
    )
    # NOTE (flagged for later refinement): relevance is derived from the alert's
    # informed_entity stops/route/direction. A notice naming a stop off your
    # path is treated as not-journey-specific. Edge cases (e.g. whole-line
    # shuttles with mixed entities) may still warrant finer review later.
    if not journey_alerts:
        st.success(
            "No active alerts match the specific segments or transfer stations "
            "of your selected journey right now."
        )
    else:
        for a in journey_alerts:
            with st.expander(f"⚠️ [{a.effect}] {a.header[:80]}"):
                alert_body(a)

    # Other alerts: route-level notices (clearly labeled) + unrelated, kept in a
    # single outer expander (Streamlit forbids nested expanders).
    with st.expander(f"Other route alerts ({len(other_bucket)})"):
        st.caption(
            "Broad line-wide notices and alerts for stops you don't pass. "
            "Route-level items are labeled because the API data only ties them "
            "to the line, not to your specific segment."
        )
        if not other_bucket:
            st.caption("No other active alerts for these lines right now.")
        for a in other_bucket:
            tag = "Route-level notice" if a.relevance == "route" else "Other line/stop"
            st.markdown(f"**⚠️ [{a.effect}] {a.header[:80]}**")
            alert_body(a, note=tag)
            st.markdown("---")

    st.caption(
        "Live predictions and alerts come directly from the MBTA V3 API "
        "(times shown in America/New_York). We never invent a prediction, and we "
        "do not claim a probability of making your connection."
    )


# ---------------------------------------------------------------------------
# Historical Reliability section (Step 4). Uses only real MBTA LAMP data from
# a small cleaned CSV. Additive: if the dataset is missing, shows a clear note.
# ---------------------------------------------------------------------------
def _render_headway_stats(stats):
    """Render one HeadwayStats block: metrics, charts, plain-language summary."""
    st.markdown(
        f"**Line:** {stats.route_id} &nbsp;·&nbsp; **Station:** {stats.station_name} "
        f"&nbsp;·&nbsp; **Dates:** {stats.date_min} → {stats.date_max} (weekdays)  \n"
        f"**Sample size:** {stats.observations:,} observed train-to-train gaps  \n"
        f"**Source:** [{hist.SOURCE_NAME}]({hist.SOURCE_URL}) (official, public)"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median gap", f"{stats.median_min} min")
    c2.metric("25th–75th", f"{stats.p25_min}–{stats.p75_min} min")
    c3.metric("90th percentile", f"{stats.p90_min} min")
    c4.metric("Mean gap", f"{stats.mean_min} min")
    if stats.scheduled_median_min is not None:
        st.caption(
            f"For comparison, the scheduled median headway in this sample was "
            f"about **{stats.scheduled_median_min} min**."
        )

    st.markdown("**Distribution of observed headways (minutes)**")
    hw_df = pd.DataFrame({"headway_min": stats.headways_min})
    hw_df["bucket"] = hw_df["headway_min"].round().astype(int)
    counts = hw_df.groupby("bucket").size().rename("trains")
    st.bar_chart(counts)

    hourly_df = hist.hourly_dataframe(stats)
    if not hourly_df.empty:
        st.markdown("**Variation by local clock hour (median vs 90th percentile gap)**")
        st.caption(
            "Hours are local clock hours 0–23 (America/New_York). After-midnight "
            "trips (service times past 24:00) are folded into the early-morning "
            "hours; the service date is unchanged."
        )
        chart_df = hourly_df.set_index("hour")[["median_min", "p90_min"]]
        st.line_chart(chart_df)

    st.info(
        f"What this means for a commuter: across {stats.observations:,} observed "
        f"gaps on recent weekdays, half of the {stats.route_id} trains at "
        f"{stats.station_name} arrived within **{stats.median_min} min** of the "
        f"previous one (the **median** — the middle value). The **average "
        f"(mean)** gap was **{stats.mean_min} min**; it differs from the median "
        f"because a few long gaps pull the average up. The **90th percentile** "
        f"of **{stats.p90_min} min** means about 1 in 10 gaps was at least that "
        f"long — a useful worst-case-ish figure. Longer, less consistent gaps "
        f"mean a missed connection could cost more waiting time. These are "
        f"historical observations for context — they do **not** predict your "
        f"specific trip or the chance of making a connection."
    )


def _historical_plain_summary(journey):
    """Short plain-English summary of the available historical data for this
    journey's transfers. Returns a markdown string (never borrows other
    stations' data)."""
    if journey.num_transfers == 0:
        return ("This is a **direct** trip, so there's no transfer to assess "
                "with historical train-gap data.")
    lines = []
    for i, transfer in enumerate(journey.transfers):
        conn = journey.legs[i + 1]
        if hist.has_data_for(transfer.at_station, conn.route_id):
            try:
                s = hist.compute_stats(conn.route_id, transfer.at_station,
                                       transfer.at_station_name)
                lines.append(
                    f"- **{conn.route_name} at {transfer.at_station_name}:** on "
                    f"recent weekdays, trains came about every **{s.median_min} "
                    f"min** (typical), stretching to **{s.p90_min} min** one time "
                    f"in ten. Based on **{s.observations:,}** observed time gaps "
                    f"between trains from {hist.SOURCE_NAME}."
                )
            except hist.HistoricalDataUnavailable:
                lines.append(f"- **{conn.route_name} at {transfer.at_station_name}:** "
                             "no verified historical data for this transfer.")
        else:
            lines.append(f"- **{conn.route_name} at {transfer.at_station_name}:** "
                         "no verified historical data for this transfer.")
    return "\n".join(lines)


def show_historical_reliability(journey, origin_name, destination_name):
    st.divider()
    st.markdown("## 📈 Historical data and experimental ML")
    st.caption(
        "Real observed **time gaps between trains** (\"headways\") at your "
        "transfer stations on recent weekdays. For context only — **not** a "
        "probability of making your connection."
    )

    if not hist.available_routes():
        st.warning(
            "ℹ️ **Historical data unavailable.** The cleaned dataset was not "
            "found. Run `.\\.venv\\Scripts\\python.exe prepare_historical.py` to "
            "build it from the MBTA LAMP archive."
        )
        return

    # --- Plain-English summary ABOVE the expander (source, station, line, size).
    st.markdown(_historical_plain_summary(journey))

    # --- All detailed charts + experimental ML live inside one expander.
    with st.expander("Explore the historical data and predictions"):
        st.markdown("### For your selected journey")
        if journey.num_transfers == 0:
            st.info(
                "This is a **direct** trip with no transfer, so there is no "
                "connection to assess with historical transfer reliability."
            )
        else:
            for i, transfer in enumerate(journey.transfers):
                connecting_leg = journey.legs[i + 1]
                station_id = transfer.at_station
                station_label = transfer.at_station_name
                route_id = connecting_leg.route_id
                st.markdown(
                    f"**Transfer {i + 1}: {connecting_leg.route_name} at "
                    f"{station_label}**"
                )
                if hist.has_data_for(station_id, route_id):
                    try:
                        stats = hist.compute_stats(route_id, station_id, station_label)
                        _render_headway_stats(stats)
                    except hist.HistoricalDataUnavailable:
                        st.info("Historical reliability data unavailable for this transfer.")
                else:
                    st.info("Historical reliability data unavailable for this transfer.")

        # Example dataset (kept, clearly labeled, never applied elsewhere).
        st.divider()
        st.markdown("#### 📚 Example dataset — Downtown Crossing (Red & Orange)")
        st.caption(
            "A reference sample from MBTA LAMP for Downtown Crossing, shown as a "
            "labeled **example** independent of your selected journey. It is "
            "never applied to other transfer stations."
        )
        example_routes = [r for r in ["Orange", "Red"] if hist.has_data_for("place-dwnxg", r)]
        if not example_routes:
            st.info("Example dataset not available.")
        else:
            chosen = st.radio(
                "Observed time gaps between trains at Downtown Crossing for:",
                options=example_routes, horizontal=True, index=0,
                key="hist_example_route",
            )
            try:
                stats = hist.compute_stats(chosen, "place-dwnxg", "Downtown Crossing")
                _render_headway_stats(stats)
            except hist.HistoricalDataUnavailable as err:
                st.info(f"Example data unavailable: {err}")

        # Experimental ML analysis, folded into the same expander.
        st.divider()
        show_experimental_ml(journey, origin_name, destination_name)


# ---------------------------------------------------------------------------
# Experimental ML Predictions (Task 5) + scenario range (Task 6). Clearly
# labeled as experimental research — never used to alter the journey plan.
# ---------------------------------------------------------------------------
def show_experimental_ml(journey, origin_name, destination_name):
    st.markdown("### 🧪 Experimental ML predictions")
    st.caption(
        "**Experimental / research only.** A model trained on real MBTA LAMP "
        "observations to predict the **next-train headway (the gap between "
        "trains) in minutes** at Downtown Crossing for the Red and Orange lines. "
        "This is **not** a journey arrival time and does **not** change any "
        "recommendation above."
    )

    if not _ML_AVAILABLE:
        st.info(
            "The ML module or scikit-learn is not available in this environment, "
            "so experimental predictions are disabled. Everything else works."
        )
        return

    try:
        res = ml.train_and_evaluate()
    except Exception as err:  # never break the page on an ML issue
        st.info(f"Experimental ML unavailable: {err}")
        return

    st.markdown(
        f"**Predicts:** {res.target_name}  \n"
        f"**Supported scope:** Red & Orange lines at **Downtown Crossing**, both "
        f"directions (the only stations/lines with verified observations).  \n"
        f"**Historical source:** [{hist.SOURCE_NAME}]({hist.SOURCE_URL}) — "
        f"{res.rows_total:,} observations.  \n"
        f"**Evaluation:** chronological split, test = newest dates "
        f"{res.test_date_min} → {res.test_date_max} ({res.rows_test:,} held-out obs)."
    )

    # Baseline vs model (same test set).
    st.markdown("**Baseline vs. model (held-out test set)**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("_Baseline: median by route/direction/hour_")
        st.metric("Mean abs error", f"{res.baseline_mae} min"
                  if res.baseline_mae is not None else "—")
        if res.baseline_within2 is not None:
            st.metric("Within ±2 min", f"{res.baseline_within2}%")
    with c2:
        if res.trained:
            st.markdown("_Model: HistGradientBoosting_")
            st.metric("Mean abs error", f"{res.model_mae} min")
            st.metric("Within ±2 min", f"{res.within2}%")
        else:
            st.markdown("_Model: not trained_")
            st.metric("Mean abs error", "—")

    if res.trained:
        st.caption(
            f"Model accuracy by tolerance (held-out): within ±1 min "
            f"**{res.within1}%**, within ±2 min **{res.within2}%**, within ±5 min "
            f"**{res.within5}%** · median abs error **{res.model_medae} min**."
        )
        better = res.model_mae is not None and res.model_mae < res.baseline_mae
        st.info(
            ("✅ On unseen data the model slightly **beats** the baseline."
             if better else
             "ℹ️ On unseen data the model does **not** beat the simple baseline — "
             "subway headways here are dominated by the scheduled frequency, "
             "which the median already captures. We keep the baseline as the "
             "reference and present the model as an experimental result.") +
            " These are headway estimates with real error (not a 'percent "
            "accuracy' claim and not a 90% confidence interval)."
        )
    else:
        st.info(f"ℹ️ {res.note} The baseline median is retained.")

    # --- Task 6: clearly-labeled scenario range (NOT a statistical interval).
    # Rendered as a plain sub-block (no nested expander — this function is shown
    # inside the historical expander).
    if journey.num_transfers > 0 and res.trained:
        st.markdown("#### 🔬 Experimental scenario range (not a confidence interval)")
        st.caption(
            "A what-if range built by applying the observed historical "
            "spread of headways (25th–90th percentile) at a verified "
            "transfer to the GTFS missed-transfer logic. It is a **scenario "
            "range**, not a validated prediction interval, and it only uses "
            "a transfer that has verified historical data for its connecting "
            "line. It does not imply one station's distribution predicts the "
            "whole journey."
        )
        shown_any = False
        for i, transfer in enumerate(journey.transfers):
            conn = journey.legs[i + 1]
            if not hist.has_data_for(transfer.at_station, conn.route_id):
                continue
            try:
                stats = hist.compute_stats(
                    conn.route_id, transfer.at_station, transfer.at_station_name
                )
            except hist.HistoricalDataUnavailable:
                continue
            shown_any = True
            sched_arrival = journey.arrive_s
            # Optimistic: typical (p25) wait; Pessimistic: p90 wait at the
            # transfer, added to the scheduled arrival as a coarse scenario.
            extra_typ = max(0.0, stats.p25_min - (transfer.usable_buffer_seconds / 60.0))
            extra_bad = max(0.0, stats.p90_min - (transfer.usable_buffer_seconds / 60.0))
            st.markdown(
                f"**Transfer: {conn.route_name} at {transfer.at_station_name}**  \n"
                f"&nbsp;&nbsp;Scheduled arrival: "
                f"**{gtfs.seconds_to_clock(sched_arrival)}**  \n"
                f"&nbsp;&nbsp;Scenario range (applying observed p25–p90 headway "
                f"spread at this transfer): about **+{extra_typ:.0f} to "
                f"+{extra_bad:.0f} min** on a missed connection.  \n"
                f"&nbsp;&nbsp;*Based on {stats.observations:,} observations; "
                f"scenario only, not a probability.*"
            )
        if not shown_any:
            st.info(
                "No transfer on this journey has verified historical data, so "
                "no scenario range is shown (we don't borrow other stations)."
            )


# ---------------------------------------------------------------------------
# Arrive-by recommendation summary (Task B). Shows the latest journey meeting
# the deadline plus earlier alternatives, with buffer and minutes-early.
# ---------------------------------------------------------------------------
def show_arrive_by_summary(arrive_info, origin_name, destination_name):
    deadline = arrive_info["deadline"]
    margin = arrive_info["margin"]
    options = arrive_info["options"]

    deadline_s = deadline.hour * 3600 + deadline.minute * 60
    effective_s = deadline_s - margin * 60

    st.success(
        f"Latest **{origin_name} → {destination_name}** journey that arrives by "
        f"**{deadline:%I:%M %p}**"
        + (f" minus your **{margin}-min** safety margin "
           f"(so by {_fmt_clock(effective_s)})" if margin else "")
        + ":"
    )
    if margin:
        st.caption(
            f"The {margin}-minute margin is your own cushion you selected — not a "
            "statistical guarantee. We require the scheduled arrival to be at or "
            f"before {_fmt_clock(effective_s)}."
        )

    def describe(journey, label):
        minutes_early = (deadline_s - journey.arrive_s) // 60
        if journey.transfers:
            tdesc = " → ".join(
                f"{t.at_station_name} ({gtfs.seconds_to_duration(max(0, t.usable_buffer_seconds))} buffer)"
                for t in journey.transfers
            )
            transfer_line = f"&nbsp;&nbsp;Transfer at **{tdesc}**"
        else:
            transfer_line = "&nbsp;&nbsp;Direct — no transfer"
        lines_desc = " + ".join(l.route_name for l in journey.legs)
        st.markdown(
            f"**{label}** ({lines_desc})  \n"
            f"&nbsp;&nbsp;Depart **{origin_name}**: "
            f"{gtfs.seconds_to_clock(journey.depart_s)}  \n"
            f"&nbsp;&nbsp;Scheduled arrival **{destination_name}**: "
            f"{gtfs.seconds_to_clock(journey.arrive_s)} "
            f"(**{minutes_early} min** before your {deadline:%I:%M %p} deadline)  \n"
            + transfer_line
        )

    describe(options[0], "✅ Recommended (latest that makes it)")
    if len(options) > 1:
        st.markdown("**Earlier alternatives:**")
        for i, alt in enumerate(options[1:], start=1):
            describe(alt, f"Alternative {i}")

    st.caption(
        "Scheduled times from the GTFS timetable. The detailed timeline, "
        "transfer risk, live status and history below all refer to the "
        "recommended journey."
    )


def _fmt_clock(total_seconds: int) -> str:
    """Local helper mirroring gtfs.seconds_to_clock for a plain seconds value."""
    return gtfs.seconds_to_clock(total_seconds)


# ---------------------------------------------------------------------------
# 7. Data sources and limitations (defined before the render block uses it).
# ---------------------------------------------------------------------------
def show_data_sources_and_limitations():
    st.divider()
    st.markdown("## 📚 Data sources and limitations")
    st.markdown(
        "- **Schedules & routing:** official MBTA GTFS feed (stops, trips, "
        "stop times, calendars, transfers).  \n"
        "- **Live predictions & alerts:** MBTA V3 API, matched to your trip "
        "near real time (roughly the next hour); times shown in America/New_York.  \n"
        "- **Historical train gaps:** MBTA LAMP Subway Performance Data — a small "
        "verified sample (Red & Orange at Downtown Crossing). Other transfers "
        "show \"unavailable\" rather than borrowing another station's data.  \n"
        "- **Experimental ML:** a research model that predicts the gap between "
        "trains (headway), **not** your arrival time. On held-out data it did "
        "**not** beat a simple median baseline, so it does not change any "
        "recommendation."
    )
    st.caption(
        "Labels used throughout: **Scheduled** (timetable) · **Live** (MBTA API) "
        "· **Simulated** (your what-if delay) · **Historical** (observed LAMP "
        "data) · **Experimental** (research model / scenario). Risk labels are "
        "timing-based categories, not probabilities. No arrival time is "
        "presented as a statistical prediction."
    )


# ---------------------------------------------------------------------------
# Journey planning form  (same controls as before)
# ---------------------------------------------------------------------------
if data_ok:
    st.markdown("## 1. Plan your journey")

    col1, col2 = st.columns(2)
    with col1:
        start_station = st.selectbox(
            "Starting Station",
            options=station_names,
            index=default_index(station_names, "Harvard", 0),
        )
    with col2:
        destination_station = st.selectbox(
            "Destination Station",
            options=station_names,
            index=default_index(station_names, "Ruggles", len(station_names) - 1),
        )

    # Mode selector (Task B): plan forward from a departure, or backward from
    # a destination arrival deadline.
    plan_mode = st.radio(
        "How do you want to plan?",
        options=["Depart at", "Arrive by"],
        horizontal=True,
    )

    col3, col4 = st.columns(2)
    with col3:
        travel_date = st.date_input("Travel date", value=datetime.date(2026, 10, 24))

    # Mode-specific controls.
    departure_time = datetime.time(9, 0)
    deadline_time = datetime.time(9, 0)
    safety_margin = 0
    if plan_mode == "Depart at":
        with col4:
            departure_time = st.time_input("Departure time", value=datetime.time(9, 0))
    else:  # Arrive by
        with col4:
            deadline_time = st.time_input(
                "Arrive by (destination deadline)", value=datetime.time(9, 0)
            )
        safety_margin = st.slider(
            "Optional extra safety margin (minutes)",
            min_value=0, max_value=20, value=0, step=1,
            help=(
                "Your own buffer: we recommend the latest journey scheduled to "
                "arrive by the deadline MINUS this margin. It is a personal "
                "cushion, not a statistical guarantee."
            ),
        )

    plan_clicked = st.button("Plan My Journey", type="primary")

    # When the button is clicked, plan the journey and remember it so that
    # moving the delay slider afterwards does not require clicking again.
    if plan_clicked:
        if start_station == destination_station:
            st.session_state.pop("journey", None)
            st.warning(
                "Your starting station and destination are the same. "
                "Pick two different stations."
            )
        elif not jp.supported_journey(start_station, destination_station):
            st.session_state.pop("journey", None)
            st.warning(
                f"**{start_station} → {destination_station}** isn't supported yet.\n\n"
                "This early version only plans **Red Line → Orange Line** journeys "
                "that transfer at **Downtown Crossing**. Try a Red Line station "
                "(like **Harvard**, **Central**, or **Alewife**) as the start and an "
                "Orange Line station (like **Ruggles**, **Back Bay**, or "
                "**Forest Hills**) as the destination."
            )
        elif plan_mode == "Arrive by":
            try:
                options = jp.arrive_by(
                    start_station, destination_station, travel_date,
                    deadline_time, safety_margin, max_results=3,
                )
            except gtfs.GTFSError as error:
                options = []
                st.session_state.pop("journey", None)
                st.error(str(error))
            if not options:
                st.session_state.pop("journey", None)
                st.warning(
                    f"No scheduled {start_station} → {destination_station} journey "
                    f"arrives by **{deadline_time:%I:%M %p}**"
                    + (f" (minus your {safety_margin}-min margin)" if safety_margin else "")
                    + f" on {travel_date:%A, %B %d}. Try a later deadline or a "
                    "smaller margin. (No train times are invented.)"
                )
            else:
                st.session_state["journey"] = {
                    "journey": options[0],
                    "origin": start_station,
                    "destination": destination_station,
                    "date": travel_date,
                    "arrive_by": {
                        "deadline": deadline_time,
                        "margin": safety_margin,
                        "options": options,
                    },
                }
        else:
            try:
                journey = jp.plan_journey(
                    start_station, destination_station, travel_date, departure_time
                )
                st.session_state["journey"] = {
                    "journey": journey,
                    "origin": start_station,
                    "destination": destination_station,
                    "date": travel_date,
                }
            except gtfs.GTFSError as error:
                st.session_state.pop("journey", None)
                st.error(str(error))

    # Auto-plan on first load so the journey + Connection Confidence slider are
    # visible immediately, without needing to click the button first. This only
    # runs when nothing has been planned yet and the current (default) selection
    # is a supported Red -> Orange journey.
    if (
        "journey" not in st.session_state
        and not plan_clicked
        and start_station != destination_station
        and jp.supported_journey(start_station, destination_station)
    ):
        try:
            auto_journey = jp.plan_journey(
                start_station, destination_station, travel_date, departure_time
            )
            st.session_state["journey"] = {
                "journey": auto_journey,
                "origin": start_station,
                "destination": destination_station,
                "date": travel_date,
            }
        except gtfs.GTFSError:
            # If the default can't be planned, just wait for a button click.
            pass

    # Render the stored journey (if any). This runs on every rerun, so the
    # delay slider stays interactive without re-planning. The sequence is
    # commuter-first: recommendation -> transfer risk -> delay -> live ->
    # history/ML -> data sources.
    saved = st.session_state.get("journey")
    if saved:
        arrive_info = saved.get("arrive_by")

        # 2. Your recommended trip (headline card).
        show_recommended_trip(
            saved["journey"], saved["origin"], saved["destination"], saved["date"],
            arrive_info=arrive_info,
        )
        # Arrive-by alternatives (if any), then the detailed timeline.
        if arrive_info:
            show_arrive_by_summary(
                arrive_info, saved["origin"], saved["destination"]
            )
        show_scheduled_journey(
            saved["journey"], saved["origin"], saved["destination"], saved["date"]
        )
        # 3. Transfer risks.
        show_transfer_risk(
            saved["journey"], saved["origin"], saved["destination"], saved["date"]
        )
        # 4. What if your train is late?
        show_delay_simulator(
            saved["journey"], saved["origin"], saved["destination"], saved["date"],
            arrive_info=arrive_info,
        )
        # 5. Live MBTA updates.
        show_live_data(
            saved["journey"], saved["origin"], saved["destination"], saved["date"]
        )
        # 6. Historical data and experimental ML (detail behind an expander).
        show_historical_reliability(
            saved["journey"], saved["origin"], saved["destination"]
        )
        # 7. Data sources and limitations.
        show_data_sources_and_limitations()
