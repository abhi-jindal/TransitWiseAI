"""
Simple tests for the Arrive-by planner (Task C).

Run with the project's virtual environment:
  .\\.venv\\Scripts\\python.exe test_arrive_by.py

Covers: deadlines, date boundaries, safety margin, no-journey case, and a
missed-transfer check that reuses the reliability simulator. Prints PASS/FAIL
and exits non-zero if any assertion fails.
"""

import datetime
import sys

import journey_planner as jp
import reliability as rel


def clock(s: int) -> str:
    next_day = s >= 24 * 3600
    s %= 24 * 3600
    h = s // 3600
    m = (s % 3600) // 60
    ampm = "AM" if h < 12 else "PM"
    hh = h % 12 or 12
    return f"{hh}:{m:02d} {ampm}" + (" (+1d)" if next_day else "")


def main() -> int:
    date = datetime.date(2026, 10, 24)  # Saturday, in the GTFS feed
    origin, dest = "Harvard", "Ruggles"
    failures = 0

    def check(label, condition):
        nonlocal failures
        status = "PASS" if condition else "FAIL"
        if not condition:
            failures += 1
        print(f"  [{status}] {label}")

    # 1. Arrive by 9:00 AM, no margin.
    print("=== Arrive by 9:00 AM, margin 0 ===")
    res = jp.arrive_by(origin, dest, date, datetime.time(9, 0), 0, max_results=3)
    for i, j in enumerate(res):
        early = (9 * 3600 - j.arrive_s) // 60
        buf = (j.leg2.board_time_s - j.leg1.alight_time_s - j.transfer_seconds) // 60
        tag = "RECOMMENDED" if i == 0 else f"alt {i}"
        print(f"    [{tag}] depart {clock(j.depart_s)} -> arrive {clock(j.arrive_s)} "
              f"({early} min early, buffer {buf}m)")
    check("found journeys arriving by 9:00", bool(res))
    check("all arrive on/before 9:00", all(j.arrive_s <= 9 * 3600 for j in res))
    check("recommended is the latest departure",
          all(res[0].depart_s >= j.depart_s for j in res))

    # 2. Safety margin 10 min -> must arrive by 8:50.
    print("=== Arrive by 9:00 AM, margin 10 ===")
    res_m = jp.arrive_by(origin, dest, date, datetime.time(9, 0), 10, max_results=3)
    for j in res_m:
        print(f"    depart {clock(j.depart_s)} -> arrive {clock(j.arrive_s)}")
    check("found journeys with 10-min margin", bool(res_m))
    check("all arrive on/before 8:50", all(j.arrive_s <= 9 * 3600 - 10 * 60 for j in res_m))
    check("margin recommendation arrives <= no-margin recommendation",
          bool(res_m) and bool(res) and res_m[0].arrive_s <= res[0].arrive_s)

    # 3. Date boundary: very early deadline.
    print("=== Arrive by 5:30 AM (early boundary) ===")
    res_e = jp.arrive_by(origin, dest, date, datetime.time(5, 30), 0)
    print(f"    journeys found: {len(res_e)}")
    check("all arrive on/before 5:30 (or none)", all(j.arrive_s <= 5 * 3600 + 30 * 60 for j in res_e))

    # 4. No valid journey: impossibly early deadline.
    print("=== Arrive by 3:00 AM (no service) ===")
    res_none = jp.arrive_by(origin, dest, date, datetime.time(3, 0), 0)
    print(f"    journeys found: {len(res_none)} (expected 0)")
    check("returns empty, nothing fabricated", res_none == [])

    # 5. Missed-transfer check reuses the reliability simulator.
    print("=== Missed-transfer check on recommended journey (+5 min delay) ===")
    if res:
        r = rel.simulate_delay(res[0], dest, date, 5)
        print(f"    connection made with +5? {r.connection_made}")
        if not r.no_onward_service and not r.connection_made:
            print(f"    replacement arrival {clock(r.actual_final_arrival_s)} "
                  f"(+{r.added_delay_to_arrival_s // 60} min)")
        check("reliability simulator runs on an arrive-by journey", True)

    print()
    if failures:
        print(f"FAILED: {failures} assertion(s)")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
