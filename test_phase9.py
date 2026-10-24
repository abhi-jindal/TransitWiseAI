"""
Phase 9 tests: journey-specific historical reliability + full-render checks.

Run with the project's virtual environment:
  .\\.venv\\Scripts\\python.exe test_phase9.py

Covers Harvard->Boston College, Airport->Boston College, Harvard->Ruggles.
Confirms that Green-line transfers do NOT get Orange/DTX historical stats, and
that the whole page renders with no exception. Prints PASS/FAIL, exits non-zero
on failure.
"""

import datetime
import sys


def main() -> int:
    import historical_reliability as hist
    import journey_planner as jp

    date = datetime.date(2026, 10, 24)
    failures = 0

    def check(label, cond):
        nonlocal failures
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures += 1

    # --- Task 2: verified-data gating by station+route.
    print("=== has_data_for ===")
    check("DTX Orange has data", hist.has_data_for("place-dwnxg", "Orange"))
    check("DTX Red has data", hist.has_data_for("place-dwnxg", "Red"))
    check("DTX Green-C has NO data", not hist.has_data_for("place-dwnxg", "Green-C"))
    check("Park St Green-C has NO data", not hist.has_data_for("place-pktrm", "Green-C"))

    # Harvard -> Boston College transfers onto a Green branch: must be unavailable.
    j = jp.plan_journey("Harvard", "Boston College", date, datetime.time(9, 0))
    print("Harvard->Boston College:", " + ".join(l.route_name for l in j.legs))
    for i, t in enumerate(j.transfers):
        conn = j.legs[i + 1]
        check(f"transfer {i+1} ({conn.route_name} @ {t.at_station_name}) has NO applied stats",
              not hist.has_data_for(t.at_station, conn.route_id))

    # --- Task 4: full render for the three journeys.
    from streamlit.testing.v1 import AppTest

    def render(o, d):
        at = AppTest.from_file("app.py", default_timeout=240)
        at.run()
        at.selectbox[0].set_value(o).run()
        at.selectbox[1].set_value(d).run()
        at.button[0].click().run()
        print(f"=== render {o} -> {d} ===")
        check(f"{o}->{d} renders with no exception", not list(at.exception))
        md = "\n".join(m.value for m in at.markdown)
        # Heading was renamed in the UX pass to "Historical data and experimental ML".
        check(f"{o}->{d} shows historical section",
              "Historical data and experimental ML" in md)
        check(f"{o}->{d} shows Live MBTA data", "Live MBTA data" in md)

    render("Harvard", "Boston College")
    render("Airport", "Boston College")
    render("Harvard", "Ruggles")

    print()
    if failures:
        print(f"FAILED: {failures} assertion(s)")
        return 1
    print("ALL PHASE 9 TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
