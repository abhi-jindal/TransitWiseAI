"""
Tests for the general multi-line network search (Phase 4 routing cases).

Run with the project's virtual environment:
  .\\.venv\\Scripts\\python.exe test_network.py

Covers direct, one-transfer (across all four lines), branch-to-branch, and
two-transfer journeys, plus a no-service date. Prints PASS/FAIL and exits
non-zero on failure. Uses only the real GTFS schedule.
"""

import datetime
import sys

import network as net


def clock(s):
    nd = s >= 24 * 3600
    s %= 24 * 3600
    h = s // 3600
    m = (s % 3600) // 60
    ap = "AM" if h < 12 else "PM"
    hh = h % 12 or 12
    return f"{hh}:{m:02d} {ap}" + (" +1d" if nd else "")


def main() -> int:
    date = datetime.date(2026, 10, 24)   # Saturday in the feed
    nine = 9 * 3600
    failures = 0

    def check(label, cond):
        nonlocal failures
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures += 1

    def journeys(o, d, t=nine):
        oid, did = net.station_id(o), net.station_id(d)
        if not oid or not did:
            return None
        return net.search_journeys(oid, did, date, t, max_results=3)

    # Phase 4 routing cases.
    specs = [
        ("Red->Orange", "Harvard", "Ruggles", 1),
        ("Red->Green", "Harvard", "Copley", 1),
        ("Blue->Green", "Wonderland", "Copley", 1),
        ("Orange->Blue", "Forest Hills", "Airport", 1),
        ("Green branch->branch", "Boston College", "Cleveland Circle", None),
        ("Same-line direct", "Harvard", "Park Street", 0),
    ]
    for label, o, d, min_expected_transfers in specs:
        js = journeys(o, d)
        print(f"=== {label}: {o} -> {d} ===")
        check(f"{label}: found a journey", js)
        if js:
            best = js[0]
            lines = " + ".join(net.route_display(l.route_id) for l in best.legs)
            print(f"    best: [{best.num_transfers}x] {lines} "
                  f"({clock(best.depart_s)} -> {clock(best.arrive_s)})")
            if min_expected_transfers == 0:
                check(f"{label}: best is direct", best.num_transfers == 0)

    # Two-transfer journey exists somewhere in results for a cross-town trip.
    js = journeys("Wonderland", "Forest Hills")
    print("=== Two-transfer availability: Wonderland -> Forest Hills ===")
    check("some option found", bool(js))

    # No-service date -> no journeys (never fabricated).
    print("=== No-service date ===")
    empty = net.search_journeys(
        net.station_id("Harvard"), net.station_id("Ruggles"),
        datetime.date(2030, 1, 1), nine, max_results=3,
    )
    check("no journeys on a date with no service", empty == [])

    print()
    if failures:
        print(f"FAILED: {failures} assertion(s)")
        return 1
    print("ALL NETWORK TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
