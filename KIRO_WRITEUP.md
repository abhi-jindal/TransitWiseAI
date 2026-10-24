# How TransitWise AI was built with Kiro

TransitWise AI was built **inside the Kiro IDE**, primarily through **iterative
conversational development** — describing goals to Kiro's agent in plain
language, having it read the existing code, make targeted edits, run commands,
and verify results, then refining over many rounds. Development was incremental:
each phase kept the app working before the next was added.

> Honesty note: this project did **not** use Kiro Agent Hooks, steering files,
> MCP servers, Kiro Powers, or the formal spec-driven (requirements → design →
> tasks) workflow. The work was conversational "vibe" coding with the agent,
> using its file-editing, terminal, and test-running tools. We're documenting
> only what was genuinely used.

## How the collaboration worked

- **Conversational, iterative building.** Each feature started as a described
  goal. Kiro inspected the relevant files first, proposed/made edits, then ran
  the app or tests to confirm behavior before moving on.
- **Agent-run terminal + tests.** Kiro used the integrated terminal (the
  project's `.venv` Python) to install dependencies, run the test suites, launch
  the Streamlit app headlessly for verification, and profile performance.
- **Direct file edits.** Kiro created and modified the Python modules, tests,
  and docs directly in the workspace rather than pasting snippets.
- **Honesty guardrails.** A recurring theme across phases was refusing to
  fabricate data: no invented predictions, clear labeling of Scheduled / Live /
  Simulated / Historical / Experimental values, and reporting negative results
  (e.g. the ML model not beating its baseline) truthfully.

## Development phases

1. **Initial Streamlit prototype.** A minimal app (title, tagline, station
   dropdowns, a departure-time control, and a "Plan My Journey" button) using
   clearly labeled sample data — establishing the UI shell.

2. **GTFS integration.** Replaced sample data with the official MBTA GTFS feed:
   loading stops/routes/trips/stop_times/calendars/transfers, de-duplicating
   platforms to parent stations, handling service calendars and GTFS times past
   24:00:00, and computing a first real scheduled journey.

3. **Reliability: Connection Confidence + delay simulator.** Added the usable
   transfer buffer, timing-based risk labels, and a delay simulator that finds
   the next real scheduled service after a missed connection — never inventing a
   waiting penalty.

4. **Live MBTA V3 API integration.** Added real-time predictions and service
   alerts (standard-library HTTP), matched to the selected trip by
   route/stop/direction/trip, with graceful fallback and date/time-aware
   matching so stale or far-off predictions are never misapplied.

5. **Historical reliability (MBTA LAMP).** Discovered and validated a real LAMP
   dataset, built a small cleaned sample (Red/Orange at Downtown Crossing), and
   added observed-headway statistics and charts — scoped so one station's data
   is never applied to another.

6. **Four-line routing expansion.** Generalized the hard-coded Red→Orange path
   into a reusable multi-line network (Red, Orange, Blue, Green B/C/D/E) with
   direct, one-transfer, and two-transfer search in both directions, including a
   performance pass (cached, preprocessed network artifacts).

7. **Arrive-by functionality.** Added backward planning from a destination
   deadline with an optional personal safety margin, reusing the routing engine
   rather than duplicating logic.

8. **Experimental ML.** Trained a scikit-learn headway model with a
   leakage-safe pipeline and a chronological train/test split, compared it
   against a median baseline, and reported honest metrics — including that the
   model did not outperform the baseline.

9. **Debugging and automated testing.** Built and repeatedly ran three test
   suites (`test_network.py`, `test_arrive_by.py`, `test_phase9.py`), fixed real
   bugs surfaced by them (e.g. an undefined-variable crash in the live section,
   a stale test assertion after a rename), and used headless app renders to
   confirm no exceptions.

10. **UI improvements.** Reorganized the page into a commuter-first sequence
    (recommendation → transfer risk → delay what-if → live → history/ML → data
    sources), translated technical values into plain English, renamed
    "Connection Confidence" to "Transfer Risk," and moved detailed charts/ML
    behind an expander.

## What Kiro was especially useful for

- Safely refactoring the hard-coded single-route planner into a general
  multi-line engine without breaking existing tests.
- Fast investigate-then-edit loops: reading the actual code before changing it,
  then verifying with the test suites and headless renders.
- Keeping the project honest — pushing back on overclaiming and surfacing
  negative results instead of hiding them.
