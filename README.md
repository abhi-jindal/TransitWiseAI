# 🚇 TransitWise AI

**Know when you'll actually arrive, not just when you're scheduled to.**

**Hackachusetts 2026 — Challenge track: Civic Action — Safer and Connected Places**

---

## The problem

Boston commuters — especially students — can miss important deadlines (a class,
an exam, a shift, a connecting trip) because a small delay on one train causes a
**missed transfer**, and a missed transfer can turn a two-minute slip into a
ten- or twenty-minute late arrival. Standard trip planners show the *scheduled*
time and stop there. They don't tell you how fragile your connection is or what
happens when things run late.

## The solution

TransitWise AI is a **reliability-aware MBTA subway journey planner**. It plans
real journeys from the official schedule, then helps you understand the *risk*
in your trip: how much spare time you have at each transfer, what happens if
your first train is late, what the live network looks like right now, and how
train frequencies have actually behaved in recent historical data.

Every number is clearly labeled by source — **Scheduled**, **Live**,
**Simulated**, **Historical**, or **Experimental** — and the app never invents a
prediction or claims a probability it hasn't earned.

### Supported subway lines
Red, Orange, Blue, and Green (branches **B / C / D / E**). Direct trips plus
journeys with one or two transfers, in both directions.

---

## Features

- **Depart at** — plan forward from a departure time.
- **Arrive by** — plan backward from a destination deadline, with an optional
  personal safety margin; shows the latest trip that still makes it.
- **Scheduled journey + alternatives** — a clear leg-by-leg timeline (board,
  ride, transfer walk/wait, arrive) with up to three useful options.
- **Transfer Risk** — the usable spare time at each transfer, translated into
  plain English, with timing-based **Risky / Medium / High** labels (not
  probabilities).
- **Missed-transfer delay simulator** — a slider imagines your first train
  running late; the app recomputes every connection, rolls to the next **real
  scheduled** train after a missed connection, and shows your revised arrival
  and whether your deadline still holds.
- **Live MBTA predictions & alerts** — real-time predictions matched to your
  selected trip (near real time), plus journey-relevant service alerts, with a
  fetch timestamp in America/New_York.
- **Historical reliability** — observed train-to-train gaps (headways) from real
  MBTA performance data, shown only for transfers that have verified data.
- **Experimental ML + scenario arrival ranges** — a research model that
  estimates next-train headway, shown transparently against a baseline, plus a
  clearly labeled schedule-based scenario arrival range.

---

## Data sources

- **Official MBTA GTFS feed** — schedules, stops, trips, stop times, calendars,
  transfers. Source: `https://cdn.mbta.com/MBTA_GTFS.zip`
  (see the MBTA's GTFS documentation).
- **MBTA V3 API** — live predictions and service alerts
  (`https://api-v3.mbta.com`). Works without a key at a low rate limit; set
  `MBTA_API_KEY` for a higher limit.
- **MBTA LAMP Subway Performance Data** — historical observed headways
  (`https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/`).

---

## Limitations (read these — we keep them honest)

- The **historical sample is small**: it covers the **Red and Orange lines at
  Downtown Crossing** only. Transfers without verified data show
  "unavailable" rather than borrowing another station's numbers.
- **Scenario arrival ranges are NOT statistical confidence intervals.** They are
  schedule-based what-if ranges derived from observed headway spread; no
  calibrated probability is claimed.
- The **experimental ML headway model did NOT outperform its baseline** (a
  simple median by route/direction/hour) on held-out data. It is shown as a
  research result and does not change any journey recommendation.
- **Live predictions only cover trips near the current time** (roughly the next
  hour). Trips outside that horizon correctly show as unavailable and fall back
  to the scheduled plan.

---

## Project layout

```
TransitWiseAI/
├── app.py                     # Streamlit UI (all sections + section ordering)
├── gtfs_loader.py             # Loads GTFS files, stations, calendar, transfers
├── network.py                 # General multi-line routing graph + search (cached)
├── journey_planner.py         # Depart-at / Arrive-by journeys built on network.py
├── reliability.py             # Transfer buffer + multi-transfer delay simulator
├── mbta_api.py                # Live MBTA V3 predictions + alerts (stdlib only)
├── historical_reliability.py  # LAMP headway stats (station+line scoped)
├── ml_model.py                # Experimental headway model + baseline evaluation
├── prepare_historical.py      # One-time build of the cleaned historical CSV
├── download_gtfs.py           # Fetches the official GTFS feed into data/gtfs/
├── requirements.txt
├── test_network.py            # Routing tests (4 lines + branches)
├── test_arrive_by.py          # Arrive-by tests
├── test_phase9.py             # Full-app render + journey-specific historical tests
└── data/
    ├── gtfs/                  # MBTA GTFS .txt files (downloaded; gitignored)
    ├── historical/
    │   ├── dtx_headways.csv   # Small cleaned LAMP sample (committed)
    │   └── raw/               # Raw LAMP parquet (gitignored)
    └── cache/                 # Runtime artifacts, rebuilt automatically (gitignored)
```

---

## Setup and running (Windows / PowerShell)

> Commands use the project's virtual environment interpreter
> (`.\.venv\Scripts\python.exe`). Run them from the project root
> `E:\Kiro\TransitWiseAI` (adjust the path if your clone is elsewhere).

### 1. Create and activate a virtual environment (first time only)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 2. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. GTFS data (downloaded automatically on first startup)

The MBTA GTFS feed is **not** committed to the repo (it's large and changes
often). You do **not** have to fetch it manually: the app **automatically
downloads the official MBTA GTFS feed on first startup** when `data/gtfs/` is
missing — this is what makes the hosted deployment (Streamlit Community Cloud)
work from a fresh GitHub clone. A one-time loading message is shown while it
downloads (~22 MB), then it's cached on disk and subsequent loads are fast. The
download runs only when the required files are missing, not on every rerun.

If you prefer to pre-populate the data locally (optional), run:

```powershell
.\.venv\Scripts\python.exe download_gtfs.py
```

Both paths fetch `https://cdn.mbta.com/MBTA_GTFS.zip` (the official MBTA source)
and extract the required `.txt` files into `data/gtfs/`. If the download fails
(e.g. no network), the app shows a clear message instead of crashing.

### 4. (Optional) Rebuild the historical sample

The cleaned historical CSV (`data/historical/dtx_headways.csv`) is already
included. To rebuild it from the MBTA LAMP archive:

```powershell
.\.venv\Scripts\python.exe prepare_historical.py
```

### 5. Run the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit opens the app in your browser (usually http://localhost:8501).
Press `Ctrl + C` in the terminal to stop it.

> We use `python -m streamlit` because the bare `streamlit` command may not be
> on PATH on Windows. Both do the same thing.

---

## Expected first-run loading time

- **First request after launch:** the app parses the large GTFS feed
  (~2.7M `stop_times` rows) and builds a routing network, then writes small
  preprocessed cache artifacts to `data/cache/`. Expect roughly **15–60 seconds**
  the very first time.
- **Subsequent requests in the same session:** fast (the network and caches are
  reused); typically a few seconds, with brief live-API calls.
- Live sections make short network calls to the MBTA V3 API (with timeouts), so
  they add a second or two and degrade gracefully if the API is unreachable.

---

## Running the tests

```powershell
.\.venv\Scripts\python.exe test_network.py
.\.venv\Scripts\python.exe test_arrive_by.py
.\.venv\Scripts\python.exe test_phase9.py
```

Each script prints `[PASS]/[FAIL]` lines and a summary, and exits non-zero on
failure. `test_phase9.py` renders the full app headlessly (via Streamlit's
AppTest) and makes live API calls, so it takes a couple of minutes.

---

## How judges can test the application

1. Follow setup steps 1–3 above (venv, install, `download_gtfs.py`).
2. Launch: `.\.venv\Scripts\python.exe -m streamlit run app.py`.
3. **Depart at demo:** keep the defaults (**Harvard → Ruggles**, travel date
   **2026-10-24**, **9:00 AM**) and click **Plan My Journey**. You'll see the
   recommended trip, a leg-by-leg timeline, transfer risk, the delay simulator,
   live data, and the historical/ML section.
4. **Missed-transfer demo:** drag the **delay slider** up a few minutes and watch
   the connection get missed and the arrival roll to the next real train.
5. **Four-line demo:** try **Wonderland → Copley** (Blue → Green) or
   **Forest Hills → Airport** (Orange → Blue), or a Green branch-to-branch trip
   like **Boston College → Cleveland Circle**.
6. **Arrive-by demo:** switch the mode to **Arrive by**, set a deadline (e.g.
   9:00 AM) and an optional safety margin, and see the latest trip that makes it.

> Exact times depend on the GTFS feed you download, so they may differ slightly
> from the examples.

---

## Troubleshooting

- **"GTFS folder not found" / missing files:** run
  the app normally downloads the feed automatically on first startup. If that
  fails, open `https://cdn.mbta.com/MBTA_GTFS.zip` in a browser to confirm
  reachability, or run `.\.venv\Scripts\python.exe download_gtfs.py`, or
  download it manually and extract the `.txt` files into `data/gtfs/`.
- **First load feels slow / seems to hang:** the first startup downloads the
  GTFS feed (~22 MB) if missing, then does a one-time GTFS parse and cache build
  (15–60s total). It is fast afterward. Delete `data/cache/` to force a clean
  rebuild.
- **`streamlit` not recognized:** use `.\.venv\Scripts\python.exe -m streamlit
  run app.py` instead of the bare `streamlit` command.
- **Live data shows "unavailable":** the MBTA V3 API may be rate-limited,
  unreachable, or your selected trip is outside the ~1-hour live prediction
  horizon. The scheduled plan and simulator still work. Optionally set
  `MBTA_API_KEY` for a higher rate limit.
- **Historical section shows "unavailable for this transfer":** expected for any
  transfer other than Downtown Crossing (Red/Orange) — the verified sample is
  intentionally narrow and we do not substitute other stations' data.

---

## A note on honesty

TransitWise AI is deliberately conservative about what it claims. Risk labels are
timing-based categories, not probabilities. Scenario ranges are not confidence
intervals. The experimental model is reported against its baseline, including the
fact that it did not beat it. Scheduled, live, simulated, historical, and
experimental values are always labeled separately so a commuter always knows
what they're looking at.
