"""
ml_model.py
-----------
Experimental machine-learning model for TransitWise AI.

TARGET (actually supported by our data): the observed next-train HEADWAY in
minutes at a station, i.e. how long after the previous same-direction train the
next one arrives. This is NOT a full-journey arrival time and must never be
presented as one.

Data: data/historical/dtx_headways.csv — real MBTA LAMP observations for the
Red and Orange lines at Downtown Crossing, both directions, 5 recent weekdays.

Method (leakage-safe):
  - Features are all knowable BEFORE the event: route, direction, hour of day,
    day of week, scheduled headway, and the PREVIOUS observed headway on the
    same route/direction/service-day (shifted, never the current target).
  - Baseline: median observed headway for the same route/direction/hour,
    computed on TRAIN data only.
  - Model: HistGradientBoostingRegressor (good for small tabular data) inside a
    scikit-learn Pipeline so encoders are fit on training data only.
  - Chronological split: the newest service dates are held out as the test set.

We report MAE, median AE, % within 1/2/5 min, test size and date range, and a
baseline comparison. If the dataset is too small, we return a baseline-only
result and say so — we never force a model or claim unvalidated accuracy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

import historical_reliability as hist

MIN_ROWS_FOR_MODEL = 500        # below this we keep baseline only
TEST_FRACTION_DATES = 0.4       # hold out the newest ~40% of service dates


@dataclass
class MLResult:
    trained: bool
    target_name: str
    rows_total: int
    rows_train: int
    rows_test: int
    test_date_min: str
    test_date_max: str
    # metrics (minutes); model vs baseline on the SAME test set
    baseline_mae: float | None = None
    baseline_medae: float | None = None
    model_mae: float | None = None
    model_medae: float | None = None
    within1: float | None = None     # % of model predictions within ±1 min
    within2: float | None = None
    within5: float | None = None
    baseline_within2: float | None = None
    note: str = ""
    feature_names: list = field(default_factory=list)


def _prepare_frame() -> pd.DataFrame:
    """Load the cleaned CSV and engineer leakage-safe features."""
    df = hist.load_dataset().copy()
    df = df[df["headway_trunk_seconds"].notna()].copy()
    # Target in minutes.
    df["headway_min"] = df["headway_trunk_seconds"] / 60.0
    # Drop clear outliers / invalid (<=0 or absurdly long service gaps).
    df = df[(df["headway_min"] > 0) & (df["headway_min"] <= 30)].copy()

    # Time-of-day features (local clock hour 0-23; after-midnight folds in).
    df["hour"] = ((df["start_time"] // 3600) % 24).astype(int)
    # Day of week from service_date (YYYYMMDD int).
    df["date"] = pd.to_datetime(df["service_date"].astype(str), format="%Y%m%d")
    df["dow"] = df["date"].dt.dayofweek
    df["sched_headway_min"] = pd.to_numeric(
        df.get("scheduled_headway_trunk"), errors="coerce"
    ) / 60.0

    # Order chronologically within each route/direction/service-day, then take
    # the PREVIOUS observed headway as a feature (known before current event).
    df = df.sort_values(["route_id", "direction_id", "service_date", "start_time"])
    df["prev_headway_min"] = (
        df.groupby(["route_id", "direction_id", "service_date"])["headway_min"]
        .shift(1)
    )
    # First train of a group has no previous observation -> fill with the
    # scheduled headway (a reasonable, leakage-free prior), else the group mean
    # of the scheduled headway.
    df["prev_headway_min"] = df["prev_headway_min"].fillna(df["sched_headway_min"])
    df["sched_headway_min"] = df["sched_headway_min"].fillna(df["sched_headway_min"].median())
    df["prev_headway_min"] = df["prev_headway_min"].fillna(df["headway_min"].median())

    return df


def _chronological_split(df: pd.DataFrame):
    """Hold out the newest service dates as the test set."""
    dates = sorted(df["service_date"].unique())
    n_test = max(1, int(round(len(dates) * TEST_FRACTION_DATES)))
    test_dates = set(dates[-n_test:])
    train = df[~df["service_date"].isin(test_dates)]
    test = df[df["service_date"].isin(test_dates)]
    return train, test


@lru_cache(maxsize=1)
def train_and_evaluate() -> MLResult:
    """Train baseline + model and evaluate on a held-out chronological test set."""
    df = _prepare_frame()
    total = len(df)

    train, test = _chronological_split(df)
    if train.empty or test.empty:
        return MLResult(
            trained=False, target_name="observed next-train headway (minutes)",
            rows_total=total, rows_train=len(train), rows_test=len(test),
            test_date_min="", test_date_max="",
            note="Not enough distinct service dates to form a chronological split.",
        )

    def fmt_date(v):
        s = str(int(v))
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    test_dmin = fmt_date(test["service_date"].min())
    test_dmax = fmt_date(test["service_date"].max())

    y_test = test["headway_min"].to_numpy()

    # --- Baseline: median headway per (route, direction, hour) from TRAIN only.
    key = ["route_id", "direction_id", "hour"]
    baseline_table = train.groupby(key)["headway_min"].median()
    global_median = float(train["headway_min"].median())

    def baseline_predict(frame):
        preds = []
        for _, r in frame.iterrows():
            k = (r["route_id"], r["direction_id"], r["hour"])
            preds.append(baseline_table.get(k, global_median))
        return np.array(preds, dtype=float)

    baseline_pred = baseline_predict(test)
    baseline_mae = float(np.mean(np.abs(baseline_pred - y_test)))
    baseline_medae = float(np.median(np.abs(baseline_pred - y_test)))
    baseline_within2 = float(np.mean(np.abs(baseline_pred - y_test) <= 2.0) * 100)

    # If too small, keep baseline only.
    if total < MIN_ROWS_FOR_MODEL:
        return MLResult(
            trained=False, target_name="observed next-train headway (minutes)",
            rows_total=total, rows_train=len(train), rows_test=len(test),
            test_date_min=test_dmin, test_date_max=test_dmax,
            baseline_mae=round(baseline_mae, 2), baseline_medae=round(baseline_medae, 2),
            baseline_within2=round(baseline_within2, 1),
            note=f"Dataset too small ({total} rows) to train a model responsibly; "
                 "baseline median retained.",
        )

    # --- Model: HistGradientBoostingRegressor in a leakage-safe pipeline.
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import HistGradientBoostingRegressor

    categorical = ["route_id", "direction_id"]
    numeric = ["hour", "dow", "sched_headway_min", "prev_headway_min"]
    features = categorical + numeric

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    model = Pipeline([
        ("pre", pre),
        ("gb", HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.08, max_iter=300,
            l2_regularization=1.0, random_state=42,
        )),
    ])

    X_train = train[features]
    y_train = train["headway_min"].to_numpy()
    model.fit(X_train, y_train)

    model_pred = model.predict(test[features])
    err = np.abs(model_pred - y_test)
    model_mae = float(np.mean(err))
    model_medae = float(np.median(err))
    within1 = float(np.mean(err <= 1.0) * 100)
    within2 = float(np.mean(err <= 2.0) * 100)
    within5 = float(np.mean(err <= 5.0) * 100)

    # Persist the fitted model for reuse by the app (best-effort).
    try:
        import joblib
        os.makedirs(os.path.join("data", "cache"), exist_ok=True)
        joblib.dump(model, os.path.join("data", "cache", "headway_model.joblib"))
    except Exception:
        pass

    return MLResult(
        trained=True, target_name="observed next-train headway (minutes)",
        rows_total=total, rows_train=len(train), rows_test=len(test),
        test_date_min=test_dmin, test_date_max=test_dmax,
        baseline_mae=round(baseline_mae, 2), baseline_medae=round(baseline_medae, 2),
        baseline_within2=round(baseline_within2, 1),
        model_mae=round(model_mae, 2), model_medae=round(model_medae, 2),
        within1=round(within1, 1), within2=round(within2, 1), within5=round(within5, 1),
        feature_names=features,
        note="HistGradientBoostingRegressor; chronological split (newest dates held out).",
    )
