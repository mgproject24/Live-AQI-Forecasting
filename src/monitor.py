"""
Reads the experiment log (models/experiment_log.jsonl) and flags two
things the per-run promotion gate in train.py can't see on its own:

1. Cumulative drift: MAE can regress by, say, 9% a day forever and never
   trip the 10%-per-run promotion gate, while still getting steadily
   worse over weeks. This compares today's MAE against MAE from N runs
   ago for the same (location, horizon) to catch that trend.
2. Repeated baseline failures: if a model has failed to beat the naive
   persistence baseline for several runs in a row, something is wrong
   with training itself (bad data, a broken feature, etc.), not just
   normal noise.

Writes models/monitor_report.json with any alerts found. Exit code stays
0 either way (this is observability, not a hard gate) - wire the alerts
into GitHub Actions to open an issue rather than fail the pipeline, since
a bad forecast being flagged is still better than no forecast at all.

Usage:
    python src/monitor.py
"""

import json
from collections import defaultdict
from datetime import datetime, timezone

from config import MODELS_DIR

EXPERIMENT_LOG_PATH = MODELS_DIR / "experiment_log.jsonl"
MONITOR_REPORT_PATH = MODELS_DIR / "monitor_report.json"

DRIFT_LOOKBACK_RUNS = 7      # compare against ~a week ago
DRIFT_THRESHOLD = 1.25        # alert if MAE is >25% worse than that run
CONSECUTIVE_REJECTION_THRESHOLD = 3


def _load_log():
    if not EXPERIMENT_LOG_PATH.exists():
        return []
    records = []
    with open(EXPERIMENT_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _group_by_key(records):
    grouped = defaultdict(list)
    for r in records:
        if r.get("status") != "trained":
            continue
        key = (r["location"], r["horizon"])
        grouped[key].append(r)
    for key in grouped:
        grouped[key].sort(key=lambda r: r["timestamp"])
    return grouped


def check_cumulative_drift(grouped):
    alerts = []
    for (location, horizon), runs in grouped.items():
        promoted_runs = [r for r in runs if r.get("mae") is not None]
        if len(promoted_runs) < DRIFT_LOOKBACK_RUNS + 1:
            continue  # not enough history yet to judge a trend

        latest = promoted_runs[-1]
        reference = promoted_runs[-(DRIFT_LOOKBACK_RUNS + 1)]

        if reference["mae"] <= 0:
            continue
        ratio = latest["mae"] / reference["mae"]
        if ratio > DRIFT_THRESHOLD:
            alerts.append({
                "type": "cumulative_drift",
                "location": location,
                "horizon": horizon,
                "detail": (
                    f"MAE drifted from {reference['mae']:.2f} to {latest['mae']:.2f} "
                    f"over the last {DRIFT_LOOKBACK_RUNS} runs "
                    f"({(ratio - 1) * 100:.0f}% worse) - no single run tripped the "
                    f"promotion gate, but the trend is real."
                ),
            })
    return alerts


def check_repeated_rejections(grouped):
    alerts = []
    for (location, horizon), runs in grouped.items():
        recent = runs[-CONSECUTIVE_REJECTION_THRESHOLD:]
        if len(recent) < CONSECUTIVE_REJECTION_THRESHOLD:
            continue
        if all(not r.get("promoted", False) for r in recent):
            alerts.append({
                "type": "repeated_rejections",
                "location": location,
                "horizon": horizon,
                "detail": (
                    f"The last {CONSECUTIVE_REJECTION_THRESHOLD} training runs were "
                    f"all rejected by the promotion gate - the model has stopped "
                    f"beating its baseline/previous version repeatedly, not just once."
                ),
            })
    return alerts


def main():
    records = _load_log()
    grouped = _group_by_key(records)

    alerts = check_cumulative_drift(grouped) + check_repeated_rejections(grouped)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_experiment_records": len(records),
        "n_location_horizon_pairs_tracked": len(grouped),
        "alerts": alerts,
    }
    with open(MONITOR_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    if alerts:
        print(f"MONITOR: {len(alerts)} alert(s) found:")
        for a in alerts:
            print(f"  [{a['type']}] {a['location']} h={a['horizon']}: {a['detail']}")
    else:
        print("MONITOR: no alerts. All tracked models within expected range.")

    print(f"Saved monitor report to {MONITOR_REPORT_PATH}")


if __name__ == "__main__":
    main()
