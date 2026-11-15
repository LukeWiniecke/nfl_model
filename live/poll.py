"""Record live NFL game snapshots (state + sportsbook line) to JSONL.

    .venv/bin/python -m live.poll                 # poll every 15s, run forever
    .venv/bin/python -m live.poll --interval 10 --once

Appends only while a game is in progress (plus one 'pre'/'post' row per game
so the closing and final lines are captured). Output: data/live/snapshots_<date>.jsonl
Rows are deduplicated: a row is written only if state or line changed.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .espn import fetch_scoreboard, parse_event

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "live"
KEY_FIELDS = ("period", "clock_display", "home_score", "away_score", "possession", "down",
              "distance", "yards_to_endzone", "spread", "total", "home_ml", "away_ml", "state")


def poll_once(last: dict) -> int:
    now = datetime.now(timezone.utc)
    ts = now.isoformat(timespec="seconds")
    rows = []
    for ev in fetch_scoreboard().get("events", []):
        row = parse_event(ev, ts)
        sig = tuple(row[k] for k in KEY_FIELDS)
        # 'pre' and 'post' only need recording on change (captures closing/final line)
        if last.get(row["game_id"]) == sig:
            continue
        last[row["game_id"]] = sig
        rows.append(row)
    if rows:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"snapshots_{now:%Y-%m-%d}.jsonl"
        with path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    last: dict = {}
    while True:
        try:
            n = poll_once(last)
            print(f"{datetime.now():%H:%M:%S} wrote {n} rows", flush=True)
        except Exception as e:  # keep recording through transient API failures
            print(f"poll failed: {e}", flush=True)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
