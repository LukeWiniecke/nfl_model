"""Always-on supervisor: nothing manual. Sleeps cheaply between games, polls
fast while a game is live or about to start, so it needs no schedule of its own
and handles odd slots (London 9:30am ET, Thursday/Saturday/Monday, flex games).

    .venv/bin/python -m live.watcher
"""

import time
from datetime import datetime, timedelta, timezone

from .espn import fetch_scoreboard
from .poll import poll_once

LEAD = timedelta(minutes=20)   # start fast polling this long before kickoff
FAST, IDLE = 15, 300           # seconds between polls: live vs waiting


def seconds_to_sleep(events: list[dict], now: datetime) -> int:
    """FAST if any game is in progress or kicks off within LEAD, else IDLE
    (shortened so we wake in time for the next kickoff)."""
    soonest = None
    for ev in events:
        state = ev["status"]["type"]["state"]
        if state == "in":
            return FAST
        if state == "pre":
            ko = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
            if ko - LEAD <= now:
                return FAST
            soonest = ko if soonest is None else min(soonest, ko)
    if soonest is None:
        return IDLE
    return max(FAST, min(IDLE, int((soonest - LEAD - now).total_seconds())))


def main():
    last: dict = {}
    while True:
        wait = IDLE
        try:
            events = fetch_scoreboard().get("events", [])
            wait = seconds_to_sleep(events, datetime.now(timezone.utc))
            n = poll_once(last) if wait == FAST else 0
            print(f"{datetime.now():%a %H:%M:%S} next in {wait}s, wrote {n}", flush=True)
        except Exception as e:
            print(f"watcher error: {e}", flush=True)
        time.sleep(wait)


if __name__ == "__main__":
    main()
