"""Turn a FINISHED game's ESPN play-by-play into snapshot rows, so the Live tab
and terminal monitor can be exercised without waiting for a live game.

    .venv/bin/python -m live.replay 401873052 --out /tmp/replay_dir
    NFL_LIVE_DIR=/tmp/replay_dir streamlit run app/streamlit_app.py

Timeouts and in-game odds are not in the feed (timeouts default to 3; the
pregame line is used throughout), so replay checks plumbing, not market gaps.
"""

import argparse
import json
from pathlib import Path

from .espn import fetch_summary


def to_rows(event_id: str) -> list[dict]:
    s = fetch_summary(event_id)
    comp = s["header"]["competitions"][0]
    teams = {c["homeAway"]: c["team"]["abbreviation"] for c in comp["competitors"]}
    tid = {c["team"]["id"]: c["team"]["abbreviation"] for c in comp["competitors"]}
    pc = (s.get("pickcenter") or [{}])[0]
    details, total, spread = pc.get("details"), pc.get("overUnder"), pc.get("spread")
    base = dict(game_id=event_id, home=teams["home"], away=teams["away"], book="replay",
                spread_details=details, spread=spread, total=total, home_ml=None, away_ml=None,
                home_timeouts=None, away_timeouts=None, last_play=None, is_red_zone=None, clock_display=None)
    rows = [dict(base, ts="2000-01-01T00:00:00+00:00", state="pre", period=0, clock_sec=0.0, home_score=0, away_score=0,
                 possession=None, down=None, distance=None, yard_line_text=None, yards_to_endzone=None)]
    plays = [p for d in s["drives"]["previous"] for p in d["plays"]]
    for i, p in enumerate(plays):
        st = p.get("start") or {}
        poss = tid.get((st.get("team") or {}).get("id")) if isinstance(st.get("team"), dict) else None
        ytg = st.get("yardsToEndzone")
        mm, ss = (p.get("clock", {}).get("displayValue", "0:00") + ":0").split(":")[:2]
        rows.append(dict(base, ts=f"2000-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00", state="in",
                         period=p["period"]["number"], clock_sec=float(int(mm) * 60 + int(ss)),
                         home_score=p["homeScore"], away_score=p["awayScore"], possession=poss,
                         down=st.get("down"), distance=st.get("distance"),
                         yard_line_text=f"{poss} {100 - ytg}" if poss and ytg is not None else None,
                         yards_to_endzone=ytg, last_play=p.get("text")))
    last = dict(rows[-1], state="post", ts="2000-01-01T05:00:00+00:00", period=4, clock_sec=0.0)
    return rows + [last]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event_id")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = to_rows(a.event_id)
    with (out / "snapshots_2000-01-01.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(len(rows), "rows ->", out)
