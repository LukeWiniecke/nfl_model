"""Thin ESPN (unofficial, keyless) client: one scoreboard call returns every
game's clock, score, situation and sportsbook line, so a poll is one request."""

import time
import requests

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"


def _get(url: str, params: dict | None = None, tries: int = 3) -> dict:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(1 + i)


def fetch_scoreboard() -> dict:
    return _get(SCOREBOARD)


def fetch_summary(event_id: str) -> dict:
    return _get(SUMMARY, {"event": event_id})


def parse_event(ev: dict, ts: str) -> dict:
    """Flatten one scoreboard event into a snapshot row. Every field uses .get
    because ESPN omits `situation`/`odds` outside live play."""
    comp = ev["competitions"][0]
    status = ev["status"]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    sit = comp.get("situation") or {}
    odds = (comp.get("odds") or [{}])[0]
    poss_id = sit.get("possession")
    poss = next((c["team"]["abbreviation"] for c in comp["competitors"] if c["team"]["id"] == poss_id), None)
    return {
        "ts": ts,
        "game_id": ev["id"],
        "state": status["type"]["state"],  # pre / in / post
        "period": status.get("period"),
        "clock_sec": status.get("clock"),
        "clock_display": status.get("displayClock"),
        "home": teams["home"]["team"]["abbreviation"],
        "away": teams["away"]["team"]["abbreviation"],
        "home_score": int(teams["home"].get("score") or 0),
        "away_score": int(teams["away"].get("score") or 0),
        "possession": poss,
        "down": sit.get("down"),
        "distance": sit.get("distance"),
        "yard_line_text": sit.get("possessionText") or sit.get("downDistanceText"),
        "yards_to_endzone": sit.get("yardLine"),
        "is_red_zone": sit.get("isRedZone"),
        "home_timeouts": sit.get("homeTimeouts"),
        "away_timeouts": sit.get("awayTimeouts"),
        "last_play": (sit.get("lastPlay") or {}).get("text"),
        "book": (odds.get("provider") or {}).get("name"),
        "spread_details": odds.get("details"),  # e.g. "KC -3"
        "spread": odds.get("spread"),
        "total": odds.get("overUnder"),
        "home_ml": (odds.get("homeTeamOdds") or {}).get("moneyLine"),
        "away_ml": (odds.get("awayTeamOdds") or {}).get("moneyLine"),
    }
