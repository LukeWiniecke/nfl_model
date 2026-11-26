"""Team tendencies as of BEFORE each game, from nflverse play-by-play (public data only).

How a team plays, beyond how good it is (the pregame spread/total already price strength):
  pace       seconds between the offense's snaps within a drive
  nohuddle   share of snaps run without a huddle
  proe       pass rate over expected (nflverse pass_oe, percentage points)
  go4        go-for-it rate on 4th-and-3-or-less from midfield in
  two_pt     2-point tries per touchdown
All measured in NEUTRAL states (quarters 1-3, score within 10) so they reflect the coach's style,
not the scoreboard. Pre-game value = decayed running average over the team's previous games,
shrunk toward the previous season's league average (prior worth PRIOR_GAMES games), with older
seasons down-weighted. Only games before the game in question are used.

    .venv/bin/python -m live.tendencies        # pull play-by-play columns (cached) and build the table
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PBP = ROOT / "data/raw/nflverse/pbp_tendency.parquet"
OUT = ROOT / "data/raw/nflverse/team_tendencies.parquet"
COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team", "play_id", "drive",
        "play_type", "down", "ydstogo", "yardline_100", "qtr", "game_seconds_remaining", "score_differential",
        "no_huddle", "pass_oe", "two_point_attempt", "extra_point_attempt", "qb_kneel", "qb_spike"]
METRICS = {"pace": ("pace_sum", "pace_n"), "nohuddle": ("nh_sum", "nh_n"), "proe": ("proe_sum", "proe_n"),
           "go4": ("go_sum", "go_n"), "two_pt": ("two_sum", "two_n")}
DECAY, SEASON_DECAY, PRIOR_GAMES = 0.9, 0.5, 4.0


def pull(seasons=None, force: bool = False) -> pd.DataFrame:
    if PBP.exists() and not force:
        return pd.read_parquet(PBP)
    import nflreadpy as nfl
    from src.data_ingestion import SEASONS
    frames = []
    for s in seasons or SEASONS:
        try:
            df = nfl.load_pbp(seasons=[s])
        except Exception as e:  # season not published yet, network, ...
            print(f"skip {s}: {e}")
            continue
        frames.append(df.select([c for c in COLS if c in df.columns]))
    p = pl.concat(frames, how="diagonal_relaxed")
    p.write_parquet(PBP)
    return p.to_pandas()


def game_metrics(p: pd.DataFrame) -> pd.DataFrame:
    """Sums and counts per (game, offense) for each tendency."""
    p = p[p.posteam.notna()].sort_values(["game_id", "play_id"]).copy()
    scrim = p.play_type.isin(["pass", "run"]) & (p.qb_kneel.fillna(0) == 0) & (p.qb_spike.fillna(0) == 0)
    neutral = (p.qtr <= 3) & (p.score_differential.abs() <= 10)
    # pace: gap to the same offense's next snap in the same drive and quarter
    s = p[scrim].copy()
    nxt = s.groupby("game_id")[["game_seconds_remaining", "drive", "posteam", "qtr"]].shift(-1)
    same = (nxt.drive == s.drive) & (nxt.posteam == s.posteam) & (nxt.qtr == s.qtr)
    s["gap"] = np.where(same, (s.game_seconds_remaining - nxt.game_seconds_remaining).clip(0, 60), np.nan)
    s_neu = s[neutral[s.index]]
    g = s_neu.groupby(["game_id", "posteam"]).agg(pace_sum=("gap", "sum"), pace_n=("gap", "count"),
                                                  nh_sum=("no_huddle", "sum"), nh_n=("no_huddle", "count"),
                                                  proe_sum=("pass_oe", "sum"), proe_n=("pass_oe", "count"))
    fourth = p[(p.down == 4) & (p.ydstogo <= 3) & (p.yardline_100 <= 50) & neutral
               & p.play_type.isin(["pass", "run", "punt", "field_goal"])]
    g4 = fourth.assign(go=fourth.play_type.isin(["pass", "run"]).astype(float)).groupby(["game_id", "posteam"]).agg(
        go_sum=("go", "sum"), go_n=("go", "count"))
    tries = p[(p.two_point_attempt == 1) | (p.extra_point_attempt == 1)]
    g2 = tries.assign(two=(tries.two_point_attempt == 1).astype(float)).groupby(["game_id", "posteam"]).agg(
        two_sum=("two", "sum"), two_n=("two", "count"))
    out = g.join(g4, how="outer").join(g2, how="outer").fillna(0.0).reset_index().rename(columns={"posteam": "team"})
    meta = p.groupby("game_id").agg(season=("season", "first"), week=("week", "first"))
    return out.merge(meta, on="game_id")


def rolling(gm: pd.DataFrame) -> pd.DataFrame:
    """Pre-game tendency per (game_id, team) from that team's earlier games only."""
    league = {m: gm.groupby("season")[s].sum() / gm.groupby("season")[n].sum() for m, (s, n) in METRICS.items()}
    rows = []
    for team, g in gm.sort_values(["season", "week", "game_id"]).groupby("team"):
        S = {m: 0.0 for m in METRICS}
        N = {m: 0.0 for m in METRICS}
        last_season = None
        for r in g.itertuples():
            if last_season is not None and r.season != last_season:
                for m in METRICS:
                    S[m] *= SEASON_DECAY
                    N[m] *= SEASON_DECAY
            rec = {"game_id": r.game_id, "team": team}
            for m, (s, n) in METRICS.items():
                prior = league[m].get(r.season - 1, league[m].get(r.season))
                rec[m] = (prior * PRIOR_GAMES * _per_game(gm, n) + S[m]) / (PRIOR_GAMES * _per_game(gm, n) + N[m])
            rows.append(rec)
            for m, (s, n) in METRICS.items():  # update AFTER recording: the game never sees itself
                S[m] = S[m] * DECAY + getattr(r, s)
                N[m] = N[m] * DECAY + getattr(r, n)
            last_season = r.season
    return pd.DataFrame(rows)


_avg_n: dict = {}


def _per_game(gm: pd.DataFrame, ncol: str) -> float:
    """Typical per-game count of a metric's observations (scales the prior to 'games')."""
    if ncol not in _avg_n:
        _avg_n[ncol] = float(gm[ncol].mean())
    return _avg_n[ncol]


def build(force_pull: bool = False) -> pd.DataFrame:
    gm = game_metrics(pull(force=force_pull))
    t = rolling(gm)
    t.to_parquet(OUT, index=False)
    return t


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT) if OUT.exists() else build()


def league_to_date(gm: pd.DataFrame) -> pd.DataFrame:
    """League average of each tendency over games BEFORE each (season, week); week 1 uses the
    previous season. Lets tendencies be expressed relative to their era (go-for-it rates doubled
    2010 -> 2025, so raw values drift and confound with team strength)."""
    rows = []
    for season, g in gm.groupby("season"):
        prev = gm[gm.season == season - 1]
        for wk in sorted(g.week.unique()):
            before = g[g.week < wk]
            base = before if len(before) else (prev if len(prev) else g)
            rows.append({"season": season, "week": wk,
                         **{f"lg_{m}": base[s].sum() / max(base[n].sum(), 1e-9) for m, (s, n) in METRICS.items()}})
    return pd.DataFrame(rows)


def attach(d: pd.DataFrame, relative: bool = True) -> pd.DataFrame:
    """Add home_*/away_* and offense/defense (off_*/def_*) tendencies plus game-level means to rows
    with game_id, home_team, away_team, pos_home. relative=True expresses each tendency as team minus
    league average to date (recommended: raw values drift across eras)."""
    t = load()
    if relative:
        gm = game_metrics(pull())
        meta = gm[["game_id", "season", "week"]].drop_duplicates("game_id")
        t = t.merge(meta, on="game_id").merge(league_to_date(gm), on=["season", "week"], how="left")
        for m in METRICS:
            t[m] = t[m] - t[f"lg_{m}"]
        t = t[["game_id", "team"] + list(METRICS)]
    d = d.merge(t.add_prefix("home_t_").rename(columns={"home_t_game_id": "game_id", "home_t_team": "home_team"}),
                on=["game_id", "home_team"], how="left")
    d = d.merge(t.add_prefix("away_t_").rename(columns={"away_t_game_id": "game_id", "away_t_team": "away_team"}),
                on=["game_id", "away_team"], how="left")
    for m in METRICS:
        h, a = d[f"home_t_{m}"], d[f"away_t_{m}"]
        d[f"off_t_{m}"] = np.where(d.pos_home == 1, h, a)
        d[f"def_t_{m}"] = np.where(d.pos_home == 1, a, h)
        d[f"game_t_{m}"] = (h + a) / 2
    return d


if __name__ == "__main__":
    t = build()
    print(t.describe().T[["mean", "std", "min", "max"]].round(3))
