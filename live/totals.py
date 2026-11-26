"""Live TOTALS: full distribution of the points still to be scored.

A multiclass LightGBM predicts P(remaining total points = r) for r = 0..75 from the game
state; P(over k) is a sum over that distribution, which handles the lumpiness of scoring
(3s and 7s).

v2 (2026-11-26) adds the decisions that change scoring as a game develops:
  - game script: margin from the offense's view, how many scores the trailing side needs,
    garbage time, leader / trailer with the ball late, two-minute drills, clock the leader
    can burn, both teams' timeouts, overtime;
  - each team's implied points from the pregame lines;
  - the real shape of scoring through a game (bumps before halftime and at the end) instead
    of a straight line, learned from the training seasons and saved with the model;
  - tempo: plays per minute so far (counted the way the live feed counts snaps);
  - kickoff weather (indoor, temperature, wind, rain, snow; live/weather.py), typical-outdoor defaults
    when unknown. Fixes a ~1-point over-prediction in windy / wet games.
Held out 2022-25 (train <= 2021): over/under log loss 0.5874 -> 0.5865, last 5 minutes
0.3366 -> 0.3270, calibration of late over/under probabilities still ~1.0.

    .venv/bin/python -m live.totals            # held-out evaluation, then fit all + save
    .venv/bin/python -m live.totals --eval     # evaluation only
    .venv/bin/python -m live.totals --fit      # fit all + save
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from .baseline import MIN_LEAF, PATH, load
from .ep import EP_FEATS, get_ep_model

TOT_PATH = Path(__file__).resolve().parents[1] / "data/live/totals.joblib"
VERSION = 3
CLIP = 75
RT = np.arange(0, CLIP + 1)  # class i <-> remaining points RT[i]
BASE = ["t", "half_t", "qtr", "cur_total", "lead", "abs_lead", "spread", "total", "pos_home", "down", "ydstogo",
        "yardline_100", "pos_to", "def_to", "ep_off", "exp_rem_tot", "pace"]
SCRIPT = ["off_imp", "def_imp", "pos_lead", "f_t", "exp_rem_curve", "garbage", "leader_ball_late", "trailer_ball_late",
          "two_min", "burn_margin", "need", "home_to", "away_to", "is_ot"]
TEMPO = ["tempo", "plays_so_far", "pts_per_play"]
WX = ["indoor", "temp_f", "wind_mph", "rain", "snow"]  # kickoff weather (live/weather.py); defaults when unknown
TF = BASE + SCRIPT + TEMPO + WX
PLAYS_PER_MIN = 2.45  # league-typical snaps (downs 1-4) per game minute: fallback when the count is unknown
PARAMS = dict(objective="multiclass", num_class=len(RT), n_estimators=120, learning_rate=0.08, num_leaves=15,
              min_child_samples=MIN_LEAF, subsample=0.6, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def base_features(d: pd.DataFrame) -> pd.DataFrame:
    """Needs t, half_t, qtr, cur_total (PRE-play), lead, spread, total (pregame line), pos_home,
    down, ydstogo, yardline_100, pos_to, def_to, plays_so_far (snaps with a down 1-4 so far)."""
    d = d.copy()
    d["abs_lead"] = d.lead.abs()
    d["ep_off"] = get_ep_model().predict(d[EP_FEATS].astype(float))  # value of the current drive to the offense
    d["exp_rem_tot"] = d.total * d.t / 3600
    d["pace"] = d.cur_total - d.total * (3600 - d.t) / 3600          # points above/below the pregame pace
    d["home_imp"], d["away_imp"] = (d.total + d.spread) / 2, (d.total - d.spread) / 2
    d["off_imp"] = np.where(d.pos_home == 1, d.home_imp, d.away_imp)
    d["def_imp"] = np.where(d.pos_home == 1, d.away_imp, d.home_imp)
    d["pos_lead"] = np.where(d.pos_home == 1, d.lead, -d.lead)
    d["garbage"] = ((d.abs_lead >= 17) & (d.t <= 900)).astype(float)
    d["leader_ball_late"] = ((d.pos_lead > 0) & (d.t <= 300)).astype(float)
    d["trailer_ball_late"] = ((d.pos_lead < 0) & (d.t <= 300)).astype(float)
    d["two_min"] = (d.half_t <= 120).astype(float)
    downs_left = np.where(d.down.between(1, 4), 5 - d.down, 4)
    d["burn_margin"] = d.t - np.maximum(downs_left * 40 - d.def_to * 40, 0)   # <= 0: leader with the ball can kneel out
    d["need"] = np.select([d.pos_lead > 16, d.pos_lead > 8, d.pos_lead > 3, d.pos_lead > 0, d.pos_lead == 0,
                           d.pos_lead >= -3, d.pos_lead >= -8, d.pos_lead >= -16], [4, 3, 2, 1, 0, -1, -2, -3], -4)
    d["home_to"] = np.where(d.pos_home == 1, d.pos_to, d.def_to)
    d["away_to"] = np.where(d.pos_home == 1, d.def_to, d.pos_to)
    d["is_ot"] = (d.qtr >= 5).astype(float)
    elapsed_min = (3600 - d.t) / 60
    if "plays_so_far" not in d or d.plays_so_far.isna().all():
        d["plays_so_far"] = np.round(elapsed_min * PLAYS_PER_MIN)
    d["plays_so_far"] = d.plays_so_far.fillna(np.round(elapsed_min * PLAYS_PER_MIN))
    d["tempo"] = d.plays_so_far / np.maximum(elapsed_min, 3)
    d["pts_per_play"] = d.cur_total / np.maximum(d.plays_so_far, 10)
    from .weather import DEFAULTS
    for c, v in DEFAULTS.items():
        d[c] = d[c].astype(float).fillna(v) if c in d else v
    return d


def scoring_curve(d: pd.DataFrame) -> pd.Series:
    """Share of the pregame total still to come, by minute of game time left (training rows)."""
    b = np.digitize(d.t, np.arange(0, 3660, 60))
    return d.y_rem_tot.groupby(b).sum() / d.total.groupby(b).sum()


def curve_features(d: pd.DataFrame, curve: pd.Series) -> pd.DataFrame:
    d = d.copy()
    b = np.digitize(d.t, np.arange(0, 3660, 60))
    d["f_t"] = pd.Series(b, index=d.index).map(curve).fillna(d.t / 3600)
    d["exp_rem_curve"] = d.total * d.f_t
    return d


def training_frame() -> pd.DataFrame:
    """baseline.load() rows plus the PRE-play total, the final total and the snap count so far.
    (load() overwrites nflverse `total`, the final total, with the pregame line.)"""
    raw = (pl.read_parquet(PATH).sort(["game_id", "play_id"])
           .with_columns(post=pl.col("total_home_score") + pl.col("total_away_score"),
                         snap=pl.col("down").is_between(1, 4).fill_null(False).cast(pl.Int32))
           .with_columns(cur_total=pl.col("post").shift(1).over("game_id").fill_null(0),
                         plays_so_far=pl.col("snap").cum_sum().shift(1).over("game_id").fill_null(0))
           .select("game_id", "play_id", "cur_total", "plays_so_far", final_total=pl.col("total")))
    from .weather import game_weather
    d = load().join(raw, on=["game_id", "play_id"], how="left").to_pandas().merge(game_weather(), on="game_id", how="left")
    d = d[d.final_total.notna() & d.cur_total.notna()]
    d["y_rem_tot"] = d.final_total - d.cur_total
    d = d[d.y_rem_tot >= 0]
    return base_features(d)


def _fit(d: pd.DataFrame) -> dict:
    curve = scoring_curve(d)
    X = curve_features(d, curve)
    y = np.clip(np.rint(X.y_rem_tot), 0, CLIP).astype(int)
    assert len(np.unique(y)) == len(RT), "a remaining-points class is missing from training data"
    return dict(clf=lgb.LGBMClassifier(**PARAMS).fit(X[TF], y), curve=curve, version=VERSION)


def pmf(model: dict, X: pd.DataFrame) -> np.ndarray:
    if "f_t" not in X:
        X = curve_features(X, model["curve"])
    return model["clf"].predict_proba(X[TF])


def prob_over(P: np.ndarray, cur_total, k: float, push: float = 0.0) -> np.ndarray:
    """P(final total > k) (+ push * P(final total == k))."""
    final = np.asarray(cur_total, dtype=float)[:, None] + RT[None, :]
    return (P * ((final > k) + push * (final == k))).sum(axis=1)


def _over_rowwise(P: np.ndarray, cur_total: np.ndarray, k: np.ndarray) -> np.ndarray:
    final = cur_total[:, None] + RT[None, :]
    return (P * (final > k[:, None])).sum(axis=1)


def evaluate():
    d = training_frame()
    tr, te = d[d.season <= 2021], d[(d.season >= 2022) & (d.season <= 2025)].copy()
    m = _fit(tr)
    P = pmf(m, te)
    final = te.final_total.values
    mu = te.cur_total.values + P @ RT
    # Gaussian baseline: remaining points ~ Normal(pregame total * t/3600, s * sqrt(t/3600)), s fit on train
    s = float(np.sqrt(((tr.y_rem_tot - tr.exp_rem_tot) ** 2 / (tr.t / 3600)).mean()))
    g_mu = te.cur_total.values + te.exp_rem_tot.values
    g_sd = s * np.sqrt(np.maximum(te.t.values / 3600, 0.02))
    ll = lambda p, y: float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
    print(f"held out 2022-2025: {len(te)} plays; final-total MAE dist {np.abs(final - mu).mean():.3f}, gaussian {np.abs(final - g_mu).mean():.3f}")
    print(f"{'offset':>7}{'plays':>8}{'gauss logloss':>15}{'dist logloss':>14}")
    tot = np.zeros(2); n = 0
    for off in (-10.5, -6.5, -3.5, -0.5, 0.5, 3.5, 6.5, 10.5):  # lines around the gaussian mean, like a live market
        k = np.floor(g_mu) + off
        y = (final > k).astype(float)
        a, b = ll(norm.sf((k - g_mu) / g_sd), y), ll(_over_rowwise(P, te.cur_total.values, k), y)
        tot += np.array([a, b]) * len(y); n += len(y)
        print(f"{off:>7}{len(y):>8}{a:>15.4f}{b:>14.4f}")
    print(f"{'all':>7}{n:>8}{tot[0]/n:>15.4f}{tot[1]/n:>14.4f}")


def score_states_totals(d: pd.DataFrame, model=None) -> np.ndarray:
    """Remaining-points distribution for canonical game-state rows (same columns as
    model.score_states, plus optional plays_so_far = snaps with a down 1-4 so far).
    Returns (n, 76) over RT."""
    from .model import half_seconds_left, seconds_left
    model = model if model is not None else get_totals()
    d = d.copy()
    d.loc[~d.stable, ["down", "distance", "ydl"]] = np.nan

    def build(pos_home):
        X = pd.DataFrame({
            "t": seconds_left(d.period.fillna(0), d.clock_sec.fillna(0)),
            "half_t": half_seconds_left(d.period.fillna(0), d.clock_sec.fillna(0)), "qtr": d.period,
            "cur_total": d.home_score + d.away_score, "lead": d.lead, "spread": d.spread_prior,
            "total": d.total.fillna(45.0), "pos_home": pos_home, "down": d.down, "ydstogo": d.distance,
            "yardline_100": d.ydl,
            "pos_to": np.where(pos_home == 1, d.home_timeouts, d.away_timeouts),
            "def_to": np.where(pos_home == 1, d.away_timeouts, d.home_timeouts),
            "plays_so_far": d["plays_so_far"] if "plays_so_far" in d else np.nan,
            **{c: d[c] for c in WX if c in d},
        }).astype(float)
        X[["pos_to", "def_to"]] = X[["pos_to", "def_to"]].fillna(3.0)
        return curve_features(base_features(X), model["curve"])

    P = pmf(model, build(d.pos_home))
    if (~d.stable).any():
        P = np.where(d.stable.values[:, None], P, (P + pmf(model, build(1 - d.pos_home))) / 2)
    return P


def fit_and_save():
    joblib.dump(_fit(training_frame()), TOT_PATH)
    print("saved", TOT_PATH)


_cache: dict = {}


def get_totals():
    if "m" not in _cache:
        m = joblib.load(TOT_PATH) if TOT_PATH.exists() else None
        if not isinstance(m, dict) or m.get("version") != VERSION:
            fit_and_save()
            m = joblib.load(TOT_PATH)
        _cache["m"] = m
    return _cache["m"]


if __name__ == "__main__":
    if "--fit" not in sys.argv:
        evaluate()
    if "--eval" not in sys.argv:
        fit_and_save()
