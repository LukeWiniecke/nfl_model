"""Live TOTALS: full distribution of the points still to be scored.

A multiclass LightGBM predicts P(remaining total points = r) for r = 0..75 from
the game state (time, current total, pregame total and spread, score margin,
possession and field position). P(over k) is then a sum over that
distribution, which handles the lumpiness of scoring (3s and 7s) and the
slower pace of blowouts and late leads.

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
from .ep import get_ep_model, EP_FEATS

TOT_PATH = Path(__file__).resolve().parents[1] / "data/live/totals.joblib"
CLIP = 75
RT = np.arange(0, CLIP + 1)  # class i <-> remaining points RT[i]
TF = ["t", "half_t", "qtr", "cur_total", "lead", "abs_lead", "spread", "total", "pos_home", "down", "ydstogo",
      "yardline_100", "pos_to", "def_to", "ep_off", "exp_rem_tot", "pace"]
PARAMS = dict(objective="multiclass", num_class=len(RT), n_estimators=120, learning_rate=0.08, num_leaves=15,
              min_child_samples=MIN_LEAF, subsample=0.6, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def add_total_features(d: pd.DataFrame) -> pd.DataFrame:
    """Needs t, half_t, qtr, cur_total (PRE-play), lead, spread, total (pregame line),
    pos_home, down, ydstogo, yardline_100, pos_to, def_to."""
    d = d.copy()
    d["abs_lead"] = d.lead.abs()
    ep_in = d[EP_FEATS].astype(float)
    d["ep_off"] = get_ep_model().predict(ep_in)  # value of the current drive to the offense
    d["exp_rem_tot"] = d.total * d.t / 3600
    d["pace"] = d.cur_total - d.total * (3600 - d.t) / 3600  # points above/below the pregame pace
    return d


def training_frame() -> pd.DataFrame:
    """baseline.load() rows plus the PRE-play total and the final total.
    (load() overwrites nflverse `total`, the final total, with the pregame line.)"""
    raw = (pl.read_parquet(PATH).sort(["game_id", "play_id"])
           .with_columns(post=pl.col("total_home_score") + pl.col("total_away_score"))
           .with_columns(cur_total=pl.col("post").shift(1).over("game_id").fill_null(0))
           .select("game_id", "play_id", "cur_total", final_total=pl.col("total")))
    d = load().join(raw, on=["game_id", "play_id"], how="left").to_pandas()
    d = d[d.final_total.notna() & d.cur_total.notna()]
    d["y_rem_tot"] = d.final_total - d.cur_total
    d = d[d.y_rem_tot >= 0]
    return add_total_features(d)


def _fit(d: pd.DataFrame) -> lgb.LGBMClassifier:
    y = np.clip(np.rint(d.y_rem_tot), 0, CLIP).astype(int)
    assert len(np.unique(y)) == len(RT), "a remaining-points class is missing from training data"
    return lgb.LGBMClassifier(**PARAMS).fit(d[TF], y)


def pmf(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[TF])


def prob_over(P: np.ndarray, cur_total, k: float, push: float = 0.0) -> np.ndarray:
    """P(final total > k) (+ push * P(final total == k))."""
    final = np.asarray(cur_total, dtype=float)[:, None] + RT[None, :]
    return (P * ((final > k) + push * (final == k))).sum(axis=1)


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
    model.score_states). Returns (n, 76) over RT."""
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
        }).astype(float)
        X[["pos_to", "def_to"]] = X[["pos_to", "def_to"]].fillna(3.0)
        return add_total_features(X)

    P = pmf(model, build(d.pos_home))
    if (~d.stable).any():
        P = np.where(d.stable.values[:, None], P, (P + pmf(model, build(1 - d.pos_home))) / 2)
    return P


def _over_rowwise(P: np.ndarray, cur_total: np.ndarray, k: np.ndarray) -> np.ndarray:
    final = cur_total[:, None] + RT[None, :]
    return (P * (final > k[:, None])).sum(axis=1)


def fit_and_save():
    joblib.dump(_fit(training_frame()), TOT_PATH)
    print("saved", TOT_PATH)


_cache: dict = {}


def get_totals():
    if "m" not in _cache:
        if not TOT_PATH.exists():
            fit_and_save()
        _cache["m"] = joblib.load(TOT_PATH)
    return _cache["m"]


if __name__ == "__main__":
    if "--fit" not in sys.argv:
        evaluate()
    if "--eval" not in sys.argv:
        fit_and_save()
