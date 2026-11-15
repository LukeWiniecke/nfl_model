"""Full distribution of the final margin, for live SPREAD pricing.

A bell curve around the expected margin gets win probability roughly right but
misprices spreads: real outcomes are wider late in two-score games, skewed by
game script, and lumpy (the next score is 3 or 7, so finals cluster on key
numbers). Here a multiclass LightGBM predicts the probability of every integer
REMAINING margin (final minus current home lead, clipped to +/-28) from the game
state; any cover or win probability is then a sum over that distribution.

    .venv/bin/python -m live.dist            # held-out evaluation, then fit all + save
    .venv/bin/python -m live.dist --eval     # evaluation only
"""

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

from .baseline import FEATS, load
from .ep import add_ep

DIST_PATH = Path(__file__).resolve().parents[1] / "data/live/dist.joblib"
CLIP = 28
REM = np.arange(-CLIP, CLIP + 1)  # class i <-> remaining margin REM[i]
DF = FEATS + ["pos_lead", "z", "home_to", "away_to", "pos_spread"]
PARAMS = dict(objective="multiclass", num_class=len(REM), n_estimators=120, learning_rate=0.08, num_leaves=15,
              min_child_samples=300, subsample=0.6, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def _fit(d: pd.DataFrame) -> lgb.LGBMClassifier:
    y = np.clip(np.rint(d.y_rem), -CLIP, CLIP).astype(int) + CLIP
    # every class must appear so predict_proba columns line up with REM
    assert len(np.unique(y)) == len(REM), "a remaining-margin class is missing from training data"
    return lgb.LGBMClassifier(**PARAMS).fit(d[DF], y)


def pmf(model, X: pd.DataFrame) -> np.ndarray:
    """(n, 57) probabilities over REM for each row."""
    return model.predict_proba(X[DF])


def prob_final_gt(p: np.ndarray, lead, k: float, push: float = 0.5) -> np.ndarray:
    """P(final home margin > k), counting a push (final == k) as `push`.
    k is the nflverse-style line: home covers a spread_line of k when final > k."""
    final = np.asarray(lead, dtype=float)[:, None] + REM[None, :]
    return (p * ((final > k) + push * (final == k))).sum(axis=1)


def mean_final(p: np.ndarray, lead) -> np.ndarray:
    return np.asarray(lead, dtype=float) + p @ REM


def evaluate():
    from .model import add_features, CF
    d = add_ep(load().to_pandas())
    d["exp_rem_brown"] = d.spread * d.t / 3600
    tr, te = d[d.season < 2024], d[d.season >= 2024].copy()
    sig = float(np.sqrt(((tr.y_rem - tr.exp_rem_brown) ** 2 / (tr.t / 3600)).mean()))
    tr, te = add_features(tr, sig), add_features(te, sig)
    final = (te.lead + te.y_rem).values

    # Gaussian baseline (the current model's shape): remaining-margin regressor + sigma*sqrt(t)
    kw = dict(learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    reg = lgb.LGBMRegressor(n_estimators=300, min_child_samples=200, **kw).fit(tr[FEATS], tr.y_rem)
    r = tr.y_rem - reg.predict(tr[FEATS])
    bsig = float((r**2 / np.maximum(tr.t / 3600, 0.02)).mean() ** 0.5)
    mu = te.lead.values + reg.predict(te[FEATS])
    sd = bsig * np.sqrt(np.maximum(te.t.values / 3600, 0.02))
    clf = lgb.LGBMClassifier(n_estimators=400, min_child_samples=100, **kw).fit(tr[CF], tr.win)
    w = np.clip((900 - te.t.values) / 600, 0, 1)
    p_blend = (1 - w) * norm.cdf(mu / sd) + w * clf.predict_proba(te[CF])[:, 1]

    m = _fit(tr)
    P = pmf(m, te)
    brier = lambda p, y: float(np.mean((p - y) ** 2))
    print(f"held out 2024-2026, {len(te)} plays")
    print(f"expected final margin MAE: gaussian {np.abs(final - mu).mean():.3f}  dist {np.abs(final - mean_final(P, te.lead)).mean():.3f}")
    win = (final > 0).astype(float)
    print(f"win prob Brier: blend {brier(p_blend, win):.4f}  dist {brier(prob_final_gt(P, te.lead, 0), win):.4f}  "
          f"vegas_wp {float(np.nanmean((np.where(te.pos_home == 1, te.vegas_wp, 1 - te.vegas_wp) - win) ** 2)):.4f}")

    print("\ncover probability, P(final home margin > line), on plays where the line is live (within 7 of expected margin)")
    print(f"{'line':>6}{'plays':>8}{'gauss Brier':>13}{'dist Brier':>12}{'gauss logloss':>15}{'dist logloss':>14}")
    tot = np.zeros(4); n = 0
    ll = lambda p, y: float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
    for k in (-10.5, -7.5, -6.5, -3.5, -2.5, 2.5, 3.5, 6.5, 7.5, 10.5):
        msk = np.abs(mu - k) <= 7
        y = (final[msk] > k).astype(float)
        pg, pd_ = norm.sf((k - mu[msk]) / sd[msk]), prob_final_gt(P[msk], te.lead.values[msk], k)
        row = (brier(pg, y), brier(pd_, y), ll(pg, y), ll(pd_, y))
        tot += np.array(row) * msk.sum(); n += msk.sum()
        print(f"{k:>6}{msk.sum():>8}{row[0]:>13.4f}{row[1]:>12.4f}{row[2]:>15.4f}{row[3]:>14.4f}")
    print(f"{'all':>6}{n:>8}{tot[0]/n:>13.4f}{tot[1]/n:>12.4f}{tot[2]/n:>15.4f}{tot[3]/n:>14.4f}")

    print("\nkey numbers: P(final margin lands exactly on +/-N), Q4 plays only")
    q4 = te.t.values <= 900
    fq, Pq, lq = final[q4], P[q4], te.lead.values[q4]
    fin = lq[:, None] + REM[None, :]
    for N in (3, 7, 10, 1, 5):
        pred = (Pq * (np.abs(fin) == N)).sum(axis=1).mean()
        g = (norm.cdf((N + .5 - mu[q4]) / sd[q4]) - norm.cdf((N - .5 - mu[q4]) / sd[q4])
             + norm.cdf((-N + .5 - mu[q4]) / sd[q4]) - norm.cdf((-N - .5 - mu[q4]) / sd[q4])).mean()
        print(f"  |margin| = {N:>2}: actual {np.mean(np.abs(fq) == N):.3f}  dist {pred:.3f}  gaussian {g:.3f}")


def fit_and_save():
    from .model import add_features, get_model
    d = add_features(add_ep(load().to_pandas()), get_model()["sig"])
    joblib.dump(_fit(d), DIST_PATH)
    print("saved", DIST_PATH)


def ladder(p_row: np.ndarray, lead: float, home: str, away: str, n: int = 4) -> pd.DataFrame:
    """Fair prices for half-point home spreads around the model's 50/50 line."""
    from src.utils import prob_to_american
    p = np.asarray(p_row, dtype=float)[None, :]
    cand = np.arange(-40, 40) + 0.5  # centre on the line closest to 50/50 (the median, not the mean: the distribution is skewed)
    center = cand[np.argmin([abs(float(prob_final_gt(p, [lead], k)[0]) - 0.5) for k in cand])]
    rows = []
    for k in center + np.arange(-n, n + 1):
        ph = float(prob_final_gt(p, [lead], k)[0])
        ph = min(max(ph, 0.001), 0.999)
        name = lambda team, pts: f"{team} {-pts:+.1f}"
        rows.append({"Home side": name(home, k), "P(cover)": ph, "Fair odds": int(prob_to_american(np.array([ph]))[0]),
                     "Away side": name(away, -k), "P(cover) ": 1 - ph, "Fair odds ": int(prob_to_american(np.array([1 - ph]))[0])})
    return pd.DataFrame(rows)


_cache: dict = {}


def get_dist():
    if "m" not in _cache:
        if not DIST_PATH.exists():
            fit_and_save()
        _cache["m"] = joblib.load(DIST_PATH)
    return _cache["m"]


if __name__ == "__main__":
    evaluate()
    if "--eval" not in sys.argv:
        fit_and_save()
