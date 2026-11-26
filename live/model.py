"""Final live win-probability model: fit on all seasons, saved to
data/live/model.joblib, and applied to recorded snapshots.

    .venv/bin/python -m live.model        # (re)fit and save

Blend (see live/baseline.py for the walk-forward evaluation): LightGBM
remaining-margin model early, direct win classifier over the last 15 minutes.
"""

import re
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

from .baseline import CF, FEATS, clf_params, load, reg_params
from .ep import add_ep

MODEL_PATH = Path(__file__).resolve().parents[1] / "data/live/model.joblib"


def add_features(d: pd.DataFrame, sig: float) -> pd.DataFrame:
    d = d.copy()
    d["exp_rem_brown"] = d.spread * d.t / 3600
    d["pos_lead"] = np.where(d.pos_home == 1, d.lead, -d.lead)
    d["z"] = (d.lead + d.exp_rem_brown) / (sig * np.sqrt(np.maximum(d.t, 1) / 3600))
    d["home_to"] = np.where(d.pos_home == 1, d.pos_to, d.def_to)
    d["away_to"] = np.where(d.pos_home == 1, d.def_to, d.pos_to)
    d["pos_spread"] = np.where(d.pos_home == 1, d.spread, -d.spread)
    return d


def fit_win_models(d: pd.DataFrame) -> dict:
    """Fit the win-probability blend on a training frame from baseline.load()
    (+ add_ep). Shared by the live model and the leakage-free backtest model."""
    d = d.copy()
    d["exp_rem_brown"] = d.spread * d.t / 3600
    sig = float(np.sqrt(((d.y_rem - d.exp_rem_brown) ** 2 / (d.t / 3600)).mean()))
    d = add_features(d, sig)
    reg = lgb.LGBMRegressor(**reg_params(FEATS)).fit(d[FEATS], d.y_rem)
    clf = lgb.LGBMClassifier(**clf_params(CF)).fit(d[CF], d.win)
    r = d.y_rem - reg.predict(d[FEATS])
    bsig = float((r**2 / np.maximum(d.t / 3600, 0.02)).mean() ** 0.5)
    return dict(reg=reg, clf=clf, sig=sig, bsig=bsig)


def fit_and_save() -> dict:
    m = fit_win_models(add_ep(load().to_pandas()))
    sig, bsig = m["sig"], m["bsig"]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, MODEL_PATH)
    print(f"saved {MODEL_PATH}  brownian sigma={sig:.2f} learned sigma={bsig:.2f}")
    return m


_cache: dict = {}


def get_model() -> dict:
    if "m" not in _cache:
        _cache["m"] = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else fit_and_save()
    return _cache["m"]


# ---- snapshot -> model inputs ------------------------------------------------

def seconds_left(period, clock_sec):
    p = np.asarray(period, dtype=float)
    c = np.asarray(clock_sec, dtype=float)
    return np.where(p >= 5, c, np.maximum(4 - p, 0) * 900 + c)


def half_seconds_left(period, clock_sec):
    p = np.asarray(period, dtype=float)
    c = np.asarray(clock_sec, dtype=float)
    return np.where(p >= 5, c, np.where(p <= 2, (2 - p) * 900 + c, (4 - p) * 900 + c))


def yardline_100(text, possession) -> float:
    """'NYG 25' with NYG in possession -> 75 yards to go; opponent's 25 -> 25."""
    m = re.search(r"([A-Z]{2,3})\s+(\d{1,2})\s*$", str(text or ""))
    if not m or not isinstance(possession, str):
        return np.nan
    n = int(m.group(2))
    return 100 - n if m.group(1) == possession else n


def spread_line_from_details(details, spread, home) -> float:
    """nflverse convention: >0 means home favored. ESPN 'details' is 'DET -2.5'
    (favorite and size); the `spread` field is signed toward the home team."""
    if isinstance(details, str):
        if details.strip().upper() in ("EVEN", "PK", "PICK"):
            return 0.0
        m = re.match(r"([A-Z]{2,3})\s+-(\d+(?:\.\d+)?)", details.strip())
        if m:
            size = float(m.group(2))
            return size if m.group(1) == home else -size
    return -float(spread) if spread is not None and not pd.isna(spread) else np.nan


def score_states(d: pd.DataFrame, m: dict | None = None, dist=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core scorer on canonical game-state rows. Returns (p_home_win, mu, pmf):
    mu = predicted REMAINING home margin, pmf = distribution over live.dist.REM.

    Needs columns: period, clock_sec (seconds left in the period), lead (home
    minus away), pos_home (0/1), down, distance, ydl (yards to the end zone for
    the team with the ball), home_timeouts, away_timeouts, spread_prior (PREGAME
    line, home-positive), total, stable (bool: possession/down/field position
    are trustworthy). A 2nd-half kickoff is down=NaN, distance=0, ydl=35,
    stable=True, matching how nflfastR stores kickoffs in the training data.

    m/dist default to the saved live models; the backtest passes models fit
    only on games before the test window.
    """
    from .dist import get_dist, pmf
    m = m or get_model()
    if dist is None:
        dist = m["dist"] if m.get("dist") is not None else get_dist()
    d = d.copy()
    d["t"] = seconds_left(d.period.fillna(0), d.clock_sec.fillna(0))
    # Unstable rows (kickoffs/PATs/reviews in the live feed): blank the play
    # state and average over who has the ball instead of trusting it.
    d.loc[~d.stable, ["down", "distance", "ydl"]] = np.nan

    def build(pos_home: pd.Series) -> pd.DataFrame:
        X = pd.DataFrame({
            "t": d.t, "lead": d.lead, "spread": d.spread_prior, "total": d.total.fillna(45.0), "pos_home": pos_home,
            "down": d.down, "ydstogo": d.distance, "yardline_100": d.ydl,
            "pos_to": np.where(pos_home == 1, d.home_timeouts, d.away_timeouts),
            "def_to": np.where(pos_home == 1, d.away_timeouts, d.home_timeouts),
            "half_t": half_seconds_left(d.period.fillna(0), d.clock_sec.fillna(0)), "qtr": d.period,
        }).astype(float)
        X[["pos_to", "def_to"]] = X[["pos_to", "def_to"]].fillna(3.0)  # unknown -> full; see Live tab check
        return add_features(add_ep(X), m["sig"])

    def predict(X: pd.DataFrame):
        mu = m["reg"].predict(X[FEATS])
        p_reg = norm.cdf((X.lead + mu) / (m["bsig"] * np.sqrt(np.maximum(X.t, 1) / 3600).clip(lower=0.141)))
        w = np.clip((900 - X.t) / 600, 0, 1)
        return (1 - w) * p_reg + w * m["clf"].predict_proba(X[CF])[:, 1], mu

    X = build(d.pos_home)
    p, mu = predict(X)
    P = pmf(dist, X)
    if (~d.stable).any():
        X0 = build(1 - d.pos_home)
        p0, mu0 = predict(X0)
        p = np.where(d.stable, p, (p + p0) / 2)
        mu = np.where(d.stable, mu, (mu + mu0) / 2)
        P = np.where(d.stable.values[:, None], P, (P + pmf(dist, X0)) / 2)
    return np.asarray(p), np.asarray(mu), P


def score_snapshots(df: pd.DataFrame, prior: pd.Series | None = None) -> pd.DataFrame:
    """Add model columns to ESPN snapshot rows (live/poll.py format).

    prior: Series game_id -> pregame spread_line (home-positive). The prior must
    be the PREGAME line; the live line moves with the score and would leak it.
    Adds: t, spread_prior, p_model, fair_home_margin (expected final),
    p_line (win prob implied by the CURRENT posted spread), line_margin, gap.
    """
    from .dist import mean_final
    m = get_model()
    d = df.copy()
    d["t"] = seconds_left(d.period.fillna(0), d.clock_sec.fillna(0))
    d["lead"] = d.home_score - d.away_score
    d["pos_home"] = (d.possession == d.home).astype(int)
    d["ydl"] = [yardline_100(a, b) for a, b in zip(d.yard_line_text, d.possession)]
    d["line_spread"] = [spread_line_from_details(a, b, h) for a, b, h in zip(d.spread_details, d.spread, d.home)]
    d["spread_prior"] = d.game_id.map(prior) if prior is not None else d.line_spread
    d["spread_prior"] = d.spread_prior.fillna(d.line_spread).fillna(0.0)
    d["stable"] = d.possession.map(lambda v: isinstance(v, str)) & d.down.between(1, 4) & d.ydl.notna()
    p, mu, P = score_states(d, m)
    d.loc[~d.stable, ["down", "distance", "ydl"]] = np.nan
    d["pmf"] = list(P)  # distribution over remaining margin (live.dist.REM), for spread pricing
    d["dist_margin"] = mean_final(P, d.lead)
    final = d.state == "post"
    p = np.where(final, (d.lead > 0).astype(float) + 0.5 * (d.lead == 0), p)
    d["p_model"] = p
    d["fair_home_margin"] = d.lead + mu
    live_final = d.line_spread  # home-positive: expected final margin implied by the current line
    d["line_margin"] = live_final
    d["p_line"] = norm.cdf(live_final / (m["sig"] * np.sqrt(np.maximum(d.t, 1) / 3600)).clip(lower=0.5))
    d["gap"] = d.p_model - d.p_line
    return d


def load_snapshots(live_dir: Path) -> pd.DataFrame:
    files = sorted(live_dir.glob("snapshots_*.jsonl"))[-3:]  # last few days is plenty
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_json(f, lines=True, dtype={"game_id": str}) for f in files], ignore_index=True)


def pregame_prior(df: pd.DataFrame) -> pd.Series:
    """Per game: the last spread seen BEFORE kickoff (closing line); falls back
    to the earliest row if we only started recording after kickoff."""
    out = {}
    for gid, g in df.sort_values("ts").groupby("game_id"):
        pre = g[g.state == "pre"]
        row = pre.iloc[-1] if len(pre) else g.iloc[0]
        out[gid] = spread_line_from_details(row.spread_details, row.spread, row.home)
    return pd.Series(out)


if __name__ == "__main__":
    fit_and_save()
