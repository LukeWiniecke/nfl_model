"""Expected points (EP) for the team with the ball: the average value of the
current drive situation (down, distance, field position, time left in half).

We distill nflfastR's `ep` column into a small LightGBM that needs only inputs
available from the live feed, so training and live scoring use the same number.

    .venv/bin/python -m live.ep      # (re)fit and save
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
EP_PATH = ROOT / "data/live/ep.joblib"
EP_FEATS = ["down", "ydstogo", "yardline_100", "half_t", "pos_to", "def_to"]
_cache: dict = {}


def fit_and_save():
    d = pl.read_parquet(ROOT / "data/raw/nflverse/pbp_states.parquet").filter(
        pl.col("ep").is_not_null() & pl.col("posteam").is_not_null() & pl.col("yardline_100").is_not_null()
    ).select(
        "down", "ydstogo", "yardline_100", "ep", "season",
        half_t=pl.col("half_seconds_remaining"),
        pos_to=pl.col("posteam_timeouts_remaining"), def_to=pl.col("defteam_timeouts_remaining"),
    ).to_pandas()
    tr, te = d[d.season < 2024], d[d.season >= 2024]
    kw = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, verbose=-1)
    chk = lgb.LGBMRegressor(**kw).fit(tr[EP_FEATS], tr.ep)
    rmse = float(np.sqrt(((chk.predict(te[EP_FEATS]) - te.ep) ** 2).mean()))
    m = lgb.LGBMRegressor(**kw).fit(d[EP_FEATS], d.ep)
    joblib.dump(m, EP_PATH)
    print(f"saved {EP_PATH}  held-out RMSE vs nflfastR ep: {rmse:.3f} points (ep sd {te.ep.std():.2f})")
    return m


def get_ep_model():
    if "m" not in _cache:
        _cache["m"] = joblib.load(EP_PATH) if EP_PATH.exists() else fit_and_save()
    return _cache["m"]


def add_ep(d: pd.DataFrame) -> pd.DataFrame:
    """Needs down, ydstogo, yardline_100, half_t, pos_to, def_to, pos_home, lead.
    Adds ep_home (EP signed toward the home team) and eff_lead = lead + ep_home."""
    d = d.copy()
    ep = get_ep_model().predict(d[EP_FEATS].astype(float))
    d["ep_home"] = np.where(d.pos_home == 1, ep, -ep)
    d["eff_lead"] = d.lead + d.ep_home
    return d


if __name__ == "__main__":
    fit_and_save()
