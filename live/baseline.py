"""Live win-probability models on the play-state table, scored by game time.

 1. brownian: final home margin ~ Normal(lead + spread * t/3600, sigma * sqrt(t/3600))
 2. learned : LightGBM predicts the REMAINING margin (final - current) from the
              game state; sigma(t) is fit on training residuals; Normal -> win prob.
Walk-forward by season (train on seasons < S). Benchmarks: nflfastR wp / vegas_wp.

    .venv/bin/python -m live.baseline
"""

import numpy as np
import polars as pl
import lightgbm as lgb
from pathlib import Path
from scipy.stats import norm

PATH = Path(__file__).resolve().parents[1] / "data/raw/nflverse/pbp_states.parquet"
FEATS = ["t", "lead", "spread", "total", "pos_home", "down", "ydstogo", "yardline_100",
         "pos_to", "def_to", "half_t", "qtr", "exp_rem_brown", "ep_home", "eff_lead"]
CF = FEATS + ["pos_lead", "z", "home_to", "away_to", "pos_spread"]

# Shared LightGBM settings for the win-prob blend (live model, backtest and this eval).
# MIN_LEAF: a game is ~150 plays, so the old 100-200-play leaves could be a handful of
# games, and game-constant features (pregame spread/total) let trees memorise those
# games' outcomes (e.g. a 55.5 total moved one home team's expected margin by 4.6 pts).
# ~3000 plays = dozens of games per leaf. Held out 2022-25: Brier 0.1561 -> 0.1554.
LGB_BASE = dict(learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
MIN_LEAF = 3000
MONO_REG = {"spread": 1, "exp_rem_brown": 1, "ep_home": 1}  # remaining home margin rises with these
MONO_CLF = {"lead": 1, "spread": 1, "exp_rem_brown": 1, "ep_home": 1, "eff_lead": 1, "z": 1}


def reg_params(cols) -> dict:
    return dict(LGB_BASE, n_estimators=300, min_child_samples=MIN_LEAF, monotone_constraints=[MONO_REG.get(c, 0) for c in cols])


def clf_params(cols) -> dict:
    return dict(LGB_BASE, n_estimators=400, min_child_samples=MIN_LEAF, monotone_constraints=[MONO_CLF.get(c, 0) for c in cols])


# ---- v2 win-prob model (2026-11-26) ---------------------------------------------------------
# Early game: the remaining-margin regressor, now told who receives the 2nd-half kickoff (an extra
# possession). From kickoff to halftime it hands over linearly to a classifier from the offense's
# point of view with nflfastR-style time-scaled features. Held out 2022-25 (train <= 2021):
# Brier 0.1555 -> 0.1539 (nflfastR vegas_wp 0.1535); first half now level with vegas_wp.
REG_FEATS = FEATS + ["home_rec2h"]
POS_FEATS = ["pos_rec2h", "spread_time", "pos_home", "half_t", "t", "diff_time_ratio", "pos_lead", "down",
             "ydstogo", "yardline_100", "pos_to", "def_to"]
# a worse down / distance / field position or more opponent timeouts can only hurt the offense
MONO_POS = {"pos_rec2h": 1, "spread_time": 1, "diff_time_ratio": 1, "pos_lead": 1, "down": -1, "ydstogo": -1,
            "yardline_100": -1, "pos_to": 1, "def_to": -1}
POS_LEAF = 300   # no game-constant inputs here, so smaller leaves don't memorise games (held-out check)
HANDOFF = (3600, 1800)  # seconds left: regressor only at kickoff -> possession classifier only from halftime


def pos_params(cols) -> dict:
    return dict(LGB_BASE, n_estimators=600, min_child_samples=POS_LEAF, monotone_constraints=[MONO_POS.get(c, 0) for c in cols])


def handoff_weight(t):
    hi, lo = HANDOFF
    return np.clip((hi - np.asarray(t, float)) / (hi - lo), 0, 1)


def load() -> "pl.DataFrame":
    d = pl.read_parquet(PATH).filter(
        pl.col("spread_line").is_not_null() & pl.col("score_differential").is_not_null() & pl.col("result").is_not_null() & pl.col("posteam").is_not_null()
        & (pl.col("season_type") == "REG") & (pl.col("game_seconds_remaining") > 0))
    d = d.with_columns(
        t=pl.col("game_seconds_remaining"),
        pos_home=(pl.col("posteam") == pl.col("home_team")).cast(pl.Int8),
        # PRE-play home lead. total_home/away_score are POST-play (they already
        # include a FG/TD/XP scored on the play), which hid the value of scoring
        # position from the model; score_differential is pre-play, posteam view.
        lead=pl.when(pl.col("posteam") == pl.col("home_team")).then(pl.col("score_differential")).otherwise(-pl.col("score_differential")),
        spread=pl.col("spread_line"),
        total=pl.col("total_line").fill_null(45.0),
        half_t=pl.col("half_seconds_remaining"),
        pos_to=pl.col("posteam_timeouts_remaining"), def_to=pl.col("defteam_timeouts_remaining"),
        home_wp=pl.when(pl.col("posteam") == pl.col("home_team")).then(pl.col("wp")).otherwise(1 - pl.col("wp")),
        home_vwp=pl.when(pl.col("posteam") == pl.col("home_team")).then(pl.col("vegas_wp")).otherwise(1 - pl.col("vegas_wp")),
        win=(pl.col("result") > 0).cast(pl.Int8),
    ).with_columns(y_rem=pl.col("result") - pl.col("lead")).with_columns(exp_rem_brown=pl.col("spread") * pl.col("t") / 3600)
    # who receives the 2nd-half kickoff (the first 3rd-quarter kickoff's receiving team); known all game
    rec = (d.filter((pl.col("qtr") == 3) & (pl.col("play_type") == "kickoff")).sort("play_id")
            .group_by("game_id").agg(rec2h=pl.col("posteam").first()))
    d = d.join(rec, on="game_id", how="left").with_columns(
        home_rec2h=pl.when(pl.col("qtr") <= 2).then((pl.col("rec2h") == pl.col("home_team")).cast(pl.Float64)))
    return d.filter(pl.col("result") != 0)


def bucket(t):  # minutes remaining
    return np.select([t > 2700, t > 1800, t > 900, t > 300], ["Q1", "Q2", "Q3", "Q4 early"], "last 5 min")


def brier(p, y): return float(np.mean((p - y) ** 2))


def main():
    import pandas as pd
    from .ep import add_ep
    from .model import add_features
    d = add_ep(load().to_pandas())
    d["q"] = bucket(d["t"].values)
    out = []
    for season in sorted(d.season.unique())[-6:]:
        tr, te = d[d.season < season].copy(), d[d.season == season].copy()
        # 1. Brownian: sigma from training remaining-margin residuals
        res = tr.y_rem - tr.exp_rem_brown
        sig = float(np.sqrt((res**2 / (tr.t / 3600)).mean()))
        te["p_brown"] = norm.cdf((te.lead + te.exp_rem_brown) / (sig * np.sqrt(te.t / 3600)))
        tr, te = add_features(tr, sig), add_features(te, sig)
        # 2. learned remaining-margin model (v1 features / v2 features with the 2nd-half kickoff receiver)
        def learned(feats):
            m = lgb.LGBMRegressor(**reg_params(feats)).fit(tr[feats], tr.y_rem)
            r = tr.y_rem - m.predict(tr[feats])
            b = (r**2 / np.maximum(tr.t / 3600, 0.02)).mean() ** 0.5
            return norm.cdf((te.lead + m.predict(te[feats])) / (b * np.sqrt(np.maximum(te.t / 3600, 0.02))))
        te["p_learn"] = learned(FEATS)
        p_learn2 = learned(REG_FEATS)
        # 3. v1: direct win classifier, blended in over the last 15 min
        c = lgb.LGBMClassifier(**clf_params(CF)).fit(tr[CF], tr.win)
        te["p_clf"] = c.predict_proba(te[CF])[:, 1]
        w = np.clip((900 - te.t) / 600, 0, 1)
        te["p_blend"] = (1 - w) * te.p_learn + w * te.p_clf
        # 4. v2: possession classifier, handed over from kickoff to halftime
        pos = lgb.LGBMClassifier(**pos_params(POS_FEATS)).fit(tr[POS_FEATS], np.where(tr.pos_home == 1, tr.win, 1 - tr.win))
        pp = pos.predict_proba(te[POS_FEATS])[:, 1]
        w2 = handoff_weight(te.t)
        te["p_v2"] = (1 - w2) * p_learn2 + w2 * np.where(te.pos_home == 1, pp, 1 - pp)
        out.append(te)
    o = pd.concat(out)
    o = o.dropna(subset=["home_wp", "home_vwp"])
    cols = [("brownian", "p_brown"), ("learned", "p_learn"), ("v1 blend", "p_blend"), ("v2", "p_v2"),
            ("nfl wp", "home_wp"), ("vegas wp", "home_vwp")]
    order = ["Q1", "Q2", "Q3", "Q4 early", "last 5 min"]
    print(f"Brier (lower better), test seasons {int(o.season.min())}-{int(o.season.max())}, walk-forward by season")
    print(f"{'bucket':<11}{'plays':>8}" + "".join(f"{n:>10}" for n, _ in cols))
    for q in order:
        g = o[o.q == q]
        print(f"{q:<11}{len(g):>8}" + "".join(f"{brier(g[c], g.win):>10.4f}" for _, c in cols))
    print(f"{'ALL':<11}{len(o):>8}" + "".join(f"{brier(o[c], o.win):>10.4f}" for _, c in cols))


if __name__ == "__main__":
    main()
