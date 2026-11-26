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
    return d.filter(pl.col("result") != 0)


def bucket(t):  # minutes remaining
    return np.select([t > 2700, t > 1800, t > 900, t > 300], ["Q1", "Q2", "Q3", "Q4 early"], "last 5 min")


def brier(p, y): return float(np.mean((p - y) ** 2))


def main():
    from .ep import add_ep
    d = add_ep(load().to_pandas())
    d["q"] = bucket(d["t"].values)
    out = []
    for season in sorted(d.season.unique())[-6:]:
        tr, te = d[d.season < season], d[d.season == season].copy()
        # 1. Brownian: sigma from training remaining-margin residuals
        res = tr.y_rem - tr.exp_rem_brown
        sig = float(np.sqrt((res**2 / (tr.t / 3600)).mean()))
        te["p_brown"] = norm.cdf((te.lead + te.exp_rem_brown) / (sig * np.sqrt(te.t / 3600)))
        # 2. learned remaining-margin model
        m = lgb.LGBMRegressor(**reg_params(FEATS))
        m.fit(tr[FEATS], tr.y_rem)
        r = tr.y_rem - m.predict(tr[FEATS])
        tr_b = (r**2 / np.maximum(tr.t / 3600, 0.02)).mean() ** 0.5
        mu = m.predict(te[FEATS])
        te["p_learn"] = norm.cdf((te.lead + mu) / (tr_b * np.sqrt(np.maximum(te.t / 3600, 0.02))))
        # 3. direct win classifier: learns clock/possession/timeout effects the
        #    Gaussian can't (kneel-downs, two-score games, hurry-up)
        for df in (tr, te):
            df["pos_lead"] = np.where(df.pos_home == 1, df.lead, -df.lead)
            df["z"] = (df.lead + df.exp_rem_brown) / (sig * np.sqrt(df.t / 3600))
            df["home_to"] = np.where(df.pos_home == 1, df.pos_to, df.def_to)
            df["away_to"] = np.where(df.pos_home == 1, df.def_to, df.pos_to)
            df["pos_spread"] = np.where(df.pos_home == 1, df.spread, -df.spread)
        c = lgb.LGBMClassifier(**clf_params(CF))
        c.fit(tr[CF], tr.win)
        te["p_clf"] = c.predict_proba(te[CF])[:, 1]
        w = np.clip((900 - te.t) / 600, 0, 1)  # learned-margin early, classifier in the last 15 min
        te["p_blend"] = (1 - w) * te.p_learn + w * te.p_clf
        out.append(te)
    o = np.concatenate([x.assign(_=0).values for x in out]) if False else __import__("pandas").concat(out)
    o = o.dropna(subset=["home_wp", "home_vwp"])
    rows = []
    for q, g in o.groupby("q"):
        rows.append((q, len(g), brier(g.p_brown, g.win), brier(g.p_learn, g.win), brier(g.p_clf, g.win), brier(g.p_blend, g.win), brier(g.home_wp, g.win), brier(g.home_vwp, g.win)))
    rows.sort(key=lambda r: ["Q1", "Q2", "Q3", "Q4 early", "last 5 min"].index(r[0]))
    print(f"Brier (lower better), test seasons {int(o.season.min())}-{int(o.season.max())}")
    print(f"{'bucket':<11}{'plays':>8}{'brownian':>10}{'learned':>9}{'clf':>8}{'blend':>8}{'nfl wp':>8}{'vegas wp':>9}")
    for r in rows: print(f"{r[0]:<11}{r[1]:>8}" + "".join(f"{x:>10.4f}" if i == 0 else f"{x:>9.4f}" for i, x in enumerate(r[2:])))
    print("ALL        ", f"{len(o):>8}", *[f"{brier(o[c], o.win):.4f}" for c in ("p_brown", "p_learn", "p_clf", "p_blend", "home_wp", "home_vwp")])


if __name__ == "__main__":
    main()
