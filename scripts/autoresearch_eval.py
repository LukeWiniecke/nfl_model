"""Fixed autoresearch metric harness -- DO NOT MODIFY during an autoresearch run.

Reads the ensemble ATS/ML backtest outputs main.py just wrote and prints/writes
a plain-text summary.txt at the repo root (tracked in git, unlike outputs/
which is gitignored) so results survive across autoresearch/<tag> branches.

Run after main.py:
    .venv\\Scripts\\python.exe main.py > run.log 2>&1
    .venv\\Scripts\\python.exe scripts\\autoresearch_eval.py >> run.log 2>&1

Extended for the autoresearch/sep19 run (frozen before that run started):
the first eleven lines are byte-for-byte the original ROI report, so every
Select-String pattern in program.md still works. After them it reports the
ensemble's ACCURACY and CALIBRATION against the market, per era, and also
writes summary.json (machine-readable, same numbers) for the run harness:

  MAE / RMSE            model vs market closing line, on every graded game
  Brier ML              model_home_prob vs de-vigged moneyline, ties excluded
  Brier ATS             model_cover_prob (and raw, pre-isotonic) vs cover
  ECE                   expected calibration error, 10 quantile bins
  calibration slope     logistic fit of outcome on logit(prob); 1.0 = calibrated,
                        <1 = overconfident, >1 = underconfident
  shrinkage slope       median walk-forward edge-shrinkage slope actually used
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from src import config, evaluation

MODEL = "ensemble"  # the flagship model this autoresearch loop optimizes


def _era_row(era_df: pl.DataFrame, era: str) -> dict:
    row = era_df.filter(pl.col("era") == era)
    if row.height == 0:
        return {"n_bets": 0, "roi": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    r = row.row(0, named=True)
    return {"n_bets": int(r["n_bets"]), "roi": float(r["roi"]), "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"])}


def _ece(p, y, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) < n_bins * 5:
        return float("nan")
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def _calib_slope(p, y) -> float:
    from sklearn.linear_model import LogisticRegression

    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = np.clip(p[ok], 1e-4, 1 - 1e-4), y[ok]
    if len(p) < 50 or len(np.unique(y)) < 2:
        return float("nan")
    x = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(x, y)
    return float(lr.coef_[0][0])


def _brier(p, y) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[ok] - y[ok]) ** 2)) if ok.any() else float("nan")


def _accuracy(ats: pl.DataFrame, ml: pl.DataFrame, era: str) -> dict:
    a = ats.filter(
        (pl.col("era") == era) & pl.col("home_margin").is_not_null() & pl.col("spread_line").is_not_null()
    ).to_pandas()
    m = ml.filter((pl.col("era") == era) & pl.col("home_margin").is_not_null()).to_pandas()

    out = {"n_games": int(len(a))}
    if len(a) == 0:
        return out

    model_err = a["predicted_margin"] - a["home_margin"]
    market_err = a["spread_line"] - a["home_margin"]
    out["mae_model"] = float(model_err.abs().mean())
    out["mae_market"] = float(market_err.abs().mean())
    out["rmse_model"] = float(np.sqrt((model_err**2).mean()))
    out["rmse_market"] = float(np.sqrt((market_err**2).mean()))
    out["mae_gap"] = out["mae_model"] - out["mae_market"]
    out["rmse_gap"] = out["rmse_model"] - out["rmse_market"]

    not_push = ~a["is_push"].to_numpy().astype(bool)
    cover = (a["home_margin"] > a["spread_line"]).to_numpy().astype(float)
    out["brier_ats_model"] = _brier(a["model_cover_prob"].to_numpy()[not_push], cover[not_push])
    out["brier_ats_raw"] = _brier(a["model_cover_prob_raw"].to_numpy()[not_push], cover[not_push])
    out["ece_ats"] = _ece(a["model_cover_prob"].to_numpy()[not_push], cover[not_push])
    out["slope_ats"] = _calib_slope(a["model_cover_prob"].to_numpy()[not_push], cover[not_push])

    shrink = a["ats_edge_shrinkage"].to_numpy(dtype=float) if "ats_edge_shrinkage" in a.columns else np.array([np.nan])
    thr = a["ats_threshold_used"].to_numpy(dtype=float) if "ats_threshold_used" in a.columns else np.array([np.nan])
    out["shrink_slope_median"] = float(np.nanmedian(shrink)) if np.isfinite(shrink).any() else float("nan")
    out["ats_threshold_median"] = float(np.nanmedian(thr)) if np.isfinite(thr).any() else float("nan")

    if len(m):
        has_ml = m["home_moneyline"].notna().to_numpy() & m["market_home_prob_no_vig"].notna().to_numpy()
        not_tie = (m["home_margin"] != 0).to_numpy()
        keep = has_ml & not_tie
        win = (m["home_margin"] > 0).to_numpy().astype(float)
        out["brier_ml_model"] = _brier(m["model_home_prob"].to_numpy()[keep], win[keep])
        out["brier_ml_raw"] = _brier(m["model_home_prob_raw"].to_numpy()[keep], win[keep])
        out["brier_ml_market"] = _brier(m["market_home_prob_no_vig"].to_numpy()[keep], win[keep])
        out["brier_ml_gap"] = out["brier_ml_model"] - out["brier_ml_market"]
        out["ece_ml"] = _ece(m["model_home_prob"].to_numpy()[keep], win[keep])
        out["slope_ml"] = _calib_slope(m["model_home_prob"].to_numpy()[keep], win[keep])
    return out


def _fmt(x, nd=4, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def main():
    try:
        ats_bets = pl.read_parquet(config.BACKTESTS_DIR / f"{MODEL}_ats_bets.parquet")
        ml_bets = pl.read_parquet(config.BACKTESTS_DIR / f"{MODEL}_ml_bets.parquet")
    except FileNotFoundError as e:
        print(f"CRASH: missing backtest output ({e})")
        sys.exit(1)

    ats_era = evaluation.summarize_by_era(ats_bets)
    ml_era = evaluation.summarize_by_era(ml_bets)

    ats_dev = _era_row(ats_era, "development")
    ats_hold = _era_row(ats_era, "holdout")
    ml_dev = _era_row(ml_era, "development")
    ml_hold = _era_row(ml_era, "holdout")

    lines = [
        "Model:                  " + MODEL,
        f"Dev ATS ROI:            {ats_dev['roi']:+.4f}",
        f"Dev ATS n_bets:         {ats_dev['n_bets']}",
        f"Dev ATS 95% CI:         [{ats_dev['ci_lo']:+.4f}, {ats_dev['ci_hi']:+.4f}]",
        f"Holdout ATS ROI:        {ats_hold['roi']:+.4f}",
        f"Holdout ATS n_bets:     {ats_hold['n_bets']}",
        f"Dev ML ROI:             {ml_dev['roi']:+.4f}",
        f"Dev ML n_bets:          {ml_dev['n_bets']}",
        f"Dev ML 95% CI:          [{ml_dev['ci_lo']:+.4f}, {ml_dev['ci_hi']:+.4f}]",
        f"Holdout ML ROI:         {ml_hold['roi']:+.4f}",
        f"Holdout ML n_bets:      {ml_hold['n_bets']}",
    ]

    acc = {}
    for era, tag in [("development", "Dev"), ("holdout", "Holdout")]:
        r = _accuracy(ats_bets, ml_bets, era)
        acc[era] = r
        if "mae_model" not in r:
            continue
        lines += [
            f"{tag} MAE model/market:   {_fmt(r['mae_model'])} / {_fmt(r['mae_market'])}  gap {_fmt(r['mae_gap'], sign=True)}",
            f"{tag} RMSE model/market:  {_fmt(r['rmse_model'])} / {_fmt(r['rmse_market'])}  gap {_fmt(r['rmse_gap'], sign=True)}",
            f"{tag} Brier ML model/market: {_fmt(r.get('brier_ml_model'))} / {_fmt(r.get('brier_ml_market'))}"
            f"  gap {_fmt(r.get('brier_ml_gap'), sign=True)}  (raw {_fmt(r.get('brier_ml_raw'))})",
            f"{tag} Brier ATS model:    {_fmt(r['brier_ats_model'])}  (raw {_fmt(r['brier_ats_raw'])})",
            f"{tag} ECE ATS / ML:       {_fmt(r['ece_ats'])} / {_fmt(r.get('ece_ml'))}",
            f"{tag} calib slope ATS/ML: {_fmt(r['slope_ats'], 3)} / {_fmt(r.get('slope_ml'), 3)}",
            f"{tag} shrink slope / thr: {_fmt(r['shrink_slope_median'], 3)} / {_fmt(r['ats_threshold_median'], 2)}",
        ]

    for line in lines:
        print(line)
    with open("summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    payload = {
        "model": MODEL,
        "ats": {"dev": ats_dev, "holdout": ats_hold},
        "ml": {"dev": ml_dev, "holdout": ml_hold},
        "acc": {"dev": acc.get("development", {}), "holdout": acc.get("holdout", {})},
    }

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    with open("summary.json", "w") as f:
        json.dump(_clean(payload), f, indent=2)


if __name__ == "__main__":
    main()
