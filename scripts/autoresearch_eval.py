"""Fixed autoresearch metric harness -- DO NOT MODIFY during an autoresearch run.

Reads the ensemble ATS/ML backtest outputs main.py just wrote and prints/writes
a plain-text summary.txt at the repo root (tracked in git, unlike outputs/
which is gitignored) so results survive across autoresearch/<tag> branches.

Run after main.py:
    .venv\\Scripts\\python.exe main.py > run.log 2>&1
    .venv\\Scripts\\python.exe scripts\\autoresearch_eval.py >> run.log 2>&1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from src import config, evaluation

MODEL = "ensemble"  # the flagship model this autoresearch loop optimizes


def _era_row(era_df: pl.DataFrame, era: str) -> dict:
    row = era_df.filter(pl.col("era") == era)
    if row.height == 0:
        return {"n_bets": 0, "roi": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    r = row.row(0, named=True)
    return {"n_bets": r["n_bets"], "roi": r["roi"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]}


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

    print("Model:                  " + MODEL)
    print(f"Dev ATS ROI:            {ats_dev['roi']:+.4f}")
    print(f"Dev ATS n_bets:         {ats_dev['n_bets']}")
    print(f"Dev ATS 95% CI:         [{ats_dev['ci_lo']:+.4f}, {ats_dev['ci_hi']:+.4f}]")
    print(f"Holdout ATS ROI:        {ats_hold['roi']:+.4f}")
    print(f"Holdout ATS n_bets:     {ats_hold['n_bets']}")
    print(f"Dev ML ROI:             {ml_dev['roi']:+.4f}")
    print(f"Dev ML n_bets:          {ml_dev['n_bets']}")
    print(f"Dev ML 95% CI:          [{ml_dev['ci_lo']:+.4f}, {ml_dev['ci_hi']:+.4f}]")
    print(f"Holdout ML ROI:         {ml_hold['roi']:+.4f}")
    print(f"Holdout ML n_bets:      {ml_hold['n_bets']}")

    with open("summary.txt", "w") as f:
        f.write(f"Model:                  {MODEL}\n")
        f.write(f"Dev ATS ROI:            {ats_dev['roi']:+.4f}\n")
        f.write(f"Dev ATS n_bets:         {ats_dev['n_bets']}\n")
        f.write(f"Dev ATS 95% CI:         [{ats_dev['ci_lo']:+.4f}, {ats_dev['ci_hi']:+.4f}]\n")
        f.write(f"Holdout ATS ROI:        {ats_hold['roi']:+.4f}\n")
        f.write(f"Holdout ATS n_bets:     {ats_hold['n_bets']}\n")
        f.write(f"Dev ML ROI:             {ml_dev['roi']:+.4f}\n")
        f.write(f"Dev ML n_bets:          {ml_dev['n_bets']}\n")
        f.write(f"Dev ML 95% CI:          [{ml_dev['ci_lo']:+.4f}, {ml_dev['ci_hi']:+.4f}]\n")
        f.write(f"Holdout ML ROI:         {ml_hold['roi']:+.4f}\n")
        f.write(f"Holdout ML n_bets:      {ml_hold['n_bets']}\n")


if __name__ == "__main__":
    main()
