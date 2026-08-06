"""Predictions for games that haven't been played yet -- the "what does the
model think about next week" view. Everything else in this project backtests
completed history; this is the live counterpart, built for one purpose: to
be one input into a real betting decision, alongside your own judgment, not
a signal to follow blindly. It is explicitly NOT walk-forward validated --
there's no result yet to validate a future game against -- so it inherits
whatever skill (or lack of it) the Diagnostics tab already measured on
completed games, nothing more.

Probability recalibration follows the exact same per-model rule established
in backtesting.build_ml_bets: kept for ridge and the ensemble's moneyline
market, skipped for LightGBM's (see main.py / README for the measured
reason -- isotonic regression overfits LightGBM's raw probability
distribution in a way that manufactures spurious underdog edges).
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import polars as pl
from sklearn.isotonic import IsotonicRegression

from scipy.stats import norm

from . import config
from .probabilities import (
    ats_cover_probability_t,
    breakeven_probability,
    fit_dof_all_history,
    fit_edge_shrinkage_all_history,
    home_win_probability_t,
    market_moneyline_no_vig,
)
from .utils import get_logger

logger = get_logger(__name__)


def _live_ats_threshold(hist_ats_bets: pl.DataFrame, live: pd.DataFrame) -> np.ndarray:
    """Live analogue of backtesting.adaptive_ats_threshold: the same
    shrinkage-and-vig derivation, but with one slope fit on ALL graded
    history rather than season-by-season (there is only one batch of upcoming
    games to apply it to). Without this the live board was selecting picks at
    the old fixed 1.5-point bar while the backtest that justified it now uses
    a derived one nearly 3x higher -- i.e. the thing being bet would not have
    been the thing that was validated."""
    hist = hist_ats_bets.filter(
        pl.col("home_margin").is_not_null() & pl.col("spread_line").is_not_null()
    ).to_pandas()
    slope, sd = fit_edge_shrinkage_all_history(
        (hist["predicted_margin"] - hist["spread_line"]).to_numpy(),
        (hist["home_margin"] - hist["spread_line"]).to_numpy(),
    )
    if not np.isfinite(slope) or not np.isfinite(sd):
        return np.full(len(live), config.ATS_EDGE_THRESHOLD_POINTS)

    home_odds = live["home_spread_odds"].fillna(config.DEFAULT_SPREAD_ODDS).to_numpy()
    away_odds = live["away_spread_odds"].fillna(config.DEFAULT_SPREAD_ODDS).to_numpy()
    p_breakeven = np.maximum(breakeven_probability(home_odds), breakeven_probability(away_odds))
    return norm.ppf(p_breakeven) * sd / slope


def _calibrate_live(hist_bets: pl.DataFrame, raw_col: str, outcome_expr: pl.Expr, live_raw: np.ndarray) -> np.ndarray:
    """Fit one isotonic calibrator on ALL historical (raw prob, outcome)
    pairs -- no season-by-season split needed, since there's only one
    "current" bucket of upcoming games to apply it to -- and use it on the
    live raw probabilities. Falls back to the raw probability, unchanged, if
    there isn't enough historical data yet (config.MIN_CALIBRATION_GAMES)."""
    hist = hist_bets.filter(pl.col(raw_col).is_not_null()).with_columns(outcome_expr.alias("_outcome"))
    hist = hist.filter(pl.col("_outcome").is_not_null()).to_pandas()
    if len(hist) < config.MIN_CALIBRATION_GAMES:
        return live_raw
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(hist[raw_col].to_numpy(dtype=float), hist["_outcome"].to_numpy(dtype=float))
    return iso.predict(live_raw)


def build_upcoming_board(
    name: str, live: pd.DataFrame, hist_ats_bets: pl.DataFrame, hist_ml_bets: pl.DataFrame, calibrate_ml: bool
) -> pd.DataFrame:
    """`live` is one model's predict_upcoming_* output: game_id/market
    columns plus predicted_margin and predicted_std. Adds probabilities and
    edges only -- bet-side and Kelly stake are computed live in the
    dashboard from the sidebar's threshold/Kelly sliders, same as every
    other tab, so dragging those sliders updates this view too."""
    df = live.copy()

    # Student's-t instead of Normal, dof fit once on all history (see
    # probabilities.fit_t_dof_walk_forward's docstring for why: raw
    # probabilities were measurably overconfident, worst for LightGBM).
    dof = fit_dof_all_history(
        hist_ats_bets["home_margin"].to_numpy(), hist_ats_bets["predicted_margin"].to_numpy(),
        hist_ats_bets["predicted_std"].to_numpy(),
    )
    df["model_cover_prob_raw"] = ats_cover_probability_t(
        df["predicted_margin"].to_numpy(), df["spread_line"].to_numpy(), df["predicted_std"].to_numpy(), dof
    )
    df["model_cover_prob"] = _calibrate_live(
        hist_ats_bets, "model_cover_prob_raw", pl.col("home_margin") > pl.col("spread_line"),
        df["model_cover_prob_raw"].to_numpy(),
    )
    df["ats_edge"] = df["predicted_margin"] - df["spread_line"]

    df["model_home_prob_raw"] = home_win_probability_t(df["predicted_margin"].to_numpy(), df["predicted_std"].to_numpy(), dof)
    if calibrate_ml:
        df["model_home_prob"] = _calibrate_live(
            hist_ml_bets, "model_home_prob_raw", pl.col("home_margin") > 0, df["model_home_prob_raw"].to_numpy()
        )
    else:
        df["model_home_prob"] = df["model_home_prob_raw"]
    market_home_prob, _ = market_moneyline_no_vig(df["home_moneyline"].to_numpy(), df["away_moneyline"].to_numpy())
    df["market_home_prob_no_vig"] = market_home_prob
    df["ml_edge"] = df["model_home_prob"] - df["market_home_prob_no_vig"]

    # Selection rules, carried over from the backtest so the live board and
    # the validated backtest agree on what counts as a pick.
    if config.USE_ADAPTIVE_THRESHOLDS:
        df["ats_threshold_used"] = _live_ats_threshold(hist_ats_bets, df)
    else:
        df["ats_threshold_used"] = config.ATS_EDGE_THRESHOLD_POINTS
    df["ats_qualifies"] = df["ats_edge"].abs() > df["ats_threshold_used"]
    df["ats_edge_shrunk"] = df["ats_edge"] * _live_shrinkage_slope(hist_ats_bets)

    df["ml_threshold_used"] = config.ML_EDGE_THRESHOLD_PROB
    ml_side = np.where(
        df["ml_edge"] > config.ML_EDGE_THRESHOLD_PROB, "home",
        np.where(df["ml_edge"] < -config.ML_EDGE_THRESHOLD_PROB, "away", "none"),
    )
    if config.ML_UNDERDOG_ONLY:
        favorite_side = np.where(df["spread_line"].to_numpy() > 0, "home", "away")
        ml_side = np.where(ml_side == favorite_side, "none", ml_side)
    df["ml_qualifies"] = ml_side != "none"

    df["model"] = name
    return df


def _live_shrinkage_slope(hist_ats_bets: pl.DataFrame) -> float:
    hist = hist_ats_bets.filter(
        pl.col("home_margin").is_not_null() & pl.col("spread_line").is_not_null()
    ).to_pandas()
    slope, _ = fit_edge_shrinkage_all_history(
        (hist["predicted_margin"] - hist["spread_line"]).to_numpy(),
        (hist["home_margin"] - hist["spread_line"]).to_numpy(),
    )
    return slope if np.isfinite(slope) else float("nan")


def build_all_boards(
    live_by_model: dict[str, pd.DataFrame | None], calibrate_ml_by_model: dict[str, bool]
) -> pd.DataFrame | None:
    """Read each model's own historical bets (for calibration) and assemble
    one combined upcoming-picks table across all models with data."""
    boards = []
    for name, live in live_by_model.items():
        if live is None or live.empty:
            continue
        hist_ats = pl.read_parquet(config.BACKTESTS_DIR / f"{name}_ats_bets.parquet")
        hist_ml = pl.read_parquet(config.BACKTESTS_DIR / f"{name}_ml_bets.parquet")
        boards.append(build_upcoming_board(name, live, hist_ats, hist_ml, calibrate_ml_by_model[name]))

    if not boards:
        return None
    return pd.concat(boards, ignore_index=True)


def merge_total_predictions(upcoming: pd.DataFrame, live_total_by_model: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """Attach each model's live TOTALS prediction (predicted_total) onto the
    upcoming ATS/ML board, purely as a display value for a derived predicted
    score (predicted_home_score/predicted_away_score = (total +/- margin)/2)
    -- no totals edge/threshold/betting logic here, that's a separate,
    bigger decision pending evidence the totals model is worth it (see
    README's "Totals -- phase 1" section)."""
    frames = []
    for name, live_total in live_total_by_model.items():
        if live_total is None or live_total.empty:
            continue
        frames.append(live_total[["game_id", "predicted_total"]].assign(model=name))
    if not frames:
        return upcoming
    total_preds = pd.concat(frames, ignore_index=True)
    return upcoming.merge(total_preds, on=["game_id", "model"], how="left")


def append_to_picks_log(upcoming: pd.DataFrame, log_path=None) -> int:
    """Append this run's upcoming picks to a persistent, append-only log,
    instead of upcoming_picks.parquet's overwrite-every-run behavior. Once
    this week's games are played and the pipeline reruns for next week, the
    live board moves on -- without this, there'd be no record of what the
    model actually said BEFORE a game, only what the walk-forward backtest
    recomputes for it after the fact (using whatever model state existed
    whenever the pipeline was next rerun, not necessarily what was on
    screen the week of). Every call appends a new timestamped snapshot
    rather than deduplicating -- deliberately: if the model's read on a game
    changes over the week (a new injury designation, a line move), that's
    a real and interesting thing to have on record, not noise to collapse
    away. Returns the number of rows appended.
    """
    path = log_path or config.PICKS_LOG_PATH
    snapshot = upcoming.copy()
    snapshot["logged_at"] = datetime.now(timezone.utc).isoformat()

    if path.exists():
        existing = pl.read_parquet(path)
        combined = pl.concat([existing, pl.from_pandas(snapshot)], how="diagonal_relaxed")
    else:
        combined = pl.from_pandas(snapshot)

    combined.write_parquet(path)
    return len(snapshot)
