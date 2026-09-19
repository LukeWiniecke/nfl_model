"""Turn model predictions + market lines into simulated flat-stake bets and
grade them against actual results.

Betting rules implemented (flat staking, no parlays/teasers/live betting):
  - ATS: bet the HOME side when predicted_margin - spread_line exceeds
    +threshold, the AWAY side when it's below -threshold, otherwise no bet.
    A push (home_margin == spread_line exactly) returns the stake.
  - ML: bet the side whose de-vigged market probability the model exceeds by
    more than the threshold. A tie (home_margin == 0) voids the bet.
"""

import numpy as np
import pandas as pd
import polars as pl

from scipy.stats import norm

from . import config
from .probabilities import (
    ats_cover_probability_t,
    breakeven_probability,
    fit_edge_shrinkage_walk_forward,
    fit_t_dof_walk_forward,
    home_win_probability_t,
    kelly_fraction,
    market_moneyline_no_vig,
    over_probability_t,
    walk_forward_calibrate,
)
from .utils import american_payout_multiplier, get_logger

logger = get_logger(__name__)


def _bet_side(edge: np.ndarray, threshold) -> np.ndarray:
    """`threshold` may be a scalar or a per-row array -- the adaptive rule
    (adaptive_ats_threshold) produces a different bar for each game, while the
    dashboard's slider and the walk-forward threshold optimizer both still
    pass a single number. np.where broadcasts either way."""
    return np.where(edge > threshold, "home", np.where(edge < -threshold, "away", "none"))


def decide_bet_side(df: pd.DataFrame, market: str, threshold, edge_col: str | None = None) -> np.ndarray:
    """The single source of truth for "which side, if any, do we bet".

    Exists because the selection rule is applied in four places (the ATS
    regrade, the ML regrade, and two ad-hoc spots in the dashboard that
    recompute it at the sidebar's threshold), and the moneyline underdog-only
    restriction was silently missing from the dashboard copies -- so the app
    would show favorite ML picks the backtest would never have placed.

    `edge_col` names the column to SELECT on, and defaults to the market's
    reported edge, which is what every existing caller (the dashboard
    included) wants. build_ats_bets passes `ats_edge_select` instead so the
    reported edge can be market-anchored while selection stays on the
    unanchored scale the thresholds were validated on -- see
    config.ATS_SELECT_ON_UNANCHORED.
    """
    if edge_col is None:
        edge_col = "ats_edge" if market == "ats" else "ml_edge"
    side = _bet_side(df[edge_col].to_numpy(), threshold)
    if market == "ml" and config.ML_UNDERDOG_ONLY:
        favorite_side = np.where(df["spread_line"].to_numpy() > 0, "home", "away")
        side = np.where(side == favorite_side, "none", side)
    return side


def adaptive_ats_threshold(df: pd.DataFrame, edge_col: str = "ats_edge") -> np.ndarray:
    """Per-game minimum RAW edge required to place an ATS bet, derived rather
    than hardcoded. Three steps, all walk-forward:

      1. How much of a claimed edge is real? -> shrinkage slope b
         (probabilities.fit_edge_shrinkage_walk_forward).
      2. What true edge, in points, clears the vig at THIS game's actual
         price? break-even win rate p* from the offered odds, converted to
         points via the walk-forward sd of home_margin - spread_line:
         true_edge_needed = Phi^-1(p*) * sd.
      3. Convert back to the raw scale the model actually reports:
         raw_needed = true_edge_needed / b.

    Worked example at the measured values: p* = 0.5238 at -110, sd = 12.86 ->
    a true edge of 0.77 points is needed; ridge's slope of 0.203 turns that
    into a raw-edge bar of 3.79 points. The old fixed threshold was 1.5.

    Returned on the RAW edge scale on purpose: `ats_edge` keeps meaning
    exactly what it always did, so the dashboard's slider, the edge-bucket
    breakdowns, and optimal_threshold_walk_forward all keep working unchanged
    and stay comparable to previously saved runs.

    Everything here -- the fitted slope, and the fixed fallback it reverts to
    when there is too little history -- is expressed on the scale of
    `edge_col`. That is why the caller must pass the SAME column it will then
    select on: ATS_EDGE_THRESHOLD_POINTS is a raw-scale constant, so deriving
    the bar from a market-anchored edge and then applying it to that same
    (4-9x smaller) edge silently turns the fallback into a 10-30 point bar.
    """
    slope, sd = fit_edge_shrinkage_walk_forward(df, edge_col, "_ats_resid")

    # Use the worse (more expensive) of the two sides' prices, so a game is
    # only bet if it clears the bar at the price actually available.
    home_odds = df["home_spread_odds"].fillna(config.DEFAULT_SPREAD_ODDS).to_numpy()
    away_odds = df["away_spread_odds"].fillna(config.DEFAULT_SPREAD_ODDS).to_numpy()
    p_breakeven = np.maximum(breakeven_probability(home_odds), breakeven_probability(away_odds))

    true_edge_needed = norm.ppf(p_breakeven) * sd
    raw_needed = true_edge_needed / slope

    return np.where(np.isfinite(raw_needed), raw_needed, config.ATS_EDGE_THRESHOLD_POINTS)


def _totals_side(edge: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(edge > threshold, "over", np.where(edge < -threshold, "under", "none"))


def regrade_ats(df: pd.DataFrame, threshold: float, edge_col: str = "ats_edge") -> pd.DataFrame:
    """Given a table that already has `ats_edge`, `home_margin`, `spread_line`,
    and spread odds columns, (re)decide bet_side at `threshold` and grade it.
    Split out from build_ats_bets so the dashboard can re-run this at a
    user-chosen threshold without recomputing predictions/edge from scratch.

    `edge_col` defaults to the reported `ats_edge`, so the dashboard's regrade
    path and optimal_threshold_walk_forward are unchanged; build_ats_bets
    passes the selection edge (see config.ATS_SELECT_ON_UNANCHORED). Only the
    SIDE decision uses it -- grading is against the realized margin either
    way, which is why moving the selection scale can change which games are
    bet but never how a bet resolves."""
    # reset_index: bet_won below gets a fresh default RangeIndex, which silently
    # misaligns (and raises) against df's own index unless it's already 0..n-1 --
    # true for a full to_pandas() conversion but not for an arbitrary filtered slice.
    df = df.reset_index(drop=True)
    df["bet_side"] = decide_bet_side(df, "ats", threshold, edge_col=edge_col)

    is_push = np.isclose(df["home_margin"], df["spread_line"])
    home_covers = df["home_margin"] > df["spread_line"]

    bet_odds = np.where(
        df["bet_side"] == "home",
        df["home_spread_odds"].fillna(config.DEFAULT_SPREAD_ODDS),
        df["away_spread_odds"].fillna(config.DEFAULT_SPREAD_ODDS),
    )
    payout_mult = american_payout_multiplier(bet_odds)

    bet_wins = np.where(
        df["bet_side"] == "home", home_covers, np.where(df["bet_side"] == "away", ~home_covers, False)
    )

    profit = np.select(
        [df["bet_side"] == "none", is_push, bet_wins],
        [0.0, 0.0, config.FLAT_STAKE_UNITS * payout_mult],
        default=-config.FLAT_STAKE_UNITS,
    )

    df["is_push"] = is_push
    bet_won = pd.Series(bet_wins, dtype="boolean")
    bet_won[df["bet_side"] == "none"] = pd.NA
    df["bet_won"] = bet_won
    df["profit_units"] = profit
    df["staked"] = np.where(df["bet_side"] == "none", 0.0, config.FLAT_STAKE_UNITS)
    df["market"] = "ATS"
    return df


def regrade_ml(df: pd.DataFrame, threshold, underdog_only: bool | None = None) -> pd.DataFrame:
    """Same idea as regrade_ats, for the moneyline market: given a table that
    already has `ml_edge`, `home_margin`, and moneylines, (re)decide bet_side
    at `threshold` and grade it.

    `underdog_only` (defaulting to config.ML_UNDERDOG_ONLY) drops any bet that
    would land on the market favorite. See that constant's note: split by
    side, the model's favorite-side moneyline bets lost money at every
    threshold tested, while its underdog bets carried the entire return.
    """
    df = df.reset_index(drop=True)  # see regrade_ats -- bet_won needs a clean 0..n-1 index to align against
    if underdog_only is None or underdog_only == config.ML_UNDERDOG_ONLY:
        df["bet_side"] = decide_bet_side(df, "ml", threshold)
    else:
        # Explicit override (used to A/B the restriction itself); spread_line
        # > 0 means the HOME team is favored, so the underdog is away.
        side = _bet_side(df["ml_edge"].to_numpy(), threshold)
        if underdog_only:
            favorite_side = np.where(df["spread_line"].to_numpy() > 0, "home", "away")
            side = np.where(side == favorite_side, "none", side)
        df["bet_side"] = side

    is_tie = df["home_margin"] == 0
    home_wins = df["home_margin"] > 0

    bet_odds = np.where(df["bet_side"] == "home", df["home_moneyline"], df["away_moneyline"])
    payout_mult = american_payout_multiplier(bet_odds)

    bet_wins = np.where(df["bet_side"] == "home", home_wins, np.where(df["bet_side"] == "away", ~home_wins, False))

    profit = np.select(
        [df["bet_side"] == "none", is_tie, bet_wins],
        [0.0, 0.0, config.FLAT_STAKE_UNITS * payout_mult],
        default=-config.FLAT_STAKE_UNITS,
    )

    df["is_push"] = is_tie
    bet_won = pd.Series(bet_wins, dtype="boolean")
    bet_won[df["bet_side"] == "none"] = pd.NA
    df["bet_won"] = bet_won
    df["profit_units"] = profit
    df["staked"] = np.where(df["bet_side"] == "none", 0.0, config.FLAT_STAKE_UNITS)
    df["market"] = "ML"
    return df


def _fit_dof_or_default(df: pd.DataFrame, actual_col: str, predicted_col: str) -> np.ndarray:
    """fit_t_dof_walk_forward needs a real per-game predicted_std column to
    standardize residuals against; falls back to config.DEFAULT_T_DOF
    (practically Normal) in the defensive edge case where the caller only
    has a single scalar margin_std/total_std (same edge case build_ats_bets/
    build_ml_bets/build_totals_bets already defend against for `std` itself)."""
    if "predicted_std" not in df.columns:
        return np.full(len(df), config.DEFAULT_T_DOF)
    return fit_t_dof_walk_forward(df, actual_col, predicted_col, "predicted_std")


def build_ats_bets(preds_with_market: pl.DataFrame, margin_std: float) -> pl.DataFrame:
    df = preds_with_market.to_pandas().copy()
    std = df["predicted_std"] if "predicted_std" in df.columns else margin_std
    df["ats_edge"] = df["predicted_margin"] - df["spread_line"]
    # Student's-t instead of Normal: calibration_bins on the raw (pre-
    # isotonic) probability showed real, measured OVERCONFIDENCE -- model
    # predictions more extreme than actual outcome rates in both directions,
    # worst for LightGBM specifically. A walk-forward-fit degrees-of-freedom
    # (probabilities.fit_t_dof_walk_forward) lets each model's own residual
    # tail shape correct for this directly, converging to Normal (dof~30)
    # when a model's raw probabilities are already well-behaved. See
    # config.py's note for the measured result.
    dof = _fit_dof_or_default(df, "home_margin", "predicted_margin")
    df["model_cover_prob_raw"] = ats_cover_probability_t(df["predicted_margin"], df["spread_line"], std, dof)
    home_covers = (df["home_margin"] > df["spread_line"]).to_numpy().astype(float)
    df["model_cover_prob"] = walk_forward_calibrate(df, "model_cover_prob_raw", home_covers)

    # Realized result against the line -- the thing the raw edge is supposed
    # to predict, and the target the shrinkage slope is fit against.
    df["_ats_resid"] = df["home_margin"] - df["spread_line"]
    slope, _ = fit_edge_shrinkage_walk_forward(df, "ats_edge", "_ats_resid")
    df["ats_edge_shrinkage"] = slope
    # The model's HONEST expected edge in points, as opposed to its claimed
    # one. Not used for bet selection directly (the threshold is converted to
    # the raw scale instead, so ats_edge keeps its historical meaning), but
    # it is the number actually worth looking at on a pick.
    df["ats_edge_shrunk"] = df["ats_edge"] * slope

    # The edge bets are SELECTED on, which is not necessarily the edge that
    # is reported. Under market anchoring the reported edge is b_S * e_raw --
    # a monotone rescaling, so it ranks games identically, but it lives on a
    # scale the betting layer's raw-scale constants do not understand. See
    # config.ATS_SELECT_ON_UNANCHORED. With the flag off, or for a model that
    # emits no unanchored column (ridge/lgbm never anchor), this is ats_edge
    # and every line below collapses to the previous behaviour.
    select_unanchored = config.ATS_SELECT_ON_UNANCHORED and "predicted_margin_unanchored" in df.columns
    if select_unanchored:
        df["ats_edge_select"] = df["predicted_margin_unanchored"] - df["spread_line"]
    else:
        df["ats_edge_select"] = df["ats_edge"]

    if config.USE_ADAPTIVE_THRESHOLDS:
        # Reported bar, on the reported edge's scale: what ats_edge has to
        # clear, kept so the saved column stays comparable with ats_edge and
        # with previous runs' parquets.
        reported_threshold = adaptive_ats_threshold(df, "ats_edge")
        df["ats_threshold_used"] = reported_threshold
        # Selection bar, on the selection edge's scale -- the one actually
        # applied below.
        threshold = adaptive_ats_threshold(df, "ats_edge_select") if select_unanchored else reported_threshold
        df["ats_select_threshold_used"] = threshold
        finite = np.isfinite(slope)
        if finite.any():
            logger.info(
                f"[ats] adaptive threshold: median={np.nanmedian(threshold):.2f} pts "
                f"(shrinkage slope median={np.nanmedian(slope[finite]):.3f}); "
                f"fixed fallback would be {config.ATS_EDGE_THRESHOLD_POINTS}"
            )
        if select_unanchored:
            logger.info(
                f"[ats] selecting on unanchored edge: select threshold median="
                f"{np.nanmedian(threshold):.2f} pts vs reported-scale median="
                f"{np.nanmedian(reported_threshold):.2f} pts"
            )
    else:
        threshold = config.ATS_EDGE_THRESHOLD_POINTS
        df["ats_threshold_used"] = threshold
        df["ats_select_threshold_used"] = threshold

    df = df.drop(columns=["_ats_resid"])
    df = regrade_ats(df, threshold, edge_col="ats_edge_select")
    return pl.from_pandas(df)


def build_ml_bets(preds_with_market: pl.DataFrame, margin_std: float, calibrate: bool = True) -> pl.DataFrame:
    df = preds_with_market.to_pandas().copy()
    std = df["predicted_std"] if "predicted_std" in df.columns else margin_std
    dof = _fit_dof_or_default(df, "home_margin", "predicted_margin")
    df["model_home_prob_raw"] = home_win_probability_t(df["predicted_margin"], std, dof)
    if calibrate:
        home_win = (df["home_margin"] > 0).to_numpy().astype(float)
        df["model_home_prob"] = walk_forward_calibrate(df, "model_home_prob_raw", home_win)
    else:
        df["model_home_prob"] = df["model_home_prob_raw"]
    market_home_prob, _ = market_moneyline_no_vig(df["home_moneyline"], df["away_moneyline"])
    df["market_home_prob_no_vig"] = market_home_prob
    df["ml_edge"] = df["model_home_prob"] - df["market_home_prob_no_vig"]
    df = regrade_ml(df, config.ML_EDGE_THRESHOLD_PROB)
    return pl.from_pandas(df)


def regrade_totals(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Given a table that already has `total_edge`, `total_points`,
    `total_line`, and over/under odds columns, (re)decide bet_side
    (over/under) at `threshold` and grade it. Same structure as
    regrade_ats/regrade_ml, for the totals market."""
    df = df.reset_index(drop=True)  # see regrade_ats -- bet_won needs a clean 0..n-1 index to align against
    df["bet_side"] = _totals_side(df["total_edge"].to_numpy(), threshold)

    is_push = np.isclose(df["total_points"], df["total_line"])
    over_hits = df["total_points"] > df["total_line"]

    bet_odds = np.where(
        df["bet_side"] == "over",
        df["over_odds"].fillna(config.DEFAULT_SPREAD_ODDS),
        df["under_odds"].fillna(config.DEFAULT_SPREAD_ODDS),
    )
    payout_mult = american_payout_multiplier(bet_odds)

    bet_wins = np.where(
        df["bet_side"] == "over", over_hits, np.where(df["bet_side"] == "under", ~over_hits, False)
    )

    profit = np.select(
        [df["bet_side"] == "none", is_push, bet_wins],
        [0.0, 0.0, config.FLAT_STAKE_UNITS * payout_mult],
        default=-config.FLAT_STAKE_UNITS,
    )

    df["is_push"] = is_push
    bet_won = pd.Series(bet_wins, dtype="boolean")
    bet_won[df["bet_side"] == "none"] = pd.NA
    df["bet_won"] = bet_won
    df["profit_units"] = profit
    df["staked"] = np.where(df["bet_side"] == "none", 0.0, config.FLAT_STAKE_UNITS)
    df["market"] = "TOTAL"
    return df


def build_totals_bets(preds_with_market: pl.DataFrame, total_std: float, calibrate: bool = True) -> pl.DataFrame:
    df = preds_with_market.to_pandas().copy()
    std = df["predicted_std"] if "predicted_std" in df.columns else total_std
    df["total_edge"] = df["predicted_total"] - df["total_line"]
    dof = _fit_dof_or_default(df, "total_points", "predicted_total")
    df["model_over_prob_raw"] = over_probability_t(df["predicted_total"], df["total_line"], std, dof)
    if calibrate:
        over_hit = (df["total_points"] > df["total_line"]).to_numpy().astype(float)
        df["model_over_prob"] = walk_forward_calibrate(df, "model_over_prob_raw", over_hit)
    else:
        df["model_over_prob"] = df["model_over_prob_raw"]
    df = regrade_totals(df, config.TOTAL_EDGE_THRESHOLD_POINTS)
    return pl.from_pandas(df)


def summarize(bets: pl.DataFrame) -> dict:
    placed = bets.filter(pl.col("bet_side") != "none")
    decided = placed.filter(~pl.col("is_push"))
    n_bets = placed.height
    n_wins = decided.filter(pl.col("bet_won")).height
    n_losses = decided.filter(~pl.col("bet_won")).height
    total_staked = placed["staked"].sum()
    total_profit = placed["profit_units"].sum()
    return {
        "n_bets": n_bets,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_pushes": n_bets - n_wins - n_losses,
        "win_rate": n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else float("nan"),
        "total_staked": total_staked,
        "total_profit_units": total_profit,
        "roi": total_profit / total_staked if total_staked else float("nan"),
    }


def simulate_bankroll(
    bets: pd.DataFrame,
    prob_col: str,
    home_odds_col: str,
    away_odds_col: str,
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.05,
    starting_bankroll: float = 100.0,
) -> pd.DataFrame:
    """Sequential, compounding bankroll simulation for placed bets, in
    chronological order. Unlike the flat-stake backtest above (a fixed 1 unit
    every time, regardless of bankroll -- fine for measuring whether an edge
    exists, but not how you'd actually size real bets), this stakes a
    fraction of the CURRENT bankroll set by the Kelly criterion, scaled down
    by `kelly_multiplier` (fractional Kelly -- full Kelly is too aggressive
    when the probability estimate feeding it is uncertain, which the
    Diagnostics tab already shows ours is) and capped at `kelly_cap` of
    bankroll per bet as a practical safeguard against a single mis-estimated
    game wiping out a big chunk of the bankroll.
    """
    placed = bets[bets["bet_side"] != "none"].sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    if placed.empty:
        placed = placed.copy()
        placed["kelly_stake_frac"] = []
        placed["bankroll"] = []
        return placed

    is_home = (placed["bet_side"] == "home").to_numpy()
    prob_for_side = np.where(is_home, placed[prob_col], 1.0 - placed[prob_col])
    odds_for_side = np.where(is_home, placed[home_odds_col], placed[away_odds_col])
    odds_for_side = pd.Series(odds_for_side).fillna(config.DEFAULT_SPREAD_ODDS).to_numpy()

    kelly_f = kelly_fraction(prob_for_side, odds_for_side) * kelly_multiplier
    kelly_f = np.clip(kelly_f, 0.0, kelly_cap)
    payout_mult = american_payout_multiplier(odds_for_side)

    is_push = placed["is_push"].to_numpy()
    bet_won = placed["bet_won"].to_numpy()

    bankroll = starting_bankroll
    bankroll_history = np.empty(len(placed))
    stake_history = np.empty(len(placed))
    for i in range(len(placed)):
        stake = bankroll * kelly_f[i]
        if is_push[i]:
            profit = 0.0
        elif bet_won[i]:
            profit = stake * payout_mult[i]
        else:
            profit = -stake
        bankroll += profit
        stake_history[i] = stake
        bankroll_history[i] = bankroll

    placed = placed.copy()
    placed["kelly_stake_frac"] = kelly_f
    placed["kelly_stake"] = stake_history
    placed["bankroll"] = bankroll_history
    return placed


def max_drawdown(bankroll: np.ndarray) -> float:
    """Largest peak-to-trough decline in a bankroll series, as a fraction
    (e.g. -0.35 means a 35% drawdown from the running peak at some point)."""
    arr = np.asarray(bankroll, dtype=float)
    if len(arr) == 0:
        return float("nan")
    running_max = np.maximum.accumulate(arr)
    drawdown = (arr - running_max) / running_max
    return float(drawdown.min())


def bootstrap_bankroll_paths(
    bets: pd.DataFrame,
    prob_col: str,
    home_odds_col: str,
    away_odds_col: str,
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.05,
    starting_bankroll: float = 100.0,
    n_boot: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """The single realized bankroll path from simulate_bankroll is one sample
    from a highly path-dependent process -- Kelly compounding means the exact
    ORDER wins and losses happened in changes the outcome, not just their
    count. A path that looks great (or terrible) can be mostly the luck of
    that order, not evidence the sizing rule itself is good or bad.

    This resamples the same placed-bet pool WITH REPLACEMENT `n_boot` times
    (same bootstrap idea as evaluation.summarize_with_ci, just applied to a
    compounding path instead of a flat mean) and re-runs the identical
    Kelly-sizing rule on each resampled draw, to show the RANGE of plausible
    outcomes instead of one line. Returns an (n_boot, n_bets) array of
    bankroll values; the recurrence is inherently sequential (bankroll[t]
    depends on bankroll[t-1]), so the loop is over bet index, vectorized
    across all n_boot paths at each step rather than looped per-path.
    """
    placed = bets[bets["bet_side"] != "none"].reset_index(drop=True)
    n = len(placed)
    if n == 0:
        return np.empty((n_boot, 0))

    is_home = (placed["bet_side"] == "home").to_numpy()
    prob_for_side = np.where(is_home, placed[prob_col], 1.0 - placed[prob_col])
    odds_for_side = np.where(is_home, placed[home_odds_col], placed[away_odds_col])
    odds_for_side = pd.Series(odds_for_side).fillna(config.DEFAULT_SPREAD_ODDS).to_numpy()

    kelly_f_pool = np.clip(kelly_fraction(prob_for_side, odds_for_side) * kelly_multiplier, 0.0, kelly_cap)
    payout_mult_pool = american_payout_multiplier(odds_for_side)
    is_push_pool = placed["is_push"].to_numpy()
    bet_won_pool = placed["bet_won"].to_numpy()

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    kelly_f = kelly_f_pool[idx]
    payout_mult = payout_mult_pool[idx]
    is_push = is_push_pool[idx]
    bet_won = bet_won_pool[idx]

    bankroll = np.full(n_boot, starting_bankroll)
    paths = np.empty((n_boot, n))
    for t in range(n):
        stake = bankroll * kelly_f[:, t]
        profit = np.where(is_push[:, t], 0.0, np.where(bet_won[:, t], stake * payout_mult[:, t], -stake))
        bankroll = bankroll + profit
        paths[:, t] = bankroll
    return paths


def sweep_kelly_multiplier(
    bets: pd.DataFrame,
    prob_col: str,
    home_odds_col: str,
    away_odds_col: str,
    multipliers: np.ndarray,
    kelly_cap: float = 0.05,
    starting_bankroll: float = 100.0,
    n_boot: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """Empirically find the growth-optimal fractional-Kelly multiplier
    instead of assuming a conventional one (e.g. 0.25) is right. For each
    candidate multiplier, bootstrap-resamples the same bet pool `n_boot`
    times (bootstrap_bankroll_paths) and measures the resulting distribution
    of GEOMETRIC growth -- mean log(bankroll_final / bankroll_start), the
    actual quantity Kelly theory optimizes -- plus two tail-risk views
    (5th-percentile final bankroll, and the fraction of resampled paths
    that ever drew down more than 50%). The multiplier that maximizes mean
    log-growth is the empirically-optimal one GIVEN this bet pool's own
    real variance; if that peak sits well below the naive full-Kelly value
    of 1.0, that's direct evidence of how much the probability estimate's
    own uncertainty should discount the textbook stake.
    """
    records = []
    for mult in multipliers:
        paths = bootstrap_bankroll_paths(
            bets, prob_col, home_odds_col, away_odds_col,
            kelly_multiplier=float(mult), kelly_cap=kelly_cap, starting_bankroll=starting_bankroll,
            n_boot=n_boot, seed=seed,
        )
        if paths.shape[1] == 0:
            continue
        final = paths[:, -1]
        log_growth = np.log(final / starting_bankroll)
        mean_log_growth = float(log_growth.mean())
        n_bets = paths.shape[1]
        per_bet_rate = float(np.exp(mean_log_growth / n_bets) - 1.0)

        running_max = np.maximum.accumulate(paths, axis=1)
        drawdown = (paths - running_max) / running_max
        worst_drawdown = drawdown.min(axis=1)

        records.append({
            "kelly_multiplier": float(mult),
            "mean_log_growth": mean_log_growth,
            "per_bet_growth_rate": per_bet_rate,
            "median_final_bankroll": float(np.median(final)),
            "p5_final_bankroll": float(np.percentile(final, 5)),
            "prob_drawdown_50": float((worst_drawdown < -0.5).mean()),
        })
    return pd.DataFrame(records)


def optimal_threshold_walk_forward(
    bets_full: pd.DataFrame,
    market: str,
    threshold_grid: np.ndarray,
    min_train_seasons: int = 3,
    min_bets: int = 20,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """For each test season, grid-search the bet-selection threshold using
    ONLY strictly earlier seasons -- selecting whichever threshold maximized
    the bootstrap 95% CI LOWER BOUND on ROI (not raw mean ROI, which rewards
    a threshold that got lucky on a handful of bets) -- then apply that
    season's chosen threshold OUT OF SAMPLE to grade the test season itself.
    Same walk-forward discipline used everywhere else in this project,
    applied to a hyperparameter instead of the model. `bets_full` needs
    ats_edge (or ml_edge) plus everything regrade_ats/regrade_ml needs,
    already computed for every game regardless of threshold. Returns one
    row per test season with the threshold chosen ahead of time and the
    REALIZED (never-peeked-at) result under that choice.
    """
    regrade_fn = regrade_ats if market == "ats" else regrade_ml
    rng = np.random.default_rng(seed)
    seasons = sorted(bets_full["season"].unique())

    records = []
    for test_season in seasons:
        train = bets_full[bets_full["season"] < test_season]
        if train["season"].nunique() < min_train_seasons:
            continue

        best_t, best_lo = None, -np.inf
        for t in threshold_grid:
            regraded = regrade_fn(train, float(t))
            placed = regraded[regraded["bet_side"] != "none"]
            n = len(placed)
            if n < min_bets:
                continue
            profit = placed["profit_units"].to_numpy()
            idx = rng.integers(0, n, size=(n_boot, n))
            lo = np.percentile(profit[idx].mean(axis=1), 2.5)
            if lo > best_lo:
                best_lo, best_t = lo, float(t)

        if best_t is None:
            continue

        test = bets_full[bets_full["season"] == test_season]
        regraded_test = regrade_fn(test, best_t)
        placed_test = regraded_test[regraded_test["bet_side"] != "none"]
        n_bets = len(placed_test)
        staked = placed_test["staked"].sum()
        profit = placed_test["profit_units"].sum()
        records.append({
            "season": test_season,
            "chosen_threshold": best_t,
            "n_bets": n_bets,
            "profit": float(profit) if n_bets else 0.0,
            "roi": float(profit / staked) if staked else np.nan,
        })
    return pd.DataFrame(records)
