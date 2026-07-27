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

from . import config
from .probabilities import (
    ats_cover_probability,
    home_win_probability,
    market_moneyline_no_vig,
)
from .utils import american_payout_multiplier, get_logger

logger = get_logger(__name__)


def _bet_side(edge: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(edge > threshold, "home", np.where(edge < -threshold, "away", "none"))


def build_ats_bets(preds_with_market: pl.DataFrame, margin_std: float) -> pl.DataFrame:
    df = preds_with_market.to_pandas().copy()

    df["ats_edge"] = df["predicted_margin"] - df["spread_line"]
    df["model_cover_prob"] = ats_cover_probability(df["predicted_margin"], df["spread_line"], margin_std)
    df["bet_side"] = _bet_side(df["ats_edge"].to_numpy(), config.ATS_EDGE_THRESHOLD_POINTS)

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
    return pl.from_pandas(df)


def build_ml_bets(preds_with_market: pl.DataFrame, margin_std: float) -> pl.DataFrame:
    df = preds_with_market.to_pandas().copy()

    df["model_home_prob"] = home_win_probability(df["predicted_margin"], margin_std)
    market_home_prob, _ = market_moneyline_no_vig(df["home_moneyline"], df["away_moneyline"])
    df["market_home_prob_no_vig"] = market_home_prob
    df["ml_edge"] = df["model_home_prob"] - df["market_home_prob_no_vig"]
    df["bet_side"] = _bet_side(df["ml_edge"].to_numpy(), config.ML_EDGE_THRESHOLD_PROB)

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
