"""Convert predicted margins into win probabilities and fair prices, and turn
market odds into de-vigged implied probabilities for comparison.
"""

import numpy as np
from scipy.stats import norm

from . import config
from .utils import american_to_prob, prob_to_american, remove_vig_two_way


def fit_margin_std(actual_margin: np.ndarray, predicted_margin: np.ndarray) -> float:
    """Std dev of the model's residuals; used as the spread of the outcome
    distribution around a predicted margin. Should be fit on TRAIN residuals
    only, never on the games being predicted."""
    resid = np.asarray(actual_margin) - np.asarray(predicted_margin)
    std = float(np.std(resid, ddof=1))
    return std if std > 0 else config.DEFAULT_MARGIN_STD


def home_win_probability(predicted_margin: np.ndarray, margin_std: float) -> np.ndarray:
    """P(home_margin > 0) under a normal approximation centered on the
    predicted margin. NFL final margins are approximately normal/logistic;
    ties are a measure-zero edge case here and handled separately in grading."""
    predicted_margin = np.asarray(predicted_margin, dtype=float)
    return norm.cdf(predicted_margin / margin_std)


def ats_cover_probability(predicted_margin: np.ndarray, spread_line: np.ndarray, margin_std: float) -> np.ndarray:
    """P(home_margin > spread_line), i.e. probability the HOME side of the
    market spread covers, given the model's predicted margin."""
    predicted_margin = np.asarray(predicted_margin, dtype=float)
    spread_line = np.asarray(spread_line, dtype=float)
    return norm.cdf((predicted_margin - spread_line) / margin_std)


def market_moneyline_no_vig(home_moneyline: np.ndarray, away_moneyline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """De-vig the market's home/away moneylines into a fair two-way probability."""
    home_raw = american_to_prob(home_moneyline)
    away_raw = american_to_prob(away_moneyline)
    return remove_vig_two_way(home_raw, away_raw)


def market_spread_no_vig(home_spread_odds: np.ndarray, away_spread_odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """De-vig the market's spread-side prices into a fair two-way probability
    of each side covering."""
    home_raw = american_to_prob(home_spread_odds)
    away_raw = american_to_prob(away_spread_odds)
    return remove_vig_two_way(home_raw, away_raw)


def fair_moneyline_price(model_home_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Model-implied fair American odds for home and away, from the model's
    home win probability (no vig, since this is our own fair value, not a
    price we're quoting)."""
    model_home_prob = np.asarray(model_home_prob, dtype=float)
    home_odds = prob_to_american(model_home_prob)
    away_odds = prob_to_american(1.0 - model_home_prob)
    return home_odds, away_odds


def moneyline_edge(model_home_prob: np.ndarray, home_moneyline: np.ndarray, away_moneyline: np.ndarray) -> np.ndarray:
    """model_home_prob - no_vig_market_home_prob. Positive => model likes home
    relative to market; negative => model likes away."""
    market_home_prob, _ = market_moneyline_no_vig(home_moneyline, away_moneyline)
    return np.asarray(model_home_prob) - market_home_prob
