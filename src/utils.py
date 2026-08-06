"""Small shared helpers: American-odds <-> probability conversions and logging."""

import logging
import sys

import numpy as np


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Force UTF-8 on the stream: polars renders its tables with box-drawing
        # characters, which a default cp1252 Windows console cannot encode --
        # every logged DataFrame (e.g. main.py's skill-vs-market tables) raised
        # a UnicodeEncodeError inside logging and printed a traceback instead
        # of the table. errors="replace" keeps that from ever being fatal.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def american_to_prob(odds: np.ndarray) -> np.ndarray:
    """Convert American odds to implied probability (still includes vig)."""
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(
            odds < 0,
            (-odds) / ((-odds) + 100.0),
            100.0 / (odds + 100.0),
        )
    return prob


def prob_to_american(prob: np.ndarray) -> np.ndarray:
    """Convert a probability into fair American odds (no vig)."""
    prob = np.asarray(prob, dtype=float)
    odds = np.where(
        prob >= 0.5,
        -100.0 * prob / (1.0 - prob),
        100.0 * (1.0 - prob) / prob,
    )
    return odds


def remove_vig_two_way(prob_a: np.ndarray, prob_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a two-way market's implied (vig-included) probabilities to sum to 1."""
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def american_payout_multiplier(odds: np.ndarray) -> np.ndarray:
    """Profit per 1 unit staked if the bet wins, given American odds (0 if it loses)."""
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, 100.0 / (-odds), odds / 100.0)


# Standardize team abbreviations that have changed over the nflverse history
# (relocations/rebrands) so rolling stats and power ratings track one franchise.
TEAM_NAME_MAP = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
}


def standardize_team(col):
    return col.replace(TEAM_NAME_MAP)
