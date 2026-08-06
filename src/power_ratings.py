"""Leakage-safe, margin-of-victory Elo power ratings.

Ratings are updated strictly game-by-game in chronological order. For every
game we first RECORD each team's pre-game rating (what a bettor could have
known beforehand), then update both teams' ratings using the actual result.
Between seasons, ratings are partially regressed toward the mean to reflect
roster turnover.
"""

from collections import defaultdict

import numpy as np
import polars as pl

from . import config


def _expected_score(rating_diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))


def _mov_multiplier(margin: float, elo_diff_winner: float) -> float:
    if not config.ELO_MOV_MULTIPLIER:
        return 1.0
    return np.log(abs(margin) + 1.0) * (2.2 / ((elo_diff_winner * 0.001) + 2.2))


def compute_elo_ratings(
    games: pl.DataFrame,
    k: float = config.ELO_K,
    home_field_advantage: float = config.ELO_HOME_FIELD_ADVANTAGE,
    regression_to_mean: float = config.SEASON_REGRESSION_TO_MEAN,
    initial: float = config.ELO_INITIAL,
    mov_multiplier: bool = config.ELO_MOV_MULTIPLIER,
    prefix: str = "elo",
) -> pl.DataFrame:
    """Returns one row per game_id with each team's PRE-game elo rating and
    the diff (home_pre - away_pre + home field advantage).

    Parameters are overridable so the same update mechanics can produce more
    than one rating "speed" -- e.g. the default fast-adapting `elo` (K=20,
    33%/year regression) alongside a slow-decaying `elo_stable` (lower K, far
    less regression) meant to track persistent organizational quality that
    the fast rating erases too quickly between seasons. Output columns are
    named from `prefix`: `home_{prefix}_pre`, `away_{prefix}_pre`, `{prefix}_diff`.
    """
    games = games.sort(["season", "week", "game_id"])
    ratings: dict[str, float] = defaultdict(lambda: initial)

    records = []
    current_season = None

    for row in games.iter_rows(named=True):
        season = row["season"]
        home, away = row["home_team"], row["away_team"]

        if current_season is not None and season != current_season:
            for team in list(ratings.keys()):
                ratings[team] = initial * regression_to_mean + ratings[team] * (1 - regression_to_mean)
        current_season = season

        home_pre = ratings[home]
        away_pre = ratings[away]
        elo_diff = (home_pre + home_field_advantage) - away_pre

        records.append(
            {
                "game_id": row["game_id"],
                f"home_{prefix}_pre": home_pre,
                f"away_{prefix}_pre": away_pre,
                f"{prefix}_diff": elo_diff,
            }
        )

        margin = row["home_margin"]
        if margin is None:
            continue

        actual_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        expected_home = _expected_score(elo_diff)
        winner_diff = elo_diff if margin >= 0 else -elo_diff
        mult = _mov_multiplier(margin, winner_diff) if mov_multiplier else 1.0

        delta = k * mult * (actual_home - expected_home)
        ratings[home] = home_pre + delta
        ratings[away] = away_pre - delta

    return pl.DataFrame(records)
