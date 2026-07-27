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


def compute_elo_ratings(games: pl.DataFrame) -> pl.DataFrame:
    """Returns one row per game_id with each team's PRE-game elo rating and
    the diff (home_elo_pre - away_elo_pre + home field advantage)."""
    games = games.sort(["season", "week", "game_id"])
    ratings: dict[str, float] = defaultdict(lambda: config.ELO_INITIAL)
    last_season_seen: dict[str, int] = {}

    records = []
    current_season = None

    for row in games.iter_rows(named=True):
        season = row["season"]
        home, away = row["home_team"], row["away_team"]

        if current_season is not None and season != current_season:
            for team in list(ratings.keys()):
                ratings[team] = (
                    config.ELO_INITIAL * config.SEASON_REGRESSION_TO_MEAN
                    + ratings[team] * (1 - config.SEASON_REGRESSION_TO_MEAN)
                )
        current_season = season

        home_pre = ratings[home]
        away_pre = ratings[away]
        elo_diff = (home_pre + config.ELO_HOME_FIELD_ADVANTAGE) - away_pre

        records.append(
            {
                "game_id": row["game_id"],
                "home_elo_pre": home_pre,
                "away_elo_pre": away_pre,
                "elo_diff": elo_diff,
            }
        )

        margin = row["home_margin"]
        if margin is None:
            continue

        actual_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        expected_home = _expected_score(elo_diff)
        winner_diff = elo_diff if margin >= 0 else -elo_diff
        mult = _mov_multiplier(margin, winner_diff)

        delta = config.ELO_K * mult * (actual_home - expected_home)
        ratings[home] = home_pre + delta
        ratings[away] = away_pre - delta

        last_season_seen[home] = season
        last_season_seen[away] = season

    return pl.DataFrame(records)
