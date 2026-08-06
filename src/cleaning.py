"""Clean raw nflverse tables into a canonical game-level table.

Sign convention (confirmed against real data, not assumed): nflverse's
`spread_line` is the market's expected HOME-team margin. Positive means the
home team is favored by that many points; negative means the away team is
favored. `result` = home_score - away_score. A home team "covers" the spread
when `result > spread_line`.
"""

import polars as pl

from . import config
from .utils import get_logger, standardize_team

logger = get_logger(__name__)

TEAM_COLS_SCHEDULES = ["home_team", "away_team"]
TEAM_COLS_PBP = ["home_team", "away_team", "posteam", "defteam"]


def clean_schedules(df: pl.DataFrame) -> pl.DataFrame:
    # Deliberately KEEPS not-yet-played games (null scores) so live_picks can
    # predict them -- home_margin/home_score/away_score just come out null
    # for those rows. Every step that trains or backtests (modeling._walk_forward,
    # skill_vs_market, etc.) already filters on home_margin.is_not_null()
    # itself, so this doesn't change any backtest result.
    df = df.filter(pl.col("game_type").is_in(config.GAME_TYPES_KEPT))

    for col in TEAM_COLS_SCHEDULES:
        df = df.with_columns(standardize_team(pl.col(col)).alias(col))

    df = df.with_columns(
        (pl.col("home_score") - pl.col("away_score")).alias("home_margin"),
        pl.col("gameday").str.to_date().alias("gameday"),
    )

    keep = [
        "game_id", "old_game_id", "season", "week", "game_type", "gameday",
        "home_team", "away_team", "home_score", "away_score", "home_margin",
        "spread_line", "total_line", "home_moneyline", "away_moneyline",
        "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
        "home_rest", "away_rest", "div_game", "roof", "surface", "temp", "wind", "location",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df.select(keep).sort(["season", "week", "game_id"])
    logger.info(f"Cleaned schedules: {df.shape[0]} games, seasons {df['season'].min()}-{df['season'].max()}")
    return df


def clean_pbp(df: pl.DataFrame) -> pl.DataFrame:
    for col in TEAM_COLS_PBP:
        if col in df.columns:
            df = df.with_columns(standardize_team(pl.col(col)).alias(col))
    return df


def build_team_game_long(games: pl.DataFrame) -> pl.DataFrame:
    """Reshape one row-per-game into two rows-per-game (one per team), which
    rolling team-form features are computed against."""
    home = games.select(
        pl.col("game_id"), pl.col("season"), pl.col("week"), pl.col("game_type"),
        pl.col("gameday"), pl.col("div_game"),
        pl.col("home_team").alias("team"), pl.col("away_team").alias("opponent"),
        pl.lit(1).alias("is_home"),
        pl.col("home_score").alias("team_score"), pl.col("away_score").alias("opp_score"),
        pl.col("home_margin").alias("margin"),
        pl.col("home_rest").alias("rest_days"),
    )
    away = games.select(
        pl.col("game_id"), pl.col("season"), pl.col("week"), pl.col("game_type"),
        pl.col("gameday"), pl.col("div_game"),
        pl.col("away_team").alias("team"), pl.col("home_team").alias("opponent"),
        pl.lit(0).alias("is_home"),
        pl.col("away_score").alias("team_score"), pl.col("home_score").alias("opp_score"),
        (-pl.col("home_margin")).alias("margin"),
        pl.col("away_rest").alias("rest_days"),
    )
    long = pl.concat([home, away]).sort(["team", "season", "week"])
    return long
