"""Build per-team-per-game EPA/success-rate stats from pbp, then turn them into
leakage-safe rolling ("form entering this game") features and assemble the
final game-level modeling table.

Leakage rule: every rolling/EWMA feature for a given game uses ONLY games that
happened strictly before it (shift(1) before the rolling window), so nothing
from the game being predicted—or the future—ever enters its own features.
"""

import polars as pl

from . import config
from .cleaning import build_team_game_long, clean_pbp
from .utils import get_logger

logger = get_logger(__name__)


def compute_game_epa_splits(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team): offensive EPA/play, success rate, pass EPA,
    rush EPA, and defensive EPA/success allowed while that team was on defense.
    """
    pbp = clean_pbp(pbp)
    plays = pbp.filter(pl.col("play_type").is_in(["run", "pass"]))

    off = (
        plays.group_by(["game_id", "posteam"])
        .agg(
            pl.col("epa").mean().alias("off_epa_play"),
            pl.col("success").mean().alias("off_success_rate"),
            pl.col("epa").filter(pl.col("pass") == 1).mean().alias("off_pass_epa"),
            pl.col("epa").filter(pl.col("rush") == 1).mean().alias("off_rush_epa"),
        )
        .rename({"posteam": "team"})
    )

    deff = (
        plays.group_by(["game_id", "defteam"])
        .agg(
            pl.col("epa").mean().alias("def_epa_allowed_play"),
            pl.col("success").mean().alias("def_success_rate_allowed"),
        )
        .rename({"defteam": "team"})
    )

    return off.join(deff, on=["game_id", "team"], how="inner")


def add_rolling_form(long: pl.DataFrame, epa_splits: pl.DataFrame) -> pl.DataFrame:
    """Join per-game team stats onto the long table, then compute shifted
    rolling means so each row only reflects form BEFORE that game."""
    long = long.join(epa_splits, on=["game_id", "team"], how="left")
    long = long.sort(["team", "season", "week"])

    stat_cols = [
        "margin", "off_epa_play", "off_success_rate", "off_pass_epa",
        "off_rush_epa", "def_epa_allowed_play", "def_success_rate_allowed",
    ]

    exprs = []
    for col in stat_cols:
        shifted = pl.col(col).shift(1).over("team")
        for window in config.ROLLING_WINDOWS:
            exprs.append(
                shifted.rolling_mean(window_size=window, min_samples=1).over("team").alias(f"{col}_roll{window}")
            )
        exprs.append(
            shifted.ewm_mean(half_life=config.EWMA_HALFLIFE_GAMES, min_samples=1).over("team").alias(f"{col}_ewm")
        )

    long = long.with_columns(exprs)

    games_played = pl.col("margin").cum_count().shift(1).over("team").fill_null(0).alias("games_played_career")
    season_games_played = (
        pl.col("margin").cum_count().shift(1).over(["team", "season"]).fill_null(0).alias("games_played_season")
    )
    long = long.with_columns(games_played, season_games_played)
    return long


def build_modeling_table(games: pl.DataFrame, team_long_form: pl.DataFrame, ratings: pl.DataFrame) -> pl.DataFrame:
    """Wide game-level table: one row per game, home/away form features as
    *_diff columns (home - away), plus power ratings and market lines."""
    form_cols = [c for c in team_long_form.columns if c.endswith(("_roll4", "_roll8", "_roll17", "_ewm"))]
    form_cols += ["games_played_career", "games_played_season"]

    home_form = team_long_form.filter(pl.col("is_home") == 1).select(["game_id", "team"] + form_cols)
    away_form = team_long_form.filter(pl.col("is_home") == 0).select(["game_id", "team"] + form_cols)

    home_form = home_form.rename({c: f"home_{c}" for c in form_cols})
    away_form = away_form.rename({c: f"away_{c}" for c in form_cols})

    df = games.join(home_form.drop("team"), on="game_id", how="left")
    df = df.join(away_form.drop("team"), on="game_id", how="left")

    df = df.join(
        ratings.select(["game_id", "home_elo_pre", "away_elo_pre", "elo_diff"]),
        on="game_id",
        how="left",
    )

    diff_exprs = []
    for c in form_cols:
        diff_exprs.append((pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(f"{c}_diff"))
    df = df.with_columns(diff_exprs)

    df = df.with_columns(
        (pl.col("home_rest") - pl.col("away_rest")).alias("rest_diff"),
    )

    return df.sort(["season", "week", "game_id"])
