"""Per-game team trend data for the dashboard's Matchup & trends tab.

Display-only: nothing here feeds the model. It is built from the cached
play-by-play and schedule files, so it needs no pipeline changes and works
for any played game, including the current season.

Opponent adjustment: a game's raw EPA/play is compared with how the
opponent performed in its OTHER games that season (leave-one-out season
average). Offense is compared with what the opponent's defense allowed
elsewhere; defense with what the opponent's offense produced elsewhere.
Positive adjusted offense / negative adjusted defense is good for the team.
The leave-one-out average uses the whole season, including games after the
one shown -- fine for looking back, and not a model feature.
"""

from __future__ import annotations

import polars as pl

from . import config
from .utils import standardize_team

# metric key -> (label, raw column, side, lower_is_better)
METRICS = {
    "off_epa": ("Offense EPA/play", "off_epa", "off", False),
    "off_pass_epa": ("Pass offense EPA/play", "off_pass_epa", "off", False),
    "off_rush_epa": ("Rush offense EPA/play", "off_rush_epa", "off", False),
    "def_epa": ("Defense EPA/play allowed", "def_epa", "def", True),
    "def_pass_epa": ("Pass defense EPA/play allowed", "def_pass_epa", "def", True),
    "def_rush_epa": ("Rush defense EPA/play allowed", "def_rush_epa", "def", True),
    "margin": ("Point margin", "margin", "off", False),
}

# raw column on the opponent that a given column is measured against
_OPP_COUNTERPART = {
    "off_epa": "def_epa",
    "off_pass_epa": "def_pass_epa",
    "off_rush_epa": "def_rush_epa",
    "def_epa": "off_epa",
    "def_pass_epa": "off_pass_epa",
    "def_rush_epa": "off_rush_epa",
}


def _play_stats() -> pl.DataFrame:
    pbp = pl.read_parquet(config.RAW_DIR / "nflverse" / "pbp.parquet")
    plays = pbp.filter(pl.col("play_type").is_in(["run", "pass"]) & pl.col("epa").is_not_null())
    off = plays.group_by(["game_id", "posteam"]).agg(
        pl.col("epa").mean().alias("off_epa"),
        pl.col("epa").filter(pl.col("pass") == 1).mean().alias("off_pass_epa"),
        pl.col("epa").filter(pl.col("rush") == 1).mean().alias("off_rush_epa"),
    ).rename({"posteam": "team"})
    deff = plays.group_by(["game_id", "defteam"]).agg(
        pl.col("epa").mean().alias("def_epa"),
        pl.col("epa").filter(pl.col("pass") == 1).mean().alias("def_pass_epa"),
        pl.col("epa").filter(pl.col("rush") == 1).mean().alias("def_rush_epa"),
    ).rename({"defteam": "team"})
    return off.join(deff, on=["game_id", "team"], how="inner")


def _schedule_long() -> pl.DataFrame:
    """One row per team per scheduled game, with coach and starting QB."""
    s = pl.read_parquet(config.RAW_DIR / "nflverse" / "schedules.parquet")
    for col in ("home_coach", "away_coach", "home_qb_name", "away_qb_name"):
        if col not in s.columns:
            s = s.with_columns(pl.lit(None, dtype=pl.String).alias(col))
    # Schedules still carry OAK/SD/STL; the play-by-play uses the current codes.
    s = s.with_columns(
        standardize_team(pl.col("home_team")).alias("home_team"),
        standardize_team(pl.col("away_team")).alias("away_team"),
    )
    frames = []
    for me, them, sign in (("home", "away", 1), ("away", "home", -1)):
        frames.append(
            s.select(
                "game_id", "season", "week", "game_type",
                pl.col("gameday").str.to_date().alias("gameday"),
                pl.col(f"{me}_team").alias("team"),
                pl.col(f"{them}_team").alias("opponent"),
                pl.lit(me == "home").alias("is_home"),
                pl.col(f"{me}_score").alias("points_for"),
                pl.col(f"{them}_score").alias("points_against"),
                pl.col(f"{me}_coach").alias("coach"),
                pl.col(f"{me}_qb_name").alias("qb"),
                (pl.col("spread_line") * sign).alias("spread_for_team"),
            )
        )
    return pl.concat(frames).with_columns((pl.col("points_for") - pl.col("points_against")).alias("margin"))


def team_game_trends() -> pl.DataFrame:
    """One row per team per PLAYED game: raw and opponent-adjusted stats."""
    long = _schedule_long().filter(pl.col("points_for").is_not_null() & (pl.col("season") >= config.FIRST_SEASON))
    long = long.join(_play_stats(), on=["game_id", "team"], how="left")

    # Leave-one-out season mean of every team's raw stats; the opponent's
    # version of it is what each game is compared against.
    raw = list(_OPP_COUNTERPART)
    loo = long.select(
        "game_id", "team", "season",
        *[pl.col(c) for c in raw],
    ).with_columns(
        *[
            ((pl.col(c).sum().over(["team", "season"]) - pl.col(c)) / (pl.col(c).count().over(["team", "season"]) - 1)).alias(f"_loo_{c}")
            for c in raw
        ]
    ).select("game_id", "team", *[f"_loo_{c}" for c in raw])

    opp = loo.rename({"team": "opponent", **{f"_loo_{c}": f"_opp_{c}" for c in raw}})
    long = long.join(opp, on=["game_id", "opponent"], how="left")
    long = long.with_columns(
        *[
            (pl.col(c) - pl.col(f"_opp_{_OPP_COUNTERPART[c]}")).alias(f"{c}_adj")
            for c in raw
        ]
    ).with_columns(pl.col("margin").alias("margin_adj"))
    return long.sort(["team", "gameday"]).drop([c for c in long.columns if c.startswith("_opp_")])


def stints(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Consecutive runs of the same coach (or QB) for a team's games, with
    games and record. `df` must be one team's games, sorted by date."""
    d = df.filter(pl.col(col).is_not_null()).with_columns(
        (pl.col(col) != pl.col(col).shift(1)).fill_null(True).cum_sum().alias("_run")
    )
    return (
        d.group_by("_run", maintain_order=True)
        .agg(
            pl.col(col).first().alias(col),
            pl.col("gameday").first().alias("from"),
            pl.col("gameday").last().alias("to"),
            pl.len().alias("games"),
            (pl.col("margin") > 0).sum().alias("wins"),
            (pl.col("margin") < 0).sum().alias("losses"),
            pl.col("margin").mean().round(1).alias("avg_margin"),
        )
        .drop("_run")
    )
