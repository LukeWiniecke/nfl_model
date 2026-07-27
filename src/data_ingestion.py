"""Pull raw data from nflverse (via nflreadpy) and cache it locally as parquet.

nflreadpy already handles remote caching of the underlying release files, but we
also snapshot what we used to data/raw/ so the pipeline is reproducible offline
and so cleaning/feature steps don't need network access on every run.
"""

import nflreadpy as nfl
import polars as pl

from . import config
from .utils import get_logger

logger = get_logger(__name__)

SEASONS = list(range(config.FIRST_SEASON, config.LAST_SEASON + 1))


def _cache_path(name: str) -> "config.Path":
    return config.NFLVERSE_RAW_DIR / f"{name}.parquet"


def load_schedules(force_refresh: bool = False) -> pl.DataFrame:
    """Game-level schedule with results and closing market lines/odds."""
    path = _cache_path("schedules")
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached schedules from {path}")
        return pl.read_parquet(path)

    logger.info(f"Downloading schedules for seasons {SEASONS[0]}-{SEASONS[-1]}")
    df = nfl.load_schedules(seasons=SEASONS)
    df = df.filter(pl.col("game_type").is_in(config.GAME_TYPES_KEPT))
    df.write_parquet(path)
    return df


def load_pbp(force_refresh: bool = False) -> pl.DataFrame:
    """Play-by-play, trimmed to the columns needed for EPA/success-rate features."""
    path = _cache_path("pbp")
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached pbp from {path}")
        return pl.read_parquet(path)

    keep_cols = [
        "game_id",
        "season",
        "week",
        "season_type",
        "home_team",
        "away_team",
        "posteam",
        "defteam",
        "play_type",
        "pass",
        "rush",
        "epa",
        "success",
        "penalty",
    ]

    logger.info(f"Downloading play-by-play for seasons {SEASONS[0]}-{SEASONS[-1]} (this can take a while)")
    frames = []
    for season in SEASONS:
        df = nfl.load_pbp(seasons=[season])
        available = [c for c in keep_cols if c in df.columns]
        df = df.select(available)
        df = df.filter(pl.col("epa").is_not_null() & pl.col("posteam").is_not_null())
        frames.append(df)
    pbp = pl.concat(frames, how="diagonal_relaxed")
    pbp.write_parquet(path)
    return pbp


def load_team_stats(force_refresh: bool = False) -> pl.DataFrame:
    """Weekly team-level box score aggregates (used for a few features pbp doesn't
    cheaply give us, e.g. fumbles/penalties totals)."""
    path = _cache_path("team_stats")
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached team_stats from {path}")
        return pl.read_parquet(path)

    logger.info(f"Downloading team stats for seasons {SEASONS[0]}-{SEASONS[-1]}")
    df = nfl.load_team_stats(seasons=SEASONS)
    df.write_parquet(path)
    return df


def load_all(force_refresh: bool = False) -> dict[str, pl.DataFrame]:
    return {
        "schedules": load_schedules(force_refresh=force_refresh),
        "pbp": load_pbp(force_refresh=force_refresh),
        "team_stats": load_team_stats(force_refresh=force_refresh),
    }


if __name__ == "__main__":
    data = load_all()
    for name, df in data.items():
        print(name, df.shape)
