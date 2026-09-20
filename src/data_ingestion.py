"""Pull raw data from nflverse (via nflreadpy) and cache it locally as parquet.

nflreadpy already handles remote caching of the underlying release files, but we
also snapshot what we used to data/raw/ so the pipeline is reproducible offline
and so cleaning/feature steps don't need network access on every run.
"""

from pathlib import Path

import nflreadpy as nfl
import polars as pl

from . import config
from .utils import get_logger, standardize_team

logger = get_logger(__name__)

SNAP_COUNTS_FIRST_SEASON = 2012  # nflreadpy's own floor for this dataset
OFFICIALS_FIRST_SEASON = 2015  # nflreadpy's own floor for this dataset
NEXTGEN_STATS_FIRST_SEASON = 2016  # nflreadpy's own floor for this dataset

SEASONS = list(range(config.FIRST_SEASON, config.LAST_SEASON + 1))

# External reference file (not part of the nflverse pull) used only for a
# couple of context columns nflverse doesn't have: prime-time and playing
# surface. Its franchise IDs already collapse relocations to one per
# franchise (like nflverse's own standardize_team), just spelled differently
# for two teams.
SPREADSPOKE_CSV_PATH = Path(r"C:\Users\USER\Downloads\spreadspoke_enhanced.csv")
SPREADSPOKE_TEAM_MAP = {"LVR": "LV", "LAR": "LA"}


def _cache_path(name: str) -> "config.Path":
    return config.NFLVERSE_RAW_DIR / f"{name}.parquet"


def _load_seasons_safe(name: str, loader_fn, seasons: list[int]) -> pl.DataFrame | None:
    """Call an nflreadpy loader over `seasons`, dropping the newest season(s)
    and retrying if nflreadpy rejects them. Several nflreadpy loaders
    validate the requested seasons against their OWN idea of the "current"
    season (raising ValueError), which can lag config.LAST_SEASON right
    after that constant gets bumped for a new year; others just try to
    download a per-season release file that doesn't exist yet and raise
    ConnectionError (a 404) instead. Either way, before that season's data
    has actually been published there'd be nothing to return anyway.
    Returns None (rather than raising) if no season in the range works.
    """
    seasons = list(seasons)
    while seasons:
        try:
            return loader_fn(seasons=seasons)
        except (ValueError, ConnectionError) as e:
            logger.warning(f"{name} unavailable for season {seasons[-1]} ({e}), dropping it and retrying")
            seasons = seasons[:-1]
    return None


def load_schedules(force_refresh: bool = False) -> pl.DataFrame:
    """Game-level schedule with results and closing market lines/odds."""
    path = _cache_path("schedules")
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached schedules from {path}")
        return pl.read_parquet(path)

    logger.info(f"Downloading schedules for seasons {SEASONS[0]}-{SEASONS[-1]}")
    df = _load_seasons_safe("schedules", nfl.load_schedules, SEASONS)
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
        # Penalties. `penalty` alone (all that was kept until 2026-09-20) is
        # not usable on its own: it flags that a penalty occurred on the play,
        # but `posteam` is the OFFENSE and ~41% of penalties are on the
        # defense, so there was no way to tell who committed it. penalty_team
        # gives the attribution, penalty_yards the magnitude, penalty_type the
        # pre-snap/discipline split. All three are non-null on every penalty
        # play in the seasons checked. Adding them invalidates any pbp cache
        # written before that date -- delete data/raw/nflverse/pbp.parquet
        # once after upgrading, or pass --refresh.
        "penalty",
        "penalty_team",
        "penalty_yards",
        "penalty_type",
    ]

    logger.info(f"Downloading play-by-play for seasons {SEASONS[0]}-{SEASONS[-1]} (this can take a while)")
    frames = []
    for season in SEASONS:
        try:
            df = nfl.load_pbp(seasons=[season])
        except (ValueError, ConnectionError) as e:
            logger.warning(f"pbp unavailable for season {season} ({e}), skipping")
            continue
        available = [c for c in keep_cols if c in df.columns]
        df = df.select(available)
        df = df.filter(pl.col("epa").is_not_null() & pl.col("posteam").is_not_null())
        frames.append(df)
    pbp = pl.concat(frames, how="diagonal_relaxed")
    pbp.write_parquet(path)
    return pbp


def load_player_stats(force_refresh: bool = False) -> pl.DataFrame:
    """Weekly player-level stats (used for starting-QB identification and
    QB-level EPA/continuity features)."""
    path = _cache_path("player_stats")
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached player_stats from {path}")
        return pl.read_parquet(path)

    logger.info(f"Downloading player stats for seasons {SEASONS[0]}-{SEASONS[-1]}")
    df = _load_seasons_safe("player_stats", nfl.load_player_stats, SEASONS)
    keep_cols = [
        "player_id", "player_display_name", "position", "season", "week",
        "season_type", "game_id", "team", "attempts", "passing_epa",
    ]
    df = df.select([c for c in keep_cols if c in df.columns])
    df.write_parquet(path)
    return df


def load_team_stats(force_refresh: bool = False) -> pl.DataFrame:
    """Weekly team-level box score aggregates (used for a few features pbp doesn't
    cheaply give us, e.g. fumbles/penalties totals)."""
    path = _cache_path("team_stats")
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached team_stats from {path}")
        return pl.read_parquet(path)

    logger.info(f"Downloading team stats for seasons {SEASONS[0]}-{SEASONS[-1]}")
    df = _load_seasons_safe("team_stats", nfl.load_team_stats, SEASONS)
    df.write_parquet(path)
    return df


def load_injuries(force_refresh: bool = False) -> pl.DataFrame:
    """Weekly injury reports (one row per player per week -- already the
    final in-week report, not daily practice snapshots)."""
    path = _cache_path("injuries")
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached injuries from {path}")
        return pl.read_parquet(path)

    logger.info(f"Downloading injury reports for seasons {SEASONS[0]}-{SEASONS[-1]}")
    df = _load_seasons_safe("injuries", nfl.load_injuries, SEASONS)
    keep_cols = ["season", "week", "game_type", "team", "gsis_id", "position", "report_status"]
    df = df.select([c for c in keep_cols if c in df.columns])
    df = df.with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
        # nflverse's injuries table still uses pre-relocation codes (OAK/SD/STL)
        # for seasons before those franchises moved; `games`' team columns are
        # already standardized (cleaning.clean_schedules), so without this the
        # injury feature silently joins to nothing for every Raiders/Chargers/
        # Rams game before 2020/2017/2016 respectively.
        standardize_team(pl.col("team")).alias("team"),
    )
    df.write_parquet(path)
    return df


def load_snap_counts(force_refresh: bool = False) -> pl.DataFrame:
    """Per-player-per-game snap share (offense/defense/special-teams %),
    used to weight the injury feature by how big a role the injured player
    actually plays, not just their position label. Only available from 2012
    on -- nflreadpy's own floor for this dataset, 2 seasons later than
    everything else this project uses."""
    path = _cache_path("snap_counts")
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached snap counts from {path}")
        return pl.read_parquet(path)

    seasons = [s for s in SEASONS if s >= SNAP_COUNTS_FIRST_SEASON]
    logger.info(f"Downloading snap counts for seasons {seasons[0]}-{seasons[-1]}")
    df = _load_seasons_safe("snap_counts", nfl.load_snap_counts, seasons)
    if df is None:
        return pl.DataFrame(
            schema={"season": pl.Int32, "week": pl.Int32, "team": pl.Utf8, "pfr_player_id": pl.Utf8, "offense_pct": pl.Float64, "defense_pct": pl.Float64}
        )
    keep_cols = ["season", "week", "team", "pfr_player_id", "offense_pct", "defense_pct"]
    df = df.select([c for c in keep_cols if c in df.columns])
    df = df.with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
        standardize_team(pl.col("team")).alias("team"),
    )
    df.write_parquet(path)
    return df


def load_officials(force_refresh: bool = False) -> pl.DataFrame:
    """Weekly officiating crew assignments -- one row per official per game.
    Only the head Referee (position == "Referee", the crew's traditional
    identifying member -- the rest of the 7-person crew rotates more) is
    used downstream (feature_engineering.compute_referee_features).
    Compiled from completed games, same as player_stats -- there's no
    advance/pre-game feed of a future assignment. Only available from 2015
    on -- nflreadpy's own floor for this dataset (a hard floor, not just a
    "too new" ceiling like most other loaders here, so _load_seasons_safe's
    trim-from-the-end retry can't work its way past it on its own -- it'd
    walk every season down to config.FIRST_SEASON and still fail)."""
    path = _cache_path("officials")
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached officials from {path}")
        return pl.read_parquet(path)

    seasons = [s for s in SEASONS if s >= OFFICIALS_FIRST_SEASON]
    logger.info(f"Downloading officiating crews for seasons {seasons[0]}-{seasons[-1]}")
    df = _load_seasons_safe("officials", nfl.load_officials, seasons)
    if df is None:
        return pl.DataFrame(
            schema={"game_id": pl.Utf8, "official_id": pl.Utf8, "position": pl.Utf8, "season": pl.Int32, "week": pl.Int32}
        )
    keep_cols = ["game_id", "official_id", "position", "season", "week"]
    df = df.select([c for c in keep_cols if c in df.columns])
    df = df.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    df.write_parquet(path)
    return df


def load_nextgen_stats_passing(force_refresh: bool = False) -> pl.DataFrame:
    """Weekly QB-level Next Gen Stats (tracking-derived passing metrics).
    Only completion_percentage_above_expectation (CPOE) is kept -- a QB-
    accuracy signal considered cleaner than raw completion% or EPA/attempt
    since it's adjusted for throw difficulty (depth, coverage tightness,
    etc.), something the pbp-derived features elsewhere in this pipeline
    don't capture at all. Only available from 2016 on -- nflreadpy's own
    floor AND ceiling for this dataset (same shape as load_officials's
    2015 floor), so seasons outside that range are filtered BEFORE calling
    nflreadpy, not left for _load_seasons_safe's trim-from-the-end retry to
    fail its way through. player_gsis_id already matches player_stats' own
    player_id format directly (both are nflverse's canonical gsis scheme,
    confirmed by direct inspection) -- no crosswalk needed, unlike
    officials' old_game_id situation."""
    path = _cache_path("nextgen_passing")
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached Next Gen Stats (passing) from {path}")
        return pl.read_parquet(path)

    seasons = [s for s in SEASONS if s >= NEXTGEN_STATS_FIRST_SEASON]
    logger.info(f"Downloading Next Gen Stats (passing) for seasons {seasons[0]}-{seasons[-1]}")
    df = _load_seasons_safe(
        "nextgen_stats_passing",
        lambda seasons: nfl.load_nextgen_stats(stat_type="passing", seasons=seasons),
        seasons,
    )
    if df is None:
        return pl.DataFrame(
            schema={
                "player_gsis_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
                "completion_percentage_above_expectation": pl.Float64,
            }
        )
    keep_cols = ["player_gsis_id", "season", "week", "completion_percentage_above_expectation"]
    df = df.select([c for c in keep_cols if c in df.columns])
    df = df.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    df.write_parquet(path)
    return df


def load_current_qb_depth_chart(force_refresh: bool = False) -> pl.DataFrame:
    """Each team's CURRENT presumed starting QB, straight from nflverse's
    depth chart data for config.LAST_SEASON. Used only to refine
    feature_engineering.compute_qb_form's live/upcoming-game carry-forward
    assumption -- never as a walk-forward training feature: for every
    PLAYED historical game, the true starter is already known exactly from
    real pass-attempt data, so a pre-game depth-chart guess can't improve
    anything there. This only matters for games that haven't been played
    yet, where the model otherwise has to assume "whoever started this
    team's most recent game is still starting" -- true most of the time,
    wrong exactly when a trade/retirement/rookie promotion happened since.

    nflreadpy's load_depth_charts returns TWO fundamentally different
    schemas depending on the season, confirmed by direct inspection: past,
    fully-archived seasons come back as a per-(season, week) snapshot with
    columns like "week"/"depth_team"/"club_code"; the CURRENT in-progress
    season instead comes from a live, continuously-updated feed (multiple
    snapshots per DAY) keyed by a "dt" timestamp instead of week, with
    columns like "dt"/"team"/"pos_abb"/"pos_rank" -- no week granularity at
    all. This function only ever wants "who's the CURRENT starter," so it
    branches on which schema came back and takes the single most recent
    snapshot per team either way.
    """
    path = _cache_path("current_qb_depth_chart")
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached current QB depth chart from {path}")
        return pl.read_parquet(path)

    logger.info(f"Downloading current depth chart (season {config.LAST_SEASON})")
    try:
        df = nfl.load_depth_charts(seasons=[config.LAST_SEASON])
    except (ValueError, ConnectionError) as e:
        logger.warning(f"Depth chart unavailable for {config.LAST_SEASON} ({e}), skipping")
        return pl.DataFrame(schema={"team": pl.Utf8, "depth_chart_qb_id": pl.Utf8})

    if "dt" in df.columns:
        # Live feed: one row per (team, position, snapshot timestamp).
        qb = df.filter((pl.col("pos_abb") == "QB") & pl.col("gsis_id").is_not_null() & (pl.col("pos_rank") == 1))
        latest_dt = qb.group_by("team").agg(pl.col("dt").max().alias("_latest_dt"))
        qb = qb.join(latest_dt, on="team").filter(pl.col("dt") == pl.col("_latest_dt"))
        out = qb.with_columns(standardize_team(pl.col("team")).alias("team")).select(
            "team", pl.col("gsis_id").alias("depth_chart_qb_id")
        )
    else:
        # Archived schema: one row per (season, week, club_code).
        qb = df.filter(
            (pl.col("position") == "QB") & (pl.col("depth_team") == "1") & pl.col("week").is_not_null()
        )
        qb = qb.sort(["club_code", "season", "week"])
        latest = qb.group_by("club_code").agg(pl.col("gsis_id").last().alias("depth_chart_qb_id"))
        out = latest.rename({"club_code": "team"}).with_columns(standardize_team(pl.col("team")).alias("team"))

    out = out.unique(subset=["team"], keep="first")
    out.write_parquet(path)
    return out


def load_player_crosswalk(force_refresh: bool = False) -> pl.DataFrame:
    """Static gsis_id <-> pfr_id crosswalk -- injuries identifies players by
    gsis_id, snap counts by pfr_id, and nflverse doesn't share one ID scheme
    between the two, so this is the join key that connects them."""
    path = _cache_path("player_crosswalk")
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached player crosswalk from {path}")
        return pl.read_parquet(path)

    logger.info("Downloading player ID crosswalk")
    df = nfl.load_players().select(["gsis_id", "pfr_id"]).filter(
        pl.col("gsis_id").is_not_null() & pl.col("pfr_id").is_not_null()
    )
    df.write_parquet(path)
    return df


def load_spreadspoke_context(force_refresh: bool = False) -> pl.DataFrame | None:
    """Day-of-week/prime-time/surface context from the Spreadspoke reference
    file, keyed by (gameday, home_team, away_team) for joining onto
    nflverse's games table. Cached locally so later runs don't depend on the
    original CSV's location. Returns None (feature skipped, not an error) if
    the source CSV isn't present and no cache exists yet.
    """
    path = config.BETTING_RAW_DIR / "spreadspoke_context.parquet"
    if path.exists() and not force_refresh:
        logger.info(f"Loading cached spreadspoke context from {path}")
        return pl.read_parquet(path)

    if not SPREADSPOKE_CSV_PATH.exists():
        logger.warning(f"Spreadspoke CSV not found at {SPREADSPOKE_CSV_PATH}, skipping its context features")
        return None

    logger.info(f"Extracting prime-time/surface context from {SPREADSPOKE_CSV_PATH}")
    df = pl.read_csv(
        SPREADSPOKE_CSV_PATH,
        columns=["schedule_date", "home_team_id", "away_team_id", "prime_time_approx", "surface_grass"],
    )
    df = df.with_columns(
        pl.col("schedule_date").str.to_date().alias("gameday"),
        pl.col("home_team_id").replace(SPREADSPOKE_TEAM_MAP).alias("home_team"),
        pl.col("away_team_id").replace(SPREADSPOKE_TEAM_MAP).alias("away_team"),
        pl.col("prime_time_approx").cast(pl.Int8),
    )
    df = df.select(["gameday", "home_team", "away_team", "prime_time_approx", "surface_grass"])
    df.write_parquet(path)
    return df


def load_all(force_refresh: bool = False) -> dict[str, pl.DataFrame]:
    return {
        "schedules": load_schedules(force_refresh=force_refresh),
        "pbp": load_pbp(force_refresh=force_refresh),
        "team_stats": load_team_stats(force_refresh=force_refresh),
        "player_stats": load_player_stats(force_refresh=force_refresh),
        "injuries": load_injuries(force_refresh=force_refresh),
        "snap_counts": load_snap_counts(force_refresh=force_refresh),
        "player_crosswalk": load_player_crosswalk(force_refresh=force_refresh),
        "officials": load_officials(force_refresh=force_refresh),
        "nextgen_passing": load_nextgen_stats_passing(force_refresh=force_refresh),
    }


if __name__ == "__main__":
    data = load_all()
    for name, df in data.items():
        print(name, df.shape)
