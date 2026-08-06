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
from .utils import get_logger, standardize_team

logger = get_logger(__name__)


def compute_game_epa_splits(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team): offensive EPA/play, success rate, pass EPA,
    rush EPA, defensive EPA/success allowed while that team was on defense,
    and off_plays (offensive snap count -- run/pass plays run by that team
    in that game, a standard pace proxy: more snaps means more scoring
    opportunities, relevant to the totals model specifically).
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
            pl.len().alias("off_plays"),
        )
        .rename({"posteam": "team"})
    )

    deff = (
        plays.group_by(["game_id", "defteam"])
        .agg(
            pl.col("epa").mean().alias("def_epa_allowed_play"),
            pl.col("success").mean().alias("def_success_rate_allowed"),
            pl.col("epa").filter(pl.col("pass") == 1).mean().alias("def_pass_epa_allowed"),
            pl.col("epa").filter(pl.col("rush") == 1).mean().alias("def_rush_epa_allowed"),
        )
        .rename({"defteam": "team"})
    )

    return off.join(deff, on=["game_id", "team"], how="inner")


def compute_qb_form(
    player_stats: pl.DataFrame, team_schedule: pl.DataFrame, depth_chart_starters: pl.DataFrame | None = None
) -> pl.DataFrame:
    """One row per (game_id, team): the starting QB's leakage-safe rolling
    EPA/dropback entering this game, tracked per QB across team changes (a
    trade/free-agent move carries their history with them, not the team's),
    plus a continuity flag for whether this team's starter changed since its
    last game.

    Starter = the QB with the most pass attempts for that team in that game.
    Team-level rolling EPA only reflects a starter change gradually, several
    games after the fact; this is meant to catch it immediately. Treating the
    starter's IDENTITY as known pre-game is reasonable (real starters are
    public via injury reports before kickoff) -- but exactly like every other
    feature here, their ROLLING PERFORMANCE only ever uses starts strictly
    before the one being predicted.

    `team_schedule` is every (game_id, team, season, week, margin) in
    scope -- played AND upcoming (`margin` is the team's own game margin,
    null exactly when the game hasn't been played yet, same convention as
    everywhere else in this pipeline). This matters because player_stats
    (and therefore everything computed above) only has rows for games that
    have actually been played AND have a recorded QB pass attempt: without
    this, an upcoming game simply has no row here at all, silently goes
    null after the join in add_rolling_form, and gets median-imputed at
    prediction time -- meaning every live prediction had ZERO real QB
    signal, discovered directly by inspecting a live upcoming game's
    modeling-table row. The fix: for any (game_id, team) not covered
    above -- a genuinely unplayed game, OR (rarer) a played game with no
    qualifying QB row in player_stats, e.g. a handful of old pre-relocation
    Raiders games -- carry forward that team's most recent PLAYED game's
    starter, using THEIR OWN rolling EPA/attribute as of that game
    (postgame, i.e. unshifted -- correct for "entering the next game"
    since nothing has happened for that team since). Continuity is assumed
    true for a carried-forward row (an actual change only becomes visible
    here once the new starter has played) -- compute_qb_injury_features is
    the pre-game substitute for "the presumed starter is actually hurt."
    Confirmed directly that this fallback, applied to those rare data-gap
    games too, is a net positive (not just harmless): forcing those few
    games to null/median-imputed instead measurably WORSENED aggregate MAE.

    `depth_chart_starters` (data_ingestion.load_current_qb_depth_chart), if
    given, refines that carry-forward guess for GENUINELY UNPLAYED games
    only (`margin` null) -- it has zero effect on any played/backtest row,
    including the data-gap ones above: the true starter there is either
    already known exactly from real pass-attempt data, or (for a gap game)
    the existing naive carry is a reasonable, already-adequate fallback
    that a depth chart reflecting TODAY'S roster has no business
    overriding. When the depth chart names a DIFFERENT player than
    "whoever started this team's most recent game" for a genuinely
    upcoming game -- a trade, retirement, or rookie promotion the naive
    carry can't see -- that player's own latest rolling stats are used
    instead (if they have any prior starts in our data; if they're a true
    rookie with zero NFL history here, this correctly falls back to
    null/median-imputed rather than guessing), and continuity is set to 0
    to flag the change.
    """
    qb = player_stats.filter((pl.col("position") == "QB") & (pl.col("attempts") > 0))
    qb = qb.with_columns(
        standardize_team(pl.col("team")).alias("team"),
        (pl.col("passing_epa") / pl.col("attempts")).alias("qb_epa_per_att"),
    )

    starter = (
        qb.sort("attempts", descending=True)
        .group_by(["game_id", "team"], maintain_order=True)
        .first()
        .select(["game_id", "team", "season", "week", "player_id", "qb_epa_per_att"])
    )

    starter = starter.sort(["player_id", "season", "week"])
    shifted = pl.col("qb_epa_per_att").shift(1).over("player_id")
    starter = starter.with_columns(
        shifted.rolling_mean(window_size=config.ROLLING_WINDOWS[0], min_samples=1)
        .over("player_id").alias("qb_epa_per_att_roll8"),
        shifted.ewm_mean(half_life=config.EWMA_HALFLIFE_GAMES, min_samples=1)
        .over("player_id").alias("qb_epa_per_att_ewm"),
        # Same two stats, unshifted (as-of-and-including this game) -- never
        # used as this row's own feature (that would leak), only to carry
        # forward into whatever this player's team's NEXT game turns out to
        # be, played or not.
        pl.col("qb_epa_per_att").rolling_mean(window_size=config.ROLLING_WINDOWS[0], min_samples=1)
        .over("player_id").alias("_qb_epa_per_att_roll8_postgame"),
        pl.col("qb_epa_per_att").ewm_mean(half_life=config.EWMA_HALFLIFE_GAMES, min_samples=1)
        .over("player_id").alias("_qb_epa_per_att_ewm_postgame"),
    )

    starter = starter.sort(["team", "season", "week"])
    prev_starter = pl.col("player_id").shift(1).over("team")
    starter = starter.with_columns(
        (pl.col("player_id") == prev_starter).cast(pl.Int8).fill_null(1).alias("qb_continuity")
    )

    played_out = starter.select(["game_id", "team", "qb_epa_per_att_roll8", "qb_epa_per_att_ewm", "qb_continuity"])

    unplayed = team_schedule.join(played_out, on=["game_id", "team"], how="anti")
    if unplayed.height == 0:
        return played_out

    carry = (
        starter.select(
            "team", "season", "week", "player_id",
            pl.col("_qb_epa_per_att_roll8_postgame").alias("qb_epa_per_att_roll8"),
            pl.col("_qb_epa_per_att_ewm_postgame").alias("qb_epa_per_att_ewm"),
        )
        .with_columns((pl.col("season") * 100 + pl.col("week")).alias("_time_key"))
        .drop(["season", "week"])
        .sort(["team", "_time_key"])
    )
    unplayed = unplayed.with_columns(
        (pl.col("season") * 100 + pl.col("week")).alias("_time_key")
    ).sort(["team", "_time_key"])

    carried = unplayed.join_asof(carry, on="_time_key", by="team", strategy="backward")

    if depth_chart_starters is not None and depth_chart_starters.height > 0:
        # Each player's own latest rolling stats, regardless of team --
        # looked up ONLY when the depth chart names someone other than the
        # naive per-team carry above.
        player_latest = (
            starter.sort(["player_id", "season", "week"])
            .group_by("player_id", maintain_order=True)
            .last()
            .select(
                pl.col("player_id").alias("_dc_lookup_id"),
                pl.col("_qb_epa_per_att_roll8_postgame").alias("_dc_roll8"),
                pl.col("_qb_epa_per_att_ewm_postgame").alias("_dc_ewm"),
            )
        )
        carried = carried.join(
            depth_chart_starters.rename({"depth_chart_qb_id": "_dc_player_id"}), on="team", how="left"
        )
        carried = carried.join(player_latest, left_on="_dc_player_id", right_on="_dc_lookup_id", how="left")

        # margin.is_null() (genuinely unplayed) gates this deliberately: the
        # rest of the "unplayed" bucket also includes PLAYED games missing
        # a QB pass-attempt row in player_stats (a real, if rare, data gap)
        # -- a depth chart reflecting TODAY'S roster has no business
        # overriding one of those, found directly when it silently
        # rewrote 15 played historical games (some from 2012) to the
        # CURRENT starter's stats.
        use_dc = (
            pl.col("_dc_player_id").is_not_null()
            & (pl.col("_dc_player_id") != pl.col("player_id"))
            & pl.col("margin").is_null()
        )
        carried = carried.with_columns(
            pl.when(use_dc).then(pl.col("_dc_roll8")).otherwise(pl.col("qb_epa_per_att_roll8"))
            .alias("qb_epa_per_att_roll8"),
            pl.when(use_dc).then(pl.col("_dc_ewm")).otherwise(pl.col("qb_epa_per_att_ewm"))
            .alias("qb_epa_per_att_ewm"),
            pl.when(use_dc).then(pl.lit(0)).otherwise(pl.lit(1)).cast(pl.Int8).alias("qb_continuity"),
        )
    else:
        carried = carried.with_columns(pl.lit(1).cast(pl.Int8).alias("qb_continuity"))

    carried_out = carried.select(["game_id", "team", "qb_epa_per_att_roll8", "qb_epa_per_att_ewm", "qb_continuity"])

    return pl.concat([played_out, carried_out], how="diagonal_relaxed")


def compute_qb_ngs_form(player_stats: pl.DataFrame, ngs_passing: pl.DataFrame, team_schedule: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team): the starting QB's leakage-safe rolling
    completion percentage over expectation (CPOE, from nflreadpy's Next Gen
    Stats) entering this game. CPOE adjusts for throw difficulty (depth,
    coverage tightness, etc.) via player tracking data, something
    compute_qb_form's pbp-derived EPA/attempt can't see at all -- a
    different KIND of QB-quality signal, not a refinement of the existing
    one. Same starter-identification and carry-forward-into-upcoming-games
    structure as compute_qb_form (see its docstring for why the carry
    exists) -- kept as a separate function rather than merged into that one
    since NGS is a different data source with its own coverage floor.

    NGS only covers 2016 on (data_ingestion.NEXTGEN_STATS_FIRST_SEASON) --
    one year later than snap counts, six years later than the rest of this
    project's core data. Every 2010-2015 game goes null here, then 0.0
    (modeling._impute's fallback for an all-null training-pool column) for
    any walk-forward fold trained ONLY on those seasons -- meaning the
    earliest test seasons get zero benefit from this feature by
    construction, not a bug. Real learned signal only starts accumulating
    as 2016+ seasons enter the training pool.
    """
    qb = player_stats.filter((pl.col("position") == "QB") & (pl.col("attempts") > 0))
    qb = qb.with_columns(standardize_team(pl.col("team")).alias("team"))
    starter = (
        qb.sort("attempts", descending=True)
        .group_by(["game_id", "team"], maintain_order=True)
        .first()
        .select(["game_id", "team", "season", "week", "player_id"])
    )

    ngs = ngs_passing.rename({"player_gsis_id": "player_id"}).select(
        "player_id", "season", "week", "completion_percentage_above_expectation"
    )
    starter = starter.join(ngs, on=["player_id", "season", "week"], how="left")

    starter = starter.sort(["player_id", "season", "week"])
    shifted = pl.col("completion_percentage_above_expectation").shift(1).over("player_id")
    starter = starter.with_columns(
        shifted.ewm_mean(half_life=config.EWMA_HALFLIFE_GAMES, min_samples=1)
        .over("player_id").alias("qb_cpoe_ewm"),
        # Unshifted (as-of-and-including this game) -- never used as this
        # row's own feature, only to carry forward into this player's
        # team's next game, played or not (see compute_qb_form).
        pl.col("completion_percentage_above_expectation")
        .ewm_mean(half_life=config.EWMA_HALFLIFE_GAMES, min_samples=1)
        .over("player_id").alias("_qb_cpoe_ewm_postgame"),
    )

    played_out = starter.select(["game_id", "team", "qb_cpoe_ewm"])

    unplayed = team_schedule.join(played_out, on=["game_id", "team"], how="anti")
    if unplayed.height == 0:
        return played_out

    carry = (
        starter.select("team", "season", "week", pl.col("_qb_cpoe_ewm_postgame").alias("qb_cpoe_ewm"))
        .with_columns((pl.col("season") * 100 + pl.col("week")).alias("_time_key"))
        .drop(["season", "week"])
        .sort(["team", "_time_key"])
    )
    unplayed = unplayed.with_columns(
        (pl.col("season") * 100 + pl.col("week")).alias("_time_key")
    ).sort(["team", "_time_key"])

    carried = unplayed.join_asof(carry, on="_time_key", by="team", strategy="backward")
    carried_out = carried.select(["game_id", "team", "qb_cpoe_ewm"])

    return pl.concat([played_out, carried_out], how="diagonal_relaxed")


def add_rolling_form(long: pl.DataFrame, epa_splits: pl.DataFrame) -> pl.DataFrame:
    """Join per-game team stats onto the long table, then compute shifted
    rolling means so each row only reflects form BEFORE that game."""
    long = long.join(epa_splits, on=["game_id", "team"], how="left")
    long = long.sort(["team", "season", "week"])

    raw_stat_cols = [
        "margin", "off_epa_play", "off_success_rate", "off_pass_epa", "off_rush_epa",
        "def_epa_allowed_play", "def_success_rate_allowed", "def_pass_epa_allowed", "def_rush_epa_allowed",
        "off_plays",
    ]
    long = long.with_columns(_rolling_exprs(raw_stat_cols))

    games_played = pl.col("margin").cum_count().shift(1).over("team").fill_null(0).alias("games_played_career")
    season_games_played = (
        pl.col("margin").cum_count().shift(1).over(["team", "season"]).fill_null(0).alias("games_played_season")
    )
    long = long.with_columns(games_played, season_games_played)

    long = add_opponent_adjusted_form(long)
    return long


def _rolling_exprs(stat_cols: list[str]) -> list[pl.Expr]:
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
    return exprs


def add_opponent_adjusted_form(long: pl.DataFrame) -> pl.DataFrame:
    """Opponent-adjust each game's raw offense/defense production by the
    specific opponent's incoming (pre-game, leakage-safe) rolling quality,
    then roll THAT adjusted quantity the same way as the raw stats.

    The raw rolling stats (off_epa_play_roll8, etc.) don't know who was
    played -- a team that faced three soft defenses in a row looks better on
    them than it should. +0.10 EPA/play against a defense that's been
    allowing +0.20 is a below-average game in context; the same +0.10 against
    a defense allowing +0.05 is a strong one. This is a single-pass
    approximation (subtract the opponent's own incoming rolling average, not
    an iterative solve like DVOA), but it's a real correction the raw stats
    don't have.
    """
    opp_pre = long.select(
        "game_id", pl.col("team").alias("opponent"),
        pl.col("off_epa_play_roll8").alias("_opp_off_epa_play_pre"),
        pl.col("off_success_rate_roll8").alias("_opp_off_success_rate_pre"),
        pl.col("off_pass_epa_roll8").alias("_opp_off_pass_epa_pre"),
        pl.col("off_rush_epa_roll8").alias("_opp_off_rush_epa_pre"),
        pl.col("def_epa_allowed_play_roll8").alias("_opp_def_epa_allowed_play_pre"),
        pl.col("def_success_rate_allowed_roll8").alias("_opp_def_success_rate_allowed_pre"),
        pl.col("def_pass_epa_allowed_roll8").alias("_opp_def_pass_epa_allowed_pre"),
        pl.col("def_rush_epa_allowed_roll8").alias("_opp_def_rush_epa_allowed_pre"),
    )
    long = long.join(opp_pre, on=["game_id", "opponent"], how="left")

    long = long.with_columns(
        (pl.col("off_epa_play") - pl.col("_opp_def_epa_allowed_play_pre")).alias("adj_off_epa_play"),
        (pl.col("off_success_rate") - pl.col("_opp_def_success_rate_allowed_pre")).alias("adj_off_success_rate"),
        (pl.col("off_pass_epa") - pl.col("_opp_def_pass_epa_allowed_pre")).alias("adj_off_pass_epa"),
        (pl.col("off_rush_epa") - pl.col("_opp_def_rush_epa_allowed_pre")).alias("adj_off_rush_epa"),
        (pl.col("def_epa_allowed_play") - pl.col("_opp_off_epa_play_pre")).alias("adj_def_epa_allowed_play"),
        (pl.col("def_success_rate_allowed") - pl.col("_opp_off_success_rate_pre")).alias("adj_def_success_rate_allowed"),
    )
    long = long.drop([c for c in long.columns if c.startswith("_opp_")])

    adj_stat_cols = [
        "adj_off_epa_play", "adj_off_success_rate", "adj_off_pass_epa",
        "adj_off_rush_epa", "adj_def_epa_allowed_play", "adj_def_success_rate_allowed",
    ]
    long = long.with_columns(_rolling_exprs(adj_stat_cols))
    return long


def compute_player_snap_share(snap_counts: pl.DataFrame, player_crosswalk: pl.DataFrame) -> pl.DataFrame:
    """One row per (gsis_id, team) per game they actually recorded snaps in:
    an EWMA of their own role size -- max(offense_pct, defense_pct) -- up to
    and including that game. `_available_from` marks the earliest later game
    this value is safe to use for (that game's own season*100+week key,
    plus 1): a player ruled OUT often has NO snap-count row at all for the
    game they missed, so looking them up by that exact week would silently
    fail for exactly the cases this feature cares about most. Callers should
    join this with join_asof(..., strategy="backward") on a matching
    season*100+week key against `_available_from`, which finds each
    player's most recent PRIOR appearance regardless of whether they have a
    row for the current week -- never the current or a future game.

    snap_counts identifies players by pfr_id, injuries by gsis_id --
    nflverse doesn't share one ID scheme between the two datasets, so
    player_crosswalk (data_ingestion.load_player_crosswalk) is the join key
    that connects them.
    """
    sc = snap_counts.rename({"pfr_player_id": "pfr_id"}).with_columns(
        pl.max_horizontal(pl.col("offense_pct").fill_null(0.0), pl.col("defense_pct").fill_null(0.0)).alias("snap_share")
    )
    sc = sc.join(player_crosswalk, on="pfr_id", how="inner")
    sc = sc.sort(["gsis_id", "team", "season", "week"])
    sc = sc.with_columns(
        pl.col("snap_share").ewm_mean(half_life=config.EWMA_HALFLIFE_GAMES, min_samples=1)
        .over(["gsis_id", "team"]).alias("snap_share_ewm"),
        (pl.col("season") * 100 + pl.col("week") + 1).alias("_available_from"),
    )
    return sc.select(["gsis_id", "team", "_available_from", "snap_share_ewm"]).sort(["gsis_id", "team", "_available_from"])


def _severity_weighted_injury_score(
    injuries: pl.DataFrame,
    player_snap_share: pl.DataFrame | None,
    position_filter: pl.Expr,
    score_col: str,
) -> pl.DataFrame:
    """Shared machinery behind compute_injury_features and
    compute_qb_injury_features: one row per (season, week, team), a
    severity-weighted sum of players on that week's official injury report
    (Out=1.0, Doubtful=0.5, Questionable=0.25) matching `position_filter`,
    each optionally scaled by their own prior rolling snap share (see
    compute_player_snap_share) -- a starter being out costs more than a
    rarely-used backup at the same report status. Missing snap-share data
    (no crosswalk match, or before 2012 when snap counts start) falls back
    to a fixed default importance instead of dropping the player entirely.
    """
    weights = {"Out": 1.0, "Doubtful": 0.5, "Questionable": 0.25}
    df = injuries.filter(pl.col("report_status").is_in(list(weights)) & position_filter)
    df = df.with_columns(pl.col("report_status").replace(weights).cast(pl.Float64).alias("_severity"))

    if player_snap_share is not None:
        # as-of, not an exact-week join: a player ruled OUT often has no
        # snap-count row at all for the game they missed (that's often WHY
        # they're on this report), so an exact (season, week) match would
        # silently miss exactly the cases this feature cares about most.
        # This instead finds each player's most recent PRIOR appearance,
        # whatever week that was.
        df = df.with_columns((pl.col("season") * 100 + pl.col("week")).alias("_time_key"))
        df = df.sort(["gsis_id", "team", "_time_key"])
        df = df.join_asof(
            player_snap_share.sort(["gsis_id", "team", "_available_from"]),
            left_on="_time_key", right_on="_available_from", by=["gsis_id", "team"], strategy="backward",
        )
        df = df.with_columns(pl.col("snap_share_ewm").fill_null(config.DEFAULT_SNAP_SHARE).alias("_importance"))
    else:
        df = df.with_columns(pl.lit(1.0).alias("_importance"))

    df = df.with_columns((pl.col("_severity") * pl.col("_importance")).alias("_weight"))
    return df.group_by(["season", "week", "team"]).agg(pl.col("_weight").sum().alias(score_col))


def compute_injury_features(
    injuries: pl.DataFrame, player_snap_share: pl.DataFrame | None = None
) -> pl.DataFrame:
    """One row per (season, week, team): severity-weighted injury score for
    every position EXCEPT QB (see compute_qb_injury_features for that) --
    the starter's rolling performance/continuity is already tracked
    directly by compute_qb_form for PLAYED games, so including QB here too
    would double-count that signal there. It doesn't cover upcoming games
    though (see compute_qb_form's carried-forward branch), which is exactly
    the gap compute_qb_injury_features fills.
    """
    return _severity_weighted_injury_score(
        injuries, player_snap_share, pl.col("position") != "QB", "team_injury_score"
    )


def compute_qb_injury_features(
    injuries: pl.DataFrame, player_snap_share: pl.DataFrame | None = None
) -> pl.DataFrame:
    """One row per (season, week, team): QB-specific severity-weighted
    injury score. compute_qb_form's carried-forward rolling EPA for an
    upcoming game assumes the same starter plays again -- reasonable by
    default, but wrong exactly when that starter is hurt, and that change
    only shows up in qb_form's own features once the new starter has
    actually played a game. The weekly injury report is public before
    kickoff, so this is the live-available substitute: scaling severity by
    the player's own prior snap share means a third-string emergency QB's
    "Questionable" tag barely moves this, while the actual starter's does.
    """
    return _severity_weighted_injury_score(
        injuries, player_snap_share, pl.col("position") == "QB", "qb_injury_score"
    )


def compute_referee_features(officials: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """One row per game_id: a leakage-safe rolling EWM of each game's
    RESIDUAL against the market's own total_line (actual total points
    minus the pre-game total_line), attributed to this game's REFEREE
    (position == "Referee" -- the crew's traditional identifying member;
    the rest of the 7-person crew rotates more), using only games strictly
    before this one. Used DIRECTLY (like wind_filled/is_dome), not diffed
    or summed, since it's shared context for both teams, not a team-
    specific quantity.

    Tried first with raw total_points instead of this residual -- reverted
    (see config.py's note): measured worse on every MAE/RMSE/Brier metric
    for every totals model. Likely reason: a referee who's randomly been
    assigned several high-powered offenses looks like a "high-scoring ref"
    on raw points when it's team quality leaking through, not a genuine
    crew tendency. total_line is already nflverse's own best pre-game
    estimate of expected scoring (informed by team quality, weather,
    injuries, everything) -- residualizing against it means what's left
    over is much closer to "did this game run hotter/colder than a
    well-informed market expected," a fairer thing to attribute to the
    officiating crew than the raw score.

    Referee assignments are only announced a day or two before kickoff, and
    nflverse's officials table is compiled from completed games (like
    player_stats) -- there's no "same ref as last week" carry-forward the
    way qb_form has for a starter, so this will be null (median-imputed)
    for essentially every LIVE upcoming prediction. This tests a narrower
    question: does it explain HISTORICAL variance better, not "can we act
    on it live."

    officials' own game_id uses nflverse's OLD numeric gsis format (e.g.
    "2015091000"), not the "{season}_{week}_{away}_{home}" format used
    everywhere else in this pipeline -- confirmed by direct inspection
    (zero id overlap between the two schemes). games.old_game_id (from
    nflverse's own schedules, which nflverse itself uses to bridge the two
    ID schemes) is the crosswalk.
    """
    crosswalk = games.select("game_id", pl.col("old_game_id").alias("_old_game_id")).filter(
        pl.col("_old_game_id").is_not_null()
    )
    ref = officials.filter(pl.col("position") == "Referee").select(
        pl.col("game_id").alias("_old_game_id"), pl.col("official_id").alias("referee_id"), "season", "week"
    )
    ref = ref.join(crosswalk, on="_old_game_id", how="inner").drop("_old_game_id")

    total_resid = games.select(
        "game_id",
        ((pl.col("home_score") + pl.col("away_score")) - pl.col("total_line")).alias("_game_total_resid"),
    )
    ref = ref.join(total_resid, on="game_id", how="left").sort(["referee_id", "season", "week"])

    shifted = pl.col("_game_total_resid").shift(1).over("referee_id")
    ref = ref.with_columns(
        shifted.ewm_mean(half_life=config.EWMA_HALFLIFE_GAMES, min_samples=1)
        .over("referee_id").alias("ref_total_resid_ewm")
    )
    return ref.select(["game_id", "ref_total_resid_ewm"])


def add_game_context_features(
    games: pl.DataFrame,
    spreadspoke_ctx: pl.DataFrame | None = None,
    injury_features: pl.DataFrame | None = None,
    qb_injury_features: pl.DataFrame | None = None,
    referee_features: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Neutral-site, weather, and (if available) prime-time/surface features,
    used directly (not as home/away diffs, since these affect both teams in
    the game equally).

    Domes and closed retractable roofs remove weather as a factor entirely, so
    those (and the handful of rows nflverse leaves null regardless of roof
    type) are filled with a fixed comfortable-conditions constant rather than
    the outdoor average, to avoid implying a dome game has "average" wind.

    prime_time_approx/surface_grass come from an external reference file
    (data_ingestion.load_spreadspoke_context) nflverse doesn't have; left
    null (median-imputed downstream, same as any other missing feature) for
    any game that source doesn't cover.

    injury_diff (home_injury_score - away_injury_score) comes from
    compute_injury_features -- a severity-weighted count of players on each
    team's official pre-game injury report, so it's positive when the HOME
    team has more/worse injuries (ridge/LightGBM learn the sign on their own,
    same as every other diff feature here).

    qb_injury_diff is the same idea from compute_qb_injury_features, kept as
    a separate column rather than folded into injury_diff since it's a
    fundamentally different signal: it's the only pre-game indicator that
    compute_qb_form's carried-forward starter assumption (see its docstring)
    might be wrong for an upcoming game.
    """
    is_dome = pl.col("roof").is_in(["dome", "closed"])
    df = games.with_columns(
        (pl.col("location") == "Neutral").cast(pl.Int8).alias("neutral_site"),
        is_dome.cast(pl.Int8).alias("is_dome"),
        pl.when(is_dome).then(70.0).otherwise(pl.col("temp").fill_null(58.0)).alias("temp_filled"),
        pl.when(is_dome).then(0.0).otherwise(pl.col("wind").fill_null(8.0)).alias("wind_filled"),
    )
    if spreadspoke_ctx is not None:
        df = df.join(spreadspoke_ctx, on=["gameday", "home_team", "away_team"], how="left")

    if injury_features is not None:
        home_inj = injury_features.rename({"team": "home_team", "team_injury_score": "home_injury_score"})
        away_inj = injury_features.rename({"team": "away_team", "team_injury_score": "away_injury_score"})
        df = df.join(home_inj, on=["season", "week", "home_team"], how="left")
        df = df.join(away_inj, on=["season", "week", "away_team"], how="left")
        df = df.with_columns((pl.col("home_injury_score") - pl.col("away_injury_score")).alias("injury_diff"))

    if qb_injury_features is not None:
        home_qb_inj = qb_injury_features.rename({"team": "home_team", "qb_injury_score": "home_qb_injury_score"})
        away_qb_inj = qb_injury_features.rename({"team": "away_team", "qb_injury_score": "away_qb_injury_score"})
        df = df.join(home_qb_inj, on=["season", "week", "home_team"], how="left")
        df = df.join(away_qb_inj, on=["season", "week", "away_team"], how="left")
        # Unlike team_injury_score, a missing match here is the COMMON case
        # (most weeks a team's QB isn't on the injury report at all), and it
        # means "no severity" (0.0), not "unknown" -- left as null, it'd
        # propagate through the subtraction below and null out qb_injury_diff
        # for the vast majority of games (measured: only 64/4363 played games
        # had BOTH teams' QBs flagged the same week; filling first restores
        # the far more common, more informative one-team-only case).
        df = df.with_columns(
            pl.col("home_qb_injury_score").fill_null(0.0), pl.col("away_qb_injury_score").fill_null(0.0)
        )
        df = df.with_columns(
            (pl.col("home_qb_injury_score") - pl.col("away_qb_injury_score")).alias("qb_injury_diff")
        )

    if referee_features is not None:
        df = df.join(referee_features, on="game_id", how="left")
    return df


# Combined (summed, not diffed) offense/defense form -- relevant to a
# TOTALS model in a way a diff isn't: two potent offenses or two leaky
# defenses point toward a high-scoring game regardless of which team is
# actually favored, which is exactly the thing home-away diffs erase.
# "margin" and the career/season-games-played counters are deliberately
# excluded -- summing a margin (roughly zero-sum league-wide) or a games-
# played count has no clean totals interpretation.
TOTAL_SUM_PREFIXES = (
    "off_epa_play", "off_success_rate", "off_pass_epa", "off_rush_epa",
    "def_epa_allowed_play", "def_success_rate_allowed", "def_pass_epa_allowed", "def_rush_epa_allowed",
    "qb_epa_per_att", "qb_cpoe", "off_plays",
)


def build_modeling_table(
    games: pl.DataFrame,
    team_long_form: pl.DataFrame,
    ratings: pl.DataFrame,
    stable_ratings: pl.DataFrame | None = None,
    spreadspoke_ctx: pl.DataFrame | None = None,
    injury_features: pl.DataFrame | None = None,
    qb_injury_features: pl.DataFrame | None = None,
    referee_features: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Wide game-level table: one row per game, home/away form features as
    *_diff columns (home - away) for the margin models, *_sum columns
    (home + away) for the totals model, plus power ratings and market lines."""
    # adj_* (opponent-adjusted) rolling stats are computed by
    # add_opponent_adjusted_form for experimentation, but excluded from the
    # default feature set here -- tried as ridge/LightGBM features directly
    # and reverted (see config.py), since they didn't move the team-bias or
    # divisional-game gap they targeted. Kept out of form_cols so LightGBM's
    # suffix-based auto-inclusion (modeling.gbm_feature_columns) doesn't pull
    # them back in alongside the raw versions.
    form_cols = [
        c for c in team_long_form.columns
        if c.endswith(("_roll8", "_ewm")) and not c.startswith("adj_")
    ]
    form_cols += ["games_played_career", "games_played_season"]
    if "qb_continuity" in team_long_form.columns:
        form_cols += ["qb_continuity"]

    home_form = team_long_form.filter(pl.col("is_home") == 1).select(["game_id", "team"] + form_cols)
    away_form = team_long_form.filter(pl.col("is_home") == 0).select(["game_id", "team"] + form_cols)

    home_form = home_form.rename({c: f"home_{c}" for c in form_cols})
    away_form = away_form.rename({c: f"away_{c}" for c in form_cols})

    df = add_game_context_features(games, spreadspoke_ctx, injury_features, qb_injury_features, referee_features)
    df = df.join(home_form.drop("team"), on="game_id", how="left")
    df = df.join(away_form.drop("team"), on="game_id", how="left")

    df = df.join(
        ratings.select(["game_id", "home_elo_pre", "away_elo_pre", "elo_diff"]),
        on="game_id",
        how="left",
    )
    if stable_ratings is not None:
        df = df.join(
            stable_ratings.select(["game_id", "home_elo_stable_pre", "away_elo_stable_pre", "elo_stable_diff"]),
            on="game_id",
            how="left",
        )

    diff_exprs = []
    sum_exprs = []
    for c in form_cols:
        diff_exprs.append((pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(f"{c}_diff"))
        if c.startswith(TOTAL_SUM_PREFIXES):
            sum_exprs.append((pl.col(f"home_{c}") + pl.col(f"away_{c}")).alias(f"{c}_sum"))
    df = df.with_columns(diff_exprs + sum_exprs)

    df = df.with_columns(
        (pl.col("home_rest") - pl.col("away_rest")).alias("rest_diff"),
        (pl.col("home_score") + pl.col("away_score")).alias("total_points"),
    )

    return df.sort(["season", "week", "game_id"])
