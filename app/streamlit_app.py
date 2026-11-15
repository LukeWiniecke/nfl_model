"""Interactive dashboard for the NFL ATS + moneyline model.

Reads the parquet/CSV outputs already written by `main.py` to
outputs/model_results/ and outputs/backtests/ -- it never recomputes the
model or the backtest from scratch, but bet-selection thresholds ARE
re-graded live in-app (see regraded_bets()) since ats_edge/ml_edge are
threshold-independent and already saved for every game, not just placed bets.

Run with:
    .venv\\Scripts\\streamlit run app\\streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src import backtesting, config, evaluation, similarity
from src.probabilities import kelly_fraction
from src.utils import american_payout_multiplier

# --- Palette (validated categorical/diverging set; see dataviz skill) ------

BLUE = "#2a78d6"      # categorical slot 1 -- "home" bets, primary series
ORANGE = "#eb6834"    # categorical slot 2 -- "away" bets
RED = "#e34948"       # diverging pole -- unprofitable / negative
GRAY = "#898781"      # muted -- "no bet" context, gridlines
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"

BET_SIDE_COLORS = {"home": BLUE, "away": ORANGE, "none": GRAY}
SIGN_COLORS = {"Profitable": BLUE, "Unprofitable": RED}

EDGE_BUCKET_ORDER = ["(-inf, 1.5]", "(1.5, 3]", "(3, 5]", "(5, 8]", "(8, inf]"]
EDGE_BUCKET_LABELS = {
    "(-inf, 1.5]": "0-1.5", "(1.5, 3]": "1.5-3", "(3, 5]": "3-5",
    "(5, 8]": "5-8", "(8, inf]": "8+",
}
SPREAD_BUCKET_ORDER = ["(-inf, 3]", "(3, 7]", "(7, 10]", "(10, 14]", "(14, inf]"]
WEEK_BUCKET_ORDER = ["Weeks 1-3", "Weeks 4-8", "Weeks 9-13", "Weeks 14-18", "Playoffs"]
WEEK_BUCKET_LABELS = {b: b for b in WEEK_BUCKET_ORDER}
SPREAD_BUCKET_LABELS = {
    "(-inf, 3]": "0-3", "(3, 7]": "3-7", "(7, 10]": "7-10",
    "(10, 14]": "10-14", "(14, inf]": "14+",
}

MODEL_LABELS = {"ridge_baseline": "Ridge baseline", "lgbm": "LightGBM", "ensemble": "Ensemble (ridge + LightGBM)"}
# Maps a margin model's name to its totals counterpart (phase 1 -- see
# README) -- used only to derive a predicted score (home/away split of the
# totals model's predicted_total using the margin model's predicted_margin),
# not a betting market of its own yet.
TOTAL_MODEL_NAME = {"ridge_baseline": "ridge_total", "lgbm": "lgbm_total", "ensemble": "ensemble_total"}
MARKET_LABELS = {"ats": "Against the spread", "ml": "Moneyline"}

CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color="#0b0b0b"),
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    margin=dict(l=10, r=10, t=40, b=10),
)


# --- Data loading -----------------------------------------------------------


@st.cache_data
def load_bets(model: str, market: str) -> pl.DataFrame | None:
    path = config.BACKTESTS_DIR / f"{model}_{market}_bets.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


@st.cache_data
def load_predictions(model: str) -> pl.DataFrame | None:
    path = config.MODEL_RESULTS_DIR / f"{model}_predictions.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


@st.cache_data
def load_skill(model: str) -> pl.DataFrame | None:
    path = config.MODEL_RESULTS_DIR / f"{model}_skill.csv"
    if not path.exists():
        return None
    return pl.read_csv(path)


@st.cache_data
def load_contributions(model: str, upcoming: bool = False) -> pl.DataFrame | None:
    """Backtest (walk-forward) breakdowns by default; with upcoming=True, the
    live model's breakdowns for games not yet played (written by main.py)."""
    suffix = "_upcoming_contributions" if upcoming else "_contributions"
    path = config.MODEL_RESULTS_DIR / f"{model}{suffix}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def selection_edge_col(pdf: pd.DataFrame, market: str) -> str:
    """The column the PIPELINE decided bets on, which is not always the one it
    reports.

    Since the ensemble became market-anchored, the reported edge and the
    selection edge are different quantities. For the spread the two are the
    same ranking (anchoring multiplies the edge by one positive scalar per
    season), but for the moneyline they are not: the anchored probability adds
    a per-game intercept and divides by a per-game std, and the rank
    correlation between the two selection magnitudes is about 0.25. So the
    pipeline reports the anchored, calibrated probability -- whose Brier now
    beats the market -- while selecting on the frozen legacy one, and the
    dashboard has to follow it or it shows a book nobody validated. Measured
    on the shipped tables before this existed: regrading the moneyline on the
    reported edge gave 11 bets against the saved book's 647, and the spread
    573 against 788.

    Falls back to the reported column when a parquet predates the split --
    but "predates" has to be checked PER ROW, not per column. The Track
    record tab's picks_log.parquet is one file accumulated across every
    `--refresh` run since July via `pl.concat(..., how="diagonal_relaxed")`
    (live_picks.append_to_picks_log); when `ats_edge_select`/`ml_edge_select`
    were added mid-season, every row logged before that point got backfilled
    with a NULL for it rather than dropped. A plain `preferred in pdf.columns`
    check sees the column present (true for the whole log once any row has
    it) and selects on it for EVERY row -- including the null ones. _bet_side
    compares edge > threshold / edge < -threshold, both False for NaN, so
    every pre-cutover pick silently graded as "no bet" and vanished from the
    chart (204 picks, every one before 2026-09-20, confirmed against the
    live log). Coalescing row-wise onto the fallback fixes it without
    touching any caller that only ever sees fully-populated current-pipeline
    output (load_bets, upcoming boards), where every value already exists
    and this is a no-op.
    """
    fallback = "ats_edge" if market == "ats" else "ml_edge"
    preferred = "ats_edge_select" if market == "ats" else "ml_edge_select"
    if preferred not in pdf.columns:
        return fallback
    if pdf[preferred].isna().any():
        pdf[preferred] = pdf[preferred].fillna(pdf[fallback])
    return preferred


def pipeline_threshold_col(pdf: pd.DataFrame, market: str) -> str | None:
    """Name of the threshold column pipeline_threshold will actually use, or
    None when it falls back to the fixed config constant. Kept separate so the
    sidebar caption describes the same bar the regrade applies -- they read
    different columns once, and the caption quietly described a bar nothing
    was using."""
    select_col = "ats_select_threshold_used" if market == "ats" else "ml_select_threshold_used"
    report_col = "ats_threshold_used" if market == "ats" else "ml_threshold_used"
    for col in (select_col, report_col):
        if col in pdf.columns:
            return col
    return None


def pipeline_threshold(pdf: pd.DataFrame, market: str):
    """The per-game bar the pipeline ACTUALLY placed its bets at, read back
    from the stored threshold column, so the dashboard's default view is the
    validated bet book rather than one fixed bar applied to every game.

    Before this existed every tab regraded at a single scalar defaulting to
    config.ATS_EDGE_THRESHOLD_POINTS, while the pipeline derives a different
    bar per game from the walk-forward shrinkage slope and the odds on offer
    (backtesting.adaptive_ats_threshold). Prefers the *_select_threshold_used
    column, which is the bar applied to the selection edge above; the older
    ats_threshold_used is on the reported scale and pairing it with the
    selection edge would be wrong in both directions. Falls back to the fixed
    config default wherever no column exists (older parquets, or a market the
    pipeline runs at a fixed bar) or a value is null.
    """
    default = config.ATS_EDGE_THRESHOLD_POINTS if market == "ats" else config.ML_EDGE_THRESHOLD_PROB
    col = pipeline_threshold_col(pdf, market)
    if col is None:
        return float(default)
    return pdf[col].astype(float).fillna(float(default)).to_numpy()


def effective_threshold(pdf: pd.DataFrame, market: str, threshold: float | None):
    """`None` means "use the pipeline's own per-game bar"; a number is the
    sidebar's manual override, applied to every game."""
    return pipeline_threshold(pdf, market) if threshold is None else threshold


@st.cache_data
def regraded_bets(model: str, market: str, threshold: float | None) -> pl.DataFrame | None:
    """Re-decide bet_side (and re-grade profit/staked) at the sidebar's
    threshold -- the pipeline's own per-game bar when `threshold` is None --
    reusing the exact regrade_ats/regrade_ml logic main.py uses.
    ats_edge/ml_edge/model probabilities are already computed for every game
    regardless of threshold, so this is a cheap re-grade, not a re-run of the
    model."""
    raw = load_bets(model, market)
    if raw is None:
        return None
    pdf = raw.to_pandas()
    thr = effective_threshold(pdf, market, threshold)
    edge_col = selection_edge_col(pdf, market)
    pdf = (
        backtesting.regrade_ats(pdf, thr, edge_col=edge_col)
        if market == "ats"
        else backtesting.regrade_ml(pdf, thr, edge_col=edge_col)
    )
    return evaluation.add_bet_context(pl.from_pandas(pdf))


@st.cache_data
def load_modeling_table() -> pl.DataFrame | None:
    path = config.PROCESSED_DIR / "modeling_table.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


@st.cache_data
def similar_games(query_game_id: str, k: int) -> pl.DataFrame:
    mt = load_modeling_table()
    if mt is None:
        return pl.DataFrame()
    return similarity.find_similar_games(mt, query_game_id, k=k)


@st.cache_data
def bankroll_sim(model: str, market: str, threshold: float, kelly_multiplier: float, kelly_cap: float) -> pl.DataFrame | None:
    bets = regraded_bets(model, market, threshold)
    if bets is None:
        return None
    prob_col = "model_cover_prob" if market == "ats" else "model_home_prob"
    home_odds_col = "home_spread_odds" if market == "ats" else "home_moneyline"
    away_odds_col = "away_spread_odds" if market == "ats" else "away_moneyline"
    sim = backtesting.simulate_bankroll(
        bets.to_pandas(), prob_col, home_odds_col, away_odds_col,
        kelly_multiplier=kelly_multiplier, kelly_cap=kelly_cap,
    )
    return pl.from_pandas(sim)


@st.cache_data
def bootstrap_paths(
    model: str, market: str, threshold: float, kelly_multiplier: float, kelly_cap: float, n_boot: int = 500
) -> np.ndarray | None:
    bets = regraded_bets(model, market, threshold)
    if bets is None:
        return None
    prob_col = "model_cover_prob" if market == "ats" else "model_home_prob"
    home_odds_col = "home_spread_odds" if market == "ats" else "home_moneyline"
    away_odds_col = "away_spread_odds" if market == "ats" else "away_moneyline"
    return backtesting.bootstrap_bankroll_paths(
        bets.to_pandas(), prob_col, home_odds_col, away_odds_col,
        kelly_multiplier=kelly_multiplier, kelly_cap=kelly_cap, n_boot=n_boot,
    )


@st.cache_data
def kelly_sweep(model: str, market: str, threshold: float, kelly_cap: float, n_boot: int = 300) -> pd.DataFrame | None:
    bets = regraded_bets(model, market, threshold)
    if bets is None:
        return None
    prob_col = "model_cover_prob" if market == "ats" else "model_home_prob"
    home_odds_col = "home_spread_odds" if market == "ats" else "home_moneyline"
    away_odds_col = "away_spread_odds" if market == "ats" else "away_moneyline"
    multipliers = np.arange(0.05, 1.55, 0.05)
    return backtesting.sweep_kelly_multiplier(
        bets.to_pandas(), prob_col, home_odds_col, away_odds_col, multipliers,
        kelly_cap=kelly_cap, n_boot=n_boot,
    )


@st.cache_data
def threshold_optimizer(model: str, market: str, n_boot: int = 1000) -> pd.DataFrame | None:
    bets = load_bets(model, market)
    if bets is None:
        return None
    grid = np.arange(0.0, 5.25, 0.25) if market == "ats" else np.arange(0.0, 0.105, 0.005)
    return backtesting.optimal_threshold_walk_forward(bets.to_pandas(), market, grid, n_boot=n_boot)


@st.cache_data
def load_upcoming() -> pl.DataFrame | None:
    path = config.MODEL_RESULTS_DIR / "upcoming_picks.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


@st.cache_data
def load_picks_log() -> pl.DataFrame | None:
    if not config.PICKS_LOG_PATH.exists():
        return None
    log = pl.read_parquet(config.PICKS_LOG_PATH)
    # Keep only the LATEST snapshot per (game_id, model) -- the model's read
    # may have been logged more than once before kickoff (a new injury
    # designation, a line move); the most recent one before the game is the
    # "official" pick to grade.
    latest = log.sort("logged_at").group_by(["game_id", "model"], maintain_order=True).last()
    games = pl.read_parquet(config.PROCESSED_DIR / "games.parquet").select(["game_id", "home_margin"])
    return latest.join(games, on="game_id", how="left")


@st.cache_data
def load_game_scores() -> pl.DataFrame:
    """Final home/away scores -- not carried in the bets parquet files
    (those only keep home_margin), so this is a separate small join purely
    for display (e.g. total points scored) in the dashboard."""
    return pl.read_parquet(config.PROCESSED_DIR / "games.parquet").select(["game_id", "home_score", "away_score"])


@st.cache_data
def load_elo_history() -> pl.DataFrame | None:
    """One row per team per SCHEDULED game, with that team's PRE-game Elo
    rating and a `played` flag saying whether that game has a result yet.

    The flag matters. games.parquet keeps not-yet-played games (cleaning
    stopped dropping them so live_picks can see upcoming fixtures), and
    power_ratings.compute_elo_ratings records a pre-game rating for every
    scheduled game while applying the post-game UPDATE only when a result
    exists. So every remaining game of the current season carries the same
    number: the team's rating as of its last completed game, repeated.

    That makes those rows useful for one thing and misleading for another. A
    team's NEXT scheduled game is exactly where its current rating should be
    read from -- that pre-game rating already includes the last result. But
    plotting the rest of them draws a flat line months into the future, and
    labelling the leaderboard off the last of them claims the ratings are
    current as of a week 18 game in January that has not been played. Callers
    take the flag and decide; see elo_trajectory and current_elo.
    """
    ratings_path = config.PROCESSED_DIR / "elo_ratings.parquet"
    games_path = config.PROCESSED_DIR / "games.parquet"
    if not ratings_path.exists() or not games_path.exists():
        return None
    ratings = pl.read_parquet(ratings_path)
    games = pl.read_parquet(games_path).select(
        ["game_id", "season", "week", "gameday", "home_team", "away_team", "home_margin"]
    )
    joined = games.join(ratings, on="game_id", how="inner").with_columns(
        pl.col("home_margin").is_not_null().alias("played")
    )
    home = joined.select(
        "game_id", "season", "week", "gameday", "played",
        pl.col("home_team").alias("team"), pl.col("home_elo_pre").alias("elo_pre"),
    )
    away = joined.select(
        "game_id", "season", "week", "gameday", "played",
        pl.col("away_team").alias("team"), pl.col("away_elo_pre").alias("elo_pre"),
    )
    return pl.concat([home, away]).sort(["team", "gameday", "week"])


def _next_scheduled(elo_hist: pl.DataFrame) -> pl.DataFrame:
    """Each team's next unplayed game -- one row per team, or empty."""
    return (
        elo_hist.filter(~pl.col("played"))
        .sort(["team", "gameday", "week"])
        .unique(subset=["team"], keep="first", maintain_order=True)
    )


def elo_trajectory(elo_hist: pl.DataFrame) -> pl.DataFrame:
    """Rating history to plot: every completed game, plus each team's next
    scheduled game so the line ends on the CURRENT rating instead of the one
    the team carried INTO its most recent game. Without that last point the
    chart is one result stale; without the filter it runs flat to January.
    """
    return pl.concat([elo_hist.filter(pl.col("played")), _next_scheduled(elo_hist)]).sort(
        ["team", "gameday", "week"]
    )


def current_elo(elo_hist: pl.DataFrame) -> pl.DataFrame:
    """Each team's current rating, labelled with the completed game it
    reflects. The value comes from the next scheduled game's pre-game rating
    (which already includes the last result); the as-of labels come from the
    last PLAYED game. A team with no game left on the schedule -- the offseason
    gap before the next season is published -- has no next-game row to read, so
    it falls back to the rating it carried INTO its last game and is marked
    reflects_last_result = False rather than being passed off as current.
    """
    played = elo_hist.filter(pl.col("played"))
    as_of = played.group_by("team", maintain_order=False).agg(
        pl.col("elo_pre").last().alias("_pre_last_played"),
        pl.col("season").last().alias("as_of_season"),
        pl.col("week").last().alias("as_of_week"),
        pl.len().alias("games_rated"),
    )
    upcoming = _next_scheduled(elo_hist).select("team", pl.col("elo_pre").alias("_next_pre"))
    return (
        as_of.join(upcoming, on="team", how="left")
        .with_columns(
            pl.coalesce(["_next_pre", "_pre_last_played"]).alias("elo"),
            pl.col("_next_pre").is_not_null().alias("reflects_last_result"),
        )
        .drop(["_next_pre", "_pre_last_played"])
        .select("team", "elo", "as_of_season", "as_of_week", "games_rated", "reflects_last_result")
        .sort("elo", descending=True)
    )


# Per-team form columns worth surfacing in the drill-down, in display order:
# (modeling-table stem, label, number format, whether LOWER is better).
# Every one of these is an actual model input or a diagnostic sitting beside
# them, not a derived statistic invented for the UI.
TEAM_FORM_FIELDS = [
    ("off_epa_play_ewm", "Offense EPA/play", "{:+.3f}", False),
    ("def_epa_allowed_play_ewm", "Defense EPA/play allowed", "{:+.3f}", True),
    ("off_success_rate_ewm", "Offense success rate", "{:.1%}", False),
    ("off_pass_epa_ewm", "Pass EPA/play", "{:+.3f}", False),
    ("off_rush_epa_ewm", "Rush EPA/play", "{:+.3f}", False),
    ("qb_epa_per_att_ewm", "QB EPA/attempt", "{:+.3f}", False),
    ("off_plays_ewm", "Plays per game", "{:.1f}", False),
    ("pen_yards_ewm", "Penalty yards/game", "{:.1f}", True),
    ("pen_presnap_ewm", "Pre-snap penalties/game", "{:.2f}", True),
]

# Injury groups. These are NOT model inputs -- measured and rejected (see
# config.py) -- but they are the most legible thing on the page for "who is
# hurt and where", which is exactly what the model reduces to one number.
TEAM_INJURY_FIELDS = [
    ("inj_ol", "O-line"), ("inj_dl", "D-line"), ("inj_lb", "Linebackers"),
    ("inj_db", "Secondary"), ("inj_rec", "Receivers"), ("inj_rb", "Backs"),
    ("qb_injury_score", "Quarterback"),
]


@st.cache_data
def load_team_game_log() -> pl.DataFrame | None:
    """One row per TEAM per game, from that team's own point of view.

    The modeling table is one row per game with home_/away_ prefixed columns;
    this unpivots it so a team can be followed down a column. Margin, spread
    and the model's prediction are all flipped to the selected team's
    perspective, so a positive number always means good for THIS team, which
    is the only way a game log is readable.
    """
    path = config.PROCESSED_DIR / "modeling_table.parquet"
    if not path.exists():
        return None
    mt = pl.read_parquet(path)

    stems = [f for f, _, _, _ in TEAM_FORM_FIELDS] + [f for f, _ in TEAM_INJURY_FIELDS]
    stems += ["elo_pre", "injury_score", "rest", "games_played_season"]
    stems = [f for f in stems if f"home_{f}" in mt.columns and f"away_{f}" in mt.columns]

    base = ["game_id", "season", "week", "gameday", "game_type", "spread_line", "total_line"]

    def side(is_home: bool) -> pl.DataFrame:
        me, them = ("home", "away") if is_home else ("away", "home")
        sign = 1 if is_home else -1
        cols = [pl.col(c) for c in base] + [
            pl.col(f"{me}_team").alias("team"),
            pl.col(f"{them}_team").alias("opponent"),
            pl.lit(is_home).alias("is_home"),
            (pl.col("home_margin") * sign).alias("margin"),
            (pl.col("spread_line") * sign).alias("spread_for_team"),
            pl.col(f"{me}_score").alias("points_for"),
            pl.col(f"{them}_score").alias("points_against"),
            pl.col(f"{them}_elo_pre").alias("opp_elo_pre"),
        ]
        cols += [pl.col(f"{me}_{f}").alias(f) for f in stems]
        return mt.select(cols).with_columns(pl.lit(sign).alias("_sign"))

    long = pl.concat([side(True), side(False)]).sort(["team", "season", "week"])

    # Elo AFTER a game is the rating the team carries into its next one. The
    # last played game gets it from the next scheduled game, which is exactly
    # what compute_elo_ratings records there.
    long = long.with_columns(pl.col("elo_pre").shift(-1).over("team").alias("elo_post"))
    long = long.with_columns((pl.col("elo_post") - pl.col("elo_pre")).alias("elo_change"))

    # Cover, from this team's side. A push is neither.
    long = long.with_columns(
        pl.when(pl.col("margin").is_null())
        .then(None)
        .when(pl.col("margin") > pl.col("spread_for_team"))
        .then(pl.lit("Cover"))
        .when(pl.col("margin") < pl.col("spread_for_team"))
        .then(pl.lit("No cover"))
        .otherwise(pl.lit("Push"))
        .alias("ats_result"),
        pl.when(pl.col("margin").is_null()).then(None)
        .when(pl.col("margin") > 0).then(pl.lit("W"))
        .when(pl.col("margin") < 0).then(pl.lit("L"))
        .otherwise(pl.lit("T")).alias("result"),
    )
    return long


@st.cache_data
def team_log_with_predictions(model: str) -> pl.DataFrame | None:
    """The team game log plus this model's predicted margin and its error,
    both flipped to the team's own perspective."""
    log = load_team_game_log()
    preds = load_predictions(model)
    if log is None or preds is None:
        return log
    p = preds.select(["game_id", "predicted_margin"])
    out = log.join(p, on="game_id", how="left").with_columns(
        (pl.col("predicted_margin") * pl.col("_sign")).alias("model_margin")
    )
    return out.with_columns(
        (pl.col("model_margin") - pl.col("margin")).alias("model_error"),
        (pl.col("model_margin") - pl.col("spread_for_team")).alias("model_vs_line"),
    )


@st.cache_data
def team_bias_table(model: str) -> pl.DataFrame | None:
    """Per team: how much the MODEL over- or under-rates it, next to how much
    the MARKET does, and the difference.

    The tab has always claimed this finding in prose -- perennially strong
    franchises underrated, weak ones overrated -- without ever showing the
    number. Positive means the source predicted that team to do better than it
    actually did, i.e. overrated it. The `excess` column is the only one that
    is really about the MODEL: a team both the model and the market read the
    same way is not a model problem.
    """
    log = team_log_with_predictions(model)
    if log is None or "model_margin" not in log.columns:
        return None
    graded = log.filter(pl.col("margin").is_not_null() & pl.col("model_margin").is_not_null())
    if graded.height == 0:
        return None
    return (
        graded.with_columns(
            (pl.col("model_margin") - pl.col("margin")).alias("_model_bias"),
            (pl.col("spread_for_team") - pl.col("margin")).alias("_market_bias"),
        )
        .group_by("team")
        .agg(
            pl.len().alias("games"),
            pl.col("_model_bias").mean().alias("model_bias"),
            pl.col("_market_bias").mean().alias("market_bias"),
        )
        .with_columns((pl.col("model_bias") - pl.col("market_bias")).alias("excess"))
        .sort("excess")
    )


def diverging_team_bars(
    pdf: pd.DataFrame, value_col: str, title: str, xaxis_title: str,
    pos_label: str, neg_label: str, hovertemplate: str, customdata: list[str] | None = None,
) -> go.Figure:
    """Horizontal diverging bars across teams, one bar per team, colored by
    SIGN only.

    Diverging because the quantity has a real zero with opposite meanings on
    each side; blue and orange are used as the two poles because they read as
    cool/warm opposites, and the neutral midpoint is the zero rule itself.
    Deliberately NOT a magnitude ramp: shading each bar darker-where-bigger
    would double-encode length as hue and burn the only free channel on
    information the bar already carries.
    """
    fig = go.Figure()
    for label, color, mask in [
        (neg_label, BLUE, pdf[value_col] < 0),
        (pos_label, ORANGE, pdf[value_col] >= 0),
    ]:
        if not mask.any():
            continue
        fig.add_bar(
            y=pdf.loc[mask, "team"], x=pdf.loc[mask, value_col],
            name=label, marker_color=color, orientation="h",
            customdata=pdf.loc[mask, customdata] if customdata else None,
            hovertemplate=hovertemplate,
        )
    fig.add_vline(x=0, line_width=1, line_color=INK_SECONDARY)
    # CHART_LAYOUT's 40px top margin is not enough once a horizontal legend
    # sits above the plotting area: the legend row lands on top of the title.
    layout = {**CHART_LAYOUT, "margin": dict(l=10, r=10, t=96, b=10)}
    fig.update_layout(
        title=dict(text=title, y=0.985, yanchor="top"),
        xaxis_title=xaxis_title, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.015, x=0),
        height=max(380, 20 * len(pdf) + 180), bargap=0.25, **layout,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
    return fig


def bucket_axis(df: pl.DataFrame, col: str, order: list[str], labels: dict[str, str]) -> pl.DataFrame:
    df = df.filter(pl.col(col).is_in(order))
    df = df.with_columns(pl.col(col).cast(pl.Enum(order)))
    df = df.sort(col)
    return df.with_columns(pl.col(col).cast(pl.Utf8).replace(labels).alias("_label"))


def diverging_bar(df: pl.DataFrame, x_col: str, title: str, x_title: str = "") -> go.Figure:
    pdf = df.to_pandas()
    has_ci = "ci_lo" in pdf.columns and "ci_hi" in pdf.columns
    sign = pdf["roi"].apply(lambda r: "Profitable" if r >= 0 else "Unprofitable")
    fig = go.Figure()
    for label, color in SIGN_COLORS.items():
        mask = sign == label
        if not mask.any():
            continue
        error_y = None
        if has_ci:
            error_y = dict(
                type="data", symmetric=False,
                array=(pdf.loc[mask, "ci_hi"] - pdf.loc[mask, "roi"]) * 100,
                arrayminus=(pdf.loc[mask, "roi"] - pdf.loc[mask, "ci_lo"]) * 100,
                color=INK_SECONDARY, thickness=1.5, width=4,
            )
        fig.add_bar(
            x=pdf.loc[mask, x_col],
            y=pdf.loc[mask, "roi"] * 100,
            name=label,
            marker_color=color,
            error_y=error_y,
            text=[f"{v:+.1%}" for v in pdf.loc[mask, "roi"]],
            textposition="outside",
            customdata=pdf.loc[mask, ["n_bets", "win_rate"]],
            hovertemplate="%{x}<br>ROI %{y:.1f}%<br>Bets: %{customdata[0]}<br>Win rate: %{customdata[1]:.1%}<extra></extra>",
        )
    fig.add_hline(y=0, line_width=1, line_color=GRID)
    fig.update_layout(
        title=title, yaxis_title="ROI (%)", xaxis_title=x_title, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **CHART_LAYOUT,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    return fig


def diverging_hbar_by_team(df: pl.DataFrame) -> go.Figure:
    pdf = df.sort("roi").to_pandas()
    has_ci = "ci_lo" in pdf.columns and "ci_hi" in pdf.columns
    sign = pdf["roi"].apply(lambda r: "Profitable" if r >= 0 else "Unprofitable")
    fig = go.Figure()
    for label, color in SIGN_COLORS.items():
        mask = sign == label
        if not mask.any():
            continue
        error_x = None
        if has_ci:
            error_x = dict(
                type="data", symmetric=False,
                array=(pdf.loc[mask, "ci_hi"] - pdf.loc[mask, "roi"]) * 100,
                arrayminus=(pdf.loc[mask, "roi"] - pdf.loc[mask, "ci_lo"]) * 100,
                color=INK_SECONDARY, thickness=1.5, width=3,
            )
        fig.add_bar(
            y=pdf.loc[mask, "bet_team"],
            x=pdf.loc[mask, "roi"] * 100,
            name=label,
            marker_color=color,
            orientation="h",
            error_x=error_x,
            customdata=pdf.loc[mask, ["n_bets", "win_rate"]],
            hovertemplate="%{y}<br>ROI %{x:.1f}%<br>Bets: %{customdata[0]}<br>Win rate: %{customdata[1]:.1%}<extra></extra>",
        )
    fig.add_vline(x=0, line_width=1, line_color=GRID)
    fig.update_layout(
        title="ROI by team", xaxis_title="ROI (%)", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=700,
        **CHART_LAYOUT,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
    return fig


def roi_by_season_line(df: pl.DataFrame) -> go.Figure:
    pdf = df.sort("season").to_pandas()
    has_ci = "ci_lo" in pdf.columns and "ci_hi" in pdf.columns
    error_y = None
    if has_ci:
        error_y = dict(
            type="data", symmetric=False,
            array=(pdf["ci_hi"] - pdf["roi"]) * 100, arrayminus=(pdf["roi"] - pdf["ci_lo"]) * 100,
            color=INK_SECONDARY, thickness=1.5, width=4,
        )
    fig = go.Figure()
    fig.add_scatter(
        x=pdf["season"], y=pdf["roi"] * 100, mode="lines+markers",
        line=dict(color=BLUE, width=2), marker=dict(size=8, color=BLUE),
        error_y=error_y,
        customdata=pdf[["n_bets", "win_rate"]],
        hovertemplate="Season %{x}<br>ROI %{y:.1f}%<br>Bets: %{customdata[0]}<br>Win rate: %{customdata[1]:.1%}<extra></extra>",
        name="ROI",
    )
    fig.add_hline(y=0, line_width=1, line_color=GRID)
    fig.update_layout(title="ROI by season", yaxis_title="ROI (%)", xaxis_title="Season", showlegend=False, **CHART_LAYOUT)
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, dtick=1)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    return fig


def cumulative_profit_line(bets: pl.DataFrame) -> go.Figure:
    placed = bets.filter(pl.col("bet_side") != "none").sort(["season", "week", "game_id"])
    pdf = placed.to_pandas().reset_index(drop=True)
    pdf["cum_profit"] = pdf["profit_units"].cumsum()
    pdf["bet_number"] = range(1, len(pdf) + 1)
    fig = go.Figure()
    fig.add_scatter(
        x=pdf["bet_number"], y=pdf["cum_profit"], mode="lines",
        line=dict(color=BLUE, width=2),
        customdata=pdf[["season", "week"]],
        hovertemplate="Bet #%{x}<br>Cumulative profit: %{y:.1f}u<br>Season %{customdata[0]} wk %{customdata[1]}<extra></extra>",
        name="Cumulative profit",
    )
    fig.add_hline(y=0, line_width=1, line_color=GRID)
    fig.update_layout(
        title="Cumulative profit (flat 1-unit stakes, in bet order)",
        yaxis_title="Profit (units)", xaxis_title="Bet number", showlegend=False,
        **CHART_LAYOUT,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    return fig


def calibration_scatter(df: pl.DataFrame, x_col: str, y_col: str, title: str, axis_title: str) -> go.Figure:
    pdf = df.to_pandas()
    fig = go.Figure()
    lo = min(pdf[x_col].min(), pdf[y_col].min())
    hi = max(pdf[x_col].max(), pdf[y_col].max())
    fig.add_scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=GRAY, width=1, dash="dash"),
        name="Model = market", hoverinfo="skip",
    )

    # Outcome of the model's own call: won (circle) / lost (x) / push (diamond).
    # "No bet" games have no call to grade, so they stay a plain circle.
    # bet_won is null exactly where bet_side == "none"; fillna(False) first so the
    # array is a plain bool (nullable-bool `==` comparisons raise on the NA entries).
    is_none = (pdf["bet_side"] == "none").to_numpy()
    is_push = pdf["is_push"].fillna(False).to_numpy(dtype=bool)
    bet_won = pdf["bet_won"].fillna(False).to_numpy(dtype=bool)
    symbol = np.select([is_none, is_push, bet_won], ["circle", "diamond", "circle"], default="x")
    outcome_label = np.select([is_none, is_push, bet_won], ["No bet", "Push", "Won"], default="Lost")

    for side, color in BET_SIDE_COLORS.items():
        mask = (pdf["bet_side"] == side).to_numpy()
        if not mask.any():
            continue
        label = {"home": "Bet home", "away": "Bet away", "none": "No bet"}[side]
        custom = np.column_stack([
            pdf.loc[mask, "home_team"].to_numpy(), pdf.loc[mask, "away_team"].to_numpy(),
            pdf.loc[mask, "season"].to_numpy(), pdf.loc[mask, "week"].to_numpy(), outcome_label[mask],
        ])
        fig.add_scatter(
            x=pdf.loc[mask, x_col], y=pdf.loc[mask, y_col], mode="markers",
            marker=dict(
                size=9 if side != "none" else 7, color=color,
                opacity=0.85 if side != "none" else 0.35,
                symbol=symbol[mask], line=dict(width=1, color=color),
            ),
            name=label,
            customdata=custom,
            hovertemplate=(
                "%{customdata[2]} wk%{customdata[3]}: %{customdata[1]} @ %{customdata[0]}<br>"
                f"Market: %{{x:.2f}}<br>Model: %{{y:.2f}}<br>" + "%{customdata[4]}<extra></extra>"
            ),
        )
    fig.update_layout(
        title=title, xaxis_title=f"Market {axis_title}", yaxis_title=f"Model {axis_title}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **CHART_LAYOUT,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
    return fig


def reliability_chart(df: pl.DataFrame, title: str) -> go.Figure:
    pdf = df.sort("mean_pred").to_pandas()
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, width=1, dash="dash"),
        name="Perfectly calibrated", hoverinfo="skip",
    )
    fig.add_scatter(
        x=pdf["mean_pred"], y=pdf["actual_rate"], mode="lines+markers",
        line=dict(color=BLUE, width=2), marker=dict(size=np.clip(pdf["n"] / pdf["n"].max() * 20, 6, 20), color=BLUE),
        customdata=pdf[["n"]],
        hovertemplate="Predicted %{x:.1%}<br>Actual %{y:.1%}<br>n=%{customdata[0]}<extra></extra>",
        name="Model",
    )
    fig.update_layout(
        title=title, xaxis_title="Predicted probability (bin mean)", yaxis_title="Actual outcome rate",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **CHART_LAYOUT,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, range=[0, 1], tickformat=".0%")
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, range=[0, 1], tickformat=".0%")
    return fig


# --- Page --------------------------------------------------------------------

st.set_page_config(page_title="NFL ATS + Moneyline Model", layout="wide")

st.title("NFL ATS + Moneyline Model")
st.caption(
    "Research/backtesting dashboard, not betting advice. Walk-forward model "
    "predictions vs. closing market lines, "
    f"seasons {config.FIRST_SEASON}-{config.LAST_SEASON}."
)

with st.sidebar:
    st.header("Controls")
    model = st.selectbox("Model", list(MODEL_LABELS), format_func=lambda m: MODEL_LABELS[m])
    market = st.radio("Market", list(MARKET_LABELS), format_func=lambda m: MARKET_LABELS[m])

    st.subheader("Bet-selection threshold")
    threshold_mode = st.radio(
        "Threshold source",
        ["Pipeline (per game, adaptive)", "Manual (one bar for every game)"],
        help="The pipeline does not bet at one fixed bar: each game's bar is derived walk-forward from "
        "the shrinkage slope and the odds on offer (backtesting.adaptive_ats_threshold) and stored with "
        "the bet. 'Pipeline' shows exactly that validated book. 'Manual' applies the slider to every "
        "game instead -- useful for what-ifs, but it is a different book from the one the README's "
        "backtest numbers describe.",
    )
    manual = threshold_mode.startswith("Manual")
    if market == "ats":
        default_threshold = config.ATS_EDGE_THRESHOLD_POINTS
        slider_value = st.slider(
            "Min. |model spread - market spread| to bet (points)",
            0.0, 10.0, float(default_threshold), 0.25, disabled=not manual,
            help="Every game already has its edge computed; this just redraws which "
            "games clear the bar to bet, live. Lower = more bets, noisier ROI. "
            f"Fixed fallback the pipeline uses before it has enough history: {default_threshold} pts.",
        )
    else:
        default_threshold = config.ML_EDGE_THRESHOLD_PROB
        slider_value = st.slider(
            "Min. |model prob - de-vigged market prob| to bet",
            0.0, 0.20, float(default_threshold), 0.005, format="%.3f", disabled=not manual,
            help="Every game already has its edge computed; this just redraws which "
            "games clear the bar to bet, live. Lower = more bets, noisier ROI. "
            f"Fixed bar the pipeline uses for this market: {default_threshold:.3f}.",
        )
    threshold = float(slider_value) if manual else None
    threshold_label = (
        "the pipeline's per-game threshold" if threshold is None else f"a manual {threshold:.3g} threshold"
    )
    if manual and abs(threshold - default_threshold) > 1e-9:
        st.caption(f"Manual bar differs from the pipeline's fixed fallback ({default_threshold:.3g}).")

    if st.button("Reload data from disk"):
        st.cache_data.clear()

predictions = load_predictions(model)
bets_ctx = regraded_bets(model, market, threshold)

if threshold is None and bets_ctx is not None:
    _pdf = bets_ctx.to_pandas()
    _thr_col = pipeline_threshold_col(_pdf, market)
    _edge_col = selection_edge_col(_pdf, market)
    _selecting = " on the frozen legacy edge" if _edge_col.endswith("_select") else ""
    if _thr_col is not None and bets_ctx[_thr_col].null_count() < bets_ctx.height:
        _thr = bets_ctx[_thr_col].drop_nulls()
        _lo, _hi = _thr.min(), _thr.max()
        _desc = f"fixed {_lo:.3g}" if _lo == _hi else f"per game, median {_thr.median():.3g}, range {_lo:.3g} to {_hi:.3g}"
        st.sidebar.caption(f"Pipeline threshold in use: {_desc}{_selecting}.")
    else:
        _fixed = config.ATS_EDGE_THRESHOLD_POINTS if market == "ats" else config.ML_EDGE_THRESHOLD_PROB
        st.sidebar.caption(
            f"Pipeline threshold in use: fixed {_fixed:.3g} (this market carries no per-game bar){_selecting}."
        )

if bets_ctx is None or predictions is None:
    st.error(
        "No outputs found for this model/market yet. Run `python main.py` first "
        "to generate `outputs/model_results/` and `outputs/backtests/`."
    )
    st.stop()

tab_live, tab_upcoming, tab_matchup, tab_league, tab_track_record, tab_overview, tab_slices, tab_diagnostics, tab_explorer, tab_ratings, tab_sizing, tab_methodology = st.tabs(
    ["Live", "Upcoming picks", "Matchup & trends", "League rankings", "Track record", "Overview", "Backtest slices", "Diagnostics", "Game explorer", "Power ratings", "Bet sizing", "Methodology"]
)

with tab_live:
    import live_tab

    live_tab.render()

# --- Matchup & trends ----------------------------------------------------------

with tab_matchup:
    import matchup_tab

    _mt_up = load_upcoming()
    matchup_tab.render(
        _mt_up.filter(pl.col("model") == model) if _mt_up is not None else None, MODEL_LABELS[model]
    )

with tab_league:
    import matchup_tab as _mt_league

    _mt_league.render_rankings()

# --- Upcoming picks ------------------------------------------------------------

with tab_upcoming:
    st.warning(
        "These are LIVE predictions for games that haven't been played yet -- there is no backtest for them, "
        "because there's nothing to validate a future game against. They inherit whatever skill (or lack of it) "
        "the Diagnostics tab already measured on completed games, nothing more. Treat this as one input alongside "
        "your own judgment, not a signal to follow on its own. Rerun `python main.py` (with fresh data) to refresh "
        "once new lines are posted or more games complete."
    )
    upcoming = load_upcoming()
    model_upcoming = upcoming.filter(pl.col("model") == model) if upcoming is not None else None

    if model_upcoming is None or model_upcoming.height == 0:
        st.info(
            f"No upcoming picks saved for {MODEL_LABELS[model]} yet -- either `python main.py` hasn't been run "
            "since a market line was posted for an unplayed game, or none are posted yet this far out."
        )
    else:
        weeks = sorted(model_upcoming["week"].unique().to_list())
        week_pick = st.selectbox("Week", weeks, index=0, key="upcoming_week")
        wk = model_upcoming.filter(pl.col("week") == week_pick)

        c1, c2 = st.columns(2)
        with c1:
            up_kelly_multiplier = st.slider(
                "Kelly multiplier (suggested stake)", 0.05, 1.0, 0.25, 0.05, key="up_kelly_mult",
                help="Same fractional-Kelly idea as the Bet sizing tab -- 0.25 stakes a quarter of what full Kelly "
                "would suggest.",
            )
        with c2:
            up_kelly_cap = st.slider(
                "Max suggested stake (% of bankroll)", 0.01, 0.25, 0.05, 0.01, format="%.2f", key="up_kelly_cap",
            )

        pdf = wk.sort("gameday").to_pandas()
        pdf["gameday"] = pd.to_datetime(pdf["gameday"]).dt.strftime("%Y-%m-%d")
        if market == "ats":
            edge_col, prob_col, home_odds_col, away_odds_col = "ats_edge", "model_cover_prob", "home_spread_odds", "away_spread_odds"
        else:
            edge_col, prob_col, home_odds_col, away_odds_col = "ml_edge", "model_home_prob", "home_moneyline", "away_moneyline"

        # Same selection rule as backtesting.py, at the sidebar's threshold --
        # by default the per-game bar the pipeline stored on this board
        # (ats_threshold_used / ml_threshold_used) -- routed through the shared
        # helper so the moneyline underdog-only restriction can't go missing.
        pdf["bet_side"] = backtesting.decide_bet_side(
            pdf, market, effective_threshold(pdf, market, threshold), edge_col=selection_edge_col(pdf, market)
        )

        is_home = pdf["bet_side"] == "home"
        prob_for_side = np.where(is_home, pdf[prob_col], 1.0 - pdf[prob_col])
        odds_for_side = np.where(is_home, pdf[home_odds_col], pdf[away_odds_col])
        odds_for_side = pd.Series(odds_for_side).fillna(config.DEFAULT_SPREAD_ODDS).to_numpy()
        stake_frac = kelly_fraction(prob_for_side, odds_for_side) * up_kelly_multiplier
        stake_frac = np.clip(stake_frac, 0.0, up_kelly_cap)
        pdf["suggested_stake_pct"] = np.where(pdf["bet_side"] == "none", 0.0, stake_frac)

        pdf["matchup"] = pdf["away_team"] + " @ " + pdf["home_team"]
        pdf["bet_side_label"] = pdf["bet_side"].map({"home": "Bet home", "away": "Bet away", "none": "No bet"})
        pdf["market_line"] = (
            pdf["spread_line"].map(lambda v: f"{v:+.1f}") if market == "ats"
            else pdf.apply(lambda r: f"{int(r['home_moneyline']):+d} / {int(r['away_moneyline']):+d}", axis=1)
        )
        if market == "ats":
            # predicted_margin uses the same sign convention as spread_line
            # (positive = home favored) -- name the favored team instead of
            # making the reader track the sign.
            favored_team = np.where(pdf["predicted_margin"] >= 0, pdf["home_team"], pdf["away_team"])
            pdf["model_number"] = [f"{t} -{abs(m):.1f}" for t, m in zip(favored_team, pdf["predicted_margin"])]
        else:
            pdf["model_number"] = pdf[prob_col].map(lambda v: f"{v:.0%}")
        pdf["edge_display"] = pdf[edge_col].map(lambda v: f"{v:+.1f} pts" if market == "ats" else f"{v:+.1%}")
        pdf["stake_display"] = pdf["suggested_stake_pct"].map(lambda v: f"{v:.1%}" if v > 0 else "--")

        # Predicted score: the margin model's predicted_margin combined with
        # the (separate, phase-1, not yet validated) totals model's
        # predicted_total, already merged into upcoming_picks.parquet by
        # main.py -- home = (total + margin)/2, away = (total - margin)/2.
        if "predicted_total" in pdf.columns:
            pred_home_score = (pdf["predicted_total"] + pdf["predicted_margin"]) / 2
            pred_away_score = (pdf["predicted_total"] - pdf["predicted_margin"]) / 2
            pdf["predicted_score"] = [
                f"{h:.0f}-{a:.0f}" if pd.notna(h) else "--" for h, a in zip(pred_home_score, pred_away_score)
            ]
        else:
            pdf["predicted_score"] = "--"

        n_bets = int((pdf["bet_side"] != "none").sum())
        st.caption(
            f"{len(pdf)} game(s) in week {week_pick}, {MODEL_LABELS[model]} / {MARKET_LABELS[market]}. "
            f"{n_bets} clear {threshold_label}."
        )

        display_cols = {
            "gameday": "Date", "matchup": "Matchup", "market_line": "Market line",
            "model_number": "Model spread" if market == "ats" else "Model win prob",
            "predicted_score": "Predicted score (H-A)",
            "edge_display": "Edge", "bet_side_label": "Suggested side", "stake_display": "Suggested stake",
        }
        pdf = pdf.sort_values(edge_col, key=lambda s: s.abs(), ascending=False)
        st.dataframe(pdf[list(display_cols)].rename(columns=display_cols), width="stretch", hide_index=True)
        st.caption(
            "Sorted by the size of the model's disagreement with the market. \"Suggested stake\" is the same "
            "fractional-Kelly math as the Bet sizing tab, just applied to a game with no result yet -- treat it as "
            "a starting point for your own sizing, not an instruction. \"Predicted score\" combines this model's "
            "margin prediction with a separate, phase-1 totals model that hasn't shown real skill yet (see the "
            "Methodology tab) -- read it as illustrative, not a confident forecast."
        )

# --- Track record --------------------------------------------------------------

with tab_track_record:
    st.warning(
        "This grades the model's ACTUAL logged picks -- what it said before each game, not a backtest recomputed "
        "after the fact -- against real results, as they come in. It only has data from the point this logging "
        "started; it is not a substitute for the full historical backtest in the other tabs, just the genuine, "
        "un-replayable record of what was actually shown before kickoff each week."
    )
    log = load_picks_log()
    if log is None or log.height == 0:
        st.info(
            "No picks have been logged yet. Each `python main.py --refresh` run appends a snapshot of that "
            "week's Upcoming picks to a permanent log (`live_picks.append_to_picks_log`) -- once the season "
            "starts and a few weekly refreshes have run, graded results will show up here as those games "
            "complete."
        )
    else:
        model_log = log.filter(pl.col("model") == model)
        graded = model_log.filter(pl.col("home_margin").is_not_null())
        ungraded = model_log.filter(pl.col("home_margin").is_null())

        if graded.height == 0:
            st.info(
                f"{model_log.height} pick(s) logged for {MODEL_LABELS[model]} so far, none of those games have "
                f"finished yet ({ungraded.height} still pending). Check back after they're played."
            )
        else:
            pdf = graded.to_pandas()
            edge_col = "ats_edge" if market == "ats" else "ml_edge"
            prob_col = "model_cover_prob" if market == "ats" else "model_home_prob"
            home_odds_col = "home_spread_odds" if market == "ats" else "home_moneyline"
            away_odds_col = "away_spread_odds" if market == "ats" else "away_moneyline"

            pdf["bet_side"] = backtesting.decide_bet_side(
                pdf, market, effective_threshold(pdf, market, threshold), edge_col=selection_edge_col(pdf, market)
            )
            if market == "ats":
                home_right = pdf["home_margin"] > pdf["spread_line"]
            else:
                home_right = pdf["home_margin"] > 0
            bet_won = np.where(pdf["bet_side"] == "home", home_right, np.where(pdf["bet_side"] == "away", ~home_right, False))
            bet_odds = np.where(pdf["bet_side"] == "home", pdf[home_odds_col], pdf[away_odds_col])
            bet_odds = pd.Series(bet_odds).fillna(config.DEFAULT_SPREAD_ODDS).to_numpy()
            profit = np.where(pdf["bet_side"] == "none", 0.0, np.where(bet_won, american_payout_multiplier(bet_odds), -1.0))
            pdf["bet_won"] = np.where(pdf["bet_side"] == "none", np.nan, bet_won.astype(float))
            pdf["profit_units"] = profit

            placed = pdf[pdf["bet_side"] != "none"]
            n_bets = len(placed)
            n_wins = int(placed["bet_won"].sum()) if n_bets else 0
            roi = placed["profit_units"].sum() / n_bets if n_bets else float("nan")

            c1, c2, c3 = st.columns(3)
            c1.metric("Graded picks placed", f"{n_bets:,}", help=f"{graded.height} games graded total at this model; {ungraded.height} still pending.")
            c2.metric("Record", f"{n_wins}-{n_bets - n_wins}" if n_bets else "--")
            c3.metric("ROI (flat stakes)", f"{roi:+.1%}" if n_bets else "--")

            if n_bets:
                pdf_placed = placed.sort_values("gameday").reset_index(drop=True)
                pdf_placed["cum_profit"] = pdf_placed["profit_units"].cumsum()
                fig = go.Figure()
                fig.add_scatter(
                    x=pdf_placed["gameday"], y=pdf_placed["cum_profit"], mode="lines+markers",
                    line=dict(color=BLUE, width=2), marker=dict(size=5), name="Cumulative profit",
                )
                fig.add_hline(y=0, line_width=1, line_color=GRID)
                fig.update_layout(
                    title="Cumulative profit on actually-logged picks (flat 1-unit stakes)",
                    xaxis_title="Game date", yaxis_title="Profit (units)",
                    showlegend=False, **CHART_LAYOUT,
                )
                fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
                fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
                st.plotly_chart(fig, width="stretch")

            display = pdf.copy()
            display["matchup"] = display["away_team"] + " @ " + display["home_team"]
            display["bet_side_label"] = display["bet_side"].map({"home": "Bet home", "away": "Bet away", "none": "No bet"})
            display["result"] = np.where(display["bet_won"] == 1.0, "Won", np.where(display["bet_won"] == 0.0, "Lost", "--"))
            st.dataframe(
                display[["gameday", "matchup", edge_col, "bet_side_label", "result", "profit_units"]]
                .rename(columns={"gameday": "Date", "matchup": "Matchup", edge_col: "Edge", "bet_side_label": "Side", "result": "Result", "profit_units": "Profit"})
                .sort_values("Date", ascending=False),
                width="stretch", hide_index=True,
            )
            st.caption(
                f"{MODEL_LABELS[model]} / {MARKET_LABELS[market]}, graded at {threshold_label}. "
                "This can change if the sidebar threshold is adjusted -- bet_side is re-decided live from the "
                "logged edge, same as every other tab, but the underlying prediction/edge itself is exactly what "
                "was logged before that game, not recomputed."
            )

# --- Overview ----------------------------------------------------------------

with tab_overview:
    placed = bets_ctx.filter(pl.col("bet_side") != "none")
    decided = placed.filter(~pl.col("is_push"))
    n_bets = placed.height
    n_wins = decided.filter(pl.col("bet_won")).height
    n_losses = decided.filter(~pl.col("bet_won")).height
    win_rate = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else float("nan")
    total_staked = placed["staked"].sum()
    total_profit = placed["profit_units"].sum()
    roi = total_profit / total_staked if total_staked else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bets placed", f"{n_bets:,}")
    c2.metric("Win rate", f"{win_rate:.1%}" if win_rate == win_rate else "n/a")
    c3.metric("ROI", f"{roi:+.1%}" if roi == roi else "n/a")
    c4.metric("Profit (units)", f"{total_profit:+.1f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(roi_by_season_line(evaluation.summarize_with_ci(bets_ctx, ["season"])), width="stretch")
    with col_b:
        st.plotly_chart(cumulative_profit_line(bets_ctx), width="stretch")
    st.caption("Error bars are bootstrap 95% confidence intervals on ROI -- where they cross zero, that season's result isn't distinguishable from no edge at all.")

# --- Backtest slices -----------------------------------------------------------

with tab_slices:
    all_seasons = sorted(bets_ctx["season"].unique().to_list())
    season_lo, season_hi = st.select_slider(
        "Season range", options=all_seasons, value=(all_seasons[0], all_seasons[-1]),
        help="Restrict every chart below to this season range -- e.g. narrow it to the last few "
        "seasons to check whether recent results look different from the full history.",
    )
    slice_bets = bets_ctx.filter(pl.col("season").is_between(season_lo, season_hi))
    st.caption(
        f"{slice_bets.height:,} games, seasons {season_lo}-{season_hi}. Error bars are bootstrap 95% "
        "confidence intervals on ROI. A bar whose interval crosses zero is not statistically distinguishable from breakeven."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        edge_df = bucket_axis(evaluation.summarize_with_ci(slice_bets, ["edge_bucket"]), "edge_bucket", EDGE_BUCKET_ORDER, EDGE_BUCKET_LABELS)
        edge_title = "ROI by ATS edge (pts)" if market == "ats" else "ROI by ML edge (prob pts)"
        st.plotly_chart(diverging_bar(edge_df, "_label", edge_title, "Model vs. market edge"), width="stretch")
    with col_b:
        spread_df = bucket_axis(evaluation.summarize_with_ci(slice_bets, ["spread_bucket"]), "spread_bucket", SPREAD_BUCKET_ORDER, SPREAD_BUCKET_LABELS)
        st.plotly_chart(diverging_bar(spread_df, "_label", "ROI by market spread size", "|Market spread|"), width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        ha_df = evaluation.summarize_with_ci(slice_bets, ["is_home_bet"]).with_columns(
            pl.col("is_home_bet").replace_strict({True: "Home", False: "Away"}).alias("_label")
        )
        st.plotly_chart(diverging_bar(ha_df, "_label", "ROI: home vs. away bets"), width="stretch")
    with col_d:
        fd_df = evaluation.summarize_with_ci(slice_bets, ["is_favorite_bet"]).with_columns(
            pl.col("is_favorite_bet").replace_strict({True: "Favorite", False: "Underdog"}).alias("_label")
        )
        st.plotly_chart(diverging_bar(fd_df, "_label", "ROI: favorite vs. underdog bets"), width="stretch")

    week_df = bucket_axis(evaluation.summarize_with_ci(slice_bets, ["week_bucket"]), "week_bucket", WEEK_BUCKET_ORDER, WEEK_BUCKET_LABELS)
    st.plotly_chart(diverging_bar(week_df, "_label", "ROI by time of season", "Week"), width="stretch")
    st.caption(
        "Playoffs are grouped by game_type, not week number, since playoff games continue the week count "
        "rather than resetting (and the regular season itself went 17 -> 18 weeks in 2021, so raw week number "
        "isn't comparable across eras past week 13 on its own). Checked directly against the hypothesis that "
        "early weeks should be worse (the model has no signal about offseason roster/coaching changes yet): the "
        "MAE-vs-market gap (Diagnostics tab's metric, not this threshold-dependent ROI) actually runs SMALLEST "
        "in Weeks 1-3 for all three models and PEAKS mid-season (Weeks 9-13), not at the start -- see the "
        "Methodology tab for the full numbers."
    )

    st.plotly_chart(diverging_hbar_by_team(evaluation.summarize_with_ci(slice_bets, ["bet_team"])), width="stretch")

    st.divider()
    st.subheader("What if the threshold were chosen automatically each season?")
    st.caption(
        "For each test season, grid-searches the bet-selection threshold using ONLY strictly earlier seasons -- "
        "picking whichever threshold maximized the bootstrap 95% CI LOWER BOUND on ROI (not raw mean ROI, which "
        "rewards a threshold that got lucky on a handful of bets) -- then applies that season's choice "
        "out-of-sample. Same walk-forward discipline as the models themselves, just for a hyperparameter instead "
        "of a prediction. Ignores the season-range slider above (it needs its own full walk-forward)."
    )
    opt = threshold_optimizer(model, market)
    if opt is None or opt.empty:
        st.info("Not enough seasons/bets to run the threshold optimizer.")
    else:
        opt_seasons = opt["season"].tolist()
        fixed_default = config.ATS_EDGE_THRESHOLD_POINTS if market == "ats" else config.ML_EDGE_THRESHOLD_PROB
        fixed_bets = regraded_bets(model, market, fixed_default)
        fixed_same_span = fixed_bets.filter(pl.col("season").is_in(opt_seasons)).filter(pl.col("bet_side") != "none").to_pandas()
        fixed_profit = float(fixed_same_span["profit_units"].sum())
        fixed_staked = float(fixed_same_span["staked"].sum())
        fixed_roi = fixed_profit / fixed_staked if fixed_staked else float("nan")

        adaptive_bets = int(opt["n_bets"].sum())
        adaptive_roi = float(opt["profit"].sum() / adaptive_bets) if adaptive_bets else float("nan")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Fixed threshold ({fixed_default:.3g})", f"{fixed_roi:+.1%} ROI", help=f"{len(fixed_same_span)} bets, seasons {opt_seasons[0]}-{opt_seasons[-1]}.")
        c2.metric("Adaptive walk-forward threshold", f"{adaptive_roi:+.1%} ROI", help=f"{adaptive_bets} bets, same seasons.")
        c3.metric(
            "Threshold stability (std dev)", f"{opt['chosen_threshold'].std():.2f}",
            help="How much the season-to-season chosen threshold swings. High = noisy, not a robust signal.",
        )

        fig = go.Figure()
        fig.add_bar(x=opt["season"], y=opt["chosen_threshold"], marker_color=BLUE, name="Chosen threshold")
        fig.add_hline(y=fixed_default, line_width=1.5, line_dash="dash", line_color=INK_SECONDARY)
        fig.update_layout(
            title="Threshold chosen each season (dashed line = current fixed default)",
            xaxis_title="Season", yaxis_title="Threshold",
            showlegend=False, **CHART_LAYOUT,
        )
        fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
        fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Measured directly over seasons {opt_seasons[0]}-{opt_seasons[-1]}: the adaptive threshold's ROI "
            f"({adaptive_roi:+.1%}) comes out {'better' if adaptive_roi > fixed_roi else 'worse'} than the fixed "
            f"{fixed_default:.3g} default ({fixed_roi:+.1%}), on roughly "
            f"{(adaptive_bets / len(fixed_same_span) if len(fixed_same_span) else float('nan')):.0%} as many bets. "
            "More telling than either ROI number: look at how much the chosen threshold swings season to season "
            "above. That instability is itself evidence this optimizer is mostly chasing noise in a small "
            "per-season sample, not locking onto one real, stable threshold -- treat any single season's "
            "\"optimal\" number with real skepticism, not as a discovery."
        )

# --- Diagnostics -----------------------------------------------------------

with tab_diagnostics:
    st.subheader("Model skill vs. the market baseline")
    st.caption(
        "The market's own closing line, read directly as its prediction (spread_line as its margin call, "
        "the de-vigged moneyline as its win probability) -- the simplest baseline there is. Lower is better "
        "for all three metrics. This is independent of any bet-selection threshold: it asks whether the "
        "model's raw prediction is more accurate than just reading the market line, full stop."
    )
    skill = load_skill(model)
    if skill is None:
        st.info("No skill-vs-market file found for this model. Re-run `python main.py` to generate it.")
    else:
        cols = st.columns(3)
        metric_labels = {"mae": "MAE (points)", "rmse": "RMSE (points)", "brier": "Brier score (probability)"}
        for i, row in enumerate(skill.iter_rows(named=True)):
            model_val, market_val = row["model"], row["market"]
            delta = market_val - model_val  # positive => model has lower error => model wins
            cols[i].metric(
                metric_labels[row["metric"]],
                f"{model_val:.4f}",
                delta=f"{delta:+.4f} vs. market",
                delta_color="normal" if delta > 0 else "inverse",
                help=f"Market baseline: {market_val:.4f} (n={row['n']:,}). A positive delta means the model beats the market.",
            )
        n_beats = sum(1 for row in skill.iter_rows(named=True) if row["model"] < row["market"])
        if n_beats == 0:
            st.warning("The model is worse than the market baseline on all three metrics here -- treat any positive backtest ROI for this model/market as very likely noise, not skill.")
        elif n_beats < skill.height:
            st.warning(f"The model beats the market baseline on {n_beats}/{skill.height} metrics -- a mixed result, not a clean win.")
        else:
            st.success("The model beats the market baseline on every metric here.")

    st.divider()
    st.subheader("Calibration (reliability curve)")
    st.caption("Games are grouped into 10 equal-sized bins by predicted probability. A well-calibrated model's line sits on the diagonal: when it says 65%, that side should win about 65% of the time.")
    if market == "ats":
        outcome_expr = pl.col("home_margin") > pl.col("spread_line")
        cal_df = evaluation.calibration_bins(bets_ctx, "model_cover_prob", outcome_expr)
        cal_title = "ATS: predicted cover probability vs. actual cover rate"
    else:
        outcome_expr = pl.col("home_margin") > 0
        cal_df = evaluation.calibration_bins(bets_ctx, "model_home_prob", outcome_expr)
        cal_title = "ML: predicted home win probability vs. actual home win rate"
    if cal_df.height == 0:
        st.info("Not enough data to build a calibration curve.")
    else:
        st.plotly_chart(reliability_chart(cal_df, cal_title), width="stretch")

# --- Game explorer -------------------------------------------------------------

with tab_explorer:
    # Games with a posted line but no result yet, from the same board the
    # Upcoming picks tab reads -- so a week that hasn't been played can be
    # explored here too.
    explorer_upcoming = load_upcoming()
    if explorer_upcoming is not None:
        explorer_upcoming = explorer_upcoming.filter(pl.col("model") == model)
    else:
        explorer_upcoming = pl.DataFrame(schema={c: pl.String for c in ["game_id", "home_team", "away_team"]} | {"season": pl.Int64, "week": pl.Int64})

    seasons = sorted(set(bets_ctx["season"].unique().to_list()) | set(explorer_upcoming["season"].unique().to_list()))
    season = st.selectbox("Season", seasons, index=len(seasons) - 1)
    season_bets = bets_ctx.filter(pl.col("season") == season)
    season_upcoming = explorer_upcoming.filter(pl.col("season") == season)

    st.subheader("Model vs. market, game by game")
    st.caption(
        "Each point is one game this season. Points above the diagonal: model more bullish on home than the "
        "market. Marker shape grades the model's own call: ● a bet it won, ✕ a bet it lost, "
        "◆ a push (stake returned), gray ● a game it sat out. (For calibration across all seasons, "
        "see the Diagnostics tab.)"
    )
    if season_bets.height == 0:
        st.info("No completed games yet this season -- pick an upcoming week below.")
    else:
        if market == "ats":
            fig = calibration_scatter(season_bets, "spread_line", "predicted_margin", f"{season}: model spread vs. market spread", "spread (home margin)")
        else:
            fig = calibration_scatter(season_bets, "market_home_prob_no_vig", "model_home_prob", f"{season}: model home win prob vs. de-vigged market prob", "home win probability")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Games by week")
    played_weeks = set(season_bets["week"].unique().to_list())
    upcoming_weeks = set(season_upcoming["week"].unique().to_list())

    def _week_label(w: int) -> str:
        if w not in upcoming_weeks:
            return str(w)
        return f"{w} (upcoming)" if w not in played_weeks else f"{w} (partly upcoming)"

    weeks = sorted(played_weeks | upcoming_weeks)
    week = st.selectbox("Week", weeks, format_func=_week_label)
    week_bets = season_bets.filter(pl.col("week") == week).sort("game_id")
    week_bets = week_bets.join(load_game_scores(), on="game_id", how="left")
    week_upcoming = season_upcoming.filter(pl.col("week") == week)
    if week_upcoming.height > 0:
        week_upcoming = week_upcoming.sort(["gameday", "game_id"])

    def _board_favorite(frame: pd.DataFrame) -> list[str]:
        # nflverse's raw spread_line sign (positive = home favored) is the
        # opposite of how a sportsbook board reads (favorite shown negative) --
        # spell out who's favored and by how much in familiar board notation
        # instead of making the reader convert the sign themselves.
        fav_team = np.where(frame["spread_line"] > 0, frame["home_team"], frame["away_team"])
        return ["Pick'em" if s == 0 else f"{t} -{abs(s):.1f}" for t, s in zip(fav_team, frame["spread_line"])]

    def _predicted_margin_display(frame: pd.DataFrame) -> list[str]:
        # predicted_margin uses the same sign convention as spread_line
        # (positive = home favored) -- show it the way a board would,
        # naming which team it is rather than making the reader track the sign.
        favored_team = np.where(frame["predicted_margin"] >= 0, frame["home_team"], frame["away_team"])
        return [f"{t} -{abs(m):.1f}" for t, m in zip(favored_team, frame["predicted_margin"])]

    def _predicted_score(frame: pd.DataFrame) -> list[str]:
        # Predicted score: the margin model already gives predicted_margin; the
        # (separate, phase-1) totals model gives predicted_total for the SAME
        # game. Combined algebraically (no new model needed): home = (total +
        # margin)/2, away = (total - margin)/2. Purely a display convenience --
        # inherits both models' own error, and the totals model isn't validated
        # to have any real skill yet (see README's "Totals -- phase 1").
        if "predicted_total" not in frame:
            return ["--"] * len(frame)
        pred_home_score = (frame["predicted_total"] + frame["predicted_margin"]) / 2
        pred_away_score = (frame["predicted_total"] - frame["predicted_margin"]) / 2
        return [f"{h:.0f}-{a:.0f}" if pd.notna(h) else "--" for h, a in zip(pred_home_score, pred_away_score)]

    rename = {"favorite": "Favorite", "total_points": "Total points", "predicted_score": "Predicted score (H-A)"}

    if week_bets.height > 0:
        wpdf = week_bets.to_pandas()
        wpdf["favorite"] = _board_favorite(wpdf)
        wpdf["total_points"] = wpdf["home_score"] + wpdf["away_score"]
        total_preds = load_predictions(TOTAL_MODEL_NAME[model])
        if total_preds is not None:
            wpdf = wpdf.merge(
                total_preds.select(["game_id", "predicted_total"]).to_pandas(), on="game_id", how="left"
            )
        wpdf["predicted_score"] = _predicted_score(wpdf)
        wpdf["predicted_margin"] = _predicted_margin_display(wpdf)

        if market == "ats":
            cols = [
                "home_team", "away_team", "spread_line", "favorite", "predicted_margin", "ats_edge",
                "model_cover_prob", "bet_side", "home_margin", "total_points", "predicted_score", "bet_won", "profit_units",
            ]
        else:
            cols = [
                "home_team", "away_team", "favorite", "home_moneyline", "away_moneyline",
                "market_home_prob_no_vig", "model_home_prob", "ml_edge",
                "bet_side", "home_margin", "total_points", "predicted_score", "bet_won", "profit_units",
            ]
        if week_upcoming.height > 0:
            st.markdown("**Played**")
        st.dataframe(wpdf[cols].rename(columns=rename), width="stretch", hide_index=True)

    if week_upcoming.height > 0:
        st.markdown("**Not played yet**")
        st.caption(
            "Numbers from the live model (trained on every completed game to date), the same ones the "
            "Upcoming picks tab shows. Bet side uses the sidebar threshold, as there."
        )
        updf = week_upcoming.to_pandas()
        updf["favorite"] = _board_favorite(updf)
        updf["predicted_score"] = _predicted_score(updf)
        updf["predicted_margin"] = _predicted_margin_display(updf)
        updf["gameday"] = pd.to_datetime(updf["gameday"]).dt.strftime("%Y-%m-%d")
        updf["bet_side"] = backtesting.decide_bet_side(
            updf, market, effective_threshold(updf, market, threshold), edge_col=selection_edge_col(updf, market)
        )
        if market == "ats":
            up_cols = [
                "gameday", "home_team", "away_team", "spread_line", "favorite", "predicted_margin", "ats_edge",
                "model_cover_prob", "bet_side", "predicted_score",
            ]
        else:
            up_cols = [
                "gameday", "home_team", "away_team", "favorite", "home_moneyline", "away_moneyline",
                "market_home_prob_no_vig", "model_home_prob", "ml_edge", "bet_side", "predicted_score",
            ]
        st.dataframe(updf[up_cols].rename(columns=rename), width="stretch", hide_index=True)

    if week_bets.height == 0 and week_upcoming.height == 0:
        st.info("No games this week.")
    else:
        upcoming_ids = set(week_upcoming["game_id"].to_list())
        game_rows = (
            week_bets.select(["game_id", "home_team", "away_team"]).to_dicts()
            + week_upcoming.select(["game_id", "home_team", "away_team"]).to_dicts()
        )
        game_labels = {
            r["game_id"]: f'{r["away_team"]} @ {r["home_team"]}' + (" (upcoming)" if r["game_id"] in upcoming_ids else "")
            for r in game_rows
        }
        game_id = st.selectbox("Game", list(game_labels), format_func=lambda g: game_labels[g])
        is_upcoming = game_id in upcoming_ids

        st.subheader("Why did the model say this?")
        contributions = load_contributions(model, upcoming=is_upcoming)
        if contributions is None and is_upcoming and load_contributions(model) is not None:
            st.info(
                "No breakdown saved for upcoming games yet -- rerun the refresh (`python main.py --refresh`) "
                "to generate one."
            )
        elif contributions is None:
            st.info(
                "Per-game breakdowns are only available for the ridge_baseline model. LightGBM's predictions come "
                "from hundreds of tree splits and don't decompose into simple additive terms the way a linear "
                "model's do -- switch to 'Ridge baseline' in the sidebar to see this."
            )
        else:
            contrib_row = contributions.filter(pl.col("game_id") == game_id)
            if contrib_row.height == 0:
                st.info(
                    "No breakdown saved for this game -- rerun the refresh to regenerate it."
                    if is_upcoming else
                    "No breakdown saved for this game (likely outside the model's predictable seasons)."
                )
            else:
                contrib_cols = [c for c in contributions.columns if c.endswith("_contrib")]
                values = contrib_row.select(contrib_cols).to_pandas().iloc[0]
                values.index = [c.removesuffix("_contrib").replace("_", " ") for c in values.index]
                values = values.sort_values()
                colors = [BLUE if v >= 0 else RED for v in values.to_numpy()]
                total = float(values.sum())
                fig = go.Figure()
                fig.add_bar(x=values.to_numpy(), y=values.index, orientation="h", marker_color=colors)
                fig.add_vline(x=0, line_width=1, line_color=GRID)
                fig.update_layout(
                    title=f"{game_labels[game_id]}: predicted home margin = {total:+.1f} pts",
                    xaxis_title="Contribution to predicted home margin (points)",
                    showlegend=False, height=420,
                    **CHART_LAYOUT,
                )
                fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
                fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "Each bar is one feature's contribution (coefficient x its value for this game) to the ridge "
                    "model's predicted home margin. Bars sum to the total."
                    + (
                        " For an upcoming game this is the live model, fit on every completed game to date, so its "
                        "coefficients differ slightly from the walk-forward fold behind that season's played games."
                        if is_upcoming else ""
                    )
                )

        st.subheader("Games like this one")
        st.caption(
            "The 10 historical games (strictly before this one) closest in feature space -- Elo diff, recent "
            "form, rest, weather, QB continuity, etc. This is an exploration tool, not a prediction: it isn't "
            "part of the backtest and carries no claim of an edge, just \"here's what happened in games that "
            "looked similar going in.\""
        )
        neighbors = similar_games(game_id, k=10)
        if neighbors.height == 0:
            st.info("No prior games to compare against (likely one of the first tracked games in the dataset).")
        else:
            npdf = neighbors.to_pandas()
            cover_rate = npdf["home_covered"].mean()
            win_rate = npdf["home_won"].mean()
            c1, c2 = st.columns(2)
            c1.metric("Home covered (among neighbors)", f"{cover_rate:.0%}", help=f"{int(npdf['home_covered'].sum())} of {len(npdf)}")
            c2.metric("Home won (among neighbors)", f"{win_rate:.0%}", help=f"{int(npdf['home_won'].sum())} of {len(npdf)}")

            display = npdf.copy()
            display["matchup"] = display["away_team"] + " @ " + display["home_team"]
            display = display[[
                "season", "week", "matchup", "spread_line", "home_margin",
                "home_covered", "home_won", "home_moneyline", "away_moneyline", "distance",
            ]]
            st.dataframe(display, width="stretch", hide_index=True)

# --- Power ratings -------------------------------------------------------------

with tab_ratings:
    st.caption(
        "The Elo power rating every model uses, entering each game. It regresses 33% back to the "
        "mean each offseason, a reasonable prior for an average team but one that erases persistent "
        "organizational quality faster than the market does -- the long-standing complaint about "
        "this rating was that it underrates perennially strong franchises and overrates weak ones. "
        "The team-bias section below is that complaint as a number, and it is also where market "
        "anchoring shows up: mean absolute excess bias across the 32 teams is 0.44 points per game "
        "for the ridge baseline against 0.12 for the anchored ensemble. Switch models in the "
        "sidebar to compare."
    )
    elo_hist = load_elo_history()
    if elo_hist is None:
        st.info("No Elo rating history found. Run `python main.py` first.")
    else:
        chart_hist = elo_trajectory(elo_hist)
        teams = sorted(chart_hist["team"].unique().to_list())
        default_teams = [t for t in ["NE", "NYJ"] if t in teams] or teams[:2]
        selected_teams = st.multiselect(
            "Teams to compare", teams, default=default_teams,
            help="NE and NYJ are pre-selected as the most- and least-underrated franchises found in the residual-bias check.",
        )
        palette_cycle = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948", "#008300"]
        if not selected_teams:
            st.info("Pick at least one team.")
        else:
            fig = go.Figure()
            for i, team in enumerate(selected_teams[:8]):
                sub = chart_hist.filter(pl.col("team") == team).to_pandas()
                fig.add_scatter(
                    x=sub["gameday"], y=sub["elo_pre"], mode="lines", name=team,
                    line=dict(color=palette_cycle[i % len(palette_cycle)], width=2),
                    customdata=sub[["season", "week"]],
                    hovertemplate=f"{team}<br>" + "%{x|%Y-%m-%d}<br>Elo %{y:.0f}<br>Season %{customdata[0]} wk %{customdata[1]}<extra></extra>",
                )
            fig.add_hline(y=config.ELO_INITIAL, line_width=1, line_color=GRID)
            fig.update_layout(
                title="Elo rating entering each game (line ends on the current rating)",
                yaxis_title="Elo (pre-game)", xaxis_title="Date",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                **CHART_LAYOUT,
            )
            fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
            fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
            st.plotly_chart(fig, width="stretch")

        st.subheader("Current ratings (after each team's most recent completed game)")
        latest = current_elo(elo_hist)
        as_of = latest.filter(pl.col("reflects_last_result")).select("as_of_season", "as_of_week")
        if as_of.height:
            st.caption(
                f"Includes every result through season {as_of['as_of_season'].max()}, "
                f"week {as_of['as_of_week'].max()}. A scheduled game that has not been played "
                "does not move a rating, so this table only advances as results come in."
            )
        st.dataframe(latest.to_pandas(), width="stretch", hide_index=True)

        # --- Does the model rate any team differently from the market? -------
        st.divider()
        st.subheader("Where this model disagrees with the market, by team")
        bias = team_bias_table(model)
        if bias is None:
            st.info("No graded predictions for this model yet.")
        else:
            st.caption(
                "The caption at the top of this tab has always asserted that the rating misjudges "
                "certain franchises; this is that claim as a number. Positive means the source "
                "expected that team to do better than it actually did, i.e. overrated it. Only the "
                "third column is about the MODEL: a team the market reads the same way is not a "
                "model problem, it is a hard team to rate. Computed over every graded game for "
                f"{MODEL_LABELS[model]}, from each team's own perspective."
            )
            bpdf = bias.to_pandas()
            mean_abs = bpdf["excess"].abs().mean()
            worst = bpdf.iloc[bpdf["excess"].abs().argmax()]
            c1, c2 = st.columns(2)
            c1.metric(
                "Mean absolute excess bias", f"{mean_abs:.2f} pts",
                help="Averaged across all 32 teams. How far the model's read of a typical team "
                "sits from the market's, in points per game.",
            )
            c2.metric(
                "Largest disagreement", f"{worst['team']}  {worst['excess']:+.2f} pts",
                help="The team this model rates most differently from the market.",
            )
            st.plotly_chart(
                diverging_team_bars(
                    bpdf, "excess",
                    "Model bias minus market bias (points per game)",
                    "Excess bias vs the market (points)",
                    pos_label="Model is higher on this team",
                    neg_label="Model is lower on this team",
                    customdata=["model_bias", "market_bias", "games"],
                    hovertemplate=(
                        "%{y}<br>Excess vs market: %{x:+.2f} pts"
                        "<br>Model bias: %{customdata[0]:+.2f}"
                        "<br>Market bias: %{customdata[1]:+.2f}"
                        "<br>Games: %{customdata[2]}<extra></extra>"
                    ),
                ),
                width="stretch",
            )

        # --- One team, in depth ---------------------------------------------
        st.divider()
        st.subheader("Team detail")
        log = team_log_with_predictions(model)
        if log is None:
            st.info("No modeling table found. Run `python main.py` first.")
        else:
            all_teams = sorted(log["team"].unique().to_list())
            rank_lookup = {r["team"]: i + 1 for i, r in enumerate(latest.to_dicts())}
            c1, c2 = st.columns([2, 3])
            team = c1.selectbox("Team", all_teams, index=all_teams.index(default_teams[0]) if default_teams else 0)
            seasons_avail = sorted(log["season"].unique().to_list(), reverse=True)
            season_pick = c2.selectbox("Season", seasons_avail, index=0)

            tl = log.filter(pl.col("team") == team).sort(["season", "week"])
            tl_season = tl.filter(pl.col("season") == season_pick)
            played = tl_season.filter(pl.col("margin").is_not_null())
            upcoming = tl_season.filter(pl.col("margin").is_null())

            row = latest.filter(pl.col("team") == team)
            cur_elo = float(row["elo"][0]) if row.height else float("nan")
            wins = int((played["result"] == "W").sum())
            losses = int((played["result"] == "L").sum())
            covers = int((played["ats_result"] == "Cover").sum())
            no_covers = int((played["ats_result"] == "No cover").sum())
            season_elo_change = float(played["elo_change"].sum()) if played.height else 0.0
            sos_played = float(played["opp_elo_pre"].mean()) if played.height else float("nan")
            sos_left = float(upcoming["opp_elo_pre"].mean()) if upcoming.height else float("nan")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Current Elo", f"{cur_elo:.0f}",
                delta=f"{season_elo_change:+.0f} in {season_pick}" if played.height else None,
                help=f"League rank {rank_lookup.get(team, '--')} of {latest.height}. "
                f"{config.ELO_INITIAL:.0f} is average; ratings regress "
                f"{config.SEASON_REGRESSION_TO_MEAN:.0%} toward it each offseason.",
            )
            m2.metric(f"{season_pick} record", f"{wins}-{losses}",
                      help="Regular season and playoffs, completed games only.")
            m3.metric("Against the spread", f"{covers}-{no_covers}",
                      help="How often this team beat the closing line. Pushes excluded.")
            m4.metric(
                "Opponent Elo faced", f"{sos_played:.0f}" if played.height else "--",
                delta=f"{sos_left - sos_played:+.0f} remaining" if upcoming.height and played.height else None,
                delta_color="inverse",
                help="Average rating of opponents played, and how much harder or easier what is "
                "left looks. A positive delta means a tougher run-in.",
            )

            # Rating trajectory for this one team. Single series, so no legend
            # box -- the title names it (see the accessibility rule).
            tpdf = tl.filter(pl.col("elo_pre").is_not_null()).to_pandas()
            fig = go.Figure()
            fig.add_scatter(
                x=tpdf["gameday"], y=tpdf["elo_pre"], mode="lines",
                line=dict(color=BLUE, width=2), name=team,
                customdata=tpdf[["season", "week", "opponent", "result"]],
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Elo %{y:.0f}"
                    "<br>%{customdata[0]} wk %{customdata[1]} vs %{customdata[2]}"
                    "<br>Result: %{customdata[3]}<extra></extra>"
                ),
            )
            fig.add_hline(y=config.ELO_INITIAL, line_width=1, line_color=GRID)
            fig.update_layout(
                title=f"{team} Elo rating entering each game, all seasons",
                yaxis_title="Elo (pre-game)", xaxis_title="Date",
                showlegend=False, height=320, **CHART_LAYOUT,
            )
            fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
            fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
            st.plotly_chart(fig, width="stretch")

            # Form: the model's own inputs, with league context. Stat tiles
            # rather than a chart -- each of these is one number, and a chart
            # of one number per metric is the classic wrong form.
            st.markdown("**Form entering the next game, against the rest of the league**")
            st.caption(
                "These are the actual model inputs, recency-weighted over prior games and carried "
                "forward from the last completed game. The percentile is against every team's "
                f"current value in {season_pick}. Penalty counts are shown because they were "
                "measured this session and rejected as model features (see Methodology) -- they are "
                "here to look at, not because the model bets on them."
            )
            latest_row = tl_season.sort(["season", "week"]).tail(1)
            league_now = (
                log.filter(pl.col("season") == season_pick)
                .sort(["team", "season", "week"])
                .group_by("team", maintain_order=False)
                .agg([pl.col(f).last().alias(f) for f, _, _, _ in TEAM_FORM_FIELDS if f in log.columns])
            )
            if latest_row.height:
                fields = [f for f in TEAM_FORM_FIELDS if f[0] in latest_row.columns]
                for chunk_start in range(0, len(fields), 3):
                    cols = st.columns(3)
                    for col, (stem, label, fmt, lower_better) in zip(cols, fields[chunk_start:chunk_start + 3]):
                        val = latest_row[stem][0]
                        if val is None:
                            col.metric(label, "--")
                            continue
                        vals = league_now[stem].drop_nulls().to_numpy()
                        pct = float((vals < val).mean() * 100) if len(vals) else float("nan")
                        if lower_better:
                            pct = 100 - pct
                        col.metric(
                            label, fmt.format(val),
                            help=f"{pct:.0f}th percentile in the league "
                            f"({'lower is better' if lower_better else 'higher is better'}).",
                        )

            # Injuries by unit -- the decomposition, which is the legible part
            # even though the model was measured not to benefit from it.
            inj_fields = [(f, lbl) for f, lbl in TEAM_INJURY_FIELDS if f in tl_season.columns]
            if latest_row.height and inj_fields:
                st.markdown("**Injury weight by unit, entering the next game**")
                st.caption(
                    "Each player on the official report counted as severity (out 1.0, doubtful 0.5, "
                    "questionable 0.25) times their own recent snap share, so a starter weighs far "
                    "more than a backup. Next week's report does not exist yet, so a zero there "
                    "means 'not published', not 'nobody hurt'."
                )
                cols = st.columns(len(inj_fields))
                for col, (stem, lbl) in zip(cols, inj_fields):
                    v = latest_row[stem][0]
                    col.metric(lbl, f"{float(v):.2f}" if v is not None else "--")

            st.markdown(f"**{season_pick} game log**")
            disp = tl_season.select(
                pl.col("week").alias("Wk"),
                pl.when(pl.col("is_home")).then(pl.lit("vs")).otherwise(pl.lit("@")).alias(""),
                pl.col("opponent").alias("Opp"),
                pl.col("opp_elo_pre").round(0).alias("Opp Elo"),
                pl.col("points_for").alias("PF"),
                pl.col("points_against").alias("PA"),
                pl.col("margin").alias("Margin"),
                pl.col("spread_for_team").alias("Line"),
                pl.col("ats_result").alias("ATS"),
                pl.col("model_margin").round(1).alias("Model"),
                pl.col("model_error").round(1).alias("Model err"),
                pl.col("elo_pre").round(0).alias("Elo before"),
                pl.col("elo_change").round(1).alias("Elo move"),
                pl.col("rest").alias("Rest"),
            ).sort("Wk")
            st.dataframe(disp.to_pandas(), width="stretch", hide_index=True)
            st.caption(
                "Margin, line and model are all from this team's point of view, so positive is "
                "always good for them. \"Model err\" is the model's predicted margin minus what "
                "actually happened: positive means it expected more than it got. \"Elo move\" is "
                "what that result did to the rating. Rows with blanks have not been played."
            )

# --- Bet sizing ----------------------------------------------------------------

with tab_sizing:
    st.warning(
        "Every other tab uses flat 1-unit stakes -- fine for testing whether an edge exists, but not how anyone "
        "would actually bet. This tab sizes each bet as a fraction of a compounding bankroll using the Kelly "
        "criterion, which exposes risk flat staking hides completely. Kelly sizing amplifies whatever the model's "
        "probability estimate says, including when that estimate is wrong -- and the Diagnostics tab already "
        "shows ours isn't clearly better than the market's. Treat this as a risk illustration, not a plan."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        kelly_multiplier = st.slider(
            "Kelly multiplier", 0.05, 1.0, 0.25, 0.05,
            help="Fraction of full Kelly to actually stake. 1.0 = full Kelly (very aggressive, assumes the "
            "probability estimate is exactly right). 0.25 (quarter-Kelly) is a common practical scale-down.",
        )
    with col_b:
        kelly_cap = st.slider(
            "Max stake per bet (% of bankroll)", 0.01, 0.25, 0.05, 0.01, format="%.2f",
            help="Hard cap regardless of what Kelly says -- a safeguard against a single mis-estimated game.",
        )

    sim = bankroll_sim(model, market, threshold, kelly_multiplier, kelly_cap)
    if sim is None or sim.height == 0:
        st.info("No placed bets at this threshold to simulate.")
    else:
        spdf = sim.to_pandas().reset_index(drop=True)
        spdf["bet_number"] = range(1, len(spdf) + 1)
        starting_bankroll = 100.0
        spdf["flat_bankroll"] = starting_bankroll + spdf["profit_units"].cumsum()

        final_kelly = spdf["bankroll"].iloc[-1]
        final_flat = spdf["flat_bankroll"].iloc[-1]
        dd = backtesting.max_drawdown(spdf["bankroll"].to_numpy())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final bankroll (flat)", f"{final_flat:.1f}", delta=f"{final_flat - starting_bankroll:+.1f}")
        c2.metric(
            "Final bankroll (Kelly)", f"{final_kelly:.1f}", delta=f"{final_kelly - final_flat:+.1f} vs. flat",
            help="Delta here is Kelly minus flat, not minus the 100 start -- this is the direct answer to "
            "\"does Kelly sizing over- or underperform just staking flat.\"",
        )
        c3.metric("Max drawdown (Kelly)", f"{dd:.0%}", help="Largest peak-to-trough decline at any point in the sequence.")
        c4.metric("Bets placed", f"{len(spdf):,}")
        if final_kelly < final_flat:
            st.caption(
                f"For {MODEL_LABELS[model]} / {MARKET_LABELS[market]} at this threshold, Kelly sizing **underperformed** "
                f"flat staking on this realized path ({final_kelly:.1f} vs. {final_flat:.1f}). Compounding cuts both "
                "ways: it also amplifies bad stretches, and the bootstrap band below shows how much of any single "
                "path's result -- good or bad -- is just the order bets happened to come in."
            )
        else:
            st.caption(
                f"For {MODEL_LABELS[model]} / {MARKET_LABELS[market]} at this threshold, Kelly sizing **outperformed** "
                f"flat staking on this realized path ({final_kelly:.1f} vs. {final_flat:.1f}). See the bootstrap band "
                "below before reading too much into that -- one path can look better than typical purely from order."
            )

        n_boot = 500
        paths = bootstrap_paths(model, market, threshold, kelly_multiplier, kelly_cap, n_boot)

        fig = go.Figure()
        if paths is not None and paths.shape[1] > 0:
            bet_n = np.arange(1, paths.shape[1] + 1)
            p10, p50, p90 = np.percentile(paths, [10, 50, 90], axis=0)
            fig.add_scatter(x=bet_n, y=p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(
                x=bet_n, y=p10, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                name=f"Bootstrap 10th-90th pct ({n_boot} resamples, same Kelly rule)", hoverinfo="skip",
            )
            fig.add_scatter(
                x=bet_n, y=p50, mode="lines", line=dict(color=BLUE, width=1, dash="dot"),
                name="Bootstrap median", hoverinfo="skip",
            )
        fig.add_scatter(
            x=spdf["bet_number"], y=spdf["flat_bankroll"], mode="lines",
            line=dict(color=GRAY, width=2), name="Flat stake (1% of start each bet)",
        )
        fig.add_scatter(
            x=spdf["bet_number"], y=spdf["bankroll"], mode="lines",
            line=dict(color=BLUE, width=2.5), name=f"Kelly x{kelly_multiplier:.2f} (realized, cap {kelly_cap:.0%})",
        )
        fig.add_hline(y=starting_bankroll, line_width=1, line_color=GRID)
        fig.update_layout(
            title="Bankroll over time: flat stake vs. Kelly sizing (both start at 100)",
            xaxis_title="Bet number", yaxis_title="Bankroll",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            **CHART_LAYOUT,
        )
        fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
        fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "The flat-stake and \"Kelly (realized)\" lines are the one sequence of bets that actually happened, in "
            "chronological order -- exactly what the older version of this chart showed. But Kelly compounding is "
            "path-dependent: the same bets in a different order produce a different ending bankroll, so one "
            "realized path can look better or worse than \"typical\" purely by chance of ordering. The shaded band "
            "resamples the same bet pool (same win rate, same odds) 500 times with replacement, reruns the "
            "identical Kelly rule on each resample, and shows the 10th-90th percentile range of outcomes -- if the "
            "realized blue line sits well outside that band, the specific order it happened in mattered a lot."
        )

        st.subheader("Which seasons would this have worked in?")
        season_end = spdf.groupby("season")["bankroll"].last()
        season_start = season_end.shift(1)
        season_start.iloc[0] = starting_bankroll
        season_pnl_pct = (season_end / season_start) - 1.0
        season_n_bets = spdf.groupby("season").size()
        season_win_rate = spdf.groupby("season").apply(
            lambda g: (~g["is_push"] & g["bet_won"].fillna(False)).mean(), include_groups=False
        )
        season_df = pl.DataFrame({
            "season": season_pnl_pct.index.to_numpy(),
            "roi": season_pnl_pct.to_numpy(),
            "n_bets": season_n_bets.to_numpy(),
            "win_rate": season_win_rate.to_numpy(),
        })
        st.plotly_chart(
            diverging_bar(season_df, "season", "Kelly bankroll % change by season", "Season"),
            width="stretch",
        )
        st.caption(
            "Season-over-season % change in the compounding Kelly bankroll (entering-season value to "
            "leaving-season value) -- unlike the chart above, this is ONLY the one realized path, not bootstrapped. "
            "A single bad season can look much worse here than its flat-stake ROI alone would suggest, since it's "
            "compounding on whatever the bankroll happened to be after every prior season."
        )

        st.divider()
        st.subheader("What Kelly multiplier actually maximizes growth?")
        st.caption(
            "The multiplier slider above is manual -- this instead sweeps it automatically and asks what the data "
            "itself says. For each candidate multiplier, the SAME bet pool at the current threshold is "
            "bootstrap-resampled 300 times and the resulting mean log-growth (the actual quantity Kelly theory "
            "maximizes) and drawdown risk are measured. The dashed line marks your current slider value; the dotted "
            "line marks the growth-maximizing one."
        )
        sweep = kelly_sweep(model, market, threshold, kelly_cap)
        if sweep is None or sweep.empty:
            st.info("Not enough placed bets at this threshold to sweep.")
        else:
            best_row = sweep.loc[sweep["mean_log_growth"].idxmax()]
            best_mult = float(best_row["kelly_multiplier"])

            fig_growth = go.Figure()
            fig_growth.add_scatter(
                x=sweep["kelly_multiplier"], y=sweep["mean_log_growth"], mode="lines+markers",
                line=dict(color=BLUE, width=2), marker=dict(size=5), name="Mean log-growth",
            )
            fig_growth.add_vline(x=kelly_multiplier, line_width=1.5, line_dash="dash", line_color=INK_SECONDARY)
            fig_growth.add_vline(x=best_mult, line_width=1.5, line_dash="dot", line_color=ORANGE)
            fig_growth.update_layout(
                title="Mean log-growth by Kelly multiplier (higher = better long-run growth)",
                xaxis_title="Kelly multiplier", yaxis_title="Mean log(final / start)",
                showlegend=False, **CHART_LAYOUT,
            )
            fig_growth.update_xaxes(gridcolor=GRID, linecolor=GRID)
            fig_growth.update_yaxes(gridcolor=GRID, linecolor=GRID)
            st.plotly_chart(fig_growth, width="stretch")

            fig_risk = go.Figure()
            fig_risk.add_scatter(
                x=sweep["kelly_multiplier"], y=sweep["prob_drawdown_50"], mode="lines+markers",
                line=dict(color=RED, width=2), marker=dict(size=5), name="P(drawdown > 50%)",
            )
            fig_risk.add_vline(x=kelly_multiplier, line_width=1.5, line_dash="dash", line_color=INK_SECONDARY)
            fig_risk.add_vline(x=best_mult, line_width=1.5, line_dash="dot", line_color=ORANGE)
            fig_risk.update_layout(
                title="P(a resampled path ever draws down more than 50%), by Kelly multiplier",
                xaxis_title="Kelly multiplier", yaxis_title="Probability", yaxis_tickformat=".0%",
                showlegend=False, **CHART_LAYOUT,
            )
            fig_risk.update_xaxes(gridcolor=GRID, linecolor=GRID)
            fig_risk.update_yaxes(gridcolor=GRID, linecolor=GRID)
            st.plotly_chart(fig_risk, width="stretch")

            risk_at_best = float(sweep.loc[sweep["kelly_multiplier"] == best_mult, "prob_drawdown_50"].iloc[0])
            risk_at_current = float(sweep.loc[(sweep["kelly_multiplier"] - kelly_multiplier).abs().idxmin(), "prob_drawdown_50"]) if len(sweep) else float("nan")
            st.caption(
                f"For {MODEL_LABELS[model]} / {MARKET_LABELS[market]} at this threshold, growth is maximized around "
                f"**{best_mult:.2f}x** -- but that comes with a **{risk_at_best:.0%}** chance of a 50%+ drawdown at "
                f"some point, vs. **{risk_at_current:.0%}** at your current {kelly_multiplier:.2f}x. Maximizing growth "
                "and avoiding painful drawdowns are different goals; the dashed (current) and dotted (growth-optimal) "
                "lines rarely land on the same value on purpose -- pick based on which risk you can actually live "
                "with, not just which line is highest."
            )

# --- Methodology -------------------------------------------------------------

with tab_methodology:
    st.markdown(
        f"""
## Data

All data comes from **nflreadpy** (the nflverse project) -- `schedules` (results plus closing market lines --
spread, total, home/away moneyline, spread odds), `pbp` (play-by-play, for EPA/success rate), `player_stats`
(QB-level form), and `injuries` (weekly injury reports) -- plus one external reference file for two columns
nflverse doesn't have (see below). Scope is **{config.FIRST_SEASON}-{config.LAST_SEASON}**, regular season +
playoffs, no preseason -- 2010 is where moneyline/spread-odds coverage in this source becomes ~100%.

## Pipeline

1. **Cleaning** (`cleaning.py`): standardizes relocated franchises' abbreviations (OAK->LV, SD->LAC, STL->LA) so
   rolling stats and Elo track one franchise across a move.
2. **Features** (`feature_engineering.py`): an 8-game rolling window and an EWMA
   ({config.EWMA_HALFLIFE_GAMES}-game half-life) of
   EPA/success-rate form per team, **QB-level form** (the starting QB each game -- most pass attempts for that
   team -- tracked with their own rolling EPA/dropback across team changes, plus a starter-continuity flag), an
   **injury feature** (ridge only -- a severity-weighted count of each team's official pre-game injury-report
   players, Out=1.0/Doubtful=0.5/Questionable=0.25), weather (temperature, wind -- filled to fixed
   comfortable-conditions constants for domes), and neutral-site flags. Every rolling/EWMA feature is shifted
   one game before the window so a game's features only ever use games (or starts, or injury reports) strictly
   before it -- no leakage from the game being predicted or the future. (Ridge only reads the EWMA versions;
   the roll8 versions feed LightGBM only.)
3. **Power ratings** (`power_ratings.py`): margin-of-victory-weighted Elo (K={config.ELO_K}, home-field edge
   {config.ELO_HOME_FIELD_ADVANTAGE} pts, {config.SEASON_REGRESSION_TO_MEAN:.0%} regression to the mean between
   seasons), computed the same way -- pre-game rating recorded before that game's result updates it. See the
   **Power ratings** tab to explore it per team.
4. **Models** (`modeling.py`): three models, all predicting **home margin**, all validated **walk-forward by
   season** (to predict season S, train only on seasons < S -- never the future):
   - **Ridge baseline**: ~15 hand-picked features (Elo diff, a few EWMA form diffs, QB EWMA diff + continuity,
     rest, weather, neutral site).
   - **LightGBM**: the full engineered feature set.
   - **Ensemble**: a meta-regression on the base models' predicted margins, refit walk-forward each season so
     the blend can shift toward whichever model has actually been more accurate so far, rather than a fixed
     split. LightGBM is currently **excluded** from the blend (`config.ENSEMBLE_INCLUDE_LGBM = False`), so this
     is a one-feature regression on ridge's prediction -- still not degenerate: it fits a slope and intercept
     that shrink and de-bias ridge's raw output against realized margins. Excluding LightGBM was originally a
     parsimony call, but has since been **measured directly** in an autoresearch sweep: re-including LightGBM
     lowered development-era ATS ROI by roughly a point, so the exclusion is now backed by data rather than
     judgment.
5. **Probabilities** (`probabilities.py`): a normal distribution around the predicted margin gives both the ATS
   cover probability and the moneyline win probability. Its width is a **per-game predicted std** (a small Ridge
   on |residual| ~ elo mismatch / divisional / playoff / weather / rest mismatch, itself walk-forward fit), not
   one constant for every game. The resulting probability is then **walk-forward recalibrated** (isotonic
   regression, fit only on strictly earlier seasons' prediction/outcome pairs) before comparing it to the
   market -- kept for ridge and the ensemble's moneyline market, but reverted for LightGBM's (see below).
6. **Backtest** (`backtesting.py`): flat 1-unit stakes, bet placed only when |model edge| clears a threshold
   (pipeline default: {config.ATS_EDGE_THRESHOLD_POINTS} pts for ATS, {config.ML_EDGE_THRESHOLD_PROB:.0%} for
   ML), graded against real market odds. **Adjustable live** via the sidebar slider on every other tab.
7. **Similarity** (`similarity.py`): a "games like this one" nearest-neighbor lookup in the same feature space
   the models use -- deliberately an interpretability tool, not a predictor. LightGBM (more flexible than k-NN
   for this kind of data) already had its shot at finding an edge and still lost to the market, so k-NN wasn't
   expected to do better; it also degrades quickly with ~15 features and a few thousand rows. Restricted to
   games strictly before the query game, same as everything else, even though it isn't part of the backtest.
8. **Bet sizing** (`backtesting.simulate_bankroll`): a compounding, Kelly-criterion bankroll simulation --
   separate from the flat 1-unit backtest everywhere else -- meant to expose risk flat staking can't show at
   all. See the **Bet sizing** tab, and the full walkthrough of the math below.
9. **Weekly automation** (`scripts/weekly_refresh.ps1`, registered as a Windows Scheduled Task by
   `scripts/setup_weekly_refresh.ps1`): runs `main.py --refresh` every Tuesday at 8am, so new results/lines/injury
   reports keep flowing in without manual intervention. Each refresh also appends a timestamped snapshot of that
   week's Upcoming picks to a permanent log (`live_picks.append_to_picks_log`) -- unlike `upcoming_picks.parquet`,
   which is overwritten every run, this is never deduplicated, so it becomes the actual, un-replayable record of
   what the model said before kickoff each week. See the **Track record** tab, which grades that log against
   real results as they come in.

## The math, model by model

**Elo power ratings** (`power_ratings.compute_elo_ratings`): every team starts at {config.ELO_INITIAL:.0f}.
Before each game, the home team's rating plus a fixed home-field edge ({config.ELO_HOME_FIELD_ADVANTAGE:.0f}
points) minus the away team's rating gives `elo_diff`, which converts to an expected win probability with the
standard Elo logistic curve: `expected_home = 1 / (1 + 10^(-elo_diff/400))`. After the result, both ratings move
by `delta = K x margin_multiplier x (actual_home - expected_home)`, where `actual_home` is 1/0.5/0 for a
win/tie/loss and K = {config.ELO_K:.0f}. The margin multiplier (`log(|margin|+1) x 2.2 / ((elo_diff_winner x
0.001) + 2.2)`) makes a 35-point blowout move the rating more than a 3-point win, but damps that effect when the
favorite was already expected to win big -- a mismatch confirmed isn't much new information. Between seasons,
every rating is pulled {config.SEASON_REGRESSION_TO_MEAN:.0%} of the way back toward {config.ELO_INITIAL:.0f}, a
simple proxy for offseason roster turnover.

**Ridge regression** (`modeling.walk_forward_ridge`, alpha=5.0): predicts home margin as a plain weighted sum of
the ~15 hand-picked features in `config.MARGIN_MODEL_FEATURES` -- `predicted_margin = intercept +
sum(weight_i x feature_i)`. The weights are fit by minimizing squared error plus a penalty on the weights' own
size (`alpha x sum(weight_i^2)`) -- ridge regression's whole reason for existing here: several of these features
are correlated (an EPA diff and a margin diff both partly measure "how good is this team"), and without the
penalty, correlated features fight each other for credit and the fitted weights swing wildly between similar
training sets. The penalty shrinks them toward zero and stabilizes the fit. Because it's a plain weighted sum,
every prediction decomposes additively into each feature's exact contribution (weight x that game's value) --
that's the Game explorer tab's "why did the model say this" bar chart, and it's only possible for ridge.

**LightGBM** (`modeling.walk_forward_lgbm`): the full engineered feature set (`modeling.gbm_feature_columns`),
fit as {config.LGBM_PARAMS['n_estimators']} shallow decision trees built one at a time -- each new tree is fit
not to the home margin itself but to the CURRENT ensemble's remaining error, and its prediction is added in at a
small learning rate ({config.LGBM_PARAMS['learning_rate']}) so no single tree can swing the model much. Each
tree has at most {config.LGBM_PARAMS['num_leaves']} leaves (shallow -- it can only split on a handful of feature
thresholds), and further regularization (at least {config.LGBM_PARAMS['min_child_samples']} games per leaf,
L1/L2 penalties on leaf weights, and each tree seeing only a random {config.LGBM_PARAMS['subsample']:.0%} sample
of games / {config.LGBM_PARAMS['colsample_bytree']:.0%} sample of features) keeps it from just memorizing the
training seasons. In exchange for all that machinery it can express interactions ridge structurally can't (e.g.
"elo_diff matters more in divisional games") -- but there's no single set of coefficients to point to, so there's
no equivalent of ridge's per-game contribution chart.

**Ensemble** (`modeling.walk_forward_ensemble`): a second-stage regression, `predicted_margin = w_ridge x
pred_ridge + w_lgbm x pred_lgbm + intercept`, with `w_ridge`/`w_lgbm` themselves fit by regression against actual
home margin -- refit every season using only prior seasons' predictions, so the blend is walk-forward validated
exactly like the two models feeding it. The fitted weights consistently land around 0.83-0.87 on ridge and
0.00-0.09 on LightGBM (e.g. 2015: 0.85 / -0.00; 2020: 0.87 / 0.04; 2025: 0.86 / 0.07) -- read that as "ridge is
still doing most of the work, LightGBM adds a small and slowly growing correction on top," not as two equal
partners.

**Turning a margin into a probability** (`probabilities.py`): every model's predicted margin becomes a
probability via a normal distribution centered on that prediction: `P(home wins) = Φ(predicted_margin / std)`;
`P(home covers) = Φ((predicted_margin - spread_line) / std)`, where Φ is the standard normal CDF. `std` is not
one constant for every game -- a divisional rematch and a weather-affected mismatch plausibly carry different
uncertainty -- it's itself a small Ridge predicting each game's expected |residual| from context features (elo
mismatch, divisional, playoff, weather, rest mismatch), converted to a std via `std = E|residual| x sqrt(pi/2)`
(the fixed ratio, for a normal distribution, between mean absolute deviation and standard deviation), then
floored/capped at {config.MARGIN_STD_FLOOR:.0f}-{config.MARGIN_STD_CAP:.0f} points as a safeguard against a
badly-fit fold.

**Calibration** (`probabilities.walk_forward_calibrate`): the raw probability above can still be systematically
over- or under-confident. Isotonic regression fits the simplest possible correction -- a monotonic step function
mapping raw probability to empirical outcome rate -- using only strictly earlier seasons' (probability, outcome)
pairs (needs at least {config.MIN_CALIBRATION_GAMES} prior graded games, else it's skipped and the raw
probability passes through unchanged), then applies that mapping to the current season. Kept for ridge and the
ensemble's moneyline market; reverted for LightGBM's (it measurably manufactured spurious underdog edges -- see
below).

**De-vigging the market** (`utils.remove_vig_two_way`): a sportsbook's own two-sided prices always imply more
than 100% combined probability -- that gap is the vig. `american_to_prob` converts each side's American odds to
its raw implied probability; `remove_vig_two_way` rescales both sides so they sum to exactly 1, giving the
market's fair (no-vig) probability to compare the model's own estimate against. Comparing a no-vig model
probability to a vig-included market number would make the model look like it has an edge it doesn't.

## Bet sizing math, end to end

1. **Kelly fraction** (`probabilities.kelly_fraction`): for a bet at model probability `p` and American odds
   implying payout `b` (profit per $1 staked if it wins), full Kelly stakes `f* = p - (1-p)/b` of the current
   bankroll, clipped at 0 (no perceived edge -> no bet, never bet against yourself).
2. **Scaled down, hard-capped**: full Kelly assumes `p` is exactly right, which is a bad assumption here (the
   Diagnostics tab shows the model's probabilities aren't clearly better than the market's). The **Kelly
   multiplier** slider scales the stake down (0.25 = quarter-Kelly, staking a quarter of what the formula says);
   the **max stake** slider hard-caps it regardless, as a backstop against one badly-wrong probability estimate.
3. **Sequential compounding** (`backtesting.simulate_bankroll`): bets are walked through in chronological order,
   `stake = current_bankroll x kelly_fraction`, and the bankroll updates after every single result -- so a
   losing stretch shrinks the next stake in absolute terms, a winning one grows it. This is what makes it
   fundamentally different from the flat 1-unit staking used everywhere else in this dashboard.
4. **Does Kelly sizing beat just staking flat?** Measured directly, quarter-Kelly vs. flat (both rescaled to
   start at 100), at the pipeline's default thresholds:

   | Model | Market | Flat final | Kelly final | Kelly max drawdown |
   |---|---|---|---|---|
   | Ridge | ATS | 86.4 | 104.9 | -61.8% |
   | Ridge | ML | 11.0 | 2.8 | -98.2% |
   | LightGBM | ATS | 50.9 | 47.3 | -69.6% |
   | LightGBM | ML | 26.3 | 1.0 | -99.3% |
   | Ensemble | ATS | 108.2 | 251.8 | -51.8% |
   | Ensemble | ML | 106.5 | 75.0 | -88.6% |

   **Kelly underperforms flat staking in 4 of 6 cases, sometimes catastrophically** (both ML markets across all
   three models lose 25-97% *more* than flat staking would have). It clearly outperforms in exactly 2 cases, both
   ATS. The pattern lines up with what you'd expect: ML odds swing much harder between favorites and heavy
   underdogs than ATS's near-uniform ~-110 pricing, so Kelly's stake sizing swings much harder too on the ML
   side -- and when the probability estimate behind that sizing is even slightly off, compounding punishes it far
   more severely than flat staking ever would. Bigger, more variable stakes aren't automatically better just
   because the formula that produced them is famous.
5. **One realized path isn't the whole story.** Kelly compounding is path-dependent -- the exact same bets in a
   different order produce a different ending bankroll, so any single historical path (including every number in
   the table above) can look better or worse than "typical" purely from the luck of ordering.
   `backtesting.bootstrap_bankroll_paths` resamples the same placed-bet pool with replacement 500 times (same
   bootstrap idea used for the ROI confidence intervals elsewhere in this dashboard), reruns the identical Kelly
   rule on each resample, and the **Bet sizing** tab's chart shows the resulting 10th-90th percentile band
   alongside the one path that actually happened. Where the realized line sits well outside that band, order
   mattered a lot.
6. **Season-by-season breakdown.** The bankroll chart's second panel shows the Kelly bankroll's percentage
   change within each individual season, which surfaces something the aggregate final-bankroll number hides:
   whether a good overall result is broad-based or came from one or two standout seasons carrying a lot of
   mediocre or losing ones.
7. **What multiplier actually maximizes growth?** (`backtesting.sweep_kelly_multiplier`) Rather than assume
   0.25 (quarter-Kelly) is the right scale-down, this sweeps the multiplier from 0.05 to 1.5 and measures mean
   log-growth (the literal quantity Kelly theory optimizes) across 300 bootstrap resamples at each value, plus
   the probability a resampled path ever draws down more than 50%. For ensemble/ATS: growth is maximized around
   **1.0x** (mean log-growth 1.24, vs. 1.00 at 1.0x proper full Kelly and 0.996 at the 0.25 default) -- but at a
   **99.3%** chance of a 50%+ drawdown at some point, vs. 74.3% at the 0.25 default. The growth-maximizing point
   and a survivable point are not the same thing; this sweep exists to make that trade-off visible and quantified
   instead of assumed.
8. **What if the threshold were chosen automatically each season?**
   (`backtesting.optimal_threshold_walk_forward`) For each test season, grid-searches the bet-selection threshold
   using only strictly earlier seasons -- selecting whichever threshold maximized the bootstrap 95% CI lower
   bound on ROI, not raw mean ROI -- then applies that season's choice out of sample. For ensemble/ATS across
   2017-2025: adaptive ROI (-0.33%, 813 bets) actually beats the fixed 1.5pt default (-0.67%, 1459 bets) on this
   metric, but **the chosen threshold itself swings from 0.0 to 4.25 points season to season (std dev 1.46)** --
   more diagnostic than either ROI number. That much season-to-season instability is itself evidence this
   optimizer is mostly chasing noise in a small per-season sample, not locking onto one real, stable threshold.
   For ensemble/ML, fixed and adaptive come out close (-3.2% vs. -3.1%) with a much tighter chosen-threshold std
   dev (0.02) -- more stable, but the conclusion (both are still losing) doesn't change.

## What a deep-dive into the errors turned up

- **QB-level form was the one change that clearly helped.** Before it, the gap to the market was worst in the
  playoffs (-0.62 pts MAE vs. market) and grew steadily from early season (-0.22) through late season (-0.39) --
  read together with games getting more variable late in the season (backups playing, seeding decided), that
  looked like an information gap (who's actually starting) more than a statistical one. Adding starter
  identification + per-QB rolling EPA + a continuity flag **shrank the playoffs gap from -0.62 to -0.30 and the
  divisional-game gap from -0.51 to -0.27** (vs. -0.22/-0.23 for non-division), and improved MAE/RMSE/Brier for
  *both* models. `qb_epa_per_att_ewm_diff` ranks 5th of ~25 features in LightGBM's importance, on par with
  `elo_diff`. The model still doesn't beat the market outright, but this closed a real chunk of the gap.
- **Two other ideas were tried for the team-level bias** (model overrates chronically weak franchises like NYJ,
  LV, WAS; underrates the era's best-run ones like NE, BAL, KC, SEA -- see the Power ratings tab) **and
  reverted**: a second, slower-decaying Elo track (91% correlated with the primary one -- didn't separate
  "organizational persistence" from "current form"), and opponent-adjusting each team's raw EPA by the specific
  opponent's incoming rolling quality (barely moved it, likely because `elo_diff` already opponent-adjusts
  through its own update rule). QB-level form did nudge this bias down slightly (0.836 -> 0.813 mean |bias|),
  but it's still a real, mostly-open problem.
- **LightGBM's feature importances had come back almost flat** across ~20 features, spending splits
  distinguishing near-duplicate rolling windows instead of finding clean signal -- this is why roll4/roll17 were
  pruned down to just roll8 + EWMA (previously 4/8/17-game windows all present at once).
- **Prime-time and playing-surface flags** (pulled from an external reference CSV, joined cleanly at a 99.1%
  match rate) were also tried and **reverted** -- motivated by the same "market is sharper on high-profile
  games" finding as the QB signal, but ridge came back flat and LightGBM got consistently slightly worse. Not
  every plausible feature pans out; this project tries to report the ones that don't as clearly as the ones
  that do.
- **A real injury-report feature, kept for ridge only.** A severity-weighted sum of each team's official
  pre-game injury designations (real report data, not a proxy). Ridge's aggregate MAE barely moved, but the
  divisional-game gap it was expected to help (rivals who scout each other closely, including injury news)
  closed further, from -0.27/-0.23 (div/non-division) to essentially zero: -0.244/-0.237. LightGBM got
  consistently slightly worse with it, so it's excluded there -- the two models don't have to share a feature
  list, and don't here.
- **Upgraded, kept: weighting each injured player by their own snap share, not just their position.**
  Each severity weight (Out=1.0/Doubtful=0.5/Questionable=0.25) is now scaled by that specific player's own
  prior rolling snap share (`feature_engineering.compute_player_snap_share`, from nflreadpy's `load_snap_counts`,
  joined to injuries through a gsis_id<->pfr_id crosswalk since the two datasets don't share an ID scheme) --
  a starter being out costs more than a rarely-used backup at the same report status. QBs are excluded from
  this feature entirely: `compute_qb_form`/`qb_continuity` already track the starter directly from real
  performance data, so including them here too would double-count the same signal. Measured directly: ridge's
  MAE/RMSE/Brier all improved (10.157->10.151 / 13.086->13.080 / 0.2172->0.2171) and ATS ROI went from -0.6%
  to +1.0% on essentially the same number of bets; ML ROI was flat. Ensemble/ML's ROI dropped (+0.3% -> -2.8%),
  but that sits well inside its own bootstrap 95% CI (-7.5% to +2.2%) -- not distinguishable from noise -- and
  ensemble/ATS improved slightly too. **Also found and fixed a real pre-existing bug along the way**: nflverse's
  injuries table uses pre-relocation team codes (OAK/SD/STL), which never matched `games`' standardized codes
  (LV/LAC/LA) -- the injury feature had silently been null for every Raiders/Chargers/Rams game before their
  respective moves. `data_ingestion.load_injuries` now standardizes team codes on load. One coverage caveat:
  snap-count data only goes back to 2012 (nflreadpy's own floor); earlier seasons, and any crosswalk miss, fall
  back to a fixed default importance (`config.DEFAULT_SNAP_SHARE`) rather than dropping the player.
- **Checked directly: does the model underperform early in the season** (the hypothesis being that rolling/EWMA
  form features are still mostly carrying over decayed values from LAST season, with no signal yet about
  offseason roster/coaching changes)? **No -- if anything, the opposite.** MAE-vs-market gap by time of season,
  all three models (Playoffs grouped by game_type, not week number, since playoff games continue the week count
  and the regular season itself went 17 -> 18 weeks in 2021):

  | | Weeks 1-3 | Weeks 4-8 | Weeks 9-13 | Weeks 14-18 | Playoffs |
  |---|---|---|---|---|---|
  | Ridge | 0.180 | 0.203 | 0.302 | 0.255 | 0.229 |
  | LightGBM | 0.610 | 0.741 | 0.778 | 0.579 | 0.600 |
  | Ensemble | 0.064 | 0.212 | 0.231 | 0.270 | 0.174 |

  (Gap = model MAE minus market MAE, in points -- lower is better, 0 would mean matching the market exactly.)
  Every model's gap is SMALLEST in Weeks 1-3 and PEAKS in Weeks 9-13, not at the start of the season. A plausible
  reason: Elo's between-season regression to the mean already gives the model a reasonable prior entering Week 1,
  and the market has just as little in-season signal that early too -- so neither side has much of an edge yet.
  The gap likely grows mid-season because that's exactly when injuries, trades, and bye-week mismatches create
  real complexity a backward-looking rolling-average model can't see coming, while the market (real-time injury
  reports, beat-writer info, line movement on late-breaking news) adjusts faster than a shifted rolling average
  can. See the **Backtest slices** tab's "ROI by time of season" chart for the threshold-dependent ROI version of
  this same cut.
- **Conditional per-game uncertainty, kept.** Replaced the single global residual std with a small Ridge
  predicting each game's own |residual| from context (elo mismatch, divisional, playoff, weather, rest
  mismatch). Essentially flat for ridge's Brier score (0.2171 -> 0.2172), a modest real cost for LightGBM's
  (0.2327 -> 0.2357, plausibly because its richer feature set already implicitly captures some of this). Kept
  anyway -- the structural idea is sound and the cost is small.
- **Walk-forward probability recalibration, mixed result.** Isotonic regression on raw probability -> outcome,
  fit only on strictly earlier seasons. A controlled ablation (identical bets, calibrated vs. raw only) found:
  ridge ML ROI -3.7% -> -3.4%, ensemble ML ROI -4.5% -> **+0.3%**, but LightGBM ML got clearly worse (-2.4% ->
  -4.8%) with its bet mix shifting from 56% favorites to 21% -- isotonic regression appears to overfit
  LightGBM's raw probability distribution into a curve that manufactures spurious underdog "edges." Kept for
  ridge and the ensemble's ML market, reverted for LightGBM's. ATS is unaffected either way (bet_side there
  comes from the points-based edge, not this probability) so it stays on there for all three models.
- **The ensemble is the best result of the three models.** Lowest MAE/RMSE of the three (10.13/13.08 vs. ridge
  10.16/13.09 and LightGBM 10.60/13.56), and the only one with positive flat-stake ROI in both markets at once
  (ATS +0.4%, ML +0.3%) -- though bootstrapped 95% CIs on both cross zero comfortably, so this is not yet
  statistically distinguishable from noise. The fitted blend leans heavily on ridge (weight ~0.83-0.87) with a
  small, growing LightGBM contribution in recent seasons -- read this as "ridge is still doing most of the
  work," not "the two models are equally useful."

## Known limitations

- **The model does not beat the market baseline on raw prediction accuracy**, in any slice checked so far.
  Check the Diagnostics tab: MAE, RMSE, and Brier score are all compared directly against just reading the
  market's own closing line. If the model doesn't win there, any positive backtest ROI elsewhere is not strong
  evidence of real skill.
- **Model-selection risk.** Two models x two markets = four backtests. Even with zero real edge anywhere,
  chance alone would produce some "profitable" results some of the time. A single green number isn't proof.
- **The normal-distribution approximation is a simplification.** Real NFL margins cluster at key numbers (3, 7,
  10) in a way a smooth normal CDF doesn't capture -- this matters most for ATS cover probabilities.
- **Bootstrap confidence intervals** shown on ROI charts account for sampling noise in the bet outcomes, but not
  for the fact that many bets in the same week share the same teams/games -- true uncertainty is likely somewhat
  wider than shown.
- **The pipeline's default thresholds** were chosen a priori, not tuned via nested cross-validation -- a
  deliberate choice to avoid overfitting the threshold to this backtest. Dragging the sidebar slider shows you
  the sensitivity directly, but it's on you to not just hunt for the threshold that looks best in hindsight.
- **Flat staking (used everywhere except the Bet sizing tab) hides risk.** A backtest ROI near zero sounds
  merely unexciting under flat 1-unit stakes; under Kelly-style compounding sizing, the same bet sequence can
  produce a peak-to-trough bankroll drawdown of 80-90%+ (see the Bet sizing tab) -- because Kelly sizing scales
  bet size with the model's estimated edge, and that estimate is exactly what the Diagnostics tab shows isn't
  clearly better than the market's. This is the honest answer to "is this tradeable": not as-is.

This is a research/backtesting tool, not betting advice.
"""
    )
