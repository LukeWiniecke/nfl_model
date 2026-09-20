"""End-to-end pipeline: ingest -> clean -> features -> power ratings ->
walk-forward models -> probabilities -> backtest -> evaluation.

Run with: .venv\\Scripts\\python.exe main.py
During the season, run with `--refresh` to pull fresh data instead of the
local cache -- new market lines, newly-final scores, and updated injury
reports don't show up otherwise, since data_ingestion caches to parquet and
reuses it indefinitely by default:
    .venv\\Scripts\\python.exe main.py --refresh
"""

import sys

import polars as pl

from src import backtesting, config, data_ingestion, evaluation, feature_engineering, live_picks, modeling, power_ratings
from src.cleaning import build_team_game_long, clean_schedules
from src.utils import get_logger

logger = get_logger("main")

MARKET_COLS = [
    "game_id", "game_type", "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
]


def run_model(name: str, modeling_table: pl.DataFrame, result: modeling.WalkForwardResult):
    logger.info(f"=== {name} ===")

    preds_with_market = result.predictions.join(
        modeling_table.select(MARKET_COLS), on="game_id", how="left"
    ).filter(pl.col("spread_line").is_not_null())

    skill = evaluation.skill_vs_market(preds_with_market, result.margin_std)
    skill.write_csv(config.MODEL_RESULTS_DIR / f"{name}_skill.csv")
    logger.info(f"[{name}] skill vs market:\n{skill}")

    if result.contributions is not None:
        result.contributions.write_parquet(config.MODEL_RESULTS_DIR / f"{name}_contributions.parquet")

    ats_bets = evaluation.add_bet_context(backtesting.build_ats_bets(preds_with_market, result.margin_std))
    # ML probability recalibration (backtesting.walk_forward_calibrate) is kept
    # per-model, not universal -- measured directly (see README/Methodology):
    # it helps ridge (ROI -0.037 -> -0.034) and the ensemble (-0.045 -> +0.003),
    # but for LightGBM specifically it roughly doubles the loss (-0.024 ->
    # -0.048) and skews selection from 56% favorites to 21% -- isotonic
    # regression appears to overfit LightGBM's raw probability distribution in
    # a way that manufactures spurious underdog "edges." ATS is unaffected by
    # this either way (bet_side there is decided by the points-based edge, not
    # this probability), so calibration stays on for ATS across all models.
    #
    # NOTE: that per-model split was diagnosed BEFORE the conditional-std bug
    # was found (modeling._fit_predict_conditional_std). LightGBM's raw
    # probabilities were overconfident because its predicted_std was ~50% too
    # small, not because isotonic is inherently wrong for it. Now that the std
    # is fit out-of-fold, this exception is a candidate for removal -- but it
    # needs re-measuring on the DEVELOPMENT era only before being changed.
    ml_bets = evaluation.add_bet_context(
        backtesting.build_ml_bets(preds_with_market, result.margin_std, calibrate=(name != "lgbm"))
    )

    ats_summary = backtesting.summarize(ats_bets)
    ml_summary = backtesting.summarize(ml_bets)
    logger.info(f"[{name}] ATS: {ats_summary}")
    logger.info(f"[{name}] ML:  {ml_summary}")

    for label, bets in [("ATS", ats_bets), ("ML", ml_bets)]:
        era = evaluation.summarize_by_era(bets)
        for row in era.iter_rows(named=True):
            logger.info(
                f"[{name}] {label} {row['era']:<11}: n={row['n_bets']:4d} roi={row['roi']:+.4f} "
                f"95% CI=[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]"
            )

    result.predictions.write_parquet(config.MODEL_RESULTS_DIR / f"{name}_predictions.parquet")
    ats_bets.write_parquet(config.BACKTESTS_DIR / f"{name}_ats_bets.parquet")
    ml_bets.write_parquet(config.BACKTESTS_DIR / f"{name}_ml_bets.parquet")

    for label, bets in [("ats", ats_bets), ("ml", ml_bets)]:
        for slicer in [
            evaluation.summarize_by_season,
            evaluation.summarize_by_team,
            evaluation.summarize_by_home_away,
            evaluation.summarize_by_favorite_underdog,
            evaluation.summarize_by_edge_bucket,
            evaluation.summarize_by_spread_bucket,
            evaluation.summarize_by_week_bucket,
            evaluation.summarize_by_era,
        ]:
            df = slicer(bets)
            out_path = config.BACKTESTS_DIR / f"{name}_{label}_by_{slicer.__name__.replace('summarize_by_', '')}.csv"
            try:
                df.write_csv(out_path)
            except OSError as e:
                logger.warning(f"Could not write {out_path} (likely open in another program), skipping: {e}")

    return {"ats": ats_summary, "ml": ml_summary}


def run_totals_model(name: str, modeling_table: pl.DataFrame, result: modeling.WalkForwardResult):
    """Totals (over/under) analogue of run_model -- phase 1: model + backtest
    + skill-vs-market only, not yet wired into Upcoming picks or the
    dashboard (see README)."""
    logger.info(f"=== {name} ===")

    preds_with_market = result.predictions.join(
        modeling_table.select(MARKET_COLS), on="game_id", how="left"
    ).filter(pl.col("total_line").is_not_null())

    skill = evaluation.skill_vs_market_totals(preds_with_market, result.margin_std)
    skill.write_csv(config.MODEL_RESULTS_DIR / f"{name}_skill.csv")
    logger.info(f"[{name}] skill vs market:\n{skill}")

    totals_bets = evaluation.add_bet_context(backtesting.build_totals_bets(preds_with_market, result.margin_std))
    totals_summary = backtesting.summarize(totals_bets)
    logger.info(f"[{name}] TOTAL: {totals_summary}")

    result.predictions.write_parquet(config.MODEL_RESULTS_DIR / f"{name}_predictions.parquet")
    totals_bets.write_parquet(config.BACKTESTS_DIR / f"{name}_totals_bets.parquet")

    for slicer in [evaluation.summarize_by_season, evaluation.summarize_by_edge_bucket, evaluation.summarize_by_week_bucket]:
        df = slicer(totals_bets)
        out_path = config.BACKTESTS_DIR / f"{name}_totals_by_{slicer.__name__.replace('summarize_by_', '')}.csv"
        try:
            df.write_csv(out_path)
        except OSError as e:
            logger.warning(f"Could not write {out_path} (likely open in another program), skipping: {e}")

    return {"total": totals_summary}


def main():
    force_refresh = "--refresh" in sys.argv
    logger.info(f"Loading raw data (nflreadpy){' -- forcing a fresh pull' if force_refresh else ' (cached)'}...")
    raw = data_ingestion.load_all(force_refresh=force_refresh)

    logger.info("Cleaning schedules...")
    games = clean_schedules(raw["schedules"])
    games.write_parquet(config.PROCESSED_DIR / "games.parquet")

    team_long = build_team_game_long(games)

    logger.info("Computing per-game EPA/success-rate splits from pbp...")
    epa_splits = feature_engineering.compute_game_epa_splits(raw["pbp"])
    # Penalties COMMITTED per team-game, attributed via penalty_team (posteam
    # is the offense, and ~41% of penalties are on the defense). Creates
    # pen_count/pen_yards/pen_presnap rolling-form columns; no model uses them
    # unless they are named in a feature list, and LightGBM's suffix
    # auto-detection explicitly excludes them (modeling.gbm_feature_columns).
    penalty_splits = feature_engineering.compute_game_penalty_splits(raw["pbp"])

    logger.info("Building rolling team-form features...")
    team_long_form = feature_engineering.add_rolling_form(team_long, epa_splits, penalty_splits)

    logger.info("Computing QB-level form (starter identification, rolling EPA, continuity)...")
    # "margin" is included so compute_qb_form can tell a genuinely UNPLAYED
    # game (margin null) apart from a PLAYED game that simply has no QB
    # pass-attempt row in player_stats (a real, if rare, data gap -- e.g.
    # old pre-relocation Raiders games). Both fall into its carry-forward
    # branch (removing that would also remove a real, if accidental,
    # benefit the naive carry-forward already provided for those gap games
    # -- confirmed directly: MAE got WORSE, not better, when gap games were
    # forced to null/median-imputed instead), but the depth-chart override
    # specifically must only ever fire for the genuinely-unplayed ones --
    # a 2026 depth chart has no business overriding a 2012 game's data gap.
    qb_schedule = team_long_form.select(["game_id", "team", "season", "week", "margin"])
    # depth_chart_starters computed but NOT applied (passed as None below):
    # the override mechanism itself is verified correct (byte-identical
    # backtest across every model once the data-gap edge case above was
    # fixed), but nflverse's LIVE depth-chart feed has a real, confirmed
    # team-to-player misalignment bug as of this data pull -- e.g. Tua
    # Tagovailoa (actually Miami's QB) listed under Atlanta, Kyler Murray
    # (Arizona's) listed under Minnesota, Miami's own entry showing Malik
    # Willis instead, Arizona's showing Jacoby Brissett instead. Not a join
    # bug on this end (confirmed directly against nflreadpy's raw output) --
    # an upstream data quality problem in nflverse's source for the current
    # season specifically. See config.py's note for the full writeup.
    depth_chart_starters = data_ingestion.load_current_qb_depth_chart()
    qb_form = feature_engineering.compute_qb_form(raw["player_stats"], qb_schedule, None)
    team_long_form = team_long_form.join(qb_form, on=["game_id", "team"], how="left")

    # QB Next Gen Stats form (CPOE) -- tried and reverted, see config.py's
    # note near MARGIN_MODEL_FEATURES. Left DISABLED here rather than just
    # dropped from the ridge/LightGBM feature lists (the pattern used for
    # ref_total_resid_ewm/prime_time_approx): qb_cpoe_ewm ends in "_ewm", so
    # build_modeling_table's generic form_cols detection would turn it into
    # qb_cpoe_ewm_diff/_sum automatically, and modeling.gbm_feature_columns/
    # gbm_total_feature_columns auto-include ANY column matching that
    # suffix -- leaving the join active would silently feed it back into
    # LightGBM regardless of what the explicit ridge lists say.
    # qb_ngs_form = feature_engineering.compute_qb_ngs_form(raw["player_stats"], raw["nextgen_passing"], qb_schedule)
    # team_long_form = team_long_form.join(qb_ngs_form, on=["game_id", "team"], how="left")

    logger.info("Computing Elo power ratings...")
    ratings = power_ratings.compute_elo_ratings(games)
    ratings.write_parquet(config.PROCESSED_DIR / "elo_ratings.parquet")

    spreadspoke_ctx = data_ingestion.load_spreadspoke_context()
    logger.info("Computing player snap shares (injury feature importance weighting)...")
    player_snap_share = feature_engineering.compute_player_snap_share(raw["snap_counts"], raw["player_crosswalk"])
    injury_features = feature_engineering.compute_injury_features(raw["injuries"], player_snap_share)
    qb_injury_features = feature_engineering.compute_qb_injury_features(raw["injuries"], player_snap_share)
    referee_features = feature_engineering.compute_referee_features(raw["officials"], games)
    # Per-unit decomposition of the same injury score (inj_ol/dl/lb/db/rec/rb).
    # Creates *_diff columns; no model uses them unless named in a feature
    # list, and they carry no _ewm_diff/_roll8_diff suffix so LightGBM's
    # auto-detection cannot pick them up either. Taking these AND injury_diff
    # would double-count -- they are a decomposition of it, not new data.
    unit_injury_features = feature_engineering.compute_unit_injury_features(raw["injuries"], player_snap_share)

    logger.info("Assembling game-level modeling table...")
    modeling_table = feature_engineering.build_modeling_table(
        games, team_long_form, ratings, spreadspoke_ctx=spreadspoke_ctx,
        injury_features=injury_features, qb_injury_features=qb_injury_features,
        referee_features=referee_features, unit_injury_features=unit_injury_features,
    )
    modeling_table.write_parquet(config.PROCESSED_DIR / "modeling_table.parquet")
    logger.info(f"Modeling table: {modeling_table.shape}")

    ridge_result = modeling.walk_forward_ridge(modeling_table)
    lgbm_result = modeling.walk_forward_lgbm(modeling_table)
    ensemble_result = modeling.walk_forward_ensemble(modeling_table, ridge_result, lgbm_result)

    results = {}
    results["ridge_baseline"] = run_model("ridge_baseline", modeling_table, ridge_result)
    results["lgbm"] = run_model("lgbm", modeling_table, lgbm_result)
    results["ensemble"] = run_model("ensemble", modeling_table, ensemble_result)

    # Totals (over/under) -- phase 1 only: model + backtest + skill-vs-market,
    # not yet wired into Upcoming picks or the dashboard.
    ridge_total_result = modeling.walk_forward_ridge_total(modeling_table)
    lgbm_total_result = modeling.walk_forward_lgbm_total(modeling_table)
    ensemble_total_result = modeling.walk_forward_ensemble_total(modeling_table, ridge_total_result, lgbm_total_result)

    totals_results = {}
    totals_results["ridge_total"] = run_totals_model("ridge_total", modeling_table, ridge_total_result)
    totals_results["lgbm_total"] = run_totals_model("lgbm_total", modeling_table, lgbm_total_result)
    totals_results["ensemble_total"] = run_totals_model("ensemble_total", modeling_table, ensemble_total_result)

    logger.info("=== SUMMARY ===")
    for name, r in results.items():
        logger.info(f"{name} | ATS n={r['ats']['n_bets']} roi={r['ats']['roi']:.3f} win_rate={r['ats']['win_rate']:.3f}")
        logger.info(f"{name} | ML  n={r['ml']['n_bets']} roi={r['ml']['roi']:.3f} win_rate={r['ml']['win_rate']:.3f}")
    for name, r in totals_results.items():
        logger.info(f"{name} | TOTAL n={r['total']['n_bets']} roi={r['total']['roi']:.3f} win_rate={r['total']['win_rate']:.3f}")

    logger.info("=== Upcoming picks (games with no result yet, but a posted market line) ===")
    # Each walk-forward result is passed through so the live conditional-std
    # model is fit on that model's OUT-OF-FOLD residuals. The live model
    # itself trains on all history with nothing held out, so its own residuals
    # are in-sample and would understate its true error -- the same bug fixed
    # inside _walk_forward. See modeling.oof_std_pool.
    live_by_model = {
        "ridge_baseline": modeling.predict_upcoming_ridge(modeling_table, ridge_result),
        "lgbm": modeling.predict_upcoming_lgbm(modeling_table, lgbm_result),
    }
    live_by_model["ensemble"] = modeling.predict_upcoming_ensemble(
        modeling_table, ridge_result, lgbm_result, live_by_model["ridge_baseline"], live_by_model["lgbm"]
    )

    # Totals live predictions -- display only (predicted_total, folded into a
    # derived predicted score below), not a betting market of its own yet.
    live_total_by_model = {
        "ridge_baseline": modeling.predict_upcoming_ridge_total(modeling_table, ridge_total_result),
        "lgbm": modeling.predict_upcoming_lgbm_total(modeling_table, lgbm_total_result),
    }
    live_total_by_model["ensemble"] = modeling.predict_upcoming_ensemble_total(
        modeling_table, ridge_total_result, lgbm_total_result,
        live_total_by_model["ridge_baseline"], live_total_by_model["lgbm"],
    )

    calibrate_ml_by_model = {"ridge_baseline": True, "lgbm": False, "ensemble": True}
    upcoming = live_picks.build_all_boards(live_by_model, calibrate_ml_by_model)
    if upcoming is None:
        logger.info("No upcoming games with a posted market line yet -- nothing to write.")
    else:
        upcoming = live_picks.merge_total_predictions(upcoming, live_total_by_model)
        pl.from_pandas(upcoming).write_parquet(config.MODEL_RESULTS_DIR / "upcoming_picks.parquet")
        n_games = upcoming["game_id"].nunique()
        logger.info(f"Upcoming picks: {n_games} games x {len(live_by_model)} models written to upcoming_picks.parquet")

        # Only log a snapshot on a --refresh run (new data actually arrived) --
        # not on every plain `python main.py`, which would otherwise re-log an
        # identical snapshot of the same unchanged picks.
        if force_refresh:
            n_logged = live_picks.append_to_picks_log(upcoming)
            logger.info(f"Picks log: appended {n_logged} rows to {config.PICKS_LOG_PATH}")


if __name__ == "__main__":
    main()
