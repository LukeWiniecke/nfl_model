"""End-to-end pipeline: ingest -> clean -> features -> power ratings ->
walk-forward models -> probabilities -> backtest -> evaluation.

Run with: .venv\\Scripts\\python.exe main.py
"""

import polars as pl

from src import backtesting, config, data_ingestion, evaluation, feature_engineering, modeling, power_ratings
from src.cleaning import build_team_game_long, clean_schedules
from src.utils import get_logger

logger = get_logger("main")

MARKET_COLS = [
    "game_id", "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_spread_odds", "away_spread_odds",
]


def run_model(name: str, modeling_table: pl.DataFrame, walk_forward_fn):
    logger.info(f"=== {name} ===")
    result = walk_forward_fn(modeling_table)

    preds_with_market = result.predictions.join(
        modeling_table.select(MARKET_COLS), on="game_id", how="left"
    ).filter(pl.col("spread_line").is_not_null())

    ats_bets = evaluation.add_bet_context(backtesting.build_ats_bets(preds_with_market, result.margin_std))
    ml_bets = evaluation.add_bet_context(backtesting.build_ml_bets(preds_with_market, result.margin_std))

    ats_summary = backtesting.summarize(ats_bets)
    ml_summary = backtesting.summarize(ml_bets)
    logger.info(f"[{name}] ATS: {ats_summary}")
    logger.info(f"[{name}] ML:  {ml_summary}")

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
        ]:
            df = slicer(bets)
            df.write_csv(config.BACKTESTS_DIR / f"{name}_{label}_by_{slicer.__name__.replace('summarize_by_', '')}.csv")

    return {"ats": ats_summary, "ml": ml_summary}


def main():
    logger.info("Loading raw data (nflreadpy)...")
    raw = data_ingestion.load_all()

    logger.info("Cleaning schedules...")
    games = clean_schedules(raw["schedules"])
    games.write_parquet(config.PROCESSED_DIR / "games.parquet")

    team_long = build_team_game_long(games)

    logger.info("Computing per-game EPA/success-rate splits from pbp...")
    epa_splits = feature_engineering.compute_game_epa_splits(raw["pbp"])

    logger.info("Building rolling team-form features...")
    team_long_form = feature_engineering.add_rolling_form(team_long, epa_splits)

    logger.info("Computing Elo power ratings...")
    ratings = power_ratings.compute_elo_ratings(games)

    logger.info("Assembling game-level modeling table...")
    modeling_table = feature_engineering.build_modeling_table(games, team_long_form, ratings)
    modeling_table.write_parquet(config.PROCESSED_DIR / "modeling_table.parquet")
    logger.info(f"Modeling table: {modeling_table.shape}")

    results = {}
    results["ridge_baseline"] = run_model("ridge_baseline", modeling_table, modeling.walk_forward_ridge)
    results["lgbm"] = run_model("lgbm", modeling_table, modeling.walk_forward_lgbm)

    logger.info("=== SUMMARY ===")
    for name, r in results.items():
        logger.info(f"{name} | ATS n={r['ats']['n_bets']} roi={r['ats']['roi']:.3f} win_rate={r['ats']['win_rate']:.3f}")
        logger.info(f"{name} | ML  n={r['ml']['n_bets']} roi={r['ml']['roi']:.3f} win_rate={r['ml']['win_rate']:.3f}")


if __name__ == "__main__":
    main()
