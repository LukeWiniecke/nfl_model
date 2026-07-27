"""Slice backtest results by season, team, edge size, home/away,
favorite/underdog, and market-spread bucket."""

import numpy as np
import polars as pl

from . import config


def add_bet_context(bets: pl.DataFrame) -> pl.DataFrame:
    df = bets.to_pandas().copy()

    df["bet_team"] = np.select(
        [df["bet_side"] == "home", df["bet_side"] == "away"],
        [df["home_team"], df["away_team"]],
        default=None,
    )

    favorite_side = np.where(df["spread_line"] > 0, "home", "away")
    df["is_favorite_bet"] = df["bet_side"] == favorite_side
    df["is_home_bet"] = df["bet_side"] == "home"

    edge_col = "ats_edge" if "ats_edge" in df.columns else "ml_edge"
    edge_units = df[edge_col].abs() if edge_col == "ats_edge" else df[edge_col].abs() * 100
    df["edge_abs"] = edge_units
    df["edge_bucket"] = (
        pl.Series(edge_units.fillna(0).to_numpy()).cut(breaks=config.EDGE_BUCKETS[1:-1]).cast(pl.Utf8).to_pandas().values
    )

    df["spread_bucket"] = (
        pl.Series(df["spread_line"].abs().fillna(0).to_numpy())
        .cut(breaks=config.SPREAD_BUCKETS[1:-1])
        .cast(pl.Utf8)
        .to_pandas()
        .values
    )

    return pl.from_pandas(df)


def summarize_by(bets: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    placed = bets.filter(pl.col("bet_side") != "none").with_columns(
        (~pl.col("is_push") & pl.col("bet_won").fill_null(False)).alias("_is_win"),
        (~pl.col("is_push") & ~pl.col("bet_won").fill_null(True)).alias("_is_loss"),
    )

    out = placed.group_by(group_cols).agg(
        pl.len().alias("n_bets"),
        pl.col("_is_win").sum().alias("n_wins"),
        pl.col("_is_loss").sum().alias("n_losses"),
        pl.col("staked").sum().alias("total_staked"),
        pl.col("profit_units").sum().alias("total_profit_units"),
    )

    out = out.with_columns(
        (pl.col("n_wins") / (pl.col("n_wins") + pl.col("n_losses"))).alias("win_rate"),
        (pl.col("total_profit_units") / pl.col("total_staked")).alias("roi"),
    )
    return out.sort(group_cols)


def summarize_by_season(bets: pl.DataFrame) -> pl.DataFrame:
    return summarize_by(bets, ["season"])


def summarize_by_team(bets: pl.DataFrame) -> pl.DataFrame:
    return summarize_by(bets, ["bet_team"])


def summarize_by_home_away(bets: pl.DataFrame) -> pl.DataFrame:
    return summarize_by(bets, ["is_home_bet"])


def summarize_by_favorite_underdog(bets: pl.DataFrame) -> pl.DataFrame:
    return summarize_by(bets, ["is_favorite_bet"])


def summarize_by_edge_bucket(bets: pl.DataFrame) -> pl.DataFrame:
    return summarize_by(bets, ["edge_bucket"])


def summarize_by_spread_bucket(bets: pl.DataFrame) -> pl.DataFrame:
    return summarize_by(bets, ["spread_bucket"])
