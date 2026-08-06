"""Slice backtest results by season, team, edge size, home/away,
favorite/underdog, and market-spread bucket."""

import numpy as np
import pandas as pd
import polars as pl

from . import config
from .probabilities import fit_t_dof_walk_forward, home_win_probability_t, market_moneyline_no_vig, over_probability_t
from .utils import american_to_prob, remove_vig_two_way


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

    if "ats_edge" in df.columns:
        edge_col = "ats_edge"
    elif "ml_edge" in df.columns:
        edge_col = "ml_edge"
    else:
        edge_col = "total_edge"
    edge_units = df[edge_col].abs() * (100 if edge_col == "ml_edge" else 1)
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

    # Raw week number isn't directly comparable across eras on its own --
    # the regular season went 17 -> 18 weeks in 2021, and playoff games
    # continue the week count rather than resetting (e.g. week 18 is a
    # Super Bowl in a 17-week season but a normal Sunday in an 18-week one).
    # Playoffs are identified by game_type, not a week cutoff, so this is
    # stable across both eras; the 4 regular-season buckets only use weeks
    # 1-13, which are numbered identically in every season.
    df["week_bucket"] = np.select(
        [df["game_type"] != "REG", df["week"] <= 3, df["week"] <= 8, df["week"] <= 13],
        ["Playoffs", "Weeks 1-3", "Weeks 4-8", "Weeks 9-13"],
        default="Weeks 14-18",
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


def summarize_by_week_bucket(bets: pl.DataFrame) -> pl.DataFrame:
    return summarize_by(bets, ["week_bucket"])


def summarize_with_ci(bets: pl.DataFrame, group_cols: list[str], n_boot: int = 2000, seed: int = 0) -> pl.DataFrame:
    """Like summarize_by, plus a bootstrap 95% CI on ROI per group. Since every
    placed bet stakes exactly 1 unit, ROI == mean(profit_units), so resampling
    profit_units with replacement is a direct bootstrap of ROI itself.
    Groups with fewer than 2 bets get a degenerate (point) interval.
    """
    base = summarize_by(bets, group_cols)
    placed = bets.filter(pl.col("bet_side") != "none")
    pdf = placed.select(group_cols + ["profit_units"]).to_pandas()
    rng = np.random.default_rng(seed)

    key_cols = group_cols[0] if len(group_cols) == 1 else group_cols
    records = []
    for key, sub in pdf.groupby(key_cols):
        profit = sub["profit_units"].to_numpy()
        n = len(profit)
        if n < 2:
            lo = hi = float(profit.mean()) if n else float("nan")
        else:
            idx = rng.integers(0, n, size=(n_boot, n))
            boot_means = profit[idx].mean(axis=1)
            lo, hi = (float(v) for v in np.percentile(boot_means, [2.5, 97.5]))
        row = {"ci_lo": lo, "ci_hi": hi}
        keys = key if isinstance(key, tuple) else (key,)
        row.update(dict(zip(group_cols, keys)))
        records.append(row)

    ci_df = pl.DataFrame(records)
    return base.join(ci_df, on=group_cols, how="left")


def calibration_bins(bets: pl.DataFrame, prob_col: str, outcome_expr: pl.Expr, n_bins: int = 10) -> pl.DataFrame:
    """Bin games by predicted probability and compare the mean prediction in
    each bin to the realized outcome rate -- the standard reliability-curve
    check of whether a probability of 0.65 actually wins ~65% of the time."""
    df = bets.filter(pl.col(prob_col).is_not_null()).with_columns(outcome_expr.alias("_outcome"))
    df = df.filter(pl.col("_outcome").is_not_null())
    pdf = df.select([prob_col, "_outcome"]).to_pandas()
    try:
        pdf["_bin"] = pd.qcut(pdf[prob_col], n_bins, duplicates="drop")
    except ValueError:
        pdf["_bin"] = pd.cut(pdf[prob_col], n_bins)
    grouped = (
        pdf.groupby("_bin", observed=True)
        .agg(mean_pred=(prob_col, "mean"), actual_rate=("_outcome", "mean"), n=("_outcome", "size"))
        .reset_index(drop=True)
    )
    return pl.from_pandas(grouped)


def skill_vs_market(preds_with_market: pl.DataFrame, margin_std: float) -> pl.DataFrame:
    """Compare the model against the simplest possible baseline: the market's
    own closing line, treated as ITS margin prediction (`spread_line`) and win
    probability (de-vigged moneyline). This is the direct answer to "is the
    model actually better than just reading the market line", independent of
    any bet-selection threshold.
    """
    df = preds_with_market.filter(pl.col("home_margin").is_not_null() & pl.col("spread_line").is_not_null()).to_pandas()

    model_err = df["predicted_margin"] - df["home_margin"]
    market_err = df["spread_line"] - df["home_margin"]

    std = df["predicted_std"].to_numpy() if "predicted_std" in df.columns else margin_std
    if "predicted_std" in df.columns:
        dof = fit_t_dof_walk_forward(df, "home_margin", "predicted_margin", "predicted_std")
    else:
        dof = np.full(len(df), config.DEFAULT_T_DOF)
    model_prob = home_win_probability_t(df["predicted_margin"].to_numpy(), std, dof)
    market_prob, _ = market_moneyline_no_vig(df["home_moneyline"].to_numpy(), df["away_moneyline"].to_numpy())
    actual_home_win = (df["home_margin"] > 0).to_numpy().astype(float)
    has_ml = df["home_moneyline"].notna().to_numpy()

    def brier(p, y):
        return float(np.mean((p - y) ** 2))

    return pl.DataFrame(
        {
            "metric": ["mae", "rmse", "brier"],
            "model": [
                float(model_err.abs().mean()),
                float(np.sqrt((model_err**2).mean())),
                brier(model_prob[has_ml], actual_home_win[has_ml]),
            ],
            "market": [
                float(market_err.abs().mean()),
                float(np.sqrt((market_err**2).mean())),
                brier(market_prob[has_ml], actual_home_win[has_ml]),
            ],
            "n": [len(df), len(df), int(has_ml.sum())],
        }
    )


def skill_vs_market_totals(preds_with_market: pl.DataFrame, total_std: float) -> pl.DataFrame:
    """Totals analogue of skill_vs_market: compares the model's predicted
    total against the market's own total_line (as its prediction) and
    de-vigged over/under odds (as its probability the over hits)."""
    df = preds_with_market.filter(
        pl.col("total_points").is_not_null() & pl.col("total_line").is_not_null()
    ).to_pandas()

    model_err = df["predicted_total"] - df["total_points"]
    market_err = df["total_line"] - df["total_points"]

    std = df["predicted_std"].to_numpy() if "predicted_std" in df.columns else total_std
    if "predicted_std" in df.columns:
        dof = fit_t_dof_walk_forward(df, "total_points", "predicted_total", "predicted_std")
    else:
        dof = np.full(len(df), config.DEFAULT_T_DOF)
    model_prob = over_probability_t(df["predicted_total"].to_numpy(), df["total_line"].to_numpy(), std, dof)
    market_prob, _ = remove_vig_two_way(
        american_to_prob(df["over_odds"].to_numpy()), american_to_prob(df["under_odds"].to_numpy())
    )
    actual_over = (df["total_points"] > df["total_line"]).to_numpy().astype(float)
    has_total_odds = df["over_odds"].notna().to_numpy()

    def brier(p, y):
        return float(np.mean((p - y) ** 2))

    return pl.DataFrame(
        {
            "metric": ["mae", "rmse", "brier"],
            "model": [
                float(model_err.abs().mean()),
                float(np.sqrt((model_err**2).mean())),
                brier(model_prob[has_total_odds], actual_over[has_total_odds]),
            ],
            "market": [
                float(market_err.abs().mean()),
                float(np.sqrt((market_err**2).mean())),
                brier(market_prob[has_total_odds], actual_over[has_total_odds]),
            ],
            "n": [len(df), len(df), int(has_total_odds.sum())],
        }
    )
