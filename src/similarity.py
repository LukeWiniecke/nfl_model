"""Find historically similar games by feature-space distance.

This is an interpretability tool ("games like this one"), not a predictive
model -- it isn't used by the backtest and doesn't feed a bet decision.
Comparison games are restricted to strictly before the query game (same
season+earlier week, or an earlier season), for consistency with the
walk-forward discipline used everywhere else in this project, even though
nothing here is part of the graded backtest.
"""

import numpy as np
import pandas as pd
import polars as pl

from . import config


def find_similar_games(
    modeling_table: pl.DataFrame,
    query_game_id: str,
    feature_cols: list[str] | None = None,
    k: int = 10,
) -> pl.DataFrame:
    feature_cols = feature_cols or config.MARGIN_MODEL_FEATURES
    df = modeling_table.filter(pl.col("home_margin").is_not_null())

    query_rows = df.filter(pl.col("game_id") == query_game_id)
    if query_rows.height == 0:
        return pl.DataFrame()
    query = query_rows.row(0, named=True)

    pool = df.filter(
        (pl.col("season") < query["season"])
        | ((pl.col("season") == query["season"]) & (pl.col("week") < query["week"]))
    )
    if pool.height == 0:
        return pl.DataFrame()

    X = pool.select(feature_cols).to_pandas()
    q = pd.Series({c: query[c] for c in feature_cols}, dtype="float64")

    means = X.mean()
    stds = X.std().replace(0, 1).fillna(1)
    Xz = (X.fillna(means) - means) / stds
    qz = (q.fillna(means) - means) / stds

    dist = np.sqrt(((Xz - qz) ** 2).sum(axis=1))

    result = pool.select(
        "game_id", "season", "week", "home_team", "away_team",
        "home_margin", "spread_line", "home_moneyline", "away_moneyline",
    ).to_pandas()
    result["distance"] = dist.to_numpy()
    result = result.sort_values("distance").head(k).reset_index(drop=True)
    result["home_covered"] = result["home_margin"] > result["spread_line"]
    result["home_won"] = result["home_margin"] > 0
    return pl.from_pandas(result)
