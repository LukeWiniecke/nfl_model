"""Two staged margin-prediction models, both evaluated with season-level
walk-forward validation: to predict season S, train ONLY on seasons < S, so
no game ever influences the prediction of a game that happened before it or
in a season that hadn't finished yet.

  1. Baseline: Ridge regression over a small, hand-picked feature set
     (power rating diff + a few recency-weighted EPA/success-rate diffs).
  2. Gradient boosting (LightGBM) over the full engineered feature set, as a
     richer point of comparison against the baseline.
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge

from . import config
from .utils import get_logger

logger = get_logger(__name__)

FIRST_PREDICTABLE_SEASON = config.FIRST_SEASON + config.MIN_TRAIN_SEASONS


def gbm_feature_columns(df: pl.DataFrame) -> list[str]:
    suffixes = ("_roll4_diff", "_roll8_diff", "_roll17_diff", "_ewm_diff")
    cols = [c for c in df.columns if c.endswith(suffixes)]
    cols += ["elo_diff", "rest_diff", "div_game"]
    return [c for c in cols if c in df.columns]


@dataclass
class WalkForwardResult:
    predictions: pl.DataFrame
    margin_std: float
    feature_cols: list[str]
    train_medians: dict = field(default_factory=dict)


def _prep_xy(df: pl.DataFrame, feature_cols: list[str]):
    pdf = df.select(feature_cols + ["home_margin"]).to_pandas()
    X = pdf[feature_cols]
    y = pdf["home_margin"]
    return X, y


def _impute(X, medians: dict):
    return X.fillna(medians)


def walk_forward_ridge(df: pl.DataFrame, feature_cols: list[str] | None = None) -> WalkForwardResult:
    feature_cols = feature_cols or config.MARGIN_MODEL_FEATURES
    return _walk_forward(df, feature_cols, lambda: Ridge(alpha=5.0), "ridge_baseline")


def walk_forward_lgbm(df: pl.DataFrame, feature_cols: list[str] | None = None) -> WalkForwardResult:
    feature_cols = feature_cols or gbm_feature_columns(df)
    return _walk_forward(df, feature_cols, lambda: lgb.LGBMRegressor(**config.LGBM_PARAMS), "lgbm")


def _walk_forward(df: pl.DataFrame, feature_cols: list[str], model_factory, model_name: str) -> WalkForwardResult:
    df = df.filter(pl.col("home_margin").is_not_null())
    seasons = sorted(df["season"].unique().to_list())
    test_seasons = [s for s in seasons if s >= FIRST_PREDICTABLE_SEASON]

    all_preds = []
    resid_train_pool = []
    last_medians = {}

    for test_season in test_seasons:
        train_df = df.filter(pl.col("season") < test_season)
        test_df = df.filter(pl.col("season") == test_season)
        if train_df.height < 50 or test_df.height == 0:
            continue

        X_train, y_train = _prep_xy(train_df, feature_cols)
        medians = X_train.median().to_dict()
        last_medians = medians
        X_train = _impute(X_train, medians)

        model = model_factory()
        model.fit(X_train, y_train)

        X_test, y_test = _prep_xy(test_df, feature_cols)
        X_test = _impute(X_test, medians)
        pred = model.predict(X_test)

        train_pred = model.predict(X_train)
        resid_train_pool.append(y_train.to_numpy() - train_pred)

        out = test_df.select(["game_id", "season", "week", "home_team", "away_team", "home_margin"]).to_pandas()
        out["predicted_margin"] = pred
        out["model"] = model_name
        all_preds.append(out)

    predictions = pl.from_pandas(pd.concat(all_preds, ignore_index=True))
    margin_std = float(np.std(np.concatenate(resid_train_pool), ddof=1))

    logger.info(f"[{model_name}] walk-forward predictions: {predictions.height} games, resid std={margin_std:.2f}")
    return WalkForwardResult(
        predictions=predictions,
        margin_std=margin_std,
        feature_cols=feature_cols,
        train_medians=last_medians,
    )
