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

# E|X| = std * sqrt(2/pi) for X ~ Normal(0, std) => std = E|X| * sqrt(pi/2).
# Lets a model of the residual's mean ABSOLUTE size (which is what a Ridge on
# non-negative targets estimates well) be converted into a std for norm.cdf.
MEAN_ABS_TO_STD = float(np.sqrt(np.pi / 2))

# Small, hand-picked context features for predicting each game's OWN residual
# spread, instead of assuming every game is equally uncertain (the previous
# single global margin_std). A blowout-prone mismatch, a divisional rematch,
# a playoff game, and a windy outdoor game plausibly have different variance
# around the point prediction, even if the point prediction itself is fine.
#
# Tried and reverted: qb_injury_diff_abs/qb_continuity_diff_abs, on the
# hypothesis that a real starter-out/backup situation should carry more
# predictive uncertainty than a normal game. Measured directly: Brier moved
# by <0.0002 for all three models (ridge 0.21690->0.21676, LightGBM
# 0.23586->0.23585, ensemble 0.21735->0.21732) -- noise, not signal. ATS ROI
# was byte-identical (bet-side/flat-stake ROI there comes from the points
# edge, untouched by std). ML ROI moved more (ridge -3.2%->-2.2%, LightGBM
# -3.4%->-3.0%, but ensemble -3.0%->-4.3%) -- larger swings than the near-zero
# Brier change justifies, i.e. threshold-crossing churn on an unchanged
# probability distribution, not a real effect. Likely reason: qb_injury_diff
# is already a MARGIN_MODEL_FEATURES input for ridge, so whatever signal it
# carries is already reflected in the point prediction (and thus partly in
# the residual this std model targets) before this ever saw it separately;
# qb_continuity_diff is also always 0 for a LIVE upcoming game specifically
# (feature_engineering.compute_qb_form's carried-forward branch always
# assumes continuity=1 for both teams), limiting it to a training-only
# signal even in principle.
STD_FEATURES = ["elo_diff_abs", "div_game", "is_playoff", "wind_filled", "is_dome", "rest_diff_abs"]


def _std_feature_frame(df: pl.DataFrame) -> pd.DataFrame:
    aug = df.with_columns(
        pl.col("elo_diff").abs().alias("elo_diff_abs"),
        (pl.col("game_type") != "REG").cast(pl.Int8).alias("is_playoff"),
        pl.col("rest_diff").abs().alias("rest_diff_abs"),
    )
    return aug.select(STD_FEATURES).to_pandas()


def _fit_predict_conditional_std(train_df: pl.DataFrame, test_df: pl.DataFrame, resid_train: np.ndarray) -> np.ndarray:
    """Fit a small Ridge on |residual| ~ STD_FEATURES using only the training
    pool, then predict a per-game std for the test fold. Floored/capped
    (config.MARGIN_STD_FLOOR/CAP) so one badly-fit fold can't collapse a
    game's probability to 0/1 or blow it up into meaninglessness."""
    std_X_train = _std_feature_frame(train_df)
    medians = std_X_train.median().to_dict()
    std_X_train = _impute(std_X_train, medians)

    std_model = Ridge(alpha=5.0)
    std_model.fit(std_X_train, np.abs(resid_train))

    std_X_test = _impute(_std_feature_frame(test_df), medians)
    pred_abs_resid = std_model.predict(std_X_test)
    return np.clip(pred_abs_resid * MEAN_ABS_TO_STD, config.MARGIN_STD_FLOOR, config.MARGIN_STD_CAP)


def gbm_feature_columns(df: pl.DataFrame) -> list[str]:
    suffixes = ("_roll8_diff", "_ewm_diff")
    cols = [c for c in df.columns if c.endswith(suffixes)]
    # off_plays (pace) is deliberately scoped to the totals model only (see
    # config.py) -- excluded here even though it'd otherwise auto-match the
    # suffix above, same as every other feature this project tests on one
    # market at a time rather than letting it leak in via auto-detection.
    cols = [c for c in cols if not c.startswith("off_plays")]
    cols += [
        "elo_diff", "rest_diff", "div_game", "qb_continuity_diff",
        "neutral_site", "is_dome", "temp_filled", "wind_filled",
    ]
    # injury_diff deliberately excluded here: kept for ridge (see config.py),
    # but reverted for LightGBM -- consistently made it slightly worse, same
    # pattern as the prime-time/surface features.
    # Both the raw-points and residual-based referee-crew tendency features
    # were tried and reverted here too -- see config.py's note near
    # MARGIN_MODEL_FEATURES.
    return [c for c in cols if c in df.columns]


def gbm_total_feature_columns(df: pl.DataFrame) -> list[str]:
    """LightGBM's totals feature set: every combined (summed) form feature
    (feature_engineering.TOTAL_SUM_PREFIXES) plus weather/context -- the
    totals analogue of gbm_feature_columns, which uses _diff features."""
    suffixes = ("_roll8_sum", "_ewm_sum")
    cols = [c for c in df.columns if c.endswith(suffixes)]
    # off_plays (pace) deliberately excluded here: kept for ridge_total (see
    # config.py), but reverted for LightGBM -- consistently made it worse.
    cols = [c for c in cols if not c.startswith("off_plays")]
    cols += ["div_game", "neutral_site", "is_dome", "temp_filled", "wind_filled"]
    return [c for c in cols if c in df.columns]


@dataclass
class WalkForwardResult:
    predictions: pl.DataFrame
    margin_std: float
    feature_cols: list[str]
    train_medians: dict = field(default_factory=dict)
    contributions: pl.DataFrame | None = None


def _prep_xy(df: pl.DataFrame, feature_cols: list[str], target_col: str = "home_margin"):
    pdf = df.select(feature_cols + [target_col]).to_pandas()
    X = pdf[feature_cols]
    y = pdf[target_col]
    return X, y


def _impute(X, medians: dict):
    # A median itself can be NaN if a feature has ZERO non-null values across
    # the whole training pool -- not hypothetical: ref_points_ewm is null for
    # every game before officials data starts (2015), so the very earliest
    # walk-forward folds (trained only on pre-2015 seasons) would otherwise
    # fillna a NaN median with another NaN and crash downstream in sklearn.
    # Falling back to 0.0 for those columns only matches how every other
    # partial-coverage feature here degrades (a neutral/no-signal default)
    # rather than raising.
    medians = {k: (v if pd.notna(v) else 0.0) for k, v in medians.items()}
    return X.fillna(medians)


def walk_forward_ridge(df: pl.DataFrame, feature_cols: list[str] | None = None) -> WalkForwardResult:
    feature_cols = feature_cols or config.MARGIN_MODEL_FEATURES
    return _walk_forward(df, feature_cols, lambda: Ridge(alpha=config.RIDGE_ALPHA_MARGIN), "ridge_baseline")


def walk_forward_lgbm(df: pl.DataFrame, feature_cols: list[str] | None = None) -> WalkForwardResult:
    feature_cols = feature_cols or gbm_feature_columns(df)
    return _walk_forward(df, feature_cols, lambda: lgb.LGBMRegressor(**config.LGBM_PARAMS), "lgbm")


def walk_forward_ridge_total(df: pl.DataFrame, feature_cols: list[str] | None = None) -> WalkForwardResult:
    """Totals (over/under) analogue of walk_forward_ridge: same walk-forward
    discipline, predicting total_points instead of home_margin."""
    feature_cols = feature_cols or config.TOTAL_MODEL_FEATURES
    return _walk_forward(
        df, feature_cols, lambda: Ridge(alpha=config.RIDGE_ALPHA_TOTAL), "ridge_total",
        target_col="total_points", pred_col="predicted_total",
    )


def walk_forward_lgbm_total(df: pl.DataFrame, feature_cols: list[str] | None = None) -> WalkForwardResult:
    """Totals analogue of walk_forward_lgbm."""
    feature_cols = feature_cols or gbm_total_feature_columns(df)
    return _walk_forward(
        df, feature_cols, lambda: lgb.LGBMRegressor(**config.LGBM_PARAMS), "lgbm_total",
        target_col="total_points", pred_col="predicted_total",
    )


def _walk_forward(
    df: pl.DataFrame, feature_cols: list[str], model_factory, model_name: str,
    target_col: str = "home_margin", pred_col: str = "predicted_margin",
) -> WalkForwardResult:
    df = df.filter(pl.col(target_col).is_not_null())
    seasons = sorted(df["season"].unique().to_list())
    test_seasons = [s for s in seasons if s >= FIRST_PREDICTABLE_SEASON]

    all_preds = []
    all_contribs = []
    resid_train_pool = []
    last_medians = {}

    for test_season in test_seasons:
        train_df = df.filter(pl.col("season") < test_season)
        test_df = df.filter(pl.col("season") == test_season)
        if train_df.height < 50 or test_df.height == 0:
            continue

        X_train, y_train = _prep_xy(train_df, feature_cols, target_col)
        medians = X_train.median().to_dict()
        last_medians = medians
        X_train = _impute(X_train, medians)

        model = model_factory()
        model.fit(X_train, y_train)

        X_test, y_test = _prep_xy(test_df, feature_cols, target_col)
        X_test = _impute(X_test, medians)
        pred = model.predict(X_test)

        train_pred = model.predict(X_train)
        resid_train = y_train.to_numpy() - train_pred
        resid_train_pool.append(resid_train)

        out = test_df.select(["game_id", "season", "week", "home_team", "away_team", target_col]).to_pandas()
        out[pred_col] = pred
        out["predicted_std"] = _fit_predict_conditional_std(train_df, test_df, resid_train)
        out["model"] = model_name
        all_preds.append(out)

        if hasattr(model, "coef_"):
            contrib = X_test * model.coef_
            contrib.columns = [f"{c}_contrib" for c in feature_cols]
            contrib.insert(0, "game_id", test_df["game_id"].to_pandas().to_numpy())
            contrib["intercept_contrib"] = model.intercept_
            all_contribs.append(contrib)

    predictions = pl.from_pandas(pd.concat(all_preds, ignore_index=True))
    margin_std = float(np.std(np.concatenate(resid_train_pool), ddof=1))
    contributions = pl.from_pandas(pd.concat(all_contribs, ignore_index=True)) if all_contribs else None

    logger.info(f"[{model_name}] walk-forward predictions: {predictions.height} games, resid std={margin_std:.2f}")
    return WalkForwardResult(
        predictions=predictions,
        margin_std=margin_std,
        feature_cols=feature_cols,
        train_medians=last_medians,
        contributions=contributions,
    )


# Context columns pulled back in from modeling_table for the ensemble's own
# conditional-std fit -- ridge_result/lgbm_result predictions only carry
# game_id/season/week/teams/margin, not the raw feature columns.
_STD_CONTEXT_COLS = ["game_id", "elo_diff", "div_game", "game_type", "wind_filled", "is_dome", "rest_diff"]


def walk_forward_ensemble(
    modeling_table: pl.DataFrame, ridge_result: WalkForwardResult, lgbm_result: WalkForwardResult,
    target_col: str = "home_margin", pred_col: str = "predicted_margin", model_name: str = "ensemble",
) -> WalkForwardResult:
    """Blend ridge and LightGBM's predictions with per-season walk-forward-fit
    weights -- a 2-feature meta-regression on each model's own prediction,
    refit each season on strictly earlier seasons -- rather than a fixed
    50/50 average. Lets the blend lean toward whichever model has actually
    been more accurate so far instead of assuming a static split forever.
    Needs at least one prior PREDICTED season to fit blend weights on, so it
    starts one season later than the base models. `target_col`/`pred_col`/
    `model_name` let this same function serve both the margin ensemble and
    the totals ensemble (walk_forward_ensemble_total)."""
    ridge_df = ridge_result.predictions.rename({pred_col: "pred_ridge"})
    lgbm_df = lgbm_result.predictions.select(["game_id", pred_col]).rename({pred_col: "pred_lgbm"})
    merged = ridge_df.join(lgbm_df, on="game_id", how="inner").join(
        modeling_table.select(_STD_CONTEXT_COLS), on="game_id", how="left"
    )

    seasons = sorted(merged["season"].unique().to_list())
    test_seasons = [s for s in seasons if s >= FIRST_PREDICTABLE_SEASON + 1]

    all_preds = []
    resid_train_pool = []
    meta_cols = ["pred_ridge", "pred_lgbm"]

    for test_season in test_seasons:
        train_df = merged.filter(pl.col("season") < test_season)
        test_df = merged.filter(pl.col("season") == test_season)
        if train_df.height < 50 or test_df.height == 0:
            continue

        X_train, y_train = _prep_xy(train_df, meta_cols, target_col)
        meta = Ridge(alpha=1.0)
        meta.fit(X_train, y_train)

        X_test, _ = _prep_xy(test_df, meta_cols, target_col)
        pred = meta.predict(X_test)
        train_pred = meta.predict(X_train)
        resid_train = y_train.to_numpy() - train_pred
        resid_train_pool.append(resid_train)

        out = test_df.select(["game_id", "season", "week", "home_team", "away_team", target_col]).to_pandas()
        out[pred_col] = pred
        out["predicted_std"] = _fit_predict_conditional_std(train_df, test_df, resid_train)
        out["model"] = model_name
        all_preds.append(out)

        logger.info(
            f"[{model_name}] {test_season}: weight(ridge)={meta.coef_[0]:.2f} weight(lgbm)={meta.coef_[1]:.2f} "
            f"intercept={meta.intercept_:.2f}"
        )

    predictions = pl.from_pandas(pd.concat(all_preds, ignore_index=True))
    margin_std = float(np.std(np.concatenate(resid_train_pool), ddof=1))

    logger.info(f"[{model_name}] walk-forward predictions: {predictions.height} games, resid std={margin_std:.2f}")
    return WalkForwardResult(predictions=predictions, margin_std=margin_std, feature_cols=meta_cols)


def walk_forward_ensemble_total(
    modeling_table: pl.DataFrame, ridge_total_result: WalkForwardResult, lgbm_total_result: WalkForwardResult
) -> WalkForwardResult:
    """Totals analogue of walk_forward_ensemble."""
    return walk_forward_ensemble(
        modeling_table, ridge_total_result, lgbm_total_result,
        target_col="total_points", pred_col="predicted_total", model_name="ensemble_total",
    )


# --- Live prediction: games that haven't been played yet ---------------------
#
# Everything above backtests completed history. These train on EVERY
# completed game to date (no season held out -- there's nothing left to
# validate a future game against) and predict games that don't have a result
# yet but already have a posted market line. Not walk-forward validated by
# construction; see live_picks.py for the probability/edge/calibration layer
# built on top of these.

LIVE_MARKET_COLS = [
    "game_id", "season", "week", "gameday", "home_team", "away_team",
    "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_spread_odds", "away_spread_odds",
]


def _predict_upcoming(
    modeling_table: pl.DataFrame, feature_cols: list[str], model_factory,
    target_col: str = "home_margin", pred_col: str = "predicted_margin",
) -> pd.DataFrame | None:
    played = modeling_table.filter(pl.col(target_col).is_not_null())
    upcoming = modeling_table.filter(pl.col(target_col).is_null() & pl.col("spread_line").is_not_null())
    if upcoming.height == 0 or played.height < 50:
        return None

    X_train, y_train = _prep_xy(played, feature_cols, target_col)
    medians = X_train.median().to_dict()
    X_train = _impute(X_train, medians)

    model = model_factory()
    model.fit(X_train, y_train)

    X_test, _ = _prep_xy(upcoming, feature_cols, target_col)
    X_test = _impute(X_test, medians)
    pred = model.predict(X_test)

    train_pred = model.predict(X_train)
    resid_train = y_train.to_numpy() - train_pred

    out = upcoming.select(LIVE_MARKET_COLS).to_pandas()
    out[pred_col] = pred
    out["predicted_std"] = _fit_predict_conditional_std(played, upcoming, resid_train)
    return out


def predict_upcoming_ridge(modeling_table: pl.DataFrame) -> pd.DataFrame | None:
    return _predict_upcoming(modeling_table, config.MARGIN_MODEL_FEATURES, lambda: Ridge(alpha=config.RIDGE_ALPHA_MARGIN))


def predict_upcoming_lgbm(modeling_table: pl.DataFrame) -> pd.DataFrame | None:
    return _predict_upcoming(modeling_table, gbm_feature_columns(modeling_table), lambda: lgb.LGBMRegressor(**config.LGBM_PARAMS))


def predict_upcoming_ridge_total(modeling_table: pl.DataFrame) -> pd.DataFrame | None:
    """Totals analogue of predict_upcoming_ridge."""
    return _predict_upcoming(
        modeling_table, config.TOTAL_MODEL_FEATURES, lambda: Ridge(alpha=config.RIDGE_ALPHA_TOTAL),
        target_col="total_points", pred_col="predicted_total",
    )


def predict_upcoming_lgbm_total(modeling_table: pl.DataFrame) -> pd.DataFrame | None:
    """Totals analogue of predict_upcoming_lgbm."""
    return _predict_upcoming(
        modeling_table, gbm_total_feature_columns(modeling_table), lambda: lgb.LGBMRegressor(**config.LGBM_PARAMS),
        target_col="total_points", pred_col="predicted_total",
    )


def predict_upcoming_ensemble(
    modeling_table: pl.DataFrame,
    ridge_result: WalkForwardResult,
    lgbm_result: WalkForwardResult,
    ridge_live: pd.DataFrame | None,
    lgbm_live: pd.DataFrame | None,
    target_col: str = "home_margin",
    pred_col: str = "predicted_margin",
) -> pd.DataFrame | None:
    """Same blend idea as walk_forward_ensemble, but the meta-regression is
    fit on ALL historical out-of-fold (pred_ridge, pred_lgbm, target) triples
    at once (no season held out) and applied to the live pair. `target_col`/
    `pred_col` let this serve both the margin and totals ensembles."""
    if ridge_live is None or lgbm_live is None or ridge_live.empty or lgbm_live.empty:
        return None

    hist_ridge = ridge_result.predictions.select(["game_id", target_col, pred_col]).rename({pred_col: "pred_ridge"})
    hist_lgbm = lgbm_result.predictions.select(["game_id", pred_col]).rename({pred_col: "pred_lgbm"})
    hist = hist_ridge.join(hist_lgbm, on="game_id", how="inner")

    X_train, y_train = _prep_xy(hist, ["pred_ridge", "pred_lgbm"], target_col)
    meta = Ridge(alpha=1.0)
    meta.fit(X_train, y_train)
    train_pred = meta.predict(X_train)
    resid_train = y_train.to_numpy() - train_pred

    live = ridge_live.rename(columns={pred_col: "pred_ridge"}).merge(
        lgbm_live[["game_id", pred_col]].rename(columns={pred_col: "pred_lgbm"}), on="game_id"
    )
    live[pred_col] = meta.predict(live[["pred_ridge", "pred_lgbm"]])
    live = live.drop(columns=["pred_ridge", "pred_lgbm", "predicted_std"])

    hist_ctx = hist.join(modeling_table.select(_STD_CONTEXT_COLS), on="game_id", how="left")
    live_ctx = pl.from_pandas(live[["game_id"]]).join(modeling_table.select(_STD_CONTEXT_COLS), on="game_id", how="left")
    live["predicted_std"] = _fit_predict_conditional_std(hist_ctx, live_ctx, resid_train)

    logger.info(f"[ensemble live] weight(ridge)={meta.coef_[0]:.2f} weight(lgbm)={meta.coef_[1]:.2f} intercept={meta.intercept_:.2f}")
    return live


def predict_upcoming_ensemble_total(
    modeling_table: pl.DataFrame,
    ridge_total_result: WalkForwardResult,
    lgbm_total_result: WalkForwardResult,
    ridge_total_live: pd.DataFrame | None,
    lgbm_total_live: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Totals analogue of predict_upcoming_ensemble."""
    return predict_upcoming_ensemble(
        modeling_table, ridge_total_result, lgbm_total_result, ridge_total_live, lgbm_total_live,
        target_col="total_points", pred_col="predicted_total",
    )
