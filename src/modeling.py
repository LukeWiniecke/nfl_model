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
#
# wind_filled was dropped here along with every other weather feature -- see
# config.WEATHER_FEATURES_DISABLED for the train/serve-skew reason.
STD_FEATURES = ["elo_diff_abs", "div_game", "is_playoff", "is_dome", "rest_diff_abs"]

# Raw modeling-table columns _std_feature_frame derives STD_FEATURES from.
# Kept as its own list because the out-of-fold std pool in _walk_forward has
# to carry these forward per fold, and the ensemble re-joins them from the
# modeling table (predictions frames don't carry feature columns).
STD_RAW_CONTEXT_COLS = ["elo_diff", "div_game", "game_type", "is_dome", "rest_diff"]


def _std_context_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in STD_RAW_CONTEXT_COLS if c in df.columns]


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
    game's probability to 0/1 or blow it up into meaninglessness.

    `train_df`/`resid_train` MUST be an OUT-OF-FOLD pair (see _walk_forward's
    oof pool) wherever one is available, not a model's own in-sample training
    residuals. This is not a stylistic preference -- feeding in-sample
    residuals here silently breaks every probability downstream for any model
    that fits its training data closely. Measured on the previous version,
    which did exactly that: LightGBM's mean predicted_std came out at 9.13
    against a true out-of-sample residual std of 13.54 (std of standardized
    residuals = 1.51, i.e. ~50% overconfident on every game), while ridge --
    which barely overfits -- was fine at 1.005. That scale error was the real
    cause of the "raw probabilities are overconfident" finding in config.py,
    and it's why LightGBM's walk-forward Student's-t fit pinned to the dof
    FLOOR: a shape parameter was being asked to absorb a scale error it
    mathematically cannot fix.
    """
    std_X_train = _std_feature_frame(train_df)
    medians = std_X_train.median().to_dict()
    std_X_train = _impute(std_X_train, medians)

    std_model = Ridge(alpha=5.0)
    std_model.fit(std_X_train, np.abs(resid_train))

    std_X_test = _impute(_std_feature_frame(test_df), medians)
    pred_abs_resid = std_model.predict(std_X_test)
    return np.clip(pred_abs_resid * MEAN_ABS_TO_STD, config.MARGIN_STD_FLOOR, config.MARGIN_STD_CAP)


def _drop_disabled(cols: list[str], df: pl.DataFrame) -> list[str]:
    """Keep only columns that exist, minus anything config has disabled.

    Belt-and-braces against this file's recurring failure mode: LightGBM's
    feature sets are built by SUFFIX auto-detection plus a hardcoded list, so
    a feature dropped from config.py's explicit ridge lists can still reach
    LightGBM through either route. Weather was doing exactly that -- removed
    from MARGIN_MODEL_FEATURES/TOTAL_MODEL_FEATURES but still named literally
    below, so ridge stopped seeing it while LightGBM did not. Routing every
    list through here means config.WEATHER_FEATURES_DISABLED is authoritative
    for both models.
    """
    disabled = set(config.WEATHER_FEATURES_DISABLED)
    return [c for c in cols if c in df.columns and c not in disabled]


def gbm_feature_columns(df: pl.DataFrame) -> list[str]:
    suffixes = ("_roll8_diff", "_ewm_diff")
    cols = [c for c in df.columns if c.endswith(suffixes)]
    # off_plays (pace) is deliberately scoped to the totals model only (see
    # config.py) -- excluded here even though it'd otherwise auto-match the
    # suffix above, same as every other feature this project tests on one
    # market at a time rather than letting it leak in via auto-detection.
    #
    # pen_* (penalties committed) excluded for the same reason. This suffix
    # auto-detection is how qb_cpoe_ewm silently re-entered LightGBM after
    # being dropped from the explicit ridge lists (see config.py), so any new
    # raw stat must be opted IN here rather than left to match by accident.
    cols = [c for c in cols if not c.startswith(("off_plays", "pen_"))]
    cols += [
        "elo_diff", "rest_diff", "div_game", "qb_continuity_diff",
        "neutral_site", "is_dome",
    ]
    # injury_diff deliberately excluded here: kept for ridge (see config.py),
    # but reverted for LightGBM -- consistently made it slightly worse, same
    # pattern as the prime-time/surface features.
    # Both the raw-points and residual-based referee-crew tendency features
    # were tried and reverted here too -- see config.py's note near
    # MARGIN_MODEL_FEATURES.
    # temp_filled/wind_filled removed -- see config.WEATHER_FEATURES_DISABLED.
    return _drop_disabled(cols, df)


def gbm_total_feature_columns(df: pl.DataFrame) -> list[str]:
    """LightGBM's totals feature set: every combined (summed) form feature
    (feature_engineering.TOTAL_SUM_PREFIXES) plus weather/context -- the
    totals analogue of gbm_feature_columns, which uses _diff features."""
    suffixes = ("_roll8_sum", "_ewm_sum")
    cols = [c for c in df.columns if c.endswith(suffixes)]
    # off_plays (pace) deliberately excluded here: kept for ridge_total (see
    # config.py), but reverted for LightGBM -- consistently made it worse.
    # pen_* excluded by the same opt-in rule as in gbm_feature_columns.
    cols = [c for c in cols if not c.startswith(("off_plays", "pen_"))]
    # temp_filled/wind_filled removed -- see config.WEATHER_FEATURES_DISABLED.
    cols += ["div_game", "neutral_site", "is_dome"]
    return _drop_disabled(cols, df)


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


def _linear_contributions(model, X: pd.DataFrame, rows: pl.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-game coefficient x (imputed) value for each feature, plus the
    intercept -- the terms sum exactly to a linear model's prediction."""
    contrib = X * model.coef_
    contrib.columns = [f"{c}_contrib" for c in feature_cols]
    contrib.insert(0, "game_id", rows["game_id"].to_pandas().to_numpy())
    contrib["intercept_contrib"] = model.intercept_
    return contrib


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
    last_medians = {}

    # Out-of-fold pool for the conditional-std model. Each completed fold's
    # test season is, by construction, a set of predictions the model never
    # trained on -- exactly the residual distribution a future game will be
    # drawn from. Accumulating them as folds complete costs nothing (the
    # predictions are already computed) and gives the std model an honest
    # target, instead of the in-sample residuals it used to get.
    #
    # Ordering matters for correctness: this fold's OWN residuals are appended
    # only AFTER its predicted_std has been computed, so the std model for
    # season S only ever sees seasons < S. Same walk-forward rule as
    # everything else here.
    oof_ctx: list[pl.DataFrame] = []
    oof_resid: list[np.ndarray] = []

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

        # The earliest fold has no completed folds behind it yet, so there are
        # no out-of-fold residuals to fit on. Fall back to in-sample residuals
        # there rather than skipping the season -- imperfect, but it affects
        # one fold's std only, and the pool takes over from the next season on.
        n_oof = sum(len(r) for r in oof_resid)
        if n_oof >= config.MIN_OOF_STD_GAMES:
            std_fit_df = pl.concat(oof_ctx, how="vertical_relaxed")
            std_fit_resid = np.concatenate(oof_resid)
        else:
            std_fit_df = train_df
            std_fit_resid = y_train.to_numpy() - model.predict(X_train)
            logger.info(
                f"[{model_name}] {test_season}: only {n_oof} out-of-fold residuals available "
                f"(<{config.MIN_OOF_STD_GAMES}), falling back to in-sample residuals for the std model"
            )

        out = test_df.select(["game_id", "season", "week", "home_team", "away_team", target_col]).to_pandas()
        out[pred_col] = pred
        out["predicted_std"] = _fit_predict_conditional_std(std_fit_df, test_df, std_fit_resid)
        out["model"] = model_name
        all_preds.append(out)

        oof_ctx.append(test_df.select(_std_context_columns(test_df)))
        oof_resid.append(y_test.to_numpy() - pred)

        if hasattr(model, "coef_"):
            all_contribs.append(_linear_contributions(model, X_test, test_df, feature_cols))

    predictions = pl.from_pandas(pd.concat(all_preds, ignore_index=True))
    # Scalar fallback std, now also out-of-fold: this is the honest spread of
    # the model's error on unseen games, not the flattered in-sample one.
    margin_std = float(np.std(np.concatenate(oof_resid), ddof=1))
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
_STD_CONTEXT_COLS = ["game_id"] + STD_RAW_CONTEXT_COLS


# --- config.ENSEMBLE_MARKET_ANCHOR == "constrained" helpers ------------------
# predicted = line + E @ b, where E holds one column per base model of
# (base prediction - line) and b is fit through the origin on
# (target - line). Frame-agnostic (polars in walk-forward, pandas live).

def _anchor_arrays(df, base_cols: list[str], market_col: str) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(df[market_col], dtype=float)
    E = np.column_stack([np.asarray(df[c], dtype=float) - m for c in base_cols])
    return m, E


def _fit_anchor_slope(E: np.ndarray, r: np.ndarray) -> np.ndarray:
    """One base model: closed form b = <e,r>/<e,e> -- the same statistic
    probabilities.fit_edge_shrinkage_walk_forward computes for the betting
    layer -- clipped to the same guard rails. Two base models: through-origin
    least squares on [e_ridge, e_lgbm] (not exercised while
    ENSEMBLE_INCLUDE_LGBM is False)."""
    if E.shape[1] == 1:
        e = E[:, 0]
        denom = float(np.dot(e, e))
        b = float(np.dot(e, r) / denom) if denom > 0 else 0.0
        return np.array([np.clip(b, config.SHRINKAGE_SLOPE_FLOOR, config.SHRINKAGE_SLOPE_CAP)])
    return np.linalg.lstsq(E, r, rcond=None)[0]


def _anchor_log(base_cols: list[str], b: np.ndarray, market_col: str) -> str:
    return " ".join(f"b({c.replace('pred_', '')})={v:.3f}" for c, v in zip(base_cols, b)) + f" anchor={market_col}"


def _legacy_meta_predict(X_train, y_train, X_test, base_cols: list[str]) -> tuple[np.ndarray, np.ndarray, str]:
    """The ENSEMBLE_MARKET_ANCHOR = None ensemble, fit on the side: the same
    Ridge(alpha=1.0) WITH intercept on the base-model predictions alone
    (predicted = a + w * pred_ridge), on the same rows the `else` branches of
    walk_forward_ensemble / predict_upcoming_ensemble would use. In
    "constrained" mode it fills `{pred_col}_unanchored`, so bet selection
    (config.ATS_SELECT_ON_UNANCHORED, config.ML_MARKET_MODE) runs on exactly
    the edge the README validated its thresholds on and reproduces that bet
    book. Frame-agnostic: pandas from _prep_xy in walk-forward, the live
    pandas frame for upcoming games -- both are indexed by column name.

    Returns (test prediction, TRAIN prediction, log line). The train
    prediction is not cosmetic: the conditional-std model's first-fold
    fallback is fit on `target - train_pred`, so a legacy std meant to equal
    the unanchored run's std bit-for-bit needs the LEGACY train prediction
    there, not the anchored one."""
    meta = Ridge(alpha=1.0)
    meta.fit(X_train[base_cols], y_train)
    weights = " ".join(f"weight({c.replace('pred_', '')})={w:.4f}" for c, w in zip(base_cols, meta.coef_))
    return (
        meta.predict(X_test[base_cols]),
        meta.predict(X_train[base_cols]),
        f"{weights} intercept={meta.intercept_:.4f}",
    )


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
    merged = ridge_df
    if config.ENSEMBLE_INCLUDE_LGBM:
        lgbm_df = lgbm_result.predictions.select(["game_id", pred_col]).rename({pred_col: "pred_lgbm"})
        merged = merged.join(lgbm_df, on="game_id", how="inner")
    merged = merged.join(modeling_table.select(_STD_CONTEXT_COLS), on="game_id", how="left")
    # Market line for this target (see config.ENSEMBLE_MARKET_ANCHOR). Joined
    # regardless of the mode; rows with no posted line can't be scored against
    # it, so they are dropped here rather than left for _prep_xy (no imputation).
    market_col = "spread_line" if target_col == "home_margin" else "total_line"
    merged = merged.join(modeling_table.select(["game_id", market_col]), on="game_id", how="left")
    merged = merged.filter(pl.col(market_col).is_not_null())

    seasons = sorted(merged["season"].unique().to_list())
    test_seasons = [s for s in seasons if s >= FIRST_PREDICTABLE_SEASON + 1]

    all_preds = []
    anchor = config.ENSEMBLE_MARKET_ANCHOR
    base_cols = [c for c in ["pred_ridge", "pred_lgbm"] if c in merged.columns]
    meta_cols = base_cols + ([market_col] if anchor == "free" else [])

    # Same out-of-fold std pool as _walk_forward -- see its comment. The meta
    # model is only a 2-feature ridge and barely overfits, so this matters
    # less here than for LightGBM, but there's no reason to leave a known
    # bias in place on one path and not the other.
    oof_ctx: list[pl.DataFrame] = []
    oof_resid: list[np.ndarray] = []
    # Second, parallel residual pool for the LEGACY (unanchored) prediction --
    # see predicted_std_unanchored below. Same rows, same order, same
    # fold-by-fold accumulation; only the prediction the residual is taken
    # against differs.
    oof_resid_unanchored: list[np.ndarray] = []

    for test_season in test_seasons:
        train_df = merged.filter(pl.col("season") < test_season)
        test_df = merged.filter(pl.col("season") == test_season)
        if train_df.height < 50 or test_df.height == 0:
            continue

        X_train, y_train = _prep_xy(train_df, meta_cols, target_col)
        X_test, y_test = _prep_xy(test_df, meta_cols, target_col)
        if anchor == "constrained":
            # b on strictly earlier seasons' out-of-fold rows only (train_df).
            m_train, E_train = _anchor_arrays(train_df, base_cols, market_col)
            b = _fit_anchor_slope(E_train, y_train.to_numpy() - m_train)
            train_pred = m_train + E_train @ b
            m_test, E_test = _anchor_arrays(test_df, base_cols, market_col)
            pred = m_test + E_test @ b
            # The LEGACY ensemble prediction: the one-feature meta-ridge on
            # the base predictions that the `else` branch below fits when
            # ENSEMBLE_MARKET_ANCHOR is None -- same Ridge(alpha=1.0), same
            # columns, same train pool (strictly earlier seasons' out-of-fold
            # rows; in this mode meta_cols == base_cols, so X_train IS that
            # pool). Reported alongside `pred` so the betting layer can SELECT
            # on exactly the edge the README validated its thresholds on
            # (a_S + w_S * pred_ridge - line) while everything still REPORTS
            # the anchored margin -- see config.ATS_SELECT_ON_UNANCHORED.
            # NOT raw pred_ridge: the betting layer's through-origin shrinkage
            # slope and the bar derived from it are not invariant to the meta
            # intercept, so selecting on raw pred_ridge moved the bet
            # universe (r3-E18: 813 dev bets against the validated 692).
            pred_unanchored, train_pred_unanchored, legacy_log = _legacy_meta_predict(
                X_train, y_train, X_test, base_cols
            )
            fit_log = f"{_anchor_log(base_cols, b, market_col)} | legacy {legacy_log}"
        else:
            meta = Ridge(alpha=1.0)
            meta.fit(X_train, y_train)
            train_pred = meta.predict(X_train)
            pred = meta.predict(X_test)
            # No anchoring applied, so there is nothing to undo: the
            # unanchored column is the prediction itself and every consumer
            # of it collapses to today's behaviour.
            pred_unanchored = pred
            train_pred_unanchored = train_pred
            weights = " ".join(f"weight({c.replace('pred_', '')})={w:.2f}" for c, w in zip(meta_cols, meta.coef_))
            fit_log = f"{weights} intercept={meta.intercept_:.2f}"

        n_oof = sum(len(r) for r in oof_resid)
        if n_oof >= config.MIN_OOF_STD_GAMES:
            std_fit_df = pl.concat(oof_ctx, how="vertical_relaxed")
            std_fit_resid = np.concatenate(oof_resid)
            std_fit_resid_unanchored = np.concatenate(oof_resid_unanchored)
        else:
            std_fit_df = train_df
            std_fit_resid = y_train.to_numpy() - train_pred
            std_fit_resid_unanchored = y_train.to_numpy() - train_pred_unanchored

        out = test_df.select(["game_id", "season", "week", "home_team", "away_team", target_col]).to_pandas()
        out[pred_col] = pred
        out[f"{pred_col}_unanchored"] = pred_unanchored
        out["predicted_std"] = _fit_predict_conditional_std(std_fit_df, test_df, std_fit_resid)
        # The conditional std the LEGACY prediction implies. Identical model,
        # identical training pool, identical fold ordering -- the ONLY
        # difference is that the |residual| target is taken against
        # pred_unanchored instead of the anchored pred.
        #
        # Why it has to exist: for the SPREAD, anchoring cancels
        # (anchored - line = b * (legacy - line), one positive scalar per
        # season), so selecting on the legacy edge needs nothing but the
        # legacy margin. In the PROBABILITY domain it does not cancel --
        # home_win_probability_t divides by a PER-GAME std -- so a legacy
        # margin divided by the ANCHORED std is neither the legacy
        # probability nor the anchored one. That mismatch is exactly why
        # round-4 E20's moneyline book missed the validated one by 8
        # development bets. When ENSEMBLE_MARKET_ANCHOR is None,
        # pred_unanchored IS pred, so this column equals predicted_std
        # exactly and every consumer collapses to previous behaviour.
        out["predicted_std_unanchored"] = _fit_predict_conditional_std(
            std_fit_df, test_df, std_fit_resid_unanchored
        )
        out["model"] = model_name
        all_preds.append(out)

        oof_ctx.append(test_df.select(_std_context_columns(test_df)))
        oof_resid.append(y_test.to_numpy() - pred)
        oof_resid_unanchored.append(y_test.to_numpy() - pred_unanchored)

        logger.info(f"[{model_name}] {test_season}: {fit_log}")

    predictions = pl.from_pandas(pd.concat(all_preds, ignore_index=True))
    margin_std = float(np.std(np.concatenate(oof_resid), ddof=1))

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


def oof_std_pool(
    modeling_table: pl.DataFrame, wf_result: WalkForwardResult, pred_col: str = "predicted_margin",
    target_col: str = "home_margin",
) -> tuple[pl.DataFrame, np.ndarray]:
    """Rebuild the (context, residual) pair the conditional-std model should
    be fit on for a LIVE prediction, from a completed walk-forward run's
    out-of-fold predictions. The live model itself trains on all history with
    nothing held out, so it has no out-of-fold residuals of its own -- but the
    walk-forward run over the same features and hyperparameters does, and
    those are the right estimate of how wrong this model is on a game it has
    never seen. Using the live model's in-sample residuals here instead is the
    same scale bug documented in _fit_predict_conditional_std."""
    resid_df = wf_result.predictions.select(["game_id", target_col, pred_col]).join(
        modeling_table.select(["game_id"] + STD_RAW_CONTEXT_COLS), on="game_id", how="inner"
    )
    resid = (resid_df[target_col] - resid_df[pred_col]).to_numpy()
    return resid_df.select(_std_context_columns(resid_df)), resid


def _predict_upcoming(
    modeling_table: pl.DataFrame, feature_cols: list[str], model_factory,
    target_col: str = "home_margin", pred_col: str = "predicted_margin",
    wf_result: WalkForwardResult | None = None,
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

    if wf_result is not None:
        std_fit_df, std_fit_resid = oof_std_pool(modeling_table, wf_result, pred_col, target_col)
    else:
        std_fit_df = played.select(_std_context_columns(played))
        std_fit_resid = y_train.to_numpy() - model.predict(X_train)

    out = upcoming.select(LIVE_MARKET_COLS).to_pandas()
    out[pred_col] = pred
    out["predicted_std"] = _fit_predict_conditional_std(std_fit_df, upcoming, std_fit_resid)
    # Linear margin models only: the live model's per-feature breakdown, for the
    # dashboard's "why did the model say this" on games not yet played.
    # Carried on attrs so the prediction frame's columns stay unchanged
    # (main.py pops it straight off again).
    if hasattr(model, "coef_") and target_col == "home_margin":
        out.attrs["contributions"] = _linear_contributions(model, X_test, upcoming, feature_cols)
    return out


def predict_upcoming_ridge(
    modeling_table: pl.DataFrame, wf_result: WalkForwardResult | None = None
) -> pd.DataFrame | None:
    return _predict_upcoming(
        modeling_table, config.MARGIN_MODEL_FEATURES, lambda: Ridge(alpha=config.RIDGE_ALPHA_MARGIN),
        wf_result=wf_result,
    )


def predict_upcoming_lgbm(
    modeling_table: pl.DataFrame, wf_result: WalkForwardResult | None = None
) -> pd.DataFrame | None:
    return _predict_upcoming(
        modeling_table, gbm_feature_columns(modeling_table), lambda: lgb.LGBMRegressor(**config.LGBM_PARAMS),
        wf_result=wf_result,
    )


def predict_upcoming_ridge_total(
    modeling_table: pl.DataFrame, wf_result: WalkForwardResult | None = None
) -> pd.DataFrame | None:
    """Totals analogue of predict_upcoming_ridge."""
    return _predict_upcoming(
        modeling_table, config.TOTAL_MODEL_FEATURES, lambda: Ridge(alpha=config.RIDGE_ALPHA_TOTAL),
        target_col="total_points", pred_col="predicted_total", wf_result=wf_result,
    )


def predict_upcoming_lgbm_total(
    modeling_table: pl.DataFrame, wf_result: WalkForwardResult | None = None
) -> pd.DataFrame | None:
    """Totals analogue of predict_upcoming_lgbm."""
    return _predict_upcoming(
        modeling_table, gbm_total_feature_columns(modeling_table), lambda: lgb.LGBMRegressor(**config.LGBM_PARAMS),
        target_col="total_points", pred_col="predicted_total", wf_result=wf_result,
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

    hist = ridge_result.predictions.select(["game_id", target_col, pred_col]).rename({pred_col: "pred_ridge"})
    live = ridge_live.rename(columns={pred_col: "pred_ridge"})
    meta_cols = ["pred_ridge"]
    if config.ENSEMBLE_INCLUDE_LGBM:
        hist_lgbm = lgbm_result.predictions.select(["game_id", pred_col]).rename({pred_col: "pred_lgbm"})
        hist = hist.join(hist_lgbm, on="game_id", how="inner")
        live = live.merge(lgbm_live[["game_id", pred_col]].rename(columns={pred_col: "pred_lgbm"}), on="game_id")
        meta_cols.append("pred_lgbm")
    # Same market column as walk_forward_ensemble (config.ENSEMBLE_MARKET_ANCHOR).
    # `live` already carries spread_line/total_line via LIVE_MARKET_COLS.
    market_col = "spread_line" if target_col == "home_margin" else "total_line"
    hist = hist.join(modeling_table.select(["game_id", market_col]), on="game_id", how="left")
    hist = hist.filter(pl.col(market_col).is_not_null())
    anchor = config.ENSEMBLE_MARKET_ANCHOR
    base_cols = list(meta_cols)
    if anchor is not None:
        # A live row with no posted line can't be anchored to it: drop it
        # rather than feed NaN into the prediction. No-op on current data
        # (_predict_upcoming already requires spread_line on upcoming rows).
        live = live[live[market_col].notna()].reset_index(drop=True)
        if live.empty:
            return None
    if anchor == "free":
        meta_cols.append(market_col)

    X_train, y_train = _prep_xy(hist, meta_cols, target_col)
    if anchor == "constrained":
        # b on ALL historical out-of-fold rows (no season held out), same
        # formula as the walk-forward path.
        m_hist, E_hist = _anchor_arrays(hist, base_cols, market_col)
        b = _fit_anchor_slope(E_hist, y_train.to_numpy() - m_hist)
        train_pred = m_hist + E_hist @ b
        m_live, E_live = _anchor_arrays(live, base_cols, market_col)
        live[pred_col] = m_live + E_live @ b
        # Legacy ensemble prediction for selection -- see walk_forward_ensemble.
        # Fit on ALL historical out-of-fold rows (X_train, no season held
        # out), exactly as the `else` branch does for the live frame when
        # ENSEMBLE_MARKET_ANCHOR is None. Set BEFORE the base-model columns
        # are dropped below, since it reads them off `live`.
        live[f"{pred_col}_unanchored"], train_pred_unanchored, legacy_log = _legacy_meta_predict(
            X_train, y_train, live, base_cols
        )
        fit_log = f"{_anchor_log(base_cols, b, market_col)} | legacy {legacy_log}"
    else:
        meta = Ridge(alpha=1.0)
        meta.fit(X_train, y_train)
        train_pred = meta.predict(X_train)
        live[pred_col] = meta.predict(live[meta_cols])
        live[f"{pred_col}_unanchored"] = live[pred_col]
        train_pred_unanchored = train_pred
        weights = " ".join(f"weight({c.replace('pred_', '')})={w:.2f}" for c, w in zip(meta_cols, meta.coef_))
        fit_log = f"{weights} intercept={meta.intercept_:.2f}"
    # Drop only the base-model prediction columns -- never the market column;
    # the upcoming board needs spread_line/total_line downstream.
    live = live.drop(columns=base_cols + ["predicted_std"])

    # The ensemble's inputs here are already out-of-fold (both base models'
    # walk-forward predictions), so the meta model's residuals against them
    # are an honest out-of-sample spread -- unlike a base model's own
    # in-sample residuals. See _fit_predict_conditional_std.
    resid = y_train.to_numpy() - train_pred
    hist_ctx = hist.join(modeling_table.select(_STD_CONTEXT_COLS), on="game_id", how="left")
    live_ctx = pl.from_pandas(live[["game_id"]]).join(modeling_table.select(_STD_CONTEXT_COLS), on="game_id", how="left")
    live["predicted_std"] = _fit_predict_conditional_std(hist_ctx, live_ctx, resid)
    # Live analogue of predicted_std_unanchored in walk_forward_ensemble: the
    # same fit on the same rows, against the LEGACY prediction's residuals, so
    # the live moneyline selection path can be the legacy pipeline end to end
    # (config.ML_MARKET_MODE). Equals predicted_std exactly when
    # ENSEMBLE_MARKET_ANCHOR is None.
    live["predicted_std_unanchored"] = _fit_predict_conditional_std(
        hist_ctx, live_ctx, y_train.to_numpy() - train_pred_unanchored
    )

    logger.info(f"[ensemble live] {fit_log}")
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
