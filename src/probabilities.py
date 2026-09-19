"""Convert predicted margins into win probabilities and fair prices, and turn
market odds into de-vigged implied probabilities for comparison.
"""

import numpy as np
from scipy.stats import norm, t as student_t
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import config
from .utils import american_payout_multiplier, american_to_prob, prob_to_american, remove_vig_two_way


def fit_margin_std(actual_margin: np.ndarray, predicted_margin: np.ndarray) -> float:
    """Std dev of the model's residuals; used as the spread of the outcome
    distribution around a predicted margin. Should be fit on TRAIN residuals
    only, never on the games being predicted."""
    resid = np.asarray(actual_margin) - np.asarray(predicted_margin)
    std = float(np.std(resid, ddof=1))
    return std if std > 0 else config.DEFAULT_MARGIN_STD


def home_win_probability(predicted_margin: np.ndarray, margin_std: float) -> np.ndarray:
    """P(home_margin > 0) under a normal approximation centered on the
    predicted margin. NFL final margins are approximately normal/logistic;
    ties are a measure-zero edge case here and handled separately in grading."""
    predicted_margin = np.asarray(predicted_margin, dtype=float)
    return norm.cdf(predicted_margin / margin_std)


def ats_cover_probability(predicted_margin: np.ndarray, spread_line: np.ndarray, margin_std: float) -> np.ndarray:
    """P(home_margin > spread_line), i.e. probability the HOME side of the
    market spread covers, given the model's predicted margin."""
    predicted_margin = np.asarray(predicted_margin, dtype=float)
    spread_line = np.asarray(spread_line, dtype=float)
    return norm.cdf((predicted_margin - spread_line) / margin_std)


def over_probability(predicted_total: np.ndarray, total_line: np.ndarray, total_std: float) -> np.ndarray:
    """P(total_points > total_line), i.e. probability the OVER hits, given
    the model's predicted total. Same normal-approximation idea as
    ats_cover_probability, just centered on the totals market's line."""
    predicted_total = np.asarray(predicted_total, dtype=float)
    total_line = np.asarray(total_line, dtype=float)
    return norm.cdf((predicted_total - total_line) / total_std)


def home_win_probability_t(predicted_margin: np.ndarray, margin_std: float, dof: float) -> np.ndarray:
    """Same as home_win_probability, but with a Student's-t distribution
    (fatter tails) instead of Normal -- see fit_t_dof_walk_forward for why."""
    predicted_margin = np.asarray(predicted_margin, dtype=float)
    return student_t.cdf(predicted_margin / margin_std, df=dof)


def ats_cover_probability_t(predicted_margin: np.ndarray, spread_line: np.ndarray, margin_std: float, dof: float) -> np.ndarray:
    """Same as ats_cover_probability, but with a Student's-t distribution."""
    predicted_margin = np.asarray(predicted_margin, dtype=float)
    spread_line = np.asarray(spread_line, dtype=float)
    return student_t.cdf((predicted_margin - spread_line) / margin_std, df=dof)


def over_probability_t(predicted_total: np.ndarray, total_line: np.ndarray, total_std: float, dof: float) -> np.ndarray:
    """Same as over_probability, but with a Student's-t distribution."""
    predicted_total = np.asarray(predicted_total, dtype=float)
    total_line = np.asarray(total_line, dtype=float)
    return student_t.cdf((predicted_total - total_line) / total_std, df=dof)


def _fit_dof(z: np.ndarray) -> float:
    """Shared MLE core behind fit_t_dof_walk_forward and fit_dof_all_history:
    fit a Student's-t degrees-of-freedom to already-standardized residuals
    (loc/scale fixed at 0/1, only the shape parameter estimated), clipped to
    [config.T_DOF_FLOOR, config.T_DOF_CAP] (see fit_t_dof_walk_forward's
    docstring for why)."""
    dof, _, _ = student_t.fit(z, floc=0.0, fscale=1.0)
    return float(np.clip(dof, config.T_DOF_FLOOR, config.T_DOF_CAP))


def fit_dof_all_history(actual: np.ndarray, predicted: np.ndarray, std: np.ndarray) -> float:
    """Live-prediction analogue of fit_t_dof_walk_forward: fits ONE
    degrees-of-freedom on ALL historical (actual, predicted, std) residuals
    -- no season-by-season split needed, mirroring live_picks._calibrate_live's
    single isotonic fit, since there's only one "current" bucket of upcoming
    games to apply it to. Falls back to config.DEFAULT_T_DOF (practically
    Normal) if there isn't enough history yet (config.MIN_CALIBRATION_GAMES)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    std = np.asarray(std, dtype=float)
    if len(actual) < config.MIN_CALIBRATION_GAMES:
        return config.DEFAULT_T_DOF
    z = (actual - predicted) / std
    return _fit_dof(z)


def fit_t_dof_walk_forward(df, actual_col: str, predicted_col: str, std_col: str) -> np.ndarray:
    """Motivation: calibration_bins on the RAW (pre-isotonic) probabilities
    showed real, measured OVERCONFIDENCE -- predictions more extreme than
    reality in both directions (e.g. LightGBM's raw ML: bin predicting 14.8%
    actually won 31.7% of the time; bin predicting 94.7% actually won only
    78.2%) -- worst for LightGBM's ML market specifically (the one left
    genuinely uncalibrated), milder but same DIRECTION for ridge/ensemble.
    A Normal distribution's CDF falls off faster than reality warrants in
    the tails if the true residual distribution is fatter-tailed than
    Normal; a Student's-t (extra free degrees-of-freedom parameter,
    controlling tail fatness -- df->infinity converges to Normal) is a
    direct, one-parameter fix for exactly that shape mismatch.

    Returns one degrees-of-freedom value per row of `df` (constant within
    a season), fit walk-forward: for each season, fits `dof` by maximum
    likelihood on the STANDARDIZED residuals ((actual - predicted) / std)
    of strictly EARLIER seasons only, via scipy.stats.t.fit with loc/scale
    fixed at 0/1 (only the shape/df parameter is estimated) -- same walk-
    forward discipline as walk_forward_calibrate and the conditional-std
    model. Requires config.MIN_CALIBRATION_GAMES prior graded games before
    fitting starts, same floor as isotonic calibration uses; before that,
    falls back to config.DEFAULT_T_DOF (a high value -- practically Normal)
    since there's not enough data yet to trust a fitted tail shape.
    Clipped to [config.T_DOF_FLOOR, config.T_DOF_CAP]: below the floor, a
    t-distribution's variance is undefined/explosive; above the cap, it's
    indistinguishable from Normal anyway, so there's nothing to gain by
    letting a noisy MLE fit an arbitrarily large number.
    """
    actual = df[actual_col].to_numpy(dtype=float)
    predicted = df[predicted_col].to_numpy(dtype=float)
    std = df[std_col].to_numpy(dtype=float)
    season = df["season"].to_numpy()
    z = (actual - predicted) / std

    out = np.full(len(df), config.DEFAULT_T_DOF, dtype=float)
    for s in np.unique(season):
        train_mask = season < s
        test_mask = season == s
        if train_mask.sum() < config.MIN_CALIBRATION_GAMES:
            continue
        out[test_mask] = _fit_dof(z[train_mask])

    return out


def fit_edge_shrinkage_walk_forward(df, edge_col: str, resid_col: str) -> tuple[np.ndarray, np.ndarray]:
    """How much of the model's claimed edge is real?

    Regresses the REALIZED result against the line (`resid_col`, i.e.
    home_margin - spread_line) on the model's RAW claimed edge (`edge_col`,
    i.e. predicted_margin - spread_line), fit for each season on strictly
    EARLIER seasons only -- same walk-forward rule as walk_forward_calibrate
    and the conditional-std model.

    The slope is the fraction of a claimed edge that survives contact with
    reality: a slope of 1.0 would mean the model's edge is exactly right, 0.0
    that it is pure noise. Measured here it is ~0.20 for ridge, ~0.09 for
    LightGBM -- so a claimed 5-point edge is really worth about 1 point. That
    is the number every bet-selection threshold in this project was
    implicitly assuming was 1.0.

    Fit through the origin (no intercept) deliberately: the market is very
    close to unbiased, so the fitted intercept is negligible (measured:
    +0.002 for ridge, -0.007 for the ensemble, +0.04 for LightGBM), and
    dropping it keeps the resulting bet rule symmetric between the home and
    away side, which _bet_side assumes.

    Returns (slope, resid_sd) as one value per row of `df`, constant within a
    season. `resid_sd` is the walk-forward std of the residual-against-line,
    which adaptive_ats_threshold needs to convert a target win rate into
    points. Seasons without config.MIN_SHRINKAGE_GAMES of prior history get
    NaN in both, signalling the caller to fall back to a fixed threshold.
    """
    edge = df[edge_col].to_numpy(dtype=float)
    resid = df[resid_col].to_numpy(dtype=float)
    season = df["season"].to_numpy()

    slope_out = np.full(len(df), np.nan)
    sd_out = np.full(len(df), np.nan)

    for s in np.unique(season):
        train_mask = (season < s) & np.isfinite(edge) & np.isfinite(resid)
        if train_mask.sum() < config.MIN_SHRINKAGE_GAMES:
            continue
        e, r = edge[train_mask], resid[train_mask]
        denom = float(np.dot(e, e))
        if denom <= 0:
            continue
        slope = float(np.dot(e, r) / denom)
        test_mask = season == s
        slope_out[test_mask] = np.clip(slope, config.SHRINKAGE_SLOPE_FLOOR, config.SHRINKAGE_SLOPE_CAP)
        sd_out[test_mask] = float(np.std(r, ddof=1))

    return slope_out, sd_out


def fit_edge_shrinkage_all_history(edge: np.ndarray, resid: np.ndarray) -> tuple[float, float]:
    """Live-prediction analogue of fit_edge_shrinkage_walk_forward: one slope
    and one residual sd fit on ALL graded history, for application to the
    current batch of upcoming games. Returns (nan, nan) if there isn't enough
    history yet."""
    edge = np.asarray(edge, dtype=float)
    resid = np.asarray(resid, dtype=float)
    ok = np.isfinite(edge) & np.isfinite(resid)
    if ok.sum() < config.MIN_SHRINKAGE_GAMES:
        return float("nan"), float("nan")
    e, r = edge[ok], resid[ok]
    denom = float(np.dot(e, e))
    if denom <= 0:
        return float("nan"), float("nan")
    slope = float(np.clip(np.dot(e, r) / denom, config.SHRINKAGE_SLOPE_FLOOR, config.SHRINKAGE_SLOPE_CAP))
    return slope, float(np.std(r, ddof=1))


def breakeven_probability(american_odds: np.ndarray) -> np.ndarray:
    """Win rate a bet at these odds must hit just to break even: 1/(1+b),
    where b is profit per unit staked. -110 gives 0.5238."""
    b = american_payout_multiplier(american_odds)
    return 1.0 / (1.0 + b)


def market_moneyline_no_vig(home_moneyline: np.ndarray, away_moneyline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """De-vig the market's home/away moneylines into a fair two-way probability."""
    home_raw = american_to_prob(home_moneyline)
    away_raw = american_to_prob(away_moneyline)
    return remove_vig_two_way(home_raw, away_raw)


def market_spread_no_vig(home_spread_odds: np.ndarray, away_spread_odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """De-vig the market's spread-side prices into a fair two-way probability
    of each side covering."""
    home_raw = american_to_prob(home_spread_odds)
    away_raw = american_to_prob(away_spread_odds)
    return remove_vig_two_way(home_raw, away_raw)


def fair_moneyline_price(model_home_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Model-implied fair American odds for home and away, from the model's
    home win probability (no vig, since this is our own fair value, not a
    price we're quoting)."""
    model_home_prob = np.asarray(model_home_prob, dtype=float)
    home_odds = prob_to_american(model_home_prob)
    away_odds = prob_to_american(1.0 - model_home_prob)
    return home_odds, away_odds


def moneyline_edge(model_home_prob: np.ndarray, home_moneyline: np.ndarray, away_moneyline: np.ndarray) -> np.ndarray:
    """model_home_prob - no_vig_market_home_prob. Positive => model likes home
    relative to market; negative => model likes away."""
    market_home_prob, _ = market_moneyline_no_vig(home_moneyline, away_moneyline)
    return np.asarray(model_home_prob) - market_home_prob


def _logit_feature(prob: np.ndarray) -> np.ndarray:
    """logit(raw prob) as a single-column design matrix for Platt scaling,
    clipped away from 0/1 first (config.PLATT_PROB_CLIP) so a saturated CDF
    can never hand the logistic fit an infinite input."""
    p = np.clip(np.asarray(prob, dtype=float), config.PLATT_PROB_CLIP, 1.0 - config.PLATT_PROB_CLIP)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def fit_calibrator(train_prob: np.ndarray, train_outcome: np.ndarray):
    """Fit ONE recalibration curve on (raw prob, outcome) training pairs and
    return it as a callable raw_prob -> calibrated_prob. The curve family is
    chosen by config.CALIBRATION_METHOD (see the note there for why):

      'platt'    -- logistic regression of outcome on logit(raw prob), two
                    parameters (slope + intercept): can stretch/compress a
                    raw probability but cannot invent structure.
      'isotonic' -- the previous method: sklearn IsotonicRegression, a
                    monotone step function.

    Shared by walk_forward_calibrate (backtest) and live_picks._calibrate_live
    (upcoming board) so the two can never disagree on the curve family.
    """
    train_prob = np.asarray(train_prob, dtype=float)
    train_outcome = np.asarray(train_outcome, dtype=float)
    method = config.CALIBRATION_METHOD
    if method == "platt":
        lr = LogisticRegression(C=config.PLATT_C, max_iter=config.PLATT_MAX_ITER)
        lr.fit(_logit_feature(train_prob), train_outcome)
        return lambda p: lr.predict_proba(_logit_feature(p))[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(train_prob, train_outcome)
        return lambda p: iso.predict(np.asarray(p, dtype=float))
    raise ValueError(f"unknown config.CALIBRATION_METHOD {method!r}; expected 'platt' or 'isotonic'")


def walk_forward_calibrate(df, prob_col: str, outcome: np.ndarray) -> np.ndarray:
    """Recalibrate a raw predicted probability season-by-season: for each
    season, fit a calibration curve (config.CALIBRATION_METHOD -- Platt
    logistic scaling by default, isotonic previously; see fit_calibrator)
    mapping raw probability -> empirical outcome rate using only STRICTLY
    EARLIER seasons' (prediction, outcome) pairs, then apply it to this
    season's predictions. Same walk-forward discipline as the margin models
    themselves -- a season's own outcomes never influence its own
    calibration. Seasons with fewer than config.MIN_CALIBRATION_GAMES prior
    graded games (i.e. the first predictable season) are passed through raw,
    unchanged.

    `df` must have a `season` column aligned with `prob_col` and `outcome`.
    """
    prob = df[prob_col].to_numpy(dtype=float)
    season = df["season"].to_numpy()
    outcome = np.asarray(outcome, dtype=float)
    out = prob.copy()

    for s in np.unique(season):
        train_mask = season < s
        test_mask = season == s
        if train_mask.sum() < config.MIN_CALIBRATION_GAMES:
            continue
        calibrate = fit_calibrator(prob[train_mask], outcome[train_mask])
        out[test_mask] = calibrate(prob[test_mask])

    return out


def kelly_fraction(prob: np.ndarray, american_odds: np.ndarray) -> np.ndarray:
    """Kelly criterion stake, as a fraction of bankroll, for a bet at model
    probability `prob` and the given American odds: f* = p - (1-p)/b, where
    b is the profit-per-unit-staked if the bet wins. Clipped at 0 -- a
    negative Kelly fraction means the model doesn't see an edge at this
    price, so it's "don't bet", not "bet against yourself". This is *full*
    Kelly; callers should scale it down (fractional Kelly) to account for
    the fact `prob` is an estimate, not a known truth -- betting full Kelly
    on a probability that's off by even a little is known to be far too
    aggressive.
    """
    b = american_payout_multiplier(american_odds)
    prob = np.asarray(prob, dtype=float)
    f = prob - (1.0 - prob) / b
    return np.clip(f, 0.0, None)
