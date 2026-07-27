"""Central configuration: paths, season scope, and modeling constants."""

from pathlib import Path

# --- Paths -----------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
NFLVERSE_RAW_DIR = RAW_DIR / "nflverse"
BETTING_RAW_DIR = RAW_DIR / "betting_lines"
PROCESSED_DIR = DATA_DIR / "processed"
DATA_OUTPUTS_DIR = DATA_DIR / "outputs"

OUTPUTS_DIR = ROOT_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
MODEL_RESULTS_DIR = OUTPUTS_DIR / "model_results"
BACKTESTS_DIR = OUTPUTS_DIR / "backtests"

for d in [
    NFLVERSE_RAW_DIR,
    BETTING_RAW_DIR,
    PROCESSED_DIR,
    DATA_OUTPUTS_DIR,
    CHARTS_DIR,
    MODEL_RESULTS_DIR,
    BACKTESTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# --- Season scope ------------------------------------------------------------

FIRST_SEASON = 2010
LAST_SEASON = 2025  # latest fully completed season as of project start

# nflverse game_type values to keep. Preseason ("PRE") is always excluded.
GAME_TYPES_KEPT = ["REG", "WC", "DIV", "CON", "SB"]

# --- Rolling feature windows -------------------------------------------------

ROLLING_WINDOWS = [4, 8, 17]  # games, per team, recency-weighted rolling stats
EWMA_HALFLIFE_GAMES = 6

# --- Power ratings ------------------------------------------------------------

ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_HOME_FIELD_ADVANTAGE = 55.0  # elo points
ELO_MOV_MULTIPLIER = True  # scale K by margin-of-victory
SEASON_REGRESSION_TO_MEAN = 0.33  # fraction of rating pulled back to 1500 each offseason

# --- Modeling -------------------------------------------------------------

# Walk-forward validation: train on all seasons strictly before the held-out
# season, never leak future games into training or feature computation.
MIN_TRAIN_SEASONS = 3  # need at least this many prior seasons before predicting

MARGIN_MODEL_FEATURES = [
    "elo_diff",
    "off_epa_play_ewm_diff",
    "def_epa_allowed_play_ewm_diff",
    "off_success_rate_ewm_diff",
    "off_pass_epa_ewm_diff",
    "off_rush_epa_ewm_diff",
    "margin_ewm_diff",
    "rest_diff",
    "div_game",
]

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "num_leaves": 15,
    "learning_rate": 0.03,
    "n_estimators": 400,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": -1,
}

# --- Probabilities & odds ------------------------------------------------------

# Std dev of actual margin vs predicted margin, used to convert a predicted
# spread into a win probability via a normal CDF. Recomputed from data at
# training time; this is only a fallback default.
DEFAULT_MARGIN_STD = 13.5

# --- Backtesting ------------------------------------------------------------

ATS_EDGE_THRESHOLD_POINTS = 1.5  # min |model_spread - market_spread| to place an ATS bet
ML_EDGE_THRESHOLD_PROB = 0.03  # min |model_prob - no_vig_market_prob| to place an ML bet
FLAT_STAKE_UNITS = 1.0
DEFAULT_SPREAD_ODDS = -110  # American odds fallback when spread_odds is missing

EDGE_BUCKETS = [0, 1.5, 3, 5, 8, 100]
SPREAD_BUCKETS = [0, 3, 7, 10, 14, 100]  # absolute market spread buckets
