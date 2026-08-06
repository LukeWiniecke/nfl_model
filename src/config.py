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

# Append-only history of every upcoming_picks snapshot ever generated (see
# live_picks.append_to_picks_log) -- unlike upcoming_picks.parquet, which is
# overwritten every run, this is the actual week-by-week track record.
PICKS_LOG_PATH = MODEL_RESULTS_DIR / "picks_log.parquet"

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
LAST_SEASON = 2026  # current/upcoming season -- bump this each year so its
# schedule (and, as they get posted, its market lines) gets pulled in. Games
# without a final score yet (home_margin null) are automatically excluded
# from every backtest/walk-forward step (they filter on home_margin.is_not_null()
# before doing anything), so including the in-progress season here is safe --
# it just makes its games available to live_picks.predict_upcoming_* once
# nflverse has posted lines for them.

# nflverse game_type values to keep. Preseason ("PRE") is always excluded.
GAME_TYPES_KEPT = ["REG", "WC", "DIV", "CON", "SB"]

# --- Rolling feature windows -------------------------------------------------

ROLLING_WINDOWS = [8]  # games, per team, recency-weighted rolling stats
# Previously [4, 8, 17]: LightGBM's feature-importance ranking came back nearly
# flat across ~20 features because roll4/roll8/roll17 of the same underlying
# stat are highly collinear -- the model was spending splits distinguishing
# near-duplicates instead of finding clean signal. Ridge never used these at
# all (it only reads the EWMA diffs below), so dropping roll4/roll17 only
# affects LightGBM, and only by removing redundancy.
EWMA_HALFLIFE_GAMES = 6

# --- Power ratings ------------------------------------------------------------

ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_HOME_FIELD_ADVANTAGE = 55.0  # elo points
ELO_MOV_MULTIPLIER = True  # scale K by margin-of-victory
SEASON_REGRESSION_TO_MEAN = 0.33  # fraction of rating pulled back to 1500 each offseason

# Tried and reverted: a second, slower-decaying Elo track (K=8, 8%/year
# regression) meant to capture persistent organizational quality that the
# primary Elo's 33%/year regression erases too fast -- diagnosed from a
# residual-bias check that found the primary Elo systematically underrates
# perennially strong franchises (NE, KC, BAL, ...) and overrates chronically
# weak ones (NYJ, WAS, ...) by 1-2 points a game. Measured directly against
# that same bias check: it made it slightly WORSE (mean |bias| across teams
# 0.84 -> 0.93), because the slow and fast Elo tracks turned out to be 0.91
# correlated -- lowering the regression rate doesn't separate "organizational
# persistence" from "current form" cleanly, since real team strength is
# already persistent most of the time. `power_ratings.compute_elo_ratings`
# still accepts arbitrary k/regression_to_mean/prefix if this is worth
# retrying with a genuinely different long-run-quality signal (e.g. a frozen
# prior-season-end rating, or multi-year win% independent of Elo's own
# update rule) -- just don't reintroduce it as a second Elo variant.

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
    "qb_epa_per_att_ewm_diff",
    "qb_continuity_diff",
    "rest_diff",
    "div_game",
    "neutral_site",
    "is_dome",
    "temp_filled",
    "wind_filled",
    "injury_diff",  # kept for ridge only -- see note below; reverted for LightGBM
    "qb_injury_diff",  # kept for ridge only, same as injury_diff -- see note below
]

# Totals (over/under) model, phase 1: a small hand-picked feature set of
# COMBINED (summed, not diffed -- see feature_engineering.TOTAL_SUM_PREFIXES)
# offense/defense form plus weather, on the theory that what predicts total
# points is "how potent are both offenses / how leaky are both defenses",
# not "which team is better" (which is what the margin diffs capture).
# Deliberately independent of MARGIN_MODEL_FEATURES -- no reason a totals
# model should share a feature list with a margin model any more than ridge
# and LightGBM share one with each other.
TOTAL_MODEL_FEATURES = [
    "off_epa_play_ewm_sum",
    "off_success_rate_ewm_sum",
    "off_pass_epa_ewm_sum",
    "off_rush_epa_ewm_sum",
    "def_epa_allowed_play_ewm_sum",
    "def_success_rate_allowed_ewm_sum",
    "qb_epa_per_att_ewm_sum",
    "off_plays_ewm_sum",  # kept for ridge/ensemble only -- see note below; reverted for LightGBM
    "wind_filled",
    "temp_filled",
    "is_dome",
    "neutral_site",
    "div_game",
]
# Phase 1 result, measured directly (main.py's ridge_total/lgbm_total/
# ensemble_total, evaluation.skill_vs_market_totals): worse than the market
# on every metric, for every model. Ridge: MAE 10.68 vs 10.51, RMSE 13.50 vs
# 13.27, Brier 0.255 vs 0.250. LightGBM: MAE 11.06 vs 10.51 (a bigger gap
# than ridge's, same pattern as the margin models). Ensemble: MAE 10.67 vs
# 10.46, essentially identical to ridge since the blend leans on it heavily.
# Flat-stake backtest ROI at a 1.5-pt threshold (untuned): ridge -1.8%, LGBM
# ~0.0%, ensemble -2.1% -- no model shows a real edge. Not wired into
# Upcoming picks or the dashboard yet (phase 2, pending a reason to believe
# this is worth the UI surface area) -- this feature set doesn't yet
# include anything QB- or injury-specific, unlike the margin models, so a
# plausible next step before writing this off is adding those (or a real
# pace/plays-per-game feature) rather than concluding totals has no signal
# from a first pass with a smaller, more generic feature set.
#
# off_plays_ewm_sum (pace/plays-per-game): the FIRST feature since phase 1
# to show a real, clean improvement here. off_plays is a new raw stat
# (feature_engineering.compute_game_epa_splits) -- offensive snap count
# (run/pass plays) per team per game, no new data ingestion needed, purely
# an extra aggregation over pbp data already cached. Flows through the
# existing team-level rolling-form machinery (add_rolling_form's
# raw_stat_cols) exactly like off_epa_play etc. does, so it's automatically
# leakage-safe and automatically forward-fills into live upcoming games the
# same way -- no bespoke carry-forward code needed, unlike every QB-specific
# feature tried this session. Deliberately scoped to the totals model only
# (modeling.gbm_feature_columns explicitly excludes the _diff version from
# LightGBM's MARGIN feature set, since it would otherwise auto-match that
# function's suffix-based detection regardless of this file) -- pace
# plausibly explains combined scoring opportunity, not which team is
# better, so there's no real hypothesis for it to help the margin model.
# Measured directly: ridge_total's MAE/RMSE/Brier ALL improved (10.678/
# 13.494/0.2551 -> 10.673/13.483/0.2549), ROI flat (-1.46%->-1.45%).
# Ensemble_total also improved on all three (10.670/13.468/0.2560 ->
# 10.663/13.456/0.2557), ROI improved (-2.12%->-1.96%). LightGBM_total got
# worse on every metric, so it's excluded there (modeling.
# gbm_total_feature_columns) -- confirmed byte-identical to its pre-pace
# baseline once excluded. Margin models (ridge/LightGBM/ensemble) are
# untouched by this change entirely, confirmed byte-identical to their own
# baseline -- the scoping actually held.

# injury_diff (severity-weighted sum of each team's official pre-game
# injury-report players, home minus away) -- kept for ridge, reverted for
# LightGBM (modeling.gbm_feature_columns excludes it). Mixed result measured
# directly: ridge's aggregate MAE/RMSE/Brier barely moved, but the
# divisional-game gap it was informally expected to help (games between
# familiar rivals where injury news is well-scouted) closed further, from
# -0.27/-0.23 (div/non-div, post-QB-signal) to -0.244/-0.237 -- essentially
# gone. LightGBM got consistently slightly worse, same pattern as the
# prime-time/surface features, so it's excluded there.
#
# Upgraded, kept: each player's severity weight (Out=1.0/Doubtful=0.5/
# Questionable=0.25) is now additionally scaled by their own prior rolling
# snap share (feature_engineering.compute_player_snap_share, from
# nflreadpy's load_snap_counts, joined to injuries via a gsis_id<->pfr_id
# crosswalk) -- a starter being out costs more than a rarely-used backup at
# the same report status, rather than every player at a severity level
# counting the same regardless of role size. QBs are excluded from this
# feature entirely (compute_qb_form/qb_continuity already track the starter
# directly from real performance data, so including them here too would
# double-count the same signal). Measured directly: ridge's MAE/RMSE/Brier
# all improved (10.157->10.151 / 13.086->13.080 / 0.2172->0.2171) and
# ATS ROI went from -0.6% to +1.0% (n_bets ~unchanged, 2142->2141); ML ROI
# was flat. Ensemble/ML's ROI dropped (+0.3%->-2.8%) but that's within its
# own bootstrap 95% CI [-7.5%, +2.2%] -- not distinguishable from noise, and
# ensemble/ATS improved slightly too. Along the way, found and fixed a real
# pre-existing bug: nflverse's injuries table uses pre-relocation team codes
# (OAK/SD/STL), which never matched `games`' standardized codes (LV/LAC/LA)
# -- the injury feature had silently been null for every Raiders/Chargers/
# Rams game before their moves (data_ingestion.load_injuries now
# standardizes team on load). Snap-count data only goes back to 2012;
# earlier seasons (and any player/game the gsis_id<->pfr_id crosswalk
# misses) fall back to a fixed default importance (config.DEFAULT_SNAP_SHARE)
# rather than dropping the player.
#
# qb_injury_diff -- kept for ridge, excluded for LightGBM (same convention as
# injury_diff above). Same severity x snap-share weighting as injury_diff,
# but for the QB position specifically, which injury_diff deliberately
# excludes (feature_engineering.compute_qb_injury_features). Built alongside
# a real bug fix in feature_engineering.compute_qb_form: it only ever had
# rows for PLAYED games (built from real pass-attempt data), so every
# upcoming game had null QB features -> median-imputed at prediction time --
# live predictions had ZERO real QB signal, invisible in the walk-forward
# backtest since every historical training row has already been played by
# construction. Fixed by carrying each team's most recent played game's
# starter forward (their own post-game rolling EPA/attempt, via join_asof on
# the team's schedule timeline) into any upcoming game, assuming continuity
# by default -- qb_injury_diff is the one live-available signal that can
# flag when that default assumption is wrong (the presumed starter is
# actually Out/Doubtful/Questionable this week), since a new starter's own
# rolling stats only show up in qb_form once they've actually played.
# Also found and fixed a second bug: naively joining left qb_injury_diff
# null whenever EITHER team had no QB flagged that week -- the common case,
# since most weeks nobody's QB is hurt -- which collapsed 64/4363 played
# games with a usable value down to almost nothing. Filling the missing
# side with 0.0 (no report entry = healthy, not unknown) before subtracting
# restored the far more common one-team-only case: 1024/4363 games now
# carry a nonzero value. Measured directly, before -> after: ridge MAE/RMSE/
# Brier 10.151/13.080/0.2171 -> 10.142/13.078/0.2169, ATS ROI +1.0%->+1.4%,
# ML ROI -3.4%->-3.2%. LightGBM 10.599/13.557/0.2357 -> 10.598/13.548/0.2359,
# ATS -1.9%->-0.9%, ML -2.4%->-3.4% -- LightGBM never sees qb_injury_diff at
# all, so its whole change here is from the carry-forward fix giving more
# complete historical training data, not the new feature. Ensemble 10.131/
# 13.073/0.2172 -> 10.117/13.063/0.2173, ATS +0.6%->+1.5%, ML -2.8%->-3.0%.
# ATS improved for all three models; ML is mixed and, given LightGBM's ML
# market's known sensitivity to small input changes (see the isotonic-
# calibration note below) and this project's own measured bootstrap-noise
# range at similar sample sizes, reads as net-positive rather than a clean
# win everywhere.
#
# Tried and reverted: ref_points_ewm, a leakage-safe rolling EWM of total
# points in games each REFEREE (position == "Referee") has officiated
# (feature_engineering.compute_referee_features, data_ingestion.
# load_officials). Motivated by real, documented crew-level tendencies in
# penalty calls/game flow -- untested and orthogonal to everything else in
# the feature set, so worth a real look, especially for the totals model,
# which had no edge at all. Two real bugs found and fixed along the way:
# (1) officials' own game_id uses nflverse's OLD numeric gsis format, not
# the season_week_away_home format used everywhere else here -- zero ID
# overlap, silently producing 100% nulls until joined through games.
# old_game_id (nflverse's own crosswalk, now kept by cleaning.clean_schedules
# instead of being dropped). (2) officials only covers 2015-2025 (a hard
# floor AND ceiling, unlike most loaders here), so the earliest walk-forward
# folds (trained only on pre-2015 seasons) had a training pool with zero
# non-null values, making even the MEDIAN null -- crashed Ridge on a NaN
# input until modeling._impute was made to fall back to 0.0 when a column's
# median itself is null. Once actually working (confirmed: null for every
# game before 2015, ~70% non-null 2015 on, 100% null for every LIVE upcoming
# game since referee assignments aren't public that far ahead and
# nflverse's officials table -- like player_stats -- is compiled from
# completed games only), measured directly: worse on every MAE/RMSE/Brier
# metric, for every TOTALS model (the actual target) -- ridge 10.678/13.494/
# 0.2551 -> 10.687/13.498/0.2553, LightGBM 11.063/13.969/0.2794 ->
# 11.088/13.981/0.2804, ensemble 10.671/13.468/0.2560 -> 10.680/13.473/
# 0.2562. Totals ROI mixed (ridge -1.5%->-1.4%, ensemble -2.1%->-1.5%, but
# LightGBM -1.0%->-2.2%). For the MARGIN models, MAE/RMSE/Brier moved less
# than 0.02 in either direction, but ATS ROI -- the metric least distorted
# by calibration noise -- got WORSE for all three (ridge +1.4%->+1.0%,
# LightGBM -0.9%->-1.5%, ensemble +1.5%->+0.8%), a consistent enough
# direction to read as real rather than scatter. Likely reason: this is a
# RAW rolling average of realized total points per referee, which doesn't
# control for which teams that ref happened to draw -- a ref who's
# randomly been assigned several high-powered offenses looks like a
# "high-scoring ref" when it's team quality leaking through, not a genuine
# crew tendency. A fairer test would use each game's RESIDUAL against its
# own expected total (from the other features), not the raw score, before
# concluding referees have no effect. The ingestion/feature machinery
# (data_ingestion.load_officials, feature_engineering.
# compute_referee_features) is left in place, unused by any feature list,
# for that retry.
#
# Retried exactly that: ref_total_resid_ewm, rolling each referee's EWM
# against (total_points - total_line) instead of raw total_points --
# properly strips out team-quality confound (confirmed: mean ~0.4, std ~4.0,
# vs. raw points' mean ~46/std ~4.2). Conceptually the right fix, but
# empirically still no win: totals MAE/RMSE/Brier moved worse again for all
# three totals models (though smaller for ridge/ensemble than the raw-points
# attempt), and margin ATS ROI got worse for all three again too (ridge
# +1.4%->+0.7%, LightGBM -0.9%->-2.4%, ensemble +1.5%->+1.1%) -- same
# direction as the raw-points version. Read as a real negative result, not
# just an implementation bug: two genuinely different framings (raw vs.
# residualized) both landed in the same place, most likely because each
# individual referee only accumulates a few hundred career games across
# ~15-17 active crews -- not enough games per ref for a rolling stat to
# separate a genuine crew tendency from noise, even with team quality
# properly stripped out. Not retried a third time.
#
# Tried and reverted: qb_cpoe_ewm_diff/_sum, the starting QB's leakage-safe
# rolling completion-% over expectation (CPOE, from nflreadpy's Next Gen
# Stats -- feature_engineering.compute_qb_ngs_form, data_ingestion.
# load_nextgen_stats_passing). CPOE adjusts for throw difficulty via player
# tracking data, a genuinely different KIND of QB signal than the pbp-
# derived EPA/attempt already in use, not just a refinement of it -- worth
# a real look. NGS only covers 2016 on (a hard floor+ceiling, same shape as
# officials' 2015 floor); unlike officials, though, the existing QB carry-
# forward mechanism (compute_qb_form's join_asof) means most LIVE upcoming
# games actually get a real, non-null value (confirmed: 43/51 real upcoming
# games), not the ~0% referee saw.
#
# Measured directly, aggregate (2013-2025): ridge MAE/RMSE/Brier all
# slightly worse (10.142/13.076/0.2169 -> 10.146/13.089/0.2171), margin ATS
# ROI worse for ridge (+1.4%->+1.0%) and LightGBM (-0.9%->-1.9%), though
# ensemble improved slightly (+1.5%->+1.7%). Totals MAE/RMSE/Brier also
# slightly worse for ridge/ensemble; LightGBM's totals RMSE was the one
# genuine improvement (13.969->13.948).
#
# Given the feature is null before 2016, ran a controlled by-era A/B
# (temporarily disabling the join, re-running, comparing season-by-season)
# to check whether the aggregate number was hiding a real win concentrated
# in the years the data actually covers. Confirmed pre-2016 deltas are
# EXACTLY zero (good leakage sanity check), but even isolated to 2016+ --
# where CPOE has real, non-imputed values -- MAE was still slightly worse,
# not better (ridge margin: 10.024 vs 10.018 without it; ridge totals:
# 10.652 vs 10.631 without it). This rules out "diluted by uninformative
# early years" as the explanation -- it's genuinely not helpful within its
# own valid coverage window, not just averaged down by years it can't see.
# Disabled by commenting out the qb_ngs_form join itself in main.py (not
# just dropping it from the ridge/LightGBM lists like the referee/prime-
# time reverts): qb_cpoe_ewm ends in "_ewm", so leaving it joined would
# auto-generate qb_cpoe_ewm_diff/_sum via build_modeling_table's generic
# form_cols detection, and modeling.gbm_feature_columns/
# gbm_total_feature_columns auto-include ANY column matching that suffix --
# LightGBM would keep using it regardless of what config.py says.
#
# Built and disabled (not a modeling failure -- an upstream data quality
# one): depth-chart-aware QB carry-forward. compute_qb_form's live/upcoming
# branch (see its own docstring) assumes "whoever started this team's most
# recent game is still starting" -- true most weeks, wrong exactly when a
# trade/retirement/rookie promotion happened since. data_ingestion.
# load_current_qb_depth_chart pulls nflverse's own depth chart for the
# presumed CURRENT starter per team, and compute_qb_form's
# depth_chart_starters param overrides the naive carry with that player's
# own rolling stats when the two disagree (never for played/backtest rows,
# confirmed byte-identical across every model with this wired in).
#
# Two real bugs found and fixed getting the mechanism itself working: (1)
# nflreadpy's load_depth_charts returns TWO different schemas depending on
# the season -- archived seasons are a per-(season, week) snapshot, the
# CURRENT season instead comes from a live feed keyed by a "dt" timestamp
# with no week at all; load_current_qb_depth_chart branches on which
# schema came back. (2) team_schedule's "unplayed" bucket (used by the
# carry-forward generally, not just the depth-chart override) turned out
# to ALSO include PLAYED historical games missing a QB pass-attempt row in
# player_stats (a real, if rare, data gap -- old pre-relocation Raiders
# games) -- initially "fixed" by restricting to genuinely-unplayed rows
# (margin.is_null()), which measurably WORSENED aggregate MAE, revealing
# that the naive carry-forward was already a net-positive fallback for
# those gap games too. Corrected by keeping the original unrestricted
# carry-forward scope, and gating ONLY the depth-chart override itself on
# margin.is_null() -- confirmed via a full pipeline run that this restores
# byte-identical backtest numbers across every model.
#
# With the mechanism verified correct, found nflverse's LIVE depth-chart
# feed itself is NOT reliable enough to trust for this: spot-checked
# against 32 teams' own most-recent-actual-starter and found real,
# confirmed team-to-player misalignment -- Tua Tagovailoa (actually
# Miami's QB) listed as Atlanta's, Kyler Murray (Arizona's) listed as
# Minnesota's, Miami's own entry showing Malik Willis instead, Arizona's
# showing Jacoby Brissett instead, Kirk Cousins appearing to duplicate
# across two different teams' entries. Confirmed directly against
# nflreadpy's raw output (not a join bug on this end) -- an upstream data
# problem in nflverse's live source for the current season specifically.
# Disabled in main.py by passing None to compute_qb_form (the override
# param, not the whole join -- there's no LightGBM auto-suffix risk here
# unlike qb_cpoe_ewm, since depth_chart_starters only ever gates internal
# logic, it doesn't create a new modeling-table column). Worth revisiting
# if nflverse's live feed improves, or with a different depth-chart source.
# Tried and reverted: prime_time_approx and surface_grass, pulled in from the
# Spreadspoke reference CSV (data_ingestion.load_spreadspoke_context) since
# nflverse doesn't have them. Motivated by a real finding (the market's edge
# grows in high-profile/late-season games, plausibly sharper money on
# marquee games) and joined cleanly (99.1% match rate). Measured directly:
# ridge was flat (mae 10.161 -> 10.163, noise), LightGBM got consistently
# slightly worse across MAE/RMSE/Brier. The ingestion + join machinery is
# still wired up (main.py computes spreadspoke_ctx and build_modeling_table
# accepts it) in case a different framing of "high-profile game" is worth
# testing later -- just not these two columns as direct features.
# Tried and reverted: opponent-adjusted versions of the 6 EPA/success-rate
# features above (subtract the specific opponent's incoming rolling
# defensive/offensive average before rolling up "form"), meant to fix the
# same team-level bias the reverted stable-Elo attempt targeted. Measured
# directly against it: barely moved (mean |bias| across teams 0.836 -> 0.831)
# and didn't touch the divisional-game gap either (-0.22/-0.51 vs market,
# same as before), while making LightGBM's aggregate MAE slightly worse.
# Likely reasons: NFL schedules aren't lopsided enough within an 8-game
# rolling window for this to extract much signal, and elo_diff already
# opponent-adjusts through its own update rule, leaving little marginal
# value. `feature_engineering.add_opponent_adjusted_form` still exists if
# this is worth retrying with e.g. season-cumulative opponent quality instead
# of rolling, or as a direct "this week's opponent quality" feature rather
# than folded into a team's own rolling form.

# Tried and reverted: hyperparameter tuning for Ridge's alpha and
# LightGBM's num_leaves/learning_rate/n_estimators, which had never
# actually been tuned via cross-validation before -- inherited defaults
# from when the model was first built, never revisited. Checked directly
# with a real walk-forward grid search (train on strictly-prior seasons,
# predict the next one, exactly as production already does -- not a
# separate CV split; one-off measurement script, not kept in the repo).
#
# By raw MAE alone, tuning looked like a clear win. Ridge (margin): swept
# [0.5..100], smooth unimodal minimum at alpha=20 (10.136 vs. the default
# alpha=5's 10.142). Ridge (totals): swept [5..50] separately -- opposite
# direction, MAE got monotonically WORSE as alpha increased, so the
# original alpha=5.0 was already better there and was left alone (the two
# targets needed different answers, which is itself a reason they're
# tuned as separate constants below rather than one shared "ridge alpha").
# LightGBM: an initial grid (num_leaves in [7,15,31], learning_rate in
# [0.02,0.03,0.05], n_estimators in [300,400,600]) found the OLD defaults
# (15/0.03/400, mae=10.598) beaten by every simpler setting tried, a
# textbook overfitting signature; a follow-up grid toward smaller/simpler
# settings found a genuine interior minimum at num_leaves=3, learning_rate
# =0.01, n_estimators=300 (mae=10.219) -- nearly HALF the old gap to
# ridge's own 10.14 MAE, and confirmed to transfer to the totals target
# too (11.063 -> 10.759 with the same values).
#
# But MAE is the wrong objective for this project, and the real backtest
# proved it emphatically: with the MAE-tuned LightGBM deployed, its ML win
# rate on the games it chose to bet COLLAPSED to 36.6% (ROI -3.4% ->
# -7.5%), its ATS ROI went -0.9% -> -3.2%, and the ensemble -- which
# blends ridge and LightGBM -- flipped from a POSITIVE +1.5% ATS ROI to
# -0.2%. Isolating ridge's alpha=20 alone (LightGBM reverted) showed the
# same pattern in miniature: ATS ROI got WORSE for both ridge (+1.4% ->
# +1.1%) and ensemble (+1.5% -> +0.8%) despite ridge's own tiny MAE gain.
# A lower-error point prediction, in aggregate, is not the same thing as
# a model that identifies genuine market mispricing -- optimizing
# hyperparameters against MAE alone, with no eye on the actual downstream
# decision (does this edge beat the market's own price), measurably threw
# away the very thing that made these hand-picked defaults useful. Both
# reverted back to their original values.
RIDGE_ALPHA_MARGIN = 5.0
RIDGE_ALPHA_TOTAL = 5.0

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
DEFAULT_TOTAL_STD = 10.0  # same idea, for the totals model's predicted-total distribution

# Per-game conditional uncertainty: a small Ridge on |residual| ~ elo
# mismatch / divisional / playoff / weather / rest mismatch (see
# modeling._std_feature_frame), refit walk-forward each season, replaces the
# single global DEFAULT_MARGIN_STD with a game-specific std. Floor/cap keep a
# badly-fit fold from producing a near-zero (probabilities collapse to 0/1)
# or wildly oversized std.
MARGIN_STD_FLOOR = 8.0
MARGIN_STD_CAP = 22.0

# Walk-forward isotonic recalibration of model probabilities (see
# probabilities.walk_forward_calibrate): fit only on strictly earlier
# seasons' (predicted prob, outcome) pairs. Below this many prior graded
# games, skip calibration and pass the raw probability through -- not enough
# data yet to trust a calibration curve.
MIN_CALIBRATION_GAMES = 150

# Walk-forward Student's-t degrees-of-freedom fit (see
# probabilities.fit_t_dof_walk_forward), an alternative to the Normal CDF
# for converting a predicted margin/total into a probability -- tests
# whether the market's own tail behavior is fatter than Normal implies.
# DEFAULT_T_DOF is the fallback before MIN_CALIBRATION_GAMES prior games
# exist: high enough (30) to be practically indistinguishable from Normal,
# since there's no data yet to justify assuming fatter tails. FLOOR keeps a
# noisy small-sample MLE fit away from near-2 (undefined/explosive
# variance); CAP (100) stops it from fitting an arbitrarily large value
# once it's already functionally Normal -- nothing to gain either way.
DEFAULT_T_DOF = 30.0
T_DOF_FLOOR = 3.0
T_DOF_CAP = 100.0

# Kept -- Student's-t replaces the Normal CDF everywhere a probability is
# computed from a predicted margin/total (probabilities.ats_cover_probability_t/
# home_win_probability_t/over_probability_t, used in backtesting.py,
# live_picks.py, and evaluation.skill_vs_market[_totals]). Motivated by a
# real, measured finding: evaluation.calibration_bins on the RAW (pre-
# isotonic) probabilities showed the model's predictions are consistently
# MORE EXTREME than actual outcome rates in both directions (overconfidence,
# not the underconfidence/compression-toward-50% that was the original
# hypothesis) -- e.g. LightGBM's raw ML: a bin predicting 14.8% win
# probability actually won 31.7% of the time; a bin predicting 94.7% only
# won 78.2%. Worst for LightGBM specifically (mean |bin gap| ~0.088 vs.
# ridge/ensemble's ~0.02-0.035) -- exactly the one market left deliberately
# uncalibrated (isotonic reverted there, see the moneyline-recalibration
# note below) because isotonic was already known to distort it. The
# walk-forward-fit degrees-of-freedom lands near-Normal for ridge/ensemble
# (~21, confirming their raw probabilities were already reasonably shaped)
# and hits the floor for LightGBM (dof=3.0, genuinely fat-tailed residuals,
# not just a noisy fit) -- the fit itself independently confirms which
# markets needed this.
#
# Measured directly, before (Normal) -> after (Student's-t), same isotonic-
# on/off pattern as before for each market:
# Brier -- margin: ridge 0.21690->0.21688 (flat), LightGBM 0.23586->0.23177
# (real improvement), ensemble 0.21735->0.21740 (flat/noise). Totals: ridge
# 0.25512->0.25477, LightGBM 0.27937->0.27417 (real improvement again),
# ensemble 0.25597->0.25566.
# ML ROI: ridge -3.22%->-2.63%, LightGBM -3.35%->-3.40% (flat), ensemble
# -2.96%->-2.44%. ATS/totals ROI unchanged everywhere, exactly as expected
# (bet SIDE there comes from the points edge, not this probability -- only
# Kelly stake sizing and the reliability-curve display depend on it).
# No regressions found on any of the six models -- kept everywhere rather
# than the usual per-model split, since for once there was no tradeoff to
# split on.

# --- Backtesting ------------------------------------------------------------

ATS_EDGE_THRESHOLD_POINTS = 1.5  # min |model_spread - market_spread| to place an ATS bet
ML_EDGE_THRESHOLD_PROB = 0.03  # min |model_prob - no_vig_market_prob| to place an ML bet
TOTAL_EDGE_THRESHOLD_POINTS = 1.5  # min |model_total - market_total| to place a totals bet -- phase-1 starting
# point, same magnitude as ATS_EDGE_THRESHOLD_POINTS; untuned, not yet checked with the threshold optimizer.
FLAT_STAKE_UNITS = 1.0
DEFAULT_SPREAD_ODDS = -110  # American odds fallback when spread_odds is missing

# Fallback "how big a role does this player play" weight for the injury
# feature when a player has no prior snap-share data (no player-crosswalk
# match, or the game is before 2012, when snap-count data starts) -- a
# modest, deliberately unremarkable default rather than 0 (ignore them) or 1
# (treat as a full-time starter).
DEFAULT_SNAP_SHARE = 0.3

EDGE_BUCKETS = [0, 1.5, 3, 5, 8, 100]
SPREAD_BUCKETS = [0, 3, 7, 10, 14, 100]  # absolute market spread buckets
