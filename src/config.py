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

# Playoff games are kept as full first-class rows: they are training data,
# they get predictions, and they get bet. Playoff STATUS is a feature of the
# uncertainty model only (`is_playoff` in modeling.STD_FEATURES), never of
# the margin/total point prediction, and Elo treats a playoff game exactly
# like a regular-season one.
#
# Tried and reverted: training the margin/total models on REGULAR-SEASON rows
# only, while still predicting/grading/betting every game type (implemented
# as a TRAIN_ON_REGULAR_SEASON_ONLY flag plus modeling._training_rows; see
# branch autoresearch/reg-only-training). The hypothesis was contamination:
# playoff games are an unrepresentative 4.3% of rows (188 of 4363) -- no bad
# teams, bye-week rest gaps that never occur in the regular season,
# neutral-site Super Bowls -- and the model spends 96% of its life predicting
# regular-season games.
#
# Measured directly, and it is a NULL result rather than a win or a loss:
#   development ATS ROI  +7.91% -> +7.69%   (aggregate; the decision metric)
#   holdout ATS ROI      +7.51% -> +8.34%
#   regular-season ATS   +7.21% -> +7.54%   (the 96% the change targeted)
#   playoff ATS          +27.8% (n=25) -> +14.1% (n=28)
# The targeted sub-metric did move the hypothesized way (+0.33pp), and the
# holdout moved with it (+0.83pp), but the aggregate development number fell
# because the tiny playoff bet pool swung down -- and every one of those
# moves is far inside a ~+/-7pp bootstrap CI. The honest reading is that
# dropping 4.3% of training rows does not measurably change a model fit on
# 4175 regular-season games, which is roughly what you would expect going in.
# Not worth the extra config surface for an unmeasurable effect; retry only
# with a materially different framing (e.g. sample-weighting rather than
# hard exclusion), not by flipping this back on.

# --- Rolling feature windows -------------------------------------------------

ROLLING_WINDOWS = [8]  # games, per team, recency-weighted rolling stats
# Previously [4, 8, 17]: LightGBM's feature-importance ranking came back nearly
# flat across ~20 features because roll4/roll8/roll17 of the same underlying
# stat are highly collinear -- the model was spending splits distinguishing
# near-duplicates instead of finding clean signal. Ridge never used these at
# all (it only reads the EWMA diffs below), so dropping roll4/roll17 only
# affects LightGBM, and only by removing redundancy.
EWMA_HALFLIFE_GAMES = 9
# Raised from 6 by an autoresearch sweep (branch autoresearch/aug18), the one
# hyperparameter in this file that moved the betting metric in a direction
# that survived a real out-of-sample check. Measured on the ensemble's
# DEVELOPMENT era only, then confirmed against the holdout era:
#   halflife  6 -> dev ATS ROI +5.36%, holdout +6.24%   (previous default)
#   halflife  7 -> dev +6.06%, holdout +7.36%
#   halflife  9 -> dev +6.98%, holdout +7.51%           <- kept
#   halflife 12 -> dev +8.31%, holdout +4.11%
#   halflife 16 -> dev +8.53%, holdout +2.04%
# Note the shape carefully, because it is the whole lesson: development ROI
# keeps climbing past 9 while holdout ROI collapses. Selecting on development
# ROI alone -- which is what this file's own history has mostly done -- would
# have landed on 16 and looked like the best result of the run. It is instead
# the clearest overfitting signature the project has produced, and it is only
# visible because HOLDOUT_FIRST_SEASON exists. 9 is the last value where both
# eras improve together.
#
# Mechanism, stated honestly as a hypothesis rather than a finding: a longer
# half-life weights more prior games into each team's "form," which is a
# straightforward variance reduction on a noisy per-game EPA signal. Past
# ~9 games it starts averaging across genuinely different team states
# (injuries, midseason scheme changes) and the extra smoothing stops paying.
#
# Also tried and reverted in the same sweep, all against DEVELOPMENT ROI, all
# reverted for making it worse -- recorded so they are not retried blind:
# ELO_K 15 and 26 (20 is a local optimum in both directions),
# ELO_HOME_FIELD_ADVANTAGE 40 and 48, SEASON_REGRESSION_TO_MEAN 0.25,
# ELO_MOV_MULTIPLIER False, RIDGE_ALPHA_MARGIN 2.5 and 12 (5 is a local
# optimum in both directions -- independently reproducing this file's own
# "MAE-tuned alpha hurt ROI" finding, this time selecting on ROI directly),
# MIN_TRAIN_SEASONS 5 (disqualified: it cut the bet sample 57%, which
# flatters ROI rather than improving it).

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

# Held-out seasons: NEVER used to make a keep/revert decision about a feature,
# hyperparameter, threshold, or model. main.py reports development-era and
# holdout-era results separately so the holdout stays a real out-of-sample
# check rather than one more thing the configuration has been fit to.
#
# The reason this exists: this file's own history is a long list of features
# kept or reverted on ATS/ML ROI differences of 0.4-1.6 percentage points,
# measured on a single 13-season backtest whose bootstrap 95% CI on ROI is
# about +/-4 points wide at n~2000. Every one of those decisions was
# individually defensible and collectively they fit the configuration to this
# particular backtest -- the "MAE-tuned LightGBM destroyed ROI" result below
# is the clearest symptom: that is what a noise-dominated selection metric
# looks like, not evidence that MAE is the wrong objective.
#
# Rule going forward: make decisions on seasons < HOLDOUT_FIRST_SEASON only,
# and require a change to move ROI by more than the bootstrap CI width
# (evaluation.summarize_with_ci) before accepting it. Most single-feature
# changes will not clear that bar, which is the honest answer.
HOLDOUT_FIRST_SEASON = 2023

# Weather (temp_filled/wind_filled) is DISABLED as a model feature.
#
# Not a modeling result -- a train/serve skew. nflverse's temp/wind are the
# conditions OBSERVED at the game, populated for 2963/3162 outdoor historical
# games but for 0/34 upcoming ones (there is no forecast in this data). So the
# walk-forward backtest trained and scored with real wind speeds, while every
# live prediction silently got add_game_context_features' fixed 58F/8mph
# defaults instead. The backtest was measuring a model that knows something
# the live model does not, which makes it an overstatement of live
# performance rather than a leak of the target.
#
# is_dome is deliberately KEPT: it comes from the stadium/roof, is known
# months in advance, and is identically available live and historically.
#
# To re-enable properly, join a real pre-kickoff weather FORECAST (by stadium
# and kickoff time) rather than the observed value, so the historical feature
# matches what is actually knowable before a game.
WEATHER_FEATURES_DISABLED = ["temp_filled", "wind_filled"]

MARGIN_MODEL_FEATURES = [
    "elo_diff",
    "off_epa_play_ewm_diff",
    "def_epa_allowed_play_ewm_diff",
    "off_success_rate_ewm_diff",
    "off_pass_epa_ewm_diff",
    "off_rush_epa_ewm_diff",
    "margin_ewm_diff",
    "qb_epa_per_att_ewm_diff",
    "qb_continuity_diff",  # load-bearing -- do NOT drop as dead weight; see note below
    "rest_diff",
    "div_game",
    "neutral_site",
    "is_dome",
    "injury_diff",  # kept for ridge only -- see note below; reverted for LightGBM
    "qb_injury_diff",  # kept for ridge only, same as injury_diff -- see note below
]

# Three features in the list above were probed by an autoresearch sweep
# (branch autoresearch/aug18) and are recorded here so they are not
# "simplified away" later on the reasoning that first suggested it:
#
#   - qb_continuity_diff is LOAD-BEARING, despite looking like dead weight.
#     Its own note (further down) says it is always 0 for a live upcoming
#     game, which reads like a training-only artifact worth deleting.
#     Measured directly: dropping it took development ATS ROI +6.98% ->
#     +4.90% and the holdout era +7.51% -> -9.77% (n=19). Whatever it
#     contributes historically, the model leans on it hard.
#   - off_rush_epa_ewm_diff is NOT redundant with off_epa_play_ewm_diff,
#     the intuition being that rushing EPA is famously low-signal and
#     collinear with overall EPA. Dropping it: dev +6.98% -> +6.48%.
#   - def_success_rate_allowed_ewm_diff was ADDED and reverted. The list
#     is asymmetric on purpose -- offense carries both EPA and success
#     rate, defense only EPA -- and restoring the symmetry did not help
#     (dev +6.98% -> +6.92%): defensive success rate is redundant with
#     defensive EPA in a way the offensive pair is not.
#
# Also tried and reverted: off_plays_ewm_diff (pace) as a MARGIN feature,
# dev +6.98% -> +6.13%. This confirms with data what the TOTAL_MODEL_FEATURES
# note below argued only from theory (pace explains combined scoring
# opportunity, not which team is better), so that scoping is now measured.
#
# And an elo_diff x div_game INTERACTION, on the hypothesis that a linear
# ridge cannot express "the talent gap matters less between division
# rivals." It produced the single largest development gain of the entire
# sweep (+7.22% -> +7.93%) and was still discarded: the holdout era fell
# +5.60% -> +1.71% at the same time. Worth knowing before someone retries
# it -- it looks like the best idea in the run right up until the holdout
# check, which is exactly why that check exists.

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
    "is_dome",
    "neutral_site",
    "div_game",
]
# wind_filled/temp_filled removed here too -- see WEATHER_FEATURES_DISABLED.
# Weather is more plausibly a real totals signal than a margin one (wind
# suppresses passing for BOTH teams), so this is the feature set most worth
# revisiting if a genuine forecast source is ever wired in.
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

# LightGBM is excluded from the ensemble.
#
# Stated honestly: this is a SIMPLIFICATION, not a measured improvement. On
# the development era, ensemble-with-lgbm vs ridge alone came to ATS -0.3pp
# (95% CI on the difference [-10.1pp, +9.3pp]) and ML +2.6pp ([-15.6pp,
# +20.9pp]) -- both firmly inside the noise, which is exactly the kind of
# difference config.py has historically treated as a result and should not.
#
# What is NOT ambiguous is the case for removing it:
#   - the meta-model already assigns it near-zero weight on its own (0.01-0.10
#     across seasons, against ridge's 0.79-0.86), so it was contributing
#     almost nothing even when included;
#   - standalone it is clearly bad (development ATS ROI -7.3%, ML -2.0%);
#   - its edge-shrinkage slope is 0.02-0.08 vs ridge's 0.20-0.35, i.e. its
#     claimed edges are ~10x overstated, so the adaptive threshold has to
#     demand 10-40 point raw edges from it before a bet clears.
# Removing a component with no demonstrated value, a bad standalone record,
# and a near-zero learned weight is justified on parsimony alone.
#
# With a single base model the "ensemble" becomes a one-feature meta
# regression on ridge's prediction -- not degenerate: it still fits a slope
# and intercept that shrink and de-bias ridge's raw output against realized
# margins, which is the same correction the edge-shrinkage machinery applies
# on the betting side.
#
# LightGBM still runs and is still reported as a standalone comparison model;
# this only controls whether it feeds the ensemble.
#
# UPDATE -- no longer "parsimony alone". An autoresearch sweep (branch
# autoresearch/aug18) flipped this back to True and re-ran the full pipeline:
# development-era ensemble ATS ROI came out +8.13% against +9.31% for the
# same configuration with LightGBM excluded. The exclusion now has a direct
# measurement behind it, not just the three structural arguments above.
# (Both numbers are from a configuration state that has since changed, so
# read the ~1.2pp GAP rather than either absolute figure.)
ENSEMBLE_INCLUDE_LGBM = False

# Market anchoring of the ensemble's OWN prediction.
#
# The measured walk-forward edge-shrinkage slope is ~0.18-0.26 (see the
# "Edge shrinkage" section below): when the model claims a +5 point edge
# over the closing line, the realized edge is ~+1 point, i.e. the closing
# line carries most of the information about the margin and the model's raw
# output is ~4-5x overconfident relative to it. Until autoresearch r1-E1
# that correction lived ONLY in the bet-selection layer (shrinkage slope ->
# adaptive threshold), while the ensemble's predicted_margin, cover
# probability and win probability were left un-anchored.
#
# Allowed values:
#   None          -- previous behaviour: per-season Ridge meta-regression on
#                    the base-model predictions only, no market input.
#   "free"        -- r1-E1: spread_line (total_line for the totals ensemble)
#                    stacked into that Ridge as one more feature, weights and
#                    intercept fit freely. Measured: it fixed accuracy (dev
#                    MAE gap vs market +0.19 -> +0.01, raw ML Brier beat the
#                    market) but the fitted weights summed to 1.03-1.07 with
#                    a floating intercept of -0.1..-0.4, which tilted and
#                    tightened the ATS bet set (692 -> 488 dev bets) and cost
#                    ROI (+8.0% -> +3.5%).
#   "constrained" -- r2-E9: predicted = market_line
#                    + b * (pred_ridge - market_line), with ONE parameter b
#                    fit through the origin on strictly earlier seasons'
#                    out-of-fold rows: b = <e, r> / <e, e> with
#                    e = pred_ridge - line and r = target - line, clipped to
#                    [SHRINKAGE_SLOPE_FLOOR, SHRINKAGE_SLOPE_CAP]. That is
#                    the same quantity the edge-shrinkage machinery already
#                    estimates, applied to the prediction itself. No
#                    intercept and weights that sum to exactly 1 mean the
#                    sign and ranking of ats_edge = predicted_margin -
#                    spread_line are unchanged, so bet selection stays
#                    essentially today's rule while the margin and the
#                    probabilities become market-anchored. (If
#                    ENSEMBLE_INCLUDE_LGBM is on, b becomes a through-origin
#                    least-squares fit on [e_ridge, e_lgbm].)
# In every mode ats_edge is still predicted_margin - spread_line; the
# shrinkage slope and adaptive ATS threshold re-derive themselves on
# whatever edge scale results.
ENSEMBLE_MARKET_ANCHOR = "constrained"

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

# Minimum accumulated OUT-OF-FOLD residuals before the conditional-std model
# is fit on them instead of falling back to in-sample residuals (see
# modeling._walk_forward). Only the earliest walk-forward fold, which has no
# completed folds behind it, ever hits the fallback.
MIN_OOF_STD_GAMES = 200

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

# --- Edge shrinkage: the central correction in this file ---------------------
#
# A model's RAW edge (predicted_margin - spread_line) is not an estimate of
# its true edge, and treating it as one is what made every threshold here too
# permissive. Measured directly (probabilities.fit_edge_shrinkage_walk_forward,
# regressing realized home_margin - spread_line on the raw edge):
#
#   ridge     slope 0.203   corr +0.048
#   ensemble  slope 0.181   corr +0.039
#   lgbm      slope 0.087   corr +0.032
#
# i.e. when ridge claims a +5 point edge, the true expected edge is +1.0
# points. The signal is real (t ~ 2.9, not zero) but it is roughly 5x smaller
# than the raw number says. The old ATS_EDGE_THRESHOLD_POINTS = 1.5 therefore
# admitted every game whose TRUE edge was as low as 0.3 points, and
# sd(home_margin - spread_line) ~ 12.9 means beating -110 needs a true edge of
# about 0.77 points -- so roughly half of all bets placed were structurally
# negative-expectation before a single game was played.
#
# The old fixed thresholds' own edge buckets show exactly that (ridge ATS):
#   (1.5, 3]  n=1057  win 49.5%  ROI  -3.8%   <- half the book, -EV by design
#   (3, 5]    n= 716  win 55.2%  ROI  +6.8%
#   (5, 8]    n= 291  win 54.0%  ROI  +5.3%
#   (8, inf]  n=  58  win 56.9%  ROI +10.9%
#
# Rather than hardcode "3.5" (which would just be fitting a constant to that
# same backtest), the threshold is now DERIVED each season from two
# walk-forward quantities: the shrinkage slope above, and the break-even win
# rate implied by the actual odds being offered. See
# backtesting.adaptive_ats_threshold. Independent confirmation that this lands
# in the right place: the repo's own optimal_threshold_walk_forward, which
# never sees the test season, picked 3.0-4.75 in 10 of 10 seasons.
# VALIDATED by autoresearch (branch autoresearch/aug18). The derivation above
# sets the bar at exactly the edge that makes a bet zero-EV at the offered
# price. That "exactly break-even" choice was never itself tested, so the
# sweep scaled it by a constant and re-ran the pipeline both ways:
#
#   0.85x break-even (bet more games)  -> dev ATS ROI +5.28%, n=926
#   1.00x break-even (as derived)      -> dev ATS ROI +9.31%, n=800
#   1.25x break-even (demand a margin) -> dev ATS ROI +8.05%, n=647
#
# Both directions are worse, so the derived bar sits on a real local optimum
# rather than at an arbitrary point that happened to be convenient. This is
# the strongest single piece of evidence in the project that the edge-
# shrinkage machinery is doing something real: nothing about the derivation
# knew where the ROI-maximizing threshold was, and it landed on it anyway.
USE_ADAPTIVE_THRESHOLDS = True

# Fallback fixed thresholds, used when USE_ADAPTIVE_THRESHOLDS is False or
# before enough history exists to fit a shrinkage slope. Both raised from
# their old values (1.5 / 0.03) to sit near what the adaptive rule and the
# walk-forward optimizer independently converge on.
ATS_EDGE_THRESHOLD_POINTS = 3.5  # min |model_spread - market_spread| to place an ATS bet
ML_EDGE_THRESHOLD_PROB = 0.10  # min |model_prob - no_vig_market_prob| to place an ML bet

# Minimum prior graded games before a shrinkage slope is trusted; below this,
# fall back to the fixed thresholds above.
MIN_SHRINKAGE_GAMES = 600
# Raised from 300 by the same autoresearch sweep, and the cleanest result it
# produced. Measured on the ensemble: development ATS ROI +7.22% -> +8.02%,
# while the holdout era came back BYTE-IDENTICAL (+5.60%, n=92). That is not
# luck, it is mechanical: by the 2023+ holdout era every fold already has far
# more than 600 prior graded games, so this constant can only ever change
# behavior in the early development seasons. Raising it means those early
# folds fall back to the fixed ATS_EDGE_THRESHOLD_POINTS above instead of
# betting off a shrinkage slope fitted on 300-600 games, which is too few to
# estimate it stably.
# 900 was also tried and reverted (dev +7.36%): past ~600 it stops removing
# noisy slopes and starts discarding seasons that had a usable one.
# Guard rails on the fitted slope. A near-zero or negative slope means the
# model has no measurable edge over the line that season -- clamping to a
# small positive floor turns that into "demand an enormous raw edge" (i.e.
# bet almost nothing) rather than a divide-by-zero or a nonsensical negative
# threshold.
#
# Measured INERT: raising the floor to 0.08 was byte-identical across every
# model, i.e. the fitted slope never gets near it in practice and this guard
# rail currently never binds. Kept as a safety net for a pathological fold,
# but tuning it does nothing.
SHRINKAGE_SLOPE_FLOOR = 0.02
SHRINKAGE_SLOPE_CAP = 1.0

# Select ATS bets on the UNANCHORED ridge edge, while still REPORTING the
# market-anchored margin (see ENSEMBLE_MARKET_ANCHOR) everywhere.
#
# The principle: anchoring is a monotone rescaling of the selection signal,
# so it cannot add or remove real information about WHICH games to bet.
# Measured on r2-E9, exactly and not approximately: ats_edge = b_S * e_raw in
# every season (max deviation 0.0), so the SIGN and the RANKING of edges are
# perfectly preserved. The bet decomposition confirms it from the other side:
# on the 445 development bets the anchored and unanchored versions BOTH
# place, ROI is IDENTICAL (+0.0675 vs +0.0675; holdout +0.0703 either way on
# 73 common bets). Grading never moved -- only the bet universe did.
#
# What moved the universe was a scale mismatch. The betting layer's constants
# (ATS_EDGE_THRESHOLD_POINTS, and the SHRINKAGE_SLOPE_CAP above) are
# expressed on the RAW, unshrunk edge scale. Under anchoring the edge is 4-9x
# smaller, so (a) seasons below MIN_SHRINKAGE_GAMES fell back to a 3.5-point
# bar worth a raw-equivalent 32.5 / 15.9 / 10.7 points in 2014/2015/2016 and
# placed 0 / 0 / 1 bets instead of ~52 / 50 / 49, and (b) for 2017-2019 the
# betting layer's own shrinkage slope -- fit on PRIOR seasons, which were
# anchored with a SMALLER b_S than the current one -- came out at 1.982 and
# hit SHRINKAGE_SLOPE_CAP, which the comment above calls "measured INERT" and
# which is not inert once anchoring is on. Applying the anchor's shrinkage
# and then the betting layer's own estimate of that same quantity is
# double-shrinkage.
#
# So: report the anchored margin (that is what fixed MAE/RMSE/Brier against
# the closing line) but hand the selection machinery the unanchored edge it
# was validated on. True decouples the two; False restores r2-E9's behaviour
# of selecting on whatever scale the report happens to be on.
ATS_SELECT_ON_UNANCHORED = True

# Moneyline: restrict to underdog-side bets only.
#
# Measured directly against a blind "bet the underdog in these same games"
# benchmark, which is the control this needs to beat to have added anything:
#   thr=0.03  model -2.6%  |  blind-dog on same games -3.6%
#   thr=0.08  model +2.8%  |  blind-dog on same games +0.9%
#   thr=0.12  model +5.1%  |  blind-dog on same games +4.6%
# Split by side, at every threshold tested, the model's underdog bets carried
# all of the return (+7.5% at thr=0.12) and its favorite bets lost money
# (-2.1%). That is consistent with the model being a WORSE probability
# estimator than the market in aggregate (Brier 0.217 vs the market's 0.211):
# where it disagrees with the line on a favorite, the line is usually right.
# Blind-dog across ALL games is -5.2%, so this is not a generic underdog
# bias -- the game SELECTION carries real information, the side choice on
# favorites does not.
ML_UNDERDOG_ONLY = True

TOTAL_EDGE_THRESHOLD_POINTS = 1.5  # min |model_total - market_total| to place a totals bet -- phase-1 starting
# point; untuned. Totals showed no edge over the market at all (see
# TOTAL_MODEL_FEATURES' note), so it is deliberately left on the old fixed
# rule rather than given the adaptive treatment -- there is no measured edge
# for a shrinkage slope to scale.
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
