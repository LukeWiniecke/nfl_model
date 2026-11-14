# Research log

Full write-up of every experiment, kept and reverted, with measurements. The project overview is in the [README](../README.md).

Research/backtesting project. **Not betting advice.** Evaluates whether a
football-fundamentals model (EPA, success rate, power ratings, rest, etc.)
can identify historical pricing inefficiencies against closing NFL market
lines — spread and moneyline — from 2010 through the latest completed season.

## What it does

1. Pulls play-by-play, schedules, closing market lines (spread, total,
   home/away moneyline, spread odds), weekly player stats, and weekly injury
   reports from `nflreadpy` — one source covers football data, betting
   lines, player-level stats, and injuries, with 100% moneyline/odds
   coverage from the 2010 season on.
2. Builds leakage-safe rolling team-form features (EPA/play, success rate,
   pass/rush EPA, defensive EPA allowed), QB-level form (starter
   identification, rolling EPA/dropback tracked per-QB across team changes,
   starter-continuity), an injury feature (severity-weighted count of each
   team's official pre-game injury-report players), weather/neutral-site
   context, and Elo-style power ratings, where every feature for a game only
   uses information that existed strictly before that game was played.
3. Trains three staged margin-prediction models with season-level walk-forward
   validation (to predict season S, train only on seasons < S):
   - **Ridge baseline** — a small, interpretable feature set (power rating
     diff + a few recency-weighted EPA/success-rate diffs).
   - **LightGBM** — the full engineered feature set, as a richer comparison.
   - **Ensemble** — a 2-feature meta-regression blending ridge's and
     LightGBM's predicted margins, itself refit walk-forward each season
     (see "What's worked" below).
4. Converts predicted margins into ATS cover probabilities and moneyline win
   probabilities via a normal approximation around the predicted margin —
   using a **per-game predicted std**, not one constant for every game (see
   "What's worked" below) — then walk-forward recalibrates those
   probabilities (isotonic regression, fit only on strictly earlier seasons)
   before comparing them against de-vigged market probabilities to compute
   edge.
5. Backtests flat-stake ATS and moneyline bets above configurable edge
   thresholds, and reports ROI/win-rate (with bootstrap 95% confidence
   intervals) sliced by season, team, edge size, home/away, favorite/underdog,
   and market-spread bucket.
6. Reports model skill directly against the market baseline (MAE/RMSE/Brier
   score vs. just reading the closing line), independent of any bet-selection
   threshold — see "Key finding" below.
7. Simulates a compounding, Kelly-criterion bankroll (separate from the flat
   1-unit backtest) to show what flat-stake ROI hides: risk. See "Bet sizing
   and execution realism" below.

## Key finding

**The ensemble now matches the closing line on MAE and beats it on RMSE and
on moneyline Brier score — but read carefully what that does and does not
mean.** It is achieved by *anchoring* the prediction to the line:
`predicted_margin = spread_line + b·(model − spread_line)`, with the single
coefficient `b` fit through the origin on strictly earlier seasons (see
`config.ENSEMBLE_MARKET_ANCHOR`). Measured walk-forward, `b` runs about
0.11–0.35, which is the honest statement that only a fifth to a third of the
model's disagreement with the market survives contact with reality. The
model is therefore not out-predicting the market from fundamentals; it is
adding a small, correctly-sized correction on top of it. That is still a
real improvement on the previous state, where the same shrinkage was applied
only at the betting layer while every *reported* number — the Diagnostics
tab, the probabilities, the live board — used the unshrunk prediction.

Development era, ensemble, against the closing line:

| | Before anchoring | Now |
|---|---|---|
| MAE gap vs market | +0.192 | **−0.001** |
| RMSE gap vs market | +0.172 | **−0.018** |
| Moneyline Brier vs market | +0.0058 | **−0.0006** |
| Moneyline calibration slope | 0.775 | **0.995** |
| Cover-probability calibration slope | 0.003 | **0.520** |

A calibration slope of 1.0 is perfectly calibrated. The moneyline Brier line
is the first time in this project that the model's win probability has beaten
the market's.

**The bet book is unchanged, deliberately.** Anchoring rescales the
selection signal, and re-deriving thresholds on the rescaled scale would
double-apply a shrinkage the betting layer already estimates — that is
exactly what cost 30% of the bets in the first attempts. Both markets
therefore select on the frozen pre-anchoring signal (`ats_edge_select` /
`ml_edge_select`), and the ensemble places precisely the bets the ROI
numbers elsewhere in this README describe: 692 development and 96 holdout
ATS bets, 493 and 154 moneyline, bet for bet, with identical profit on every
one. Two independent reviews verified that against the pre-change tables.

Flat-stake ROI is therefore unchanged: ATS +8.0% development (95% CI
[+1.1%, +15.2%]) and +5.3% holdout; moneyline +12.1% development (95% CI
[+0.2%, +24.6%]) and +2.1% holdout. Those intervals are wide and the holdout
samples small, so this remains "the most promising model so far," not "found
an edge."

**Why the moneyline selector is frozen rather than rebuilt.** For the spread,
anchoring cancels: `anchored − spread = b·(model − spread)`, a single
positive scalar per season, so the ranking of games is identical and
reporting one while selecting on the other is information-free. In the
probability domain it does not cancel — each game picks up a different
intercept `(1−b)·spread_line` and is then divided by its own predicted std —
so an anchored probability is a genuinely *different* selector, not a
rescaling. Measured Kendall tau between the old and new selection
magnitudes: 0.25–0.32. Six attempts to build a moneyline selector under
anchoring (logistic stacking plus an adaptive bar, a break-even bar, and
selecting on the anchored-chain legacy probability) either collapsed out of
sample or bought volume by loosening the bar. All are recorded in
`results.tsv`; the open question is flagged in `BUGS.md`.

## What's worked, and what hasn't (in order tried)

1. **A second, slower-decaying "stable" Elo track** (organizational-quality
   bias fix) — tried, **reverted**. Made the team-level bias slightly worse;
   the fast/slow Elo tracks were 91% correlated.
2. **LightGBM rolling-window pruning** ([4,8,17] → [8]) — **kept**. Its
   feature importances had come back nearly flat across ~20 near-duplicate
   columns; pruning gave a small, real MAE/Brier improvement.
3. **Opponent-adjusted EPA** (schedule-strength correction) — tried,
   **reverted**. Barely moved the team-bias or divisional-game gaps it
   targeted, and slightly hurt LightGBM. `elo_diff` likely already captures
   most of the available schedule-adjustment signal.
4. **QB-level form** (starter identification + rolling EPA/dropback tracked
   per-QB across team changes, plus a starter-continuity flag) — **kept, and
   the best result so far.** Improved MAE/RMSE/Brier for *both* models, and
   specifically closed most of two previously-diagnosed gaps: the
   divisional-game MAE deficit vs. market shrank from -0.51 to -0.27 (vs.
   -0.22/-0.23 for non-division games), and the playoffs deficit went from
   -0.62 to -0.30. `qb_epa_per_att_ewm_diff` ranks 5th of ~25 features in
   LightGBM's importance, on par with `elo_diff`.
5. **Prime-time and playing-surface flags**, pulled in from the Spreadspoke
   reference CSV (`data_ingestion.load_spreadspoke_context`, since nflverse
   doesn't have either) — tried, **reverted**. Joined cleanly (99.1% match
   rate by date+teams), motivated by a real finding (the market's edge grows
   in high-profile/late-season games), but ridge came back flat (MAE 10.161
   -> 10.163, noise) and LightGBM got consistently slightly worse across
   MAE/RMSE/Brier. The ingestion/join machinery is still wired up if a
   different framing of "high-profile game" is worth testing later.
6. **A real injury-report feature** (severity-weighted sum of each team's
   official pre-game injury designations — Out/Doubtful/Questionable — from
   nflverse's own weekly injury reports) — **mixed result, kept for ridge
   only.** Ridge's aggregate MAE barely moved, but the divisional-game gap
   (rivals who scout each other closely, including injury news) closed
   further, from -0.27/-0.23 (div/non-division) to essentially zero:
   -0.244/-0.237. LightGBM got consistently slightly worse with it, so it's
   excluded there — ridge and LightGBM have independent feature lists in this
   codebase, so this isn't an all-or-nothing choice.
7. **Upgraded, kept: weight each injured player by their own snap share, not
   just their position.** Each severity weight is now scaled by that
   player's own prior rolling snap share (`feature_engineering.
   compute_player_snap_share`, from nflreadpy's `load_snap_counts`, joined
   via a gsis_id↔pfr_id crosswalk since injuries and snap counts don't
   share an ID scheme) — a starter being out costs more than a rarely-used
   backup at the same report status. QBs are excluded entirely: the
   dedicated QB-form/continuity feature already tracks the starter from
   real performance data, so including them here too would double-count
   the same signal. Measured directly: ridge's MAE/RMSE/Brier all improved
   (10.157→10.151 / 13.086→13.080 / 0.2172→0.2171), ATS ROI went from -0.6%
   to +1.0% on essentially the same bet count, ML ROI was flat. Ensemble/ML
   ROI dropped (+0.3%→-2.8%) but sits well inside its own bootstrap 95% CI
   (-7.5% to +2.2%) — not distinguishable from noise — and ensemble/ATS
   improved slightly too. **Also found and fixed a real pre-existing bug**:
   nflverse's injuries table uses pre-relocation team codes (OAK/SD/STL),
   never matching `games`' standardized codes (LV/LAC/LA) — the injury
   feature had silently been null for every Raiders/Chargers/Rams game
   before their moves. `data_ingestion.load_injuries` now standardizes team
   codes on load. Snap-count data only goes back to 2012 (nflreadpy's own
   floor); earlier seasons and any crosswalk miss fall back to a fixed
   default importance (`config.DEFAULT_SNAP_SHARE`) rather than dropping
   the player.
8. **Checked directly: does the model underperform early in the season?**
   The hypothesis: rolling/EWMA form features still mostly carry over
   decayed values from *last* season in the first few weeks, with no signal
   yet about offseason roster/coaching changes. **Measured directly — no,
   if anything the opposite.** MAE-vs-market gap by time of season, all
   three models (playoffs grouped by `game_type`, not week number, since
   playoff games continue the week count and the regular season itself went
   17 → 18 weeks in 2021, making raw week number incomparable across eras
   past week 13):

   | | Weeks 1-3 | Weeks 4-8 | Weeks 9-13 | Weeks 14-18 | Playoffs |
   |---|---|---|---|---|---|
   | Ridge | 0.180 | 0.203 | 0.302 | 0.255 | 0.229 |
   | LightGBM | 0.610 | 0.741 | 0.778 | 0.579 | 0.600 |
   | Ensemble | 0.064 | 0.212 | 0.231 | 0.270 | 0.174 |

   (Gap = model MAE − market MAE, points; lower is better.) Every model's
   gap is *smallest* in Weeks 1-3 and *peaks* in Weeks 9-13. Plausible
   reason: Elo's between-season regression already gives the model a
   reasonable Week 1 prior, and the market has just as little in-season
   signal that early — neither side has much of an edge yet. The gap likely
   grows mid-season because that's exactly when injuries/trades/bye-week
   mismatches create complexity a backward-looking rolling average can't
   see coming, while the market (real-time injury reports, line movement on
   breaking news) adjusts faster than a shifted rolling average can. New
   `evaluation.summarize_by_week_bucket` / dashboard's Backtest slices tab
   ("ROI by time of season") has the threshold-dependent ROI version of this
   same cut.
9. **Conditional (per-game) uncertainty** — replaced the single global
   residual std (one number for every game, forever) with a small Ridge
   predicting each game's OWN |residual| from elo mismatch, divisional,
   playoff, weather, and rest-mismatch context (`modeling._std_feature_frame`,
   walk-forward fit exactly like the margin models), floored/capped at
   8-22 points. **Kept.** Essentially flat for ridge (Brier 0.2171 ->
   0.2172), and a modest real cost for LightGBM's Brier (0.2327 -> 0.2357) —
   plausibly because LightGBM's richer feature set already implicitly
   captures some of this context, so re-modeling it on the residual is
   partly redundant noise. Kept anyway since the structural idea is sound and
   the cost is small, but worth revisiting if a cleaner conditional-variance
   approach turns up. **Tried and reverted: adding `qb_injury_diff_abs`/
   `qb_continuity_diff_abs`** to this same std feature set, on the
   hypothesis that a real starter-out/backup situation should carry more
   predictive uncertainty than a normal game. Measured directly: Brier moved
   by less than 0.0002 for all three models — noise, not signal. ATS ROI was
   byte-identical (bet-side there comes from the points edge, untouched by
   std). ML ROI swung more than that near-zero Brier change justifies (ridge
   -3.2%→-2.2%, LightGBM -3.4%→-3.0%, but ensemble -3.0%→**-4.3%**) — read as
   threshold-crossing churn on an unchanged probability distribution, not a
   real effect. Likely reason: `qb_injury_diff` is already a
   `MARGIN_MODEL_FEATURES` input for ridge, so its signal is already baked
   into the point prediction (and thus partly into the residual this std
   model targets) before this feature set ever saw it separately; and
   `qb_continuity_diff` is always 0 for a live upcoming game specifically
   (`compute_qb_form`'s carried-forward branch always assumes continuity for
   both teams), limiting it to a training-only signal even in principle.
10. **Walk-forward probability recalibration** (isotonic regression on raw
   model probability -> outcome, fit only on strictly earlier seasons,
   `probabilities.walk_forward_calibrate`) — **mixed result, kept for ridge
   and the ensemble's moneyline market, reverted for LightGBM's.** Measured
   with a controlled ablation (identical bets, calibrated vs. raw
   probability only): ridge ML ROI -3.7% -> -3.4%, ensemble ML ROI -4.5% ->
   **+0.3%**. But LightGBM ML got clearly worse (-2.4% -> -4.8%) and its bet
   mix shifted from 56% favorites to 21% — isotonic regression appears to
   overfit LightGBM's raw probability distribution into a curve that
   manufactures spurious underdog "edges." ATS is unaffected either way
   (bet_side there comes from the points-based edge, not this probability),
   so calibration stays on for ATS across all models; it only changes Kelly
   stake sizing there, never bet selection.
11. **Ensemble (ridge + LightGBM blend)** — a 2-feature meta-regression on
   each model's own predicted margin, refit walk-forward each season so the
   blend weight can shift toward whichever model has actually been more
   accurate so far, rather than a fixed 50/50 split (`modeling.
   walk_forward_ensemble`). **Kept — best result of the three models.**
   Lowest MAE/RMSE of the three (10.13/13.08, vs. ridge 10.16/13.09 and
   LightGBM 10.60/13.56), and the only one with positive flat-stake ROI in
   both markets (see "Key finding" above, including the caveat that it's not
   yet statistically significant). The fitted blend consistently leans
   heavily on ridge (weight ~0.83-0.87) with a small, growing LightGBM
   contribution in recent seasons (~0.00 in 2015, ~0.09 by 2023-24) — read
   this as "ridge is still doing most of the work," not "the two models are
   equally useful."
12. **Fixed a real bug found while adding a QB-specific injury feature: live
   predictions had ZERO real QB signal.** `compute_qb_form` only ever had
   rows for games that had actually been played (it's built from real
   pass-attempt data), so an upcoming game — which hasn't happened yet —
   simply had no row, went null after the join, and got median-imputed at
   prediction time. Verified directly: every one of a real batch of 51
   upcoming games had `qb_epa_per_att_ewm_diff`/`qb_continuity_diff` = null
   before this fix. This is invisible in the walk-forward backtest, since
   every *historical* game being predicted there has already been played by
   the time it's used as a training row — it only ever affected the live
   board. **Fixed**: `compute_qb_form` now carries each team's most recent
   played game's starter forward (their own post-game rolling EPA/attempt,
   found via `join_asof` on the team's schedule timeline) into any upcoming
   game, assuming continuity by default. Confirmed fixed: 0/51 nulls after.
   Also **added a QB-specific injury-severity feature**
   (`qb_injury_diff`, `feature_engineering.compute_qb_injury_features`) —
   the same severity × snap-share weighting as the existing injury feature,
   but for the QB position specifically (which the existing one deliberately
   excludes), and the one part of this that's genuinely knowable *live*: if
   the presumed carried-forward starter is actually listed as
   Out/Doubtful/Questionable this week, this is the only feature that can
   flag it before a new starter has actually played to prove it. Found and
   fixed a second bug along the way: naively joining left `qb_injury_diff`
   null whenever EITHER team had no QB flagged that week — the *common*
   case, since most weeks nobody's QB is hurt at all — collapsing 64/4363
   played games with a usable value down to almost nothing. Filling the
   missing side with 0.0 (no report entry = healthy, not unknown) before
   subtracting restored the far more common one-team-only case: 1024/4363
   games now carry a real nonzero value. Kept for ridge (matching
   `injury_diff`'s precedent), excluded for LightGBM by default. Measured
   directly, before → after:

   | | Ridge MAE/RMSE/Brier | Ridge ATS / ML ROI | LightGBM MAE/RMSE/Brier | LightGBM ATS / ML ROI | Ensemble MAE/RMSE/Brier | Ensemble ATS / ML ROI |
   |---|---|---|---|---|---|---|
   | Before | 10.151 / 13.080 / 0.2171 | +1.0% / -3.4% | 10.599 / 13.557 / 0.2357 | -1.9% / -2.4% | 10.131 / 13.073 / 0.2172 | +0.6% / -2.8% |
   | After | 10.142 / 13.078 / 0.2169 | +1.4% / -3.2% | 10.598 / 13.548 / 0.2359 | -0.9% / -3.4% | 10.117 / 13.063 / 0.2173 | +1.5% / -3.0% |

   ATS improved for all three models, consistent in direction if modest in
   size. LightGBM never sees `qb_injury_diff` at all (excluded from its
   feature list), so its whole change here — and roughly a third of
   ridge/ensemble's — comes purely from the carry-forward bug fix giving
   more complete historical training data, not the new feature itself.
   ML is mixed (ridge improved, LightGBM and ensemble got slightly worse);
   given LightGBM's ML market is already known to be sensitive to small
   input changes (see the isotonic-calibration note above), and these ROI
   deltas are within the range this project's own bootstrap CIs have shown
   to be noise at this sample size, this reads as a net positive on the
   metric (ATS) least distorted by calibration quirks, not a clean win
   everywhere.
13. **Tried and reverted: referee-crew tendency** (`ref_points_ewm`, a
   leakage-safe rolling EWM of total points in games each REFEREE has
   officiated — `feature_engineering.compute_referee_features`,
   `data_ingestion.load_officials`). Motivated by real, documented
   crew-level tendencies in penalty calls and game flow — untested and
   orthogonal to everything else here, and aimed squarely at the totals
   model, which had no edge at all. **Two real bugs found and fixed
   getting there:** (1) `officials`' own `game_id` uses nflverse's OLD
   numeric gsis format (e.g. `"2015091000"`), not the
   `season_week_away_home` format used everywhere else in this pipeline —
   zero ID overlap, silently producing 100% nulls until joined through
   `games.old_game_id` (nflverse's own crosswalk, previously dropped by
   `cleaning.clean_schedules` and now kept). (2) `officials` only covers
   2015–2025 (a hard floor *and* ceiling, unlike most loaders here), so the
   earliest walk-forward folds — trained only on pre-2015 seasons — had a
   training pool with zero non-null values, making even the *median* null
   and crashing Ridge on a NaN input; fixed by having `modeling._impute`
   fall back to 0.0 when a column's own median is itself null. Once
   actually working (confirmed: null before 2015, ~70%+ non-null from 2015
   on, and — as expected going in — 100% null for every LIVE upcoming game,
   since referee assignments aren't public that far ahead and nflverse's
   officials table, like `player_stats`, is compiled from completed games
   only), **measured directly: worse on every MAE/RMSE/Brier metric, for
   every totals model** — the one it was actually built for:

   | | Ridge | LightGBM | Ensemble |
   |---|---|---|---|
   | Totals MAE/RMSE/Brier, before | 10.678 / 13.494 / 0.2551 | 11.063 / 13.969 / 0.2794 | 10.671 / 13.468 / 0.2560 |
   | Totals MAE/RMSE/Brier, after | 10.687 / 13.498 / 0.2553 | 11.088 / 13.981 / 0.2804 | 10.680 / 13.473 / 0.2562 |
   | Totals ROI, before → after | -1.5% → -1.4% | -1.0% → **-2.2%** | -2.1% → -1.5% |

   For the margin models, MAE/RMSE/Brier moved less than 0.02 either way,
   but ATS ROI — the metric least distorted by calibration noise — got
   **worse for all three** (ridge +1.4%→+1.0%, LightGBM -0.9%→-1.5%,
   ensemble +1.5%→+0.8%), consistent enough in direction to read as real
   rather than scatter. Likely reason: this is a *raw* rolling average of
   realized total points per referee, which doesn't control for which
   teams that ref happened to draw — a ref randomly assigned several
   high-powered offenses in a row looks like a "high-scoring ref" when
   it's team quality leaking through, not a genuine crew tendency. A fairer
   test would use each game's residual against its own expected total
   (from the other features), not the raw score. The ingestion/feature
   machinery is left in place, unused by any feature list, for that retry.

   **Retried exactly that** (`ref_total_resid_ewm`): rolled each referee's
   EWM against `total_points - total_line` instead of raw total points —
   properly strips out the team-quality confound (confirmed: mean ~0.4, std
   ~4.0, vs. raw points' mean ~46/std ~4.2). Conceptually the right fix, but
   **empirically still no win**: totals MAE/RMSE/Brier moved worse again
   for all three totals models (smaller for ridge/ensemble than the
   raw-points attempt, but LightGBM's totals ROI still went from -1.0% to
   **-2.5%**), and margin ATS ROI got worse for all three again (ridge
   +1.4%→+0.7%, LightGBM -0.9%→-2.4%, ensemble +1.5%→+1.1%) — same
   direction as the raw-points version. Read as a real negative result
   rather than an implementation bug: two genuinely different framings both
   landed in the same place, most likely because each individual referee
   only accumulates a few hundred career games across ~15-17 active crews —
   not enough games per ref for a rolling stat to separate a genuine crew
   tendency from noise, even with team quality properly stripped out. Not
   retried a third time.
14. **Tried and reverted: QB Next Gen Stats (CPOE)** (`qb_cpoe_ewm_diff`/
   `_sum`, the starting QB's leakage-safe rolling completion-% over
   expectation — `feature_engineering.compute_qb_ngs_form`,
   `data_ingestion.load_nextgen_stats_passing`). CPOE adjusts for throw
   difficulty via player-tracking data — a genuinely different *kind* of
   QB signal than the pbp-derived EPA/attempt already in the model, not
   just a refinement of it. NGS only covers 2016 on (a hard floor+ceiling,
   like `officials`), but unlike the referee feature, the existing QB
   carry-forward mechanism means most live upcoming games actually get a
   real, non-null value (confirmed: 43/51 real upcoming games), not ~0%.

   **Measured directly, aggregate (2013–2025)**: ridge MAE/RMSE/Brier all
   slightly worse (10.142/13.076/0.2169 → 10.146/13.089/0.2171); margin ATS
   ROI worse for ridge (+1.4%→+1.0%) and LightGBM (-0.9%→-1.9%), though
   ensemble improved slightly (+1.5%→+1.7%). Since the feature is null
   before 2016, **ran a controlled by-era A/B** (temporarily disabled the
   join, re-ran, compared season-by-season) to check whether the aggregate
   number was hiding a real win concentrated in the years it actually
   covers. Confirmed pre-2016 deltas are *exactly* zero (a good leakage
   sanity check), but even isolated to 2016+ — where CPOE has real,
   non-imputed values — MAE was still slightly worse, not better (ridge
   margin: 10.024 vs. 10.018 without it; ridge totals: 10.652 vs. 10.631
   without it). This rules out "diluted by uninformative early years" as
   the explanation — it's genuinely not helpful within its own valid
   coverage window, not just averaged down by years it can't see.

   Disabled by commenting out the `qb_ngs_form` join itself in `main.py`,
   not just dropping it from the feature lists (the pattern used for the
   referee/prime-time reverts): `qb_cpoe_ewm` ends in `_ewm`, so leaving it
   joined would auto-generate `qb_cpoe_ewm_diff`/`_sum` via
   `build_modeling_table`'s generic form-column detection, and LightGBM's
   suffix-based auto-inclusion would keep using it regardless of what the
   explicit ridge lists say.
15. **Built and disabled — not a modeling failure, an upstream data quality
   one: depth-chart-aware QB carry-forward.** `compute_qb_form`'s
   live/upcoming branch assumes "whoever started this team's most recent
   game is still starting" — true most weeks, wrong exactly when a
   trade/retirement/rookie promotion happened since. `data_ingestion.
   load_current_qb_depth_chart` pulls nflverse's own depth chart for each
   team's presumed current starter; `compute_qb_form`'s new
   `depth_chart_starters` param overrides the naive carry with that
   player's own rolling stats when the two disagree — never for
   played/backtest rows, confirmed byte-identical across every model.

   **Two real bugs found and fixed building the mechanism itself.**
   (1) nflreadpy's depth-chart loader returns two different schemas
   depending on the season — archived seasons are a per-(season, week)
   snapshot, the *current* season instead comes from a live feed keyed by
   a timestamp with no week at all; the loader branches on which schema
   came back. (2) The existing "unplayed" bucket the general carry-forward
   already used turned out to *also* include played historical games
   missing a QB pass-attempt row in `player_stats` (a real, if rare, data
   gap — old pre-relocation Raiders games). Initially "fixed" by
   restricting to genuinely-unplayed rows — which **measurably worsened**
   aggregate MAE, revealing the naive carry-forward was already a
   net-positive fallback for those gap games too. Corrected by keeping the
   original carry-forward scope intact and gating *only* the new
   depth-chart override on "genuinely unplayed" — confirmed via a full
   pipeline run that this restores byte-identical backtest numbers
   everywhere.

   **With the mechanism verified correct, found nflverse's live
   depth-chart feed itself isn't reliable enough to trust.** Spot-checked
   against each team's own actual most-recent starter and found real,
   confirmed team-to-player misalignment: Tua Tagovailoa (actually Miami's
   QB) listed as Atlanta's starter, Kyler Murray (Arizona's) listed as
   Minnesota's, Miami's own entry showing a different player entirely,
   Arizona's likewise, Kirk Cousins appearing to duplicate across two
   teams' entries. Confirmed directly against nflreadpy's raw output —
   not a join bug here, an upstream data problem in nflverse's live source
   for the current season specifically. Disabled by passing `None` for the
   override param in `main.py` (not the whole join, unlike the CPOE
   revert — there's no LightGBM auto-suffix risk here, since this param
   only gates internal logic and never creates a new modeling-table
   column). Worth revisiting if nflverse's live feed improves.
16. **Student's-t replaces the Normal CDF everywhere a probability is
   derived from a predicted margin/total — kept, no regressions found on
   any of the six models.** Prompted by actually pulling the reliability
   curve's real numbers rather than assuming: the RAW (pre-isotonic)
   probabilities are consistently *more extreme* than reality in both
   directions — overconfidence, not the underconfidence/compression-toward-
   50% originally hypothesized. Concretely, LightGBM's raw ML: a bin
   predicting a 14.8% win probability actually won 31.7% of the time; a bin
   predicting 94.7% only won 78.2%. Worst for LightGBM specifically (mean
   |bin gap| ~0.088 vs. ridge/ensemble's ~0.02–0.035) — which happens to be
   exactly the one market left deliberately uncalibrated, since isotonic
   was already known to distort it (item 10 above). A Student's-t
   distribution (one extra free parameter controlling tail fatness, → ∞
   converges to Normal) directly targets that shape mismatch.

   `probabilities.fit_t_dof_walk_forward` fits the degrees-of-freedom
   parameter walk-forward — same discipline as the isotonic calibration
   and conditional-std model, using only strictly earlier seasons'
   standardized residuals each time. The fit itself independently confirms
   the diagnosis: it lands near-Normal for ridge/ensemble (dof≈21,
   consistent with their raw probabilities already being reasonably
   shaped) and hits the floor for LightGBM (dof=3.0 — genuinely fat-tailed
   residuals, not just a noisy fit).

   **Measured directly, before (Normal) → after (Student's-t):**

   | | Ridge | LightGBM | Ensemble |
   |---|---|---|---|
   | Margin Brier | 0.21690 → 0.21688 | 0.23586 → **0.23177** | 0.21735 → 0.21740 |
   | Totals Brier | 0.25512 → 0.25477 | 0.27937 → **0.27417** | 0.25597 → 0.25566 |
   | ML ROI | -3.22% → **-2.63%** | -3.35% → -3.40% | -2.96% → **-2.44%** |

   ATS and totals ROI are unchanged everywhere, exactly as expected — bet
   *side* in those markets comes from the points edge, not this
   probability; only Kelly stake sizing and the reliability-curve display
   depend on it. Wired into `backtesting.py` (all three bet-building
   functions), `live_picks.py` (a single all-history dof fit, mirroring how
   live isotonic calibration already works), and `evaluation.skill_vs_market
   [_totals]` (so the reported Brier reflects the same probability formula
   actually used for betting, not a stale Normal-based one computed
   separately). Kept everywhere rather than the usual per-model split —
   for once, nothing to trade off.
17. **Tried and reverted: hyperparameter tuning for Ridge's alpha and
   LightGBM's num_leaves/learning_rate/n_estimators** — never actually
   tuned via cross-validation before, inherited defaults from when the
   model was first built. Checked with a real walk-forward grid search
   (train on strictly-prior seasons, predict the next one, exactly as
   production already does). **By raw MAE alone this looked like a clear
   win**: ridge's margin MAE improved at alpha=20 (10.136 vs. the default
   alpha=5's 10.142; totals needed the opposite direction and kept alpha=5
   unchanged). LightGBM's was dramatic — the old defaults (15 leaves,
   lr=0.03, 400 trees, MAE 10.598) were beaten by every simpler setting
   tried, a textbook overfitting signature, bottoming out at 3 leaves,
   lr=0.01, 300 trees (MAE 10.219 — nearly half the old gap to ridge's own
   10.14 closed by hyperparameters alone), and this transferred to the
   totals target too.

   **But MAE is the wrong objective here, and the real backtest proved it
   emphatically.** With the MAE-tuned LightGBM deployed: its ML win rate
   on the games it chose to bet **collapsed to 36.6%** (ROI -3.4% →
   **-7.5%**), its ATS ROI went -0.9% → -3.2%, and the ensemble — which
   blends ridge and LightGBM — flipped from a **positive** +1.5% ATS ROI
   to -0.2%. Isolating ridge's alpha=20 alone (LightGBM reverted) showed
   the same pattern in miniature: ATS ROI got worse for both ridge
   (+1.4% → +1.1%) and ensemble (+1.5% → +0.8%) despite ridge's own tiny
   MAE gain. A lower-error point prediction, in aggregate, is not the same
   thing as a model that identifies genuine market mispricing —
   optimizing hyperparameters against MAE alone, with no eye on the actual
   downstream decision, measurably threw away the very thing that made
   these hand-picked defaults useful. Both reverted to their original
   values; this is a real, if counterintuitive, lesson worth keeping in
   mind before ever tuning against an accuracy metric here again.
18. **A 29-experiment autonomous sweep (`program.md`, branch
    `autoresearch/aug18`) — two changes kept, and a methodology lesson
    worth more than either.** An LLM ran the loop defined in `program.md`:
    propose a change, run the full pipeline, score the ensemble's
    development-era ATS ROI against a fixed harness
    (`scripts/autoresearch_eval.py`), keep or `git reset --hard`. Kept:
    **`EWMA_HALFLIFE_GAMES` 6 → 9** and **`MIN_SHRINKAGE_GAMES` 300 → 600**.
    Together: development ATS ROI **+5.36% → +7.91%** and holdout ATS ROI
    **+6.24% → +7.51%** — both eras improved, which matters more than the
    headline number (see below). Still not statistically significant: the
    development 95% CI is [+0.8%, +15.0%] at n=691.

    **The lesson: the loop's first version reproduced this project's
    signature failure mode, mechanically and fast.** Its original keep rule
    was "development ATS ROI improved" — the same rule most of items 1-17
    were decided under. Run against an EWMA-halflife sweep it kept every
    step of 6 → 7 → 9 → 12 → 16, because development ROI rose monotonically
    (+6.06% → +8.53%) the whole way. Holdout ROI fell monotonically over
    the same steps (+7.36% → +2.04%). Every individual step was "an
    improvement," every one was inside the ~13pp bootstrap CI, and the
    end state would have been reported as the best result of the run.
    Adding a **net-improvement gate** (a change must gain more development
    ROI than it gives up in holdout ROI) reverted halflife to 9 and killed
    3 of the 5 originally-kept changes — including an Elo home-field
    change that had looked like the most credible result of the sweep
    until it was re-measured on the corrected base, where its entire gain
    turned out to be an interaction with the overfit halflife.

    **Negative results, recorded so they aren't retried blind** (all
    development-era ATS ROI, all reverted): Elo K 15 and 26, Elo HFA 40 and
    48, season regression 0.25, MOV multiplier off, ridge alpha 2.5 and 12,
    `MIN_TRAIN_SEASONS` 5, dropping `off_rush_epa_ewm_diff`, adding
    `def_success_rate_allowed_ewm_diff`, adding `off_plays_ewm_diff` (pace)
    to the margin model, and an `elo_diff × div_game` interaction — that
    last one produced the sweep's single largest development gain (+0.71pp)
    while holdout fell 3.89pp, i.e. it is exactly the trap the gate exists
    to catch. Three incidental findings worth keeping:
    - **`qb_continuity_diff` is load-bearing**, despite its own note saying
      it is always 0 for live games. Dropping it: development +6.98% →
      +4.90%, holdout +7.51% → **-9.77%**.
    - **LightGBM's exclusion from the ensemble is now measured**, not just
      parsimony as `config.py` admitted: re-including it cost ~1.2pp of
      development ATS ROI.
    - **The adaptive ATS threshold sits on a real local optimum.** Scaling
      the derived break-even bar by 0.85× gave +5.28% and by 1.25× gave
      +8.05%, against +9.31% at the derived 1.0×. Nothing in the
      edge-shrinkage derivation knew where the ROI-maximizing threshold
      was, and it landed on it — the strongest evidence so far that that
      machinery is doing something real.

## Totals (over/under) — phase 1

A third market, built alongside ATS/ML rather than sharing their pipeline:
same walk-forward discipline, same three models (ridge/LightGBM/ensemble),
but predicting **total points** (`home_score + away_score`) against the
market's `total_line`, with its own feature set
(`config.TOTAL_MODEL_FEATURES`, `modeling.gbm_total_feature_columns`) —
**combined** (summed) offense/defense form instead of the margin models'
diffs, since what predicts a high- or low-scoring game is "how potent are
both offenses / how leaky are both defenses," not "which team is better."
Data was already fully there: `total_line`/`over_odds`/`under_odds` have
100% coverage back to 2010, same as spread/moneyline.

**Phase 1 only** — model, backtest, and skill-vs-market check, deliberately
not yet wired into Upcoming picks or the dashboard, to see if there's any
signal before investing in that UI surface area. Measured directly: **worse
than the market on every metric, for every model.** Ridge: MAE 10.68 vs.
10.51, RMSE 13.50 vs. 13.27, Brier 0.255 vs. 0.250. LightGBM: MAE 11.06 vs.
10.51 — a bigger gap than ridge's, the same pattern seen in the margin
models. Ensemble: essentially identical to ridge (10.67/13.47), since the
blend leans on it heavily. Flat-stake ROI at an untuned 1.5-point threshold:
ridge -1.8%, LightGBM ~0.0%, ensemble -2.1% — no real edge anywhere.

This feature set didn't include anything QB- or injury-specific (mixed
result when added, see the QB-injury item above) or a real pace measure —
**pace turned out to be the one that actually helped.**

**`off_plays_ewm_sum` (pace/plays-per-game) — kept for ridge and ensemble,
the first genuine improvement here since phase 1.** `off_plays` is a new
raw stat (`feature_engineering.compute_game_epa_splits`) — offensive snap
count per team per game — needing no new data ingestion at all, just an
extra aggregation over pbp data already cached. It flows through the
existing team-level rolling-form machinery exactly like the EPA features
do, so it's automatically leakage-safe and automatically populated for live
upcoming games too, with no bespoke carry-forward code needed (unlike every
QB-specific feature tried this session). Deliberately scoped to totals
only — LightGBM's margin feature list explicitly excludes it, since pace
explains combined scoring *opportunity*, not which team is better, and
confirmed byte-identical to baseline for all three margin models.

Measured directly: **ridge_total's MAE/RMSE/Brier all improved**
(10.678/13.494/0.2551 → 10.673/13.483/0.2549, ROI flat at -1.46%→-1.45%),
and **ensemble_total improved on all three too** (10.670/13.468/0.2560 →
10.663/13.456/0.2557, ROI -2.12%→-1.96%). LightGBM_total got worse on every
metric, so it's excluded there (confirmed byte-identical to its own
pre-pace baseline once removed) — same independent-feature-lists pattern
as `injury_diff`. Still nowhere close to beating the market outright, but
the first feature in this whole totals effort to move MAE, RMSE, *and*
Brier in the same direction for more than one model at once.

Not yet wired into Upcoming picks or the dashboard (still phase 1) — that
remains gated on the totals model actually beating the market, not just
narrowing the gap.

## Bet sizing and execution realism

Every backtest number above uses flat 1-unit stakes — necessary for testing
whether an edge exists, but not how anyone would actually bet. The **Bet
sizing** dashboard tab (`backtesting.simulate_bankroll`) instead sizes each
bet as a fraction of a *compounding* bankroll via the Kelly criterion
(fractional, with a hard cap — full Kelly is far too aggressive when the
probability estimate feeding it is uncertain, which the Diagnostics tab
already shows ours is). This is deliberately a risk illustration, not a
staking recommendation. At the pipeline's default thresholds, quarter-Kelly
(100 starting units):

| Model | Market | Final bankroll | Max drawdown |
|---|---|---|---|
| Ridge | ATS | 104.9 | -61.8% |
| Ridge | ML | 2.8 | -98.2% |
| LightGBM | ATS | 47.3 | -69.6% |
| LightGBM | ML | 1.0 | -99.3% |
| Ensemble | ATS | 251.8 | -51.8% |
| Ensemble | ML | 75.0 | -88.6% |

Even the best case (ensemble/ATS) carries a 51.8% max drawdown, and
ensemble/ML *loses* money under compounding despite a positive flat-stake
ROI — Kelly sizing amplifies a small, uncertain edge into large swings in
both directions. A backtest ROI near zero looks merely unexciting under flat
staking; under realistic compounding sizing, the same bet sequence makes the
"not tradeable as-is" conclusion much harder to miss.

**Does Kelly sizing actually beat flat staking? Measured directly — no.**
Comparing flat vs. Kelly final bankroll (both rescaled to start at 100):
Kelly **underperforms in 4 of the 6** model/market combinations above, worst
on the moneyline markets (loses an *additional* 25-97% versus what flat
staking alone would have lost/made). It clearly wins in exactly 2 cases,
both ATS. The split lines up with market structure: moneyline odds swing
far more between favorites and underdogs than ATS's near-uniform ~-110
pricing, so Kelly's stake size swings much harder there too — and when the
probability estimate behind that sizing is even slightly off (which the
Diagnostics tab already shows it is), compounding punishes that harder than
flat staking ever would.

Also worth knowing: Kelly compounding is **path-dependent** — the same
bets in a different order produce a different ending bankroll, so any
single realized path (every number above included) can look better or
worse than "typical" purely from the luck of ordering.
`backtesting.bootstrap_bankroll_paths` resamples the same placed-bet pool
with replacement 500 times, reruns the identical Kelly rule on each, and the
Bet sizing tab's chart plots the resulting 10th-90th percentile band next to
the one path that actually happened — plus a season-by-season breakdown of
the realized path, to see whether a good result is broad-based or carried
by one or two standout years.

**Calculating the optimal bet size and threshold, rather than assuming
them:**

- `backtesting.sweep_kelly_multiplier` sweeps the fractional-Kelly
  multiplier from 0.05–1.5 and measures mean log-growth (the actual
  quantity Kelly theory optimizes) across 300 bootstrap resamples at each
  value, alongside the probability a resampled path ever draws down more
  than 50%. For ensemble/ATS, growth is maximized around **1.0x** (mean
  log-growth 1.24 vs. 0.996 at the 0.25 default) — but at a **99.3%**
  chance of a 50%+ drawdown, vs. 74.3% at 0.25x. The growth-maximizing
  multiplier and a survivable one aren't the same number; the Bet sizing
  tab plots both curves with the current slider and the growth-optimal
  point both marked, so that trade-off is visible rather than assumed.
- `backtesting.optimal_threshold_walk_forward` grid-searches the
  bet-selection threshold each season using only strictly earlier seasons
  (selecting whichever threshold maximized the bootstrap 95% CI *lower
  bound* on ROI, not raw mean ROI), then applies that season's choice out
  of sample. For ensemble/ATS across 2017-2025, the adaptive threshold's
  ROI (-0.33%, 813 bets) edges out the fixed 1.5pt default (-0.67%, 1459
  bets) — but **the chosen threshold swings from 0.0 to 4.25 points
  season to season (std dev 1.46)**, which matters more than either ROI
  number: that much instability is itself evidence the optimizer is mostly
  chasing noise in a small per-season sample, not locking onto one real,
  stable threshold. Ensemble/ML's chosen threshold is far more stable (std
  dev 0.02, fixed -3.2% vs. adaptive -3.1%) — but the conclusion (still
  losing) doesn't change either way. See the Backtest slices tab.

## Upcoming picks (live predictions, not backtested)

Every section above is a backtest over completed history. The **Upcoming
picks** dashboard tab (`src/live_picks.py`, `modeling.predict_upcoming_*`) is
the live counterpart: it trains each model on every completed game to date
(no season held out — there's nothing left to validate a future game
against) and predicts games that don't have a result yet but already have a
posted market line, with the same probability/calibration/edge math as the
backtest, plus a Kelly-suggested stake. It is explicitly **not walk-forward
validated** — it inherits whatever skill (or lack of it) the Diagnostics tab
measured on completed games, nothing more, and is meant as one input
alongside your own judgment, not a signal to follow on its own.

To keep it current during the season, `main.py` needs a **fresh data pull**
— by default it reuses whatever's cached in `data/raw/`, which won't pick up
newly-posted lines, newly-final scores, or updated injury reports on its
own:

```powershell
.venv\Scripts\python.exe main.py --refresh
```

Also bumped `config.LAST_SEASON` to the current season (2026) so its
schedule — and, as they get posted, its market lines — gets pulled at all;
this needs bumping again each year. `cleaning.clean_schedules` was changed
to keep not-yet-played games (null scores) rather than dropping them, since
every backtest/walk-forward step already filters on `home_margin.is_not_null()`
itself — this doesn't change any backtest number, it just lets unplayed
games flow through to `live_picks`.

**This now runs on its own, automatically.** A Windows Scheduled Task
("NFL Model Weekly Refresh", set up by `scripts/setup_weekly_refresh.ps1`)
runs `scripts/weekly_refresh.ps1` every **Tuesday at 8:00 AM** — after
Monday Night Football has finished, so the prior week's results are final
and the next week's lines are usually posted. It calls `main.py --refresh`
and logs stdout/stderr to `logs/refresh_<timestamp>.{out,err}.log`. Useful
commands:

```powershell
Get-ScheduledTaskInfo -TaskName "NFL Model Weekly Refresh"   # next/last run time, last result
Start-ScheduledTask -TaskName "NFL Model Weekly Refresh"     # run it right now
Unregister-ScheduledTask -TaskName "NFL Model Weekly Refresh" -Confirm:$false   # remove it
```

Two things worth knowing about the data itself, discovered while wiring
this up: several nflreadpy loaders (`load_pbp`, `load_team_stats`,
`load_player_stats`, `load_injuries`, `load_snap_counts`) reject or 404 on
a season before that season's release files actually exist upstream —
`data_ingestion._load_seasons_safe` now catches this (both `ValueError` and
`ConnectionError`, since different loaders fail differently) and drops the
unavailable season rather than crashing the whole pipeline. `load_schedules`
itself is unaffected — nflverse publishes the full season's schedule
(including future weeks, without lines yet) well before those release
files exist.

**A permanent, append-only picks log.** `upcoming_picks.parquet` is
overwritten every run — once this week's games are played and the pipeline
moves on to next week, there'd be no record of what the model actually said
*before* a given game, only what a rerun backtest recomputes for it
afterward (using whatever model state exists at rerun time, not necessarily
what was on screen that week). Every `--refresh` run instead appends a
timestamped snapshot to `outputs/model_results/picks_log.parquet`
(`live_picks.append_to_picks_log`) — deliberately never deduplicated, since
a model's read on a game legitimately changing over the week (a new injury
designation, a line move) is worth keeping, not collapsing away. The
dashboard's **Track record** tab grades the latest logged snapshot of each
game against its real result once known, live at the sidebar's threshold —
this is the actual, un-replayable record of what was shown before kickoff,
which is what makes the earlier "treat 2026 as a genuine holdout" idea
checkable rather than just a good intention.

## Dashboard

```powershell
.venv\Scripts\streamlit run app\streamlit_app.py
```

Reads the parquet/CSV outputs `main.py` already wrote — it never recomputes
the model, but **bet-selection thresholds are re-graded live in the browser**
(every game's edge is already computed regardless of threshold, so dragging
the sidebar slider just redraws which games clear the bar — instant, no
pipeline rerun). Nine tabs:

- **Upcoming picks** — live predictions for games with no result yet (see
  "Upcoming picks" above); reacts to the same threshold slider plus its own
  Kelly-suggested stake sliders.
- **Track record** — the permanent picks log graded against real results as
  they come in; the actual, un-replayable record of what the model said
  before kickoff each week (see "A permanent, append-only picks log" above).
- **Overview** — KPIs, ROI by season, cumulative profit curve (with bootstrap CIs).
- **Backtest slices** — ROI by edge size, market-spread size, home/away,
  favorite/underdog, and team, each with 95% confidence intervals, restrictable
  to any season range.
- **Diagnostics** — model vs. market skill (MAE/RMSE/Brier, independent of any
  bet threshold) and a calibration/reliability curve.
- **Game explorer** — model-vs-market scatter for a season, a per-week game
  table, (ridge only) a per-game additive breakdown of the prediction, and a
  "games like this one" nearest-neighbor lookup (any model) showing what
  actually happened in the 10 historically most similar games.
- **Power ratings** — each team's Elo trajectory over time, plus a current
  leaderboard; defaults to the two teams the residual-bias check flagged as
  most mis-rated (NE, NYJ).
- **Bet sizing** — Kelly-criterion bankroll simulation vs. flat staking, with
  adjustable Kelly multiplier and per-bet cap, showing max drawdown alongside
  final bankroll.
- **Methodology** — data, pipeline, deep-dive findings, and known limitations.

Pick the model (ridge baseline / LightGBM), market (ATS / moneyline), and
bet-selection threshold from the sidebar — the whole dashboard reacts live.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Run

```powershell
.venv\Scripts\python main.py
```

This downloads/caches raw data to `data/raw/`, writes the cleaned game table
and modeling table to `data/processed/`, and writes predictions, per-bet
detail, and sliced summaries to `outputs/model_results/` and
`outputs/backtests/`.

## Key design decisions

- **Data source**: `nflreadpy.load_schedules()` already includes
  `home_moneyline`, `away_moneyline`, `home_spread_odds`, `away_spread_odds`,
  `spread_line`, `total_line`, `temp`, `wind`, `roof`, and neutral-site
  `location`, with full coverage from 2010 onward — a separate betting-lines
  or weather dataset (e.g. Spreadspoke) isn't needed for this scope.
- **Spread sign convention** (confirmed against real data, not assumed):
  `spread_line > 0` means the home team is favored by that many points;
  `spread_line < 0` means the away team is favored. Home covers when
  `home_margin > spread_line`.
- **Validation**: season-level walk-forward, not random k-fold — a game's
  prediction never depends on data from its own season's future games or
  later seasons.
- **Win probabilities**: derived from the *same* predicted margin (via a
  normal CDF using the model's residual std dev) rather than a separately
  trained classifier, keeping the spread and moneyline views of a game
  consistent with each other.
- **Weather/neutral-site features**: `temp`/`wind` are only meaningful for
  outdoor games — domes and closed roofs are filled with a fixed
  comfortable-conditions constant rather than the outdoor average, so a dome
  game isn't implied to have "average" wind.
- **"Games like this one" (`similarity.py`) is deliberately an interpretability
  tool, not a new predictive model.** A more flexible nonlinear model
  (LightGBM) already had its shot at finding an edge and still lost to the
  market on raw accuracy, so a k-NN predictor wasn't expected to do better —
  k-NN also degrades quickly with ~15 features and a few thousand rows (curse
  of dimensionality). It's used here only to answer "what happened in
  historically similar games," restricted to games strictly before the query
  game for the same walk-forward-consistency reason as everything else, even
  though it isn't part of the graded backtest.
- **Per-game contribution breakdown**: only implemented for the ridge model,
  since it's linear (`coefficient × feature value` sums exactly to the
  prediction). LightGBM's tree-based prediction doesn't decompose this simply.
- **LightGBM's rolling windows were pruned from [4, 8, 17] to just [8]** after
  its feature importances came back almost flat across ~20 features — it was
  spending splits distinguishing near-duplicate windows of the same stat
  rather than finding clean signal. Ridge is unaffected (it only ever read the
  EWMA versions, never the windowed ones).
- **A second, slower-decaying "stable" Elo track was tried and reverted.** It
  was meant to fix a real, diagnosed bias (the model systematically underrates
  perennially strong franchises and overrates chronically weak ones), but
  measured directly against that bias, it made it slightly worse — the fast
  and slow Elo tracks are 91% correlated, so lowering the regression rate
  doesn't cleanly separate "organizational persistence" from "current form."
  `power_ratings.compute_elo_ratings` still takes arbitrary k/regression/prefix
  if this is worth another attempt with a genuinely different signal.
- **Opponent-adjusted EPA was also tried and reverted.** Adjusted each game's
  raw offense/defense EPA by the specific opponent's incoming rolling quality
  before rolling it up (`feature_engineering.add_opponent_adjusted_form`,
  still present but unused by default) — aimed at the same team-bias pattern
  plus the divisional-game gap. Measured directly: barely moved either one
  (team bias 0.836→0.831 mean |bias|; divisional gap unchanged at -0.22/-0.51
  vs. market), and made LightGBM's aggregate MAE slightly worse. Likely
  because `elo_diff` already opponent-adjusts through its own update rule, and
  NFL schedules aren't lopsided enough within an 8-game window for a second
  adjustment to extract much more.

## Not yet built (deferred)

- Notebooks under `notebooks/` are minimal stubs for loading the parquet
  outputs, not polished analysis.
- Player props, live/in-game betting, and injury modeling are out of scope
  for v1 per the original outline.

## Repository structure

```text
nfl_model/
├── data/{raw,processed,outputs}/
├── notebooks/            # stubs
├── src/                  # pipeline modules
├── outputs/{charts,model_results,backtests}/
├── app/streamlit_app.py  # dashboard: run with `streamlit run app/streamlit_app.py`
├── program.md            # autonomous-research spec (see "Autoresearch" below)
├── scripts/
│   ├── autoresearch_eval.py  # fixed scoring harness -- do not modify mid-run
│   ├── plot_progress.py      # experiment progress chart -> progress.png
│   ├── setup_weekly_refresh.ps1
│   └── weekly_refresh.ps1
├── requirements.txt
└── main.py               # run this
```

## Autoresearch

`program.md` defines an autonomous experiment loop over this model's
feature engineering, hyperparameters, and bet-selection logic — the
protocol that produced item 18 above. An agent proposes a change, runs
`main.py`, scores it with a fixed read-only harness, logs the result, and
keeps or reverts it. To start a run:

```powershell
git checkout -b autoresearch/<tag>
# then point an agent at program.md and have it follow the loop
```

Scoring is deliberately split from the code under test:
`scripts/autoresearch_eval.py` reports the ensemble's development- and
holdout-era ATS/ML ROI (using the existing `config.HOLDOUT_FIRST_SEASON`
split) into `summary.txt`, and `scripts/plot_progress.py` renders
`progress.png` from the `results.tsv` experiment log. Both are declared
read-only during a run so the metric can't drift to meet the result.

The keep/discard gates are the important part, and are the way they are
because the first version of them failed (see item 18): a change must not
collapse the bet count, must improve development ROI, **and** must gain
more development ROI than it gives up in holdout ROI. Statistical
significance is reported but deliberately not gated on — at n≈700 bets the
bootstrap CI on ROI is roughly ±13pp, so requiring significance would halt
the search entirely. Individual keeps are provisional; only a consistent
direction across many experiments means anything.

## Later screens (market-residual tests, not built)

Screened against the market residual (result minus closing line) before building anything, as the `config.py` notes advise. None held up, so none were added to the model:

- **Schedule spots** (short week, bye, travel distance, time-zone shifts, third road game; 4,416 games): no effect distinguishable from noise after allowing for the number of variables tested. Only a 3+ time-zone gap reached |t| of about 2.5, with an implausible sign.
- **Garbage-time filtering** of EPA/success features (win probability outside 5-95%): dev ATS ROI +8.03% to +7.01%, MAE gap slightly worse. Reverted.
- **Neutral-situation pass rate**: no relationship to the margin or total residual.
- **New head coach** (first 8 games): about -0.9 points against the line (t about -1.9), shrinking in the holdout era. Not significant once the number of tests is counted; not built.
- **Offensive line**: already covered above (unit strength and unit-routed injuries both negative).
- **Line movement** would need an opening-line source; no free one with 2010+ coverage was found.
