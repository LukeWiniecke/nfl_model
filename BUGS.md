# Bug ledger — autoresearch/sep19 session

Every defect found while researching the model on 2026-09-19/20, what it
affects, and its status. Kept current through the session; the newest
state is always at the top of each entry's status line.

Legend: **FIXED** (in this branch, verified) · **OPEN** (not yet fixed) ·
**EXPERIMENT-ONLY** (only bites in an experimental configuration, not in
the production model) · **LIMITATION** (known, not fixable with current data).

---

## 1. Team rolling form is wrong for games beyond the current week — FIXED (2afa00d, completed by f306990)

**Status:** fixed in two passes, because the first was only half of it.

`2afa00d` added `feature_engineering._carry_form_into_unplayed`, forward-filling
nulls within team into unplayed rows only. That repaired the `*_ewm` columns:
the six team-form features went from 16 of 31 upcoming games null to 0, and
0 nulls across all 255 unplayed 2026 games.

**An adversarial review then showed the `*_roll8` columns were untouched by
that fix, because they were never null.** `rolling_mean(min_samples=1)` skips
nulls inside its window instead of returning null, so a team's k-th unplayed
game averaged its last `window - k + 1` completed games: a sample that shrinks
week by week until it is a single game, and then sticks there. Measured drift
on `margin_roll8` against the team's true current form: 1.4 points at the next
game, 2.5 at the one after, 11.1 by the eighth. LightGBM consumes every
`*_roll8_diff` (`modeling.gbm_feature_columns`), so this reached the LightGBM
rows of the live board; ridge reads only the `_ewm` columns and was unaffected
by this second failure, and the ensemble excludes LightGBM.

`f306990` sources the carried value explicitly from each team's FIRST unplayed
game, the only scheduled row whose window is entirely completed games, then
forward-fills. Verified: every form column is now constant across each team's
unplayed rows with zero nulls (Kansas City holds -2.25 for every remaining
week, against -2.25 / -3.0 / ... / 21.0 before), the backtest is byte-identical
across 13 prediction, bet and contribution tables, and played rows of the
modeling table are unchanged.

**Lesson worth keeping:** the two rolling functions in `_rolling_exprs` have
opposite null semantics. Any future change there must check both.

**Where:** `feature_engineering._rolling_exprs` / `add_rolling_form`.
**What:** rolling form is `shift(1).over(team)` then `ewm_mean(min_samples=1)`.
An unplayed game has null raw stats, and polars' `ewm_mean` returns null at
every position whose shifted input is null, so a team's NEXT game has real
form and every game after it has none. `modeling._impute` then substitutes
the training median.
**Impact:** 6 of ridge's 15 features (all `*_ewm_diff` team-form features)
are median-imputed for next-week games. On the 2026-09-19 board that was
16 of 31 games and 4 of the 6 qualifying picks. Invisible in the backtest
because every historical game has been played. Same class as the QB
carry-forward bug in README item 12. The `*_roll8` versions survive because
`rolling_mean` skips nulls inside its window, so LightGBM degrades less.
**Fix:** carry each team's last real rolling-form value forward into unplayed
rows only. Validation: backtest output must stay byte-identical (no played
row may change) and live nulls must go to zero.

## 2. Dashboard threshold slider does not match the pipeline's bet book — FIXED (dba6ed2)

**Status:** sidebar "Threshold source" switch, default "Pipeline (per game,
adaptive)", regrades with the stored `ats_threshold_used` /
`ml_threshold_used` column and falls back to the fixed default where a
parquet has none; "Manual" keeps the old slider behaviour for what-ifs.
Verified headless with Streamlit's AppTest in ATS-pipeline, ATS-manual and
ML-pipeline modes: no exceptions.

**Where:** `app/streamlit_app.py` `regraded_bets()` and the two
`decide_bet_side` call sites in the Upcoming picks and Track record tabs.
**What:** every tab regrades bets at a single scalar threshold defaulting to
`config.ATS_EDGE_THRESHOLD_POINTS` / `ML_EDGE_THRESHOLD_PROB`, while the
pipeline places bets at a per-game adaptive threshold stored in
`ats_threshold_used` / `ml_threshold_used`. The dashboard therefore shows a
different (and differently sized) bet book than the validated backtest.
Gets far worse under any market-anchored ensemble, whose edges are ~4x
smaller than the 3.5-point slider default.
**Fix:** default the dashboard to the pipeline's own per-game threshold;
keep the slider as an explicit override.

## 3. Power ratings tab plotted future scheduled games as history — FIXED (b15520d)

**Where:** `app/streamlit_app.py` `load_elo_history` and the Power ratings tab.
**What:** `games.parquet` keeps unplayed games and Elo records a pre-game
rating for every scheduled game, so the chart drew a flat line to January
2027 and the leaderboard said "as of 2026 week 18" for all 32 teams. The
rating values were correct; only the presentation was wrong.
**Fix:** `played` flag on the history; chart ends on the current rating;
leaderboard labelled from the last completed game.

## 4. Weekly refresh scheduled task dies silently after wake-from-sleep — PARTIALLY FIXED

**Status:** task re-registered 2026-09-20 with `-WindowStyle Hidden` (the
console that could be closed no longer appears). The more robust fix, an
S4U principal so the task does not depend on the interactive session at
all, was refused with "Access is denied" from an unelevated shell;
`scripts/setup_weekly_refresh.ps1` now tries S4U first and falls back, so
rerunning it from an elevated PowerShell upgrades the task. A manual
`Start-ScheduledTask` test is scheduled after round 3 finishes (it rewrites
the outputs the reviewers are reading). Next scheduled run: Tue 2026-09-22
08:00. Root cause remains inferred, not proven.

**Where:** Windows task "NFL Model Weekly Refresh" (`scripts/setup_weekly_refresh.ps1`).
**What:** died 2026-08-18 and 2026-09-15 with result 0xC000013A
(STATUS_CONTROL_C_EXIT) about 7 s after launch, both times when the task
started late right after the laptop woke; log files 0 bytes. Root cause is
inferred, not proven: the task runs interactively with a visible console
that can be closed, and only while the user is logged on.
**Fix:** register the action with `-WindowStyle Hidden` and an S4U principal
so it does not depend on the interactive session; verify with a manual run.

## 11. Dashboard would read the wrong edge column under an anchored ensemble — OPEN (blocks adoption of E19)

**Where:** `app/streamlit_app.py` `regraded_bets` and the two `decide_bet_side`
call sites; `backtesting.regrade_ats` / `decide_bet_side` default to `ats_edge`.
**What:** the adopted candidate E19 reports the anchored margin but selects
bets on `ats_edge_select` / `ats_select_threshold_used`. The dashboard reads
`ats_edge` / `ats_threshold_used`, which under E19 are the anchored (roughly
4x smaller) display values. The reviewer measured it against the E19 parquet:
"Pipeline (per game)" mode yields 573 bets agreeing with the stored book on
only 2,987 of 3,312 rows, and the manual 3.5-point slider yields 4 bets.
**Not yet a live defect** — E19 is not merged, and against today's parquets the
dashboard is correct. Must be fixed in the same change that merges E19: prefer
the `_select` columns when present, fall back to the current ones.

## 12. Column semantics change silently under anchoring — OPEN (documentation, blocks adoption)

**What:** under E19, `ats_edge_shrinkage` (median 0.911 vs the baseline's
0.258) and `ats_edge_shrunk` describe shrinkage of an already-anchored edge,
and the scorer's reported shrink slope and threshold (0.921 / 1.14 against
0.258 / 3.50) are on the reported scale. Same names, different meaning, so
any comparison against a pre-anchoring run is invalid unless it is stated.
Needs a note in `config.py` and the dashboard's Methodology tab at merge time.

## 5. Adaptive moneyline threshold can go negative — EXPERIMENT-ONLY

**Where:** round-1 E6's `backtesting.adaptive_ml_threshold` and
`live_picks._live_ml_threshold`.
**What:** a plus-money-both-sides game (2020_01_LV_CAR, +134/-124) gives a
negative vig cushion, hence a negative bar, so a -0.009 edge was graded as
a bet against the model's own sign. Only exists in the adaptive-ML variants
(not in production). Fixed inside E10/E12 with `max(cushion, 0)`; any
adopted variant must carry that floor.

## 6. Raw-scale betting constants silently assume an unshrunk edge — EXPERIMENT-ONLY

**Where:** `config.ATS_EDGE_THRESHOLD_POINTS`, `ML_EDGE_THRESHOLD_PROB`,
`SHRINKAGE_SLOPE_CAP`; `backtesting.adaptive_ats_threshold`'s fallback.
**What:** under a market-anchored ensemble the edge is ~4x smaller, so the
3.5-point fallback (used for 2014-16, below `MIN_SHRINKAGE_GAMES`) becomes a
raw-equivalent 11-33 points and those seasons place 0 bets; the slope cap
of 1.0, documented "measured inert", binds in 2017-19. Harmless in the
production (unanchored) model; the reason round-2's anchored variants lost
bets. Round-3 E13 is the fix.

## 7. `is_dome` null for upcoming games at retractable-roof stadiums — FIXED (2afa00d)

**Where:** `feature_engineering.add_game_context_features`, from `games.roof`.
**What:** nflverse leaves `roof` null on future fixtures at retractable-roof
stadiums (ATL, HOU, ARI, DAL, IND), whose open/closed state is only
recorded once the game is played; 41 of 255 unplayed 2026 games, 6 of the
31 with a line. All five teams have been "closed" in effectively every
recent home game, so the median imputation (0, "outdoors") told the model
the opposite of the truth for exactly the teams that play indoors.
**Fix:** `_stadium_roof_by_home_team` fills from the team's most common
roof over its last three seasons of played home games; neutral-site games
stay null (2 remain, both neutral). Played rows untouched; backtest
byte-identical.

## 8. Injury features null for next-week games — LIMITATION

**Where:** `injury_diff` (19 of 31 upcoming games null on 2026-09-19).
**What:** next week's injury reports do not exist yet, so the feature is
median-imputed (approximately "no injuries"). Not a code defect; the data is
genuinely not knowable that early. The Tuesday refresh happens before that
week's reports, so the board's first snapshot of a week never has them.

## 9. Backtest uses closing lines, the live board sees early-week lines — LIMITATION

**What:** `spread_line` in history is the closing line; the Tuesday board
sees the opening/early line. Pre-existing and independent of any change
here, but a market-anchored ensemble makes 0.7-0.9 of the prediction BE
that line, so backtest and live differ by late-week line moves. Not fixable
without a line-history source.

## 10. Live fits use all history including the current season — LIMITATION (by design)

**What:** every live fit (calibration, shrinkage, anchoring, stacking) uses
all graded games including already-played games of the current season,
while the backtest for season S uses strictly earlier seasons. This is the
project's established live design (it is the correct "next season"
analogue) and was noted by every reviewer as pre-existing, not introduced.
