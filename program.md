# NFL ATS/ML model autoresearch

This is an experiment to have the LLM autonomously improve the ensemble
model's ATS/moneyline betting edge, following this project's own existing
walk-forward + development/holdout discipline (see README.md and
src/config.py's history of kept/reverted features for the established
norms this loop should keep following).

## Background

Full context lives in README.md -- read it first. Short version: three
models (ridge, LightGBM, ensemble) predict NFL game margins from
leakage-safe rolling team/QB form, injuries, and Elo power ratings, convert
the predicted margin into ATS cover / moneyline win probabilities, and bet
flat 1-unit stakes above an edge threshold whenever the model disagrees
with the market. **The ensemble is the flagship model** (best MAE, only
model with positive flat-stake ROI in both markets) -- this loop optimizes
it specifically, though ridge and LightGBM still run every cycle as
comparison points.

Validation is season-level walk-forward (never train on a season's own
future or later seasons), split into a **development era** (seasons before
`config.HOLDOUT_FIRST_SEASON`, currently 2023) and a **holdout era**
(2023+). This split is not new for autoresearch -- it is this project's
existing rule: `src/evaluation.py`'s `summarize_by_era`/
`decision_is_significant` and `src/config.py`'s own comment history already
establish that every keep/revert decision must be made on the development
era only, and that a change should move ROI by more than the bootstrap 95%
CI width before being trusted. This loop inherits that rule rather than
inventing a new one.

## Setup

To set up a new experiment:

1. **Agree on a run tag**: propose a tag based on today's date (e.g.
   `aug18`). The branch `autoresearch/<tag>` must not already exist -- this
   is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the
   current branch tip.
3. **Read the in-scope files** (the repo is not tiny -- read selectively):
   - `README.md` -- full project context, and the "What's worked, and what
     hasn't" section specifically -- ~17 prior experiments, each with a
     measured result. Do not re-try something already tried and reverted
     without a genuinely different angle; if you do retry one, say why this
     attempt differs.
   - `src/config.py` -- feature lists, hyperparameters, thresholds, and a
     long inline comment history of what was tried. Read this fully before
     proposing a change; it is the project's own experiment log.
   - `src/evaluation.py` -- the fixed metric harness (read-only, see below).
   - `scripts/autoresearch_eval.py` -- the fixed autoresearch scorer
     (read-only, see below).
4. **Verify data exists**: check that `data/processed/modeling_table.parquet`
   exists (main.py's cached pipeline output). If not, run
   `.venv\Scripts\python.exe main.py --refresh` once to build it, and tell
   the human this took a fresh data pull.
5. **Initialize `results.tsv`**: already created with just the header row.
   The baseline gets recorded after the first run.
6. **Confirm and go**: confirm setup looks good before starting the loop.

Once confirmed, kick off the experimentation.

## Experimentation

**What you CAN modify:**
- `src/config.py` -- feature lists (`MARGIN_MODEL_FEATURES`,
  `TOTAL_MODEL_FEATURES`), hyperparameters, thresholds, rolling windows.
- `src/feature_engineering.py` -- new or changed features.
- `src/modeling.py` -- model structure, the ensemble blend, conditional-std.
- `src/probabilities.py` -- probability calibration, edge shrinkage.
- `src/backtesting.py` -- bet selection / sizing logic.
- `src/power_ratings.py` -- Elo mechanics.

**What you CANNOT modify:**
- `src/evaluation.py` -- the fixed metric harness (era split, bootstrap CI,
  significance test). Read-only.
- `scripts/autoresearch_eval.py` -- the fixed autoresearch scorer that
  reads the ensemble's backtest output and writes `summary.txt`. Read-only.
- `src/cleaning.py`, `src/data_ingestion.py`, `src/utils.py`, `main.py`,
  `app/streamlit_app.py` -- pipeline plumbing and the dashboard, not the
  model itself.
- `data/` -- raw and processed data are fixed inputs.
- Do not install new packages -- only use what `requirements.txt` already
  provides (polars, pandas, numpy, scikit-learn, lightgbm, streamlit, ...).

**The goal**: maximize development-era ensemble ATS ROI (see "Primary
metric" below), without silently degrading the holdout era or the ML
market. Given this project's own measured history (single-feature changes
move ROI by 0.4-1.6pp against a bootstrap CI roughly +/-4pp wide at this
sample size), do not expect most individual ideas to be a clear, real win
-- that is the honest baseline this project has already established, not a
sign something is broken.

**Simplicity criterion**: all else equal, simpler is better. A tiny ROI
improvement that adds real complexity (a new data join, a new hand-tuned
constant) is not obviously worth it, especially if it isn't statistically
distinguishable from noise. Removing something and getting an equal or
better result is a great outcome.

**The first run**: always establish the baseline first -- run the pipeline
as-is before making any change.

## Running an experiment

```powershell
# Step 1: run the full pipeline -- produces the ensemble's backtest parquet outputs
.venv\Scripts\python.exe main.py > run.log 2>&1

# Step 2: score it -- reads the ensemble outputs, writes summary.txt
.venv\Scripts\python.exe scripts\autoresearch_eval.py >> run.log 2>&1
```

Both steps must complete without error for a result to be valid. A full run
takes about **1m45s** with cached raw data (walk-forward refits 6 models
across ~15 seasons) -- not as fast as a single backtest script, but fast
enough to iterate many ideas in a session.

## Output format

`scripts/autoresearch_eval.py` writes `summary.txt`, e.g.:

```
Model:                  ensemble
Dev ATS ROI:            +0.0150
Dev ATS n_bets:         1842
Dev ATS 95% CI:         [-0.0180, +0.0480]
Holdout ATS ROI:        -0.0090
Holdout ATS n_bets:     299
Dev ML ROI:             -0.0300
Dev ML n_bets:          1120
Dev ML 95% CI:          [-0.0620, +0.0010]
Holdout ML ROI:         +0.0040
Holdout ML n_bets:      187
```

Extract the key metric:

```powershell
Select-String -Path summary.txt -Pattern '^Dev ATS ROI|^Dev ATS n_bets|^Holdout ATS ROI'
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT
comma-separated -- commas break in descriptions).

Columns: `commit  dev_ats_roi  dev_ats_n  holdout_ats_roi  status  description  notes`

- `commit`: git commit hash (short, 7 chars)
- `dev_ats_roi`: development-era ensemble ATS ROI (e.g. `0.0150`) -- `0.0000`
  for crashes
- `dev_ats_n`: number of development-era ATS bets placed -- `0` for crashes
- `holdout_ats_roi`: holdout-era ensemble ATS ROI, informational only, never
  used to decide keep/discard
- `status`: `keep`, `discard`, or `crash`
- `description`: short text description of what this experiment tried (no
  tabs)
- `notes`: optional -- dev ML ROI, CI width, or other observations (no tabs)

Example:

```
commit	dev_ats_roi	dev_ats_n	holdout_ats_roi	status	description	notes
a1b2c3d	0.0150	1842	-0.0090	keep	baseline	ml_roi -0.030 CI[-.018,+.048]
b2c3d4e	0.0210	1839	-0.0050	keep	add opponent pace-adjusted EPA	ml_roi -0.028
c3d4e5f	0.0090	1850	-0.0130	discard	5-game rolling window instead of EWMA	weaker signal
d4e5f6g	0.0000	0	0.0000	crash	QB CPOE feature (join key mismatch)	
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/aug18`).

**LOOP FOREVER:**

1. Look at the git state: the current branch/commit we're on.
2. Tune one of the modifiable files with an experimental idea by directly
   hacking the code.
3. `git commit`
4. Run the experiment:
   ```powershell
   .venv\Scripts\python.exe main.py > run.log 2>&1; .venv\Scripts\python.exe scripts\autoresearch_eval.py >> run.log 2>&1
   ```
5. Read out the results:
   ```powershell
   Select-String -Path run.log -Pattern '^Dev ATS|^Dev ML|^Holdout'
   ```
   If the output is empty or shows `CRASH`, the run failed. Run
   `Get-Content run.log -Tail 50` to read the traceback and attempt a fix.
   If you cannot fix it after a few attempts, skip and log as `crash`.
6. Record the results in `results.tsv` (do NOT `git commit results.tsv` --
   leave it untracked).
7. Regenerate the progress chart:
   ```powershell
   .venv\Scripts\python.exe scripts\plot_progress.py
   ```
   Reads every row logged in `results.tsv` so far and writes `progress.png`
   -- development-era ensemble ATS ROI per experiment, kept (green) vs.
   discarded/crashed (red), with a running-best step line. Run this after
   every `results.tsv` update, not just at the end.
8. Commit the refreshed artifacts: after each iteration, `git add
   summary.txt progress.png` and commit them (e.g. `git commit -m "outputs:
   refresh summary.txt and progress.png"`) so they reflect HEAD's run. Do
   NOT commit `data/`, `outputs/`, `run_log.txt`, `results.tsv`, or `logs/`
   -- these are already gitignored.
9. **Apply the keep/discard gates below, in order.** A change is kept only
   if it passes ALL THREE. Any failure means log the reason and
   `git reset --hard HEAD~1` (this also rolls back the summary/progress
   commit, which is fine).

   **Gate A -- bet count.** If development ATS `n_bets` drops by more than
   50% versus the baseline's, discard regardless of ROI -- a rule that only
   bets a tiny cherry-picked slice of games is not an edge, it's
   overfitting the selection rule. Do not even compare ROI.

   **Gate B -- development ROI improved.** Development ATS ROI must be
   strictly higher than the current baseline's.

   **Gate C -- net improvement across both eras.** The development ROI gain
   must be LARGER than any holdout ROI loss:
   `(dev_after - dev_before) > (holdout_before - holdout_after)`.
   A change that buys +0.2pp of development ROI by giving up 1.9pp of
   holdout ROI is a net loss and must be discarded.

   Gate C was originally written as a flat "holdout must not fall by more
   than 2.0pp" and that proved too loose in practice: EWMA 9->10 passed it
   at exactly -1.91pp holdout for a +0.24pp development gain -- inside the
   letter of the rule, obviously net-negative, and it had to be reverted by
   hand in a later audit. The net-improvement form above rejects that case
   directly and needs no magic constant.

   Gate C exists because the loop's own history proved Gates A+B alone are
   too permissive: an EWMA-halflife sweep (6->7->9->12->16) passed both at
   every step while development ROI climbed monotonically (+6.06% ->
   +8.53%) and holdout ROI fell monotonically (+7.36% -> +2.04%). Every
   step was individually "an improvement" and inside the ~13pp bootstrap
   CI, i.e. indistinguishable from noise. That is exactly the failure mode
   `src/config.py` documents at `HOLDOUT_FIRST_SEASON`, reproduced
   mechanically.

   **Honest cost of Gate C**: consulting the holdout partially contaminates
   it -- it stops being a fully untouched out-of-sample check and becomes a
   weak selection input. This is a deliberate trade: a *directional
   consistency* check is far weaker contamination than optimizing on it,
   and the alternative (no guard) demonstrably ratchets noise. Treat
   holdout numbers as a sanity check from here on, not as a clean
   out-of-sample result.

10. **Report significance, don't gate on it.** Note in `results.tsv`
    whether the development 95% CI excludes zero. Do NOT require
    statistical significance to keep a change: at n~800 bets the bootstrap
    CI on ROI is roughly +/-13pp, so essentially no single change can ever
    clear that bar, and gating on it would stop the search entirely. The
    honest framing is that individual keeps are provisional and only a
    consistent direction across many changes means anything.

**Timeout**: if either script hangs beyond 15 minutes, kill it and treat as
a crash.

**Crashes**: if the crash is a trivial bug (typo, wrong column name), fix
and re-run. If the idea is fundamentally broken (e.g. depends on data that
doesn't exist), log as `crash` and move on.

**NEVER STOP**: once the experiment loop has begun, do NOT pause to ask the
human if you should continue. Do NOT ask "should I keep going?" or "is this
a good stopping point?". You are autonomous. If you run out of ideas, think
harder:

- Re-read README.md's "What's worked, and what hasn't" for a genuinely
  different angle on something already tried (a different framing, not the
  same idea again).
- Try feature interactions the current set doesn't capture (e.g. a
  rest-mismatch x divisional-game interaction).
- Try adjusting `config.ROLLING_WINDOWS`, `EWMA_HALFLIFE_GAMES`,
  `RIDGE_ALPHA_MARGIN`, or `LGBM_PARAMS` -- but read config.py's own
  hyperparameter-tuning writeup first (item 17 in the README): tuning
  against a point-accuracy metric alone previously destroyed ROI even
  though it improved MAE. If you retune anything, verify against ROI, not
  MAE, before keeping it.
- Try a different edge-shrinkage or calibration approach in
  `probabilities.py`.
- Try restricting bets further (e.g. a new context filter analogous to
  `ML_UNDERDOG_ONLY`).

The loop runs until the human interrupts you, period.

## In-scope ideas to try (non-exhaustive)

| Idea | What to change |
|---|---|
| Rest-mismatch x divisional interaction | new feature in `feature_engineering.py`, add to `MARGIN_MODEL_FEATURES` |
| Alternative EWMA halflife | `config.EWMA_HALFLIFE_GAMES` |
| Wider/narrower rolling window | `config.ROLLING_WINDOWS` |
| Re-include LightGBM in the ensemble with a regularized weight | `config.ENSEMBLE_INCLUDE_LGBM`, `modeling.walk_forward_ensemble` |
| Tighter/looser adaptive ATS threshold | `backtesting.adaptive_ats_threshold`, `config.SHRINKAGE_SLOPE_FLOOR/CAP` |
| Home-favorite / road-underdog specific filter | `backtesting.py` bet-side logic, analogous to `ML_UNDERDOG_ONLY` |
| Divisional-game-specific edge threshold | `backtesting.py`, using existing `div_game` feature |
| Different conditional-std feature set | `modeling._std_feature_frame` |
| Playoff-specific handling (currently pooled with regular season) | `backtesting.py` / `config.py` |

## File map

```
main.py                    — full pipeline entry point (do not modify)
src/config.py               — feature lists, hyperparameters, thresholds (agent modifies this)
src/feature_engineering.py  — feature construction (agent modifies this)
src/modeling.py             — model training, ensemble (agent modifies this)
src/probabilities.py        — calibration, edge shrinkage (agent modifies this)
src/backtesting.py          — bet selection, sizing (agent modifies this)
src/power_ratings.py        — Elo ratings (agent modifies this)
src/evaluation.py           — fixed metric harness (do not modify)
scripts/autoresearch_eval.py — fixed autoresearch scorer, writes summary.txt (do not modify)
scripts/plot_progress.py     — fixed progress-chart generator, writes progress.png (do not modify)
data/                        — raw + processed data (fixed, gitignored)
results.tsv                  — experiment log (gitignored, untracked)
summary.txt                  — output of autoresearch_eval.py (committed each iteration)
progress.png                 — output of plot_progress.py (committed each iteration)
run.log                      — output of main.py + autoresearch_eval.py (gitignored)
program.md                   — these instructions (human edits over time)
```

## Run 2 (branch `autoresearch/sep19`): accuracy & calibration track

The first run (item 18 in README.md) optimized development-era ATS ROI
alone. This run adds a second objective, at the human's request: make the
ensemble **more accurate and better calibrated against the market**, not
just more profitable on the backtest. The motivating measurement is the
edge-shrinkage slope of ~0.18: only about a fifth of the ensemble's
disagreement with the closing line is real, yet its reported margin, cover
probability and win probability all use the unshrunk prediction. That is
why the Diagnostics tab shows the model losing to the market on MAE/Brier
and why the reliability curve is off.

**Scorer.** `scripts/autoresearch_eval.py` was extended BEFORE this run
started and is frozen for its duration. Its first eleven lines are
byte-for-byte the original ROI report (every `Select-String` pattern above
still works). After them it reports, per era, the ensemble's MAE/RMSE vs
the closing line, Brier vs the de-vigged moneyline, Brier of the cover
probability, expected calibration error (ECE) and calibration slope of
both probabilities, and the median shrinkage slope / ATS threshold actually
used. The same numbers are written to `summary.json`.

**Keep rules for this run.** A change is kept if it passes EITHER track
below, subject to the common guards. Both tracks are evaluated on the
development era only; the holdout is a directional sanity check, exactly as
in Gate C.

Common guards (all must hold, no exceptions):
- **G1 (= Gate A):** development ATS `n_bets` >= 50% of the baseline's.
- **G2 (moneyline guard):** development ML `n_bets` >= 50% of the
  baseline's, unless the experiment itself changes ML bet selection on
  purpose -- in which case the new ML bet count and both eras' ML ROI must
  be recorded in `results.tsv` notes and judged explicitly.

ROI track (unchanged from above): Gate B (development ATS ROI strictly
higher) and Gate C (net improvement across both eras).

Accuracy track:
- development MAE gap vs market improves by at least 0.01 points, AND the
  development Brier-ML gap vs market does not get worse (a change that
  targets only one of the two must leave the other within 0.0005);
- AND holdout MAE gap does not get worse;
- AND development ATS ROI stays within 1.5pp of the baseline AND holdout
  ATS ROI within 2.0pp -- accuracy is never bought with ROI. README item 17
  (MAE-tuned hyperparameters destroyed ROI) is the failure this clause
  exists to stop.

When a change passes one track and narrowly fails the other, record both
verdicts in `results.tsv` and let the human in the loop decide; the notes
column must say why.

**Parallel protocol.** Experiments run in detached git worktrees
(sibling directories `nfl_wt/w0` .. `w5`), each with its own copy
of `data/raw`, driven by the main repo's `.venv\Scripts\python.exe` with
`OMP_NUM_THREADS=3` and `POLARS_MAX_THREADS=3`. Verified before the run: a
worktree run is byte-identical to the main-repo run for the ensemble's ATS
and ML bet tables. Each experiment: reset the worktree to the base commit,
make ONE change, commit it there, run `main.py` then the scorer, report
`summary.json` and the commit hash. Winners are cherry-picked onto
`autoresearch/sep19` in the main repo, one per round (greedy forward
selection); other passing candidates are re-tested on the new base in the
next round, since kept changes interact. `results.tsv` keeps its columns;
the notes column carries the accuracy numbers. The previous run's ledger is
preserved as `results_aug18.tsv`.

Everything else in this file applies unchanged.
