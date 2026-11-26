# Live in-game betting model: instructions for Claude

Referencing this folder means: you are working on the **in-game win-probability / live odds model** for the NFL project in the parent directory. Read this file and `STATUS.md` first. The pregame model (`src/`, `main.py`) is separate; see the repo `README.md` and `docs/RESEARCH_LOG.md` for it.

## Goal
Keep it simple: estimate how odds should shift as plays progress. Model P(home wins), the final-margin distribution (spreads) and the remaining-points distribution (totals) from the game state, compare them with the live market, and look for alpha. Research tool, **not betting advice**.

## Code map (all under `live/`, plus `app/live_tab.py`)
| File | Role |
|---|---|
| `espn.py` | Keyless ESPN client. `parse_event` flattens a scoreboard event into a snapshot row. |
| `poll.py` | `poll_once` appends changed rows to `data/live/snapshots_<date>.jsonl`. |
| `watcher.py` | Always-on supervisor. Idle polls every 300s, switches to 15s from 20 min before kickoff and while any game is live. |
| `states.py` | Builds `data/raw/nflverse/pbp_states.parquet` (793k plays, 28 cols; the main pipeline's pbp cache has only 18 and cannot be used). |
| `ep.py` | Expected points for the offense, distilled from nflfastR `ep` using only live-available inputs. Adds `ep_home`, `eff_lead`. |
| `dist.py` | Multiclass model of the full remaining-margin distribution; `prob_final_gt` gives cover probability for any line, `ladder` builds the fair spread ladder. Saved to `data/live/dist.joblib`. |
| `totals.py` | Multiclass model of the remaining total points (0-75) with game-script/decision features, tempo and a learned scoring curve (saved with the model); `prob_over` prices any total. Callers may pass `plays_so_far` (snaps with a down 1-4 so far); otherwise it is estimated from the clock. Saved to `data/live/totals.joblib`. |
| `baseline.py` | Walk-forward evaluation: Brownian, learned margin, direct classifier, blend vs nflfastR `wp`/`vegas_wp`. Run `python -m live.baseline`. |
| `model.py` | Final model fit on all seasons -> `data/live/model.joblib`; `score_snapshots` turns snapshot rows into probabilities. |
| `monitor.py` | Terminal table (`python -m live.monitor`). |
| `replay.py` | Turns a finished game into snapshots for testing without a live game. |
| `app/live_tab.py` | "Live" tab in the Streamlit dashboard. |

Run everything with `.venv/bin/python` from the repo root.

## Model
- Target: final home margin. Prior = **pregame** closing spread (never the live line; it leaks the score).
- Win probability (v2): at kickoff a LightGBM remaining-margin model (Normal around it); it hands over linearly by halftime to a LightGBM win classifier from the offense's point of view with nflfastR-style inputs (spread fading with time, score differential scaled up as time runs out, monotone in down/distance/field position/timeouts). Both know which team receives the 2nd-half kickoff (live: inferred from the opening possession; unknown -> averaged over both).
- Validation: walk-forward by season (train on earlier seasons), regular season only, 2021-2026 test. Never split by play; plays within a game are correlated.
- **Leaf size matters:** a game is ~150 plays, so small LightGBM leaves can be a handful of games, and game-constant inputs (pregame spread/total) let the trees memorise those games. Shared settings live in `baseline.py` (`MIN_LEAF` = 3000 plays for models with game-constant inputs; the possession classifier has none and uses `POS_LEAF` = 300); `dist.py` and `totals.py` use 3000. Re-check on held-out seasons when adding features.
- Brier by time left (lower better): see `STATUS.md` for the current table. Benchmark to beat is nflfastR `vegas_wp`; v2 is 0.0007 behind overall, level in the first half, behind mainly in the last 5 minutes. STATUS.md also lists what was tried and rejected; check it before re-trying an idea.
- Fair line = -(lead + predicted remaining margin), home-positive convention below.
- Spread prices come from `dist.py` (full distribution: key numbers, skew, late-game width), not from the Gaussian. Win probability and spread ladder are two models and can disagree slightly; that is known.

## Conventions and gotchas
- **Spread sign:** nflverse `spread_line > 0` means the home team is favored. ESPN's `details` is `"DET -2.5"` (favorite and size); `model.spread_line_from_details` converts. Do not trust the raw ESPN `spread` sign without checking.
- **Score columns:** nflverse `total_home_score`/`total_away_score` are POST-play; use pre-play `score_differential` (posteam view) for the lead. Using post-play scores hides the value of scoring position.
- ESPN removes odds from the scoreboard feed once a game kicks off, so there is no in-game line from it.
- `home_wp`/`home_vwp` in the eval are derived from posteam-based `wp`/`vegas_wp`.
- Field position is parsed from ESPN `possessionText` (`"NYG 25"`; own side means 100-N, opponent side means N). `situation.yardLine` is NOT used.
- `line_margin`/`p_line` use the *posted* spread as the market's expected final margin with a Brownian sigma. This is an approximation, not the book's price. The scoreboard feed has **no moneyline** (`pickcenter` in the per-game summary endpoint does).
- Timeouts missing from the feed default to 3 (the Live tab flags this).
- Local timezone of the machine is CST; kickoff times from ESPN are UTC.
- 2nd-half kickoff receiver = the team that kicked off to start the game (true in all 2,924 games checked). In the training data a kickoff row is down NaN, distance 0, yardline 35, timeouts 3.
- Backtesting: the saved live model is refit weekly on everything, so it has seen any past game. Fit a model with a cutoff before the test window.

## Automation (macOS launchd, nothing manual)
- `com.nflmodel.livewatcher` (KeepAlive, wrapped in `caffeinate -i`): runs `live.watcher`. Log: `logs/watcher.log`.
- `com.nflmodel.weeklyrefresh`: Tuesdays 09:00, runs `scripts/weekly_refresh_mac.sh` (main pipeline refresh + `live.states`).
- Plists live in `~/Library/LaunchAgents`. Check with `launchctl list | grep nflmodel`.
- **Known weakness:** `caffeinate -i` does not stop sleep when the lid is closed or on battery; the log showed multi-hour gaps overnight. Hosting on an always-on machine would fix this.
- Dashboard: `streamlit run app/streamlit_app.py` (port 8501). Streamlit caches imported modules, so after editing a tab module restart the server rather than relying on hot reload.

## Private material
- The repo is public. Some data, code and notes for this project are confidential and live only in gitignored paths (`live/private/`, `live_betting_model/private/`, `data/raw/`, `CLAUDE.local.md`). Read `CLAUDE.local.md` before working with them, never move anything from those paths into tracked files, and keep tracked files free of names, numbers or descriptions that come from them. A local pre-commit hook blocks this, but don't rely on it.

## How to work here
- Verify, don't assume. The ESPN live fields (timeouts, in-game spread movement, field-position text) were untested on a real in-progress game until the first live recording; check `STATUS.md` for what has been confirmed.
- Honest reporting: say plainly when the model trails `vegas_wp` or when a feed field is missing. Do not claim an edge.
- Prefer extending `live/` over touching `src/`. Keep new dependencies in `requirements.txt`.
- Do not commit or push unless asked. Do not write personal details into external requests.
- Test with `live.replay` + `NFL_LIVE_DIR=<dir>` (the monitor and tab both honor it) instead of waiting for games.

## Next ideas (in rough priority)
1. Close the late-game gap: explicit clock mechanics (two-score threshold, first downs to kill clock, timeouts) or a drive-level simulation.
2. Use the market as an input: a blend of model and market that is better than either, if the model ever proves to add information.
3. Live halftime receiver in `score_snapshots` (opening kicker rule) so halftime isn't averaged over possession.
4. Alerts when |gap| exceeds a threshold; always-on hosting (VM or Tailscale for access).
