# Live model status (update as things change)

Last updated: 2026-11-15

## Walk-forward Brier, 2021-2026 (lower better), after the pre-play score fix
| Time left | Brownian | Learned | Classifier | Blend | nflfastR wp | vegas_wp |
|---|---|---|---|---|---|---|
| Q1 | 0.2072 | 0.2051 | 0.2100 | 0.2051 | 0.2337 | 0.2028 |
| Q2 | 0.1807 | 0.1777 | 0.1817 | 0.1777 | 0.1938 | 0.1743 |
| Q3 | 0.1461 | 0.1436 | 0.1484 | 0.1436 | 0.1505 | 0.1417 |
| Q4 early | 0.1164 | 0.1130 | 0.1161 | 0.1137 | 0.1142 | 0.1108 |
| Last 5 min | 0.1061 | 0.0934 | 0.0916 | 0.0916 | 0.0867 | 0.0858 |
| All | 0.1597 | 0.1558 | 0.1594 | 0.1557 | 0.1675 | 0.1527 |

Earlier numbers (blend 0.1554) were slightly flattered by the post-play score leak.

## Situation calibration, held out 2024-2026, possessing team's view (model / actual / vegas_wp)
- 4th down in FG range: 0.509 / 0.511 / 0.520 (fixed; was biased before)
- 1st down red zone: 0.558 / 0.576 / 0.586 (model too low)
- 1st down backed up (own 10 or worse): 0.527 / 0.518 / 0.513 (model too high)
- 4th down own territory (punt): 0.450 / 0.425 / 0.424 (model too high)

After adding the EP feature (`live/ep.py`, distilled from nflfastR `ep`, held-out RMSE 0.22 pts): 4th FG range 0.522 / 0.511, red zone 0.573 / 0.576, backed up 0.519 / 0.518, punt 0.438 / 0.425. Walk-forward blend Brier 0.1557 -> 0.1555 (last 5 min 0.0916 -> 0.0907). An offset variant (EP + spread as a fixed offset) calibrated punts slightly better (0.430) with the same Brier; not adopted.

## Margin-distribution model (`live/dist.py`), held out 2024-2026
Multiclass LightGBM over integer remaining margin (+/-28). Used for SPREAD pricing; win probability still comes from the blend.
- Cover probability on live-relevant lines (10 lines, 476k play-lines): Brier 0.2100 gaussian -> 0.2066 dist; log loss 0.6080 -> 0.5969. Better on 9 of 10 lines (-3.5 was 0.2122 vs 0.2141).
- Key numbers in Q4, P(|final| = 3): actual 0.163, dist 0.156, gaussian 0.069. P(|final| = 7): 0.090 / 0.081 / 0.051.
- Win-prob Brier: blend 0.1515, dist 0.1526 (so the blend stays the win-prob model). Expected-margin MAE identical (6.838).
- Single train/test split (train < 2024), not season walk-forward; walk-forward is slow (~75s per fit).

## Built and tested
- Poller/watcher, play-state table, baseline eval, final model, terminal monitor, Streamlit Live tab, replay tool. Tested on a replayed finished game (WSH @ NYG, event 401873052), not yet on a live game.

## Open verification (first live game: NE vs DET, Sun 2026-11-15 9:30am ET)
- [x] `situation` provides down/distance/possession/timeouts in-game (confirmed NE-DET). Between plays it sends down=-1, no possession, junk field position; `score_snapshots` marks these `stable=False` and averages over possession.
- [x] `possessionText` parses correctly. ESPN `situation.yardLine` = yards from the HOME goal line.
- [x] Posted spread: **ESPN drops odds from the scoreboard once the game starts** (all in-game rows null). No live line from this feed; `p_line`/gap are empty in-game. Need another odds source.
- [ ] Is the spread sign right for an away-favorite game (e.g. KC @ ATL)?
- [ ] Machine stayed awake through the game (see sleep note in CLAUDE.md).

## Log
- 2026-11-14: built everything above. Overnight the Mac slept (watcher log gaps 16:47->17:27, ..., 01:51->06:01); it woke ~08:08 CST on 11-15, before kickoff.
- 2026-11-15: first live game recorded. Found and fixed: (1) training used `total_home_score - total_away_score`, which is POST-play, so points scored on the play were already in the lead (FG value invisible; user spotted the 4th-down-in-FG-range bias). Now uses pre-play `score_differential`. (2) NaN possession was treated as a team in `yardline_100`. (3) between-play rows caused probability blips. Model refit.
- 2026-11-15: added EP feature (`ep_home`, `eff_lead`), refit model. Remaining bias: punting situations still ~1 pt too high for the punting team.
- 2026-11-15: added margin-distribution model and a fair live spread ladder in the Live tab (centred on the 50/50 line). Checked Q4 behaviour by score: model's mean remaining margin is within ~0.5 pt per lead bucket, but the gaussian sd was 10-25% too narrow late, which motivated this.
