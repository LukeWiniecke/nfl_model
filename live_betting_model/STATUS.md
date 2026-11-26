# Live model status (update as things change)

Last updated: 2026-11-26

## v2 win-prob model (2026-11-26): walk-forward Brier 2021-2026 (lower better)
| Time left | Brownian | v1 blend | **v2** | nflfastR wp | vegas_wp |
|---|---|---|---|---|---|
| Q1 | 0.2068 | 0.2034 | **0.2019** | 0.2334 | 0.2015 |
| Q2 | 0.1797 | 0.1751 | **0.1726** | 0.1928 | 0.1725 |
| Q3 | 0.1450 | 0.1412 | **0.1409** | 0.1492 | 0.1402 |
| Q4 early | 0.1153 | 0.1114 | **0.1102** | 0.1127 | 0.1091 |
| Last 5 min | 0.1055 | 0.0881 | **0.0846** | 0.0850 | 0.0825 |
| All | 0.1588 | 0.1533 | **0.1516** | 0.1665 | 0.1509 |

Gap to vegas_wp: 0.0024 -> 0.0007. What v2 changed:
- **2nd-half kickoff receiver** as a feature (an extra possession; nflfastR has it, v1 didn't). Helped Q1-Q2.
- **Possession-perspective classifier** with nflfastR-style inputs (spread fading as exp(-4 x elapsed), score differential scaled up as time runs out, monotone in down / distance / field position / timeouts), handed over linearly from kickoff to halftime from the remaining-margin regressor. Helped everything after halftime, most in the last 5 minutes.
- The v1 late-game classifier never gets weight in this handoff, so it was dropped.

Tried on held-out 2022-25 (train <= 2021) and NOT adopted:
| Idea | Result |
|---|---|
| Extra inputs in the possession model (EP, total, spread) | worse (0.1544 -> 0.1557) |
| Training on 1999-2021 instead of 2010-2021 | late game better, first half worse (era drift); net worse; recency weighting didn't fix it |
| Late-game features (clock the leader can burn, scores needed, possessions left, overtime flag) | ~0 (0.1539 -> 0.1538) |
| Dedicated late-game model (last 20 min, smaller leaves) | slightly worse |
| Calibration layer per time bucket (Platt / isotonic, fit 2018-21) | worse: home-field advantage drifted, so the correction didn't transfer |
| Different early/late handoff windows | no effect |
| In-game EPA so far | no effect |
Reference only: averaging v2 with vegas_wp beats vegas_wp alone (0.1534 vs 0.1535 held out), so v2 carries some information nflfastR's model lacks.

## Older: walk-forward Brier after the leaf-size fix (2026-11-25)
| Time left | Brownian | Learned | Classifier | Blend | nflfastR wp | vegas_wp |
|---|---|---|---|---|---|---|
| Q1 | 0.2068 | 0.2034 | 0.2074 | 0.2034 | 0.2334 | 0.2015 |
| Q2 | 0.1797 | 0.1751 | 0.1783 | 0.1751 | 0.1928 | 0.1725 |
| Q3 | 0.1450 | 0.1412 | 0.1443 | 0.1412 | 0.1492 | 0.1402 |
| Q4 early | 0.1153 | 0.1108 | 0.1132 | 0.1114 | 0.1127 | 0.1091 |
| Last 5 min | 0.1055 | 0.0902 | 0.0881 | 0.0881 | 0.0850 | 0.0825 |
| All | 0.1588 | 0.1534 | 0.1561 | 0.1533 | 0.1665 | 0.1509 |

Test set now includes 2026 through week 11, so vegas_wp moved too; compare gaps: blend - vegas_wp = +0.0024 (was +0.0028).

## Leaf-size / memorisation fix (2026-11-25), held out 2022-2025 (train <= 2021)
- Old leaves (100-200 plays) could be a few games; the pregame total then moved one team's expected margin by 4.6 pts (DAL-SF, total 55.5).
- Win-prob blend: Brier 0.1561 -> 0.1554 (checkpoint rows 0.1618 -> 0.1613) with MIN_LEAF 3000 + monotone constraints. Dropping the total hurt (0.1562); nflfastR-style exponential spread decay added nothing.
- Margin distribution: cover log loss 0.5928 -> 0.5908, win Brier 0.1576 -> 0.1562 (min leaf 300 -> 3000).

## Totals v2 (2026-11-26): decisions and game script, held out 2022-25 (train <= 2021)
Diagnosis of v1 (mean final total minus expected, points): leader with the ball late -> model 0.3-0.6
too HIGH (clock-killing); slow-tempo games -> ~0.5 too high after halftime; first half 0.5-1.4 too low.
| Variant | CRPS | O/U log loss | Q1 | Q3 | Last 5 min | Late calibration slope |
|---|---|---|---|---|---|---|
| v1 | 5.034 | 0.5874 | **0.6563** | 0.6177 | 0.3366 | 1.00 |
| + game script / decision features | 5.040 | 0.5873 | 0.6604 | **0.6158** | 0.3272 | 1.00 |
| **v2 = + tempo (snaps per minute so far)** | **5.031** | **0.5865** | 0.6594 | 0.6164 | **0.3270** | 1.01 |
Decision features: offense-view margin, scores the trailer needs, garbage time, leader/trailer with the
ball late, two-minute drills, clock the leader can burn, both teams' timeouts, overtime, implied team
points from the pregame lines, the real scoring curve (end-of-half bumps). Live callers pass the number
of snaps so far from their game feed; otherwise it is estimated from the clock.
Rejected: more trees/leaves (overfits: 0.5980), dedicated late-game model (small leaves collapse with 76
outcome classes), in-game offensive efficiency (EPA/play, success rate: 0.5865 -> 0.5863, not worth it).
Takeaway: on history the game state explains only a little more of remaining scoring; the decision
effects are real but mostly matter in the last 5-10 minutes.

## Totals model v1 (`live/totals.py`), held out 2022-2025
- Remaining-points distribution vs a pace-based Gaussian: over/under log loss 0.5097 vs 0.5428 (8 lines around the expected total); final-total MAE 7.18 vs 7.32 pts.

## Live-market comparison
- Results that depend on confidential market data are kept in gitignored private notes, not here.

## Older: walk-forward Brier after the pre-play score fix (2026-11-15)
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
- 2026-11-24: weekly refresh failed (no network after wake, DNS errors); refresh script now retries 6x and refits the live models.
- 2026-11-26: totals v2 (game-script/decision features + tempo).
- 2026-11-26: v2 win-prob model (2nd-half kickoff receiver + possession classifier from halftime); live model refit.
- 2026-11-25: built `totals.py`. Found and fixed game memorisation in all LightGBM models (leaf size), refit live models. `score_states` split out of `score_snapshots` (identical output on 1,708 recorded snapshots). Checked in-game EPA features (held-out nflverse seasons): no gain.
