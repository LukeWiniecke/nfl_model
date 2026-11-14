# NFL Spread, Moneyline and Totals Modeling

A research and backtesting project: can a football-fundamentals model find pricing inefficiencies against closing NFL lines, 2010 to the present? It also ships a Streamlit dashboard for exploring teams and games.

**Not betting advice.** The honest answer so far is "barely, if at all" (see [Results](#results)).

## What it does

- **Data.** Play-by-play, schedules, closing lines (spread, total, moneyline), player stats, snap counts and injury reports, all from [`nflreadpy`](https://github.com/nflverse/nflreadpy).
- **Features.** Leakage-safe rolling team form (EPA, success rate, pass/rush splits), starting-QB form, injury severity weighted by snap share, rest, weather context, and Elo power ratings. Every feature uses only information from before kickoff.
- **Models.** Ridge, LightGBM and a ridge/LightGBM ensemble predict the margin, validated season by season (to predict season S, train only on seasons before S). Separate models predict totals.
- **Probabilities.** Predicted margins become cover and win probabilities with a per-game uncertainty estimate and walk-forward calibration, then are compared with de-vigged market probabilities to get an edge.
- **Backtest.** Flat-stake ATS and moneyline bets above an edge threshold, with bootstrap confidence intervals and slices by season, team, edge size, favorite or underdog and spread size. A Kelly bankroll simulation shows the risk that flat staking hides.

## Results

- **The ensemble matches the closing line, it does not clearly beat it.** The prediction is anchored to the line: `pred = line + b * (model - line)`, with `b` fit on prior seasons. `b` comes out at about 0.1 to 0.35, so only a fifth to a third of the model's disagreement with the market survives.
- **Skill against the market (development era, ensemble):** MAE gap -0.001, RMSE gap -0.018, moneyline Brier gap -0.0006 (all vs. reading the closing line; negative is better).
- **Flat-stake backtest ROI (ensemble):** ATS +8.0% development (95% CI +1.1% to +15.2%, 692 bets) and +5.3% holdout (96 bets); moneyline +12.1% development (CI +0.2% to +24.6%) and +2.1% holdout. The intervals are wide and the holdout samples small, so this is "the most promising model so far," not a demonstrated edge.
- **Totals:** worse than the market on every metric.
- **Kelly sizing** amplifies a small, uncertain edge into large drawdowns (51.8% max drawdown for the best case), so it is a risk illustration, not a staking plan.
- **Most ideas did not work.** Opponent-adjusted EPA, referee tendencies, penalties, QB CPOE, unit-level injuries and strength, schedule spots, garbage-time filtering and hyperparameter tuning were all measured and reverted. An autonomous tuning loop ([`program.md`](program.md)) gained about 2.5pp of development ROI over 30 experiments, much of it likely overfit to the development era.

Full details of every experiment, kept or reverted, are in [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md). Known bugs and open questions are in [`BUGS.md`](BUGS.md).

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

Reads the files `main.py` writes and never refits the model. The bet-selection threshold, model and market are set live from the sidebar. Tabs:

| Tab | Purpose |
|---|---|
| Upcoming picks | Live predictions for games with a posted line (not backtested) |
| Matchup & trends | Two teams' form over time, adjusted for opponent, with coach and QB changes |
| League rankings | All 32 teams ranked on recent EPA metrics, with an offense-vs-defense chart |
| Track record | Logged pre-kickoff picks graded against real results |
| Overview, Backtest slices, Diagnostics | ROI, slices with confidence intervals, model vs. market skill and calibration |
| Game explorer | Per-game breakdown and a "games like this one" lookup |
| Power ratings | Elo history and per-team detail |
| Bet sizing | Kelly bankroll simulation |
| Methodology | Pipeline and limitations |

## Setup and run

Tested with Python 3.11.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
.venv/bin/python main.py                       # build features, train, backtest (uses cached data)
.venv/bin/python main.py --refresh             # pull fresh data first; run weekly in season
streamlit run app/streamlit_app.py
```

`main.py` caches raw data in `data/raw/`, writes the modeling table to `data/processed/`, and writes predictions and backtests to `outputs/`. Bump `config.LAST_SEASON` each year. `scripts/weekly_refresh.ps1` is a Windows Task Scheduler example for the weekly refresh.

## Design notes

- **Spread convention:** `spread_line > 0` means the home team is favored. Home covers when `home_margin > spread_line`.
- **Validation:** season-level walk-forward, split into a development era and a holdout era (2023 onward). A change is only kept if it improves the development era without giving back more in the holdout.
- **Probabilities** come from the same predicted margin (a Student's-t around it), so the spread and moneyline views of a game stay consistent.
- **Per-game contribution breakdown** exists for ridge only, since it is linear.

## Repository layout

```text
main.py            pipeline entry point
src/               ingestion, features, Elo, models, probabilities, backtesting, live picks
app/               Streamlit dashboard (streamlit_app.py, matchup_tab.py)
scripts/           experiment scoring and plotting, weekly refresh
docs/              full research log
program.md         spec for the autonomous experiment loop
BUGS.md            bug ledger and open questions
```

## Data

Football data and betting lines come from nflverse via `nflreadpy`; check their terms before redistributing any data. Nothing here is investment or betting advice.
