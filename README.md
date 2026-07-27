# NFL ATS + Moneyline Modeling

Research/backtesting project. **Not betting advice.** Evaluates whether a
football-fundamentals model (EPA, success rate, power ratings, rest, etc.)
can identify historical pricing inefficiencies against closing NFL market
lines — spread and moneyline — from 2010 through the latest completed season.

## What it does

1. Pulls play-by-play, schedules, and closing market lines (spread, total,
   home/away moneyline, spread odds) from `nflreadpy` — one source covers
   both football data and betting lines, with 100% moneyline/odds coverage
   from the 2010 season on.
2. Builds leakage-safe rolling team-form features (EPA/play, success rate,
   pass/rush EPA, defensive EPA allowed) and Elo-style power ratings, where
   every feature for a game only uses information that existed strictly
   before that game was played.
3. Trains two staged margin-prediction models with season-level walk-forward
   validation (to predict season S, train only on seasons < S):
   - **Ridge baseline** — a small, interpretable feature set (power rating
     diff + a few recency-weighted EPA/success-rate diffs).
   - **LightGBM** — the full engineered feature set, as a richer comparison.
4. Converts predicted margins into ATS cover probabilities and moneyline win
   probabilities (normal approximation around the predicted margin), and
   compares them against de-vigged market probabilities to compute edge.
5. Backtests flat-stake ATS and moneyline bets above configurable edge
   thresholds, and reports ROI/win-rate sliced by season, team, edge size,
   home/away, favorite/underdog, and market-spread bucket.

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
  `spread_line`, and `total_line` with full coverage from 2010 onward, so a
  separate betting-lines dataset (e.g. Spreadspoke) isn't needed for this
  scope. It can be merged in later for extra weather/stadium detail.
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

## Not yet built (deferred)

- `app/streamlit_app.py` is a placeholder — the interactive dashboard is a
  follow-up phase.
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
├── app/streamlit_app.py  # placeholder
├── requirements.txt
└── main.py               # run this
```
