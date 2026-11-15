"""Full-column play-by-play for the live model. The main pipeline's pbp cache
keeps only 18 columns, so this writes a separate state table (one row per play)
to data/raw/nflverse/pbp_states.parquet.

    .venv/bin/python -m live.states
"""

import polars as pl
import nflreadpy as nfl

from pathlib import Path

from src.data_ingestion import SEASONS

COLS = [
    "game_id", "season", "week", "season_type", "home_team", "away_team", "posteam", "defteam",
    "play_id", "qtr", "game_seconds_remaining", "half_seconds_remaining", "down", "ydstogo",
    "yardline_100", "score_differential", "total_home_score", "total_away_score",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "spread_line", "total_line",
    "result", "total", "epa", "ep", "wp", "vegas_wp", "play_type",
]


def build() -> pl.DataFrame:
    frames = []
    for s in SEASONS:
        try:
            df = nfl.load_pbp(seasons=[s])
        except Exception as e:
            print(f"skip {s}: {e}")
            continue
        df = df.select([c for c in COLS if c in df.columns]).filter(pl.col("game_seconds_remaining").is_not_null())
        frames.append(df)
    return pl.concat(frames, how="diagonal_relaxed")


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data/raw/nflverse/pbp_states.parquet"
    df = build()
    df.write_parquet(out)
    print(df.shape, "->", out)
