"""Game weather at kickoff, from nflverse (public data): roof type, temperature and wind from the
schedule, rain/snow parsed from the play-by-play weather description.

    game_weather() -> DataFrame game_id, indoor, temp_f, wind_mph, rain, snow

Indoor games (dome / closed roof) get temp 70, wind 0 and no precipitation. Live, the ESPN scoreboard
gives indoor, temperature and a conditions text (rain/snow) but NOT wind speed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TEXT = ROOT / "data/raw/nflverse/game_weather_text.parquet"
FEATS = ["indoor", "temp_f", "wind_mph", "rain", "snow"]
DEFAULTS = {"indoor": 0.0, "temp_f": 59.0, "wind_mph": 8.0, "rain": 0.0, "snow": 0.0}  # typical outdoor game


def _weather_text(force: bool = False) -> pd.DataFrame:
    if TEXT.exists() and not force:
        return pd.read_parquet(TEXT)
    import nflreadpy as nfl
    from src.data_ingestion import SEASONS
    frames = []
    for s in SEASONS:
        try:
            frames.append(nfl.load_pbp(seasons=[s]).select("game_id", "weather").drop_nulls().unique(subset="game_id"))
        except Exception as e:
            print(f"skip {s}: {e}")
    w = pl.concat(frames)
    w.write_parquet(TEXT)
    return w.to_pandas()


def game_weather() -> pd.DataFrame:
    sch = pl.read_parquet(ROOT / "data/raw/nflverse/schedules.parquet").select("game_id", "roof", "temp", "wind").to_pandas()
    w = sch.merge(_weather_text(), on="game_id", how="left")
    txt = w.weather.fillna("").str.lower()
    w["indoor"] = w.roof.isin(["dome", "closed"]).astype(float)
    w["temp_f"] = np.where(w.indoor == 1, 70.0, w.temp.fillna(DEFAULTS["temp_f"]))
    w["wind_mph"] = np.where(w.indoor == 1, 0.0, w.wind.fillna(DEFAULTS["wind_mph"]))
    w["rain"] = ((w.indoor == 0) & txt.str.contains("rain|shower|drizzle")).astype(float)
    w["snow"] = ((w.indoor == 0) & txt.str.contains("snow")).astype(float)
    return w[["game_id"] + FEATS]
