"""Game weather at kickoff, from nflverse (public data): roof type, temperature and wind from the
schedule, rain/snow parsed from the play-by-play weather description.

    game_weather() -> DataFrame game_id, indoor, temp_f, wind_mph, rain, snow

Indoor games (dome / closed roof) get temp 70, wind 0 and no precipitation. Live, the ESPN scoreboard
gives indoor, temperature and a conditions text (rain/snow) but NOT wind speed, so live_conditions()
takes kickoff-hour weather from Open-Meteo, calibrated to nflverse's scale (see WIND_A / WET_MM).
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


# ---------------------------------------------------------------------------------------------
# Live / backfill conditions from Open-Meteo (free, no key). Only a city's coordinates and a date
# are sent. Results are cached in data/live/weather_cache.json (one call per stadium per day).
# ---------------------------------------------------------------------------------------------

import json
import time
from datetime import datetime, timezone

import requests

CACHE = ROOT / "data/live/weather_cache.json"
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,wind_speed_10m,precipitation,weather_code"
RAIN_CODES = set(range(51, 68)) | set(range(80, 83)) | set(range(95, 100))   # drizzle, rain, showers, storms
SNOW_CODES = set(range(71, 78)) | {85, 86}
# Calibrated on 363 outdoor home games 2024-25 against nflverse's recorded kickoff weather:
#  - wind: Open-Meteo 10 m wind reads low in windy games (corr 0.76); nflverse ~= 2.54 + 0.75 x Open-Meteo
#  - temperature: corr 0.99, used as is
#  - wet: kickoff-hour precipitation >= 0.3 mm matches nflverse's rain/snow best (95% agreement; weather
#    codes alone flag light drizzle and gave 44 false alarms in 363 games)
WIND_A, WIND_B, WET_MM = 2.54, 0.75, 0.3
# Home cities (stadium locations) for teams; neutral-site games use ESPN's venue city instead.
TEAM_CITY = {"ARI": ("Glendale", "Arizona"), "ATL": ("Atlanta", "Georgia"), "BAL": ("Baltimore", "Maryland"),
             "BUF": ("Orchard Park", "New York"), "CAR": ("Charlotte", "North Carolina"), "CHI": ("Chicago", "Illinois"),
             "CIN": ("Cincinnati", "Ohio"), "CLE": ("Cleveland", "Ohio"), "DAL": ("Arlington", "Texas"),
             "DEN": ("Denver", "Colorado"), "DET": ("Detroit", "Michigan"), "GB": ("Green Bay", "Wisconsin"),
             "HOU": ("Houston", "Texas"), "IND": ("Indianapolis", "Indiana"), "JAX": ("Jacksonville", "Florida"),
             "KC": ("Kansas City", "Missouri"), "LV": ("Paradise", "Nevada"), "LAC": ("Inglewood", "California"),
             "LA": ("Inglewood", "California"), "LAR": ("Inglewood", "California"), "MIA": ("Miami Gardens", "Florida"),
             "MIN": ("Minneapolis", "Minnesota"), "NE": ("Foxborough", "Massachusetts"), "NO": ("New Orleans", "Louisiana"),
             "NYG": ("East Rutherford", "New Jersey"), "NYJ": ("East Rutherford", "New Jersey"),
             "PHI": ("Philadelphia", "Pennsylvania"), "PIT": ("Pittsburgh", "Pennsylvania"), "SF": ("Santa Clara", "California"),
             "SEA": ("Seattle", "Washington"), "TB": ("Tampa", "Florida"), "TEN": ("Nashville", "Tennessee"),
             "WAS": ("Landover", "Maryland"), "WSH": ("Landover", "Maryland")}
STATE = {"AZ": "Arizona", "GA": "Georgia", "MD": "Maryland", "NY": "New York", "NC": "North Carolina", "IL": "Illinois",
         "OH": "Ohio", "TX": "Texas", "CO": "Colorado", "MI": "Michigan", "WI": "Wisconsin", "IN": "Indiana",
         "FL": "Florida", "MO": "Missouri", "NV": "Nevada", "CA": "California", "MN": "Minnesota", "MA": "Massachusetts",
         "LA": "Louisiana", "NJ": "New Jersey", "PA": "Pennsylvania", "WA": "Washington", "TN": "Tennessee"}


def _cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {"geo": {}, "hourly": {}}


def _save(c: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c))


def _get(url: str, params: dict, tries: int = 3) -> dict:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(1 + i)


def geocode(city: str, state: str | None = None, country: str | None = None) -> tuple[float, float] | None:
    """City -> (lat, lon), preferring a match on state (US full name or abbreviation) / country."""
    state = STATE.get(state, state)
    key = f"{city}|{state}|{country}"
    c = _cache()
    if key in c["geo"]:
        return tuple(c["geo"][key]) if c["geo"][key] else None
    res = _get(GEO_URL, {"name": city, "count": 10, "language": "en", "format": "json"}).get("results", [])
    pick = [r for r in res if (not state or r.get("admin1") == state) and (not country or r.get("country") == country
                                                                           or r.get("country_code") == country)]
    hit = (pick or res or [None])[0]
    c["geo"][key] = [hit["latitude"], hit["longitude"]] if hit else None
    _save(c)
    return tuple(c["geo"][key]) if hit else None


def conditions_at(lat: float, lon: float, when_utc: datetime) -> dict | None:
    """Temperature (F), wind (mph) and rain/snow flags for the hour nearest `when_utc`."""
    when_utc = when_utc.astimezone(timezone.utc)
    day = when_utc.strftime("%Y-%m-%d")
    key = f"{lat:.3f},{lon:.3f},{day}"
    c = _cache()
    now = datetime.now(timezone.utc)
    days_ago = (now - when_utc).days
    stale = days_ago < 2 and now.timestamp() - c["hourly"].get(key, {}).get("_fetched", 0) > 1800
    if key not in c["hourly"] or stale:   # recent days: forecast may be revised; refresh at most every 30 min
        common = {"latitude": lat, "longitude": lon, "hourly": HOURLY, "wind_speed_unit": "mph",
                  "temperature_unit": "fahrenheit", "timezone": "UTC"}
        if days_ago > 6:
            h = _get(ARCHIVE_URL, dict(common, start_date=day, end_date=day))["hourly"]
        else:
            h = _get(FORECAST_URL, dict(common, past_days=7, forecast_days=16))["hourly"]
            keep = [i for i, t in enumerate(h["time"]) if t.startswith(day)]
            h = {k: [v[i] for i in keep] for k, v in h.items()}
        c["hourly"][key] = dict(h, _fetched=now.timestamp())
        _save(c)
    h = c["hourly"][key]
    if not h.get("time"):
        return None
    i = min(range(len(h["time"])), key=lambda j: abs(datetime.fromisoformat(h["time"][j]).replace(tzinfo=timezone.utc) - when_utc))
    code, temp, precip = h["weather_code"][i], h["temperature_2m"][i], h["precipitation"][i] or 0.0
    wet = precip >= WET_MM
    snow = wet and (code in SNOW_CODES or temp <= 32)
    return {"temp_f": temp, "wind_mph_raw": h["wind_speed_10m"][i], "wind_mph": WIND_A + WIND_B * h["wind_speed_10m"][i],
            "precip_mm": precip, "rain": float(wet and not snow), "snow": float(snow)}


def live_conditions(kickoff_utc: datetime, indoor: bool | None = None, home_team: str | None = None,
                    city: str | None = None, state: str | None = None, country: str | None = None) -> dict:
    """Model-ready kickoff weather (indoor, temp_f, wind_mph, rain, snow) for a live game.
    Location: ESPN venue city/state when given (neutral sites, international games), else the home
    team's city. Indoor venues get the indoor defaults. Falls back to typical-outdoor DEFAULTS if the
    location or the weather service is unavailable."""
    if indoor:
        return {"indoor": 1.0, "temp_f": 70.0, "wind_mph": 0.0, "rain": 0.0, "snow": 0.0, "source": "indoor venue"}
    try:
        if city:
            ll = geocode(city, state, country if country not in (None, "USA", "US") else None)
        elif home_team in TEAM_CITY:
            ll = geocode(*TEAM_CITY[home_team])
        else:
            ll = None
        c = conditions_at(*ll, kickoff_utc) if ll else None
    except Exception:
        c = None
    if not c:
        return dict(DEFAULTS, source="defaults (weather unavailable)")
    return {"indoor": 0.0, "temp_f": c["temp_f"], "wind_mph": c["wind_mph"], "rain": c["rain"], "snow": c["snow"],
            "source": f"open-meteo ({c['wind_mph_raw']:.0f} mph raw wind, {c['precip_mm']:.1f} mm/h)"}
