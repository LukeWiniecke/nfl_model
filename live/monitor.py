"""Terminal live view: one row per game, refreshed in place.

    .venv/bin/python -m live.monitor            # reads data/live (or $NFL_LIVE_DIR)
"""

import os
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.table import Table

from .model import load_snapshots, pregame_prior, score_snapshots

LIVE_DIR = Path(os.environ.get("NFL_LIVE_DIR", Path(__file__).resolve().parents[1] / "data/live"))


def latest_per_game() -> pd.DataFrame:
    df = load_snapshots(LIVE_DIR)
    if df.empty:
        return df
    scored = score_snapshots(df, pregame_prior(df))
    return scored.sort_values("ts").groupby("game_id").tail(1)


def fmt_line(margin: float, home: str, away: str) -> str:
    if pd.isna(margin):
        return "-"
    if abs(margin) < 0.05:
        return "EVEN"
    return f"{home if margin > 0 else away} -{abs(margin):.1f}"


def build_table(g: pd.DataFrame) -> Table:
    t = Table(title=f"NFL live  {time.strftime('%a %H:%M:%S')}", expand=True)
    for c, j in [("Game", "left"), ("Score", "center"), ("Clock", "center"), ("Situation", "left"),
                 ("Model P(home)", "right"), ("Line P(home)", "right"), ("Gap", "right"),
                 ("Fair line", "right"), ("Posted line", "right")]:
        t.add_column(c, justify=j)
    order = {"in": 0, "pre": 1, "post": 2}
    for _, r in g.assign(o=g.state.map(order)).sort_values(["o", "ts"]).iterrows():
        live = r.state == "in"
        sit = "" if not live else f"{r.possession or ''}{'' if pd.isna(r.down) else ' ' + f'{int(r.down)}&{int(r.distance)}'}".strip()
        gap = r.gap if live else float("nan")
        style = "bold green" if live and gap > 0.05 else "bold red" if live and gap < -0.05 else ("dim" if r.state == "post" else "")
        t.add_row(
            f"{r.away} @ {r.home}", f"{int(r.away_score)}-{int(r.home_score)}",
            f"Q{int(r.period)} {r.clock_display}" if live else {"pre": "pre", "post": "final"}.get(r.state, r.state), sit,
            f"{r.p_model:.1%}", f"{r.p_line:.1%}" if live and pd.notna(r.p_line) else "-",
            f"{gap:+.1%}" if live and pd.notna(gap) else "-",
            fmt_line(r.fair_home_margin, r.home, r.away) if live else "-",
            fmt_line(r.line_margin, r.home, r.away), style=style)
    return t


def main():
    console = Console()
    with Live(console=console, refresh_per_second=1, screen=False) as live:
        while True:
            try:
                g = latest_per_game()
                live.update(build_table(g) if len(g) else "Waiting for snapshots in " + str(LIVE_DIR))
            except Exception as e:
                live.update(f"error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
