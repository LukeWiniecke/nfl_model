"""Live tab: in-game model win probability vs. the sportsbook line, per game.

Reads the snapshots written by live/watcher.py (or $NFL_LIVE_DIR) and rescored
on every refresh; never refits. Gap = model P(home) minus P(home) implied by the
posted spread. Display only, not a pick.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live.model import load_snapshots, pregame_prior, score_snapshots  # noqa: E402
from live.monitor import fmt_line  # noqa: E402

LIVE_DIR = Path(os.environ.get("NFL_LIVE_DIR", Path(__file__).resolve().parents[1] / "data/live"))
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"


@st.cache_data(ttl=10, show_spinner=False)
def _scored() -> pd.DataFrame:
    df = load_snapshots(LIVE_DIR)
    if df.empty:
        return df
    s = score_snapshots(df, pregame_prior(df)).sort_values("ts").reset_index(drop=True)
    s["play_n"] = s.groupby("game_id").cumcount()
    return s


def _chart(g: pd.DataFrame, home: str) -> go.Figure:
    x = 3600 - g.t  # elapsed game seconds
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=g.p_model, name="Model", line=dict(color=BLUE, width=2.5),
                             customdata=g.last_play.fillna(""), hovertemplate="%{y:.1%}<br>%{customdata}<extra>Model</extra>"))
    live = g[g.p_line.notna()]
    fig.add_trace(go.Scatter(x=3600 - live.t, y=live.p_line, name="Posted line", line=dict(color=ORANGE, width=2, dash="dot"),
                             hovertemplate="%{y:.1%}<extra>Line</extra>"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, width=1))
    for q in (900, 1800, 2700):
        fig.add_vline(x=q, line=dict(color="#e1e0d9", width=1))
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(tickformat=".0%", range=[0, 1], title=f"P({home} wins)"),
                      xaxis=dict(title="Game seconds elapsed", range=[0, max(3600, x.max())]),
                      legend=dict(orientation="h", y=1.1))
    return fig


def total_ladder(pmf_tot, cur_total: float, n: int = 4) -> pd.DataFrame:
    """Fair over/under prices for half-point totals around the model's 50/50 total."""
    import numpy as np
    from live.totals import RT
    from src.utils import prob_to_american
    P = np.asarray(pmf_tot, dtype=float)
    final = cur_total + RT
    cand = np.arange(int(cur_total), int(cur_total) + 80) + 0.5
    center = cand[np.argmin([abs(float(P[final > k].sum()) - 0.5) for k in cand])]
    rows = []
    for k in center + np.arange(-n, n + 1):
        po = min(max(float(P[final > k].sum()), 0.001), 0.999)
        rows.append({"Total": k, "P(over)": po, "Fair over": int(prob_to_american(np.array([po]))[0]),
                     "P(under)": 1 - po, "Fair under": int(prob_to_american(np.array([1 - po]))[0])})
    return pd.DataFrame(rows)


@st.fragment(run_every=15)
def _body() -> None:
    s = _scored()
    if s.empty:
        st.info(f"No snapshots yet in {LIVE_DIR}. The watcher starts recording ~20 minutes before kickoff.")
        return
    latest = s.groupby("game_id").tail(1).copy()
    latest["o"] = latest.state.map({"in": 0, "pre": 1, "post": 2})
    latest = latest.sort_values(["o", "ts"])
    labels = {r.game_id: f"{r.away} @ {r.home}  ({'LIVE' if r.state == 'in' else r.state.upper()})" for r in latest.itertuples()}
    gid = st.selectbox("Game", list(labels), format_func=labels.get, key="live_game")

    # overview strip
    cols = st.columns(min(len(latest), 4) or 1)
    for i, r in enumerate(latest[latest.state == "in"].head(4).itertuples()):
        cols[i].metric(f"{r.away} @ {r.home}", f"{int(r.away_score)}-{int(r.home_score)}",
                       f"gap {r.gap:+.1%}" if pd.notna(r.gap) else None, delta_color="off")

    g = s[s.game_id == gid]
    r = g.iloc[-1]
    st.subheader(f"{r.away} {int(r.away_score)} @ {r.home} {int(r.home_score)}")
    live = r.state == "in"
    st.caption(f"Q{int(r.period)} {r.clock_display}" if live else {"pre": "Not started", "post": "Final"}.get(r.state, r.state))
    if live:
        sit = f"{r.possession} ball" + ("" if pd.isna(r.down) else f", {int(r.down)} & {int(r.distance)}")
        st.write(sit + (f" - {r.last_play}" if isinstance(r.last_play, str) else ""))

    c = st.columns(5)
    c[0].metric(f"Model P({r.home})", f"{r.p_model:.1%}")
    c[1].metric("Line-implied", f"{r.p_line:.1%}" if live and pd.notna(r.p_line) else "-")
    c[2].metric("Gap (model - line)", f"{r.gap:+.1%}" if live and pd.notna(r.gap) else "-")
    c[3].metric("Model fair line", fmt_line(r.fair_home_margin, r.home, r.away) if live else "-")
    c[4].metric("Posted line", fmt_line(r.line_margin, r.home, r.away))
    st.plotly_chart(_chart(g, r.home), width="stretch")

    if live:
        from live.dist import ladder

        st.markdown("**Fair live spread ladder** (model's full margin distribution; no vig)")
        lad = ladder(r.pmf, r.lead, r.home, r.away)
        st.dataframe(lad.style.format({"P(cover)": "{:.1%}", "P(cover) ": "{:.1%}", "Fair odds": "{:+d}", "Fair odds ": "{:+d}"}),
                     hide_index=True, width="stretch")
        st.caption("Compare a book's live spread price with the fair odds on the same line. Prices account for key "
                   "numbers (3, 7) and game script; they do not know about injuries or anything not in the game state.")

        st.markdown(f"**Fair live total ladder** (model's remaining-points distribution; no vig) - expected final total "
                    f"{r.exp_total:.1f}")
        st.dataframe(total_ladder(r.pmf_tot, r.home_score + r.away_score).style.format(
            {"P(over)": "{:.1%}", "P(under)": "{:.1%}", "Fair over": "{:+d}", "Fair under": "{:+d}"}),
            hide_index=True, width="stretch")
        wx = ("indoors" if r.indoor == 1 else f"{r.temp_f:.0f}F, wind {r.wind_mph:.0f} mph"
              + (", rain" if r.rain == 1 else "") + (", snow" if r.snow == 1 else ""))
        st.caption(f"Kickoff weather used by the totals model: {wx} ({r.weather_source}).")

    # Moves of the posted line during play tell us whether it is really live.
    n_lines = g.line_margin.nunique()
    if live and n_lines <= 1:
        st.warning("Posted spread has not changed since kickoff. ESPN may be showing the pregame line, in which "
                   "case 'gap' compares the model with a stale number.")

    with st.expander("Verify inputs (raw feed vs. what the model sees)"):
        cols_raw = ["ts", "period", "clock_display", "possession", "down", "distance", "yard_line_text", "ydl",
                    "home_timeouts", "away_timeouts", "spread_details", "spread", "spread_prior", "line_margin", "t"]
        st.dataframe(g[cols_raw].tail(8).iloc[::-1], hide_index=True)
        miss = [c for c in ("yard_line_text", "home_timeouts", "away_timeouts") if live and g[c].tail(20).isna().all()]
        if miss:
            st.error(f"Feed is not providing: {', '.join(miss)}. The model fills these with defaults (timeouts = 3, "
                     "field position = missing), so late-game probabilities are less reliable.")


def render() -> None:
    st.caption("Model vs. posted line, rescored every 15 seconds from recorded snapshots. "
               "Line-implied probability is derived from the current spread; it is not the book's moneyline.")
    _body()
