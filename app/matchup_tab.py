"""Matchup & trends tab: how two teams have progressed over time, adjusted for
who they played, with coach and quarterback context.

Display only -- the model's number is shown next to the market's as a
cross-check, not as a pick. All trend data comes from src/team_trends.py.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src import team_trends

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "#e1e0d9"
GRAY = "#898781"
LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color="#0b0b0b"),
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    margin=dict(l=10, r=10, t=50, b=10),
)


@st.cache_data
def _trends() -> pl.DataFrame:
    return team_trends.team_game_trends()


def _spread_text(home: str, away: str, spread_line: float) -> str:
    """spread_line is the home team's expected margin (positive = home favored)."""
    if spread_line > 0:
        return f"{home} -{spread_line:g}"
    if spread_line < 0:
        return f"{away} -{abs(spread_line):g}"
    return "pick'em"


def _fmt_margin(home: str, away: str, margin: float) -> str:
    return _spread_text(home, away, round(margin, 1))


def _trend_chart(
    games: pl.DataFrame, teams: list[tuple[str, str]], metric: str, adjusted: bool, smooth: int
) -> go.Figure:
    label, _, _, lower_better = team_trends.METRICS[metric]
    col = f"{metric}_adj" if adjusted else metric
    fig = go.Figure()
    for team, color in teams:
        t = games.filter(pl.col("team") == team).sort("gameday").to_pandas()
        if t.empty:
            continue
        roll = t[col].rolling(smooth, min_periods=1).mean()
        hover = (
            t["gameday"].astype(str) + " " + t["is_home"].map({True: "vs", False: "@"}) + " " + t["opponent"]
            + "<br>" + t["points_for"].astype(int).astype(str) + "-" + t["points_against"].astype(int).astype(str)
            + "<br>Coach: " + t["coach"].fillna("?") + "<br>QB: " + t["qb"].fillna("?")
        )
        fig.add_scatter(
            x=t["gameday"], y=t[col], mode="markers", name=f"{team} game",
            marker=dict(color=color, size=7, opacity=0.35), showlegend=False,
            customdata=hover, hovertemplate="%{customdata}<br>" + label + ": %{y:.3f}<extra>" + team + "</extra>",
        )
        fig.add_scatter(
            x=t["gameday"], y=roll, mode="lines", name=f"{team} ({smooth}-game avg)",
            line=dict(color=color, width=2.5), hovertemplate="%{y:.3f}<extra>" + team + " avg</extra>",
        )
        # Coach changes for this team inside the window.
        c = t[t["coach"].notna()]
        change = c[c["coach"] != c["coach"].shift(1)].iloc[1:]
        for _, r in change.iterrows():
            fig.add_vline(x=r["gameday"], line_dash="dot", line_color=color, line_width=1)
            fig.add_annotation(
                x=r["gameday"], y=1, yref="paper", text=f"{team}: {r['coach']}", showarrow=False,
                font=dict(size=10, color=color), textangle=-90, xanchor="right", yanchor="top",
            )
    fig.add_hline(y=0, line_width=1, line_color=GRID)
    direction = "lower is better" if lower_better else "higher is better"
    adj_note = ", adjusted for opponent" if adjusted else ""
    fig.update_layout(
        title=f"{label}{adj_note} ({direction})", height=420, legend=dict(orientation="h", y=-0.15), **LAYOUT
    )
    if lower_better:
        fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
    return fig


def _recent_table(games: pl.DataFrame, a: str, b: str, last_n: int) -> pd.DataFrame:
    rows = []
    for key, (label, _, _, lower) in team_trends.METRICS.items():
        row = {"Metric": label + (" (lower better)" if lower else "")}
        for team in (a, b):
            t = games.filter(pl.col("team") == team).sort("gameday").tail(last_n)
            season_all = games.filter(pl.col("team") == team)
            raw = t[key].mean()
            adj = t[f"{key}_adj"].mean()
            row[f"{team} last {last_n}"] = None if raw is None else round(raw, 3)
            row[f"{team} adj"] = None if adj is None else round(adj, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _team_context(games: pl.DataFrame, team: str, color: str) -> None:
    t = games.filter(pl.col("team") == team).sort("gameday")
    st.markdown(f"**{team}**")
    coaches = team_trends.stints(t, "coach")
    qbs = team_trends.stints(t, "qb")
    if coaches.height:
        st.caption("Head coach (consecutive stints in this window)")
        st.dataframe(coaches.to_pandas(), width="stretch", hide_index=True)
    if qbs.height:
        st.caption("Starting quarterback (consecutive stints)")
        st.dataframe(qbs.to_pandas(), width="stretch", hide_index=True)
    st.caption("Most recent games")
    last = t.tail(8).select(
        pl.col("gameday").alias("Date"),
        pl.when(pl.col("is_home")).then(pl.lit("vs")).otherwise(pl.lit("@")).alias(""),
        pl.col("opponent").alias("Opp"),
        pl.concat_str(pl.col("points_for").cast(pl.Int64), pl.lit("-"), pl.col("points_against").cast(pl.Int64)).alias("Score"),
        pl.col("margin").alias("Margin"),
        pl.col("spread_for_team").alias("Line"),
        pl.col("off_epa_adj").round(3).alias("Off EPA adj"),
        pl.col("def_epa_adj").round(3).alias("Def EPA adj"),
        pl.col("qb").alias("QB"),
    )
    st.dataframe(last.to_pandas(), width="stretch", hide_index=True)


def render(upcoming: pl.DataFrame | None, model_label: str) -> None:
    st.info(
        "Built for forming your own view, not for following the model. The model's number is shown "
        "beside the market line as a cross-check; the Diagnostics tab measures how little it adds. "
        "Rolling averages of 8 games or fewer are noisy -- the faint dots are the individual games."
    )
    games_all = _trends()

    # --- Choose the matchup ----------------------------------------------
    teams_all = sorted(games_all["team"].unique().to_list())
    game_row = None
    options = ["Pick any two teams"]
    labels: dict[str, dict] = {}
    if upcoming is not None and upcoming.height:
        for r in upcoming.sort(["week", "gameday", "game_id"]).to_dicts():
            lab = f"Wk {r['week']}: {r['away_team']} @ {r['home_team']}"
            labels[lab] = r
            options.append(lab)
    pick = st.selectbox("Matchup", options, index=1 if len(options) > 1 else 0, key="mt_game")
    if pick in labels:
        game_row = labels[pick]
        team_a, team_b = game_row["away_team"], game_row["home_team"]
    else:
        c1, c2 = st.columns(2)
        team_a = c1.selectbox("Team A", teams_all, index=teams_all.index("BUF") if "BUF" in teams_all else 0, key="mt_a")
        team_b = c2.selectbox("Team B", teams_all, index=teams_all.index("KC") if "KC" in teams_all else 1, key="mt_b")

    # --- Market vs model (cross-check) -----------------------------------
    if game_row is not None:
        home, away = game_row["home_team"], game_row["away_team"]
        spread = game_row["spread_line"]
        pred = game_row["predicted_margin"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Market line", _spread_text(home, away, spread))
        m2.metric(f"{model_label} number", _fmt_margin(home, away, pred))
        m3.metric(
            "Model vs. market", f"{abs(pred - spread):.1f} pts",
            help=f"Positive value means the model likes {home} more than the market does."
            if pred > spread else f"The model likes {away} more than the market does.",
        )
        total_line = game_row.get("total_line")
        m4.metric("Total", f"{total_line:g}" if total_line is not None else "--")

    # --- Controls --------------------------------------------------------
    seasons = sorted(games_all["season"].unique().to_list())
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    metric_keys = list(team_trends.METRICS)
    metric = c1.selectbox(
        "Metric", metric_keys, format_func=lambda k: team_trends.METRICS[k][0], key="mt_metric"
    )
    adjusted = c2.toggle(
        "Adjust for opponent", value=True, key="mt_adj", disabled=(metric == "margin"),
        help="Compares each game with how that opponent did in its other games that season. "
        "A good number against a strong defense counts for more than against a weak one.",
    )
    since = c3.selectbox("Since season", seasons, index=max(len(seasons) - 2, 0), key="mt_since")
    smooth = c4.slider("Rolling window (games)", 3, 12, 6, key="mt_smooth")

    games = games_all.filter(pl.col("season") >= since)
    fig = _trend_chart(games, [(team_a, ORANGE), (team_b, BLUE)], metric, adjusted and metric != "margin", smooth)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Dotted vertical lines mark head-coach changes. Hover a dot for the opponent, score, coach and "
        "quarterback. Opponent adjustment compares each game with that opponent's average in its OTHER "
        "games that season, so it uses the whole season and is a look-back, not a model input."
    )

    # --- Side by side ----------------------------------------------------
    st.subheader("Side by side")
    last_n = st.slider("Games to average", 3, 17, 8, key="mt_lastn")
    st.dataframe(_recent_table(games_all, team_a, team_b, last_n), width="stretch", hide_index=True)
    st.caption(
        "Raw columns are plain averages of the last games; 'adj' columns subtract what each opponent "
        "usually gives up (or produces) elsewhere. Defense EPA is points allowed per play in expected "
        "terms, so LOWER is better for the three defense rows."
    )

    # --- Context ---------------------------------------------------------
    st.subheader("Coaching, quarterback and recent games")
    ctx = games_all.filter(pl.col("season") >= max(since, seasons[-1] - 2))
    left, right = st.columns(2)
    with left:
        _team_context(ctx, team_a, ORANGE)
    with right:
        _team_context(ctx, team_b, BLUE)


def render_rankings() -> None:
    st.info(
        "Every team's recent form, ranked as of its most recent completed game. Rank 1 is best "
        "(lowest for the defense columns). Short windows are noisy -- an 8-game average of EPA can "
        "move a team many spots on one result."
    )
    games_all = _trends()
    seasons = sorted(games_all["season"].unique().to_list())
    c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
    last_n = c1.slider("Games to average", 3, 17, 8, key="lr_n")
    adjusted = c2.toggle("Adjust for opponent", value=True, key="lr_adj")
    this_season = c3.toggle("Current season only", value=False, key="lr_season",
                            help="Off: the last N games regardless of season, so early in a season "
                            "the window reaches back into last year.")

    g = games_all
    if this_season:
        g = g.filter(pl.col("season") == seasons[-1])
    keys = [k for k in team_trends.METRICS if k != "margin"]
    suffix = "_adj" if adjusted else ""
    recent = (
        g.sort(["team", "gameday"]).group_by("team", maintain_order=True).tail(last_n)
        .group_by("team")
        .agg(
            *[pl.col(f"{k}{suffix}").mean().alias(k) for k in keys],
            pl.col("margin").mean().alias("margin"),
            pl.len().alias("games"),
            pl.col("gameday").max().alias("through"),
        )
    ).to_pandas().set_index("team")

    metric_cols = keys + ["margin"]
    lower = {k: team_trends.METRICS[k][3] for k in metric_cols}
    ranks = pd.DataFrame({k: recent[k].rank(ascending=lower[k], method="min") for k in metric_cols})
    # Overall: mean rank across the six EPA metrics (point margin excluded, it is an outcome).
    recent["overall"] = ranks[keys].mean(axis=1)
    ranks["overall"] = recent["overall"].rank(method="min")

    sort_options = ["overall"] + metric_cols
    labels = {"overall": "Overall (mean rank of the six EPA metrics)", **{k: team_trends.METRICS[k][0] for k in metric_cols}}
    sort_by = c4.selectbox("Sort by", sort_options, format_func=lambda k: labels[k], key="lr_sort")
    order = ranks[sort_by].sort_values(kind="stable").index

    table = recent.loc[order, ["overall"] + metric_cols].copy()
    table.insert(0, "Rank", range(1, len(table) + 1))
    table.columns = ["Rank", "Overall rank score"] + [
        {"off_epa": "Off EPA", "off_pass_epa": "Pass off", "off_rush_epa": "Rush off",
         "def_epa": "Def EPA ↓", "def_pass_epa": "Pass def ↓", "def_rush_epa": "Rush def ↓",
         "margin": "Avg margin"}[k] for k in metric_cols
    ]
    gmap = ranks.loc[order, ["overall"] + metric_cols].copy()
    gmap.insert(0, "Rank", 0.0)
    gmap.columns = table.columns
    gmap["Rank"] = gmap["Overall rank score"]
    fmt = {c: "{:+.3f}" for c in table.columns if c not in ("Rank", "Overall rank score", "Avg margin")}
    fmt["Overall rank score"] = "{:.1f}"
    fmt["Avg margin"] = "{:+.1f}"
    styled = (
        table.style.format(fmt)
        .background_gradient(cmap="RdYlGn_r", gmap=gmap, axis=None, vmin=1, vmax=len(table))
    )
    st.dataframe(styled, width="stretch", height=38 * (len(table) + 1))
    through = recent["through"].max()
    st.caption(
        f"Through {through}. Averages of each team's last {last_n} games"
        + (" this season" if this_season else "")
        + (", adjusted for opponent" if adjusted else ", raw")
        + ". Colors follow rank within each column (green best, red worst). "
        "'Overall rank score' is the average of a team's ranks on the six EPA columns -- lower is better."
    )

    # Offense vs defense quadrant.
    fig = go.Figure()
    fig.add_scatter(
        x=recent["off_epa"], y=recent["def_epa"], mode="markers+text", text=recent.index,
        textposition="top center", marker=dict(color=BLUE, size=9),
        hovertemplate="%{text}<br>Off %{x:+.3f}<br>Def allowed %{y:+.3f}<extra></extra>",
    )
    fig.add_hline(y=recent["def_epa"].mean(), line_width=1, line_color=GRID)
    fig.add_vline(x=recent["off_epa"].mean(), line_width=1, line_color=GRID)
    fig.update_yaxes(autorange="reversed", title="Defense EPA/play allowed (better is up)", gridcolor=GRID)
    fig.update_xaxes(title="Offense EPA/play (better is right)", gridcolor=GRID)
    fig.update_layout(
        title="Offense vs. defense" + (" (opponent-adjusted)" if adjusted else ""), height=560,
        showlegend=False, **LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")
