"""
Run:
  streamlit run wallii_app.py

Requires:
  pip install streamlit requests matplotlib numpy pandas

CSV (bundled with app):
  Put export.csv next to this file (same folder).
  Expected columns: lb_mmr, current_mmr, avg_place, games
"""

import re
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone, date


# ── Config ────────────────────────────────────────────────────────────────────

SEASON_START       = "2025-12-01"
THRESHOLD_BASE     = 9000
THRESHOLD_INCREASE = 1000
VALID_REGIONS      = ["NA", "EU", "AP", "CN"]
DEFAULT_CSV_NAME   = "export.csv"

SUPABASE_URL = "https://xtivasurpzvcbomieuba.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh0aXZhc3VycHp2Y2JvbWlldWJhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQzMTUzODgsImV4cCI6MjA1OTg5MTM4OH0.Opd3c-esvzBd-CWBDSSV7XFB2JCF2LlyevrE2Yr054U"
SUPABASE_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}
MIN_GAMES_NEIGHBOR = 300


# ── Helpers ───────────────────────────────────────────────────────────────────

def style_dark_axes(ax):
    ax.xaxis.label.set_color("#aaa")
    ax.yaxis.label.set_color("#aaa")
    ax.tick_params(axis="x", colors="#aaa")
    ax.tick_params(axis="y", colors="#aaa")
    ax.grid(True, alpha=0.2, color="#444")
    for spine in ax.spines.values():
        spine.set_visible(False)


def get_threshold(snapshot_time_str):
    season_start = datetime.fromisoformat(SEASON_START).replace(tzinfo=timezone.utc)
    game_time    = datetime.fromisoformat(snapshot_time_str)
    if game_time.tzinfo is None:
        game_time = game_time.replace(tzinfo=timezone.utc)
    days_in = max(0, (game_time - season_start).days)
    return THRESHOLD_BASE + (days_in // 20) * THRESHOLD_INCREASE


def est_place(mmr, gain, snapshot_time=None):
    mmr  = float(mmr)
    gain = float(gain)
    placements = [1, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    dex_avg    = mmr if mmr < 8200 else (mmr - 0.85 * (mmr - 8200))
    threshold  = get_threshold(snapshot_time) if snapshot_time else THRESHOLD_BASE
    best_placement, best_delta = placements[0], None
    for p in placements:
        avg_opp = mmr - 148.1181435 * (100 - ((p - 1) * (200 / 7) + gain))
        if avg_opp > threshold:
            continue
        delta = abs(dex_avg - avg_opp)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_placement = p
    return best_placement


# ── Single-player fetch & calculate ───────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_and_calculate(player_name, region):
    url     = f"https://www.wallii.gg/stats/{player_name}?region={region.lower()}&mode=solo&view=all"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

    match = re.search(r'\\"data\\":\[(\{\\"player_name.*?)\],\\"availableModes\\"', r.text, re.DOTALL)
    if not match:
        raise ValueError("Player not found — check spelling and region.")

    data_str  = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
    snapshots = json.loads("[" + data_str + "]")
    snapshots = [s for s in snapshots if s["region"].upper() == region.upper() and s["game_mode"] == "0"]
    snapshots = sorted(snapshots, key=lambda x: x["snapshot_time"])

    if len(snapshots) < 2:
        raise ValueError("Not enough snapshots to compute games.")

    games = []
    for i in range(1, len(snapshots)):
        prev, curr = snapshots[i - 1], snapshots[i]
        gain = curr["rating"] - prev["rating"]
        games.append({
            "mmr_before": prev["rating"],
            "mmr_after":  curr["rating"],
            "gain":       gain,
            "placement":  est_place(prev["rating"], gain, snapshot_time=curr["snapshot_time"]),
            "time":       curr["snapshot_time"],
        })

    return games


# ── Normalize ─────────────────────────────────────────────────────────────────

def normalized_counts(games):
    counts      = {p: 0 for p in range(1, 9)}
    half_counts = {}
    for g in games:
        p = g["placement"]
        if p == int(p):
            counts[int(p)] += 1
        else:
            low, high = int(p), int(p) + 1
            half_counts[(low, high)] = half_counts.get((low, high), 0) + 1
    for (low, high), n in half_counts.items():
        counts[low]  += n // 2
        counts[high] += n // 2 + n % 2
    return counts


def norm_to_pct(games):
    """Return normalized distribution as percentages (sums to 100)."""
    counts = normalized_counts(games)
    total  = sum(counts.values())
    if total == 0:
        return {p: 0.0 for p in range(1, 9)}
    return {p: counts[p] / total * 100 for p in range(1, 9)}


# ── Leaderboard neighbor lookup ───────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_player_rank(player_name, region):
    """Returns (rank, games_played) for a player on today's leaderboard."""
    today = date.today().isoformat()
    # Fetch a wide range and search by name client-side (Supabase join filter syntax is tricky)
    url = (
        f"{SUPABASE_URL}/rest/v1/daily_leaderboard_stats"
        f"?select=rank,games_played,players!inner(player_name)"
        f"&region=eq.{region}&game_mode=eq.0&day_start=eq.{today}"
        f"&order=rank.asc&limit=1000"
    )
    r = requests.get(url, headers=SUPABASE_HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    for row in data:
        if row["players"]["player_name"].lower() == player_name.lower():
            return row["rank"], row.get("games_played", 0)
    return None, None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_neighbor_names(player_rank, region, n=5):
    """
    Returns up to n names above and n names below player_rank.
    Fetches without games_played filter — we check game count after fetching.
    """
    today   = date.today().isoformat()
    search  = 30
    lo_rank = max(1, player_rank - search)
    hi_rank = player_rank + search

    url = (
        f"{SUPABASE_URL}/rest/v1/daily_leaderboard_stats"
        f"?select=rank,players!inner(player_name)"
        f"&region=eq.{region}&game_mode=eq.0&day_start=eq.{today}"
        f"&rank=gte.{lo_rank}&rank=lte.{hi_rank}"
        f"&order=rank.asc&limit=100"
    )
    r = requests.get(url, headers=SUPABASE_HEADERS, timeout=10)
    r.raise_for_status()
    rows = r.json()

    above = [row for row in rows if row["rank"] < player_rank][-n:]
    below = [row for row in rows if row["rank"] > player_rank][:n]

    return (
        [row["players"]["player_name"] for row in above],
        [row["players"]["player_name"] for row in below],
        [row["rank"] for row in above],
        [row["rank"] for row in below],
    )


# ── Charts ─────────────────────────────────────────────────────────────────────

def make_chart(games):
    norm   = normalized_counts(games)
    labels = [str(p) for p in range(1, 9)]
    values = [norm[p] for p in range(1, 9)]
    colors = ["#d4a843" if p == 1 else "#4a8c5c" if p <= 4 else "#8c3a2a" for p in range(1, 9)]
    total  = len(games)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    bars = ax.bar(labels, values, color=colors, edgecolor="#222", linewidth=0.8, width=0.55)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val}\n{val/total*100:.1f}%",
                ha="center", va="bottom", color="#aaa", fontsize=13
            )

    ax.set_ylim(0, max(values) * 1.45)
    ax.set_xlabel("Placement", fontsize=12, labelpad=10)
    ax.tick_params(labelsize=10)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    style_dark_axes(ax)

    ax.legend(handles=[
        mpatches.Patch(color="#d4a843", label="1st"),
        mpatches.Patch(color="#4a8c5c", label="Top 4"),
        mpatches.Patch(color="#8c3a2a", label="Bot 4"),
    ], facecolor="#161616", labelcolor="#aaa", fontsize=11,
       edgecolor="#555", framealpha=1, loc="upper right")

    plt.tight_layout(pad=1.2)
    return fig


def make_neighbor_chart(all_pcts, names, ranks, player_name, player_rank):
    """
    all_pcts : list of {1..8: pct} dicts (one per neighbor)
    Shows averaged distribution as a bar chart.
    """
    avg_pcts = {p: np.mean([d[p] for d in all_pcts]) for p in range(1, 9)}
    labels   = [str(p) for p in range(1, 9)]
    values   = [avg_pcts[p] for p in range(1, 9)]
    colors   = ["#d4a843" if p == 1 else "#4a8c5c" if p <= 4 else "#8c3a2a" for p in range(1, 9)]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    bars = ax.bar(labels, values, color=colors, edgecolor="#222", linewidth=0.8, width=0.55)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.1f}%",
                ha="center", va="bottom", color="#aaa", fontsize=16
            )

    n = len(all_pcts)
    above = sum(1 for r in ranks if r < player_rank)
    below = sum(1 for r in ranks if r > player_rank)
    ax.set_title(
        f"Neighbor average  ({above} above · {below} below · {n} players · rank {player_rank})",
        color="#666", fontsize=9, pad=10
    )
    ax.set_xlabel("Placement", fontsize=12, labelpad=10)
    ax.tick_params(labelsize=10)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.set_ylim(0, max(values) * 1.45)
    style_dark_axes(ax)

    ax.legend(handles=[
        mpatches.Patch(color="#d4a843", label="1st"),
        mpatches.Patch(color="#4a8c5c", label="Top 4"),
        mpatches.Patch(color="#8c3a2a", label="Bot 4"),
    ], facecolor="#161616", labelcolor="#aaa", fontsize=11,
       edgecolor="#555", framealpha=1, loc="upper right")

    plt.tight_layout(pad=1.2)
    return fig


# ── Weighted binned curve (RatingAvg tab) ─────────────────────────────────────

def weighted_quantile(values, weights, q):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[m], weights[m]
    if len(values) == 0:
        return np.nan
    idx = np.argsort(values)
    values, weights = values[idx], weights[idx]
    cum = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(q, cum, values))


def binned_weighted_curve(df, x_col, y_col="avg_place", w_col="games",
                          bin_size=500, mode="wquant", q=0.5, min_games=0):
    d = df[[x_col, y_col, w_col]].dropna().copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d[w_col] = pd.to_numeric(d[w_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col, w_col])
    d = d[np.isfinite(d[x_col]) & np.isfinite(d[y_col]) & np.isfinite(d[w_col])]
    d = d[(d[y_col] >= 1) & (d[y_col] <= 8)]
    d = d[d[w_col] > 0]
    if min_games > 0:
        d = d[d[w_col] >= min_games]
    if d.empty:
        return np.array([]), np.array([]), d

    d["bin"] = (d[x_col] // bin_size) * bin_size
    xs, ys = [], []
    for b, g in d.groupby("bin"):
        y = g[y_col].to_numpy(float)
        w = g[w_col].to_numpy(float)
        val = float((w * y).sum() / w.sum()) if mode == "wmean" else weighted_quantile(y, w, q)
        if np.isfinite(val):
            xs.append(float(b))
            ys.append(val)

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    order = np.argsort(xs)
    return xs[order], ys[order], d


# ── Page styling ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Placement Stats", layout="centered")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Georgia', serif; }
.stApp { background-color: #0e0e0e; color: #ccc; }

.stTextInput input, .stNumberInput input {
    background-color: #161616 !important;
    border: 1px solid #2a2a2a !important;
    color: #eee !important;
    border-radius: 4px !important;
}
.stTextInput input:focus, .stNumberInput input:focus { border-color: #d4a843 !important; }
.stTextInput label, .stSelectbox label, .stNumberInput label {
    color: #666 !important; font-size: 0.75rem !important;
    text-transform: uppercase !important; letter-spacing: 0.07em !important;
}
.stSelectbox > div > div {
    background-color: #161616 !important;
    border: 1px solid #2a2a2a !important;
    color: #eee !important;
    border-radius: 4px !important;
}
.stFormSubmitButton button, .stButton button {
    background-color: #161616 !important;
    color: #d4a843 !important;
    border: 1px solid #d4a843 !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.stFormSubmitButton button:hover, .stButton button:hover {
    background-color: #d4a843 !important;
    color: #0e0e0e !important;
}
.streamlit-expanderHeader {
    background-color: #161616 !important;
    border: 1px solid #2a2a2a !important;
    color: #666 !important; font-size: 0.8rem !important;
    border-radius: 4px !important;
}
.stTable th { color: #555 !important; font-size: 0.75rem !important; text-transform: uppercase !important; }
.stTable td { color: #bbb !important; }
hr { border-color: #1e1e1e !important; }
#MainMenu, footer, header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='color:#eee; font-weight:normal; margin-bottom:0.2rem;'>Placement Statistics</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#555; font-size:0.8rem; margin-bottom:1.0rem; text-transform:uppercase; letter-spacing:0.08em;'>Hearthstone Battlegrounds</p>", unsafe_allow_html=True)

tabs = st.tabs(["Single player", "RatingAvg"])


# ── Single player tab ─────────────────────────────────────────────────────────

with tabs[0]:
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            player = st.text_input("Player", placeholder="Name")
        with col2:
            region = st.selectbox("Region", VALID_REGIONS, index=VALID_REGIONS.index("EU"))
        submitted = st.form_submit_button("Search", use_container_width=True)

    if submitted and player:
        st.session_state["sp_player"] = player.strip().lower()
        st.session_state["sp_region"] = region
        st.session_state["sp_games"]  = None
        st.session_state["nb_result"] = None

    sp_player = st.session_state.get("sp_player")
    sp_region = st.session_state.get("sp_region")

    if sp_player and sp_region:
        if st.session_state.get("sp_games") is None:
            with st.spinner("Fetching data..."):
                try:
                    st.session_state["sp_games"] = fetch_and_calculate(sp_player, sp_region)
                except Exception as e:
                    st.error(str(e))
                    st.session_state["sp_games"] = []

        games = st.session_state.get("sp_games", [])

        if games:
            try:
                norm  = normalized_counts(games)
                total = len(games)
                avg   = sum(g["placement"] for g in games) / total
                wins  = norm[1]
                top4  = sum(norm[p] for p in [1, 2, 3, 4])
                current_mmr = games[-1]["mmr_after"]
                peak_mmr = max(
                    max(g["mmr_before"] for g in games),
                    max(g["mmr_after"]  for g in games),
                )
                diff_to_cr = current_mmr - peak_mmr

                def stat(label, value):
                    return (
                        '<div style="background:#161616;border:1px solid #2a2a2a;border-radius:4px;padding:0.6rem 0.8rem;">'
                        '<div style="color:#555;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">' + label + '</div>'
                        '<div style="color:#eee;font-size:1.1rem;font-weight:600;">' + str(value) + '</div>'
                        '</div>'
                    )

                player_rank_display, _ = fetch_player_rank(sp_player, sp_region)
                rank_str = f" <span style='color:#999;font-size:0.8rem;margin-left:0.5rem;'>#{player_rank_display}</span>" if player_rank_display else ""
                st.markdown(
                    "<p style='color:#eee;font-size:1.1rem;margin:1.2rem 0 0.8rem;'>"
                    + sp_player
                    + " <span style='color:#d4a843;font-size:0.8rem;margin-left:0.5rem;'>" + sp_region + "</span>"
                    + rank_str + "</p>",
                    unsafe_allow_html=True
                )

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.markdown(stat("Games",     str(total)),                                    unsafe_allow_html=True)
                c2.markdown(stat("Avg Place", f"{avg:.2f}"),                                  unsafe_allow_html=True)
                c3.markdown(stat("1st",       f"{wins} ({wins/total*100:.0f}%)"),             unsafe_allow_html=True)
                c4.markdown(stat("Top 4",     f"{top4} ({top4/total*100:.0f}%)"),             unsafe_allow_html=True)
                c5.markdown(stat("CR",        f"{current_mmr:,}"),                            unsafe_allow_html=True)
                peak_time_raw = max(games, key=lambda g: g["mmr_after"])["time"]
                try:
                    peak_time_dt  = datetime.fromisoformat(peak_time_raw)
                    peak_time_str = peak_time_dt.strftime("%b %-d, %Y")
                except Exception:
                    peak_time_str = peak_time_raw[:10]
                c5.markdown(
                    "<div title='Peak date: " + peak_time_str + "' style='margin-top:0.45rem;color:#777;font-size:0.85rem;'>Peak: "
                    "<span style='color:#aaa;font-weight:600;'>" + f"{peak_mmr:,}" + "</span> "
                    "<span style='color:#666;'>(" + f"{diff_to_cr:+,}" + ")</span></div>",

                    unsafe_allow_html=True
                )

                st.pyplot(make_chart(games))

                # Tilt factor — avg placement in next 5 games after a 7 or 8, no overlapping windows
                placements  = [round(g["placement"]) for g in games]
                after_bot2  = []
                skip_until  = 0
                for i, p in enumerate(placements):
                    if i < skip_until:
                        continue
                    if p >= 7:
                        window = placements[i+1 : i+4]
                        after_bot2.extend(window)
                        skip_until = i + 4

                if len(after_bot2) >= 3:
                    after_avg  = sum(after_bot2) / len(after_bot2)
                    factor     = after_avg / avg if avg > 0 else 1.0
                    factor     = 1 + (factor - 1) * 3
                    tilt_color = (
                        "#8c3a2a" if factor >= 1.15
                        else "#c47c2a" if factor >= 1.06
                        else "#555" if factor >= 1.00
                        else "#7ab87a" if factor >= 0.90
                        else "#4a8c5c"
                    )
                    tooltip    = f"Avg next 3 after 7-8: {after_avg:.2f} / overall avg: {avg:.2f}"
                    trigger_count = sum(1 for p in placements if p >= 7)
                    asterisk      = "*" if trigger_count < 40 else ""
                    st.markdown(
                        f"<div style='margin:0.6rem 0 0.8rem;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Tilt factor"
                        f"<span style='color:{tilt_color};font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{factor:.2f}{asterisk}</span>"
                        f"<span title='{tooltip}' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with st.expander("View as table"):
                    rows = [{"Place": p, "Count": norm[p], "%": f"{norm[p]/total*100:.1f}%"} for p in range(1, 9)]
                    st.table(rows)

                st.markdown("<hr>", unsafe_allow_html=True)

                if st.button("Compare with leaderboard neighbors", use_container_width=True):
                    st.session_state["nb_result"] = None
                    with st.spinner("Looking up leaderboard rank..."):
                        player_rank, _ = fetch_player_rank(sp_player, sp_region)

                    if player_rank is None:
                        st.session_state["nb_result"] = {"error": "Player not found on today's leaderboard."}
                    else:
                        with st.spinner(f"Finding neighbors around rank {player_rank}..."):
                            names_above, names_below, ranks_above, ranks_below = fetch_neighbor_names(
                                player_rank, sp_region
                            )

                        all_names = names_above + names_below
                        all_ranks = ranks_above + ranks_below

                        if not all_names:
                            st.session_state["nb_result"] = {"error": f"No neighbors found with {MIN_GAMES_NEIGHBOR}+ games nearby."}
                        else:
                            all_pcts = []
                            failed   = []
                            progress = st.progress(0)
                            status   = st.empty()

                            for i, (name, rank) in enumerate(zip(all_names, all_ranks)):
                                status.markdown(
                                    "<p style='color:#666;font-size:0.8rem;'>Fetching "
                                    + name + " (rank " + str(rank) + ")...</p>",
                                    unsafe_allow_html=True
                                )
                                try:
                                    ng = fetch_and_calculate(name, sp_region)
                                    if len(ng) >= MIN_GAMES_NEIGHBOR:
                                        all_pcts.append(norm_to_pct(ng))
                                except Exception:
                                    failed.append(name)
                                progress.progress((i + 1) / len(all_names))

                            progress.empty()
                            status.empty()

                            st.session_state["nb_result"] = {
                                "pcts":        all_pcts,
                                "names":       all_names,
                                "ranks":       all_ranks,
                                "failed":      failed,
                                "player_rank": player_rank,
                            }

                nb = st.session_state.get("nb_result")
                if nb:
                    if "error" in nb:
                        st.warning(nb["error"])
                    elif nb.get("pcts"):
                        if nb["failed"]:
                            st.caption("Could not fetch: " + ", ".join(nb["failed"]))
                        st.markdown(
                            "<p style='color:#666;font-size:0.8rem;margin-bottom:0.5rem;'>Neighbors: "
                            + ", ".join(nb["names"]) + "</p>",
                            unsafe_allow_html=True
                        )
                        st.pyplot(make_neighbor_chart(nb["pcts"], nb["names"], nb["ranks"], sp_player, nb["player_rank"]))

                        # Horizontal diff row
                        player_pct = norm_to_pct(games)
                        avg_pct    = {p: sum(d[p] for d in nb["pcts"]) / len(nb["pcts"]) for p in range(1, 9)}

                        cells = ""
                        for p in range(1, 9):
                            diff  = player_pct[p] - avg_pct[p]
                            color = "#4a8c5c" if diff > 0.5 else "#8c3a2a" if diff < -0.5 else "#555"
                            cells += (
                                f"<div style='text-align:center;flex:1;'>"
                                f"<div style='color:#444;font-size:0.65rem;margin-bottom:0.2rem;'>{p}</div>"
                                f"<div style='color:{color};font-size:1.1rem;font-weight:600;'>{diff:+.1f}%</div>"
                                f"</div>"
                            )

                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;padding:0.3rem 4.5%;border-top:1px solid #1e1e1e;margin-top:0.2rem;'>"
                            f"{cells}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning("Could not retrieve enough data from neighbors.")

            except Exception as e:
                st.error(str(e))

    elif submitted:
        st.warning("Enter a player name.")

# ── RatingAvg tab (CSV) ───────────────────────────────────────────────────────

with tabs[1]:
    csv_path = Path(__file__).parent / DEFAULT_CSV_NAME

    if not csv_path.exists():
        st.error(f"Can't find CSV: {csv_path}\n\nPut your export CSV next to wallii_app.py and name it {DEFAULT_CSV_NAME}.")
        st.stop()

    df = pd.read_csv(csv_path)

    required = {"lb_mmr", "current_mmr", "avg_place", "games"}
    missing  = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV is missing columns: {', '.join(missing)}")
        st.stop()

    c1, c2, c3 = st.columns([1.2, 1.0, 1.2])
    with c1:
        x_choice = st.selectbox("MMR source", ["lb_mmr", "current_mmr"], index=0)
    with c2:
        bin_size = st.select_slider("Bin size", options=[250, 500, 750, 1000], value=1000)
    with c3:
        max_games       = int(pd.to_numeric(df["games"], errors="coerce").max() or 0)
        default_min_games = 300 if max_games >= 300 else max_games
        min_games       = st.slider("Min games", min_value=0, max_value=max_games, value=default_min_games, step=50)

    mode = st.selectbox(
        "Curve",
        [
            ("Weighted 10th percentile", "wquant", 0.10),
            ("Weighted 25th percentile", "wquant", 0.25),
            ("Weighted median",          "wquant", 0.50),
            ("Weighted mean",            "wmean",  None),
        ],
        format_func=lambda t: t[0],
        index=3,
    )
    mode_kind = mode[1]
    q         = mode[2] if mode_kind == "wquant" else 0.5

    bx, by, d = binned_weighted_curve(
        df,
        x_col=x_choice,
        y_col="avg_place",
        w_col="games",
        bin_size=int(bin_size),
        mode=mode_kind,
        q=float(q) if mode_kind == "wquant" else 0.5,
        min_games=int(min_games),
    )

    if len(bx) < 2:
        st.warning("Too few bins after filtering. Lower min games or change bin size.")
        st.stop()

    st.markdown(
        "<div style='color:#666;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>"
        f"Estimate Avg Place at {x_choice}</div>",
        unsafe_allow_html=True
    )

    q_mmr = st.number_input(
        label="MMR",
        min_value=float(np.min(bx)),
        max_value=float(np.max(bx)),
        value=float(np.percentile(bx, 75)),
        step=100.0,
        label_visibility="collapsed",
    )
    est = float(np.interp(float(q_mmr), bx, by))

    st.markdown(
        f"<div style='margin-top:0.35rem;margin-bottom:0.8rem;'>"
        f"<span style='color:#eee;font-size:1.6rem;font-weight:700;'>{est:.2f}</span>"
        f"<span style='color:#777;font-size:0.9rem;margin-left:0.6rem;'>at {q_mmr:,.0f} {x_choice}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")
    ax.plot(bx, by, linewidth=3)
    ax.set_xlabel(x_choice)
    ax.set_ylabel("Avg Place")
    style_dark_axes(ax)
    st.pyplot(fig)