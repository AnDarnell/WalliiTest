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
from datetime import datetime, timezone


# ── Config ────────────────────────────────────────────────────────────────────

SEASON_START       = "2025-12-01"
THRESHOLD_BASE     = 9000
THRESHOLD_INCREASE = 1000
VALID_REGIONS      = ["NA", "EU", "AP", "CN"]

DEFAULT_CSV_NAME = "export.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def style_dark_axes(ax):
    """Make matplotlib axes readable on a dark background."""
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


# ── Single-player chart ───────────────────────────────────────────────────────

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
                ha="center", va="bottom", color="#aaa", fontsize=20
            )

    ax.set_ylim(0, max(values) * 1.45)
    ax.set_xlabel("Placement", fontsize=9, labelpad=8)
    ax.tick_params(labelsize=10)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_tick_params(length=0)

    style_dark_axes(ax)

    ax.legend(handles=[
        mpatches.Patch(color="#d4a843", label="1st"),
        mpatches.Patch(color="#4a8c5c", label="Top 4"),
        mpatches.Patch(color="#8c3a2a", label="Bot 4"),
    ], facecolor="#161616", labelcolor="#aaa", fontsize=8,
       edgecolor="#333", framealpha=1, loc="upper right")

    plt.tight_layout(pad=1.2)
    return fig


# ── Option C: weighted binned curve ───────────────────────────────────────────

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
    """
    mode:
      - "wmean"  : weighted mean in each bin (weights = games)
      - "wquant" : weighted quantile in each bin (q controls quantile; q=0.5 => weighted median)
    """
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
        if mode == "wmean":
            val = float((w * y).sum() / w.sum())
        else:
            val = weighted_quantile(y, w, q)
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
            player = st.text_input("Player", placeholder="jeef")
        with col2:
            region = st.selectbox("Region", VALID_REGIONS, index=VALID_REGIONS.index("EU"))
        submitted = st.form_submit_button("Search", use_container_width=True)

    if submitted and player:
        with st.spinner("Fetching data."):
            try:
                games = fetch_and_calculate(player.strip().lower(), region)

                norm  = normalized_counts(games)
                total = len(games)
                avg   = sum(g["placement"] for g in games) / total
                wins  = norm[1]
                top4  = sum(norm[p] for p in [1, 2, 3, 4])
                current_mmr = games[-1]["mmr_after"]

                def stat(label, value):
                    return f"""<div style="background:#161616;border:1px solid #2a2a2a;border-radius:4px;padding:0.6rem 0.8rem;text-align:left;">
                        <div style="color:#555;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">{label}</div>
                        <div style="color:#eee;font-size:1.1rem;font-weight:600;">{value}</div>
                    </div>"""

                st.markdown(
                    f"<p style='color:#eee; font-size:1.1rem; margin:1.2rem 0 0.8rem;'>"
                    f"{player.lower()} <span style='color:#d4a843; font-size:0.8rem; margin-left:0.5rem;'>{region}</span>"
                    f"</p>",
                    unsafe_allow_html=True
                )

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.markdown(stat("Games",     f"{total}"),                             unsafe_allow_html=True)
                c2.markdown(stat("Avg Place", f"{avg:.2f}"),                           unsafe_allow_html=True)
                c3.markdown(stat("1st",       f"{wins} ({wins/total*100:.0f}%)"),      unsafe_allow_html=True)
                c4.markdown(stat("Top 4",     f"{top4} ({top4/total*100:.0f}%)"),      unsafe_allow_html=True)
                c5.markdown(stat("CR",        f"{current_mmr:,}"),                     unsafe_allow_html=True)

                st.pyplot(make_chart(games))

            except Exception as e:
                st.error(str(e))

    elif submitted:
        st.warning("Enter a player name.")


# ── RatingAvg tab (CSV) ──────────────────────────────────────────────────────

with tabs[1]:
    csv_path = Path(__file__).parent / DEFAULT_CSV_NAME

    if not csv_path.exists():
        st.error(
            f"Can't find CSV: {csv_path}\n\n"
            f"Put your export CSV next to wallii_app.py and name it {DEFAULT_CSV_NAME}."
        )
        st.stop()

    df = pd.read_csv(csv_path)

    required = {"lb_mmr", "current_mmr", "avg_place", "games"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV is missing columns: {', '.join(missing)}")
        st.stop()

    # Minimal controls (defaults per your request)
    c1, c2, c3 = st.columns([1.2, 1.0, 1.2])
    with c1:
        x_choice = st.selectbox("MMR source", ["lb_mmr", "current_mmr"], index=0)
    with c2:
        bin_size = st.select_slider("Bin size", options=[250, 500, 750, 1000], value=1000)
    with c3:
        max_games = int(pd.to_numeric(df["games"], errors="coerce").max() or 0)
        default_min_games = 300 if max_games >= 300 else max_games
        min_games = st.slider("Min games", min_value=0, max_value=max_games, value=default_min_games, step=50)

    mode = st.selectbox(
        "Curve",
        [
            ("Weighted 10th percentile", "wquant", 0.10),
            ("Weighted 25th percentile", "wquant", 0.25),
            ("Weighted median", "wquant", 0.50),
            ("Weighted mean", "wmean", None),
        ],
        format_func=lambda t: t[0],
        index=3
    )
    mode_kind = mode[1]
    q = mode[2] if mode_kind == "wquant" else 0.5

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

    # Estimate (above chart), minimalist
    st.markdown(
        "<div style='color:#666;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>"
        f"Estimate Avg Place at {x_choice}"
        "</div>",
        unsafe_allow_html=True
    )

    q_mmr = st.number_input(
        label="",
        min_value=float(np.min(bx)),
        max_value=float(np.max(bx)),
        value=float(np.percentile(bx, 75)),
        step=100.0,
        label_visibility="collapsed",
    )
    est = float(np.interp(float(q_mmr), bx, by))

    st.markdown(
        f"<div style='margin-top:0.35rem; margin-bottom:0.8rem;'>"
        f"<span style='color:#eee; font-size:1.6rem; font-weight:700;'>{est:.2f}</span>"
        f"<span style='color:#777; font-size:0.9rem; margin-left:0.6rem;'>at {q_mmr:,.0f} {x_choice}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Plot (minimal)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    ax.plot(bx, by, linewidth=3)

    ax.set_xlabel(x_choice)
    ax.set_ylabel("Avg Place")
    style_dark_axes(ax)

    st.pyplot(fig)