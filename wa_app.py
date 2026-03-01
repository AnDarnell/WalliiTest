"""
Run app: streamlit run wallii_app.py
Requires: pip install requests matplotlib streamlit
"""

import re
import json
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from collections import Counter
from datetime import datetime, timezone


# ── Config ────────────────────────────────────────────────────────────────────

SEASON_START       = "2025-12-01"
THRESHOLD_BASE     = 9000
THRESHOLD_INCREASE = 1000
VALID_REGIONS      = ["NA", "EU", "AP", "CN"]


# ── Formula ───────────────────────────────────────────────────────────────────

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


# ── Fetch & calculate ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_and_calculate(player_name, region):
    url     = f"https://www.wallii.gg/stats/{player_name}?region={region.lower()}&mode=solo&view=all"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    match = re.search(r'\\"data\\":\[(\{\\"player_name.*?)\],\\"availableModes\\"', r.text, re.DOTALL)
    if not match:
        raise ValueError("Player not found — check spelling and region.")

    data_str  = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
    snapshots = json.loads("[" + data_str + "]")
    snapshots = [s for s in snapshots if s["region"].upper() == region.upper() and s["game_mode"] == "0"]
    snapshots = sorted(snapshots, key=lambda x: x["snapshot_time"])

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


# ── Chart ─────────────────────────────────────────────────────────────────────

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
    ax.set_xlabel("Placement", color="#555", fontsize=9, labelpad=8)
    ax.tick_params(colors="#555", labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_tick_params(length=0)

    ax.legend(handles=[
        mpatches.Patch(color="#d4a843", label="1st"),
        mpatches.Patch(color="#4a8c5c", label="Top 4"),
        mpatches.Patch(color="#8c3a2a", label="Bot 4"),
    ], facecolor="#161616", labelcolor="#aaa", fontsize=8,
       edgecolor="#333", framealpha=1, loc="upper right")

    plt.tight_layout(pad=1.2)
    return fig


# ── CSS ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Placement Stats", layout="centered")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Georgia', serif; }
.stApp { background-color: #0e0e0e; color: #ccc; }

.stTextInput input {
    background-color: #161616 !important;
    border: 1px solid #2a2a2a !important;
    color: #eee !important;
    border-radius: 4px !important;
}
.stTextInput input:focus { border-color: #d4a843 !important; }
.stTextInput label, .stSelectbox label {
    color: #666 !important; font-size: 0.75rem !important;
    text-transform: uppercase !important; letter-spacing: 0.07em !important;
}
.stSelectbox > div > div {
    background-color: #161616 !important;
    border: 1px solid #2a2a2a !important;
    color: #eee !important;
    border-radius: 4px !important;
}
.stFormSubmitButton button {
    background-color: #161616 !important;
    color: #d4a843 !important;
    border: 1px solid #d4a843 !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.stFormSubmitButton button:hover {
    background-color: #d4a843 !important;
    color: #0e0e0e !important;
}
[data-testid="metric-container"] {
    background-color: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 0.8rem 1rem;
}
[data-testid="metric-container"] label {
    color: #555 !important; font-size: 0.7rem !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #eee !important; font-size: 0.65rem !important;
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


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("<h2 style='color:#eee; font-weight:normal; margin-bottom:0.2rem;'>Placement Statistics</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#555; font-size:0.8rem; margin-bottom:1.5rem; text-transform:uppercase; letter-spacing:0.08em;'>Hearthstone Battlegrounds</p>", unsafe_allow_html=True)

with st.form("search_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        player = st.text_input("Player", placeholder="jeef")
    with col2:
        region = st.selectbox("Region", VALID_REGIONS)
    submitted = st.form_submit_button("Search", use_container_width=True)

if submitted and player:
    with st.spinner("Fetching data..."):
        try:
            games = fetch_and_calculate(player.strip().lower(), region)

            norm  = normalized_counts(games)
            total = len(games)
            avg   = sum(g["placement"] for g in games) / total
            wins  = norm[1]
            top4  = sum(norm[p] for p in [1, 2, 3, 4])
            netto = games[-1]["mmr_after"] - games[0]["mmr_before"]

            st.markdown(f"<p style='color:#eee; font-size:1.1rem; margin:1.2rem 0 0.8rem;'>{player.lower()} <span style='color:#d4a843; font-size:0.8rem; margin-left:0.5rem;'>{region}</span></p>", unsafe_allow_html=True)

            def stat(label, value):
                return f"""<div style="background:#161616;border:1px solid #2a2a2a;border-radius:4px;padding:0.6rem 0.8rem;text-align:left;">
                    <div style="color:#555;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">{label}</div>
                    <div style="color:#eee;font-size:1.1rem;font-weight:600;">{value}</div>
                </div>"""

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(stat("Games",     f"{total}"),                             unsafe_allow_html=True)
            c2.markdown(stat("Avg Place", f"{avg:.2f}"),                           unsafe_allow_html=True)
            c3.markdown(stat("1st",       f"{wins} ({wins/total*100:.0f}%)"),      unsafe_allow_html=True)
            c4.markdown(stat("Top 4",     f"{top4} ({top4/total*100:.0f}%)"),      unsafe_allow_html=True)
            c5.markdown(stat("CR",        f"{games[-1]['mmr_after']:,}"),          unsafe_allow_html=True)               

            st.pyplot(make_chart(games))

            with st.expander("View as table"):
                rows = [{"Place": p, "Count": norm[p], "%": f"{norm[p]/total*100:.1f}%"} for p in range(1, 9)]
                st.table(rows)

        except Exception as e:
            st.error(str(e))

elif submitted:
    st.warning("Enter a player name.")