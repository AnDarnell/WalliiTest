"""
Run:
  streamlit run wa2_app.py

Requires:
  pip install streamlit requests matplotlib numpy pandas

CSV (bundled with app):
  Put your region CSVs next to this file.
  Expected columns: lb_mmr, current_mmr, avg_place, games

Recommended filenames (auto-picked by region):
  export_eu.csv, export_na.csv, export_ap.csv, export_cn.csv

Fallback:
  export.csv (used if the region-specific file is missing)
"""

APP_VERSION = "1.0.0"

import json
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import importlib
_st_components = importlib.import_module("streamlit.components.v1")
from pathlib import Path
from datetime import datetime, timezone, timedelta, date
from urllib.parse import urlencode
import html


# ── Config ────────────────────────────────────────────────────────────────────

DEBUG = False  # set True if you want debug panels + logs

SEASON_START       = "2025-12-01"
THRESHOLD_BASE     = 9000
THRESHOLD_INCREASE = 1000
VALID_REGIONS      = ["NA", "EU", "AP", "CN"]

DEFAULT_CSV_NAME   = "export.csv"

CSV_BY_REGION = {
    "EU": "export_eu.csv",
    "NA": "export_na.csv",
    "AP": "export_ap.csv",
    "CN": "export_cn.csv",
}

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = (
    st.secrets.get("SUPABASE_KEY")
    or st.secrets.get("KEY")
    or st.secrets.get("APIKEY")
    or st.secrets.get("SUPABASE_APIKEY")
    or ""
)

SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)

SUPABASE_HEADERS = {
    "apikey":          SUPABASE_KEY,
    "Authorization":   f"Bearer {SUPABASE_KEY}",
    "Accept-Profile":  "public",
    "Content-Profile": "public",
}

MIN_GAMES_NEIGHBOR = 300

ENABLE_SESSION_TOPLISTS = True
TOPLIST_BACKEND = "supabase"     # "supabase" or "session"
PLAYER_STATS_TABLE = "player_stats"
TOP_N = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def dlog(*args):
    if DEBUG:
        print(*args)

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

def interp_with_extrap(x, xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(xs) < 2:
        return float(ys[0]) if len(ys) else float("nan")

    x = float(x)
    if x <= xs[0]:
        m = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return float(ys[0] + (x - xs[0]) * m)
    if x >= xs[-1]:
        m = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return float(ys[-1] + (x - xs[-1]) * m)

    return float(np.interp(x, xs, ys))

def get_csv_for_region(region: str) -> Path:
    base = Path(__file__).parent
    region = (region or "").upper().strip()
    candidate = base / CSV_BY_REGION.get(region, "")
    if candidate.name and candidate.exists():
        return candidate
    return base / DEFAULT_CSV_NAME

def go_home():
    st.session_state.pop("sp_player", None)
    st.session_state.pop("sp_region", None)
    st.session_state.pop("sp_games", None)
    st.session_state.pop("sp_rank", None)
    st.session_state.pop("nb_result", None)

# ── Query-param navigation (from leaderboard name links) ─────────────────────
_qp = st.query_params
if "goto_player" in _qp:
    st.session_state["sp_player"] = _qp["goto_player"].lower()
    st.session_state["sp_region"] = _qp.get("goto_region", "EU")
    st.session_state.pop("sp_games", None)
    st.session_state.pop("sp_rank", None)
    st.session_state.pop("nb_result", None)
    st.query_params.clear()
    st.rerun()
if "goto_home" in _qp:
    go_home()
    st.query_params.clear()
    st.rerun()


# ── Toplists — session or Supabase ────────────────────────────────────────────

def _lb_init():
    if "toplists" not in st.session_state:
        st.session_state["toplists"] = {"players": {}}

def _lb_key(region, player_name):
    return f"{(region or '').upper()}::{(player_name or '').lower()}"

# ---------- session backend ----------

def _session_upsert(region, player_name, record: dict):
    _lb_init()
    key = _lb_key(region, player_name)
    st.session_state["toplists"]["players"][key] = {
        **record,
        "region": (region or "").upper(),
        "player": (player_name or "").lower(),
    }

def _session_top_n(metric, n=TOP_N, higher_is_better=True):
    _lb_init()
    rows = list(st.session_state["toplists"]["players"].values())
    rows = [r for r in rows if metric in r and r[metric] is not None and np.isfinite(r[metric])]
    rows.sort(key=lambda r: r[metric], reverse=higher_is_better)
    return rows[:n]

# ---------- Supabase backend ----------

def _cache_bust_toplists():
    try:
        _sb_fetch_all.clear()
        _sb_top_n.clear()
    except Exception:
        pass

def _sb_upsert(region, player_name, record: dict):
    payload = {
        "player":       (player_name or "").lower(),
        "region":       (region or "").upper(),
        "games":        record.get("games"),
        "hot_streak":   record.get("hot_streak"),
        "roach_streak": record.get("roach_streak"),
        "first_pct":    record.get("first_pct"),
        "top4_pct":     record.get("top4_pct"),
        "tilt_factor":  record.get("tilt_factor"),
        "avg_place":    record.get("avg_place"),
        "form_diff":    record.get("form_diff"),
        "max_drawdown": record.get("max_drawdown"),
        "dd_detail":      record.get("dd_detail"),
        "first_10k_date": record.get("first_10k_date"),
        "cr":             record.get("cr"),
        "u_score":        record.get("u_score"),
        "bot2_count":     record.get("bot2_count"),
        "updated_at":   datetime.utcnow().isoformat() + "Z",
    }

    if not SUPABASE_ENABLED:
        if DEBUG:
            st.session_state["sb_upsert_status"] = ("DISABLED", "SUPABASE_URL/KEY missing.")
        return

    try:
        url = f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}?on_conflict=player,region"
        r = requests.post(
            url,
            headers={
                **SUPABASE_HEADERS,
                "Content-Type": "application/json",
                "Prefer":       "resolution=merge-duplicates,return=minimal",
            },
            json=payload,
            timeout=10,
        )
        if DEBUG:
            st.session_state["sb_upsert_status"] = (str(r.status_code), (r.text or "")[:300])

        dlog("UPSERT STATUS:", r.status_code)
        dlog("UPSERT RESPONSE:", (r.text or "")[:500])

        r.raise_for_status()
        _cache_bust_toplists()

    except Exception as e:
        if DEBUG:
            st.session_state["sb_upsert_status"] = ("ERROR", f"{type(e).__name__}: {e}"[:300])
        dlog("UPSERT ERROR:", e)

_ALL_FIELDS = "player,region,games,first_pct,top4_pct,hot_streak,roach_streak,tilt_factor,avg_place,form_diff,max_drawdown,dd_detail,first_10k_date,cr,u_score,bot2_count,updated_at"

@st.cache_data(show_spinner=False, ttl=60)
def _sb_fetch_all():
    if not SUPABASE_ENABLED:
        return []
    try:
        url = f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}"
        r = requests.get(
            url,
            headers=SUPABASE_HEADERS,
            params={"select": _ALL_FIELDS, "limit": "500"},
            timeout=10,
        )
        if DEBUG:
            st.session_state["sb_topn_status"] = (str(r.status_code), (r.text or "")[:200])
        r.raise_for_status()
        return r.json()
    except Exception as e:
        if DEBUG:
            st.session_state["sb_topn_status"] = ("ERROR", f"{type(e).__name__}: {e}"[:200])
        return []

@st.cache_data(show_spinner=False, ttl=60)
def _sb_top_n(metric, n=TOP_N, higher_is_better=True):
    rows = [r for r in _sb_fetch_all() if r.get(metric) is not None]
    rows.sort(key=lambda r: (r[metric], r.get("cr") or 0), reverse=higher_is_better)
    return rows[:n]

# ---------- unified API ----------

def lb_upsert_player(region, player_name, record: dict):
    if TOPLIST_BACKEND == "supabase":
        _sb_upsert(region, player_name, record)
    else:
        _session_upsert(region, player_name, record)

def compute_and_upsert(player_name, region, games):
    if not games or len(games) < 50:
        return
    norm  = normalized_counts(games)
    total = len(games)
    avg   = sum(g["placement"] for g in games) / total
    wins  = norm[1]
    top4  = sum(norm[p] for p in [1, 2, 3, 4])
    _eps    = 0.5
    _part1  = np.log((norm[1] + _eps) / (norm[2] + norm[3] + norm[4] + _eps))
    _part2  = np.log((norm[7] + norm[8] + _eps) / (norm[5] + norm[6] + _eps))
    u_score  = 0.5 * (_part1 + _part2)
    bot2_count = norm[7] + norm[8]
    current_mmr = games[-1]["mmr_after"]

    longest_streak, streak = 0, 0
    for g in games:
        streak = streak + 1 if round(g["placement"]) == 1 else 0
        longest_streak = max(longest_streak, streak)

    longest_roach, roach = 0, 0
    for g in games:
        roach = roach + 1 if round(g["placement"]) <= 4 else 0
        longest_roach = max(longest_roach, roach)

    placements = [round(g["placement"]) for g in games]
    after_bot2, skip_until = [], 0
    for i, p in enumerate(placements):
        if i < skip_until:
            continue
        if p >= 7:
            after_bot2.extend(placements[i+1 : i+4])
            skip_until = i + 4

    tilt_factor_val = None
    if len(after_bot2) >= 3:
        after_avg = sum(after_bot2) / len(after_bot2)
        tilt_factor_val = float(1 + ((after_avg / avg) - 1) * 2) if avg > 0 else None

    form_diff = None
    if total >= 60:
        recent_avg = sum(g["placement"] for g in games[-50:]) / 50
        form_diff = recent_avg - avg

    max_dd = 0
    peak_so_far = games[0]["mmr_after"]
    peak_so_far_game = games[0]
    dd_peak_game = games[0]
    dd_trough_game = games[0]
    for g in games:
        if g["mmr_after"] > peak_so_far:
            peak_so_far = g["mmr_after"]
            peak_so_far_game = g
        dd = peak_so_far - g["mmr_after"]
        if dd > max_dd:
            max_dd = dd
            dd_peak_game = peak_so_far_game
            dd_trough_game = g
    dd_detail = (
        f"{dd_peak_game['mmr_after']:,} → {dd_trough_game['mmr_after']:,} "
        f"({dd_peak_game['time'][:10]} – {dd_trough_game['time'][:10]})"
    )

    first_10k_date = None
    for g in games:
        if g["mmr_after"] >= 10000:
            first_10k_date = g["time"]
            break

    lb_upsert_player(region, player_name, {
        "games":          int(total),
        "hot_streak":     int(longest_streak),
        "roach_streak":   int(longest_roach),
        "first_pct":      float(wins / total * 100),
        "top4_pct":       float(top4 / total * 100),
        "tilt_factor":    tilt_factor_val,
        "avg_place":      float(avg),
        "form_diff":      float(form_diff) if form_diff is not None else None,
        "max_drawdown":   int(max_dd),
        "dd_detail":      dd_detail,
        "first_10k_date": first_10k_date,
        "cr":             int(current_mmr),
        "u_score":        float(u_score),
        "bot2_count":     int(bot2_count),
        "updated_at":     datetime.utcnow().isoformat() + "Z",
    })

def lb_top_n(metric, n=TOP_N, higher_is_better=True):
    if TOPLIST_BACKEND == "supabase":
        return _sb_top_n(metric, n, higher_is_better)
    else:
        return _session_top_n(metric, n, higher_is_better)


# ── Single-player fetch & calculate ───────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def fetch_and_calculate(player_name, region):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def _fetch(rgn):
        s = requests.Session()
        s.max_redirects = 5
        u = f"https://www.wallii.gg/stats/{player_name}?region={rgn.lower()}&mode=solo&view=all"
        return s.get(u, headers=headers, timeout=20)

    try:
        r = _fetch(region)
    except requests.exceptions.TooManyRedirects:
        raise ValueError("Player not found — check if correct region.")
    r.raise_for_status()

    match = re.search(r'\\"data\\":\[(\{\\"player_name.*?)\],\\"availableModes\\"', r.text, re.DOTALL)
    if not match:
        raise ValueError("Player not found — check spelling and region.")

    data_str  = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
    snapshots_all     = json.loads("[" + data_str + "]")
    snapshots_all     = [s for s in snapshots_all if s["game_mode"] == "0"]
    available_regions = list({s["region"].upper() for s in snapshots_all})
    snapshots         = [s for s in snapshots_all if s["region"].upper() == region.upper()]

    rank_match   = re.search(r'text-2xl text-white">(\d+)<', r.text)
    current_rank = int(rank_match.group(1)) if rank_match else None
    dlog(
        "DEBUG rank_match:", rank_match,
        "| snippet:",
        repr(r.text[r.text.find("text-2xl"):r.text.find("text-2xl")+60]) if "text-2xl" in r.text else "text-2xl NOT FOUND"
    )

    if not snapshots and available_regions:
        other = ", ".join(r for r in sorted(available_regions) if r != region.upper())
        raise ValueError(f"No data found for {region}. Player appears to be in: {other}.")

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

    return games, region, current_rank


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
    counts = normalized_counts(games)
    total  = sum(counts.values())
    if total == 0:
        return {p: 0.0 for p in range(1, 9)}
    return {p: counts[p] / total * 100 for p in range(1, 9)}


# ── Leaderboard neighbor lookup ───────────────────────────────────────────────

def fetch_player_rank(player_name, region):
    # Ranken skrapas redan i fetch_and_calculate från wallii.gg — används därifrån
    return st.session_state.get("sp_rank"), None, None

# ── Wallii leaderboard (Supabase) ──────────────────────────────────────────────
# Wallii.gg loads leaderboard data via Supabase REST (seen in DevTools Network).
# Put your key in .streamlit/secrets.toml:
#   SUPABASE_ANON_KEY = "..."
SUPABASE_BASE = "https://xtivasurpzvcbomieuba.supabase.co"


def _supabase_headers():
    key = st.secrets.get("SUPABASE_ANON_KEY")
    if not key:
        raise ValueError(
            "Missing SUPABASE_ANON_KEY in Streamlit secrets. "
            "Add it to .streamlit/secrets.toml (local) and Streamlit Cloud Secrets (deploy)."
        )
    return {
        "apikey": key,
        "authorization": f"Bearer {key}",
        "accept": "application/json",
    }


@st.cache_data(show_spinner=False, ttl=300)
def _latest_day_start(game_mode=0):
    """
    Fetch latest available day_start from Supabase.
    """
    url = f"{SUPABASE_BASE}/rest/v1/daily_leaderboard_stats"
    params = {
        "select": "day_start",
        "game_mode": f"eq.{int(game_mode)}",
        "order": "day_start.desc",
        "limit": "1",
    }
    r = requests.get(url, headers=_supabase_headers(), params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError("Could not fetch latest day_start from Wallii Supabase.")
    return data[0]["day_start"]


@st.cache_data(show_spinner=False, ttl=120)
def fetch_neighbor_names(player_rank, region, day_start=None, n=5):
    """
    Returns neighbours around a given 1-indexed rank on Wallii leaderboard:
      (names_above, names_below, ranks_above, ranks_below)

    This version calls the same Supabase PostgREST endpoint that wallii.gg uses
    (see DevTools -> Network).
    """
    player_rank = int(player_rank)
    n = int(n)

    if day_start is None:
        day_start = _latest_day_start(game_mode=0)

    lo_rank = max(1, player_rank - n)
    hi_rank = player_rank + n

    url = f"{SUPABASE_BASE}/rest/v1/daily_leaderboard_stats"

    # Match wallii.gg query style:
    #   select=player_id,rating,rank,region,players!inner(player_name)
    #   region=eq.EU (or NA/AS etc), game_mode=eq.0, day_start=eq.YYYY-MM-DD
    # Use rank range directly (no need for offset pagination).
    params = {
        "select": "player_id,rating,rank,region,updated_at,players!inner(player_name)",
        "region": f"eq.{region.upper()}",
        "game_mode": "eq.0",
        "day_start": f"eq.{day_start}",
        "rank": f"gte.{lo_rank}",
        # PostgREST doesn't allow two 'rank' keys in a dict; use 'and' via query string.
        # We'll add the upper bound manually below.
        "order": "rank.asc",
    }

    # Build query string manually so we can include both rank bounds.
    # (requests will otherwise drop the duplicated key)
    qs_parts = [
        ("select", params["select"]),
        ("region", params["region"]),
        ("game_mode", params["game_mode"]),
        ("day_start", params["day_start"]),
        ("rank", params["rank"]),
        ("rank", f"lte.{hi_rank}"),
        ("order", params["order"]),
    ]
    full_url = f"{url}?{urlencode(qs_parts)}"

    r = requests.get(full_url, headers=_supabase_headers(), timeout=20)
    r.raise_for_status()
    rows = r.json()

    names_above, names_below, ranks_above, ranks_below = [], [], [], []

    for row in rows:
        try:
            rank = int(row.get("rank"))
        except Exception:
            continue

        # Join result sometimes comes as list or dict depending on how PostgREST is configured.
        p = row.get("players")
        if isinstance(p, list) and p:
            name = p[0].get("player_name")
        elif isinstance(p, dict):
            name = p.get("player_name")
        else:
            name = None

        if not name:
            continue

        if rank < player_rank:
            names_above.append(name)
            ranks_above.append(rank)
        elif rank > player_rank:
            names_below.append(name)
            ranks_below.append(rank)

    return names_above[-n:], names_below[:n], ranks_above[-n:], ranks_below[:n]

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def load_rating_curve(csv_path_str, file_mtime, x_choice="current_mmr",
                      bin_size=1000, mode_kind="wmean", q=0.5, min_games=300):
    df = pd.read_csv(csv_path_str)
    bx, by, _ = binned_weighted_curve(
        df,
        x_col=x_choice,
        y_col="avg_place",
        w_col="games",
        bin_size=int(bin_size),
        mode=mode_kind,
        q=float(q) if mode_kind == "wquant" else 0.5,
        min_games=int(min_games),
    )
    return bx, by

def delta_color(delta):
    if delta <= -0.16:
        return "#4a8c5c"
    elif delta <= 0.01:
        return "#7ab87a"
    elif delta <= 0.15:
        return "#d4a843"
    else:
        return "#8c3a2a"

def diff_pct_color(diff):
    if diff <= -10:
        return "#8c3a2a"
    elif diff < 0:
        return "#c47a75"
    elif diff == 0:
        return "#555"
    elif diff < 10:
        return "#7ab87a"
    else:
        return "#4a8c5c"


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
h2 a[data-testid], h1 a[data-testid], h3 a[data-testid] { display: none !important; }

/* Show-more toggle */
.lb-show-more button {
    background: transparent !important;
    border: none !important;
    color: #444 !important;
    font-size: 0.72rem !important;
    padding: 0.05rem 0 !important;
    height: auto !important;
    min-height: 0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
.lb-show-more button:hover { color: #888 !important; }

/* Icon button wrapper (Home arrow) */
.icon-btn button {
    background: transparent !important;
    border: 1px solid #2a2a2a !important;
    color: #999 !important;
    padding: 0.15rem 0.45rem !important;
    border-radius: 6px !important;
    font-size: 1.0rem !important;
    line-height: 1.1rem !important;
}
.icon-btn button:hover {
    border-color: #d4a843 !important;
    color: #d4a843 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='color:#eee; font-weight:normal; margin-bottom:0.2rem;'><a href='?goto_home=1' style='color:inherit;text-decoration:none;' onmouseover=\"this.style.opacity='0.7'\" onmouseout=\"this.style.opacity='1'\">Placement Statistics</a></h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#555; font-size:0.8rem; margin-bottom:1.0rem; text-transform:uppercase; letter-spacing:0.08em;'>Hearthstone Battlegrounds Stats</p>", unsafe_allow_html=True)

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
        st.session_state["sp_rank"]   = None
        st.session_state["nb_result"] = None
        st.rerun()

    if submitted and not player:
        st.warning("Enter a player name.")

    # ── Render-funktion för topplistor ────────────────────────────────────────

    def render_list(container, title, items, fmt, tooltip=None, asterisk_tip=None):
        HEADER_COLOR = "#8a8a8a"

        tip_html = ""
        if tooltip:
            safe_tip = html.escape(str(tooltip))
            tip_html = (
                f"<span title='{safe_tip}' "
                f"style='color:#444;font-size:0.8rem;margin-left:0.35rem;cursor:help;'>?</span>"
            )

        asterisk_html = ""
        if asterisk_tip:
            safe_ast = html.escape(str(asterisk_tip))
            asterisk_html = (
                f"<span title='{safe_ast}' "
                f"style='color:#666;font-size:0.75rem;margin-left:0.2rem;cursor:help;'>*</span>"
            )

        container.markdown(
            f"<div style='color:{HEADER_COLOR};font-size:0.85rem;text-transform:uppercase;"
            f"letter-spacing:0.08em;margin:0.25rem 0 0.45rem;font-weight:600;'>{title}{asterisk_html}{tip_html}</div>",
            unsafe_allow_html=True
        )

        if not items:
            container.markdown(
                "<div style='color:#444;font-size:0.8rem;padding:0.3rem 0;'>—</div>",
                unsafe_allow_html=True
            )
            return

        def row_html(i, r):
            row_color = value_color = HEADER_COLOR
            if i == 1:
                row_color = value_color = "#d4a843"
            elif i == 2:
                row_color = value_color = "#bfc4c8"
            elif i == 3:
                row_color = value_color = "#b57a4a"
            player = r['player']
            region = r.get('region', '')
            link   = f"?goto_player={html.escape(player)}&goto_region={html.escape(region)}"
            return (
                "<div style='display:flex;justify-content:space-between;"
                "border:1px solid #1e1e1e;background:#121212;border-radius:4px;"
                "padding:0.35rem 0.5rem;margin-bottom:0.25rem;'>"
                f"<span style='color:{row_color};font-weight:700'>{i}. "
                f"<a href='{link}' target='_self' style='color:inherit;text-decoration:none;' "
                f"onmouseover=\"this.style.textDecoration='underline'\" "
                f"onmouseout=\"this.style.textDecoration='none'\">{player}</a> "
                f"<span style='color:#666'>({region})</span></span>"
                f"<span style='color:{value_color};font-weight:700'>{fmt(r)}</span>"
                "</div>"
            )

        # Topp 5 visas alltid
        for i, r in enumerate(items[:5], 1):
            container.markdown(row_html(i, r), unsafe_allow_html=True)

        # 6–10 som inline-toggle utan expander-box
        if len(items) > 5:
            expand_key = f"lb_expanded_{title}"
            if expand_key not in st.session_state:
                st.session_state[expand_key] = False
            expanded = st.session_state[expand_key]

            if expanded:
                for i, r in enumerate(items[5:10], 6):
                    container.markdown(row_html(i, r), unsafe_allow_html=True)

            toggle_label = "▲ Show less" if expanded else "▼ Show more"
            container.markdown("<div class='lb-show-more'>", unsafe_allow_html=True)
            if container.button(toggle_label, key=f"lb_toggle_{title}"):
                st.session_state[expand_key] = not expanded
                st.rerun()
            container.markdown("</div>", unsafe_allow_html=True)

    # ── Visa antingen topplistor ELLER spelarsida ─────────────────────────────

    sp_player = st.session_state.get("sp_player")
    sp_region = st.session_state.get("sp_region")

    if not sp_player:
        # ── Topplistor (startsida) ────────────────────────────────────────────
        if ENABLE_SESSION_TOPLISTS:
            _lb_init()
            st.markdown("<hr>", unsafe_allow_html=True)

            if DEBUG:
                with st.expander("Supabase debug", expanded=False):
                    st.caption(f"TOPLIST_BACKEND={TOPLIST_BACKEND} | SUPABASE_ENABLED={SUPABASE_ENABLED}")
                    st.caption(f"SUPABASE_URL={SUPABASE_URL}")
                    st.caption(f"SUPABASE_KEY length={len(SUPABASE_KEY)}")
                    if "sb_topn_status" in st.session_state:
                        code, txt = st.session_state["sb_topn_status"]
                        st.caption(f"TopN fetch: {code} | {txt}")

            backend_label = "all time" if TOPLIST_BACKEND == "supabase" else "this session"
            _lbh_col, _eu_col, _na_col, _ap_col, _cn_col = st.columns([5, 1, 1, 1, 1])
            with _lbh_col:
                st.markdown(
                    f"<p style='color:#ccc;font-size:1.0rem;font-weight:600;margin:0.3rem 0 0.1rem;'>Leaderboards ({backend_label}) <span style='color:#666;font-size:0.75rem;font-weight:400;'>(Players are added when first searched, if eligible)</span></p>",
                    unsafe_allow_html=True
                )
            with _eu_col:
                _inc_eu = st.checkbox("EU", value=True,  key="lb_inc_eu")
            with _na_col:
                _inc_na = st.checkbox("NA", value=True,  key="lb_inc_na")
            with _ap_col:
                _inc_ap = st.checkbox("AP", value=True,  key="lb_inc_ap")
            with _cn_col:
                _inc_cn = st.checkbox("CN", value=False, key="lb_inc_cn")

            _lb_regions = {r for r, v in [("EU", _inc_eu), ("NA", _inc_na), ("AP", _inc_ap), ("CN", _inc_cn)] if v}

            def _lb(metric, higher_is_better=True, n=50):
                rows = lb_top_n(metric, higher_is_better=higher_is_better, n=n)
                rows = [r for r in rows if r.get("region") in _lb_regions]
                return rows[:TOP_N]

            cols = st.columns(2)

            lists = [
                ("Top 1 %",              _lb("first_pct",    higher_is_better=True),   lambda r: f"{r['first_pct']:.1f}%",  "Percentage of games finished in 1st place."),
                ("Hot streak",         _lb("hot_streak",   higher_is_better=True),   lambda r: f"{int(r['hot_streak'])}",  "Longest consecutive 1st streak of placement."),
                ("Top 4 %",            _lb("top4_pct",     higher_is_better=True),   lambda r: f"{r['top4_pct']:.1f}%",   "Percentage of games finished in top 4."),
                ("Roach streak",       _lb("roach_streak", higher_is_better=True),   lambda r: f"{int(r['roach_streak'])}", "Longest consecutive streak of Top 4 place finishes."),
                ("# Games",              _lb("games",        higher_is_better=True),   lambda r: f"{int(r['games'])}",       "Total number of games played this season while on the leaderboard."),
                ("Largest MMR drop",   _lb("max_drawdown", higher_is_better=True),   lambda r: f"<span title='{html.escape(r['dd_detail'])}' style='cursor:help;'>-{int(r['max_drawdown']):,}</span>" if r.get("dd_detail") else (f"-{int(r['max_drawdown']):,}" if r.get("max_drawdown") is not None else "—"), "Largest MMR drop from a peak to a subsequent low."),
                ("Lowest tilt factor",  [r for r in _lb("tilt_factor",  higher_is_better=False) if (r.get("bot2_count") or 0) >= 30][:TOP_N],  lambda r: f"{r['tilt_factor']:.2f}" if r.get("tilt_factor") is not None else "—", "Comparison of performance following a 7th/8th and overall performance. (Lower = better)", "Min 30 games with 7th/8th placement"),
                ("Highest tilt factor", [r for r in _lb("tilt_factor",  higher_is_better=True)  if (r.get("bot2_count") or 0) >= 30][:TOP_N],  lambda r: f"{r['tilt_factor']:.2f}" if r.get("tilt_factor") is not None else "—", "Comparison of performance following a 7th/8th and overall performance. (Higher = worse)", "Min 30 games with 7th/8th placement"),
                ("Best form",          _lb("form_diff",    higher_is_better=False),  lambda r: f"{r['form_diff']:+.2f}" if r.get("form_diff") is not None else "—", "Difference between form (last 50) and overall avg place. More negative = better form relative to baseline."),
                ("First to 10k",       _lb("first_10k_date", higher_is_better=False), lambda r: datetime.fromisoformat(r["first_10k_date"].replace("Z", "+00:00")).strftime("%b %d").replace(" 0", " ") if r.get("first_10k_date") else "—", "Date when the player first reached 10,000 MMR."),
                ("Most aggressive",    _lb("u_score",        higher_is_better=True),   lambda r: f"{r['u_score']:+.2f}" if r.get("u_score") is not None else "—", "Style score: high 1st relative to 2–4, and high 7+8 relative to 5–6. Positive = aggressive/swingy."),
                ("Most defensive",     _lb("u_score",        higher_is_better=False),  lambda r: f"{r['u_score']:+.2f}" if r.get("u_score") is not None else "—", "Style score: low 1st relative to 2–4, and low 7+8 relative to 5–6. Negative = defensive/consistent."),

            ]

            for idx, (title, items, fmt, tip, *rest) in enumerate(lists):
                render_list(cols[idx % 2], title, items, fmt, tooltip=tip, asterisk_tip=rest[0] if rest else None)

            if st.button("Refresh leaderboards", use_container_width=True):
                _cache_bust_toplists()
                st.rerun()

            if "prefetch_done" not in st.session_state:
                st.session_state["prefetch_done"] = True
                import threading
                def _prefetch():
                    seen = set()
                    for row in _sb_fetch_all():
                        key = (row["player"], row.get("region", "EU"))
                        if key not in seen:
                            seen.add(key)
                            try:
                                fetch_and_calculate(key[0], key[1])
                            except Exception:
                                pass
                threading.Thread(target=_prefetch, daemon=True).start()

            with st.expander("Secret stuff"):
                pwd = st.text_input("Password", type="password", key="admin_pwd")
                if pwd == st.secrets.get("ADMIN_PASSWORD", ""):
                    st.caption("Fetches all players in the leaderboard and recalculates their stats.")
                    if st.button("Refresh all players", use_container_width=True):
                        try:
                            url = f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}"
                            resp = requests.get(url, headers=SUPABASE_HEADERS, params={"select": "player,region", "limit": "500"}, timeout=10)
                            resp.raise_for_status()
                            all_players = resp.json()
                        except Exception as e:
                            st.error(f"Could not fetch player list: {e}")
                            all_players = []
                        if all_players:
                            bar = st.progress(0, text="Starting...")
                            for i, row in enumerate(all_players):
                                name, region = row["player"], row["region"]
                                bar.progress((i + 1) / len(all_players), text=f"({i+1}/{len(all_players)}) {name} [{region}]")
                                try:
                                    fetch_and_calculate.clear()
                                    games_r, _, _ = fetch_and_calculate(name, region)
                                    compute_and_upsert(name, region, games_r)
                                except Exception:
                                    pass
                            bar.progress(1.0, text="Done!")
                            st.success(f"Done! {len(all_players)} players refreshed.")
                elif pwd:
                    st.caption("Wrong password.")

    else:
        # ── Spelarsida ────────────────────────────────────────────────────────
        if st.session_state.get("sp_games") is None:
            with st.spinner("Fetching data..."):
                try:
                    st.session_state["sp_games"], st.session_state["sp_region"], st.session_state["sp_rank"] = fetch_and_calculate(sp_player, sp_region)
                except ValueError as e:
                    msg = str(e)
                    m = re.search(r"appears to be in: ([A-Z,\s]+)", msg)
                    if m:
                        detected = m.group(1).strip().split(", ")[0]
                        if detected in VALID_REGIONS:
                            st.info(f"Not found in {sp_region} — retrying as {detected}...")
                            try:
                                st.session_state["sp_region"] = detected
                                st.session_state["sp_games"], st.session_state["sp_region"], st.session_state["sp_rank"] = fetch_and_calculate(sp_player, detected)
                                sp_region = detected
                            except Exception as e2:
                                st.error(str(e2))
                                st.session_state["sp_games"] = []
                        else:
                            st.error(msg)
                            st.session_state["sp_games"] = []
                    else:
                        st.error(msg)
                        st.session_state["sp_games"] = []
                except Exception as e:
                    st.error(str(e))
                    st.session_state["sp_games"] = []

        games = st.session_state.get("sp_games", [])
        sp_region = st.session_state.get("sp_region", sp_region)

        if games:
            try:
                norm  = normalized_counts(games)
                total = len(games)
                avg   = sum(g["placement"] for g in games) / total
                wins  = norm[1]
                top4  = sum(norm[p] for p in [1, 2, 3, 4])
                _eps        = 0.5
                _part1      = np.log((norm[1] + _eps) / (norm[2] + norm[3] + norm[4] + _eps))
                _part2      = np.log((norm[7] + norm[8] + _eps) / (norm[5] + norm[6] + _eps))
                u_score_val = 0.5 * (_part1 + _part2)
                current_mmr = games[-1]["mmr_after"]
                peak_mmr = max(
                    max(g["mmr_before"] for g in games),
                    max(g["mmr_after"]  for g in games),
                )
                diff_to_cr = current_mmr - peak_mmr

                def stat(label, value, value_color="#eee", label_tip=None):
                    tip_html = ""
                    if label_tip:
                        safe_tip = html.escape(str(label_tip))
                        tip_html = (
                            "<span title='" + safe_tip + "' "
                            "style='color:#444;font-size:0.8rem;margin-left:0.35rem;cursor:help;'>?</span>"
                        )
                    return (
                        '<div style="background:#161616;border:1px solid #2a2a2a;border-radius:4px;padding:0.6rem 0.8rem;">'
                        '<div style="color:#555;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">'
                        + label + tip_html +
                        '</div>'
                        '<div style="color:' + value_color + ';font-size:1.1rem;font-weight:600;">'
                        + str(value) +
                        '</div>'
                        '</div>'
                    )

                expected_avg = None
                delta = None
                curve_csv = get_csv_for_region(sp_region)

                if curve_csv.exists():
                    try:
                        mtime = curve_csv.stat().st_mtime
                        bx, by = load_rating_curve(
                            str(curve_csv),
                            mtime,
                            x_choice="current_mmr",
                            bin_size=1000,
                            mode_kind="wmean",
                            min_games=300,
                        )
                        if len(bx) >= 2:
                            expected_avg = interp_with_extrap(current_mmr, bx, by)
                            expected_avg = float(np.clip(expected_avg, 1.0, 8.0))
                            delta = float(avg - expected_avg)
                    except Exception:
                        pass

                avg_color = "#eee"
                avg_tip = None
                if expected_avg is not None and delta is not None:
                    avg_color = delta_color(delta)
                    sign = "+" if delta >= 0 else ""
                    avg_tip = (
                        f"Curve file: {curve_csv.name} | "
                        f"Expected at CR {current_mmr:,}: {expected_avg:.2f} | "
                        f"Δ: {sign}{delta:.2f} (avg - expected)"
                    )

                # Header row: [←] player [region] [#rank]
                hL, hR = st.columns([0.7, 9.3], vertical_alignment="center")

                with hL:
                    st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
                    if st.button("←", key="home_btn_icon", use_container_width=True):
                        go_home()
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                with hR:
                    player_rank_display = st.session_state.get("sp_rank")
                    rank_str = (
                        f" <span style='color:#999;font-size:0.8rem;margin-left:0.5rem;'>#{player_rank_display}</span>"
                        if player_rank_display else ""
                    )
                    st.markdown(
                        "<p style='color:#eee;font-size:1.1rem;margin:1.2rem 0 0.8rem;'>"
                        + sp_player
                        + " <span style='color:#d4a843;font-size:0.8rem;margin-left:0.5rem;'>" + sp_region + "</span>"
                        + rank_str
                        + "</p>",
                        unsafe_allow_html=True
                    )

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.markdown(stat("Games",     str(total)),                                              unsafe_allow_html=True)
                c2.markdown(stat("Avg Place", f"{avg:.2f}", value_color=avg_color, label_tip=avg_tip),  unsafe_allow_html=True)
                c3.markdown(stat("1st",       f"{wins} ({wins/total*100:.0f}%)"),                       unsafe_allow_html=True)
                c4.markdown(stat("Top 4",     f"{top4} ({top4/total*100:.0f}%)"),                       unsafe_allow_html=True)
                c5.markdown(stat("CR",        f"{current_mmr:,}"),                                      unsafe_allow_html=True)

                max_dd = 0
                peak_so_far = games[0]["mmr_after"]
                peak_so_far_game = games[0]
                dd_peak_game = games[0]
                dd_trough_game = games[0]
                for g in games:
                    if g["mmr_after"] > peak_so_far:
                        peak_so_far = g["mmr_after"]
                        peak_so_far_game = g
                    dd = peak_so_far - g["mmr_after"]
                    if dd > max_dd:
                        max_dd = dd
                        dd_peak_game = peak_so_far_game
                        dd_trough_game = g

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
                dd_tip = (
                    f"{dd_peak_game['mmr_after']:,} → {dd_trough_game['mmr_after']:,} "
                    f"({dd_peak_game['time'][:10]} – {dd_trough_game['time'][:10]})"
                )
                c5.markdown(
                    "<div title='" + dd_tip + "' style='margin-top:0.2rem;color:#777;font-size:0.85rem;cursor:help;'>Max MMR drop: "
                    "<span style='color:#c07070;font-weight:600;'>-" + f"{max_dd:,}" + "</span></div>",
                    unsafe_allow_html=True
                )

                st.pyplot(make_chart(games))

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

                longest_streak, streak = 0, 0
                for g in games:
                    streak = streak + 1 if round(g["placement"]) == 1 else 0
                    longest_streak = max(longest_streak, streak)

                longest_roach, roach = 0, 0
                for g in games:
                    roach = roach + 1 if round(g["placement"]) <= 4 else 0
                    longest_roach = max(longest_roach, roach)

                roach_html = (
                    f"<span style='float:right;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Roach streak"
                    f"<span style='color:#4a8c5c;font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{longest_roach}</span>"
                    f"<span title='Longest consecutive streak of Top 4 place finishes.' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                    f"</span>"
                )

                st.markdown(
                    f"<div style='margin:0.3rem 0 0.8rem;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Hot streak"
                    f"<span style='color:#d4a843;font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{longest_streak}</span>"
                    f"<span title='Longest consecutive streak of 1st placement.' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                    f"{roach_html}"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # ── Form (last 50) ────────────────────────────────────────
                if total < 60:
                    form_html = (
                        "<span style='float:right;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Form (last 50)"
                        "<span style='color:#333;font-size:0.8rem;margin-left:0.8rem;font-weight:400;text-transform:none;letter-spacing:0;'>— requires 60+ games</span>"
                        "</span>"
                    )
                else:
                    recent_games = games[-50:]
                    recent_avg   = sum(g["placement"] for g in recent_games) / len(recent_games)
                    form_diff    = recent_avg - avg
                    form_color   = delta_color(form_diff)
                    form_sign = "+" if form_diff >= 0 else ""
                    form_tip  = f"Avg last 50 games: {recent_avg:.2f} vs overall: {avg:.2f}"
                    form_html = (
                        f"<span style='float:right;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Form (last 50)"
                        f"<span style='color:{form_color};font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{recent_avg:.2f}</span>"
                        f"<span style='color:{form_color};font-size:0.8rem;margin-left:0.4rem;'>({form_sign}{form_diff:.2f})</span>"
                        f"<span title='{form_tip}' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                        f"</span>"
                    )

                # ── Tilt factor ───────────────────────────────────────────────
                tilt_factor_val = None
                if total < 60:
                    st.markdown(
                        "<div style='margin:0.3rem 0 0.8rem;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>"
                        "Tilt factor"
                        "<span style='color:#333;font-size:0.8rem;margin-left:0.8rem;font-weight:400;text-transform:none;letter-spacing:0;'>— requires 60+ games</span>"
                        f"{form_html}"
                        "</div>",
                        unsafe_allow_html=True
                    )
                elif len(after_bot2) >= 3:
                    after_avg  = sum(after_bot2) / len(after_bot2)
                    factor     = after_avg / avg if avg > 0 else 1.0
                    factor     = 1 + (factor - 1) * 2
                    tilt_factor_val = float(factor)

                    tilt_color = (
                        "#8c3a2a" if factor >= 1.15
                        else "#c47a75" if factor >= 1.06
                        else "#aaa"   if factor >= 1.00
                        else "#7ab87a" if factor >= 0.90
                        else "#4a8c5c"
                    )
                    tooltip       = f"Comparison of performance following a 7th/8th and overall performance. (Lower = better) 1 + (({after_avg:.2f} / {avg:.2f}) - 1) * 2."
                    trigger_count = sum(1 for p in placements if p >= 7)
                    asterisk      = "*" if trigger_count < 40 else ""
                    asterisk_tip  = f" title='Low sample size: only {trigger_count} games with placement 7–8'" if trigger_count < 40 else ""

                    st.markdown(
                        f"<div style='margin:0.3rem 0 0.8rem;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>"
                        f"Tilt factor"
                        f"<span style='color:{tilt_color};font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{factor:.2f}</span>"
                        f"<span{asterisk_tip} style='color:{tilt_color};font-size:0.8rem;cursor:help;'>{asterisk}</span>"
                        f"<span title='{tooltip}' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                        f"{form_html}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                u_color = (
                    "#d4a843" if u_score_val >= 0.2
                    else "#aaa"  if u_score_val >= -0.1
                    else "#5b8fd4"
                )
                u_tip = "Aggression score: ((p7+p8)/(p5+p6) − 1) × log(p1/avg(p2–p4)). Positive = U-shaped distribution (spikes at both ends). Negative = non-aggressive."
                st.markdown(
                    f"<div style='margin:0.3rem 0 0.8rem;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>"
                    f"Aggression"
                    f"<span style='color:{u_color};font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{u_score_val:+.2f}</span>"
                    f"<span title='{u_tip}' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                if ENABLE_SESSION_TOPLISTS and total >= 50:
                    first_pct = wins / total * 100 if total else 0.0
                    top4_pct  = top4 / total * 100 if total else 0.0
                    lb_upsert_player(
                        sp_region,
                        sp_player,
                        {
                            "games":        int(total),
                            "hot_streak":   int(longest_streak),
                            "roach_streak": int(longest_roach),
                            "first_pct":    float(first_pct),
                            "top4_pct":     float(top4_pct),
                            "tilt_factor":  float(tilt_factor_val) if tilt_factor_val is not None else None,
                            "avg_place":    float(avg),
                            "form_diff":    float(form_diff) if total >= 60 else None,
                            "max_drawdown":   int(max_dd),
                            "dd_detail":      dd_tip,
                            "first_10k_date": next((g["time"] for g in games if g["mmr_after"] >= 10000), None),
                            "cr":             int(current_mmr),
                            "u_score":        float(u_score_val),
                            "updated_at":   datetime.utcnow().isoformat() + "Z",
                        }
                    )

                if DEBUG:
                    with st.expander("Supabase upsert debug", expanded=False):
                        st.caption(f"TOPLIST_BACKEND={TOPLIST_BACKEND} | SUPABASE_ENABLED={SUPABASE_ENABLED}")
                        if "sb_upsert_status" in st.session_state:
                            code, txt = st.session_state["sb_upsert_status"]
                            st.caption(f"Upsert: {code} | {txt}")

                import altair as alt
                period = st.radio("Period", ["Season", "Week", "Day"], horizontal=True, label_visibility="collapsed", key="rg_period")
                now = datetime.now(timezone.utc)
                cutoff = {"Season": None, "Week": now - timedelta(days=7), "Day": now - timedelta(days=1)}[period]
                filtered = [
                    g for g in games
                    if cutoff is None or datetime.fromisoformat(g["time"].replace("Z", "+00:00")) >= cutoff
                ]
                if not filtered:
                    st.caption("No games in this period.")
                else:
                    df_mmr = pd.DataFrame([
                        {"Game": i + 1, "MMR": g["mmr_after"], "Date": g["time"][:10]}
                        for i, g in enumerate(filtered)
                    ])
                    peak_idx = df_mmr["MMR"].idxmax()
                    df_peak = df_mmr.loc[[peak_idx]]

                    milestone_rows = []
                    crossed = set()
                    for i, g in enumerate(filtered):
                        threshold = (g["mmr_after"] // 1000) * 1000
                        if g["mmr_before"] < threshold <= g["mmr_after"] and threshold not in crossed:
                            crossed.add(threshold)
                            milestone_rows.append({"Game": i + 1, "MMR": g["mmr_after"], "Milestone": f"{threshold:,}", "Date": g["time"][:10]})
                    df_milestones = pd.DataFrame(milestone_rows) if milestone_rows else pd.DataFrame(columns=["Game", "MMR", "Milestone", "Date"])

                    nearest = alt.selection_point(nearest=True, on="mouseover", encodings=["x"], empty=False)
                    line = (
                        alt.Chart(df_mmr)
                        .mark_line(color="#7ab87a")
                        .encode(
                            x=alt.X("Game:Q", title="Game"),
                            y=alt.Y("MMR:Q", title="MMR", scale=alt.Scale(zero=False, padding=20)),
                        )
                    )
                    hover_points = (
                        alt.Chart(df_mmr)
                        .mark_point(color="#7ab87a", size=60, filled=True)
                        .encode(
                            x="Game:Q",
                            y="MMR:Q",
                            opacity=alt.condition(nearest, alt.value(1), alt.value(0.15)),
                            tooltip=[alt.Tooltip("Game:Q", title="Game"), alt.Tooltip("MMR:Q", title="MMR"), alt.Tooltip("Date:N", title="Date")],
                        )
                        .add_params(nearest)
                    )
                    rule = (
                        alt.Chart(df_mmr)
                        .mark_rule(color="#444", strokeWidth=1)
                        .encode(x="Game:Q")
                        .transform_filter(nearest)
                    )
                    peak_dot = (
                        alt.Chart(df_peak)
                        .mark_point(color="#d4a843", size=80, filled=True)
                        .encode(
                            x="Game:Q",
                            y="MMR:Q",
                            tooltip=[alt.Tooltip("Game:Q", title="Game"), alt.Tooltip("MMR:Q", title="Peak MMR"), alt.Tooltip("Date:N", title="Date")],
                        )
                    )
                    milestone_dots = (
                        alt.Chart(df_milestones)
                        .mark_point(color="#aaaaaa", size=40, filled=True, opacity=0.6)
                        .encode(
                            x="Game:Q",
                            y="MMR:Q",
                            tooltip=[alt.Tooltip("Milestone:N", title="First time crossing"), alt.Tooltip("Game:Q", title="Game"), alt.Tooltip("Date:N", title="Date")],
                        )
                    )
                    chart = line + milestone_dots + peak_dot + rule + hover_points
                    st.altair_chart(chart.properties(height=250).configure_view(strokeWidth=0), use_container_width=True)

                if st.button("Compare with leaderboard neighbors", use_container_width=True):
                    st.session_state["nb_result"] = None
                    player_rank = st.session_state.get("sp_rank")

                    if player_rank is None:
                        st.session_state["nb_result"] = {"error": "Ingen rank hittades för den här spelaren — de kanske inte är på leaderboard."}
                    else:
                        with st.spinner(f"Finding neighbors around rank {player_rank}..."):
                            names_above, names_below, ranks_above, ranks_below = fetch_neighbor_names(
                                player_rank, sp_region
                            )

                        all_names = names_above + names_below
                        all_ranks = ranks_above + ranks_below

                        if not all_names:
                            st.session_state["nb_result"] = {"error": "No neighbors found nearby."}
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
                                    ng, _, _rank = fetch_and_calculate(name, sp_region)
                                    if len(ng) >= MIN_GAMES_NEIGHBOR:
                                        all_pcts.append(norm_to_pct(ng))
                                except Exception:
                                    failed.append(name)
                                progress.progress((i + 1) / len(all_names))

                            progress.empty()
                            status.empty()

                            if not all_pcts:
                                st.session_state["nb_result"] = {"error": f"No neighbors had {MIN_GAMES_NEIGHBOR}+ games to compare."}
                            else:
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

                        player_pct = norm_to_pct(games)
                        avg_pct    = {p: sum(d[p] for d in nb["pcts"]) / len(nb["pcts"]) for p in range(1, 9)}

                        cells = ""
                        for p in range(1, 9):
                            diff  = player_pct[p] - avg_pct[p]
                            color = diff_pct_color(diff)
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

                with st.expander("View as table"):
                    rows = [{"Place": p, "Count": norm[p], "%": f"{norm[p]/total*100:.1f}%"} for p in range(1, 9)]
                    st.table(rows)

            except Exception as e:
                st.error(str(e))


# ── RatingAvg tab (CSV) ───────────────────────────────────────────────────────

with tabs[1]:
    st.info("Ignore this, just backend stuff for debug/testign. Used for estimating expected average placement at a given MMR based on currently uploaded CSV curves (regression between MMR and avgPlace) for some of the values.")

with tabs[1]:
    rr = st.selectbox("Curve region (CSV)", ["EU"], index=0)

    if rr == "Fallback export.csv":
        csv_path = Path(__file__).parent / DEFAULT_CSV_NAME
    else:
        csv_path = get_csv_for_region(rr)

    if not csv_path.exists():
        st.error(
            f"Can't find CSV: {csv_path}\n\n"
            f"Put it next to this file. Expected columns: lb_mmr, current_mmr, avg_place, games."
        )
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
        max_games         = int(pd.to_numeric(df["games"], errors="coerce").max() or 0)
        default_min_games = 300 if max_games >= 300 else max_games
        min_games         = st.slider("Min games", min_value=0, max_value=max_games, value=default_min_games, step=50)

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
        f"Estimate Avg Place at {x_choice}  ·  using {csv_path.name}</div>",
        unsafe_allow_html=True
    )

    q_mmr = st.number_input(
        label="MMR",
        min_value=float(np.min(bx)) - 5000.0,
        max_value=float(np.max(bx)) + 5000.0,
        value=float(np.percentile(bx, 75)),
        step=100.0,
        label_visibility="collapsed",
    )

    est = interp_with_extrap(q_mmr, bx, by)
    est = float(np.clip(est, 1.0, 8.0))

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

    # streamlit run wa2_app.py
    # streamlit run wa2_app.py --server.runOnSave true (auto-reload on save)

# ── Footer ────────────────────────────────────────────────────────────────────
_all_rows = _sb_fetch_all()
_ts_values = [r["updated_at"] for r in _all_rows if r.get("updated_at")]
if _ts_values:
    _latest_ts = max(_ts_values)
    _st_components.html(
        f"""<div style='text-align:center;color:#333;font-size:0.75rem;font-family:sans-serif;' id="fts">Data last updated ...</div>
<script>
var d = new Date("{_latest_ts}");
document.getElementById("fts").textContent = "Data last updated " + d.toLocaleString(undefined, {{day:"numeric",month:"short",year:"numeric",hour:"2-digit",minute:"2-digit"}}) + "  \u00b7  v{APP_VERSION}";
</script>""",
        height=30,
    )
