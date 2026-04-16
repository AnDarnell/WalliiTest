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

SEASONS = {
    12: {"start": "2025-12-01", "end": "2026-04-13"},
    13: {"start": "2026-04-14", "end": None},  # None = pågående säsong
    }
CURRENT_SEASON = 13
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

TWITCH_CLIENT_ID     = st.secrets.get("TWITCH_CLIENT_ID", "")
TWITCH_CLIENT_SECRET = st.secrets.get("TWITCH_CLIENT_SECRET", "")
YOUTUBE_API_KEY      = st.secrets.get("YOUTUBE_API_KEY", "")

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

def get_threshold(snapshot_time_str, season_start_str=None):
    if season_start_str is None:
        season_start_str = SEASONS[CURRENT_SEASON]["start"]
    season_start = datetime.fromisoformat(season_start_str).replace(tzinfo=timezone.utc)
    game_time    = datetime.fromisoformat(snapshot_time_str)
    if game_time.tzinfo is None:
        game_time = game_time.replace(tzinfo=timezone.utc)
    days_in = max(0, (game_time - season_start).days)
    return THRESHOLD_BASE + (days_in // 20) * THRESHOLD_INCREASE

def est_place(mmr, gain, snapshot_time=None, season_start_str=None):
    mmr  = float(mmr)
    gain = float(gain)
    placements = [1, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    dex_avg    = mmr if mmr < 8200 else (mmr - 0.85 * (mmr - 8200))
    threshold  = get_threshold(snapshot_time, season_start_str) if snapshot_time else THRESHOLD_BASE
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
    st.session_state.pop("h2h_games", None)
    st.session_state.pop("h2h_label", None)
    st.session_state.pop("h2h_region", None)
    st.session_state.pop("h2h_error", None)
    st.session_state.pop("nb_result", None)

# ── Query-param navigation (from leaderboard name links) ─────────────────────
_qp = st.query_params
if "goto_player" in _qp:
    st.session_state["sp_player"] = _qp["goto_player"].lower()
    st.session_state["sp_region"] = _qp.get("goto_region", "EU")
    st.session_state.pop("sp_games", None)
    st.session_state.pop("sp_rank", None)
    st.session_state.pop("h2h_games", None)
    st.session_state.pop("h2h_label", None)
    st.session_state.pop("h2h_region", None)
    st.session_state.pop("h2h_error", None)
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
        "season":       record.get("season", CURRENT_SEASON),
        "games":        record.get("games"),
        "hot_streak":   record.get("hot_streak"),
        "roach_streak": record.get("roach_streak"),
        "first_pct":    record.get("first_pct"),
        "top4_pct":     record.get("top4_pct"),
        "tilt_factor":  record.get("tilt_factor"),
        "avg_place":    record.get("avg_place"),
        "form_diff":    record.get("form_diff"),
        "form_rating":  record.get("form_rating"),
        "max_drawdown": record.get("max_drawdown"),
        "dd_detail":      record.get("dd_detail"),
        "first_10k_date": record.get("first_10k_date"),
        "cr":             record.get("cr"),
        "u_score":        record.get("u_score"),
        "bot2_count":       record.get("bot2_count"),
        "mmr_milestones":    record.get("mmr_milestones"),
        **({"matchup_scaling": record["matchup_scaling"]} if record.get("matchup_scaling") is not None else {}),
        "updated_at":        datetime.utcnow().isoformat() + "Z",
    }

    if not SUPABASE_ENABLED:
        if DEBUG:
            st.session_state["sb_upsert_status"] = ("DISABLED", "SUPABASE_URL/KEY missing.")
        return

    try:
        url = f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}?on_conflict=player,region,season"   
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

_ALL_FIELDS = "player,region,season,games,first_pct,top4_pct,hot_streak,roach_streak,tilt_factor,avg_place,form_diff,form_rating,max_drawdown,dd_detail,first_10k_date,cr,u_score,bot2_count,mmr_milestones,matchup_scaling,updated_at"

@st.cache_data(show_spinner=False, ttl=300)
def _sb_fetch_all(season=CURRENT_SEASON):
    if not SUPABASE_ENABLED:
        return []
    try:
        url = f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}"
        r = requests.get(
            url,
            headers=SUPABASE_HEADERS,
            params={"select": _ALL_FIELDS, "season": f"eq.{season}", "limit": "2000"},
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

@st.cache_data(show_spinner=False, ttl=300)
def _country_flag(code):
    if not code or len(code) != 2:
        return ""
    c = code.lower()
    return f"<img src='https://flagcdn.com/16x12/{c}.png' alt='{code.upper()}' style='vertical-align:middle;margin-left:4px;border-radius:1px;'>"

@st.cache_data(show_spinner=False, ttl=300)
def _sb_fetch_player_links():
    """Returns dict {player_name_lower: {twitch_url, youtube_url, nationality}}."""
    for cols in ("player_name,twitch_url,youtube_url,nationality", "player_name,twitch_url,youtube_url"):
        try:
            r = requests.get(
                f"{SUPABASE_URL}/rest/v1/player_links",
                headers=SUPABASE_HEADERS,
                params={"select": cols},
                timeout=10,
            )
            r.raise_for_status()
            return {row["player_name"].lower(): row for row in r.json()}
        except Exception:
            continue
    return {}

@st.cache_data(show_spinner=False, ttl=120)
def _twitch_get_token():
    """Hämtar ett app access token från Twitch."""
    if not TWITCH_CLIENT_ID or not TWITCH_CLIENT_SECRET:
        return None
    try:
        r = requests.post(
            "https://id.twitch.tv/oauth2/token",
            params={"client_id": TWITCH_CLIENT_ID, "client_secret": TWITCH_CLIENT_SECRET, "grant_type": "client_credentials"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=120)
def _twitch_get_live_streams():
    """Returnerar lista med live-streams för spelare i player_links, sorterade efter viewer-antal."""
    token = _twitch_get_token()
    if not token:
        return []
    links = _sb_fetch_player_links()
    usernames = []
    player_by_login = {}
    for player, row in links.items():
        url = row.get("twitch_url", "")
        if url:
            login = url.rstrip("/").split("/")[-1].lower().strip()
            usernames.append(login)
            player_by_login[login] = player
    if not usernames:
        return []
    # Hämta stats från alla säsonger, prioritera nyaste
    _all_stats = []
    for _s in sorted(SEASONS.keys()):
        _all_stats += _sb_fetch_all(season=_s)
    stats_by_player = {}
    for r in _all_stats:
        if r.get("player"):
            stats_by_player[r["player"].lower()] = r  # nyaste skriver över äldre
    try:
        params = [("user_login", u) for u in usernames]
        r = requests.get(
            "https://api.twitch.tv/helix/streams",
            headers={"Client-ID": TWITCH_CLIENT_ID, "Authorization": f"Bearer {token}"},
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        streams = r.json().get("data", [])
        result = []
        for s in streams:
            if "hearthstone" not in s.get("game_name", "").lower():
                continue
            login = s["user_login"].lower()
            player = player_by_login.get(login, s["user_name"])
            pstats = stats_by_player.get(player.lower(), {})
            result.append({
                "player":       player,
                "login":        login,
                "title":        s["title"],
                "viewers":      s["viewer_count"],
                "region":       pstats.get("region", ""),
                "cr":           pstats.get("cr") or 0,
                "nationality":  links.get(player.lower(), {}).get("nationality", ""),
                "twitch_url":   links.get(player_by_login.get(login, ""), {}).get("twitch_url", f"https://twitch.tv/{login}"),
            })
        result.sort(key=lambda x: x["cr"], reverse=True)
        result = result[:10]


        return result
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=3600)
def _yt_fetch_subscribers():
    """Returnerar lista med {player, youtube_url, nationality, subscribers} sorterad efter subscribers."""
    if not YOUTUBE_API_KEY:
        return []
    links = _sb_fetch_player_links()
    entries = [(player, row) for player, row in links.items() if row.get("youtube_url")]
    if not entries:
        return []

    def _parse_yt_url(url):
        url = url.rstrip("/")
        if "/@" in url:
            handle = url.split("/@")[-1]
            return ("forHandle", f"@{handle}")
        if "/channel/" in url:
            cid = url.split("/channel/")[-1]
            return ("id", cid)
        if "/user/" in url:
            uname = url.split("/user/")[-1]
            return ("forUsername", uname)
        # Fallback: sista segmentet som handle (t.ex. youtube.com/Shadybunny)
        last = url.split("/")[-1]
        if last and "youtube.com" in url:
            return ("forHandle", f"@{last}")
        return None

    result = []
    for player, row in entries:
        parsed = _parse_yt_url(row["youtube_url"])
        if not parsed:
            continue
        param_key, param_val = parsed
        try:
            r = requests.get(
                "https://www.googleapis.com/youtube/v3/channels",
                params={"part": "statistics,snippet", param_key: param_val, "key": YOUTUBE_API_KEY},
                timeout=10,
            )
            r.raise_for_status()
            items = r.json().get("items", [])
            if not items:
                continue
            subs = int(items[0]["statistics"].get("subscriberCount", 0))
            result.append({
                "player":      player,
                "youtube_url": row["youtube_url"],
                "nationality": row.get("nationality", ""),
                "subscribers": subs,
            })
        except Exception:
            continue
    result.sort(key=lambda x: x["subscribers"], reverse=True)
    return result

@st.cache_data(show_spinner=False, ttl=60)
def _sb_top_n(metric, n=TOP_N, higher_is_better=True, season=CURRENT_SEASON):
    rows = [r for r in _sb_fetch_all(season=season) if r.get(metric) is not None]
    rows.sort(key=lambda r: (r[metric], r.get("cr") or 0), reverse=higher_is_better)
    return rows[:n]

# ---------- unified API ----------

def lb_upsert_player(region, player_name, record: dict):
    if TOPLIST_BACKEND == "supabase":
        _sb_upsert(region, player_name, record)
    else:
        _session_upsert(region, player_name, record)

def compute_and_upsert(player_name, region, games, season=CURRENT_SEASON):
    if not games:
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
    _tilt_diffs = []
    for i, p in enumerate(placements):
        if p >= 7:
            before = placements[max(0, i-50):i]
            after  = placements[i+1:i+4]
            if len(before) >= 10 and len(after) >= 1:
                _tilt_diffs.append(sum(after)/len(after) - sum(before)/len(before))

    tilt_factor_val = None
    if len(_tilt_diffs) >= 3 and avg > 0:
        tilt_factor_val = float(1 + (sum(_tilt_diffs) / len(_tilt_diffs) / avg) * 2)

    form_diff   = None
    form_rating = None
    if total >= 60:
        recent_avg = sum(g["placement"] for g in games[-50:]) / 50
        form_diff  = recent_avg - avg
        try:
            _bx, _by = _sb_load_regression("ALL")
            if _bx is not None and len(_bx) >= 2:
                form_rating = int(round(float(np.interp(recent_avg, _by[::-1], _bx[::-1]))))
        except Exception:
            pass

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
        f"({dd_peak_game['time'][:10]} - {dd_trough_game['time'][:10]})"
    )

    first_10k_date = None
    _mmr_milestones = {}
    for g in games:
        mmr = g["mmr_after"]
        if first_10k_date is None and mmr >= 10000:
            first_10k_date = g["time"]
        for _thresh in range(10000, 22000, 1000):
            if str(_thresh) not in _mmr_milestones and mmr >= _thresh:
                _mmr_milestones[str(_thresh)] = g["time"]

    if total < 50:
        if first_10k_date is not None:
            lb_upsert_player(region, player_name, {
                "season":         season,
                "games":          int(total),
                "first_10k_date": first_10k_date,
                "cr":             int(current_mmr),
                "mmr_milestones": json.dumps(_mmr_milestones),
                "updated_at":     datetime.utcnow().isoformat() + "Z",
            })
        return

    lb_upsert_player(region, player_name, {
        "season":          season,
        "games":           int(total),
        "hot_streak":      int(longest_streak),
        "roach_streak":    int(longest_roach),
        "first_pct":       float(wins / total * 100),
        "top4_pct":        float(top4 / total * 100),
        "tilt_factor":     tilt_factor_val,
        "avg_place":       float(avg),
        "form_diff":       float(form_diff) if form_diff is not None else None,
        "form_rating":     form_rating,
        "max_drawdown":    int(max_dd),
        "dd_detail":       dd_detail,
        "first_10k_date":  first_10k_date,
        "cr":              int(current_mmr),
        "u_score":         float(u_score),
        "bot2_count":      int(bot2_count),
        "mmr_milestones":  json.dumps(_mmr_milestones),
        "matchup_scaling": compute_matchup_scaling(games),
        "updated_at":      datetime.utcnow().isoformat() + "Z",
    })
    _save_opp_buckets(player_name, region, games)

def compute_matchup_scaling(games):
    """Returns matchup scaling score or None if insufficient data."""
    games_10k = [g for g in games if g.get("mmr_before", 0) >= 10000]
    if len(games_10k) < 300:
        return None
    buckets = {}
    for g in games_10k:
        gp, gmmr, ggain = g.get("placement"), g.get("mmr_before"), g.get("gain")
        if gp is None or gmmr is None or ggain is None:
            continue
        avg_opp = gmmr - 148.1181435 * (100 - ((gp - 1) * (200 / 7) + ggain))
        bucket = int(avg_opp // 1000) * 1000
        if bucket not in buckets:
            buckets[bucket] = {"placements": [], "expected": []}
        buckets[bucket]["placements"].append(gp)
        buckets[bucket]["expected"].append(1 + (7 / 200) * (100 - (gmmr - avg_opp) / 148.1181435))
    scaling_buckets = [7000, 8000, 9000, 10000]
    points = []
    for sbk in scaling_buckets:
        bv = buckets.get(sbk)
        if not bv or len(bv["placements"]) < 30 or not bv["expected"]:
            return None
        bavg = sum(bv["placements"]) / len(bv["placements"])
        bexp = sum(bv["expected"]) / len(bv["expected"])
        points.append((sbk + 500, bexp - bavg))
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    ws = np.array([0.5, 1.0, 1.0, 0.5])
    xs_norm = (xs - xs.mean()) / xs.std()
    return float(np.polyfit(xs_norm, ys, 1, w=ws)[0])

def _save_opp_buckets(player_name, region, games):
    if not SUPABASE_ENABLED or not games:
        return
    buckets = {}
    for g in games:
        gp = g.get("placement")
        gmmr = g.get("mmr_before")
        ggain = g.get("gain")
        if gp is None or gmmr is None or ggain is None:
            continue
        avg_opp = gmmr - 148.1181435 * (100 - ((gp - 1) * (200 / 7) + ggain))
        bucket = int(avg_opp // 1000) * 1000
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(gp)
    payload = [
        {
            "player": player_name.lower(),
            "region": region.upper(),
            "bucket_start": bk,
            "avg_placement": round(sum(bv) / len(bv), 4),
            "game_count": len(bv),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        for bk, bv in buckets.items() if len(bv) >= 1
    ]
    if not payload:
        return
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/player_opp_buckets",
            headers={**SUPABASE_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"},
            json=payload,
            timeout=10,
        ).raise_for_status()
    except Exception:
        pass

def lb_top_n(metric, n=TOP_N, higher_is_better=True, season=CURRENT_SEASON):
    if TOPLIST_BACKEND == "supabase":
        return _sb_top_n(metric, n, higher_is_better, season=season)
    else:
        return _session_top_n(metric, n, higher_is_better)


# ── Single-player fetch & calculate ───────────────────────────────────────────

def _snapshots_to_games(snapshots, season_start_str=None):
    games = []
    for i in range(1, len(snapshots)):
        prev, curr = snapshots[i - 1], snapshots[i]
        gain = curr["rating"] - prev["rating"]
        games.append({
            "mmr_before": prev["rating"],
            "mmr_after":  curr["rating"],
            "gain":       gain,
            "placement":  est_place(prev["rating"], gain, snapshot_time=curr["snapshot_time"], season_start_str=season_start_str),
            "time":       curr["snapshot_time"],
        })
    return games

def _sb_fetch_snapshots_range(player_name, region, date_from, date_to=None):
    """Hämtar snapshots från Supabase för ett givet datumintervall."""
    player_name = player_name.lower()
    rows = []
    offset = 0
    while True:
        # Bygg params som lista av tupler för att kunna ha två snapshot_time-filter
        params = [
            ("player_name", f"eq.{player_name}"),
            ("region",      f"eq.{region.upper()}"),
            ("game_mode",   "eq.0"),
            ("snapshot_time", f"gte.{date_from}"),
            ("order",       "snapshot_time.asc"),
            ("limit",       "1000"),
            ("offset",      str(offset)),
            ("select",      "snapshot_time,rating"),
        ]
        if date_to:
            params.append(("snapshot_time", f"lte.{date_to}"))
        sr = requests.get(
            f"{SUPABASE_URL}/rest/v1/snapshots",
            headers=SUPABASE_HEADERS,
            params=params,
            timeout=10,
        )
        sr.raise_for_status()
        batch = sr.json()
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return rows if rows else None


def _sb_get_cached_snapshots(player_name, region, season=CURRENT_SEASON):
    """Returns (snapshots, last_fetched, cached_rank) from Supabase, or (None, None, None)."""
    season_cfg  = SEASONS[season]
    date_from   = season_cfg["start"]
    date_to     = season_cfg["end"]

    # Historiska säsonger: data är komplett, skippa cache-ålder-check
    if season != CURRENT_SEASON:
        try:
            rows = _sb_fetch_snapshots_range(player_name, region, date_from, date_to)
            return rows, None, None
        except Exception:
            return None, None, None

    # Pågående säsong: kolla cache-ålder
    player_name = player_name.lower()
    try:
        cr = requests.get(
            f"{SUPABASE_URL}/rest/v1/player_cache",
            headers=SUPABASE_HEADERS,
            params={"player_name": f"eq.{player_name}", "region": f"eq.{region.upper()}", "select": "last_fetched,current_rank"},
            timeout=10,
        )
        cr.raise_for_status()
        cache_rows = cr.json()
        if not cache_rows:
            return None, None, None
        last_fetched = cache_rows[0]["last_fetched"]
        cached_rank  = cache_rows[0].get("current_rank")
        age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(last_fetched.replace("Z", "+00:00"))).total_seconds() / 3600
        if age_hours >= 12:
            return None, None, None
        rows = _sb_fetch_snapshots_range(player_name, region, date_from, date_to)
        return rows, last_fetched, cached_rank
    except Exception:
        return None, None, None


def _sb_get_cached_games(player_name, region, season=CURRENT_SEASON):
    """Returns (games, cached_rank) using only Supabase snapshot cache."""
    season_start_str = SEASONS[season]["start"]
    snapshots, _, cached_rank = _sb_get_cached_snapshots(player_name, region, season=season)
    if not snapshots:
        return None, cached_rank
    return _snapshots_to_games(snapshots, season_start_str=season_start_str), cached_rank

def _sb_save_snapshots(player_name, region, snapshots, current_rank=None):
    """Save snapshots to Supabase in batches, then update player_cache."""
    player_name = player_name.lower()
    rows = [
        {"player_name": player_name, "region": region.upper(), "snapshot_time": s["snapshot_time"], "rating": s["rating"], "game_mode": "0"}
        for s in snapshots
    ]
    for i in range(0, len(rows), 100):
        requests.post(
            f"{SUPABASE_URL}/rest/v1/snapshots?on_conflict=player_name,region,snapshot_time",
            headers={**SUPABASE_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"},
            json=rows[i:i + 100], timeout=30,
        ).raise_for_status()
    requests.post(
        f"{SUPABASE_URL}/rest/v1/player_cache?on_conflict=player_name,region",
        headers={**SUPABASE_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"},
        json={"player_name": player_name, "region": region.upper(), "last_fetched": datetime.now(timezone.utc).isoformat(), "current_rank": current_rank},
        timeout=10,
    ).raise_for_status()

@st.cache_data(show_spinner=False, ttl=300)
def _sb_load_regression(region):
    """Load regression curve from Supabase. Returns (bx, by) or (None, None)."""
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/regression_cache",
            headers=SUPABASE_HEADERS,
            params={"region": f"eq.{region.upper()}", "select": "bx_json,by_json"},
            timeout=10,
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return None, None
        bx = np.array(json.loads(rows[0]["bx_json"]))
        by = np.array(json.loads(rows[0]["by_json"]))
        return bx, by
    except Exception:
        return None, None

def _sb_save_regression(region, bx, by, n_players):
    """Save regression curve to Supabase."""
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/regression_cache?on_conflict=region",
            headers={**SUPABASE_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"},
            json={
                "region":     region.upper(),
                "bx_json":    json.dumps(bx.tolist()),
                "by_json":    json.dumps(by.tolist()),
                "n_players":  n_players,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            timeout=10,
        )
    except Exception:
        pass

def _compute_player_stats(games):
    """Compute all displayable stats from a games list."""
    if not games:
        return None
    norm        = normalized_counts(games)
    total       = len(games)
    avg         = sum(g["placement"] for g in games) / total
    wins        = norm[1]
    top4        = sum(norm[p] for p in [1, 2, 3, 4])
    current_mmr = games[-1]["mmr_after"]
    peak_mmr    = max(max(g["mmr_before"] for g in games), max(g["mmr_after"] for g in games))

    max_dd, peak_so_far = 0, games[0]["mmr_after"]
    for g in games:
        if g["mmr_after"] > peak_so_far:
            peak_so_far = g["mmr_after"]
        dd = peak_so_far - g["mmr_after"]
        if dd > max_dd:
            max_dd = dd

    longest_streak, streak = 0, 0
    for g in games:
        streak = streak + 1 if round(g["placement"]) == 1 else 0
        longest_streak = max(longest_streak, streak)

    longest_roach, roach = 0, 0
    for g in games:
        roach = roach + 1 if round(g["placement"]) <= 4 else 0
        longest_roach = max(longest_roach, roach)

    form_diff = None
    if total >= 60:
        recent_avg = sum(g["placement"] for g in games[-50:]) / 50
        form_diff  = recent_avg - avg

    placements, _tilt_diffs = [round(g["placement"]) for g in games], []
    for i, p in enumerate(placements):
        if p >= 7:
            before = placements[max(0, i-50):i]
            after  = placements[i+1:i+4]
            if len(before) >= 10 and len(after) >= 1:
                _tilt_diffs.append(sum(after)/len(after) - sum(before)/len(before))
    tilt_factor = None
    if len(_tilt_diffs) >= 3 and avg > 0:
        tilt_factor = float(1 + (sum(_tilt_diffs) / len(_tilt_diffs) / avg) * 2)

    _eps   = 0.5
    _part1 = np.log((norm[1] + _eps) / (norm[2] + norm[3] + norm[4] + _eps))
    _part2 = np.log((norm[7] + norm[8] + _eps) / (norm[5] + norm[6] + _eps))
    u_score = 0.5 * (_part1 + _part2)

    _ms = compute_matchup_scaling(games)
    farmer_factor = -_ms if _ms is not None else None

    return {
        "total":         total,
        "avg":           avg,
        "first_pct":     wins / total * 100,
        "top4_pct":      top4 / total * 100,
        "current_mmr":   current_mmr,
        "peak_mmr":      peak_mmr,
        "max_drawdown":  max_dd,
        "hot_streak":    longest_streak,
        "roach_streak":  longest_roach,
        "form_diff":     form_diff,
        "tilt_factor":   tilt_factor,
        "u_score":       u_score,
        "farmer_factor": farmer_factor,
    }

def fetch_and_calculate(player_name, region, season=CURRENT_SEASON):
    season_cfg       = SEASONS[season]
    season_start_str = season_cfg["start"]
    season_end_str   = season_cfg["end"]

    # ── 1. Historisk säsong: hämta bara från Supabase, inget wallii-anrop ────
    if season != CURRENT_SEASON:
        if SUPABASE_ENABLED:
            cached, _, _ = _sb_get_cached_snapshots(player_name, region, season=season)
            if cached and len(cached) >= 2:
                return _snapshots_to_games(cached, season_start_str=season_start_str), region, None
        raise ValueError(f"No cached data found for season {season}.")

    # ── 2. Pågående säsong: försök Supabase-cache först ──────────────────────
    if SUPABASE_ENABLED:
        cached, _, cached_rank = _sb_get_cached_snapshots(player_name, region, season=season)
        if cached and len(cached) >= 2:
            return _snapshots_to_games(cached, season_start_str=season_start_str), region, cached_rank

    # ── 2. Fetch from wallii.gg ───────────────────────────────────────────────
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def _fetch(rgn):
        s = requests.Session()
        s.max_redirects = 5
        u = f"https://www.wallii.gg/stats/{player_name}?region={rgn.lower()}&mode=solo&view=all"
        return s.get(u, headers=headers, timeout=20)

    try:
        r = _fetch(region)
    except requests.exceptions.TooManyRedirects:
        raise ValueError("Player not found - check if correct region.")
    r.raise_for_status()

    match = re.search(r'\\"data\\":\[(\{\\"player_name.*?)\],\\"availableModes\\"', r.text, re.DOTALL)
    if not match:
        raise ValueError("Player not found - check spelling and region.")

    data_str      = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
    snapshots_all = json.loads("[" + data_str + "]")
    snapshots_all = [s for s in snapshots_all if s["game_mode"] == "0"]
    available_regions = list({s["region"].upper() for s in snapshots_all})
    snapshots     = [s for s in snapshots_all if s["region"].upper() == region.upper()]

    rank_match   = re.search(r'text-2xl text-white">(\d+)<', r.text)
    current_rank = int(rank_match.group(1)) if rank_match else None
    dlog("DEBUG rank_match:", rank_match, "| snippet:", repr(r.text[r.text.find("text-2xl"):r.text.find("text-2xl")+60]) if "text-2xl" in r.text else "text-2xl NOT FOUND")

    if not snapshots and available_regions:
        other = ", ".join(r for r in sorted(available_regions) if r != region.upper())
        raise ValueError(f"No data found for {region}. Player appears to be in: {other}.")

    snapshots = sorted(snapshots, key=lambda x: x["snapshot_time"])

    if len(snapshots) < 2:
        raise ValueError("Not enough snapshots to compute games.")

    # ── 3. Save to Supabase ───────────────────────────────────────────────────
    if SUPABASE_ENABLED:
        _sb_save_snapshots(player_name, region, snapshots, current_rank)

    return _snapshots_to_games(snapshots, season_start_str=season_start_str), region, current_rank


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


def fetch_top_n_for_scan(region, n=100):
    """Fetch top N player names for a region from wallii.gg leaderboard."""
    day_start = _latest_day_start(game_mode=0)
    url = f"{SUPABASE_BASE}/rest/v1/daily_leaderboard_stats"
    qs = urlencode([
        ("select",    "rank,rating,region,players!inner(player_name)"),
        ("region",    f"eq.{region.upper()}"),
        ("game_mode", "eq.0"),
        ("day_start", f"eq.{day_start}"),
        ("order",     "rank.asc"),
        ("limit",     str(n)),
    ])
    r = requests.get(f"{url}?{qs}", headers=_supabase_headers(), timeout=20)
    r.raise_for_status()
    rows = r.json()
    names = []
    for row in rows:
        p = row.get("players")
        name = (p[0].get("player_name") if isinstance(p, list) and p else p.get("player_name") if isinstance(p, dict) else None)
        if name:
            names.append(name)
    return names


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

    avg_place = sum(g["placement"] for g in games) / total
    ax.text(0.02, 0.97, f"Avg: {avg_place:.2f}", transform=ax.transAxes,
            ha="left", va="top", color="#aaa", fontsize=12)

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
    if delta <= -0.40:
        return "#4a8c5c"
    elif delta <= -0.20:
        return "#6aab6a"
    elif delta <= 0.02:
        return "#7ab87a"
    elif delta <= 0.20:
        return "#d4a843"
    elif delta <= 0.40:
        return "#c47a75"
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

# ── Card Browser ──────────────────────────────────────────────────────────────

from pathlib import Path

CARDS_ROOT = Path(__file__).parent / "cards"  # ändra till din mapp

TRIBES = [
    "beast", "demons", "dragons", "elementals",
    "mechs", "murloc", "naga", "neutral",
    "pirates", "quilboar", "undead",
]
TRIBE_LABELS = {t: t.capitalize() for t in TRIBES}
TIERS = [1, 2, 3, 4, 5, 6]


@st.cache_data(show_spinner=False)
def _get_minion_images(tribes, tiers):
    from collections import defaultdict
    by_name = defaultdict(lambda: defaultdict(list))

    # Skanna ALLA tribes för att hitta delade kort
    for tribe in TRIBES:
        folder = CARDS_ROOT / f"S13_{tribe}" / "minions"
        for tier in tiers:
            tier_folder = folder / f"tier {tier}"
            if tier_folder.exists():
                for img in sorted(tier_folder.glob("*.png")):
                    by_name[img.name][tier].append((tribe, img))

    # Filtrera: visa bara kort som tillhör minst en vald tribe
    result = []
    selected_set = set(tribes)
    for name, tiers_dict in by_name.items():
        for tier, entries in tiers_dict.items():
            tribes_list = [t for t, _ in entries]
            if not selected_set.intersection(tribes_list):
                continue
            path = entries[0][1]
            result.append((tribes_list, tier, path))

    result.sort(key=lambda x: (x[1], x[2].name))
    return result


@st.cache_data(show_spinner=False)
def _get_trinket_images(tribes, trinket_type):
    """trinket_type: 'trinket_greater' eller 'trinket_lesser'"""
    result = []
    for tribe in tribes:
        folder = CARDS_ROOT / f"S13_{tribe}" / trinket_type
        if folder.exists():
            for img in sorted(folder.glob("*.png")):
                result.append((tribe, img))
    return result


@st.cache_data(show_spinner=False)
def _get_spell_images():
    folder = CARDS_ROOT / "S13_spells"
    if not folder.exists():
        return []
    return list(sorted(folder.glob("*.png")))

@st.cache_data(show_spinner=False)
def _get_all_trinket_cards():
    """Cache all trinket image paths once — filesystem scan happens only on first call."""
    result = []
    for trinket_type in ("trinket_greater", "trinket_lesser"):
        for tribe in TRIBES:
            folder = CARDS_ROOT / f"S13_{tribe}" / trinket_type
            if folder.exists():
                for img in sorted(folder.glob("*.png")):
                    result.append({
                        "tribe": tribe,
                        "trinket_type": trinket_type,
                        "name": img.stem,
                        "path_str": str(img),
                    })
    return result

@st.cache_data(show_spinner=False)
def _get_all_minion_cards():
    """Cache all minion card metadata + paths once — filesystem scan happens only on first call."""
    from collections import defaultdict
    by_name = defaultdict(lambda: defaultdict(list))
    for tribe in TRIBES:
        folder = CARDS_ROOT / f"S13_{tribe}" / "minions"
        for tier in TIERS:
            tier_folder = folder / f"tier {tier}"
            if tier_folder.exists():
                for img in sorted(tier_folder.glob("*.png")):
                    by_name[img.stem][tier].append((tribe, img))
    result = []
    for name, tiers_dict in by_name.items():
        for tier, entries in tiers_dict.items():
            tribes_list = [t for t, _ in entries]
            path = entries[0][1]
            result.append({
                "name": name,
                "display_name": name.replace("_", " "),
                "tier": tier,
                "tribes": tribes_list,
                "path_str": str(path),
            })
    result.sort(key=lambda x: (x["tier"], x["name"]))
    return result


def show_card_browser():
    st.markdown("""
    <style>
    div[data-testid="stCheckbox"] label p { font-size: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## Card Library – Season 13")

    # ── Typ-väljare ───────────────────────────────────────────────────────────
    card_type = st.radio(
        "Type",
        ["Minions", "Trinkets", "Spells*"],
        horizontal=True,
        key="cb_type",
    )

    if card_type == "Spells*":
        images = _get_spell_images()
        if not images:
            st.info("Updating soon...")
            return
        cols_per_row = 6
        cols = st.columns(cols_per_row)
        for i, img in enumerate(images):
            cols[i % cols_per_row].image(str(img), use_container_width=True)
        return

    # ── Tribe & Tier filters ──────────────────────────────────────────────────
    _fcol1, _fcol2 = st.columns(2)
    tribe_options = ["All"] + [TRIBE_LABELS[t] for t in TRIBES]
    selected_label = _fcol1.selectbox("Tribe", tribe_options, index=tribe_options.index("Beast"), key="cb_tribe_select")
    if selected_label == "All":
        selected_tribes = list(TRIBES)
    else:
        selected_tribes = [t for t in TRIBES if TRIBE_LABELS[t] == selected_label]

    # ── Minions ───────────────────────────────────────────────────────────────
    # NY KOD:
    if card_type == "Minions":
        tier_options = ["All"] + [f"Tier {t}" for t in TIERS]
        selected_tier_label = _fcol2.selectbox("Tier", tier_options, index=0, key="cb_tier_select")
        selected_tiers = TIERS if selected_tier_label == "All" else [int(selected_tier_label.split(" ")[1])]

        all_cards = _get_all_minion_cards()  # cachat med b64
        selected_set = set(selected_tribes)
        filtered = [
            c for c in all_cards
            if c["tier"] in selected_tiers and selected_set.intersection(c["tribes"])
        ]

        if not filtered:
            st.info("Inga kort matchade filtret.")
            return

        for tier in selected_tiers:
            tier_cards = [c for c in filtered if c["tier"] == tier]
            if not tier_cards:
                continue
            st.markdown(f"### ⭐ Tier {tier}")
            cols_per_row = max(4, min(8, len(selected_tribes) * 2))
            cols = st.columns(cols_per_row)
            for i, card in enumerate(tier_cards):
                col = cols[i % cols_per_row]
                col.image(card["path_str"], use_container_width=True)

    # ── Trinkets ──────────────────────────────────────────────────────────────
    elif card_type == "Trinkets":
        trinket_type = st.radio(
            "Trinket-typ", ["Greater", "Lesser"], horizontal=True, key="cb_trinket_type"
        )
        folder_name = "trinket_greater" if trinket_type == "Greater" else "trinket_lesser"

        all_trinkets = _get_all_trinket_cards()
        selected_set = set(selected_tribes)
        filtered = [
            t for t in all_trinkets
            if t["trinket_type"] == folder_name and t["tribe"] in selected_set
        ]

        if not filtered:
            st.info("Inga trinkets matchade filtret.")
            return

        cols_per_row = max(4, min(8, len(selected_tribes) * 2))
        cols = st.columns(cols_per_row)
        for i, card in enumerate(filtered):
            cols[i % cols_per_row].image(card["path_str"], use_container_width=True)

# ── Page styling ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Placement Stats", layout="centered", page_icon="nerdbob2.png")
st.logo("nerdbob.png")

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
.lb-show-more button:disabled, .lb-show-more button[disabled] { text-decoration: line-through !important; opacity: 1 !important; }

.lb-hover-row {
    position: relative;
    overflow: visible !important;
}
.lb-hover-card {
    position: absolute;
    left: 0;
    top: calc(100% + 6px);
    min-width: 220px;
    max-width: 260px;
    background: rgba(14, 14, 14, 0.98);
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.45);
    opacity: 0;
    visibility: hidden;
    transform: translateY(-4px);
    transition: opacity 120ms ease, transform 120ms ease, visibility 120ms ease;
    pointer-events: none;
    z-index: 100;
}
.lb-hover-name {
    position: relative;
    display: inline-block;
}
.lb-hover-name:hover {
    z-index: 40;
}
.lb-hover-name:hover .lb-hover-card {
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
}
.lb-hover-title {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.75rem;
    color: #eee;
    font-size: 0.8rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}
.lb-hover-meta {
    color: #777;
    font-size: 0.72rem;
    font-weight: 500;
    white-space: nowrap;
}
.lb-hover-grid {
    display: grid;
    grid-template-columns: auto auto;
    column-gap: 0.8rem;
    row-gap: 0.22rem;
    align-items: baseline;
}
.lb-hover-label {
    color: #666;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.lb-hover-value {
    color: #ddd;
    font-size: 0.8rem;
    font-weight: 600;
    text-align: right;
}

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

import base64 as _b64
_logo_b64 = _b64.b64encode(open("nerdbob.png", "rb").read()).decode()
st.markdown(f"""
<div style='display:flex; align-items:center; gap:1rem; margin-bottom:1.0rem;'>
  <img src='data:image/png;base64,{_logo_b64}' style='height:72px; width:72px; object-fit:cover; border-radius:8px; flex-shrink:0;'>
  <div style='line-height:1.2;'>
    <div style='color:#eee; font-size:1.5rem; font-weight:normal; margin:0 0 0.2rem 0;'><a href='?goto_home=1' style='color:inherit;text-decoration:none;' onmouseover="this.style.opacity='0.7'" onmouseout="this.style.opacity='1'">Placement Statistics</a></div>
    <div style='color:#555; font-size:0.8rem; margin:0; text-transform:uppercase; letter-spacing:0.08em;'>Hearthstone Battlegrounds Leaderboard Stats</div>
  </div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["Single player", "Calculator", "Info/Explanations", "TestingStuff", "Card Library"])


# ── Single player tab ─────────────────────────────────────────────────────────

with tabs[0]:
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            player = st.text_input("Player", placeholder="Name")
        with col2:
            region = st.selectbox("Region", VALID_REGIONS, index=VALID_REGIONS.index("EU"))
        season_options = sorted(SEASONS.keys(), reverse=True)
        season_labels  = [f"Season {s}" + (" (current)" if SEASONS[s]["end"] is None else "") for s in season_options]
        default_season_index = season_options.index(13) if 13 in season_options else 0
        season_choice  = st.selectbox("Season", options=season_options, format_func=lambda s: f"Season {s}" + (" (current)" if SEASONS[s]["end"] is None else ""), index=default_season_index)
        submitted = st.form_submit_button("Search", width='stretch')

    if submitted and player:
        import time as _time
        _last  = st.session_state.get("last_search_time", 0)
        _count = st.session_state.get("search_count", 0)
        if _count >= 2 and _time.time() - _last < 15:
            st.warning("Please wait a moment before searching again.")
        else:
            st.session_state["last_search_time"] = _time.time()
            st.session_state["search_count"]     = _count + 1
            st.session_state["sp_player"] = player.strip().lower()
            st.session_state["sp_region"] = region
            st.session_state["sp_season"] = season_choice
            st.session_state["sp_games"]  = None
            st.session_state["sp_rank"]   = None
            st.session_state.pop("h2h_games", None)
            st.session_state.pop("h2h_label", None)
            st.session_state.pop("h2h_error", None)
            st.session_state["nb_result"] = None
            st.rerun()

    if submitted and not player:
        st.warning("Enter a player name.")


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

            @st.fragment
            def _leaderboard_section():
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

                    _plinks = _sb_fetch_player_links()
                    _twitch_svg = "<svg width='12' height='12' viewBox='0 0 24 24' fill='#9146FF' style='vertical-align:middle;margin-left:4px;'><path d='M11.571 4.714h1.715v5.143H11.57zm4.715 0H18v5.143h-1.714zM6 0L1.714 4.286v15.428h5.143V24l4.286-4.286h3.428L22.286 12V0zm14.571 11.143l-3.428 3.428h-3.429l-3 3v-3H6.857V1.714h13.714z'/></svg>"
                    _yt_svg     = "<svg width='12' height='12' viewBox='0 0 24 24' fill='#FF0000' style='vertical-align:middle;margin-left:4px;'><path d='M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z'/></svg>"
                    _live_players = {s["player"].lower(): s["twitch_url"] for s in _twitch_get_live_streams()}
                    _hover_bx, _hover_by = _sb_load_regression("ALL")

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
                        _pl    = _plinks.get(player.lower(), {})
                        _icons = ""
                        if _pl.get("twitch_url"):
                            _icons += f"<a href='{html.escape(_pl['twitch_url'])}' target='_blank' title='Twitch'>{_twitch_svg}</a>"
                        if _pl.get("youtube_url"):
                            _icons += f"<a href='{html.escape(_pl['youtube_url'])}' target='_blank' title='YouTube'>{_yt_svg}</a>"
                        _flag = _country_flag(_pl.get("nationality", ""))
                        _live_url = _live_players.get(player.lower())
                        _live_badge = (
                            f"<a href='{html.escape(_live_url)}' target='_blank' "
                            f"style='margin-left:6px;font-size:0.65rem;font-weight:700;color:#fff;"
                            f"background:#e53935;border-radius:3px;padding:1px 5px;text-decoration:none;"
                            f"vertical-align:middle;letter-spacing:0.04em;'>LIVE</a>"
                        ) if _live_url else ""
                        _region_meta = html.escape(str(region).upper()) if region else ""
                        _cr_meta = r.get("cr")
                        _meta = " ".join(x for x in [_region_meta, f"{int(_cr_meta):,}" if _cr_meta is not None else ""] if x)
                        _avg_val = r.get("avg_place")
                        _avg = f"{_avg_val:.2f}" if _avg_val is not None else "—"
                        _avg_color = "#777"
                        _hover_cr = r.get("cr")
                        if _avg_val is not None and _hover_cr is not None and _hover_bx is not None and len(_hover_bx) >= 2:
                            try:
                                _expected_avg = interp_with_extrap(_hover_cr, _hover_bx, _hover_by)
                                _expected_avg = float(np.clip(_expected_avg, 1.0, 8.0))
                                _avg_color = delta_color(float(_avg_val - _expected_avg))
                            except Exception:
                                _avg_color = "#ddd"
                        _top1 = f"{r['first_pct']:.1f}%" if r.get("first_pct") is not None else "—"
                        _top4 = f"{r['top4_pct']:.1f}%" if r.get("top4_pct") is not None else "—"
                        _u_val = r.get("u_score")
                        _u_color = (
                            "#b388e8" if _u_val is not None and _u_val >= 0.2
                            else "#aaa" if _u_val is not None and _u_val >= -0.1
                            else "#5b8fd4" if _u_val is not None
                            else "#777"
                        )
                        _u_text = f"{_u_val:+.2f}" if _u_val is not None else "—"
                        _ff_raw = r.get("matchup_scaling")
                        _ff_val = -_ff_raw if _ff_raw is not None else None
                        _ff_color = (
                            "#7ab87a" if _ff_val is not None and _ff_val >= 0.35
                            else "#d4a843" if _ff_val is not None and _ff_val >= -0.15
                            else "#c47a75" if _ff_val is not None
                            else "#777"
                        )
                        _ff_text = f"{_ff_val:+.2f}" if _ff_val is not None else "—"
                        _hover_card = (
                            f"<div class='lb-hover-card'>"
                            f"<div class='lb-hover-title'><span>{html.escape(player)}</span><span class='lb-hover-meta'>{_meta}</span></div>"
                            f"<div class='lb-hover-grid'>"
                            f"<span class='lb-hover-label'>Avg</span><span class='lb-hover-value' style='color:{_avg_color};'>{_avg}</span>"
                            f"<span class='lb-hover-label'>Top 1%</span><span class='lb-hover-value'>{_top1}</span>"
                            f"<span class='lb-hover-label'>Top 4%</span><span class='lb-hover-value'>{_top4}</span>"
                            f"<span class='lb-hover-label'>Aggression</span><span class='lb-hover-value' style='color:{_u_color};'>{_u_text}</span>"
                            f"<span class='lb-hover-label'>Farmer Factor</span><span class='lb-hover-value' style='color:{_ff_color};'>{_ff_text}</span>"
                            f"</div></div>"
                        )
                        return (
                            "<div class='lb-hover-row' style='display:flex;justify-content:space-between;"
                            "border:1px solid #1e1e1e;background:#121212;border-radius:4px;"
                            "padding:0.35rem 0.5rem;margin-bottom:0.25rem;'>"
                            f"<span style='color:{row_color};font-weight:700'>{i}. "
                            f"<span class='lb-hover-name'>"
                            f"<a href='{link}' target='_self' style='color:inherit;text-decoration:none;' "
                            f"onmouseover=\"this.style.textDecoration='underline'\" "
                            f"onmouseout=\"this.style.textDecoration='none'\">{player}</a>"
                            f"{_hover_card}</span> "
                            f"<span style='color:#666'>({region})</span>"
                            f"{'&nbsp;' + _flag if _flag else ''}{_icons}{_live_badge}</span>"
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
                            st.rerun(scope="fragment")
                        container.markdown("</div>", unsafe_allow_html=True)

                _lb_season_options = sorted(SEASONS.keys(), reverse=True)
                _lb_season_index = _lb_season_options.index(13) if 13 in _lb_season_options else 0
                st.markdown(
                    "<div style='border:1px solid #4a8c5c; background:#12221b; color:#d4e8d4; " \
                    "padding:0.75rem 1rem; border-radius:10px; margin-bottom:0.8rem; box-shadow:0 0 0 1px rgba(74,140,92,0.1);'>" \
                    "<strong style='display:block; color:#b8dfb8; margin-bottom:0.2rem;'>Note:</strong>" \
                    "Season 13 is out! You can still view stats from last season by ticking the box below. You can also still search for people's stats from last season." \
                    " The leaderboards might look weird for a while until people get enough games. Some metrics has a game-requirement (usually 50).</div>",
                    unsafe_allow_html=True,
                )
                _lb_season = st.radio(
                    "Leaderboard season",
                    options=_lb_season_options,
                    format_func=lambda s: f"Season {s}" + (" (current)" if SEASONS[s]["end"] is None else ""),
                    horizontal=True,
                    index=_lb_season_index,
                    key="lb_season_selector",
                    label_visibility="collapsed",
                )
                st.markdown(
                    f"<p style='color:#ccc;font-size:1.0rem;font-weight:600;margin:0.3rem 0 0.1rem;'>Leaderboards (Season {_lb_season}) <span style='color:#666;font-size:0.75rem;font-weight:400;'>(Players are added when first searched, if eligible)</span></p>",
                    unsafe_allow_html=True
                )
                _mmr_col, _eu_col, _na_col, _ap_col, _cn_col = st.columns([4, 1, 1, 1, 1])
                with _mmr_col:
                    _mmr_filter = st.radio("MMR filter", ["All", "Top 25", "Top 50"], index=0, horizontal=True, key="lb_mmr_filter", label_visibility="collapsed")
                with _eu_col:
                    _inc_eu = st.checkbox("EU", value=True,  key="lb_inc_eu")
                with _na_col:
                    _inc_na = st.checkbox("NA", value=True,  key="lb_inc_na")
                with _ap_col:
                    _inc_ap = st.checkbox("AP", value=True,  key="lb_inc_ap")
                with _cn_col:
                    _inc_cn = st.checkbox("CN", value=False, key="lb_inc_cn", help="CN sends inconsistent MMR updates, which means estimated placements may be slightly misleading in some cases.")

                _lb_regions = {r for r, v in [("EU", _inc_eu), ("NA", _inc_na), ("AP", _inc_ap), ("CN", _inc_cn)] if v}
                if _mmr_filter != "All":
                    _mmr_n = int(_mmr_filter.split()[1])
                    _all_by_mmr = sorted(_sb_fetch_all(season=_lb_season), key=lambda r: r.get("cr") or 0, reverse=True)
                    _top_mmr_players = set()
                    for _rgn in _lb_regions:
                        _rgn_rows = [r for r in _all_by_mmr if r.get("region") == _rgn]
                        _top_mmr_players.update(r["player"] for r in _rgn_rows[:_mmr_n])
                else:
                    _top_mmr_players = None

                def _lb(metric, higher_is_better=True, n=9999, limit=TOP_N):
                    rows = lb_top_n(metric, higher_is_better=higher_is_better, n=n, season=_lb_season)
                    rows = [r for r in rows if r.get("region") in _lb_regions]
                    if _top_mmr_players is not None:
                        rows = [r for r in rows if r.get("player") in _top_mmr_players]
                    return rows[:limit] if limit is not None else rows

                _all_stats_by_player = {r["player"].lower(): r for r in _sb_fetch_all(season=_lb_season) if r.get("player")}
                _hover_bx, _hover_by = _sb_load_regression("ALL")

                def _hover_card_html(player_name, stats_row):
                    if not stats_row:
                        return ""
                    _region = html.escape(str(stats_row.get("region", "")).upper()) if stats_row.get("region") else ""
                    _cr = stats_row.get("cr")
                    _meta = " ".join(x for x in [_region, f"{int(_cr):,}" if _cr is not None else ""] if x)
                    _avg_val = stats_row.get("avg_place")
                    _avg = f"{_avg_val:.2f}" if _avg_val is not None else "—"
                    _avg_color = "#777"
                    _hover_cr = stats_row.get("cr")
                    if _avg_val is not None and _hover_cr is not None and _hover_bx is not None and len(_hover_bx) >= 2:
                        try:
                            _expected_avg = interp_with_extrap(_hover_cr, _hover_bx, _hover_by)
                            _expected_avg = float(np.clip(_expected_avg, 1.0, 8.0))
                            _avg_color = delta_color(float(_avg_val - _expected_avg))
                        except Exception:
                            _avg_color = "#ddd"
                    _top1 = f"{stats_row['first_pct']:.1f}%" if stats_row.get("first_pct") is not None else "—"
                    _top4 = f"{stats_row['top4_pct']:.1f}%" if stats_row.get("top4_pct") is not None else "—"
                    _u_val = stats_row.get("u_score")
                    _u_color = (
                        "#b388e8" if _u_val is not None and _u_val >= 0.2
                        else "#aaa" if _u_val is not None and _u_val >= -0.1
                        else "#5b8fd4" if _u_val is not None
                        else "#777"
                    )
                    _u_text = f"{_u_val:+.2f}" if _u_val is not None else "—"
                    _ff_raw = stats_row.get("matchup_scaling")
                    _ff_val = -_ff_raw if _ff_raw is not None else None
                    _ff_color = (
                        "#7ab87a" if _ff_val is not None and _ff_val >= 0.35
                        else "#d4a843" if _ff_val is not None and _ff_val >= -0.15
                        else "#c47a75" if _ff_val is not None
                        else "#777"
                    )
                    _ff_text = f"{_ff_val:+.2f}" if _ff_val is not None else "—"
                    return (
                        f"<div class='lb-hover-card'>"
                        f"<div class='lb-hover-title'><span>{html.escape(player_name)}</span><span class='lb-hover-meta'>{_meta}</span></div>"
                        f"<div class='lb-hover-grid'>"
                        f"<span class='lb-hover-label'>Avg</span><span class='lb-hover-value' style='color:{_avg_color};'>{_avg}</span>"
                        f"<span class='lb-hover-label'>Top 1%</span><span class='lb-hover-value'>{_top1}</span>"
                        f"<span class='lb-hover-label'>Top 4%</span><span class='lb-hover-value'>{_top4}</span>"
                        f"<span class='lb-hover-label'>Aggression</span><span class='lb-hover-value' style='color:{_u_color};'>{_u_text}</span>"
                        f"<span class='lb-hover-label'>Farmer Factor</span><span class='lb-hover-value' style='color:{_ff_color};'>{_ff_text}</span>"
                        f"</div></div>"
                    )

                lists = [
                    ("Avg placement",       _lb("avg_place",    higher_is_better=False),  lambda r: f"{r['avg_place']:.2f}",    "Mean placement across all recorded games. Lower is better."),
                    ("Top 1 %",             _lb("first_pct",    higher_is_better=True),   lambda r: f"{r['first_pct']:.1f}%",   "Percentage of games finished in 1st place."),
                    ("Hot streak",          _lb("hot_streak",   higher_is_better=True),   lambda r: f"{int(r['hot_streak'])} games",   "Longest consecutive 1st streak of placement."),
                    ("Top 4 %",             _lb("top4_pct",     higher_is_better=True),   lambda r: f"{r['top4_pct']:.1f}%",    "Percentage of games finished in top 4."),
                    ("Roach streak",        _lb("roach_streak", higher_is_better=True),   lambda r: f"{int(r['roach_streak'])} games", "Longest consecutive streak of Top 4 place finishes."),
                    ("Lowest tilt factor",  [r for r in _lb("tilt_factor", higher_is_better=False, limit=None) if (r.get("bot2_count") or 0) >= 30][:TOP_N], lambda r: f"{r['tilt_factor']:.2f}<span style='color:#555;font-size:0.78em;margin-left:2px;'>x</span>" if r.get("tilt_factor") is not None else "—", "Measures how much a player is affected by a bad placement. The value shows how much worse their avg placement becomes after a 7th/8th compared to their overall avg. Lower = less affected by tilt.", "Min 30 games with 7th/8th placement"),
                    ("Highest tilt factor", [r for r in _lb("tilt_factor", higher_is_better=True,  limit=None) if (r.get("bot2_count") or 0) >= 30][:TOP_N], lambda r: f"{r['tilt_factor']:.2f}<span style='color:#555;font-size:0.78em;margin-left:2px;'>x</span>" if r.get("tilt_factor") is not None else "—", "Measures how much a player is affected by a bad placement. The value shows how much worse their avg placement becomes after a 7th/8th compared to their overall avg. Higher = more affected by tilt.", "Min 30 games with 7th/8th placement"),
                    ("Most aggressive",     _lb("u_score",      higher_is_better=True),   lambda r: f"{r['u_score']:+.2f}<span style='color:#555;font-size:0.85em;margin-left:3px;'>u</span>" if r.get("u_score") is not None else "—", "Measures play style based on placement distribution. Aggressive players finish at the extremes more often - more 1st and 7th/8th places - suggesting a high-risk, high-reward approach. Higher = more aggressive."),
                    ("Most defensive",      _lb("u_score",      higher_is_better=False),  lambda r: f"{r['u_score']:+.2f}<span style='color:#555;font-size:0.85em;margin-left:3px;'>&#8745;</span>" if r.get("u_score") is not None else "—", "Measures play style based on placement distribution. Defensive players finish in the middle more often - fewer 1st and 7th/8th places - suggesting a consistent, low-risk approach. Lower = more defensive."),
                    ("Best form",           _lb("form_diff",    higher_is_better=False),  lambda r: f"{(r['avg_place'] + r['form_diff']):.2f}<span style='color:#555;font-size:0.78em;margin-left:3px;'>avg</span> ({r['form_diff']:+.2f})" if r.get("form_diff") is not None and r.get("avg_place") is not None else "—", "Difference between form (last 50) and overall avg place. More negative = better form relative to baseline."),
                    ("Best 'form rating'",    [r for r in _lb("form_rating", higher_is_better=True, limit=None) if r.get("form_rating") is not None][:TOP_N], lambda r: f"{r['form_rating']:,}<span style='color:#555;font-size:0.78em;margin-left:3px;'>mmr</span>", "Estimated MMR based on last 50 games avg placement on the regression curve."),
                    ("Largest MMR drop",    _lb("max_drawdown", higher_is_better=True),   lambda r: f"<span title='{html.escape(r['dd_detail'])}' style='cursor:help;'>-{int(r['max_drawdown']):,} MMR</span>" if r.get("dd_detail") else (f"-{int(r['max_drawdown']):,} MMR" if r.get("max_drawdown") is not None else "—"), "Largest MMR drop from a peak to a subsequent low."),
                    ("# Games",             _lb("games",        higher_is_better=True),   lambda r: f"{int(r['games'])} games",  "Total number of games played this season while on the leaderboard."),
                    ("Farmer factor",       [r for r in _lb("matchup_scaling", higher_is_better=False, limit=None) if r.get("matchup_scaling") is not None][:TOP_N], lambda r: f"{-r['matchup_scaling']:+.2f}", "Measures how much better a player performs against weaker opponents relative to stronger ones. Higher = more dominant vs weaker lobbies. Experimental", "Experimental - min 300 games at 10k+ MMR"),
                    ("Lowest farmer factor", [r for r in _lb("matchup_scaling", higher_is_better=True,  limit=None) if r.get("matchup_scaling") is not None][:TOP_N], lambda r: f"{-r['matchup_scaling']:+.2f}", "Measures how much better a player performs against stronger opponents relative to weaker ones. Lower farmer factor = scales better with competition. Experimental.", "Experimental - min 300 games at 10k+ MMR"),

                ]

                # ── Rad 0: Avg placement (vänster) + Live Now (höger) ─────────────
                _row0 = st.columns(2)
                _t0, _it0, _fm0, _tp0, *_rst0 = lists[0]
                render_list(_row0[0], _t0, _it0, _fm0, _tp0, asterisk_tip=_rst0[0] if _rst0 else None)

                # ── Live streams (höger, rad 0) ────────────────────────────────────
                _live_streams_lb = [s for s in _twitch_get_live_streams() if not _lb_regions or s.get("region", "").upper() in _lb_regions]
                _live_col = _row0[1]
                HEADER_COLOR = "#c45151"
                _total_viewers = sum(s["viewers"] for s in _live_streams_lb)
                _viewers_str = f" <span style='color:#666;font-size:0.78em;font-weight:400;text-transform:none;letter-spacing:0;'>({_total_viewers:,} total viewers)</span>" if _total_viewers > 0 else ""
                _live_col.markdown(
                    f"<div style='color:{HEADER_COLOR};font-size:0.85rem;text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin:0.25rem 0 0.45rem;font-weight:600;'>"
                    f"<svg width='10' height='10' viewBox='0 0 24 24' fill='#9146FF' style='vertical-align:middle;margin-right:5px;'><path d='M11.571 4.714h1.715v5.143H11.57zm4.715 0H18v5.143h-1.714zM6 0L1.714 4.286v15.428h5.143V24l4.286-4.286h3.428L22.286 12V0zm14.571 11.143l-3.428 3.428h-3.429l-3 3v-3H6.857V1.714h13.714z'/></svg>"
                    f"Live now{_viewers_str}</div>",
                    unsafe_allow_html=True,
                )
                if "lb_live_sort" not in st.session_state:
                    st.session_state["lb_live_sort"] = "mmr"
                _live_sort_key = st.session_state["lb_live_sort"]
                _live_streams_lb = sorted(
                    _live_streams_lb,
                    key=lambda x: x["viewers"] if _live_sort_key == "viewers" else x["cr"],
                    reverse=True,
                )
                _empty_row = "<div style='display:flex;border:1px solid #1e1e1e;background:#121212;border-radius:4px;padding:0.35rem 0.5rem;margin-bottom:0.25rem;'><span style='color:#1e1e1e;'>—</span></div>"

                def _live_row_html(si, s):
                    s_twitch = html.escape(s["twitch_url"])
                    s_title  = html.escape(s["title"][:50] + ("..." if len(s["title"]) > 50 else ""))
                    s_color  = "#d4a843" if si == 1 else "#bfc4c8" if si == 2 else "#b57a4a" if si == 3 else "#8a8a8a"
                    s_profile = f"?goto_player={html.escape(s['player'])}&goto_region={html.escape(s.get('region',''))}"
                    _hover_card = _hover_card_html(s["player"], _all_stats_by_player.get(s["player"].lower()))
                    return (
                        f"<div class='lb-hover-row' style='display:flex;justify-content:space-between;"
                        f"border:1px solid #1e1e1e;background:#121212;border-radius:4px;"
                        f"padding:0.35rem 0.5rem;margin-bottom:0.25rem;'>"
                        f"<span style='color:{s_color};font-weight:700'>{si}. "
                        f"<span class='lb-hover-name'>"
                        f"<a href='{s_profile}' target='_self' style='color:inherit;text-decoration:none;' "
                        f"onmouseover=\"this.style.textDecoration='underline'\" "
                        f"onmouseout=\"this.style.textDecoration='none'\">{s['player']}</a>"
                        f"{_hover_card}</span>"
                        f"{_country_flag(s.get('nationality',''))}"
                        f"<span style='color:#666;font-size:0.8rem;font-weight:400;margin-left:0.4rem;'>({s.get('region','').upper()} {s.get('cr',''):,})</span>"
                        f"<a href='{s_twitch}' target='_blank' title='{s_title}' style='margin-left:5px;'>"
                        f"<svg width='11' height='11' viewBox='0 0 24 24' fill='#9146FF' style='vertical-align:middle;'><path d='M11.571 4.714h1.715v5.143H11.57zm4.715 0H18v5.143h-1.714zM6 0L1.714 4.286v15.428h5.143V24l4.286-4.286h3.428L22.286 12V0zm14.571 11.143l-3.428 3.428h-3.429l-3 3v-3H6.857V1.714h13.714z'/></svg>"
                        f"</a>"
                        f"</span>"
                        f"<span style='color:{s_color};font-weight:700'><span style='color:#eb0400;font-size:0.6rem;vertical-align:middle;margin-right:4px;'>&#9679;</span>{s['viewers']:,}</span>"
                        f"</div>"
                    )

                for _si, _s in enumerate(_live_streams_lb[:5], 1):
                    _live_col.markdown(_live_row_html(_si, _s), unsafe_allow_html=True)
                for _ in range(max(0, 5 - len(_live_streams_lb))):
                    _live_col.markdown(_empty_row, unsafe_allow_html=True)

                if len(_live_streams_lb) > 5:
                    _live_exp_key = "lb_expanded_live"
                    if _live_exp_key not in st.session_state:
                        st.session_state[_live_exp_key] = False
                    if st.session_state[_live_exp_key]:
                        for _si, _s in enumerate(_live_streams_lb[5:], 6):
                            _live_col.markdown(_live_row_html(_si, _s), unsafe_allow_html=True)
                    _live_toggle = "▲ Show less" if st.session_state[_live_exp_key] else "▼ Show more"
                    _live_col.markdown("<div class='lb-show-more'>", unsafe_allow_html=True)
                    _live_btn_cols = _live_col.columns([3, 2])
                    if _live_btn_cols[0].button(_live_toggle, key="lb_toggle_live"):
                        st.session_state[_live_exp_key] = not st.session_state[_live_exp_key]
                        st.rerun(scope="fragment")
                    _sort_label = "MMR" if _live_sort_key == "mmr" else "Views"
                    if _live_btn_cols[1].button(f"Sort: {_sort_label}", key="lb_live_sort_btn"):
                        st.session_state["lb_live_sort"] = "viewers" if _live_sort_key == "mmr" else "mmr"
                        st.rerun(scope="fragment")
                    _live_col.markdown("</div>", unsafe_allow_html=True)
                else:
                    _live_col.markdown("<div class='lb-show-more'>", unsafe_allow_html=True)
                    _live_btn_cols = _live_col.columns([3, 2])
                    _live_btn_cols[0].button("▼ Show more", key="lb_toggle_live_placeholder", disabled=True)
                    _sort_label = "MMR" if _live_sort_key == "mmr" else "Views"
                    if _live_btn_cols[1].button(f"Sort: {_sort_label}", key="lb_live_sort_btn"):
                        st.session_state["lb_live_sort"] = "viewers" if _live_sort_key == "mmr" else "mmr"
                        st.rerun(scope="fragment")
                    _live_col.markdown("</div>", unsafe_allow_html=True)

                # ── Återstående listor i par (vänster/höger per rad) ───────────────
                for _i in range(0, len(lists) - 1, 2):
                    _pair = st.columns(2)
                    _li, _ri = _i + 1, _i + 2
                    _t, _it, _fm, _tp, *_rst = lists[_li]
                    render_list(_pair[0], _t, _it, _fm, _tp, asterisk_tip=_rst[0] if _rst else None)
                    if _ri < len(lists):
                        _t, _it, _fm, _tp, *_rst = lists[_ri]
                        render_list(_pair[1], _t, _it, _fm, _tp, asterisk_tip=_rst[0] if _rst else None)

                # ── Sista raden: YouTube (vänster) + First to Xk (höger) ──────────
                _final_row = st.columns(2)

                # ── YouTube leaderboard ────────────────────────────────────────────
                _yt_subs = _yt_fetch_subscribers()
                _yt_svg = "<svg width='11' height='11' viewBox='0 0 24 24' fill='#FF0000' style='vertical-align:middle;margin-right:5px;'><path d='M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z'/></svg>"
                _next_col = _final_row[0]
                _next_col.markdown(
                    f"<div style='color:#8a8a8a;font-size:0.85rem;text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin:0.25rem 0 0.45rem;font-weight:600;'>"
                    f"{_yt_svg}Most subscribers</div>",
                    unsafe_allow_html=True,
                )

                def _yt_row_html(yi, y):
                    y_color  = "#d4a843" if yi == 1 else "#bfc4c8" if yi == 2 else "#b57a4a" if yi == 3 else "#8a8a8a"
                    y_url    = html.escape(y["youtube_url"])
                    subs     = y["subscribers"]
                    subs_fmt = f"{subs/1_000_000:.1f}M" if subs >= 1_000_000 else f"{subs/1_000:.1f}k" if subs >= 1_000 else str(subs)
                    return (
                        f"<div style='display:flex;justify-content:space-between;"
                        f"border:1px solid #1e1e1e;background:#121212;border-radius:4px;"
                        f"padding:0.35rem 0.5rem;margin-bottom:0.25rem;'>"
                        f"<span style='color:{y_color};font-weight:700'>{yi}. "
                        f"<a href='{y_url}' target='_blank' style='color:inherit;text-decoration:none;' "
                        f"onmouseover=\"this.style.textDecoration='underline'\" "
                        f"onmouseout=\"this.style.textDecoration='none'\">{y['player']}</a>"
                        f"{_country_flag(y.get('nationality',''))}"
                        f"<a href='{y_url}' target='_blank' style='margin-left:5px;'>"
                        f"<svg width='11' height='11' viewBox='0 0 24 24' fill='#FF0000' style='vertical-align:middle;'><path d='M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z'/></svg>"
                        f"</a>"
                        f"</span>"
                        f"<span style='color:{y_color};font-weight:700'>{subs_fmt}<span style='color:#555;font-size:0.78em;margin-left:3px;'>subs</span></span>"
                        f"</div>"
                    )

                _yt_empty_row = "<div style='display:flex;border:1px solid #1e1e1e;background:#121212;border-radius:4px;padding:0.35rem 0.5rem;margin-bottom:0.25rem;'><span style='color:#1e1e1e;'>—</span></div>"
                for _yi, _y in enumerate(_yt_subs[:5], 1):
                    _next_col.markdown(_yt_row_html(_yi, _y), unsafe_allow_html=True)
                for _ in range(max(0, 5 - len(_yt_subs))):
                    _next_col.markdown(_yt_empty_row, unsafe_allow_html=True)

                if len(_yt_subs) > 5:
                    _yt_exp_key = "lb_expanded_yt"
                    if _yt_exp_key not in st.session_state:
                        st.session_state[_yt_exp_key] = False
                    if st.session_state[_yt_exp_key]:
                        for _yi, _y in enumerate(_yt_subs[5:], 6):
                            _next_col.markdown(_yt_row_html(_yi, _y), unsafe_allow_html=True)
                    _yt_toggle = "▲ Show less" if st.session_state[_yt_exp_key] else "▼ Show more"
                    _next_col.markdown("<div class='lb-show-more'>", unsafe_allow_html=True)
                    if _next_col.button(_yt_toggle, key="lb_toggle_yt"):
                        st.session_state[_yt_exp_key] = not st.session_state[_yt_exp_key]
                        st.rerun(scope="fragment")
                    _next_col.markdown("</div>", unsafe_allow_html=True)
                else:
                    _next_col.markdown("<div class='lb-show-more'>", unsafe_allow_html=True)
                    _next_col.button("▼ Show more", key="lb_toggle_yt_placeholder", disabled=True)
                    _next_col.markdown("</div>", unsafe_allow_html=True)

                # ── First to Xk (dynamic milestone card) ──────────────────────────
                st.markdown(
                    "<style>label[for='lb_milestone']{"
                    "color:#8a8a8a !important;font-size:0.85rem !important;"
                    "text-transform:uppercase !important;letter-spacing:0.08em !important;"
                    "font-weight:600 !important;}</style>",
                    unsafe_allow_html=True,
                )
                _milestone_k = _final_row[1].selectbox(
                    "First to",
                    [f"{i}k" for i in range(10, 22)],
                    key="lb_milestone",
                )
                _milestone_thresh = str(int(_milestone_k[:-1]) * 1000)
                _milestone_rows = []
                for _r in _sb_fetch_all():
                    if _r.get("region") not in _lb_regions:
                        continue
                    if _top_mmr_players is not None and _r.get("player") not in _top_mmr_players:
                        continue
                    _ms_raw = _r.get("mmr_milestones")
                    if not _ms_raw:
                        continue
                    try:
                        _ms = json.loads(_ms_raw)
                        _date = _ms.get(_milestone_thresh)
                        if _date:
                            _milestone_rows.append({**_r, "_mdate": _date})
                    except Exception:
                        continue
                _milestone_rows.sort(key=lambda r: r["_mdate"])
                _milestone_rows = _milestone_rows[:TOP_N]
                render_list(
                    _final_row[1],
                    f"First to {_milestone_k}",
                    _milestone_rows,
                    lambda r: datetime.fromisoformat(r["_mdate"].replace("Z", "+00:00")).strftime("%b %d"),
                    tooltip=f"First players to reach {_milestone_k} MMR this season.",
                )


            _leaderboard_section()

            if st.button("Refresh leaderboards", width='stretch'):
                _cache_bust_toplists()
                st.rerun(scope="app")


            with st.expander("Secret stuff"):
                pwd = st.text_input("Password", type="password", key="admin_pwd")
                if pwd == st.secrets.get("ADMIN_PASSWORD", ""):
                    st.caption("Add or update Twitch/YouTube links for players.")
                    _COUNTRIES = [("","- None -"),("AF","Afghanistan"),("AL","Albania"),("DZ","Algeria"),("AR","Argentina"),("AM","Armenia"),("AU","Australia"),("AT","Austria"),("AZ","Azerbaijan"),("BE","Belgium"),("BR","Brazil"),("BG","Bulgaria"),("BY","Belarus"),("CA","Canada"),("CL","Chile"),("CN","China"),("CO","Colombia"),("HR","Croatia"),("CZ","Czech Republic"),("DK","Denmark"),("EG","Egypt"),("EE","Estonia"),("FI","Finland"),("FR","France"),("GE","Georgia"),("DE","Germany"),("GR","Greece"),("HK","Hong Kong"),("HU","Hungary"),("IN","India"),("ID","Indonesia"),("IE","Ireland"),("IL","Israel"),("IT","Italy"),("JP","Japan"),("KZ","Kazakhstan"),("KR","South Korea"),("LV","Latvia"),("LT","Lithuania"),("LU","Luxembourg"),("MY","Malaysia"),("MX","Mexico"),("NL","Netherlands"),("NZ","New Zealand"),("NO","Norway"),("PH","Philippines"),("PL","Poland"),("PT","Portugal"),("RO","Romania"),("RU","Russia"),("SA","Saudi Arabia"),("RS","Serbia"),("SG","Singapore"),("SK","Slovakia"),("SI","Slovenia"),("ZA","South Africa"),("ES","Spain"),("SE","Sweden"),("CH","Switzerland"),("TW","Taiwan"),("TH","Thailand"),("TR","Turkey"),("UA","Ukraine"),("GB","United Kingdom"),("US","United States"),("UZ","Uzbekistan"),("VN","Vietnam")]
                    _lnk_player  = st.text_input("Player name", key="lnk_player").strip().lower()
                    _lnk_twitch  = st.text_input("Twitch URL (leave blank to clear)", value="https://www.twitch.tv/", key="lnk_twitch").strip()
                    _lnk_youtube = st.text_input("YouTube URL (leave blank to clear)", key="lnk_youtube").strip()
                    _nat_options = [f"{code} - {name}" if code else f"- {name} -" for code, name in _COUNTRIES]
                    _nat_sel     = st.selectbox("Nationality", _nat_options, key="lnk_nat_sel")
                    _lnk_nat     = _nat_sel.split(" - ")[0].strip() if " - " in _nat_sel and not _nat_sel.startswith("- ") else ""
                    if st.button("Save link", key="lnk_save", width='stretch'):
                        if _lnk_player:
                            try:
                                requests.post(
                                    f"{SUPABASE_URL}/rest/v1/player_links?on_conflict=player_name",
                                    headers={**SUPABASE_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"},
                                    json={k: v for k, v in {"player_name": _lnk_player, "twitch_url": (_lnk_twitch if _lnk_twitch not in ("https://www.twitch.tv/", "") else None), "youtube_url": _lnk_youtube or None, "nationality": _lnk_nat or None}.items() if k == "player_name" or v is not None},
                                    timeout=10,
                                ).raise_for_status()
                                _sb_fetch_player_links.clear()
                                _twitch_get_live_streams.clear()
                                st.success(f"Saved links for {_lnk_player}.")
                            except Exception as _le:
                                st.error(str(_le))
                        else:
                            st.warning("Enter a player name.")

                    st.divider()
                    st.caption("Fetches all players in the leaderboard and recalculates their stats.")
                    if st.button("Refresh all players", width='stretch'):
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
                                    games_r, _, _ = fetch_and_calculate(name, region)
                                    compute_and_upsert(name, region, games_r)
                                except Exception:
                                    pass
                            bar.progress(1.0, text="Done!")
                            _cache_bust_toplists()
                            st.success(f"Done! {len(all_players)} players refreshed.")
                            st.rerun(scope="app")

                    st.divider()
                    st.caption("Fetches top N players per region from wallii.gg leaderboard and upserts their stats.")
                    _scan_regions = st.multiselect("Regions to scan", ["EU", "NA", "AP", "CN"], default=["EU", "NA", "AP", "CN"], key="scan_regions")
                    _scan_limit   = st.number_input("Players per region", min_value=10, max_value=500, value=100, step=10, key="scan_limit")
                    if st.button("Scan leaderboard top N", width='stretch'):
                        _scan_ok, _scan_err = 0, 0
                        for _scan_rgn in _scan_regions:
                            try:
                                _scan_names = fetch_top_n_for_scan(_scan_rgn, int(_scan_limit))
                            except Exception as e:
                                st.warning(f"{_scan_rgn}: failed to fetch leaderboard - {e}")
                                continue
                            _bar = st.progress(0, text=f"{_scan_rgn}: starting...")
                            for _si, _sname in enumerate(_scan_names):
                                _bar.progress((_si + 1) / len(_scan_names), text=f"{_scan_rgn} ({_si+1}/{len(_scan_names)}) {_sname}")
                                try:
                                    fetch_and_calculate.clear()
                                    _sgames, _, _ = fetch_and_calculate(_sname, _scan_rgn)
                                    compute_and_upsert(_sname, _scan_rgn, _sgames)
                                    _scan_ok += 1
                                except Exception:
                                    _scan_err += 1
                            _bar.progress(1.0, text=f"{_scan_rgn}: done!")
                        _cache_bust_toplists()
                        st.success(f"Scan complete - {_scan_ok} ok, {_scan_err} errors.")
                        st.rerun(scope="app")

                    st.divider()
                    st.caption("Rebuild avg-placement regression curves from player_stats data (last 7 days, min 100 games).")
                    _cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
                    _fresh_counts = {}
                    for _reg in ["EU", "NA", "AP", "CN"]:
                        try:
                            _rc = requests.get(
                                f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}",
                                headers=SUPABASE_HEADERS,
                                params={"region": f"eq.{_reg}", "updated_at": f"gte.{_cutoff_7d}", "games": "gte.100", "select": "player", "limit": "10000"},
                                timeout=10,
                            )
                            _fresh_counts[_reg] = len(_rc.json())
                        except Exception:
                            _fresh_counts[_reg] = "?"
                    _total_fresh = sum(v for v in _fresh_counts.values() if isinstance(v, int))
                    st.caption("Fresh players (7d): " + "  |  ".join(f"{r}: {n}" for r, n in _fresh_counts.items()) + f"  |  **Total: {_total_fresh}**")
                    if st.button("Rebuild regression curve", width='stretch'):
                        try:
                            _all_rows = []
                            for _reg in ["EU", "NA", "AP", "CN"]:
                                _rd = requests.get(
                                    f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}",
                                    headers=SUPABASE_HEADERS,
                                    params={"region": f"eq.{_reg}", "updated_at": f"gte.{_cutoff_7d}", "games": "gte.100", "select": "cr,avg_place,games", "limit": "10000"},
                                    timeout=15,
                                )
                                _rd.raise_for_status()
                                _all_rows.extend(_rd.json())
                            if len(_all_rows) < 5:
                                st.warning(f"Not enough data ({len(_all_rows)} players total)")
                            else:
                                _df_all = pd.DataFrame(_all_rows).rename(columns={"cr": "current_mmr"})
                                _rbx, _rby, _ = binned_weighted_curve(_df_all, x_col="current_mmr", y_col="avg_place", w_col="games", bin_size=500, mode="wmean", min_games=0)
                                if len(_rbx) < 2:
                                    st.warning("Regression failed (not enough bins)")
                                else:
                                    _poly = np.polyfit(_rbx, _rby, deg=2)
                                    _smooth_x = np.linspace(_rbx.min(), _rbx.max() + 5000, 100)
                                    _smooth_y = np.clip(np.polyval(_poly, _smooth_x), 1.0, 8.0)
                                    _sb_save_regression("ALL", _smooth_x, _smooth_y, len(_all_rows))
                                    st.success(f"Curve updated ({len(_all_rows)} players, {len(_rbx)} bins)")
                        except Exception as _re:
                            st.error(str(_re))

                    st.divider()
                    st.caption("Recalculates Form Rating for all players using existing player_stats data and the current regression curve. No wallii.gg calls.")
                    if st.button("Rebuild Form Ratings", width='stretch'):
                        try:
                            _bx_fr, _by_fr = _sb_load_regression("ALL")
                            if _bx_fr is None or len(_bx_fr) < 2:
                                st.error("No regression curve found. Rebuild regression curve first.")
                            else:
                                _ps_resp = requests.get(
                                    f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}",
                                    headers=SUPABASE_HEADERS,
                                    params={"select": "player,region,avg_place,form_diff", "limit": "1000"},
                                    timeout=15,
                                )
                                _ps_resp.raise_for_status()
                                _ps_rows = [r for r in _ps_resp.json() if r.get("avg_place") is not None and r.get("form_diff") is not None]
                                _ok, _fail = 0, 0
                                for _pr in _ps_rows:
                                    try:
                                        _ra = _pr["avg_place"] + _pr["form_diff"]
                                        _fr = int(round(float(np.interp(_ra, _by_fr[::-1], _bx_fr[::-1]))))
                                        requests.post(
                                            f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}?on_conflict=player,region",
                                            headers={**SUPABASE_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"},
                                            json={"player": _pr["player"], "region": _pr["region"], "form_rating": _fr},
                                            timeout=10,
                                        ).raise_for_status()
                                        _ok += 1
                                    except Exception:
                                        _fail += 1
                                st.success(f"Updated {_ok} players. Failed: {_fail}.")
                                _sb_fetch_all.clear()
                                _sb_top_n.clear()
                        except Exception as _fre:
                            st.error(str(_fre))

                    st.divider()
                    st.caption("Recalculates Matchup Scaling for all players using existing player_cache snapshots. No wallii.gg calls.")
                    if st.button("Rebuild Matchup Scaling", width='stretch'):
                        try:
                            _pc_resp = requests.get(
                                f"{SUPABASE_URL}/rest/v1/player_cache",
                                headers=SUPABASE_HEADERS,
                                params={"select": "player_name,region", "limit": "2000"},
                                timeout=30,
                            )
                            _pc_resp.raise_for_status()
                            _pc_rows = _pc_resp.json()
                            _ok, _skip, _fail = 0, 0, 0
                            _bar = st.progress(0.0)
                            for _pi, _pc in enumerate(_pc_rows):
                                _bar.progress((_pi + 1) / max(len(_pc_rows), 1), text=f"{_pc.get('player_name','?')}...")
                                try:
                                    _snaps, _, _ = _sb_get_cached_snapshots(_pc["player_name"], _pc["region"])
                                    if not _snaps or len(_snaps) < 2:
                                        # fetch directly without cache age check
                                        _sr = requests.get(
                                            f"{SUPABASE_URL}/rest/v1/snapshots",
                                            headers=SUPABASE_HEADERS,
                                            params={"player_name": f"eq.{_pc['player_name'].lower()}", "region": f"eq.{_pc['region'].upper()}", "game_mode": "eq.0", "snapshot_time": f"gte.{SEASON_START}", "order": "snapshot_time.asc", "limit": "2000", "select": "snapshot_time,rating"},
                                            timeout=15,
                                        )
                                        _sr.raise_for_status()
                                        _snaps = _sr.json()
                                    if not _snaps or len(_snaps) < 2:
                                        _skip += 1
                                        continue
                                    _ms_games = _snapshots_to_games(_snaps)
                                    _ms_val = compute_matchup_scaling(_ms_games)
                                    if _ms_val is None:
                                        _skip += 1
                                        continue
                                    requests.patch(
                                        f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}?player=eq.{_pc['player_name'].lower()}&region=eq.{_pc['region'].upper()}",
                                        headers={**SUPABASE_HEADERS, "Content-Type": "application/json", "Prefer": "return=minimal"},
                                        json={"matchup_scaling": _ms_val},
                                        timeout=10,
                                    ).raise_for_status()
                                    _ok += 1
                                except Exception:
                                    _fail += 1
                            _bar.progress(1.0, text="Done!")
                            _cache_bust_toplists()
                            st.success(f"Updated {_ok} players. Skipped: {_skip}. Failed: {_fail}.")
                        except Exception as _mse:
                            st.error(str(_mse))

                elif pwd:
                    st.caption("Wrong password.")

    else:
        # ── Spelarsida ────────────────────────────────────────────────────────
        if st.session_state.get("sp_games") is None:
            with st.spinner("Fetching data..."):
                try:
                    _sp_season = st.session_state.get("sp_season", CURRENT_SEASON)
                    st.session_state["sp_games"], st.session_state["sp_region"], st.session_state["sp_rank"] = fetch_and_calculate(sp_player, sp_region, season=_sp_season)
                    compute_and_upsert(sp_player, st.session_state["sp_region"], st.session_state["sp_games"], season=_sp_season)
                    _save_opp_buckets(sp_player, st.session_state["sp_region"], st.session_state["sp_games"])
                except ValueError as e:
                    msg = str(e)
                    m = re.search(r"appears to be in: ([A-Z,\s]+)", msg)
                    if m:
                        detected = m.group(1).strip().split(", ")[0]
                        if detected in VALID_REGIONS:
                            st.info(f"Not found in {sp_region} - retrying as {detected}...")
                            try:
                                st.session_state["sp_region"] = detected
                                st.session_state["sp_games"], st.session_state["sp_region"], st.session_state["sp_rank"] = fetch_and_calculate(sp_player, detected)
                                sp_region = detected
                                _save_opp_buckets(sp_player, detected, st.session_state["sp_games"])
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
                _curve_source = None
                bx, by = _sb_load_regression("ALL")
                if bx is not None and len(bx) >= 2:
                    _curve_source = "supabase"
                else:
                    # fallback to CSV
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
                            _curve_source = "csv"
                        except Exception:
                            bx, by = None, None

                if bx is not None and len(bx) >= 2:
                    try:
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
                        f"Expected at CR {current_mmr:,}: {expected_avg:.2f} | "
                        f"Δ: {sign}{delta:.2f} (avg - expected) | "
                        f"Source: {_curve_source}"
                    )

                # Header row: [←] player [region] [#rank]
                hL, hR, hInfo = st.columns([0.7, 5.3, 4.0], vertical_alignment="center")

                with hL:
                    st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
                    if st.button("←", key="home_btn_icon", width='stretch'):
                        go_home()
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                with hInfo:
                    st.markdown(
                        "<p style='color:#888;font-size:0.8rem;text-align:right;margin:0;'>"
                        "See the <strong>Info &amp; Explanations</strong> tab for details."
                        "</p>",
                        unsafe_allow_html=True
                    )

                with hR:
                    player_rank_display = st.session_state.get("sp_rank")
                    rank_str = (
                        f" <span style='color:#999;font-size:0.8rem;margin-left:0.5rem;'>#{player_rank_display}</span>"
                        if player_rank_display else ""
                    )
                    _pl_links  = _sb_fetch_player_links().get(sp_player.lower(), {})
                    _pl_flag   = _country_flag(_pl_links.get("nationality", ""))
                    _hdr_icons = ""
                    if _pl_links.get("twitch_url"):
                        _hdr_icons += f"<a href='{html.escape(_pl_links['twitch_url'])}' target='_blank' title='Twitch' style='margin-left:6px;'><svg width='14' height='14' viewBox='0 0 24 24' fill='#9146FF' style='vertical-align:middle;'><path d='M11.571 4.714h1.715v5.143H11.57zm4.715 0H18v5.143h-1.714zM6 0L1.714 4.286v15.428h5.143V24l4.286-4.286h3.428L22.286 12V0zm14.571 11.143l-3.428 3.428h-3.429l-3 3v-3H6.857V1.714h13.714z'/></svg></a>"
                    if _pl_links.get("youtube_url"):
                        _hdr_icons += f"<a href='{html.escape(_pl_links['youtube_url'])}' target='_blank' title='YouTube' style='margin-left:6px;'><svg width='14' height='14' viewBox='0 0 24 24' fill='#FF0000' style='vertical-align:middle;'><path d='M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z'/></svg></a>"
                    st.markdown(
                        "<p style='color:#eee;font-size:1.1rem;margin:1.2rem 0 0.8rem;'>"
                        + sp_player
                        + (" &nbsp;" + _pl_flag if _pl_flag else "")
                        + " <span style='color:#d4a843;font-size:0.8rem;margin-left:0.5rem;'>" + sp_region + "</span>"
                        + rank_str
                        + _hdr_icons
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
                _wallii_url = f"https://www.wallii.gg/stats/{sp_player}?region={sp_region.lower()}&mode=solo&view=all"
                c1.markdown(
                    f"<div style='margin-top:0.45rem;font-size:0.85rem;color:#777;'>"
                    f"Live updates: <a href='{html.escape(_wallii_url)}' target='_blank' style='color:#9ab8d8;text-decoration:none;' "
                    f"onmouseover=\"this.style.textDecoration='underline'\" onmouseout=\"this.style.textDecoration='none'\">wallii.gg</a></div>",
                    unsafe_allow_html=True
                )
                c5.markdown(
                    "<div title='Peak date: " + peak_time_str + "' style='margin-top:0.45rem;color:#777;font-size:0.85rem;'>Peak: "
                    "<span style='color:#aaa;font-weight:600;'>" + f"{peak_mmr:,}" + "</span> "
                    "<span style='color:#666;'>(" + f"{diff_to_cr:+,}" + ")</span></div>",
                    unsafe_allow_html=True
                )
                dd_tip = (
                    f"{dd_peak_game['mmr_after']:,} → {dd_trough_game['mmr_after']:,} "
                    f"({dd_peak_game['time'][:10]} - {dd_trough_game['time'][:10]})"
                )
                c5.markdown(
                    "<div title='" + dd_tip + "' style='margin-top:0.2rem;color:#777;font-size:0.85rem;cursor:help;'>Max MMR drop: "
                    "<span style='color:#c07070;font-weight:600;'>-" + f"{max_dd:,}" + "</span></div>",
                    unsafe_allow_html=True
                )

                _dist_mode = st.radio("Distribution period", ["All time", "Last 30 days", "Last 7 days"], horizontal=True, label_visibility="collapsed", key="dist_mode")
                if _dist_mode in ("Last 7 days", "Last 30 days"):
                    _days = 7 if _dist_mode == "Last 7 days" else 30
                    _cutoff = datetime.now(timezone.utc) - timedelta(days=_days)
                    _chart_games = [g for g in games if datetime.fromisoformat(g["time"].replace("Z", "+00:00")) >= _cutoff]
                else:
                    _chart_games = games
                if _chart_games:
                    st.pyplot(make_chart(_chart_games))
                else:
                    st.caption("No games in the last 7 days.")

                placements  = [round(g["placement"]) for g in games]
                _tilt_diffs = []
                for i, p in enumerate(placements):
                    if p >= 7:
                        before = placements[max(0, i-50):i]
                        after  = placements[i+1:i+4]
                        if len(before) >= 10 and len(after) >= 1:
                            _tilt_diffs.append(sum(after)/len(after) - sum(before)/len(before))

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
                    f"<span style='color:#aaa;font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{longest_roach}</span>"
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
                        "<span style='color:#333;font-size:0.8rem;margin-left:0.8rem;font-weight:400;text-transform:none;letter-spacing:0;'>- requires 60+ games</span>"
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
                        "<span style='color:#333;font-size:0.8rem;margin-left:0.8rem;font-weight:400;text-transform:none;letter-spacing:0;'>- requires 60+ games</span>"
                        f"{form_html}"
                        "</div>",
                        unsafe_allow_html=True
                    )
                elif len(_tilt_diffs) >= 3 and avg > 0:
                    factor          = 1 + (sum(_tilt_diffs) / len(_tilt_diffs) / avg) * 2
                    tilt_factor_val = float(factor)

                    tilt_color = (
                        "#8c3a2a" if factor >= 1.15
                        else "#c47a75" if factor >= 1.06
                        else "#aaa"   if factor >= 1.00
                        else "#7ab87a" if factor >= 0.90
                        else "#4a8c5c"
                    )
                    _mean_diff    = sum(_tilt_diffs) / len(_tilt_diffs)
                    tooltip       = f"Avg placement change in the 3 games after a 7th/8th, compared to the 50-game baseline before it. Mean diff: {_mean_diff:+.2f}. (Lower = better)"
                    trigger_count = sum(1 for p in placements if p >= 7)
                    asterisk      = "*" if trigger_count < 40 else ""
                    asterisk_tip  = f" title='Low sample size: only {trigger_count} games with placement 7-8'" if trigger_count < 40 else ""

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
                    "#b388e8" if u_score_val >= 0.2
                    else "#aaa"  if u_score_val >= -0.1
                    else "#5b8fd4"
                )
                u_tip = "Measures play style based on placement distribution. Aggressive (positive) = U-shaped - more 1st and 7th/8th relative to 2nd-4th and 5th-6th. Defensive (negative) = flatter - more consistent mid-range finishes. See Info/Explanations for details."

                # ── Form Rating (omvänd interpolation: placement → MMR) ───────
                _form_rating_html = ""
                if total >= 60 and bx is not None and len(bx) >= 2:
                    try:
                        _by_rev = by[::-1]
                        _bx_rev = bx[::-1]
                        _form_mmr_int = int(round(float(np.interp(recent_avg, _by_rev, _bx_rev))))
                        _fmmr_tip = f"The MMR that corresponds to a {recent_avg:.2f} avg placement on the regression curve."
                        _d = _form_mmr_int - current_mmr
                        _fmmr_color = (
                            "#4a8c5c" if _d >=  2000 else
                            "#6aab6a" if _d >=   750 else
                            "#7ab87a" if _d >=     0 else
                            "#d4a843" if _d >=  -750 else
                            "#c47a75" if _d >= -2000 else
                            "#8c3a2a"
                        )
                        _fmmr_diff  = _d
                        _fmmr_sign  = "+" if _fmmr_diff >= 0 else ""
                        _form_rating_html = (
                            f"<span style='float:right;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Form Rating"
                            f"<span style='color:{_fmmr_color};font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{_form_mmr_int:,}</span>"
                            f"<span style='color:{_fmmr_color};font-size:0.8rem;margin-left:0.4rem;'>({_fmmr_sign}{_fmmr_diff:,})</span>"
                            f"<span title='{_fmmr_tip}' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                            f"</span>"
                        )
                    except Exception:
                        pass

                st.markdown(
                    f"<div style='margin:0.3rem 0 0.8rem;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>"
                    f"Aggression"
                    f"<span style='color:{u_color};font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{u_score_val:+.2f}</span>"
                    f"<span title='{u_tip}' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                    f"{_form_rating_html}"
                    f"</div>",
                    unsafe_allow_html=True
                )

                first_10k_date = next((g["time"] for g in games if g["mmr_after"] >= 10000), None)
                first_10k_date = next((g["time"] for g in games if g["mmr_after"] >= 10000), None)
                first_10k_date = next((g["time"] for g in games if g["mmr_after"] >= 10000), None)
                if ENABLE_SESSION_TOPLISTS and total >= 50:
                    first_pct = wins / total * 100 if total else 0.0
                    top4_pct  = top4 / total * 100 if total else 0.0
                    lb_upsert_player(
                        sp_region,
                        sp_player,
                        {
                            "season":       CURRENT_SEASON,
                            "games":        int(total),
                            "hot_streak":   int(longest_streak),
                            "roach_streak": int(longest_roach),
                            "first_pct":    float(first_pct),
                            "top4_pct":     float(top4_pct),
                            "tilt_factor":  float(tilt_factor_val) if tilt_factor_val is not None else None,
                            "avg_place":    float(avg),
                            "form_diff":    float(form_diff) if total >= 60 else None,
                            "form_rating":  _form_mmr_int if total >= 60 and _form_rating_html else None,
                            "max_drawdown":   int(max_dd),
                            "dd_detail":      dd_tip,
                            "first_10k_date": first_10k_date,
                            "cr":             int(current_mmr),
                            "u_score":        float(u_score_val),
                            "bot2_count":     int(norm[7] + norm[8]),
                            "mmr_milestones": json.dumps({
                                str(t): next((g["time"] for g in games if g["mmr_after"] >= t), None)
                                for t in range(10000, 22000, 1000)
                                if any(g["mmr_after"] >= t for g in games)
                            }),
                            "matchup_scaling": compute_matchup_scaling(games),
                            "updated_at":   datetime.utcnow().isoformat() + "Z",
                        }
                    )
                elif ENABLE_SESSION_TOPLISTS and total > 0:
                    lb_upsert_player(
                        sp_region,
                        sp_player,
                        {
                            "season":         CURRENT_SEASON,
                            "games":          int(total),
                            "hot_streak":     int(longest_streak),
                            "roach_streak":   int(longest_roach),
                            "max_drawdown":   int(max_dd),
                            "dd_detail":      dd_tip,
                            "first_10k_date": first_10k_date,
                            "cr":             int(current_mmr),
                            "mmr_milestones": json.dumps({
                                str(t): next((g["time"] for g in games if g["mmr_after"] >= t), None)
                                for t in range(10000, 22000, 1000)
                                if any(g["mmr_after"] >= t for g in games)
                            }),
                            "updated_at":     datetime.utcnow().isoformat() + "Z",
                        }
                    )
                elif ENABLE_SESSION_TOPLISTS and first_10k_date:
                    lb_upsert_player(
                        sp_region,
                        sp_player,
                        {
                            "season":         CURRENT_SEASON,
                            "games":          int(total),
                            "first_10k_date": first_10k_date,
                            "cr":             int(current_mmr),
                            "mmr_milestones": json.dumps({
                                str(t): next((g["time"] for g in games if g["mmr_after"] >= t), None)
                                for t in range(10000, 22000, 1000)
                                if any(g["mmr_after"] >= t for g in games)
                            }),
                            "updated_at":     datetime.utcnow().isoformat() + "Z",
                        }
                    )

                if DEBUG:
                    with st.expander("Supabase upsert debug", expanded=False):
                        st.caption(f"TOPLIST_BACKEND={TOPLIST_BACKEND} | SUPABASE_ENABLED={SUPABASE_ENABLED}")
                        if "sb_upsert_status" in st.session_state:
                            code, txt = st.session_state["sb_upsert_status"]
                            st.caption(f"Upsert: {code} | {txt}")

                # ── Head-to-Head ──────────────────────────────────────────────
                st.markdown("<hr style='border-color:#1e1e1e;margin:0.8rem 0;'>", unsafe_allow_html=True)
                st.markdown("<p style='color:#8a8a8a;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;'>Head-to-Head comparison ⚔️</p>", unsafe_allow_html=True)
                with st.form("h2h_form"):
                    _h2h_cols = st.columns([3, 1, 1])
                    _h2h_name = _h2h_cols[0].text_input("H2H player", placeholder="Compare stats with player…", label_visibility="collapsed", key="h2h_name_input")
                    _h2h_region = _h2h_cols[1].selectbox("H2H region", VALID_REGIONS, key="h2h_region_input", label_visibility="collapsed")
                    _h2h_submitted = _h2h_cols[2].form_submit_button("Compare", use_container_width=True)
                if _h2h_submitted and _h2h_name.strip():
                    import time as _time
                    _h2h_last  = st.session_state.get("h2h_last_time", 0)
                    _h2h_count = st.session_state.get("h2h_count", 0)
                    if _h2h_count >= 2 and _time.time() - _h2h_last < 15:
                        st.warning("Please wait a moment before comparing again.")
                    else:
                        st.session_state["h2h_last_time"] = _time.time()
                        st.session_state["h2h_count"]     = _h2h_count + 1
                        with st.spinner(f"Fetching {_h2h_name.strip()}…"):
                            try:
                                _h2h_result = fetch_and_calculate(_h2h_name.strip().lower(), _h2h_region)
                                st.session_state["h2h_games"]  = _h2h_result[0]
                                st.session_state["h2h_label"]  = _h2h_name.strip()
                                st.session_state["h2h_region"] = _h2h_region
                                st.session_state.pop("h2h_error", None)
                            except Exception as _e:
                                st.session_state["h2h_games"] = None
                                st.session_state["h2h_error"] = str(_e)

                if st.session_state.get("h2h_error"):
                    st.error(st.session_state["h2h_error"])
                elif st.session_state.get("h2h_games"):
                    _h2h_games = st.session_state["h2h_games"]
                    _h2h_label = st.session_state.get("h2h_label", "Opponent")
                    _s1 = _compute_player_stats(games)
                    _s2 = _compute_player_stats(_h2h_games)
                    _n1 = sp_player.title()
                    _n2 = _h2h_label.title()
                    if _n1.lower() == _n2.lower():
                        _n1 = f"{_n1} ({sp_region.upper()})"
                        _n2 = f"{_n2} ({st.session_state.get('h2h_region', '?').upper()})"

                    _WIN  = "color: #7ab87a"
                    _LOSE = "color: #c47a75"

                    def _fmt_diff(s1_val, s2_val, higher_is_better=True, fmt=None, template="{name} has {val} more"):
                        if s1_val is None or s2_val is None:
                            return "—"
                        d = s1_val - s2_val
                        if abs(d) < 0.01:
                            return "Similar"
                        better = _n1 if (d > 0) == higher_is_better else _n2
                        val    = fmt(abs(d)) if fmt else f"{abs(d):.2f}"
                        return template.format(name=better, val=val)

                    def _winner(s1_val, s2_val, higher_is_better=True):
                        if s1_val is None or s2_val is None:
                            return None
                        d = s1_val - s2_val
                        if abs(d) < 0.01:
                            return None
                        return _n1 if (d > 0) == higher_is_better else _n2

                    _stat_defs = [
                        ("Games",          str(_s1["total"]),                                                        str(_s2["total"]),                                                        _fmt_diff(_s1["total"],        _s2["total"],        higher_is_better=True,  fmt=lambda x: str(int(x)),   template="{name} has played {val} more games"),                  _winner(_s1["total"],        _s2["total"],        higher_is_better=True)),
                        ("Avg Placement",  f"{_s1['avg']:.2f}",                                                     f"{_s2['avg']:.2f}",                                                     _fmt_diff(round(_s1["avg"],2),round(_s2["avg"],2),higher_is_better=False, fmt=lambda x: f"{x:.2f}",    template="{name} has a {val} lower average placement"),            _winner(_s1["avg"],          _s2["avg"],          higher_is_better=False)),
                        ("1st %",          f"{_s1['first_pct']:.1f}%",                                              f"{_s2['first_pct']:.1f}%",                                              _fmt_diff(_s1["first_pct"],    _s2["first_pct"],    higher_is_better=True,  fmt=lambda x: f"{x:.1f}%",   template="{name} has a {val} higher 1st place rate"),                    _winner(_s1["first_pct"],    _s2["first_pct"],    higher_is_better=True)),
                        ("Top 4 %",        f"{_s1['top4_pct']:.1f}%",                                               f"{_s2['top4_pct']:.1f}%",                                               _fmt_diff(_s1["top4_pct"],     _s2["top4_pct"],     higher_is_better=True,  fmt=lambda x: f"{x:.1f}%",   template="{name} has a {val} higher top 4 rate"),                _winner(_s1["top4_pct"],     _s2["top4_pct"],     higher_is_better=True)),
                        ("Current MMR",    f"{_s1['current_mmr']:,}",                                                f"{_s2['current_mmr']:,}",                                                _fmt_diff(_s1["current_mmr"],  _s2["current_mmr"],  higher_is_better=True,  fmt=lambda x: f"{int(x):,}", template="{name} has {val} higher Current MMR"),                 _winner(_s1["current_mmr"],  _s2["current_mmr"],  higher_is_better=True)),
                        ("Peak MMR",       f"{_s1['peak_mmr']:,}",                                                   f"{_s2['peak_mmr']:,}",                                                   _fmt_diff(_s1["peak_mmr"],     _s2["peak_mmr"],     higher_is_better=True,  fmt=lambda x: f"{int(x):,}", template="{name} has {val} higher Peak MMR"),                    _winner(_s1["peak_mmr"],     _s2["peak_mmr"],     higher_is_better=True)),
                        ("Max MMR Drop",   f"-{_s1['max_drawdown']:,}",                                              f"-{_s2['max_drawdown']:,}",                                              _fmt_diff(_s1["max_drawdown"], _s2["max_drawdown"], higher_is_better=True,  fmt=lambda x: f"{int(x):,}", template="{name} has a {val} larger Max MMR Drop"),                _winner(_s1["max_drawdown"], _s2["max_drawdown"], higher_is_better=True)),
                        ("Hot Streak",     str(_s1["hot_streak"]),                                                   str(_s2["hot_streak"]),                                                   _fmt_diff(_s1["hot_streak"],   _s2["hot_streak"],   higher_is_better=True,  fmt=lambda x: str(int(x)),   template="{name} has a {val} game longer streak of 1st places"),  _winner(_s1["hot_streak"],   _s2["hot_streak"],   higher_is_better=True)),
                        ("Roach Streak",   str(_s1["roach_streak"]),                                                 str(_s2["roach_streak"]),                                                 _fmt_diff(_s1["roach_streak"], _s2["roach_streak"], higher_is_better=True,  fmt=lambda x: str(int(x)),   template="{name} has a {val} game longer top 4 streak"),              _winner(_s1["roach_streak"], _s2["roach_streak"], higher_is_better=True)),
                        ("Form (last 50)", f"{_s1['form_diff']:+.2f}" if _s1["form_diff"] is not None else "—",     f"{_s2['form_diff']:+.2f}" if _s2["form_diff"] is not None else "—",     _fmt_diff(_s1["form_diff"],    _s2["form_diff"],    higher_is_better=False, fmt=lambda x: f"{x:.2f}",    template="{name} has a {val} better current form"),                _winner(_s1["form_diff"],    _s2["form_diff"],    higher_is_better=False)),
                        ("Tilt Factor",    f"{_s1['tilt_factor']:.2f}" if _s1["tilt_factor"] is not None else "—", f"{_s2['tilt_factor']:.2f}" if _s2["tilt_factor"] is not None else "—", _fmt_diff(_s1["tilt_factor"],  _s2["tilt_factor"],  higher_is_better=False, fmt=lambda x: f"{x:.2f}",    template="{name} has a {val} lower tilt factor"),                _winner(_s1["tilt_factor"],  _s2["tilt_factor"],  higher_is_better=False)),
                        ("Aggression",     f"{_s1['u_score']:+.2f}",                                                f"{_s2['u_score']:+.2f}",                                                (f"{_n1} has a {'slightly ' if abs(_s1['u_score'] - _s2['u_score']) < 0.3 else ''}more aggressive style") if _s1["u_score"] > _s2["u_score"] else (f"{_n2} has a {'slightly ' if abs(_s1['u_score'] - _s2['u_score']) < 0.3 else ''}more aggressive style" if _s2["u_score"] > _s1["u_score"] else "Similar style"), _winner(_s1["u_score"], _s2["u_score"], higher_is_better=True)),
                        ("Farmer Factor",  f"{_s1['farmer_factor']:+.2f}" if _s1["farmer_factor"] is not None else "N/A (need 300+ games at 10k+)", f"{_s2['farmer_factor']:+.2f}" if _s2["farmer_factor"] is not None else "N/A (need 300+ games at 10k+)", _fmt_diff(_s1["farmer_factor"], _s2["farmer_factor"], higher_is_better=True, fmt=lambda x: f"{x:.2f}", template="{name} performs relatively better vs weaker opponents"), _winner(_s1["farmer_factor"], _s2["farmer_factor"], higher_is_better=True)),
                    ]
                    _rows    = [{"Stat": s, _n1: v1, _n2: v2, "Comparison": cmp} for s, v1, v2, cmp, _ in _stat_defs]
                    _winners = [w for *_, w in _stat_defs]

                    _df_h2h = pd.DataFrame(_rows).set_index("Stat")

                    def _color_h2h(df):
                        styles = pd.DataFrame("", index=df.index, columns=df.columns)
                        for i, w in enumerate(_winners):
                            if w == _n1:
                                styles.iloc[i, df.columns.get_loc(_n1)] = _WIN
                                styles.iloc[i, df.columns.get_loc(_n2)] = _LOSE
                            elif w == _n2:
                                styles.iloc[i, df.columns.get_loc(_n1)] = _LOSE
                                styles.iloc[i, df.columns.get_loc(_n2)] = _WIN
                        return styles

                    st.dataframe(
                        _df_h2h.style.apply(_color_h2h, axis=None),
                        use_container_width=True,
                        height=495,
                        column_config={
                            _n1:    st.column_config.TextColumn(width="small"),
                            _n2:    st.column_config.TextColumn(width="small"),
                        }
                    )

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
                        {"Game": i + 1, "MMR": g["mmr_after"], "Date": g["time"][:10], "Placement": f"{g['placement']:.1f}".rstrip('0').rstrip('.') + f" ({g['mmr_after'] - g['mmr_before']:+d})"}
                        for i, g in enumerate(filtered)
                    ])
                    peak_idx = df_mmr["MMR"].idxmax()
                    df_peak = df_mmr.loc[[peak_idx]]
                    df_peak = df_peak.assign(PeakLabel=df_peak["MMR"].map(lambda v: f"Peak: {int(v):,}"))

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
                            tooltip=[alt.Tooltip("Game:Q", title="Game"), alt.Tooltip("MMR:Q", title="MMR"), alt.Tooltip("Placement:N", title="Placement"), alt.Tooltip("Date:N", title="Date")],
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
                    peak_text = (
                        alt.Chart(df_peak)
                        .mark_text(color="#d4a843", fontSize=11, fontWeight=600, dx=10, dy=-10, align="left")
                        .encode(
                            x="Game:Q",
                            y="MMR:Q",
                            text="PeakLabel:N",
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
                    chart = line + milestone_dots + peak_dot + peak_text + rule + hover_points
                    st.altair_chart(chart.properties(height=250).configure_view(strokeWidth=0), width='stretch')

                if st.button("Compare with leaderboard neighbors", width='stretch'):
                    st.session_state["nb_result"] = None
                    player_rank = st.session_state.get("sp_rank")

                    if player_rank is None:
                        st.session_state["nb_result"] = {"error": "Ingen rank hittades för den här spelaren - de kanske inte är på leaderboard."}
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
                                    "<p style='color:#666;font-size:0.8rem;'>Loading cached stats for "
                                    + name + " (rank " + str(rank) + ")...</p>",
                                    unsafe_allow_html=True
                                )
                                try:
                                    ng, _rank = _sb_get_cached_games(name, sp_region)
                                    if ng is not None and len(ng) >= MIN_GAMES_NEIGHBOR:
                                        all_pcts.append(norm_to_pct(ng))
                                    else:
                                        failed.append(name)
                                except Exception:
                                    failed.append(name)
                                progress.progress((i + 1) / len(all_names))

                            progress.empty()
                            status.empty()

                            if not all_pcts:
                                st.session_state["nb_result"] = {"error": f"No cached neighbors had {MIN_GAMES_NEIGHBOR}+ games to compare in Supabase."}
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

            # ── Opponent MMR analysis ─────────────────────────────────────────
            st.divider()
            st.markdown("<p style='color:#8a8a8a;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;'>Performance by opponent MMR range</p>", unsafe_allow_html=True)

            _opp_buckets = {}
            for _g in games:
                _gp   = _g.get("placement")
                _gmmr = _g.get("mmr_before")
                _ggain = _g.get("gain")
                if _gp is None or _gmmr is None or _ggain is None:
                    continue
                if _gmmr < 10000:
                    continue
                _avg_opp = _gmmr - 148.1181435 * (100 - ((_gp - 1) * (200 / 7) + _ggain))
                _bucket  = int(_avg_opp // 1000) * 1000
                # expected placement for this player's MMR at this moment
                _exp = 1 + (7 / 200) * (100 - (_gmmr - _avg_opp) / 148.1181435)
                if _bucket not in _opp_buckets:
                    _opp_buckets[_bucket] = {"placements": [], "expected": []}
                _opp_buckets[_bucket]["placements"].append(_gp)
                if _exp is not None:
                    _opp_buckets[_bucket]["expected"].append(_exp)

            _MIN_GAMES_BUCKET = 5
            _games_above_10k = sum(1 for g in games if g.get("mmr_before", 0) >= 10000)
            _valid_buckets = {k: v for k, v in _opp_buckets.items() if len(v["placements"]) >= _MIN_GAMES_BUCKET and k >= 7000} if _games_above_10k >= 300 else {}
            if _valid_buckets:
                _opp_rows = ""
                for _bk in sorted(_valid_buckets.keys()):
                    _bv   = _valid_buckets[_bk]
                    _bavg = sum(_bv["placements"]) / len(_bv["placements"])
                    _bexp = sum(_bv["expected"]) / len(_bv["expected"]) if _bv["expected"] else None
                    _dev  = (_bavg - _bexp) if _bexp is not None else None
                    if _dev is not None:
                        _dev_color = "#81c784" if _dev < -0.15 else "#e57373" if _dev > 0.15 else "#8a8a8a"
                        _dev_str   = f"{_dev:+.2f}"
                        _dev_html  = f"<span style='color:{_dev_color};font-size:0.78rem;min-width:44px;text-align:right;'>{_dev_str}</span>"
                    else:
                        _dev_html = ""
                    _bar_w     = max(4, int((8 - _bavg) / 7 * 100))
                    _bar_color = "#81c784" if _bavg <= 3 else "#fff176" if _bavg <= 4.5 else "#e57373"
                    _uncertain = _bk >= 11000
                    _ast = "<span title='Data in this interval is unreliable - the placement estimation formula tends to systematically overestimate results at this opponent MMR level, as well as an issue of sample size.' style='color:#666;cursor:help;margin-left:2px;'>*</span>" if _uncertain else ""
                    _top_margin = "1rem" if _bk == 11000 else "0"
                    _opp_rows += (
                        f"<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.3rem;margin-top:{_top_margin};'>"
                        f"<span style='color:#666;font-size:0.78rem;min-width:90px;'>{_bk:,}–{_bk+1000:,}{_ast}</span>"
                        f"<div style='flex:1;background:#1e1e1e;border-radius:3px;height:8px;'>"
                        f"<div style='width:{_bar_w}%;background:{_bar_color};border-radius:3px;height:8px;'></div></div>"
                        f"<span style='color:#ccc;font-size:0.82rem;min-width:32px;text-align:right;'>{_bavg:.2f}</span>"
                        f"{_dev_html}"
                        f"<span style='color:#444;font-size:0.72rem;min-width:50px;'>({len(_bv['placements'])} games)</span>"
                        f"</div>"
                    )
                st.markdown(_opp_rows, unsafe_allow_html=True)
                _legend = "Avg placement per 1k MMR interval. Only games played at 10k+ MMR (early season affects the stats a bit much otherwise)."
                st.caption(_legend)

                # ── Matchup scaling score ─────────────────────────────────────
                _slope = compute_matchup_scaling(games)
                if _slope is not None:
                    _ff = -_slope  # flip: positive = good at beating weaker opponents
                    _ff_color = "#81c784" if _ff > 0.2 else "#e57373" if _ff < -0.2 else "#8a8a8a"
                    if _ff > 0.45:
                        _ff_label = "way better vs weaker opponents"
                    elif _ff > 0.2:
                        _ff_label = "performs better vs weaker opponents"
                    elif _ff > 0:
                        _ff_label = "slightly better vs weaker opponents"
                    elif _ff > -0.2:
                        _ff_label = "slightly better vs stronger opponents"
                    elif _ff > -0.45:
                        _ff_label = "performs better vs stronger opponents"
                    else:
                        _ff_label = "way better vs stronger opponents"
                    st.markdown(
                        f"<div style='margin-top:0.6rem;'>"
                        f"<span style='color:#666;font-size:0.78rem;'>Farmer factor: </span>"
                        f"<span style='color:{_ff_color};font-size:0.9rem;font-weight:700;'>{_ff:+.2f}</span>"
                        f"<span style='color:#444;font-size:0.75rem;margin-left:0.4rem;'>({_ff_label})</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("Not enough data to show opponent MMR breakdown.")


# ── RatingAvg tab (CSV) ───────────────────────────────────────────────────────

with tabs[3]:
    st.info("Ignore this (if you want), just backend stuff for debug/testign. Used for estimating expected average placement at a given MMR based on currently uploaded CSV curves (regression between MMR and avgPlace) for some of the values.")

    st.markdown("<p style='color:#8a8a8a;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Supabase regression curve (all regions combined)</p>", unsafe_allow_html=True)
    _sb_bx, _sb_by = _sb_load_regression("ALL")
    if _sb_bx is None or len(_sb_bx) < 2:
        st.caption("No Supabase regression found. Run 'Rebuild regression curve' in admin.")
    else:
        _sb_q_mmr = st.number_input(
            "MMR",
            min_value=float(np.min(_sb_bx)) - 5000.0,
            max_value=float(np.max(_sb_bx)) + 5000.0,
            value=float(np.percentile(_sb_bx, 75)),
            step=100.0,
            label_visibility="collapsed",
            key="sb_reg_mmr",
        )
        _sb_est = float(np.clip(interp_with_extrap(_sb_q_mmr, _sb_bx, _sb_by), 1.0, 8.0))
        st.markdown(
            f"<div style='margin-top:0.35rem;margin-bottom:0.8rem;'>"
            f"<span style='color:#eee;font-size:1.6rem;font-weight:700;'>{_sb_est:.2f}</span>"
            f"<span style='color:#777;font-size:0.9rem;margin-left:0.6rem;'>at {_sb_q_mmr:,.0f} MMR (all regions)</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        _sb_fig, _sb_ax = plt.subplots(figsize=(10, 5))
        _sb_fig.patch.set_facecolor("#0e0e0e")
        _sb_ax.set_facecolor("#0e0e0e")
        _sb_ax.plot(_sb_bx, _sb_by, color="#d4a843", linewidth=3)
        _sb_ax.set_xlabel("Current MMR")
        _sb_ax.set_ylabel("Avg Place")
        style_dark_axes(_sb_ax)
        st.pyplot(_sb_fig)

    st.divider()

with tabs[3]:
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

    def _remove_outliers(xs, ys, z_thresh=2.5):
        mask = (np.abs((xs - xs.mean()) / xs.std()) < z_thresh) & (np.abs((ys - ys.mean()) / ys.std()) < z_thresh)
        return xs[mask], ys[mask], int((~mask).sum())

    st.divider()
    st.markdown("<h3 style='text-decoration:none;'>Farmer Factor vs MMR</h3>", unsafe_allow_html=True)
    _corr_mmr_rows = [r for r in _sb_fetch_all() if r.get("matchup_scaling") is not None and r.get("cr") is not None]
    if len(_corr_mmr_rows) < 5:
        st.caption("Not enough data yet.")
    else:
        _ff_mmr_vals = np.array([-r["matchup_scaling"] for r in _corr_mmr_rows])
        _cr_vals = np.array([r["cr"] for r in _corr_mmr_rows])
        _cr_vals, _ff_mmr_vals, _mmr_removed = _remove_outliers(_cr_vals, _ff_mmr_vals)
        _corr_mmr = float(np.corrcoef(_ff_mmr_vals, _cr_vals)[0, 1])
        _fig_m, _ax_m = plt.subplots(figsize=(8, 5))
        _fig_m.patch.set_facecolor("#0e0e0e")
        _ax_m.set_facecolor("#0e0e0e")
        _ax_m.scatter(_cr_vals, _ff_mmr_vals, color="#9146FF", alpha=0.6, s=30)
        _m_m, _b_m = np.polyfit(_cr_vals, _ff_mmr_vals, 1)
        _xs_line_m = np.linspace(_cr_vals.min(), _cr_vals.max(), 100)
        _ax_m.plot(_xs_line_m, _m_m * _xs_line_m + _b_m, color="#d4a843", linewidth=2)
        _ax_m.set_xlabel("MMR (cr)")
        _ax_m.set_ylabel("Farmer Factor")
        style_dark_axes(_ax_m)
        st.pyplot(_fig_m)
        _removed_note = f", {_mmr_removed} outliers removed" if _mmr_removed else ""
        st.caption(f"Pearson correlation: **{_corr_mmr:.3f}** ({len(_cr_vals)} players{_removed_note})")

    st.divider()
    st.markdown("<h3 style='text-decoration:none;'>Farmer Factor vs Aggression Score</h3>", unsafe_allow_html=True)
    _corr_rows = [r for r in _sb_fetch_all() if r.get("matchup_scaling") is not None and r.get("u_score") is not None]
    if len(_corr_rows) < 5:
        st.caption("Not enough data yet.")
    else:
        _ff_vals = np.array([-r["matchup_scaling"] for r in _corr_rows])
        _us_vals = np.array([r["u_score"] for r in _corr_rows])
        _us_vals, _ff_vals, _us_removed = _remove_outliers(_us_vals, _ff_vals)
        _corr = float(np.corrcoef(_ff_vals, _us_vals)[0, 1])
        _fig_c, _ax_c = plt.subplots(figsize=(8, 5))
        _fig_c.patch.set_facecolor("#0e0e0e")
        _ax_c.set_facecolor("#0e0e0e")
        _ax_c.scatter(_us_vals, _ff_vals, color="#9146FF", alpha=0.6, s=30)
        _m, _b = np.polyfit(_us_vals, _ff_vals, 1)
        _xs_line = np.linspace(_us_vals.min(), _us_vals.max(), 100)
        _ax_c.plot(_xs_line, _m * _xs_line + _b, color="#d4a843", linewidth=2)
        _ax_c.set_xlabel("Aggression Score (u_score)")
        _ax_c.set_ylabel("Farmer Factor")
        style_dark_axes(_ax_c)
        st.pyplot(_fig_c)
        _removed_note = f", {_us_removed} outliers removed" if _us_removed else ""
        st.caption(f"Pearson correlation: **{_corr:.3f}** ({len(_us_vals)} players{_removed_note}")

    st.divider()
    st.markdown("<h3 style='text-decoration:none;'>Aggression Score vs Tilt Factor</h3>", unsafe_allow_html=True)
    _corr_tilt_rows = [r for r in _sb_fetch_all() if r.get("u_score") is not None and r.get("tilt_factor") is not None]
    if len(_corr_tilt_rows) < 5:
        st.caption("Not enough data yet.")
    else:
        _us_tilt_vals = np.array([r["u_score"] for r in _corr_tilt_rows])
        _tf_vals = np.array([r["tilt_factor"] for r in _corr_tilt_rows])
        _us_tilt_vals, _tf_vals, _tilt_removed = _remove_outliers(_us_tilt_vals, _tf_vals)
        _corr_tilt = float(np.corrcoef(_us_tilt_vals, _tf_vals)[0, 1])
        _fig_t, _ax_t = plt.subplots(figsize=(8, 5))
        _fig_t.patch.set_facecolor("#0e0e0e")
        _ax_t.set_facecolor("#0e0e0e")
        _ax_t.scatter(_us_tilt_vals, _tf_vals, color="#9146FF", alpha=0.6, s=30)
        _m_t, _b_t = np.polyfit(_us_tilt_vals, _tf_vals, 1)
        _xs_line_t = np.linspace(_us_tilt_vals.min(), _us_tilt_vals.max(), 100)
        _ax_t.plot(_xs_line_t, _m_t * _xs_line_t + _b_t, color="#d4a843", linewidth=2)
        _ax_t.set_xlabel("Aggression Score (u_score)")
        _ax_t.set_ylabel("Tilt Factor")
        style_dark_axes(_ax_t)
        st.pyplot(_fig_t)
        _removed_note = f", {_tilt_removed} outliers removed" if _tilt_removed else ""
        st.caption(f"Pearson correlation: **{_corr_tilt:.3f}** ({len(_us_tilt_vals)} players{_removed_note})")

with tabs[4]:
    show_card_browser()

with tabs[1]:
    st.markdown("<h2 style='text-decoration:none;'>Placement Calculator</h2>", unsafe_allow_html=True)
    st.markdown("Given your MMR and the MMR change from a game, estimate what average MMR your opponents had for your most likely placements, and vice versa:")
    _calc_cols = st.columns(2)
    _calc_mmr  = _calc_cols[0].number_input("Your MMR", min_value=0, max_value=30000, value=8000, step=50, key="calc_mmr")
    _calc_gain = _calc_cols[1].number_input("...and MMR gain/loss", min_value=-500, max_value=500, value=100, step=1, key="calc_gain")

    _placements_full = [1, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    _dex_avg   = float(_calc_mmr) if _calc_mmr < 8200 else (_calc_mmr - 0.85 * (_calc_mmr - 8200))
    _p_to_avg = {}
    for _p in _placements_full:
        _avg_opp = _calc_mmr - 148.1181435 * (100 - ((_p - 1) * (200 / 7) + _calc_gain))
        _p_to_avg[_p] = _avg_opp

    _ranked = sorted(
        [(_p, _p_to_avg[_p], abs(_dex_avg - _p_to_avg[_p])) for _p in _placements_full],
        key=lambda x: x[2]
    )
    if _ranked:
        # Certainty: diff between 2nd smallest and smallest delta (all 13 steps, matching sheet formula)
        _all_deltas_full = sorted(
            abs(_dex_avg - (_calc_mmr - 148.1181435 * (100 - ((_p - 1) * (200 / 7) + _calc_gain))))
            for _p in _placements_full
        )
        _cert_diff = (_all_deltas_full[1] - _all_deltas_full[0]) if len(_all_deltas_full) >= 2 else 9999
        if _cert_diff <= 350:
            _cert_label, _cert_color = "Very uncertain", "#e57373"
        elif _cert_diff <= 700:
            _cert_label, _cert_color = "Kinda uncertain", "#ffb74d"
        elif _cert_diff <= 1000:
            _cert_label, _cert_color = "Certain", "#fff176"
        else:
            _cert_label, _cert_color = "Very certain", "#81c784"

        _rows_html = ""
        for _i, (_pd, _avg_opp, _) in enumerate(_ranked[:2]):
            _is_best = _i == 0
            _row_bg  = f"background:{_cert_color}22;border-color:{_cert_color}66;" if _is_best else "background:#111;border-color:#1e1e1e;"
            _row_col = _cert_color if _is_best else "#8a8a8a"
            _suffixes = {1: "st", 2: "nd", 3: "rd"}
            if _pd != int(_pd):
                _pd_str = f"{_pd:g}th"
            else:
                _pd_str = f"{int(_pd)}{_suffixes.get(int(_pd), 'th')}"
            _prefix  = "Most likely" if _is_best else "Also possible"
            _cert_note = f" <span style='font-size:0.78em;font-weight:400;color:{_cert_color};'>({_cert_label.lower()})</span>" if _is_best else ""
            _rows_html += (
                f"<div style='{_row_bg}border:1px solid;border-radius:4px;"
                f"padding:0.5rem 0.7rem;margin-bottom:0.3rem;'>"
                f"<div style='color:{_row_col};font-weight:{'700' if _is_best else '400'};'>"
                f"{_prefix}: <strong>{_pd_str} place</strong> "
                f"<span style='color:{_row_col};'>against an avg opponent MMR of "
                f"<strong>{_avg_opp:,.0f}</strong></span>{_cert_note}</div>"
                f"</div>"
            )
        st.markdown(_rows_html, unsafe_allow_html=True)
    else:
        st.caption("No valid placement found for these values.")

with tabs[2]:
    st.markdown("<h2 style='text-decoration:none;'>About</h2>", unsafe_allow_html=True)
    st.markdown(
        "This app fetches data from [wallii.gg](https://wallii.gg) (timestamped MMR changes). "
        "Placements and all other metrics are derived from these values alone. Most metrics (that might be unclear) are explained below. <br><br>"
        "The placements are not publicly available through an API, so they are estimated using a formula. <br>"
        "Players are added to the leaderboards automatically once searched/fetched (and are eligible).<br>"
        "If you notice any issues or wrong calculations, please report them in the form at the bottom. <br><br>"
        "Created by Darnell/Brugdar.",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("<h2 style='text-decoration:none;'>Explanations</h2>", unsafe_allow_html=True)
    _info_metrics = {
        "Top 1 %": "Percentage of games finished in <strong>1st place</strong>.",
        "Top 4 %": 'Percentage of games finished in <strong>top 4</strong>.',
        "Hot Streak": "The longest consecutive streak of <strong>1st-places</strong>.",
        "Roach Streak": "The longest consecutive streak of <strong>top-4</strong> finishes (i.e. avoiding 5th-8th). This includes 1st places as well.",
        "Tilt Factor": (
            "This measures how a player performs <em>after</em> a bad game (7th or 8th place) compared to their baseline.<br><br>"
            "For each 7th/8th placement, the 50 games before it form a local baseline average, "
            "and the 3 games immediately after are the \"reaction window\". "
            "Tilt Factor = <code>1 + (mean_diff / baseline_avg) × 2</code>.<br><br>"
            "<strong>&lt; 1.0</strong> → plays <em>better</em> after a bad game on average (bounces back)<br>"
            "<strong>= 1.0</strong> → no change<br>"
            "<strong>&gt; 1.0</strong> → plays <em>worse</em> after a bad game (tilts)<br><br>"
            "Requires at least <strong>30 games with a 7th or 8th placement</strong> to appear in leaderboards to avoid outliers. Therefore a lot of good players might not make it on the lists (because of low bottom 2 placements)."
        ),
        "Form": (
            "Difference between a player's <strong>last 50 games</strong> average placement and their overall average. "
            "A negative number means they are playing <em>better</em> recently than their historical baseline.<br><br>"
            "The value is therefore a difference in average placement (for example 3.50 (last 50 games) - 3.20 (overall) = 0.30).<br><br> Negative: better form. Positive: worse form." 
        ),
        "Form Rating": (
            "An estimated MMR value that corresponds to the player's current <strong>Form (last 50 games)</strong> average placement, "
            "based on the regression curve between MMR and average placement.<br><br>"
            "For example, if a player's last 50 games average placement corresponds to what players with 14 000 MMR typically achieve, "
            "the Form Rating will show 14 000 - regardless of the player's current MMR.<br><br>"
            "The difference shown in parentheses is Form Rating minus current MMR. "
            "<strong>Positive = playing above your current rank. Negative = playing below.</strong><br><br>"
            "Think of it as an <em>event horizon</em>: if a player keeps performing at their current form, "
            "their MMR will naturally trend towards the Form Rating over time - "
            "since their results will push the rating up (or down) until it stabilizes at that level."
        ),
        "Largest MMR Drop": "The largest MMR drop from a <strong>peak</strong> to a subsequent <strong>low</strong> in the player's history. This includes any upswing during this time. For every recorded peak it will look for the lowest MMR reached before a new peak is achieved, and the largest of these drops is shown.",
        "Aggression Score": (
            "This measures play style on a spectrum from <strong>aggressive/swingy</strong> to <strong>defensive/consistent</strong> "
            "- basically it describes how U-shaped the placement distribution is. Also known as \"1st-or-8th\".<br><br>"
            "<strong>Part 1:</strong> How often the player finishes 1st vs 2/3/4th.<br>"
            "<strong>Part 2:</strong> How often the player finishes 7/8th vs 5/6th.<br><br>"
            "<code>part1 = ln( place_1 / (place_2 + place_3 + place_4) )</code><br>"
            "<code>part2 = ln( (place_7 + place_8) / (place_5 + place_6) )</code><br>"
            "<code>score&nbsp;= 0.5 × (part1 + part2)</code><br><br>"
            "(Since all players here have a majority of games in top 4, logarithmic ratios are used to make it more of a spectrum. The score is normalized so that 0 means even distribution between 1st/2-4th and 7-8th/5-6th, positive means more 1st and 7-8th, and negative means more 2-4th and 5-6th.)<br><br>"
            "<strong>Positive = aggressive</strong>, <strong>Negative = consistent/defensive</strong>."
        ),
        "Farmer Factor": (
            "This measures how a player's performance changes depending on the strength of their opponents.<br><br>"
            "For each game (only counting games played at 10k+ MMR), the <strong>break-even placement</strong> is calculated - "
            "the placement needed to gain exactly 0 MMR given the player's rating and the lobby's average MMR. "
            "The deviation between actual and break-even placement is then tracked across four opponent MMR intervals (7-8k, 8-9k, 9-10k, 10-11k).<br><br>"
            "A weighted linear regression through these four data points produces the final score:<br><br>"
            "<strong>Positive (high farmer factor)</strong> = performs relatively better against weaker opponents than stronger ones.<br>"
            "<strong>Negative (low farmer factor)</strong> = performs relatively better against stronger opponents than weaker ones.<br><br>"
            "Requires at least <strong>300 games at 10k+ MMR</strong> and <strong>30 games in each of the four opponent brackets</strong>.<br><br>"
        ),
    }
    _selected_metric = st.selectbox("Select a metric", list(_info_metrics.keys()), label_visibility="collapsed")
    st.markdown(
        f"<div style='border:1px solid #333;border-top:none;border-radius:0 0 6px 6px;"
        f"padding:0.8rem 1rem;font-size:0.82rem;line-height:1.7;margin-top:-8px;'>"
        f"{_info_metrics[_selected_metric]}</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("<h2 style='text-decoration:none;'>Report an issue</h2>", unsafe_allow_html=True)
    with st.form("report_form", clear_on_submit=True):
        _report_player  = st.text_input("Player name")
        _report_message = st.text_area("What seems wrong?", height=100)
        _submitted = st.form_submit_button("Send report")
    _now = datetime.utcnow()
    _last_report = st.session_state.get("last_report_time")
    _cooldown_secs = 300  # 5 minutes
    if _submitted:
        if _last_report and (_now - _last_report).total_seconds() < _cooldown_secs:
            _wait = int(_cooldown_secs - (_now - _last_report).total_seconds())
            st.warning(f"Please wait {_wait}s before sending another report.")
        elif _report_player.strip() and _report_message.strip():
            _webhook = st.secrets.get("DISCORD_WEBHOOK", "")
            if _webhook:
                try:
                    requests.post(_webhook, json={"content": f"<@230731312124788736> **Report - {_report_player.strip()}**\n{_report_message.strip()}"}, timeout=5)
                    st.session_state["last_report_time"] = _now
                    st.success("Report sent, thanks!")
                except Exception:
                    st.error("Could not send report, try again later.")
        else:
            st.warning("Please fill in both fields.")

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

# streamlit run wa2_app.py
# streamlit run wa2_app.py --server.runOnSave true (auto-reload on save)