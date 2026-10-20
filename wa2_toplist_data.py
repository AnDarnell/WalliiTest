"""Supabase-backed player statistics and leaderboard data."""

import json
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import numpy as np
import requests
import streamlit as st

from domain import stats as dstats
from wa2_config import CURRENT_SEASON, DEBUG, PLAYER_STATS_TABLE, SEASONS, TOP_N


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
    "apikey": SUPABASE_KEY,
    "Authorization": f"******",
    "Accept-Profile": "public",
    "Content-Profile": "public",
}
SUPABASE_BASE = "https://xtivasurpzvcbomieuba.supabase.co"

SERVER_TOP10_CACHE_TABLE = "server_top10_cache"
SERVER_TOP10_CACHE_KEY = "current_top10_eu_na_ap_v2"
SERVER_TOP10_REGIONS = ("EU", "NA", "AP")
SERVER_TOP10_REFRESH_HOURS = 12
SERVER_TOP10_PAYLOAD_FIELDS = (
    "player",
    "region",
    "current_rank",
    "cr",
    "avg_place",
    "first_pct",
    "top4_pct",
    "u_score",
    "matchup_scaling",
)

_ALL_FIELDS = (
    "player,region,season,games,first_pct,top4_pct,hot_streak,roach_streak,"
    "tilt_factor,avg_place,form_diff,form_rating,max_drawdown,dd_detail,"
    "first_10k_date,cr,peak_mmr,u_score,bot2_count,mmr_milestones,"
    "matchup_scaling,updated_at"
)


def _utc_now():
    return dstats.utc_now()


def _utc_now_iso_z():
    return dstats.utc_now_iso_z()


def _parse_iso_utc(ts):
    return dstats.parse_iso_utc(ts)


def dlog(*args):
    if DEBUG:
        print(*args)


def _lb_init():
    if "toplists" not in st.session_state:
        st.session_state["toplists"] = {"players": {}}

def _lb_key(region, player_name):
    return f"{(region or '').upper()}::{(player_name or '').lower()}"

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

def _cache_bust_toplists():
    try:
        _sb_fetch_all.clear()
        _sb_top_n.clear()
        _lb_top_mmr_players.clear()
        _sb_cached_top_rank_players.clear()
        _lb_metric_rows.clear()
        _lb_stats_by_player.clear()
        _lb_milestone_rows.clear()
        _lb_highest_milestone_threshold.clear()
        _latest_stats_update_label.clear()
    except Exception:
        pass

def _toplist_payload_fingerprint(payload):
    stable_payload = {k: v for k, v in payload.items() if k != "updated_at"}
    return json.dumps(stable_payload, sort_keys=True, default=str)

def _sb_upsert(region, player_name, record: dict):
    season = record.get("season")
    if season is None or season not in SEASONS:
        if DEBUG:
            dlog(
                "Skipping Supabase upsert because missing or invalid season",
                player_name,
                region,
                season,
                record.get("games"),
            )
        return

    payload = {
        "player":       (player_name or "").lower(),
        "region":       (region or "").upper(),
        "season":       int(season),
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
        **({"peak_mmr": record["peak_mmr"]} if record.get("peak_mmr") is not None else {}),
        "u_score":        record.get("u_score"),
        "bot2_count":       record.get("bot2_count"),
        "mmr_milestones":    record.get("mmr_milestones"),
        **({"matchup_scaling": record["matchup_scaling"]} if record.get("matchup_scaling") is not None else {}),
        "updated_at":        _utc_now_iso_z(),
    }

    upsert_key = f"toplist_upsert:{payload['season']}:{payload['region']}:{payload['player']}"
    payload_fingerprint = _toplist_payload_fingerprint(payload)
    if st.session_state.get(upsert_key) == payload_fingerprint:
        return

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
        st.session_state[upsert_key] = payload_fingerprint
        _cache_bust_toplists()

    except Exception as e:
        if DEBUG:
            st.session_state["sb_upsert_status"] = ("ERROR", f"{type(e).__name__}: {e}"[:300])
        dlog("UPSERT ERROR:", e)

@st.cache_data(show_spinner=False, ttl=300, max_entries=10)
def _sb_fetch_all(season=CURRENT_SEASON):
    if not SUPABASE_ENABLED:
        return []
    try:
        url = f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}"
        rows = []
        offset = 0
        page_size = 1000

        while True:
            r = requests.get(
                url,
                headers=SUPABASE_HEADERS,
                params={
                    "select": _ALL_FIELDS,
                    "season": f"eq.{season}",
                    "limit": str(page_size),
                    "offset": str(offset),
                },
                timeout=10,
            )
            if DEBUG:
                st.session_state["sb_topn_status"] = (str(r.status_code), (r.text or "")[:200])
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size

        return rows
    except Exception as e:
        if DEBUG:
            st.session_state["sb_topn_status"] = ("ERROR", f"{type(e).__name__}: {e}"[:200])
        return []

@st.cache_data(show_spinner=False, ttl=300, max_entries=5)
def _sb_current_rank_map():
    if not SUPABASE_ENABLED:
        return {}
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/player_cache",
            headers=SUPABASE_HEADERS,
            params={"select": "player_name,region,current_rank", "limit": "10000"},
            timeout=10,
        )
        r.raise_for_status()
        return {
            (str(row.get("player_name", "")).lower(), str(row.get("region", "")).upper()): int(row["current_rank"])
            for row in r.json()
            if row.get("player_name") and row.get("region") is not None and row.get("current_rank") is not None
        }
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=300, max_entries=300)
def _country_flag(code):
    if not code or len(code) != 2:
        return ""
    c = code.lower()
    return f"<img src='https://flagcdn.com/16x12/{c}.png' alt='{code.upper()}' style='vertical-align:middle;margin-left:4px;border-radius:1px;'>"

@st.cache_data(show_spinner=False, ttl=300, max_entries=5)
def _sb_fetch_player_links():
    """Returns dict {player_name_lower: {twitch_url, youtube_url, nationality, display_name}}."""
    for cols in ("player_name,display_name,twitch_url,youtube_url,nationality", "player_name,twitch_url,youtube_url,nationality", "player_name,twitch_url,youtube_url"):
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

def _split_player_link_names(raw_names):
    """Parse admin input into unique lowercase player names."""
    seen = set()
    names = []
    for part in re.split(r"[,;\n]+", raw_names or ""):
        name = part.strip().lower()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names

def _stats_sort_key(stats):
    return (stats.get("season") or 0, stats.get("cr") or 0)

def _latest_stats_by_player():
    rows_by_player = _stats_rows_by_player()
    return {
        player: _best_stats_for_players([player], rows_by_player)
        for player in rows_by_player
    }

def _stats_rows_by_player():
    all_stats = []
    for season in sorted(SEASONS.keys()):
        all_stats += _sb_fetch_all(season=season)
    rows_by_player = {}
    for row in all_stats:
        player = (row.get("player") or "").lower()
        if player:
            rows_by_player.setdefault(player, []).append(row)
    return rows_by_player

def _best_stats_for_players(players, rows_by_player, regions=None):
    region_set = {r.upper() for r in (regions or [])}
    candidates = []
    for player in players:
        for row in rows_by_player.get((player or "").lower(), []):
            if not region_set or row.get("region", "").upper() in region_set:
                candidates.append(row)
    if not candidates:
        return {}
    return max(candidates, key=_stats_sort_key)

def _regions_for_players_current_season(players, rows_by_player, season):
    return sorted({
        row.get("region", "").upper()
        for player in players
        for row in rows_by_player.get((player or "").lower(), [])
        if row.get("region") and row.get("season") == season
    })

def _best_linked_player(players, stats_by_player):
    if not players:
        return "", {}
    best_player = max(players, key=lambda p: _stats_sort_key(stats_by_player.get(p.lower(), {})))
    return best_player, stats_by_player.get(best_player.lower(), {})

def _display_name_for_linked_players(players, links, stats_by_player):
    stats_player, stats_row = _best_linked_player(players, stats_by_player)
    for player in players:
        display_name = _player_link_display_name(player, links)
        if display_name:
            return display_name, stats_row
    return stats_player, stats_row

def _player_link_display_name(player, links):
    player = (player or "").lower()
    session_display_names = st.session_state.get("player_link_display_names", {})
    return (
        (links.get(player, {}).get("display_name") or "").strip().lower()
        or (session_display_names.get(player) or "").strip().lower()
    )

def _post_player_links(payload):
    url = f"{SUPABASE_URL}/rest/v1/player_links?on_conflict=player_name"
    headers = {**SUPABASE_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"}
    try:
        requests.post(url, headers=headers, json=payload, timeout=10).raise_for_status()
        return True
    except Exception as first_error:
        fallback_payload = [{k: v for k, v in row.items() if k != "display_name"} for row in payload]
        try:
            requests.post(url, headers=headers, json=fallback_payload, timeout=10).raise_for_status()
            return False
        except Exception:
            raise first_error

@st.cache_data(show_spinner=False, ttl=60, max_entries=30)
def _sb_top_n(metric, n=TOP_N, higher_is_better=True, season=CURRENT_SEASON):
    rows = [r for r in _sb_fetch_all(season=season) if r.get(metric) is not None]
    rows.sort(key=lambda r: (r[metric], r.get("cr") or 0), reverse=higher_is_better)
    return rows[:n]

@st.cache_data(show_spinner=False, ttl=300, max_entries=30)
def _lb_top_mmr_players(season, regions_key, n):
    top_keys = set()
    fallback_regions = set(regions_key)

    if season == CURRENT_SEASON and SUPABASE_ENABLED:
        for region in regions_key:
            try:
                rows = _sb_cached_top_rank_players(region, n)
                for row in rows:
                    top_keys.add(_lb_player_key(row.get("region"), row.get("player_name")))
                if rows:
                    fallback_regions.discard(region)
            except Exception as e:
                dlog("Cached top-rank filter lookup failed:", region, e)

    if not fallback_regions:
        return tuple(sorted(top_keys))

    regions = set(fallback_regions)
    rows = [r for r in _sb_fetch_all(season=season) if r.get("region") in regions]
    rows.sort(key=lambda r: r.get("cr") or 0, reverse=True)
    for region in fallback_regions:
        region_rows = [r for r in rows if r.get("region") == region]
        top_keys.update(_lb_player_key(region, r["player"]) for r in region_rows[:n] if r.get("player"))
    return tuple(sorted(top_keys))

def _lb_player_key(region, player):
    return f"{(region or '').upper()}::{(player or '').lower()}"

@st.cache_data(show_spinner=False, ttl=300, max_entries=30)
def _sb_cached_top_rank_players(region, n):
    if not SUPABASE_ENABLED:
        return []
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/player_cache",
        headers=SUPABASE_HEADERS,
        params={
            "select": "player_name,region,current_rank",
            "region": f"eq.{region.upper()}",
            "current_rank": f"lte.{int(n)}",
            "order": "current_rank.asc",
            "limit": str(int(n)),
        },
        timeout=10,
    )
    r.raise_for_status()
    return [row for row in r.json() if row.get("player_name") and row.get("current_rank") is not None]

@st.cache_data(show_spinner=False, ttl=300, max_entries=50)
def _lb_metric_rows(metric, higher_is_better, season, regions_key, top_players_key):
    regions = set(regions_key)
    top_players = set(top_players_key or ())
    rows = []
    for r in _sb_fetch_all(season=season):
        value = r.get(metric)
        if value is None or r.get("region") not in regions:
            continue
        if top_players and _lb_player_key(r.get("region"), r.get("player")) not in top_players:
            continue
        rows.append(r)
    rows.sort(key=lambda r: (r[metric], r.get("cr") or 0), reverse=higher_is_better)
    return rows

@st.cache_data(show_spinner=False, ttl=300, max_entries=10)
def _lb_stats_by_player(season):
    return {r["player"].lower(): r for r in _sb_fetch_all(season=season) if r.get("player")}

@st.cache_data(show_spinner=False, ttl=300, max_entries=50)
def _lb_milestone_rows(season, milestone_thresh, regions_key, top_players_key):
    regions = set(regions_key)
    top_players = set(top_players_key or ())
    rows = []
    for r in _sb_fetch_all(season=season):
        if r.get("region") not in regions:
            continue
        if top_players and _lb_player_key(r.get("region"), r.get("player")) not in top_players:
            continue
        ms_raw = r.get("mmr_milestones")
        if not ms_raw:
            continue
        try:
            milestone_date = json.loads(ms_raw).get(milestone_thresh)
        except Exception:
            continue
        if milestone_date:
            rows.append({**r, "_mdate": milestone_date})
    rows.sort(key=lambda r: r["_mdate"])
    return rows

@st.cache_data(show_spinner=False, ttl=300, max_entries=50)
def _lb_highest_milestone_threshold(season, regions_key, top_players_key):
    regions = set(regions_key)
    top_players = set(top_players_key or ())
    highest = None
    for row in _sb_fetch_all(season=season):
        if row.get("region") not in regions:
            continue
        if top_players and _lb_player_key(row.get("region"), row.get("player")) not in top_players:
            continue
        try:
            milestones = json.loads(row.get("mmr_milestones") or "{}")
        except (TypeError, ValueError):
            continue
        for threshold, reached_at in milestones.items():
            if not reached_at:
                continue
            try:
                threshold = int(threshold)
            except (TypeError, ValueError):
                continue
            highest = threshold if highest is None else max(highest, threshold)
    return highest

@st.cache_data(show_spinner=False, ttl=300, max_entries=10)
def _latest_stats_update_label(season=CURRENT_SEASON):
    ts_values = [r["updated_at"] for r in _sb_fetch_all(season=season) if r.get("updated_at")]
    if not ts_values:
        return None
    latest_ts = max(ts_values)
    try:
        latest_dt = datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
        if latest_dt.tzinfo is None:
            latest_dt = latest_dt.replace(tzinfo=timezone.utc)
        return latest_dt.astimezone().strftime("%d %b %Y, %H:%M")
    except Exception:
        return latest_ts

def _save_opp_buckets(player_name, region, games):
    if not SUPABASE_ENABLED or not games:
        return
    buckets = dstats.compute_opp_buckets(games)
    payload = [
        {
            "player": player_name.lower(),
            "region": region.upper(),
            "bucket_start": bk,
            "avg_placement": round(sum(bv) / len(bv), 4),
            "game_count": len(bv),
            "updated_at": _utc_now_iso_z(),
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

@st.cache_data(show_spinner=False, ttl=300, max_entries=20)
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

@st.cache_data(show_spinner=False, ttl=300, max_entries=10)
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

@st.cache_data(show_spinner=False, ttl=120, max_entries=200)
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

def _server_top10_cache_age_ok(updated_at):
    if not updated_at:
        return False
    try:
        return _utc_now() - _parse_iso_utc(updated_at) < timedelta(hours=SERVER_TOP10_REFRESH_HOURS)
    except Exception:
        return False

def _server_top10_payload_row(row):
    return {field: row.get(field) for field in SERVER_TOP10_PAYLOAD_FIELDS if row.get(field) is not None}

def _sb_load_server_top10_cache_direct(n=10):
    if not SUPABASE_ENABLED:
        return None, None, False
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{SERVER_TOP10_CACHE_TABLE}",
        headers=SUPABASE_HEADERS,
        params={
            "select": "payload_json,updated_at",
            "cache_key": f"eq.{SERVER_TOP10_CACHE_KEY}_{int(n)}",
            "limit": "1",
        },
        timeout=10,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return None, None, True
    return rows[0].get("payload_json") or {}, rows[0].get("updated_at"), True

@st.cache_data(show_spinner=False, ttl=300, max_entries=20)
def _sb_load_server_top10_cache(n=10):
    try:
        return _sb_load_server_top10_cache_direct(n)
    except Exception as e:
        dlog("Server top 10 cache load failed:", e)
        return None, None, False

def _sb_save_server_top10_cache(payload, n=10, clear_cached=True):
    if not SUPABASE_ENABLED:
        return False
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{SERVER_TOP10_CACHE_TABLE}?on_conflict=cache_key",
            headers={**SUPABASE_HEADERS, "Prefer": "resolution=merge-duplicates"},
            json={
                "cache_key": f"{SERVER_TOP10_CACHE_KEY}_{int(n)}",
                "payload_json": payload,
                "updated_at": _utc_now_iso_z(),
            },
            timeout=10,
        )
        r.raise_for_status()
        if clear_cached:
            _sb_load_server_top10_cache.clear()
        return True
    except Exception as e:
        dlog("Server top 10 cache save failed:", e)
        return False

def _sb_server_top10_stats_by_key():
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}",
        headers=SUPABASE_HEADERS,
        params={
            "select": "player,region,avg_place,first_pct,top4_pct,u_score,matchup_scaling",
            "season": f"eq.{CURRENT_SEASON}",
            "limit": "2000",
        },
        timeout=10,
    )
    r.raise_for_status()
    return {
        _lb_player_key(row.get("region"), row.get("player")): row
        for row in r.json()
        if row.get("player") and row.get("region")
    }

def _wallii_refresh_server_top10_bundle(n=10, wallii_headers=None):
    """Refresh all server top 10 rows from Wallii's leaderboard backend in one request."""
    url = f"{SUPABASE_BASE}/rest/v1/daily_leaderboard_stats"
    qs = urlencode([
        ("select",    "rank,rating,region,day_start,players!inner(player_name)"),
        ("region",    f"in.({','.join(SERVER_TOP10_REGIONS)})"),
        ("game_mode", "eq.0"),
        ("rank",      f"lte.{int(n)}"),
        ("order",     "day_start.desc,region.asc,rank.asc"),
        ("limit",     str(int(n) * len(SERVER_TOP10_REGIONS))),
    ])
    r = requests.get(f"{url}?{qs}", headers=wallii_headers or _supabase_headers(), timeout=20)
    r.raise_for_status()
    wallii_rows = r.json()
    if not wallii_rows:
        return {region: [] for region in SERVER_TOP10_REGIONS}

    latest_day = max(row.get("day_start") or "" for row in wallii_rows)
    stats_by_key = _sb_server_top10_stats_by_key()
    bundle = {region: [] for region in SERVER_TOP10_REGIONS}
    for row in wallii_rows:
        if row.get("day_start") != latest_day:
            continue
        p = row.get("players")
        name = (p[0].get("player_name") if isinstance(p, list) and p else p.get("player_name") if isinstance(p, dict) else None)
        if not name:
            continue
        region = str(row.get("region", "")).upper()
        if region not in bundle:
            continue
        merged = dict(stats_by_key.get(_lb_player_key(region, name), {}))
        merged.update({
            "player": name,
            "region": region,
            "current_rank": row.get("rank"),
            "cr": row.get("rating"),
        })
        bundle[region].append(_server_top10_payload_row(merged))
    return bundle

@st.cache_data(ttl=3600, max_entries=20)
def _fetch_server_top10_cached(n=10):
    """Cachad i Streamlits minne i 1h — anropar inte Supabase vid varje knapptryckning."""
    payload, updated_at, cache_available = _sb_load_server_top10_cache(n)
    if payload and _server_top10_cache_age_ok(updated_at):
        return payload, "cache hit"
    if not cache_available:
        return {}, "cache unavailable"
    try:
        fresh_payload = _wallii_refresh_server_top10_bundle(n)
    except Exception:
        if payload:
            return payload, "stale fallback"
        raise
    _sb_save_server_top10_cache(fresh_payload, n)
    return fresh_payload, "refreshed"

def fetch_cached_server_top10_bundle(n=10):
    t0 = time.perf_counter()
    result, status = _fetch_server_top10_cached(n)
    st.session_state["server_top10_status"] = f"{status} ({(time.perf_counter() - t0) * 1000:.0f} ms)"
    return result

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