"""External Twitch, YouTube, and patch notes integrations."""

import re
from dataclasses import dataclass
from typing import Any, Callable

import requests
import streamlit as st

from wa2_config import CURRENT_SEASON


TWITCH_CLIENT_ID = st.secrets.get("TWITCH_CLIENT_ID", "")
TWITCH_CLIENT_SECRET = st.secrets.get("TWITCH_CLIENT_SECRET", "")
YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")


@dataclass(frozen=True)
class IntegrationServices:
    fetch_player_links: Callable[..., Any]
    stats_rows_by_player: Callable[..., Any]
    best_stats_for_players: Callable[..., Any]
    display_name_for_linked_players: Callable[..., Any]
    regions_for_players_current_season: Callable[..., Any]
    latest_stats_by_player: Callable[..., Any]


@st.cache_data(show_spinner=False, ttl=120, max_entries=5)
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

@st.cache_data(show_spinner=False, ttl=120, max_entries=5)
def _twitch_get_live_streams(_services):
    """Returnerar lista med live-streams för spelare i player_links, sorterade efter viewer-antal."""
    _sb_fetch_player_links = _services.fetch_player_links
    _stats_rows_by_player = _services.stats_rows_by_player
    _best_stats_for_players = _services.best_stats_for_players
    _display_name_for_linked_players = _services.display_name_for_linked_players
    _regions_for_players_current_season = _services.regions_for_players_current_season
    token = _twitch_get_token()
    if not token:
        return []
    links = _sb_fetch_player_links()
    usernames = set()
    players_by_login = {}
    for player, row in links.items():
        url = row.get("twitch_url", "")
        if url:
            login = url.rstrip("/").split("/")[-1].lower().strip()
            usernames.add(login)
            players_by_login.setdefault(login, []).append(player)
    if not usernames:
        return []
    stats_rows_by_player = _stats_rows_by_player()
    stats_by_player = {
        player: _best_stats_for_players([player], stats_rows_by_player)
        for player in stats_rows_by_player
    }
    try:
        streams = []
        username_list = sorted(usernames)
        for i in range(0, len(username_list), 100):
            params = [("user_login", u) for u in username_list[i:i + 100]]
            r = requests.get(
                "https://api.twitch.tv/helix/streams",
                headers={"Client-ID": TWITCH_CLIENT_ID, "Authorization": f"Bearer {token}"},
                params=params,
                timeout=10,
            )
            r.raise_for_status()
            streams.extend(r.json().get("data", []))
        result = []
        for s in streams:
            if "hearthstone" not in s.get("game_name", "").lower():
                continue
            login = s["user_login"].lower()
            linked_players = players_by_login.get(login) or [s["user_name"].lower()]
            player, pstats = _display_name_for_linked_players(linked_players, links, stats_by_player)
            link_row = links.get(player.lower(), {})
            best_season = pstats.get("season") or CURRENT_SEASON
            linked_regions = _regions_for_players_current_season(linked_players, stats_rows_by_player, best_season)
            linked_stats_rows = [
                row
                for p in linked_players
                for row in stats_rows_by_player.get(p.lower(), [])
            ]
            result.append({
                "player":       player,
                "linked_players": linked_players,
                "regions":      linked_regions,
                "stats_rows":   linked_stats_rows,
                "login":        login,
                "title":        s["title"],
                "viewers":      s["viewer_count"],
                "region":       pstats.get("region", ""),
                "season":       pstats.get("season") or 0,
                "cr":           pstats.get("cr") or 0,
                "nationality":  link_row.get("nationality", ""),
                "twitch_url":   link_row.get("twitch_url", f"https://twitch.tv/{login}"),
            })
        result.sort(key=lambda x: (x["season"], x["cr"]), reverse=True)
        return result
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=3600, max_entries=5)
def _yt_fetch_subscribers(_services):
    """Returnerar lista med {player, youtube_url, nationality, subscribers} sorterad efter subscribers."""
    _sb_fetch_player_links = _services.fetch_player_links
    _latest_stats_by_player = _services.latest_stats_by_player
    _display_name_for_linked_players = _services.display_name_for_linked_players
    if not YOUTUBE_API_KEY:
        return []
    links = _sb_fetch_player_links()
    entries_by_url = {}
    for player, row in links.items():
        if row.get("youtube_url"):
            key = row["youtube_url"].rstrip("/").lower()
            entries_by_url.setdefault(key, []).append((player, row))
    stats_by_player = _latest_stats_by_player()
    entries = []
    for grouped_entries in entries_by_url.values():
        players = [player for player, _ in grouped_entries]
        best_player, _ = _display_name_for_linked_players(players, links, stats_by_player)
        row = next((r for player, r in grouped_entries if player == best_player), grouped_entries[0][1])
        entries.append((best_player, row))
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

@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def _fetch_patch_notes(n=5):
    try:
        r = requests.get(
            "https://hearthstone.blizzard.com/en-us/api/blog/articleList",
            params={"page": 1, "pageSize": 30, "category": "patchnotes", "locale": "en_US"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        r.raise_for_status()
        articles = r.json()
        result = []
        for a in articles:
            title = a.get("title", "")
            if re.match(r'^\d{2,}\.\d+', title):
                result.append({
                    "title": title,
                    "url": f"https://hearthstone.blizzard.com/en-us/news/{a['id']}/{a['slug']}"
                })
            if len(result) >= n:
                break
        return result
    except Exception:
        return []