"""Wallii + Supabase snapshot stuff."""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from urllib.parse import urlencode

import requests


@dataclass(frozen=True)
class PlayerDataServices:
    supabase_url: str
    supabase_headers: Mapping[str, str]
    supabase_enabled: bool
    seasons: Mapping[int, Mapping[str, Any]]
    current_season: int
    player_cache_fresh: Callable[..., bool]
    snapshots_to_games: Callable[..., Any]
    latest_day_start: Callable[..., str]
    supabase_base: str
    supabase_headers_for_leaderboard: Callable[..., Mapping[str, str]]
    dlog: Callable[..., Any]


def _sb_fetch_snapshots_range(services, player_name, region, date_from, date_to=None):
    """Fetch snapshots from Supabase for a date range."""
    player_name = player_name.lower()
    rows = []
    offset = 0
    while True:
        params = [
            ("player_name", f"eq.{player_name}"),
            ("region", f"eq.{region.upper()}"),
            ("game_mode", "eq.0"),
            ("snapshot_time", f"gte.{date_from}"),
            ("order", "snapshot_time.asc"),
            ("limit", "1000"),
            ("offset", str(offset)),
            ("select", "snapshot_time,rating"),
        ]
        if date_to:
            params.append(("snapshot_time", f"lte.{date_to}"))
        response = requests.get(
            f"{services.supabase_url}/rest/v1/snapshots",
            headers=services.supabase_headers,
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        batch = response.json()
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return rows if rows else None


def _sb_get_cached_snapshots(services, player_name, region, season=None):
    """Return Supabase snapshots, last fetch time, and cached rank."""
    if season is None:
        season = services.current_season
    season_cfg = services.seasons[season]
    date_from = season_cfg["start"]
    date_to = season_cfg["end"]

    if season != services.current_season:
        try:
            rows = _sb_fetch_snapshots_range(services, player_name, region, date_from, date_to)
            return rows, None, None
        except Exception:
            return None, None, None

    player_name = player_name.lower()
    try:
        response = requests.get(
            f"{services.supabase_url}/rest/v1/player_cache",
            headers=services.supabase_headers,
            params={
                "player_name": f"eq.{player_name}",
                "region": f"eq.{region.upper()}",
                "select": "last_fetched,current_rank",
            },
            timeout=10,
        )
        response.raise_for_status()
        cache_rows = response.json()
        last_fetched = cache_rows[0].get("last_fetched") if cache_rows else None
        cached_rank = cache_rows[0].get("current_rank") if cache_rows else None

        rows = _sb_fetch_snapshots_range(services, player_name, region, date_from, date_to)
        return rows, last_fetched, cached_rank
    except Exception:
        return None, None, None


def _sb_get_cached_games(services, player_name, region, season=None):
    """Return cached games and rank using only Supabase snapshot data."""
    if season is None:
        season = services.current_season
    season_start_str = services.seasons[season]["start"]
    snapshots, _, cached_rank = _sb_get_cached_snapshots(services, player_name, region, season=season)
    if not snapshots:
        return None, cached_rank
    return services.snapshots_to_games(snapshots, season_start_str=season_start_str), cached_rank


def _sb_save_snapshots(services, player_name, region, snapshots, current_rank=None):
    """Save snapshots to Supabase in batches, then update player_cache."""
    player_name = player_name.lower()
    rows = [
        {
            "player_name": player_name,
            "region": region.upper(),
            "snapshot_time": snapshot["snapshot_time"],
            "rating": snapshot["rating"],
            "game_mode": "0",
        }
        for snapshot in snapshots
    ]
    for index in range(0, len(rows), 100):
        requests.post(
            f"{services.supabase_url}/rest/v1/snapshots?on_conflict=player_name,region,snapshot_time",
            headers={
                **services.supabase_headers,
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            json=rows[index:index + 100],
            timeout=30,
        ).raise_for_status()
    requests.post(
        f"{services.supabase_url}/rest/v1/player_cache?on_conflict=player_name,region",
        headers={
            **services.supabase_headers,
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
        json={
            "player_name": player_name,
            "region": region.upper(),
            "last_fetched": datetime.now(timezone.utc).isoformat(),
            "current_rank": current_rank,
        },
        timeout=10,
    ).raise_for_status()


def fetch_and_calculate(services, player_name, region, season=None):
    if season is None:
        season = services.current_season
    season_cfg = services.seasons[season]
    season_start_str = season_cfg["start"]
    season_end_str = season_cfg["end"]

    if season != services.current_season:
        if services.supabase_enabled:
            cached, _, _ = _sb_get_cached_snapshots(services, player_name, region, season=season)
            if cached and len(cached) >= 2:
                return services.snapshots_to_games(cached, season_start_str=season_start_str), region, None
        raise ValueError(f"No cached data found for season {season}.")

    if services.supabase_enabled:
        cached, last_fetched, cached_rank = _sb_get_cached_snapshots(
            services, player_name, region, season=season
        )
        if cached and len(cached) >= 2 and services.player_cache_fresh(last_fetched):
            return services.snapshots_to_games(cached, season_start_str=season_start_str), region, cached_rank

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def _fetch(region_name):
        session = requests.Session()
        session.max_redirects = 5
        url = (
            f"https://www.wallii.gg/stats/{player_name}"
            f"?region={region_name.lower()}&mode=solo&view=all"
        )
        return session.get(url, headers=headers, timeout=20)

    try:
        response = _fetch(region)
    except requests.exceptions.TooManyRedirects:
        raise ValueError("Player not found - check if correct region.")
    response.raise_for_status()

    match = re.search(r'\\"data\\":\[(\{\\"player_name.*?)\],\\"availableModes\\"', response.text, re.DOTALL)
    if not match:
        raise ValueError("Player not found - check spelling and region.")

    data_str = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
    snapshots_all = json.loads("[" + data_str + "]")
    snapshots_all = [snapshot for snapshot in snapshots_all if snapshot["game_mode"] == "0"]
    available_regions = list({snapshot["region"].upper() for snapshot in snapshots_all})
    snapshots = [snapshot for snapshot in snapshots_all if snapshot["region"].upper() == region.upper()]

    current_rank = None
    if services.supabase_enabled:
        try:
            day_start = services.latest_day_start(game_mode=0)
            url = f"{services.supabase_base}/rest/v1/daily_leaderboard_stats"
            query = urlencode([
                ("select", "rank,players!inner(player_name)"),
                ("region", f"eq.{region.upper()}"),
                ("game_mode", "eq.0"),
                ("day_start", f"eq.{day_start}"),
                ("players.player_name", f"ilike.{player_name}"),
                ("limit", "1"),
            ])
            rank_response = requests.get(
                f"{url}?{query}",
                headers=services.supabase_headers_for_leaderboard(),
                timeout=10,
            )
            rank_rows = rank_response.json()
            if rank_rows:
                current_rank = int(rank_rows[0]["rank"])
        except Exception as error:
            services.dlog("DEBUG rank lookup failed:", error)

    if not snapshots and available_regions:
        other = ", ".join(r for r in sorted(available_regions) if r != region.upper())
        raise ValueError(f"No data found for {region}. Player appears to be in: {other}.")

    snapshots = sorted(snapshots, key=lambda snapshot: snapshot["snapshot_time"])

    def _parse_snapshot_time(timestamp):
        if isinstance(timestamp, str) and timestamp.endswith("Z"):
            timestamp = timestamp[:-1] + "+00:00"
        parsed = datetime.fromisoformat(timestamp)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

    season_start = _parse_snapshot_time(season_start_str)
    season_end = _parse_snapshot_time(season_end_str) if season_end_str is not None else None
    snapshots = [
        snapshot
        for snapshot in snapshots
        if season_start
        <= _parse_snapshot_time(snapshot["snapshot_time"])
        <= (season_end if season_end is not None else datetime.max.replace(tzinfo=timezone.utc))
    ]

    if len(snapshots) < 2:
        raise ValueError(f"No snapshots found for season {season}. Please check season selection or player history.")

    if services.supabase_enabled:
        _sb_save_snapshots(services, player_name, region, snapshots, current_rank)

    return services.snapshots_to_games(snapshots, season_start_str=season_start_str), region, current_rank
