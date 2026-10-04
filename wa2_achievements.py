"""
Achievements based on earlier Blizzard-leaderboards (top 25 per season).

Data in leaderboard_history.json (created by scripts/fetch_leaderboard_history.py).
"""
import html
import json
from functools import lru_cache
from pathlib import Path

_HISTORY_PATH = Path(__file__).parent / "leaderboard_history.json"
_HISTORY_REGIONS = {"EU": "EU", "NA": "US", "AP": "AP", "CN": "CN"}


@lru_cache(maxsize=1)
def _load_finishes():
    if not _HISTORY_PATH.exists():
        return {}
    by_key = {}
    for r in json.loads(_HISTORY_PATH.read_text(encoding="utf-8")):
        by_key.setdefault((r["region"].upper(), r["player"].lower()), []).append(r)
    return by_key


def _alias_names(player, links):
    """Player's own name + all names that share the same display_name in player_links."""
    player = (player or "").lower()
    names = {player}
    display = ((links or {}).get(player, {}).get("display_name") or "").strip().lower()
    if display:
        names.add(display)
        for name, row in (links or {}).items():
            if (row.get("display_name") or "").strip().lower() == display:
                names.add(name.lower())
    return names


def achievements_for(region, player, links=None):
    finishes = _load_finishes()
    names = _alias_names(player, links)
    by_region = {}
    for display_region, history_region in _HISTORY_REGIONS.items():
        rows = [
            result
            for name in names
            for result in finishes.get((history_region, name), [])
        ]
        by_region[display_region] = {
            "top1": sum(result["rank"] == 1 for result in rows),
            "top10": sum(result["rank"] <= 10 for result in rows),
            "top25": len(rows),
            "top1_seasons": sorted(result["season"] for result in rows if result["rank"] == 1),
        }
    top1_total = sum(stats["top1"] for stats in by_region.values())
    is_record_holder = False
    if top1_total:
        aliases = _alias_names(player, links)
        leaderboard = historical_finish_leaderboard("top1", links)
        leader = leaderboard[0] if leaderboard else None
        leader_names = set(leader.get("linked_players", ())) if leader else set()
        if leader:
            leader_names.add(leader["player"].lower())
        is_record_holder = bool(aliases & leader_names)
    return {
        "top1": top1_total,
        "top10": sum(stats["top10"] for stats in by_region.values()),
        "top25": sum(stats["top25"] for stats in by_region.values()),
        "top1_seasons": [
            (region, season)
            for region, stats in by_region.items()
            for season in stats["top1_seasons"]
        ],
        "by_region": by_region,
        "is_record_holder": is_record_holder,
    }


@lru_cache(maxsize=32)
def _cached_finish_leaderboard(metric, link_signature, selected_regions):
    display_names = dict(link_signature)
    selected_regions = set(selected_regions)
    players = {}
    for (history_region, player), results in _load_finishes().items():
        display_name = display_names.get(player, "").strip()
        group_key = display_name.lower() if display_name else player
        entry = players.setdefault(group_key, {
            "player": display_name or player,
            "count": 0,
            "region_counts": {region: 0 for region in _HISTORY_REGIONS},
            "linked_players": set(),
        })
        entry["linked_players"].add(player)
        display_region = next(
            (region for region, source_region in _HISTORY_REGIONS.items() if source_region == history_region),
            None,
        )
        if display_region is None or display_region not in selected_regions:
            continue
        qualifying = [result for result in results if metric == "top25" or result["rank"] == 1]
        count = len(qualifying)
        entry["count"] += count
        entry["region_counts"][display_region] += count

    rows = []
    for entry in players.values():
        if not entry["count"]:
            continue
        profile_region = max(
            selected_regions,
            key=lambda region: entry["region_counts"][region],
        )
        label_regions = tuple(region for region in _HISTORY_REGIONS if region in selected_regions)
        region_label = "All" if set(label_regions) == set(_HISTORY_REGIONS) else " · ".join(label_regions)
        rows.append({
            "player": entry["player"],
            "count": entry["count"],
            "region": "ALL",
            "profile_region": profile_region,
            "all_regions": True,
            "region_label": region_label,
            "selected_regions": label_regions,
            "metric": metric,
            "region_counts": dict(entry["region_counts"]),
            "linked_players": tuple(sorted(entry["linked_players"])),
        })
    rows.sort(key=lambda row: (-row["count"], row["player"].lower()))
    return tuple(tuple(row.items()) for row in rows)


def historical_finish_leaderboard(metric, links=None, regions=None):
    """Return cached Top 1 or Top 25 totals for the selected historical regions."""
    if metric not in {"top1", "top25"}:
        raise ValueError("metric must be 'top1' or 'top25'")
    requested_regions = {str(region).upper() for region in (regions or _HISTORY_REGIONS)}
    selected_regions = tuple(region for region in _HISTORY_REGIONS if region in requested_regions)
    link_signature = tuple(sorted(
        (name.lower(), (row.get("display_name") or "").strip())
        for name, row in (links or {}).items()
    ))
    cached_rows = _cached_finish_leaderboard(metric, link_signature, selected_regions)
    return [dict(row) for row in cached_rows]


def trophy_html(region, player, links=None):
    """Record-holder crown or a gold trophy, with a regional tooltip."""
    a = achievements_for(region, player, links)
    if not a["top1"]:
        return ""
    icon = "👑" if a["is_record_holder"] else "🏆"
    regional_counts = " · ".join(
        f"{r} {a['by_region'][r]['top1']}" for r in _HISTORY_REGIONS
    )
    tooltip = html.escape(f"Top 1: {a['top1']}x ({regional_counts})", quote=True)
    return f" <span title='{tooltip}' style='cursor:help;font-size:0.9em;'>{icon}</span>"


def _achievement_counts(a):
    items = []
    for key, label, emoji in (
        ("top1", "Top 1", "👑" if a["is_record_holder"] else "🏆"),
        ("top10", "Top 10", "🥈"),
        ("top25", "Top 25", "🥉"),
    ):
        if not a[key]:
            continue
        regional_counts = " · ".join(
            f"{region} {a['by_region'][region][key]}"
            for region in _HISTORY_REGIONS
        )
        tooltip = html.escape(f"{label}: {regional_counts}", quote=True)
        items.append(
            f"<span title='{tooltip}' style='cursor:help;margin-right:0.8rem;'>"
            f"{label}: {emoji} <span style='color:#aaa;'>x{a[key]}</span></span>"
        )
    return "".join(items)


def achievements_html(region, player, links=None):
    """Compact achievement counts for the player profile."""
    a = achievements_for(region, player, links)
    counts = _achievement_counts(a)
    if not counts:
        return ""
    return f"<div style='font-size:0.85rem;line-height:1.5;color:#777;'>{counts}</div>"
