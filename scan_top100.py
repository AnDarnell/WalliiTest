"""
scan_top100.py
Fetches top 100 players per region from wallii.gg leaderboard,
computes stats, and upserts to Supabase.

Usage:
    python scan_top100.py
    python scan_top100.py --regions EU NA
    python scan_top100.py --limit 50
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import requests

# ── Config ────────────────────────────────────────────────────────────────────

SUPABASE_URL   = "https://nxhoueugwbtkhfswanlf.supabase.co"
SUPABASE_KEY   = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im54aG91ZXVnd2J0a2hmc3dhbmxmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI0NjMxODUsImV4cCI6MjA4ODAzOTE4NX0.C2dyacBcotpIuVso96SIxGNXJX8KRSMgThD-n8yLSC8"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh0aXZhc3VycHp2Y2JvbWlldWJhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQzMTUzODgsImV4cCI6MjA1OTg5MTM4OH0.Opd3c-esvzBd-CWBDSSV7XFB2JCF2LlyevrE2Yr054U"

WALLII_SUPABASE = "https://xtivasurpzvcbomieuba.supabase.co"
PLAYER_STATS_TABLE = "player_stats"

SEASON_START       = "2025-12-01"
THRESHOLD_BASE     = 9000
THRESHOLD_INCREASE = 1000

ALL_REGIONS = ["EU", "NA", "AP", "CN"]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

SUPABASE_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}

WALLII_HEADERS = {
    "apikey":        SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

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

# ── Fetch top 100 from wallii.gg leaderboard ─────────────────────────────────

def fetch_top_n(region, n=100):
    """Fetch top N players for a region from wallii.gg's Supabase."""
    # Get latest day_start
    r = requests.get(
        f"{WALLII_SUPABASE}/rest/v1/daily_leaderboard_stats",
        headers=WALLII_HEADERS,
        params={"select": "day_start", "game_mode": "eq.0", "order": "day_start.desc", "limit": "1"},
        timeout=20,
    )
    r.raise_for_status()
    day_start = r.json()[0]["day_start"]

    # Fetch top N players
    r = requests.get(
        f"{WALLII_SUPABASE}/rest/v1/daily_leaderboard_stats",
        headers=WALLII_HEADERS,
        params={
            "select":    "rank,rating,region,players!inner(player_name)",
            "region":    f"eq.{region}",
            "game_mode": "eq.0",
            "day_start": f"eq.{day_start}",
            "order":     "rank.asc",
            "limit":     str(n),
        },
        timeout=20,
    )
    r.raise_for_status()
    rows = r.json()
    return [(row["players"]["player_name"], region) for row in rows]

# ── Fetch player games from wallii.gg ─────────────────────────────────────────

def fetch_games(player_name, region):
    s = requests.Session()
    s.max_redirects = 5
    url = f"https://www.wallii.gg/stats/{player_name}?region={region.lower()}&mode=solo&view=all"
    r = s.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    match = re.search(r'\\"data\\":\[(\{\\"player_name.*?)\],\\"availableModes\\"', r.text, re.DOTALL)
    if not match:
        raise ValueError("Player not found")

    data_str      = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
    snapshots_all = json.loads("[" + data_str + "]")
    snapshots_all = [s for s in snapshots_all if s["game_mode"] == "0"]
    snapshots     = [s for s in snapshots_all if s["region"].upper() == region.upper()]
    snapshots     = sorted(snapshots, key=lambda x: x["snapshot_time"])

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

# ── Compute stats ─────────────────────────────────────────────────────────────

def compute_stats(games):
    norm  = normalized_counts(games)
    total = len(games)
    avg   = sum(g["placement"] for g in games) / total
    wins  = norm[1]
    top4  = sum(norm[p] for p in [1, 2, 3, 4])

    _eps    = 0.5
    _part1  = np.log((norm[1] + _eps) / (norm[2] + norm[3] + norm[4] + _eps))
    _part2  = np.log((norm[7] + norm[8] + _eps) / (norm[5] + norm[6] + _eps))
    u_score = 0.5 * (_part1 + _part2)

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
            after_bot2.extend(placements[i + 1: i + 4])
            skip_until = i + 4

    tilt_factor = None
    if len(after_bot2) >= 3:
        after_avg   = sum(after_bot2) / len(after_bot2)
        tilt_factor = float(1 + ((after_avg / avg) - 1) * 2) if avg > 0 else None

    form_diff = None
    if total >= 60:
        recent_avg = sum(g["placement"] for g in games[-50:]) / 50
        form_diff  = recent_avg - avg

    max_dd, peak = 0, games[0]["mmr_after"]
    dd_peak_game = dd_trough_game = peak_game = games[0]
    for g in games:
        if g["mmr_after"] > peak:
            peak      = g["mmr_after"]
            peak_game = g
        dd = peak - g["mmr_after"]
        if dd > max_dd:
            max_dd        = dd
            dd_peak_game  = peak_game
            dd_trough_game = g

    dd_detail = (
        f"{dd_peak_game['mmr_after']:,} → {dd_trough_game['mmr_after']:,} "
        f"({dd_peak_game['time'][:10]} – {dd_trough_game['time'][:10]})"
    )

    first_10k = next((g["time"] for g in games if g["mmr_after"] >= 10000), None)

    return {
        "games":          int(total),
        "hot_streak":     int(longest_streak),
        "roach_streak":   int(longest_roach),
        "first_pct":      float(wins / total * 100),
        "top4_pct":       float(top4 / total * 100),
        "tilt_factor":    tilt_factor,
        "avg_place":      float(avg),
        "form_diff":      float(form_diff) if form_diff is not None else None,
        "max_drawdown":   int(max_dd),
        "dd_detail":      dd_detail,
        "first_10k_date": first_10k,
        "cr":             int(games[-1]["mmr_after"]),
        "u_score":        float(u_score),
        "bot2_count":     int(norm[7] + norm[8]),
    }

# ── Upsert to Supabase ────────────────────────────────────────────────────────

def upsert(player_name, region, stats):
    payload = {
        "player":     player_name.lower(),
        "region":     region.upper(),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        **stats,
    }
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}?on_conflict=player,region",
        headers={
            **SUPABASE_HEADERS,
            "Content-Type": "application/json",
            "Prefer":       "resolution=merge-duplicates,return=minimal",
        },
        json=payload,
        timeout=10,
    )
    r.raise_for_status()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scan top N players per region into Supabase.")
    parser.add_argument("--regions", nargs="+", default=ALL_REGIONS, choices=ALL_REGIONS)
    parser.add_argument("--limit",   type=int, default=100, help="Players per region (default 100)")
    parser.add_argument("--delay",   type=float, default=1.5, help="Seconds between requests (default 1.5)")
    args = parser.parse_args()

    total_ok, total_err = 0, 0

    for region in args.regions:
        print(f"\n── {region} ──────────────────────────")
        try:
            players = fetch_top_n(region, args.limit)
        except Exception as e:
            print(f"  Failed to fetch leaderboard: {e}")
            continue

        for i, (name, rgn) in enumerate(players, 1):
            print(f"  [{i}/{len(players)}] {name} ({rgn})", end=" ... ", flush=True)
            try:
                games = fetch_games(name, rgn)
                if len(games) < 50:
                    print(f"skipped (only {len(games)} games)")
                    continue
                stats = compute_stats(games)
                upsert(name, rgn, stats)
                print(f"ok  cr={stats['cr']:,}  1st={stats['first_pct']:.1f}%  u={stats['u_score']:+.2f}")
                total_ok += 1
            except Exception as e:
                print(f"ERROR: {e}")
                total_err += 1

            time.sleep(args.delay)

    print(f"\nDone. {total_ok} ok, {total_err} errors.")


if __name__ == "__main__":
    main()
