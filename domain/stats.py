"""Statistical and date helpers used by the Wallii app."""

from datetime import datetime, timedelta, timezone

import numpy as np


def utc_now():
    return datetime.now(timezone.utc)


def utc_now_iso_z():
    return utc_now().isoformat().replace("+00:00", "Z")


def parse_iso_utc(ts):
    if isinstance(ts, str) and ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def is_cache_fresh(last_fetched, ttl_hours):
    if not last_fetched:
        return False
    try:
        return utc_now() - parse_iso_utc(last_fetched) < timedelta(hours=ttl_hours)
    except (TypeError, ValueError):
        return False


def season_tracking_start(season_start_str):
    return parse_iso_utc(season_start_str) + timedelta(days=1)


def mmr_milestones_after_tracking_start(games, season_start_str):
    tracking_start = season_tracking_start(season_start_str)
    milestones = {}
    first_10k_date = None
    for game in games:
        if parse_iso_utc(game["time"]) < tracking_start:
            continue
        mmr = game["mmr_after"]
        if first_10k_date is None and mmr >= 10000:
            first_10k_date = game["time"]
        for threshold in range(10000, 22000, 1000):
            if str(threshold) not in milestones and mmr >= threshold:
                milestones[str(threshold)] = game["time"]
    return first_10k_date, milestones


def get_threshold(snapshot_time_str, season_start_str, threshold_base, threshold_increase):
    season_start = parse_iso_utc(season_start_str)
    game_time = parse_iso_utc(snapshot_time_str)
    days_in = max(0, (game_time - season_start).days)
    return threshold_base + (days_in // 20) * threshold_increase


def est_place(mmr, gain, threshold):
    mmr, gain = float(mmr), float(gain)
    placements = [1, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    dex_avg = mmr if mmr < 8200 else (mmr - 0.85 * (mmr - 8200))
    best_placement, best_delta = placements[0], None
    for placement in placements:
        avg_opp = mmr - 148.1181435 * (100 - ((placement - 1) * (200 / 7) + gain))
        if avg_opp > threshold:
            continue
        delta = abs(dex_avg - avg_opp)
        if best_delta is None or delta < best_delta:
            best_delta, best_placement = delta, placement
    return best_placement


def snapshots_to_games(snapshots, season_start_str, threshold_base, threshold_increase):
    games = []
    for previous, current in zip(snapshots, snapshots[1:]):
        gain = current["rating"] - previous["rating"]
        threshold = get_threshold(current["snapshot_time"], season_start_str, threshold_base, threshold_increase)
        games.append({
            "mmr_before": previous["rating"],
            "mmr_after": current["rating"],
            "gain": gain,
            "placement": est_place(previous["rating"], gain, threshold),
            "time": current["snapshot_time"],
        })
    return games


def _opponent_bucket(game):
    placement, mmr, gain = game.get("placement"), game.get("mmr_before"), game.get("gain")
    if placement is None or mmr is None or gain is None:
        return None
    average_opponent = mmr - 148.1181435 * (100 - ((placement - 1) * (200 / 7) + gain))
    return int(average_opponent // 1000) * 1000


def compute_matchup_scaling(games):
    if len(games) < 300:
        return None
    games_10k = [g for g in games if g.get("mmr_before", 0) >= 10000]
    if len(games_10k) < 300:
        return None
    buckets = {}
    for game in games_10k:
        bucket = _opponent_bucket(game)
        if bucket is None:
            continue
        data = buckets.setdefault(bucket, {"placements": [], "expected": []})
        placement, mmr, gain = game["placement"], game["mmr_before"], game["gain"]
        average_opponent = mmr - 148.1181435 * (100 - ((placement - 1) * (200 / 7) + gain))
        data["placements"].append(placement)
        data["expected"].append(1 + (7 / 200) * (100 - (mmr - average_opponent) / 148.1181435))
    points = []
    for bucket in [7000, 8000, 9000, 10000]:
        values = buckets.get(bucket)
        if not values or len(values["placements"]) < 30 or not values["expected"]:
            return None
        points.append((bucket + 500, sum(values["expected"]) / len(values["expected"]) - sum(values["placements"]) / len(values["placements"])))
    xs, ys = np.array([p[0] for p in points]), np.array([p[1] for p in points])
    return float(np.polyfit((xs - xs.mean()) / xs.std(), ys, 1, w=np.array([0.5, 1.0, 1.0, 0.5]))[0])


def compute_opp_buckets(games):
    buckets = {}
    for game in games:
        bucket = _opponent_bucket(game)
        if bucket is not None:
            buckets.setdefault(bucket, []).append(game["placement"])
    return buckets


def compute_core_stats(games, normalized_counts):
    norm = normalized_counts(games)
    total = len(games)
    avg = sum(g["placement"] for g in games) / total
    wins = norm[1]
    top4 = sum(norm[p] for p in [1, 2, 3, 4])
    longest_streak = streak = 0
    longest_roach = roach = 0
    peak_so_far = games[0]["mmr_after"]
    max_drawdown = 0
    peak_game = dd_peak_game = dd_trough_game = games[0]
    for game in games:
        streak = streak + 1 if round(game["placement"]) == 1 else 0
        longest_streak = max(longest_streak, streak)
        roach = roach + 1 if round(game["placement"]) <= 4 else 0
        longest_roach = max(longest_roach, roach)
        if game["mmr_after"] > peak_so_far:
            peak_so_far, peak_game = game["mmr_after"], game
        drawdown = peak_so_far - game["mmr_after"]
        if drawdown > max_drawdown:
            max_drawdown, dd_peak_game, dd_trough_game = drawdown, peak_game, game
    placements = [round(g["placement"]) for g in games]
    tilt_diffs = []
    for i, placement in enumerate(placements):
        if placement >= 7:
            before, after = placements[max(0, i - 50):i], placements[i + 1:i + 4]
            if len(before) >= 10 and after:
                tilt_diffs.append(sum(after) / len(after) - sum(before) / len(before))
    tilt_factor = float(1 + (sum(tilt_diffs) / len(tilt_diffs) / avg) * 2) if len(tilt_diffs) >= 3 and avg > 0 else None
    form_diff = sum(g["placement"] for g in games[-50:]) / 50 - avg if total >= 60 else None
    eps = 0.5
    u_score = 0.5 * (np.log((norm[1] + eps) / (norm[2] + norm[3] + norm[4] + eps)) + np.log((norm[7] + norm[8] + eps) / (norm[5] + norm[6] + eps)))
    dd_detail = f"{dd_peak_game['mmr_after']:,} → {dd_trough_game['mmr_after']:,} ({dd_peak_game['time'][:10]} - {dd_trough_game['time'][:10]})"
    return {
        "total": total, "avg": avg, "current_mmr": games[-1]["mmr_after"],
        "hot_streak": longest_streak, "roach_streak": longest_roach,
        "first_pct": wins / total * 100, "top4_pct": top4 / total * 100,
        "tilt_factor": tilt_factor, "form_diff": form_diff,
        "max_drawdown": max_drawdown, "dd_detail": dd_detail, "u_score": float(u_score),
        "bot2_count": norm[7] + norm[8],
    }


def compute_player_stats(games, normalized_counts):
    if not games:
        return None
    norm = normalized_counts(games)
    total = len(games)
    avg = sum(g["placement"] for g in games) / total
    wins = norm[1]
    top4 = sum(norm[p] for p in [1, 2, 3, 4])
    current_mmr = games[-1]["mmr_after"]
    peak_mmr = max(max(g["mmr_before"] for g in games), max(g["mmr_after"] for g in games))
    max_dd, peak = 0, games[0]["mmr_after"]
    for game in games:
        peak = max(peak, game["mmr_after"])
        max_dd = max(max_dd, peak - game["mmr_after"])
    hot = streak = 0
    for game in games:
        streak = streak + 1 if round(game["placement"]) == 1 else 0
        hot = max(hot, streak)
    # Recompute roach streak separately to preserve the maximum across runs.
    longest_roach = current_roach = 0
    for game in games:
        current_roach = current_roach + 1 if round(game["placement"]) <= 4 else 0
        longest_roach = max(longest_roach, current_roach)
    form_diff = sum(g["placement"] for g in games[-50:]) / 50 - avg if total >= 60 else None
    placements = [round(g["placement"]) for g in games]
    tilt_diffs = []
    for i, placement in enumerate(placements):
        if placement >= 7:
            before, after = placements[max(0, i - 50):i], placements[i + 1:i + 4]
            if len(before) >= 10 and after:
                tilt_diffs.append(sum(after) / len(after) - sum(before) / len(before))
    tilt_factor = float(1 + (sum(tilt_diffs) / len(tilt_diffs) / avg) * 2) if len(tilt_diffs) >= 3 and avg > 0 else None
    eps = 0.5
    u_score = 0.5 * (np.log((norm[1] + eps) / (norm[2] + norm[3] + norm[4] + eps)) + np.log((norm[7] + norm[8] + eps) / (norm[5] + norm[6] + eps)))
    matchup = compute_matchup_scaling(games)
    return {"total": total, "avg": avg, "first_pct": wins / total * 100, "top4_pct": top4 / total * 100,
            "current_mmr": current_mmr, "peak_mmr": peak_mmr, "max_drawdown": max_dd,
            "hot_streak": hot, "roach_streak": longest_roach, "form_diff": form_diff,
            "tilt_factor": tilt_factor, "u_score": u_score, "farmer_factor": -matchup if matchup is not None else None}
