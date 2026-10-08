# scripts/fetch_leaderboard_history.py
#
#   python fetch_leaderboard_history.py          
#   python fetch_leaderboard_history.py --cn-only
import argparse, json, os, time, requests

OUT_PATH = "leaderboard_history.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; wallii-history/1.0)"}
ID_OFFSET = 5                       # seasonId = säsong + 5 (gäller även CN)
SEASON_IDS = range(7, 19)           # Id 7..18 = säsong 2..13 (Id 19 pågår)
PAGE_SIZE = 25
TOP_N = 100

GLOBAL_URL = "https://hearthstone.blizzard.com/en-us/api/community/leaderboardsData"
GLOBAL_REGIONS = ["EU", "US", "AP"]
CN_URL = "https://webapi.blizzard.cn/hs-rank-api-server/api/game/ranks"


def fetch_global(region, season_id):
    rows = []
    for page in range(1, TOP_N // PAGE_SIZE + 1):
        r = requests.get(GLOBAL_URL, headers=HEADERS, timeout=15, params={
            "region": region, "leaderboardId": "battlegrounds",
            "page": page, "seasonId": season_id})
        if r.status_code != 200:
            print(f"!! {region} Id={season_id} page={page}: HTTP {r.status_code}")
            return []
        page_rows = r.json().get("leaderboard", {}).get("rows", [])[:PAGE_SIZE]
        rows.extend(page_rows)
        if len(page_rows) < PAGE_SIZE:
            break
        time.sleep(0.25)
    return [
        {"rank": row["rank"], "player": row["accountid"].lower(), "rating": row.get("rating")}
        for row in rows[:TOP_N]
    ]


def fetch_cn(season_id):
    rows = []
    for page in range(1, TOP_N // PAGE_SIZE + 1):
        r = requests.get(CN_URL, headers=HEADERS, timeout=15, params={
            "page": page, "page_size": PAGE_SIZE,
            "mode_name": "battlegrounds", "season_id": season_id})
        if r.status_code != 200:
            print(f"!! CN Id={season_id} page={page}: HTTP {r.status_code}")
            return []
        body = r.json()
        if body.get("code") != 0:
            print(f"!! CN Id={season_id} page={page}: code={body.get('code')} {body.get('message')}")
            return []
        page_rows = (body.get("data") or {}).get("list") or []
        rows.extend(page_rows[:PAGE_SIZE])
        if len(page_rows) < PAGE_SIZE:
            break
        time.sleep(0.25)
    return [
        {"rank": row["position"], "player": row["battle_tag"].lower(), "rating": row.get("score")}
        for row in rows[:TOP_N]
    ]


def collect(region, fetch, seen_signatures=None):
    """Fetches all seasons for a region. fetch(season_id) -> list of rank/player/rating."""
    out = []
    for season_id in SEASON_IDS:
        rows = fetch(season_id)
        if not rows:
            print(f"!! {region} Id={season_id}: no rows, skipping")
            continue
        if len(rows) < TOP_N:
            print(f"!! {region} Id={season_id}: bara {len(rows)} rader")
        # Skydd: om API:t ignorerar season_id och returnerar samma lista flera gånger.
        if seen_signatures is not None:
            sig = tuple((r["player"], r["rating"]) for r in rows[:5])
            if sig in seen_signatures:
                print(f"!! {region} Id={season_id}: identisk med Id={seen_signatures[sig]}, hoppar över")
                continue
            seen_signatures[sig] = season_id
        for row in rows:
            out.append({"region": region, "season": season_id - ID_OFFSET, **row})
        print(f"ok {region} Id={season_id} ({len(rows)} rader)")
        time.sleep(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cn-only", action="store_true",
                    help="fetch only CN and merge with existing leaderboard_history.json")
    args = ap.parse_args()

    if args.cn_only:
        existing = []
        if os.path.exists(OUT_PATH):
            with open(OUT_PATH, encoding="utf-8") as f:
                existing = [r for r in json.load(f) if r["region"] != "CN"]
        out = existing
    else:
        out = []
        for region in GLOBAL_REGIONS:
            out += collect(region, lambda sid, region=region: fetch_global(region, sid))

    out += collect("CN", fetch_cn, seen_signatures={})

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print("Totalt", len(out), "rader")


if __name__ == "__main__":
    main()
