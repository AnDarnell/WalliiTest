# scripts/fetch_leaderboard_history.py  (kör lokalt, en gång)
import json, time, requests

URL = "https://hearthstone.blizzard.com/en-us/api/community/leaderboardsData"
REGIONS = ["EU", "US", "AP"]
ID_OFFSET = 5                       # seasonId = säsong + 5
SEASON_IDS = range(7, 19)           # Id 7..18 = säsong 2..13 (Id 19 pågår)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; wallii-history/1.0)"}

out = []
for region in REGIONS:
    for season_id in SEASON_IDS:
        r = requests.get(URL, headers=HEADERS, timeout=15, params={
            "region": region, "leaderboardId": "battlegrounds",
            "page": 1, "seasonId": season_id})
        if r.status_code != 200:
            print(f"!! {region} Id={season_id}: HTTP {r.status_code}")
            continue
        rows = r.json().get("leaderboard", {}).get("rows", [])[:25]
        if len(rows) < 25:
            print(f"!! {region} Id={season_id}: bara {len(rows)} rader")
        for row in rows:
            out.append({
                "region": region,
                "season": season_id - ID_OFFSET,
                "rank": row["rank"],
                "player": row["accountid"].lower(),
                "rating": row.get("rating"),
            })
        print(f"ok {region} Id={season_id} ({len(rows)} rader)")
        time.sleep(1)

with open("leaderboard_history.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=0)
print("Totalt", len(out), "rader")