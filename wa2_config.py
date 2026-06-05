APP_VERSION = "1.0.0"

DEBUG = False 

SEASONS = {
    12: {"start": "2025-12-01", "end": "2026-04-13"},
    13: {"start": "2026-04-14", "end": None},
}
CURRENT_SEASON = 13
THRESHOLD_BASE = 9000
THRESHOLD_INCREASE = 1000
VALID_REGIONS = ["NA", "EU", "AP", "CN"]

DEFAULT_CSV_NAME = "export.csv"

CSV_BY_REGION = {
    "EU": "export_eu.csv",
    "NA": "export_na.csv",
    "AP": "export_ap.csv",
    "CN": "export_cn.csv",
}

MIN_GAMES_NEIGHBOR = 300

ENABLE_SESSION_TOPLISTS = True
TOPLIST_BACKEND = "supabase"
PLAYER_STATS_TABLE = "player_stats"
TOP_N = 10
