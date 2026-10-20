
"""Requires:
  pip install streamlit requests matplotlib numpy pandas"""


import json
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone, timedelta, date
from urllib.parse import urlencode
import html
import time
import threading
from functools import partial

from wa2_config import (
    APP_VERSION, DEBUG, SEASONS, CURRENT_SEASON,
    THRESHOLD_BASE, THRESHOLD_INCREASE, VALID_REGIONS,
    DEFAULT_CSV_NAME, CSV_BY_REGION, MIN_GAMES_NEIGHBOR,
    ENABLE_SESSION_TOPLISTS, TOPLIST_BACKEND, PLAYER_STATS_TABLE, TOP_N,
)
from wa2_charts import (
    style_dark_axes, normalized_counts, norm_to_pct,
    make_chart, make_neighbor_chart, delta_color, diff_pct_color,
    summarize_neighbor_differences,
)
from wa2_rating import (
    interp_with_extrap, weighted_quantile, binned_weighted_curve, load_rating_curve,
)
from wa2_cards import show_card_browser
from wa2_achievements import (
    trophy_html, achievements_html, historical_finish_leaderboard,
    historical_finish_hover_html, all_time_mmr_leaderboard,
)

from domain import stats as dstats
from wa2_icons import twitch_svg, youtube_svg, twitch_link, youtube_link
from wa2_leaderboard import LeaderboardServices, render_leaderboard
from wa2_integrations import (
    IntegrationServices, _fetch_patch_notes, _twitch_get_live_streams,
    _yt_fetch_subscribers,
)
from wa2_profile import ProfileServices, render_player_profile
from wa2_admin_views import AdminServices, render_admin_panel
from wa2_testing_views import TestingViewServices, render_testing_views
from wa2_theme import apply_page_theme
from wa2_toplist_data import (
    SUPABASE_BASE, SUPABASE_ENABLED, SUPABASE_HEADERS, SUPABASE_KEY, SUPABASE_URL,
    SERVER_TOP10_CACHE_KEY, SERVER_TOP10_CACHE_TABLE, SERVER_TOP10_PAYLOAD_FIELDS,
    SERVER_TOP10_REFRESH_HOURS, SERVER_TOP10_REGIONS, _ALL_FIELDS,
    _best_linked_player, _best_stats_for_players, _cache_bust_toplists,
    _country_flag, _display_name_for_linked_players, _fetch_server_top10_cached,
    _latest_day_start, _latest_stats_by_player, _latest_stats_update_label,
    _lb_highest_milestone_threshold, _lb_init, _lb_key, _lb_metric_rows,
    _lb_milestone_rows, _lb_player_key, _lb_stats_by_player, _lb_top_mmr_players,
    _player_link_display_name,
    _post_player_links, _regions_for_players_current_season, _save_opp_buckets,
    _sb_cached_top_rank_players, _sb_current_rank_map, _sb_fetch_all,
    _sb_fetch_player_links, _sb_load_regression, _sb_load_server_top10_cache,
    _sb_load_server_top10_cache_direct, _sb_save_regression,
    _sb_save_server_top10_cache, _sb_server_top10_stats_by_key, _sb_top_n,
    _sb_upsert, _session_top_n, _session_upsert, _split_player_link_names,
    _supabase_headers,
    _stats_rows_by_player, _stats_sort_key, _toplist_payload_fingerprint,
    _wallii_refresh_server_top10_bundle, fetch_cached_server_top10_bundle,
    fetch_neighbor_names, fetch_top_n_for_scan,
)

from wa2_player_data import (
    PlayerDataServices,
    _sb_get_cached_games as _get_cached_player_games,
    _sb_get_cached_snapshots as _get_cached_player_snapshots,
    fetch_and_calculate as _fetch_player_and_calculate,
)


def _utc_now():
    return dstats.utc_now()


def _utc_now_iso_z():
    return dstats.utc_now_iso_z()


def _parse_iso_utc(ts):
    return dstats.parse_iso_utc(ts)


def _player_cache_fresh(last_fetched):
    return dstats.is_cache_fresh(last_fetched, PLAYER_CACHE_TTL_HOURS)


def _season_tracking_start(season):
    season_cfg = SEASONS.get(season, SEASONS[CURRENT_SEASON])
    return dstats.season_tracking_start(season_cfg["start"])


def _mmr_milestones_after_tracking_start(games, season):
    season_cfg = SEASONS.get(season, SEASONS[CURRENT_SEASON])
    return dstats.mmr_milestones_after_tracking_start(games, season_cfg["start"])



PLAYER_CACHE_TTL_HOURS = 2





# Helpers

def dlog(*args):
    if DEBUG:
        print(*args)

def get_threshold(snapshot_time_str, season_start_str=None):
    if season_start_str is None:
        season_start_str = SEASONS[CURRENT_SEASON]["start"]
    return dstats.get_threshold(snapshot_time_str, season_start_str, THRESHOLD_BASE, THRESHOLD_INCREASE)

def est_place(mmr, gain, snapshot_time=None, season_start_str=None):
    threshold = get_threshold(snapshot_time, season_start_str) if snapshot_time else THRESHOLD_BASE
    return dstats.est_place(mmr, gain, threshold)

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
    st.session_state.pop("sp_profile_cache", None)
    st.session_state.pop("sp_previous_season", None)
    st.session_state.pop("sp_season_notice", None)



# ── Query-param navigation (from leaderboard name links) ─────────────────────
_qp = st.query_params
if "goto_player" in _qp:
    st.session_state["sp_player"] = _qp["goto_player"].lower()
    st.session_state["sp_region"] = _qp.get("goto_region", "EU")
    season_param = _qp.get("goto_season", None)
    if isinstance(season_param, list):
        season_param = season_param[0] if season_param else None
    try:
        season_choice = int(season_param) if season_param is not None else CURRENT_SEASON
    except ValueError:
        season_choice = CURRENT_SEASON
    st.session_state["sp_season"] = season_choice if season_choice in SEASONS else CURRENT_SEASON
    st.session_state.pop("sp_games", None)
    st.session_state.pop("sp_rank", None)
    st.session_state.pop("h2h_games", None)
    st.session_state.pop("h2h_label", None)
    st.session_state.pop("h2h_region", None)
    st.session_state.pop("h2h_error", None)
    st.session_state.pop("nb_result", None)
    st.session_state.pop("sp_profile_cache", None)
    st.session_state.pop("sp_previous_season", None)
    st.session_state.pop("sp_season_notice", None)
    st.query_params.clear()
    st.rerun()
if "goto_home" in _qp:
    go_home()
    st.query_params.clear()
    st.rerun()


# ── Toplists — session or Supabase ────────────────────────────────────────────



# ---------- session backend ----------



# ---------- Supabase backend ----------

















































@st.cache_data(show_spinner=False, max_entries=20)
def _read_curve_csv(path_str, mtime_ns):
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False, max_entries=100)
def _cached_binned_curve(path_str, mtime_ns, x_choice, bin_size, mode_kind, q, min_games):
    df = _read_curve_csv(path_str, mtime_ns)
    return binned_weighted_curve(
        df,
        x_col=x_choice,
        y_col="avg_place",
        w_col="games",
        bin_size=int(bin_size),
        mode=mode_kind,
        q=float(q) if mode_kind == "wquant" else 0.5,
        min_games=int(min_games),
    )



# ---------- unified API ----------

def lb_upsert_player(region, player_name, record: dict):
    if TOPLIST_BACKEND == "supabase":
        _sb_upsert(region, player_name, record)
    else:
        _session_upsert(region, player_name, record)

def compute_and_upsert(player_name, region, games, season=CURRENT_SEASON):
    if not games:
        return
    if not any(g["gain"] != 0 for g in games):
            if DEBUG:
                dlog("Skipping upsert - no actual games (all gains are 0)", player_name, region, season)
            return

    if season not in SEASONS:
        season = CURRENT_SEASON
    season_cfg = SEASONS[season]
    season_start = datetime.fromisoformat(season_cfg["start"]).replace(tzinfo=timezone.utc)
    season_end = datetime.fromisoformat(season_cfg["end"]).replace(tzinfo=timezone.utc) if season_cfg["end"] else None

    def _parse_snapshot_time(ts):
        if isinstance(ts, str) and ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    game_times = [_parse_snapshot_time(g["time"]) for g in games]
    if not all(season_start <= t for t in game_times):
        if DEBUG:
            dlog(
                "Skipping upsert because games start before selected season",
                player_name,
                region,
                season,
                game_times[0].isoformat(),
                game_times[-1].isoformat(),
            )
        return
    if season_end is not None and not all(t <= season_end for t in game_times):
        if DEBUG:
            dlog(
                "Skipping upsert because games extend beyond selected season",
                player_name,
                region,
                season,
                game_times[0].isoformat(),
                game_times[-1].isoformat(),
            )
        return

    core = dstats.compute_core_stats(games, normalized_counts)
    total, avg, current_mmr = core["total"], core["avg"], core["current_mmr"]
    _peak_mmr = max(
        max(g["mmr_before"] for g in games),
        max(g["mmr_after"] for g in games),
    )

    form_rating = None
    if core["form_diff"] is not None:
        recent_avg = avg + core["form_diff"]
        try:
            _bx, _by = _sb_load_regression("ALL")
            if _bx is not None and len(_bx) >= 2:
                form_rating = int(round(float(np.interp(recent_avg, _by[::-1], _bx[::-1]))))
        except Exception:
            pass

    first_10k_date, _mmr_milestones = _mmr_milestones_after_tracking_start(games, season)
    _save_opp_buckets(player_name, region, games)

    if total < 50:
        if first_10k_date is not None:
            lb_upsert_player(region, player_name, {
                "season":         season,
                "games":          int(total),
                "first_10k_date": first_10k_date,
                "cr":             int(current_mmr),
                "peak_mmr":       int(_peak_mmr),
                "mmr_milestones": json.dumps(_mmr_milestones),
                "updated_at":     _utc_now_iso_z(),
            })
        return

    lb_upsert_player(region, player_name, {
        "season":          season,
        "games":           int(total),
        "hot_streak":      int(core["hot_streak"]),
        "roach_streak":    int(core["roach_streak"]),
        "first_pct":       float(core["first_pct"]),
        "top4_pct":        float(core["top4_pct"]),
        "tilt_factor":     core["tilt_factor"],
        "avg_place":       float(avg),
        "form_diff":       float(core["form_diff"]) if core["form_diff"] is not None else None,
        "form_rating":     form_rating,
        "max_drawdown":    int(core["max_drawdown"]),
        "dd_detail":       core["dd_detail"],
        "first_10k_date":  first_10k_date,
        "cr":              int(current_mmr),
        "peak_mmr":        int(_peak_mmr),
        "u_score":         float(core["u_score"]),
        "bot2_count":      int(core["bot2_count"]),
        "mmr_milestones":  json.dumps(_mmr_milestones),
        "matchup_scaling": compute_matchup_scaling(games),
        "updated_at":      _utc_now_iso_z(),
    })

def compute_matchup_scaling(games):
    return dstats.compute_matchup_scaling(games)


def lb_top_n(metric, n=TOP_N, higher_is_better=True, season=CURRENT_SEASON):
    if TOPLIST_BACKEND == "supabase":
        return _sb_top_n(metric, n, higher_is_better, season=season)
    else:
        return _session_top_n(metric, n, higher_is_better)


# ── Single-player fetch & calculate ───────────────────────────────────────────

def _snapshots_to_games(snapshots, season_start_str=None):
    if season_start_str is None:
        season_start_str = SEASONS[CURRENT_SEASON]["start"]
    return dstats.snapshots_to_games(snapshots, season_start_str, THRESHOLD_BASE, THRESHOLD_INCREASE)











# ── Leaderboard neighbor lookup ───────────────────────────────────────────────

def fetch_player_rank(player_name, region):
    # Ranken skrapas redan i fetch_and_calculate från wallii.gg — används därifrån
    return st.session_state.get("sp_rank"), None, None

# ── Wallii leaderboard (Supabase) ──────────────────────────────────────────────





























_INTEGRATION_SERVICES = IntegrationServices(
    fetch_player_links=_sb_fetch_player_links,
    stats_rows_by_player=_stats_rows_by_player,
    best_stats_for_players=_best_stats_for_players,
    display_name_for_linked_players=_display_name_for_linked_players,
    regions_for_players_current_season=_regions_for_players_current_season,
    latest_stats_by_player=_latest_stats_by_player,
)

apply_page_theme()

_PLAYER_DATA_SERVICES = PlayerDataServices(
    supabase_url=SUPABASE_URL,
    supabase_headers=SUPABASE_HEADERS,
    supabase_enabled=SUPABASE_ENABLED,
    seasons=SEASONS,
    current_season=CURRENT_SEASON,
    player_cache_fresh=_player_cache_fresh,
    snapshots_to_games=_snapshots_to_games,
    latest_day_start=_latest_day_start,
    supabase_base=SUPABASE_BASE,
    supabase_headers_for_leaderboard=_supabase_headers,
    dlog=dlog,
)
_sb_get_cached_snapshots = partial(_get_cached_player_snapshots, _PLAYER_DATA_SERVICES)
_sb_get_cached_games = partial(_get_cached_player_games, _PLAYER_DATA_SERVICES)
fetch_and_calculate = partial(_fetch_player_and_calculate, _PLAYER_DATA_SERVICES)

_ADMIN_SERVICES = AdminServices(
    supabase_headers=SUPABASE_HEADERS,
    supabase_url=SUPABASE_URL,
    cache_bust_toplists=_cache_bust_toplists,
    post_player_links=_post_player_links,
    sb_fetch_all=_sb_fetch_all,
    sb_fetch_player_links=_sb_fetch_player_links,
    sb_get_cached_snapshots=_sb_get_cached_snapshots,
    sb_load_regression=_sb_load_regression,
    sb_save_regression=_sb_save_regression,
    sb_top_n=_sb_top_n,
    split_player_link_names=_split_player_link_names,
    twitch_get_live_streams=_twitch_get_live_streams,
    yt_fetch_subscribers=_yt_fetch_subscribers,
    snapshots_to_games=_snapshots_to_games,
    binned_weighted_curve=binned_weighted_curve,
    compute_and_upsert=compute_and_upsert,
    compute_matchup_scaling=compute_matchup_scaling,
    fetch_and_calculate=fetch_and_calculate,
    fetch_top_n_for_scan=fetch_top_n_for_scan,
)
_TESTING_VIEW_SERVICES = TestingViewServices(
    app_dir=Path(__file__).parent,
    cached_binned_curve=_cached_binned_curve,
    get_csv_for_region=get_csv_for_region,
    read_curve_csv=_read_curve_csv,
    sb_fetch_all=_sb_fetch_all,
    sb_load_regression=_sb_load_regression,
)

_PROFILE_SERVICES = ProfileServices(
    country_flag=_country_flag,
    lb_stats_by_player=_lb_stats_by_player,
    mmr_milestones_after_tracking_start=_mmr_milestones_after_tracking_start,
    save_opp_buckets=_save_opp_buckets,
    sb_fetch_player_links=_sb_fetch_player_links,
    sb_get_cached_games=_sb_get_cached_games,
    sb_load_regression=_sb_load_regression,
    utc_now_iso_z=_utc_now_iso_z,
    compute_and_upsert=compute_and_upsert,
    compute_matchup_scaling=compute_matchup_scaling,
    fetch_and_calculate=fetch_and_calculate,
    fetch_neighbor_names=fetch_neighbor_names,
    get_csv_for_region=get_csv_for_region,
    go_home=go_home,
    lb_upsert_player=lb_upsert_player,
    supabase_enabled=SUPABASE_ENABLED,
)

tabs = st.tabs(["Single player", "Calculator", "Info/Explanations", "TestingStuff", "Card Library"], on_change="rerun")


# ── Single player tab ─────────────────────────────────────────────────────────

if tabs[0].open:
    with tabs[0]:
        with st.form("search_form"):
            col1, col2 = st.columns([3, 1])
            with col1:
                player = st.text_input("Player", placeholder="Name")
            with col2:
                region = st.selectbox("Region", VALID_REGIONS, index=VALID_REGIONS.index("EU"))
            season_options = sorted(SEASONS.keys(), reverse=True)
            season_labels  = [f"Season {s}" + (" (current)" if SEASONS[s]["end"] is None else "") for s in season_options]
            default_season_index = season_options.index(CURRENT_SEASON) if CURRENT_SEASON in season_options else 0
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
                st.session_state.pop("sp_profile_cache", None)
                st.session_state.pop("sp_previous_season", None)
                st.session_state.pop("sp_season_notice", None)
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

                if DEBUG:
                    with st.expander("Supabase debug", expanded=False):
                        st.caption(f"TOPLIST_BACKEND={TOPLIST_BACKEND} | SUPABASE_ENABLED={SUPABASE_ENABLED}")
                        st.caption(f"SUPABASE_URL={SUPABASE_URL}")
                        st.caption(f"SUPABASE_KEY length={len(SUPABASE_KEY)}")
                        if "sb_topn_status" in st.session_state:
                            code, txt = st.session_state["sb_topn_status"]
                            st.caption(f"TopN fetch: {code} | {txt}")

                render_leaderboard(LeaderboardServices(
                    country_flag=_country_flag,
                    fetch_patch_notes=_fetch_patch_notes,
                    lb_highest_milestone_threshold=_lb_highest_milestone_threshold,
                    lb_metric_rows=_lb_metric_rows,
                    lb_milestone_rows=_lb_milestone_rows,
                    lb_stats_by_player=_lb_stats_by_player,
                    lb_top_mmr_players=_lb_top_mmr_players,
                    player_link_display_name=_player_link_display_name,
                    sb_current_rank_map=_sb_current_rank_map,
                    sb_fetch_all=_sb_fetch_all,
                    sb_fetch_player_links=_sb_fetch_player_links,
                    sb_load_regression=_sb_load_regression,
                    stats_rows_by_player=_stats_rows_by_player,
                    stats_sort_key=_stats_sort_key,
                    twitch_get_live_streams=partial(_twitch_get_live_streams, _INTEGRATION_SERVICES),
                    yt_fetch_subscribers=partial(_yt_fetch_subscribers, _INTEGRATION_SERVICES),
                    fetch_cached_server_top10_bundle=fetch_cached_server_top10_bundle,
                    server_top10_regions=SERVER_TOP10_REGIONS,
                    dlog=dlog,
                ))
                st.markdown("<hr class='lb-row-divider lb-row-divider-paired'>", unsafe_allow_html=True)
                st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

                if st.button("Refresh leaderboards", width='stretch'):
                    _cache_bust_toplists()
                    st.rerun(scope="app")


                render_admin_panel(_ADMIN_SERVICES)

            st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

        else:
            render_player_profile(sp_player, sp_region, _PROFILE_SERVICES)


    # ── RatingAvg tab (CSV) ───────────────────────────────────────────────────────

render_testing_views(tabs, _TESTING_VIEW_SERVICES)


if tabs[4].open:
    with tabs[4]:
        show_card_browser()

if tabs[1].open:
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

            _best_pd = _ranked[0][0]
            _best_index = _placements_full.index(_best_pd)
            _result_placements = [_best_pd]
            if _best_index > 0:
                _result_placements.append(_placements_full[_best_index - 1])
            if _best_index < len(_placements_full) - 1:
                _result_placements.append(_placements_full[_best_index + 1])

            _rows_html = ""
            for _i, _pd in enumerate(_result_placements):
                _avg_opp = _p_to_avg[_pd]
                _is_best = _i == 0
                _row_bg  = f"background:{_cert_color}22;border-color:{_cert_color}66;" if _is_best else "background:#111;border-color:#1e1e1e;"
                _row_col = _cert_color if _is_best else "#8a8a8a"
                _suffixes = {1: "st", 2: "nd", 3: "rd"}
                if _pd != int(_pd):
                    _pd_str = f"{_pd:g}th"
                else:
                    _pd_str = f"{int(_pd)}{_suffixes.get(int(_pd), 'th')}"
                _prefix = "Most likely" if _is_best else "Also possible"
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

if tabs[2].open:
    with tabs[2]:
        st.markdown("<h2 style='text-decoration:none;'>About</h2>", unsafe_allow_html=True)
        st.markdown(
            "This app fetches data from [wallii.gg](https://wallii.gg) (timestamped MMR changes). "
            "Placements and all other metrics are derived from these values alone. Most metrics (that might be unclear) are explained below. <br><br>"
            "- Placements are **not** publicly available through an API. I've reverse-engineered the rating formula used by Blizzard, which is then used to calculate estimated placements using an algorithm based on MMR (same as Walli uses).<br>"
            "- Players are added to the leaderboards automatically once searched/fetched (and are eligible).<br>"
            "- If you notice any issues or wrong calculations, please report them in the form at the bottom. <br><br>"
            "Created by Darnell/Brugdar. <br><br>"
            "Contact: <br>"
            "Brugdar (Discord)<br>"
            "Darnell#21191 (Battle.net).",
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
        _now = _utc_now()
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
    _latest_label = _latest_stats_update_label(CURRENT_SEASON)
    if _latest_label:
        st.markdown(
            "<div style='text-align:center;color:#333;font-size:0.75rem;font-family:sans-serif;'>"
            f"Data last updated {html.escape(_latest_label)} &nbsp;&middot;&nbsp; v{APP_VERSION}"
            "</div>",
            unsafe_allow_html=True,
        )

    plt.close("all")

    # streamlit run wa2_app.py
    # streamlit run wa2_app.py --server.runOnSave true (auto-reload on save)
