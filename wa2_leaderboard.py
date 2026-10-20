"""Leaderboard view + dependencies."""

import html
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import streamlit as st

from wa2_achievements import (
    all_time_mmr_leaderboard,
    historical_finish_hover_html,
    historical_finish_leaderboard,
    trophy_html,
)
from wa2_config import CURRENT_SEASON, SEASONS, TOP_N, VALID_REGIONS
from wa2_icons import twitch_svg, youtube_svg
from wa2_charts import delta_color
from wa2_rating import interp_with_extrap


@dataclass(frozen=True)
class LeaderboardServices:
    country_flag: Callable[..., Any]
    fetch_patch_notes: Callable[..., Any]
    lb_highest_milestone_threshold: Callable[..., Any]
    lb_metric_rows: Callable[..., Any]
    lb_milestone_rows: Callable[..., Any]
    lb_stats_by_player: Callable[..., Any]
    lb_top_mmr_players: Callable[..., Any]
    player_link_display_name: Callable[..., Any]
    sb_current_rank_map: Callable[..., Any]
    sb_fetch_all: Callable[..., Any]
    sb_fetch_player_links: Callable[..., Any]
    sb_load_regression: Callable[..., Any]
    stats_rows_by_player: Callable[..., Any]
    stats_sort_key: Callable[..., Any]
    twitch_get_live_streams: Callable[..., Any]
    yt_fetch_subscribers: Callable[..., Any]
    fetch_cached_server_top10_bundle: Callable[..., Any]
    server_top10_regions: tuple[str, ...]
    dlog: Callable[..., Any]


@st.fragment
def render_leaderboard(services: LeaderboardServices):
    _country_flag = services.country_flag
    _fetch_patch_notes = services.fetch_patch_notes
    _lb_highest_milestone_threshold = services.lb_highest_milestone_threshold
    _lb_metric_rows = services.lb_metric_rows
    _lb_milestone_rows = services.lb_milestone_rows
    _lb_stats_by_player = services.lb_stats_by_player
    _lb_top_mmr_players = services.lb_top_mmr_players
    _player_link_display_name = services.player_link_display_name
    _sb_current_rank_map = services.sb_current_rank_map
    _sb_fetch_all = services.sb_fetch_all
    _sb_fetch_player_links = services.sb_fetch_player_links
    _sb_load_regression = services.sb_load_regression
    _stats_rows_by_player = services.stats_rows_by_player
    _stats_sort_key = services.stats_sort_key
    _twitch_get_live_streams = services.twitch_get_live_streams
    _yt_fetch_subscribers = services.yt_fetch_subscribers
    fetch_cached_server_top10_bundle = services.fetch_cached_server_top10_bundle
    SERVER_TOP10_REGIONS = services.server_top10_regions
    dlog = services.dlog

    def render_list(container, title, items, fmt, tooltip=None, asterisk_tip=None, initial_rows=5):
        HEADER_COLOR = "#8a8a8a"

        tip_html = ""
        if tooltip:
            safe_tip = html.escape(str(tooltip))
            tip_html = (
                f"<span title='{safe_tip}' "
                f"style='color:#444;font-size:0.8rem;margin-left:0.35rem;cursor:help;'>?</span>"
            )

        asterisk_html = ""
        if asterisk_tip:
            safe_ast = html.escape(str(asterisk_tip))
            asterisk_html = (
                f"<span title='{safe_ast}' "
                f"style='color:#666;font-size:0.75rem;margin-left:0.2rem;cursor:help;'>*</span>"
            )

        container.markdown(
            f"<div style='color:{HEADER_COLOR};font-size:0.85rem;text-transform:uppercase;"
            f"letter-spacing:0.08em;margin:0.25rem 0 0.45rem;font-weight:600;'>{title}{asterisk_html}{tip_html}</div>",
            unsafe_allow_html=True
        )

        if not items:
            container.markdown(
                "<div style='color:#444;font-size:0.8rem;padding:0.3rem 0;'>—</div>",
                unsafe_allow_html=True
            )
            return

        _plinks = _sb_fetch_player_links()
        _twitch_svg = twitch_svg(12, "vertical-align:middle;margin-left:4px;")
        _yt_svg     = youtube_svg(12, "vertical-align:middle;margin-left:4px;")
        _live_players = {s["player"].lower(): s["twitch_url"] for s in _twitch_get_live_streams()}
        _hover_bx, _hover_by = _sb_load_regression("ALL")

        def row_html(i, r):
            row_color = value_color = HEADER_COLOR
            if i == 1:
                row_color = value_color = "#d4a843"
            elif i == 2:
                row_color = value_color = "#bfc4c8"
            elif i == 3:
                row_color = value_color = "#b57a4a"
            player = r['player']
            region = r.get('region', '')
            link_region = r.get('profile_region', region)
            link_season = _lb_season
            linkable = True
            _profile_stats = {}
            if r.get("all_regions"):
                _historical_names = {player.lower(), *r.get("linked_players", ())}
                _profile_stats_candidates = [
                    stats_row
                    for name in _historical_names
                    for stats_row in _profile_stats_by_player.get(name, [])
                ]
                if _profile_stats_candidates:
                    _profile_stats = max(_profile_stats_candidates, key=_stats_sort_key)
                    link_region = _profile_stats.get("region", link_region)
                    link_season = _profile_stats.get("season", link_season)
                else:
                    linkable = False
            elif r.get("all_time_mmr"):
                _profile_stats_candidates = [
                    stats_row
                    for stats_row in _profile_stats_by_player.get(player.lower(), [])
                    if str(stats_row.get("region", "")).upper() == str(region).upper()
                ]
                if _profile_stats_candidates:
                    _profile_stats = max(_profile_stats_candidates, key=_stats_sort_key)
                    link_region = _profile_stats.get("region", region)
                    link_season = _profile_stats.get("season", link_season)
                else:
                    linkable = False
            link = f"?goto_player={html.escape(player)}&goto_region={html.escape(str(link_region))}&goto_season={html.escape(str(link_season))}"
            _pl    = _plinks.get(player.lower(), {})
            _icons = ""
            if _pl.get("twitch_url"):
                _icons += f"<a href='{html.escape(_pl['twitch_url'])}' target='_blank' title='Twitch'>{_twitch_svg}</a>"
            if _pl.get("youtube_url"):
                _icons += f"<a href='{html.escape(_pl['youtube_url'])}' target='_blank' title='YouTube'>{_yt_svg}</a>"
            _flag = _country_flag(_pl.get("nationality", ""))
            _live_url = _live_players.get(player.lower())
            _live_badge = (
                f"<a href='{html.escape(_live_url)}' target='_blank' "
                f"style='margin-left:6px;font-size:0.65rem;font-weight:700;color:#fff;"
                f"background:#e53935;border-radius:3px;padding:1px 5px;text-decoration:none;"
                f"vertical-align:middle;letter-spacing:0.04em;'>LIVE</a>"
            ) if _live_url else ""
            _hover_stats = _profile_stats if r.get("all_regions") or r.get("all_time_mmr") else r
            _hover_card = _hover_card_html(player, _hover_stats)
            _trophy = trophy_html(region, player, _plinks)
            if r.get("all_regions"):
                region_label = historical_finish_hover_html(r)
            else:
                region_label = html.escape(str(region))
            player_label = (
                f"<a href='{link}' target='_self' style='color:inherit;text-decoration:none;' "
                "onmouseover=\"this.style.textDecoration='underline'\" "
                f"onmouseout=\"this.style.textDecoration='none'\">{html.escape(player)}</a>"
                if linkable else html.escape(player)
            )
            return (
                "<div class='lb-hover-row' style='display:flex;justify-content:space-between;"
                "border:1px solid #1e1e1e;background:#121212;border-radius:4px;"
                "padding:0.35rem 0.5rem;margin-bottom:0.25rem;'>"
                f"<span style='color:{row_color};font-weight:700'>{i}. "
                f"<span class='lb-hover-name'>"
                f"{player_label}"
                f"{_trophy}"
                f"{_hover_card}</span> "
                f"<span style='color:#666'>({region_label})</span>"
                f"{'&nbsp;' + _flag if _flag else ''}{_icons}{_live_badge}</span>"
                f"<span style='color:{value_color};font-weight:700'>{fmt(r)}</span>"
                "</div>"
            )

        for i, r in enumerate(items[:initial_rows], 1):
            container.markdown(row_html(i, r), unsafe_allow_html=True)

        if len(items) > initial_rows:
            _more_rows = "".join(
                row_html(i, r)
                for i, r in enumerate(items[initial_rows:10], initial_rows + 1)
            )
            container.markdown(
                "<details class='lb-show-more'>"
                "<div class='lb-more-rows'>" + _more_rows + "</div>"
                "<summary><span class='lb-more-label'>▼ Show more</span>"
                "<span class='lb-less-label'>▲ Show less</span></summary>"
                "</details>",
                unsafe_allow_html=True,
            )

    _lb_season_options = sorted(SEASONS.keys(), reverse=True)
    _lb_season_index = _lb_season_options.index(CURRENT_SEASON) if CURRENT_SEASON in _lb_season_options else 0
    with st.container(key="updates_expander_wrap"):
        with st.expander("Latest updates 🔨", expanded=False):
            st.markdown(
                "- Players that have historically finished top 1 on any server now have a 🏆 next to their name. "
                "Historical achievements (top 1/10/25/50/100 leaderboard finishes) can now be seen on each player profile.\n\n"
                "- There is also a leaderboard for the most top 1 finishes and another for top 25 finishes further down. "
                "The player with the most top 1 finishes is highlighted with a 👑.\n\n"
                "- A leaderboard for the highest all-time MMR reached has also been added. Note that CN has higher historical "
                "MMR numbers due to its seasonal reset system, which was removed recently.\n\n"
                "Any ideas or feedback are greatly appreciated! Contact info can be found under the Info/Explanation tab."
            )
    _lb_season = st.radio(
        "Leaderboard season",
        options=_lb_season_options,
        format_func=lambda s: f"Season {s}" + (" (current)" if SEASONS[s]["end"] is None else ""),
        horizontal=True,
        index=_lb_season_index,
        key="lb_season_selector",
        label_visibility="collapsed",
    )
    st.markdown(
        f"<p style='color:#ccc;font-size:1.0rem;font-weight:600;margin:0.3rem 0 0.1rem;'>Leaderboards: Season {_lb_season} <span style='color:#666;font-size:0.75rem;font-weight:400;'>(Players are added when first searched, if eligible)</span></p>",
        unsafe_allow_html=True
    )
    _mmr_col, _eu_col, _na_col, _ap_col, _cn_col = st.columns([4, 0.5, 0.5, 0.5, 0.7])
    with _mmr_col:
        _mmr_filter = st.radio("MMR filter", ["All", "Top 25", "Top 50"], index=0, horizontal=True, key="lb_mmr_filter", label_visibility="collapsed")
    with _eu_col:
        _inc_eu = st.checkbox("EU", value=True,  key="lb_inc_eu")
    with _na_col:
        _inc_na = st.checkbox("NA", value=True,  key="lb_inc_na")
    with _ap_col:
        _inc_ap = st.checkbox("AP", value=True,  key="lb_inc_ap")
    with _cn_col:
        _inc_cn = st.checkbox("CN", value=False, key="lb_inc_cn", help="CN sends inconsistent (hourly) MMR updates, which means estimated placements often are slightly misleading.")

    _lb_regions = {r for r, v in [("EU", _inc_eu), ("NA", _inc_na), ("AP", _inc_ap), ("CN", _inc_cn)] if v}
    _lb_regions_key = tuple(r for r in VALID_REGIONS if r in _lb_regions)
    if _mmr_filter != "All":
        _mmr_n = int(_mmr_filter.split()[1])
        _top_mmr_players_key = _lb_top_mmr_players(_lb_season, _lb_regions_key, _mmr_n)
    else:
        _top_mmr_players_key = ()

    def _lb(metric, higher_is_better=True, n=9999, limit=TOP_N):
        rows = _lb_metric_rows(metric, higher_is_better, _lb_season, _lb_regions_key, _top_mmr_players_key)
        rows = rows[:n]
        return rows[:limit] if limit is not None else rows

    _all_stats_by_player = _lb_stats_by_player(_lb_season)
    _profile_stats_by_player = _stats_rows_by_player()
    _hover_bx, _hover_by = _sb_load_regression("ALL")
    _rank_lookup = _sb_current_rank_map()

    def _hover_card_html(player_name, stats_row):
        if not stats_row:
            return ""
        _region = html.escape(str(stats_row.get("region", "")).upper()) if stats_row.get("region") else ""
        _cr = stats_row.get("cr")
        _meta = " ".join(x for x in [_region, f"{int(_cr):,}" if _cr is not None else ""] if x)
        _rank = _rank_lookup.get((str(player_name).lower(), str(stats_row.get("region", "")).upper()))
        _rank_html = f" <span class='lb-hover-rank'>#{_rank}</span>" if _rank is not None else ""
        _avg_val = stats_row.get("avg_place")
        _avg = f"{_avg_val:.2f}" if _avg_val is not None else "—"
        _avg_color = "#777"
        _hover_cr = stats_row.get("cr")
        if _avg_val is not None and _hover_cr is not None and _hover_bx is not None and len(_hover_bx) >= 2:
            try:
                _expected_avg = interp_with_extrap(_hover_cr, _hover_bx, _hover_by)
                _expected_avg = float(np.clip(_expected_avg, 1.0, 8.0))
                _avg_color = delta_color(float(_avg_val - _expected_avg))
            except Exception:
                _avg_color = "#ddd"
        _top1 = f"{stats_row['first_pct']:.1f}%" if stats_row.get("first_pct") is not None else "—"
        _top4 = f"{stats_row['top4_pct']:.1f}%" if stats_row.get("top4_pct") is not None else "—"
        _u_val = stats_row.get("u_score")
        _u_color = (
            "#b388e8" if _u_val is not None and _u_val >= 0.2
            else "#aaa" if _u_val is not None and _u_val >= -0.1
            else "#5b8fd4" if _u_val is not None
            else "#777"
        )
        _u_text = f"{_u_val:+.2f}" if _u_val is not None else "—"
        _ff_raw = stats_row.get("matchup_scaling")
        _ff_val = -_ff_raw if _ff_raw is not None else None
        _ff_color = (
            "#7ab87a" if _ff_val is not None and _ff_val >= 0.35
            else "#d4a843" if _ff_val is not None and _ff_val >= -0.15
            else "#c47a75" if _ff_val is not None
            else "#777"
        )
        _ff_text = f"{_ff_val:+.2f}" if _ff_val is not None else "—"
        return (
            f"<div class='lb-hover-card'>"
            f"<div class='lb-hover-title'><span>{html.escape(player_name)}{_rank_html}</span><span class='lb-hover-meta'>{_meta}</span></div>"
            f"<div class='lb-hover-grid'>"
            f"<span class='lb-hover-label'>Avg</span><span class='lb-hover-value' style='color:{_avg_color};'>{_avg}</span>"
            f"<span class='lb-hover-label'>Top 1%</span><span class='lb-hover-value'>{_top1}</span>"
            f"<span class='lb-hover-label'>Top 4%</span><span class='lb-hover-value'>{_top4}</span>"
            f"<span class='lb-hover-label'>Aggression</span><span class='lb-hover-value' style='color:{_u_color};'>{_u_text}</span>"
            f"<span class='lb-hover-label'>Farmer Factor</span><span class='lb-hover-value' style='color:{_ff_color};'>{_ff_text}</span>"
            f"</div></div>"
        )

    if _lb_season == CURRENT_SEASON:
        if "lb_show_server_top10" not in st.session_state:
            st.session_state["lb_show_server_top10"] = False
        _server_toggle_label = (
            "Hide server top 10"
            if st.session_state["lb_show_server_top10"]
            else "Show Top 10 (EU / NA / AP)"
        )
        _patch_col1, _patch_col2 = st.columns([4, 1])
        with _patch_col1:
            if st.button(_server_toggle_label, key="lb_server_top10_toggle"):
                st.session_state["lb_show_server_top10"] = not st.session_state["lb_show_server_top10"]
                st.rerun(scope="fragment")
        with _patch_col2:
            _latest_patch = _fetch_patch_notes(1)
            if _latest_patch:
                st.link_button(f"📋 {_latest_patch[0]['title']}", _latest_patch[0]["url"], use_container_width=True)
        if st.session_state["lb_show_server_top10"]:
            try:
                _server_top10_bundle = fetch_cached_server_top10_bundle(10)
            except Exception as e:
                dlog("Cached server top 10 bundle failed:", e)
                _server_top10_bundle = {}
            _server_cols = st.columns(3)
            for _server_col, _server_region in zip(_server_cols, SERVER_TOP10_REGIONS):
                _server_rows = _server_top10_bundle.get(_server_region, [])
                render_list(
                    _server_col,
                    f"{_server_region} Top 10",
                    _server_rows,
                    lambda r: f"{int(r['cr']):,}" if r.get("cr") is not None else f"#{int(r['current_rank'])}" if r.get("current_rank") is not None else "-",
                    "Cached current Wallii server leaderboard. CN is excluded.",
                )
        st.markdown("<hr class='lb-row-divider lb-row-divider-top'>", unsafe_allow_html=True)

    lists = [
        ("Avg placement🏅",       _lb("avg_place",    higher_is_better=False),  lambda r: f"{r['avg_place']:.2f}<span style='color:#555;font-size:0.78em;margin-left:3px;'>avg</span>",    "Average placement across recorded games this season."),
        ("Top 1 %📈",             _lb("first_pct",    higher_is_better=True),   lambda r: f"{r['first_pct']:.1f}%",   "Percentage of games finished in 1st place this season."),
        ("Hot streak🔥",          _lb("hot_streak",   higher_is_better=True),   lambda r: f"{int(r['hot_streak'])} games",   "Longest consecutive streak of 1st placements."),
        ("Top 4 %📈",             _lb("top4_pct",     higher_is_better=True),   lambda r: f"{r['top4_pct']:.1f}%",    "Percentage of games finished in top 4 this season."),
        ("Roach streak🪳",        _lb("roach_streak", higher_is_better=True),   lambda r: f"{int(r['roach_streak'])} games", "Longest consecutive streak of top 4 finishes."),
        ("Top 1 region finishes🥇",      historical_finish_leaderboard("top1", _sb_fetch_player_links(), _lb_regions_key),  lambda r: f"{int(r['count'])}", "Total historical first-place finishes in the selected regions. 'bS1' refers to the (beta) seasons before the game got formally released."),
        ("Top 25 region finishes🥈",     historical_finish_leaderboard("top25", _sb_fetch_player_links(), _lb_regions_key), lambda r: f"{int(r['count'])}", "Total historical Top 25 finishes in the selected regions. 'bS1' refers to the (beta) seasons before the game got formally released."),
        ("Lowest tilt factor😎",  [r for r in _lb("tilt_factor", higher_is_better=False, limit=None) if (r.get("bot2_count") or 0) >= 30][:TOP_N], lambda r: f"{r['tilt_factor']:.2f}<span style='color:#555;font-size:0.78em;margin-left:2px;'>x</span>" if r.get("tilt_factor") is not None else "—", "Measures how much a player is affected by a bad placements. The value shows how much worse their avg placement becomes after a 7th/8th compared to their overall avg. Lower = less affected by tilt.", "(Min 30 games with 7th/8th placement)"),
        ("Highest tilt factor😤", [r for r in _lb("tilt_factor", higher_is_better=True,  limit=None) if (r.get("bot2_count") or 0) >= 30][:TOP_N], lambda r: f"{r['tilt_factor']:.2f}<span style='color:#555;font-size:0.78em;margin-left:2px;'>x</span>" if r.get("tilt_factor") is not None else "—", "Measures how much a player is affected by a bad placement. The value shows how much worse their avg placement becomes after a 7th/8th compared to their overall avg. Higher = more affected by tilt.", "(Min 30 games with 7th/8th placement)"),
        ("Most aggressive👺",     _lb("u_score",      higher_is_better=True),   lambda r: f"{r['u_score']:+.2f}<span style='color:#555;font-size:0.85em;margin-left:3px;'>u</span>" if r.get("u_score") is not None else "—", "Measures play style based on placement distribution. Aggressive players finish at the extremes more often; more 1st and 7th/8th places - suggesting a high-risk/high-reward style. Higher = more aggressive."),
        ("Most defensive🫣",      [r for r in _lb("u_score", higher_is_better=False, limit=None) if (r.get("cr") or 0) >= 10000][:TOP_N],  lambda r: f"{r['u_score']:+.2f}<span style='color:#555;font-size:0.85em;margin-left:3px;'>&#8745;</span>" if r.get("u_score") is not None else "—", "Measures play style based on placement distribution. Defensive players finish in the middle more often; fewer 1st and 7th/8th places - suggesting a safer low-risk style. Lower = more defensive.", "Min 10,000 MMR"),
        ("Best form🏋️‍♂️",           _lb("form_diff",    higher_is_better=False),  lambda r: f"{(r['avg_place'] + r['form_diff']):.2f}<span style='color:#555;font-size:0.78em;margin-left:3px;'>avg</span> ({r['form_diff']:+.2f})" if r.get("form_diff") is not None and r.get("avg_place") is not None else "—", "Difference between form (last 50 games) and overall avg. More negative = better form relative to baseline."),
        ("Best 'form rating🏋️‍♂️'",    [r for r in _lb("form_rating", higher_is_better=True, limit=None) if r.get("form_rating") is not None and (r.get("games") or 0) >= 300][:TOP_N], lambda r: f"{r['form_rating']:,}<span style='color:#555;font-size:0.78em;margin-left:3px;'>mmr</span>", "Estimated MMR based on last 50 games avg placement on the regression curve. Requires at least 300 games this season."),
        ("Largest MMR drop🍂",    _lb("max_drawdown", higher_is_better=True),   lambda r: f"<span title='{html.escape(r['dd_detail'])}' style='cursor:help;'>-{int(r['max_drawdown']):,} MMR</span>" if r.get("dd_detail") else (f"-{int(r['max_drawdown']):,} MMR" if r.get("max_drawdown") is not None else "—"), "Largest MMR drop from a peak to a subsequent low."),
        ("# Games🎲",             _lb("games",        higher_is_better=True),   lambda r: f"{int(r['games'])} games",  "Total number of games played this season while on the leaderboard."),
        ("Farmer factor 🧑‍🌾",       [r for r in _lb("matchup_scaling", higher_is_better=False, limit=None) if r.get("matchup_scaling") is not None and (r.get("games") or 0) >= 300][:TOP_N], lambda r: f"{-r['matchup_scaling']:+.2f}<span style='color:#555;font-size:0.78em;margin-left:2px;'>x</span>", "Measures how much better a player performs against weaker opponents relative to stronger ones. Higher = more dominant vs weaker lobbies. (Experimental)", "Min 300 games at 10k+ MMR"),
        ("Lowest farmer factor 🧑‍🌾", [r for r in _lb("matchup_scaling", higher_is_better=True,  limit=None) if r.get("matchup_scaling") is not None and (r.get("games") or 0) >= 300][:TOP_N], lambda r: f"{-r['matchup_scaling']:+.2f}<span style='color:#555;font-size:0.78em;margin-left:2px;'>x</span>", "Measures how much better a player performs against stronger opponents relative to weaker ones. Lower farmer factor = better in higher mmr lobbies. (Experimental)", "Min 300 games at 10k+ MMR"),

    ]
    _all_time_mmr_rows = all_time_mmr_leaderboard(
        _lb_regions_key,
        current_rows=_sb_fetch_all(season=CURRENT_SEASON),
        limit=TOP_N,
    )
    for _row in _all_time_mmr_rows:
        _row["all_time_mmr"] = True
    lists.append((
        "All-time highest MMR🐐",
        _all_time_mmr_rows,
        lambda r: f"<span title='{html.escape('bS1 rating; an exploit allowed unusually high external MMRs this season.' if r.get('season_label') == 'bS1' else str(r.get('season_label') or ('Season %s' % r['season'])))}' style='cursor:help;'>{r['rating']:,}{'*' if r.get('season_label') == 'bS1' else ''}</span>",
        "Highest recorded leaderboard rating per player. Hover over a rating to see the season it was reached.",
    ))
    lists[2], lists[3] = lists[3], lists[2]
    lists[13], lists[17] = lists[17], lists[13]

    # ── Rad 0: Avg placement (vänster) + Live Now (höger) ─────────────
    _row0 = st.columns(2)
    _t0, _it0, _fm0, _tp0, *_rst0 = lists[0]
    render_list(_row0[0], _t0, _it0, _fm0, _tp0, asterisk_tip=_rst0[0] if _rst0 else None)

    # ── Live streams (höger, rad 0) ────────────────────────────────────
    _live_streams_lb = _twitch_get_live_streams()
    _live_col = _row0[1]
    HEADER_COLOR = "#c45151"
    _total_viewers = sum(s["viewers"] for s in _live_streams_lb)
    _viewers_str = f" <span style='color:#666;font-size:0.78em;font-weight:400;text-transform:none;letter-spacing:0;'>({_total_viewers:,} total viewers)</span>" if _total_viewers > 0 else ""
    _live_col.markdown(
        f"<div style='color:{HEADER_COLOR};font-size:0.85rem;text-transform:uppercase;"
        f"letter-spacing:0.08em;margin:0.25rem 0 0.45rem;font-weight:600;'>"
        f"{twitch_svg(10, 'vertical-align:middle;margin-right:5px;')}"
        f"Live now{_viewers_str}</div>",
        unsafe_allow_html=True,
    )
    if "lb_live_sort" not in st.session_state:
        st.session_state["lb_live_sort"] = "mmr"
    _live_sort_key = st.session_state["lb_live_sort"]
    _live_streams_lb = sorted(
        _live_streams_lb,
        key=lambda x: x["viewers"] if _live_sort_key == "viewers" else (x.get("season", 0), x.get("cr", 0)),
        reverse=True,
    )[:10]
    _empty_row = "<div style='display:flex;border:1px solid #1e1e1e;background:#121212;border-radius:4px;padding:0.35rem 0.5rem;margin-bottom:0.25rem;'><span style='color:#1e1e1e;'>—</span></div>"

    def _live_display_player(s):
        _live_links = _sb_fetch_player_links()
        stats_candidates = s.get("stats_rows", [])
        stats_row = (
            max(stats_candidates, key=_stats_sort_key)
            if stats_candidates
            else _all_stats_by_player.get(s["player"].lower(), {})
        )
        for linked_player in s.get("linked_players", []):
            display_name = _player_link_display_name(linked_player, _live_links)
            if display_name:
                return display_name, stats_row
        return (stats_row.get("player") or s["player"]), stats_row

    def _live_row_html(si, s):
        display_player, display_stats = _live_display_player(s)
        display_region = display_stats.get("region", s.get("region", ""))
        display_cr = display_stats.get("cr", s.get("cr", ""))
        display_cr_text = f"{display_cr:,}" if isinstance(display_cr, (int, float)) else str(display_cr)
        display_links = _sb_fetch_player_links().get(display_player.lower(), {})
        s_twitch = html.escape(s["twitch_url"])
        s_title  = html.escape(s["title"][:50] + ("..." if len(s["title"]) > 50 else ""))
        s_color  = "#d4a843" if si == 1 else "#bfc4c8" if si == 2 else "#b57a4a" if si == 3 else "#8a8a8a"
        s_profile = f"?goto_player={html.escape(display_player)}&goto_region={html.escape(display_region)}&goto_season={_lb_season}"
        _hover_card = _hover_card_html(display_player, display_stats)
        _trophy = trophy_html(display_region, display_player, _sb_fetch_player_links())
        return (
            f"<div class='lb-hover-row' style='display:flex;justify-content:space-between;"
            f"border:1px solid #1e1e1e;background:#121212;border-radius:4px;"
            f"padding:0.35rem 0.5rem;margin-bottom:0.25rem;'>"
            f"<span style='color:{s_color};font-weight:700'>{si}. "
            f"<span class='lb-hover-name'>"
            f"<a href='{s_profile}' target='_self' style='color:inherit;text-decoration:none;' "
            f"onmouseover=\"this.style.textDecoration='underline'\" "
            f"onmouseout=\"this.style.textDecoration='none'\">{display_player}</a>"
            f"{_trophy}"
            f"{_hover_card}</span>"
            f"{_country_flag(display_links.get('nationality', s.get('nationality','')))}"
            f"<span style='color:#666;font-size:0.8rem;font-weight:400;margin-left:0.4rem;'>({str(display_region).upper()} {display_cr_text})</span>"
            f"<a href='{s_twitch}' target='_blank' title='{s_title}' style='margin-left:5px;'>"
            f"{twitch_svg(11)}"
            f"</a>"
            f"<a href='{s_twitch}' target='_blank' style='margin-left:6px;font-size:0.65rem;"
            f"font-weight:700;color:#fff;background:#e53935;border-radius:3px;padding:1px 5px;"
            f"text-decoration:none;vertical-align:middle;letter-spacing:0.04em;'>LIVE</a>"
            f"</span>"
            f"<span style='color:{s_color};font-weight:700'><span style='color:#eb0400;font-size:0.6rem;vertical-align:middle;margin-right:4px;'>&#9679;</span>{s['viewers']:,}<span style='color:#555;font-size:0.78em;margin-left:3px;'>viewers</span></span>"
            f"</div>"
        )

    for _si, _s in enumerate(_live_streams_lb[:5], 1):
        _live_col.markdown(_live_row_html(_si, _s), unsafe_allow_html=True)
    for _ in range(max(0, 5 - len(_live_streams_lb))):
        _live_col.markdown(_empty_row, unsafe_allow_html=True)

    _live_more_rows = "".join(
        _live_row_html(_si, _s)
        for _si, _s in enumerate(_live_streams_lb[5:10], 6)
    )
    _live_extra_count = min(5, max(0, len(_live_streams_lb) - 5))
    _live_btn_cols = _live_col.columns([1, 1])
    if _live_more_rows:
        _live_btn_cols[0].markdown(
            f"<details class='lb-show-more lb-show-more-live lb-live-extra-{_live_extra_count}'>"
            "<div class='lb-more-rows'>" + _live_more_rows + "</div>"
            "<summary><span class='lb-more-label'>▼ Show more</span>"
            "<span class='lb-less-label'>▲ Show less</span></summary>"
            "</details>",
            unsafe_allow_html=True,
        )
    else:
        _live_btn_cols[0].markdown(
            "<div class='lb-show-more'><span class='lb-show-more-disabled'>▼ Show more</span></div>",
            unsafe_allow_html=True,
        )
    _sort_label = "MMR" if _live_sort_key == "mmr" else "Views"
    if _live_btn_cols[1].button(f"Sort: {_sort_label}", key="lb_live_sort_btn"):
        st.session_state["lb_live_sort"] = "viewers" if _live_sort_key == "mmr" else "mmr"
        st.rerun(scope="fragment")

    st.markdown("<hr class='lb-row-divider lb-row-divider-top'>", unsafe_allow_html=True)

    # ── Återstående listor i par (vänster/höger per rad) ───────────────
    for _i in range(0, len(lists) - 1, 2):
        _pair = st.columns(2)
        _li, _ri = _i + 1, _i + 2
        _t, _it, _fm, _tp, *_rst = lists[_li]
        render_list(_pair[0], _t, _it, _fm, _tp, asterisk_tip=_rst[0] if _rst else None)
        if _ri < len(lists):
            _t, _it, _fm, _tp, *_rst = lists[_ri]
            render_list(_pair[1], _t, _it, _fm, _tp, asterisk_tip=_rst[0] if _rst else None)
        st.markdown("<hr class='lb-row-divider lb-row-divider-paired'>", unsafe_allow_html=True)

    # ── Sista raden: YouTube (vänster) + First to Xk (höger) ──────────
    _final_row = st.columns(2)

    # ── YouTube leaderboard ────────────────────────────────────────────
    _yt_subs = _yt_fetch_subscribers()
    _yt_svg = youtube_svg(11, "vertical-align:middle;margin-right:5px;")
    _next_col = _final_row[0]
    _next_col.markdown(
        f"<div style='color:#8a8a8a;font-size:0.85rem;text-transform:uppercase;"
        f"letter-spacing:0.08em;margin:0.25rem 0 0.45rem;font-weight:600;'>"
        f"{_yt_svg}Most subscribers</div>",
        unsafe_allow_html=True,
    )

    def _yt_row_html(yi, y):
        y_color  = "#d4a843" if yi == 1 else "#bfc4c8" if yi == 2 else "#b57a4a" if yi == 3 else "#8a8a8a"
        y_url    = html.escape(y["youtube_url"])
        subs     = y["subscribers"]
        subs_fmt = f"{subs/1_000_000:.1f}M" if subs >= 1_000_000 else f"{subs/1_000:.1f}k" if subs >= 1_000 else str(subs)
        return (
            f"<div style='display:flex;justify-content:space-between;"
            f"border:1px solid #1e1e1e;background:#121212;border-radius:4px;"
            f"padding:0.35rem 0.5rem;margin-bottom:0.25rem;'>"
            f"<span style='color:{y_color};font-weight:700'>{yi}. "
            f"<a href='{y_url}' target='_blank' style='color:inherit;text-decoration:none;' "
            f"onmouseover=\"this.style.textDecoration='underline'\" "
            f"onmouseout=\"this.style.textDecoration='none'\">{y['player']}</a>"
            f"{_country_flag(y.get('nationality',''))}"
            f"<a href='{y_url}' target='_blank' style='margin-left:5px;'>"
            f"{youtube_svg(11)}"
            f"</a>"
            f"</span>"
            f"<span style='color:{y_color};font-weight:700'>{subs_fmt}<span style='color:#555;font-size:0.78em;margin-left:3px;'>subs</span></span>"
            f"</div>"
        )

    _yt_empty_row = "<div style='display:flex;border:1px solid #1e1e1e;background:#121212;border-radius:4px;padding:0.35rem 0.5rem;margin-bottom:0.25rem;'><span style='color:#1e1e1e;'>—</span></div>"
    for _yi, _y in enumerate(_yt_subs[:5], 1):
        _next_col.markdown(_yt_row_html(_yi, _y), unsafe_allow_html=True)
    for _ in range(max(0, 5 - len(_yt_subs))):
        _next_col.markdown(_yt_empty_row, unsafe_allow_html=True)

    _yt_more_rows = "".join(
        _yt_row_html(_yi, _y)
        for _yi, _y in enumerate(_yt_subs[5:10], 6)
    )
    if _yt_more_rows:
        _next_col.markdown(
            "<details class='lb-show-more'>"
            "<div class='lb-more-rows'>" + _yt_more_rows + "</div>"
            "<summary><span class='lb-more-label'>▼ Show more</span>"
            "<span class='lb-less-label'>▲ Show less</span></summary>"
            "</details>",
            unsafe_allow_html=True,
        )
    else:
        _next_col.markdown(
            "<div class='lb-show-more'><span class='lb-show-more-disabled'>▼ Show more</span></div>",
            unsafe_allow_html=True,
        )

    # ── First to Xk (dynamic milestone card) ──────────────────────────
    st.markdown(
        "<style>label[for='lb_milestone']{"
        "color:#8a8a8a !important;font-size:0.85rem !important;"
        "text-transform:uppercase !important;letter-spacing:0.08em !important;"
        "font-weight:600 !important;}</style>",
        unsafe_allow_html=True,
    )
    _milestone_options = [f"{i}k" for i in range(10, 22)]
    _highest_milestone = _lb_highest_milestone_threshold(
        _lb_season,
        _lb_regions_key,
        _top_mmr_players_key,
    )
    _default_milestone = min(21, max(10, (_highest_milestone or 10000) // 1000))
    _milestone_k = _final_row[1].selectbox(
        "First to",
        _milestone_options,
        index=_default_milestone - 10,
        key=(
            f"lb_milestone_{_lb_season}_"
            f"{'_'.join(_lb_regions_key)}_{_mmr_filter.replace(' ', '')}"
        ),
    )
    _milestone_thresh = str(int(_milestone_k[:-1]) * 1000)
    _milestone_rows = _lb_milestone_rows(
        _lb_season,
        _milestone_thresh,
        _lb_regions_key,
        _top_mmr_players_key,
    )[:TOP_N]
    render_list(
        _final_row[1],
        f"First to {_milestone_k}",
        _milestone_rows,
        lambda r: datetime.fromisoformat(r["_mdate"].replace("Z", "+00:00")).strftime("%b %d"),
        tooltip=f"First players to reach {_milestone_k} MMR this season.",
        initial_rows=3,
    )
