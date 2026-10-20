"""Profile workflow and rendering."""

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

from domain.stats import compute_player_stats
from wa2_achievements import achievements_html
from wa2_charts import (
    delta_color,
    diff_pct_color,
    make_chart,
    make_neighbor_chart,
    norm_to_pct,
    normalized_counts,
    summarize_neighbor_differences,
)
from wa2_config import (
    CURRENT_SEASON,
    DEBUG,
    ENABLE_SESSION_TOPLISTS,
    SEASONS,
    TOPLIST_BACKEND,
    VALID_REGIONS,
)
from wa2_icons import twitch_link, youtube_link
from wa2_rating import interp_with_extrap, load_rating_curve


@dataclass(frozen=True)
class ProfileServices:
    country_flag: Callable[..., Any]
    lb_stats_by_player: Callable[..., Any]
    mmr_milestones_after_tracking_start: Callable[..., Any]
    save_opp_buckets: Callable[..., Any]
    sb_fetch_player_links: Callable[..., Any]
    sb_get_cached_games: Callable[..., Any]
    sb_load_regression: Callable[..., Any]
    utc_now_iso_z: Callable[..., Any]
    compute_and_upsert: Callable[..., Any]
    compute_matchup_scaling: Callable[..., Any]
    fetch_and_calculate: Callable[..., Any]
    fetch_neighbor_names: Callable[..., Any]
    get_csv_for_region: Callable[..., Any]
    go_home: Callable[..., Any]
    lb_upsert_player: Callable[..., Any]
    supabase_enabled: bool


def _cache_profile_season(season, games, region, rank):
    """Keep already visited profile seasons in this browser session."""
    cache = st.session_state.setdefault("sp_profile_cache", {})
    cache[season] = {"games": games, "region": region, "rank": rank}


def _restore_previous_profile_season(message):
    """Return to the last working season if the newly selected one has no data."""
    previous = st.session_state.get("sp_previous_season")
    cached = st.session_state.get("sp_profile_cache", {}).get(previous)
    if not cached:
        return False
    st.session_state["sp_season"] = previous
    st.session_state["sp_games"] = cached["games"]
    st.session_state["sp_region"] = cached["region"]
    st.session_state["sp_rank"] = cached["rank"]
    st.session_state["profile_season_toggle"] = previous
    st.session_state["sp_season_notice"] = (
        f"No data was found for Season {st.session_state.get('sp_requested_season')}. "
        f"You're still viewing Season {previous}."
    )
    return True


def _compute_player_stats(games):
    return compute_player_stats(games, normalized_counts)


def render_player_profile(sp_player, sp_region, services: ProfileServices):
    _country_flag = services.country_flag
    _lb_stats_by_player = services.lb_stats_by_player
    _mmr_milestones_after_tracking_start = services.mmr_milestones_after_tracking_start
    _save_opp_buckets = services.save_opp_buckets
    _sb_fetch_player_links = services.sb_fetch_player_links
    _sb_get_cached_games = services.sb_get_cached_games
    _sb_load_regression = services.sb_load_regression
    _utc_now_iso_z = services.utc_now_iso_z
    compute_and_upsert = services.compute_and_upsert
    compute_matchup_scaling = services.compute_matchup_scaling
    fetch_and_calculate = services.fetch_and_calculate
    fetch_neighbor_names = services.fetch_neighbor_names
    get_csv_for_region = services.get_csv_for_region
    go_home = services.go_home
    lb_upsert_player = services.lb_upsert_player
    SUPABASE_ENABLED = services.supabase_enabled

    _sp_season = st.session_state.get("sp_season", CURRENT_SEASON)
    _cached_profile = st.session_state.get("sp_profile_cache", {}).get(_sp_season)
    if st.session_state.get("sp_games") is None and _cached_profile:
        st.session_state["sp_games"] = _cached_profile["games"]
        st.session_state["sp_region"] = _cached_profile["region"]
        st.session_state["sp_rank"] = _cached_profile["rank"]

    if st.session_state.get("sp_games") is None:
        with st.spinner("Loading player profile..."):
            try:
                st.session_state["sp_games"], st.session_state["sp_region"], st.session_state["sp_rank"] = fetch_and_calculate(sp_player, sp_region, season=_sp_season)
                compute_and_upsert(sp_player, st.session_state["sp_region"], st.session_state["sp_games"], season=_sp_season)
                _save_opp_buckets(sp_player, st.session_state["sp_region"], st.session_state["sp_games"])
                _cache_profile_season(_sp_season, st.session_state["sp_games"], st.session_state["sp_region"], st.session_state["sp_rank"])
                st.session_state.pop("sp_season_notice", None)
            except ValueError as e:
                msg = str(e)
                m = re.search(r"appears to be in: ([A-Z,\s]+)", msg)
                if m:
                    detected = m.group(1).strip().split(", ")[0]
                    if detected in VALID_REGIONS:
                        st.info(f"Not found in {sp_region} - retrying as {detected}...")
                        try:
                            st.session_state["sp_region"] = detected
                            st.session_state["sp_games"], st.session_state["sp_region"], st.session_state["sp_rank"] = fetch_and_calculate(sp_player, detected, season=_sp_season)
                            sp_region = detected
                            compute_and_upsert(sp_player, st.session_state["sp_region"], st.session_state["sp_games"], season=_sp_season)
                            _save_opp_buckets(sp_player, detected, st.session_state["sp_games"])
                            _cache_profile_season(_sp_season, st.session_state["sp_games"], st.session_state["sp_region"], st.session_state["sp_rank"])
                            st.session_state.pop("sp_season_notice", None)
                        except Exception as e2:
                            if _restore_previous_profile_season(str(e2)):
                                st.rerun()
                            st.error(str(e2))
                            st.session_state["sp_games"] = []
                    else:
                        if _restore_previous_profile_season(msg):
                            st.rerun()
                        st.error(msg)
                        st.session_state["sp_games"] = []
                else:
                    if _restore_previous_profile_season(msg):
                        st.rerun()
                    st.error(msg)
                    st.session_state["sp_games"] = []
            except Exception as e:
                if _restore_previous_profile_season(str(e)):
                    st.rerun()
                st.error(str(e))
                st.session_state["sp_games"] = []


    games = st.session_state.get("sp_games", [])
    sp_region = st.session_state.get("sp_region", sp_region)

    if _season_notice := st.session_state.pop("sp_season_notice", None):
        st.warning(_season_notice)

    if games:
        try:
            norm  = normalized_counts(games)
            total = len(games)
            avg   = sum(g["placement"] for g in games) / total
            wins  = norm[1]
            top4  = sum(norm[p] for p in [1, 2, 3, 4])
            _eps        = 0.5
            _part1      = np.log((norm[1] + _eps) / (norm[2] + norm[3] + norm[4] + _eps))
            _part2      = np.log((norm[7] + norm[8] + _eps) / (norm[5] + norm[6] + _eps))
            u_score_val = 0.5 * (_part1 + _part2)
            current_mmr = games[-1]["mmr_after"]
            peak_mmr = max(
                max(g["mmr_before"] for g in games),
                max(g["mmr_after"]  for g in games),
            )
            diff_to_cr = current_mmr - peak_mmr

            def stat(label, value, value_color="#eee", label_tip=None):
                tip_html = ""
                if label_tip:
                    safe_tip = html.escape(str(label_tip))
                    tip_html = (
                        "<span title='" + safe_tip + "' "
                        "style='color:#444;font-size:0.8rem;margin-left:0.35rem;cursor:help;'>?</span>"
                    )
                return (
                    '<div style="background:#161616;border:1px solid #2a2a2a;border-radius:4px;padding:0.6rem 0.8rem;">'
                    '<div style="color:#555;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">'
                    + label + tip_html +
                    '</div>'
                    '<div style="color:' + value_color + ';font-size:1.1rem;font-weight:600;">'
                    + str(value) +
                    '</div>'
                    '</div>'
                )

            expected_avg = None
            delta = None
            _curve_source = None
            bx, by = _sb_load_regression("ALL")
            if bx is not None and len(bx) >= 2:
                _curve_source = "supabase"
            else:
                # fallback to CSV
                curve_csv = get_csv_for_region(sp_region)
                if curve_csv.exists():
                    try:
                        mtime = curve_csv.stat().st_mtime
                        bx, by = load_rating_curve(
                            str(curve_csv),
                            mtime,
                            x_choice="current_mmr",
                            bin_size=1000,
                            mode_kind="wmean",
                            min_games=300,
                        )
                        _curve_source = "csv"
                    except Exception:
                        bx, by = None, None

            if bx is not None and len(bx) >= 2:
                try:
                    expected_avg = interp_with_extrap(current_mmr, bx, by)
                    expected_avg = float(np.clip(expected_avg, 1.0, 8.0))
                    delta = float(avg - expected_avg)
                except Exception:
                    pass

            avg_color = "#eee"
            avg_tip = None
            if expected_avg is not None and delta is not None:
                avg_color = delta_color(delta)
                sign = "+" if delta >= 0 else ""
                avg_tip = (
                    f"Expected at CR {current_mmr:,}: {expected_avg:.2f} | "
                    f"Δ: {sign}{delta:.2f} (avg - expected) | "
                    f"Source: {_curve_source}"
                )

            # Header row: [←] player [region] [#rank]  season toggle on the right
            hL, hR, hT = st.columns([0.7, 5.0, 3.5], vertical_alignment="center")

            with hL:
                st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
                if st.button("←", key="home_btn_icon", width='stretch'):
                    go_home()
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            with hT:
                recent_seasons = sorted(SEASONS.keys(), reverse=True)
                current_profile_season = st.session_state.get("sp_season", CURRENT_SEASON)
                if current_profile_season not in recent_seasons:
                    current_profile_season = CURRENT_SEASON
                season_toggle = st.radio(
                    "",
                    recent_seasons,
                    index=recent_seasons.index(current_profile_season),
                    format_func=lambda s: f"Season {s}",
                    horizontal=True,
                    key="profile_season_toggle",
                    label_visibility="collapsed",
                )
                if season_toggle != st.session_state.get("sp_season"):
                    current_season = st.session_state.get("sp_season", CURRENT_SEASON)
                    _cache_profile_season(
                        current_season,
                        st.session_state.get("sp_games", []),
                        st.session_state.get("sp_region", sp_region),
                        st.session_state.get("sp_rank"),
                    )
                    st.session_state["sp_previous_season"] = current_season
                    st.session_state["sp_requested_season"] = season_toggle
                    st.session_state["sp_season"] = season_toggle
                    cached_season = st.session_state.get("sp_profile_cache", {}).get(season_toggle)
                    if cached_season:
                        st.session_state["sp_games"] = cached_season["games"]
                        st.session_state["sp_region"] = cached_season["region"]
                        st.session_state["sp_rank"] = cached_season["rank"]
                    else:
                        st.session_state["sp_games"] = None
                        st.session_state["sp_rank"] = None
                    st.rerun()

            with hR:
                player_rank_display = st.session_state.get("sp_rank")
                rank_str = (
                    f" <span style='color:#999;font-size:0.8rem;margin-left:0.5rem;'>#{player_rank_display}</span>"
                    if player_rank_display else ""
                )
                _pl_links  = _sb_fetch_player_links().get(sp_player.lower(), {})
                _pl_flag   = _country_flag(_pl_links.get("nationality", ""))
                _hdr_icons = ""
                if _pl_links.get("twitch_url"):
                    _hdr_icons += twitch_link(_pl_links["twitch_url"], size=14)
                if _pl_links.get("youtube_url"):
                    _hdr_icons += youtube_link(_pl_links["youtube_url"], size=14)
                st.markdown(
                    "<p style='color:#eee;font-size:1.1rem;margin:1.2rem 0 0.8rem;'>"
                    + sp_player
                    + (" &nbsp;" + _pl_flag if _pl_flag else "")
                    + " <span style='color:#d4a843;font-size:0.8rem;margin-left:0.5rem;'>" + sp_region + "</span>"
                    + rank_str
                    + _hdr_icons
                    + "</p>",
                    unsafe_allow_html=True
                )

            _achievement_row = achievements_html(sp_region, sp_player, _sb_fetch_player_links())
            if _achievement_row:
                st.markdown(_achievement_row, unsafe_allow_html=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(stat("Games",     str(total)),                                              unsafe_allow_html=True)
            c2.markdown(stat("Avg Place", f"{avg:.2f}", value_color=avg_color, label_tip=avg_tip),  unsafe_allow_html=True)
            c3.markdown(stat("1st",       f"{wins} ({wins/total*100:.0f}%)"),                       unsafe_allow_html=True)
            c4.markdown(stat("Top 4",     f"{top4} ({top4/total*100:.0f}%)"),                       unsafe_allow_html=True)
            c5.markdown(stat("CR",        f"{current_mmr:,}"),                                      unsafe_allow_html=True)

            max_dd = 0
            peak_so_far = games[0]["mmr_after"]
            peak_so_far_game = games[0]
            dd_peak_game = games[0]
            dd_trough_game = games[0]
            for g in games:
                if g["mmr_after"] > peak_so_far:
                    peak_so_far = g["mmr_after"]
                    peak_so_far_game = g
                dd = peak_so_far - g["mmr_after"]
                if dd > max_dd:
                    max_dd = dd
                    dd_peak_game = peak_so_far_game
                    dd_trough_game = g

            peak_time_raw = max(games, key=lambda g: g["mmr_after"])["time"]
            try:
                peak_time_dt  = datetime.fromisoformat(peak_time_raw)
                peak_time_str = peak_time_dt.strftime("%b %-d, %Y")
            except Exception:
                peak_time_str = peak_time_raw[:10]
            _wallii_url = f"https://www.wallii.gg/stats/{sp_player}?region={sp_region.lower()}&mode=solo&view=all"
            c1.markdown(
                f"<div style='margin-top:0.45rem;font-size:0.85rem;color:#777;'>"
                f"Live updates: <a href='{html.escape(_wallii_url)}' target='_blank' style='color:#9ab8d8;text-decoration:none;' "
                f"onmouseover=\"this.style.textDecoration='underline'\" onmouseout=\"this.style.textDecoration='none'\">wallii.gg</a></div>",
                unsafe_allow_html=True
            )
            c5.markdown(
                "<div title='Peak date: " + peak_time_str + "' style='margin-top:0.45rem;color:#777;font-size:0.85rem;'>Peak: "
                "<span style='color:#aaa;font-weight:600;'>" + f"{peak_mmr:,}" + "</span> "
                "<span style='color:#666;'>(" + f"{diff_to_cr:+,}" + ")</span></div>",
                unsafe_allow_html=True
            )
            dd_tip = (
                f"{dd_peak_game['mmr_after']:,} → {dd_trough_game['mmr_after']:,} "
                f"({dd_peak_game['time'][:10]} - {dd_trough_game['time'][:10]})"
            )
            c5.markdown(
                "<div title='" + dd_tip + "' style='margin-top:0.2rem;color:#777;font-size:0.85rem;cursor:help;'>Max MMR drop: "
                "<span style='color:#c07070;font-weight:600;'>-" + f"{max_dd:,}" + "</span></div>",
                unsafe_allow_html=True
            )

            _dist_mode = st.radio("Distribution period", ["All time", "Last 30 days", "Last 7 days"], horizontal=True, label_visibility="collapsed", key="dist_mode")
            if _dist_mode in ("Last 7 days", "Last 30 days"):
                _days = 7 if _dist_mode == "Last 7 days" else 30
                _cutoff = datetime.now(timezone.utc) - timedelta(days=_days)
                _chart_games = [g for g in games if datetime.fromisoformat(g["time"].replace("Z", "+00:00")) >= _cutoff]
            else:
                _chart_games = games
            if _chart_games:
                st.pyplot(make_chart(_chart_games))
            else:
                st.caption("No games in the last 7 days.")

            placements  = [round(g["placement"]) for g in games]
            _tilt_diffs = []
            for i, p in enumerate(placements):
                if p >= 7:
                    before = placements[max(0, i-50):i]
                    after  = placements[i+1:i+4]
                    if len(before) >= 10 and len(after) >= 1:
                        _tilt_diffs.append(sum(after)/len(after) - sum(before)/len(before))

            longest_streak, streak = 0, 0
            for g in games:
                streak = streak + 1 if round(g["placement"]) == 1 else 0
                longest_streak = max(longest_streak, streak)

            longest_roach, roach = 0, 0
            for g in games:
                roach = roach + 1 if round(g["placement"]) <= 4 else 0
                longest_roach = max(longest_roach, roach)

            roach_html = (
                f"<span style='float:right;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Roach streak"
                f"<span style='color:#aaa;font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{longest_roach}</span>"
                f"<span title='Longest consecutive streak of Top 4 place finishes.' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                f"</span>"
            )

            st.markdown(
                f"<div style='margin:0.3rem 0 0.8rem;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Hot streak"
                f"<span style='color:#d4a843;font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{longest_streak}</span>"
                f"<span title='Longest consecutive streak of 1st placement.' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                f"{roach_html}"
                f"</div>",
                unsafe_allow_html=True
            )

            # ── Form (last 50) ────────────────────────────────────────
            if total < 60:
                form_html = (
                    "<span style='float:right;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Form (last 50)"
                    "<span style='color:#333;font-size:0.8rem;margin-left:0.8rem;font-weight:400;text-transform:none;letter-spacing:0;'>- requires 60+ games</span>"
                    "</span>"
                )
            else:
                recent_games = games[-50:]
                recent_avg   = sum(g["placement"] for g in recent_games) / len(recent_games)
                form_diff    = recent_avg - avg
                form_color   = delta_color(form_diff)
                form_sign = "+" if form_diff >= 0 else ""
                form_tip  = f"Avg last 50 games: {recent_avg:.2f} vs overall: {avg:.2f}"
                form_html = (
                    f"<span style='float:right;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Form (last 50)"
                    f"<span style='color:{form_color};font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{recent_avg:.2f}</span>"
                    f"<span style='color:{form_color};font-size:0.8rem;margin-left:0.4rem;'>({form_sign}{form_diff:.2f})</span>"
                    f"<span title='{form_tip}' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                    f"</span>"
                )

            # ── Tilt factor ───────────────────────────────────────────────
            tilt_factor_val = None
            if total < 30:
                st.markdown(
                    "<div style='margin:0.3rem 0 0.8rem;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>"
                    "Tilt factor"
                    "<span style='color:#333;font-size:0.8rem;margin-left:0.8rem;font-weight:400;text-transform:none;letter-spacing:0;'>- requires 30 games</span>"
                    f"{form_html}"
                    "</div>",
                    unsafe_allow_html=True
                )
            elif len(_tilt_diffs) >= 3 and avg > 0:
                factor          = 1 + (sum(_tilt_diffs) / len(_tilt_diffs) / avg) * 2
                tilt_factor_val = float(factor)

                tilt_color = (
                    "#8c3a2a" if factor >= 1.15
                    else "#c47a75" if factor >= 1.06
                    else "#aaa"   if factor >= 1.00
                    else "#7ab87a" if factor >= 0.90
                    else "#4a8c5c"
                )
                _mean_diff    = sum(_tilt_diffs) / len(_tilt_diffs)
                tooltip       = f"How much better or worse a player performs following a bad result, where 1.00 means no change. Mean diff: {_mean_diff:+.2f}. (Lower = better)"
                trigger_count = sum(1 for p in placements if p >= 7)
                asterisk      = "*" if trigger_count < 40 else ""
                asterisk_tip  = f" title='Low sample size: only {trigger_count} games with placement 7-8'" if trigger_count < 40 else ""

                st.markdown(
                    f"<div style='margin:0.3rem 0 0.8rem;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>"
                    f"Tilt factor"
                    f"<span style='color:{tilt_color};font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{factor:.2f}</span>"
                    f"<span{asterisk_tip} style='color:{tilt_color};font-size:0.8rem;cursor:help;'>{asterisk}</span>"
                    f"<span title='{tooltip}' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                    f"{form_html}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            if total >= 30:
                u_color = (
                    "#b388e8" if u_score_val >= 0.2
                    else "#aaa"  if u_score_val >= -0.1
                    else "#5b8fd4"
                )
                u_label = f"{u_score_val:+.2f}"
                u_tip = "Measures play style based on placement distribution. Aggressive (positive) = U-shaped - more 1st and 7th/8th relative to 2nd-4th and 5th-6th. Defensive (negative) = flatter - more consistent mid-range finishes. See Info/Explanations for details."
            else:
                u_color = "#777"
                u_label = "—"
                u_tip = "Requires 30 games to show aggression."

            # ── Form Rating (omvänd interpolation: placement → MMR) ───────
            _form_rating_html = ""
            if total >= 60 and bx is not None and len(bx) >= 2:
                try:
                    _by_rev = by[::-1]
                    _bx_rev = bx[::-1]
                    _form_mmr_int = int(round(float(np.interp(recent_avg, _by_rev, _bx_rev))))
                    _fmmr_tip = f"The MMR that corresponds to a {recent_avg:.2f} avg placement on the regression curve."
                    _d = _form_mmr_int - current_mmr
                    _fmmr_color = (
                        "#4a8c5c" if _d >=  2000 else
                        "#6aab6a" if _d >=   750 else
                        "#7ab87a" if _d >=     0 else
                        "#d4a843" if _d >=  -750 else
                        "#c47a75" if _d >= -2000 else
                        "#8c3a2a"
                    )
                    _fmmr_diff  = _d
                    _fmmr_sign  = "+" if _fmmr_diff >= 0 else ""
                    _form_rating_html = (
                        f"<span style='float:right;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Form Rating"
                        f"<span style='color:{_fmmr_color};font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{_form_mmr_int:,}</span>"
                        f"<span style='color:{_fmmr_color};font-size:0.8rem;margin-left:0.4rem;'>({_fmmr_sign}{_fmmr_diff:,})</span>"
                        f"<span title='{_fmmr_tip}' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                        f"</span>"
                    )
                except Exception:
                    pass

            st.markdown(
                f"<div style='margin:0.3rem 0 0.8rem;color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>"
                f"Aggression"
                f"<span style='color:{u_color};font-size:1.0rem;font-weight:600;margin-left:0.8rem;'>{u_label}</span>"
                f"<span title='{u_tip}' style='color:#444;font-size:0.8rem;margin-left:0.5rem;cursor:help;'>?</span>"
                f"{_form_rating_html}"
                f"</div>",
                unsafe_allow_html=True
            )

            first_10k_date, _mmr_milestones = _mmr_milestones_after_tracking_start(games, current_profile_season)
            if ENABLE_SESSION_TOPLISTS and total >= 50:
                first_pct = wins / total * 100 if total else 0.0
                top4_pct  = top4 / total * 100 if total else 0.0
                lb_upsert_player(
                    sp_region,
                    sp_player,
                    {
                        "season":       current_profile_season,
                        "games":        int(total),
                        "hot_streak":   int(longest_streak),
                        "roach_streak": int(longest_roach),
                        "first_pct":    float(first_pct),
                        "top4_pct":     float(top4_pct),
                        "tilt_factor":  float(tilt_factor_val) if tilt_factor_val is not None else None,
                        "avg_place":    float(avg),
                        "form_diff":    float(form_diff) if total >= 60 else None,
                        "form_rating":  _form_mmr_int if total >= 60 and _form_rating_html else None,
                        "max_drawdown":   int(max_dd),
                        "dd_detail":      dd_tip,
                        "first_10k_date": first_10k_date,
                        "cr":             int(current_mmr),
                        "peak_mmr":       int(peak_mmr),
                        "u_score":        float(u_score_val),
                        "bot2_count":     int(norm[7] + norm[8]),
                        "mmr_milestones": json.dumps(_mmr_milestones),
                        "matchup_scaling": compute_matchup_scaling(games),
                        "updated_at":   _utc_now_iso_z(),
                    }
                )
            elif ENABLE_SESSION_TOPLISTS and total > 0:
                lb_upsert_player(
                    sp_region,
                    sp_player,
                    {
                        "season":         current_profile_season,
                        "games":          int(total),
                        "hot_streak":     int(longest_streak),
                        "roach_streak":   int(longest_roach),
                        "max_drawdown":   int(max_dd),
                        "dd_detail":      dd_tip,
                        "first_10k_date": first_10k_date,
                        "cr":             int(current_mmr),
                        "peak_mmr":       int(peak_mmr),
                        "mmr_milestones": json.dumps(_mmr_milestones),
                        "updated_at":     _utc_now_iso_z(),
                    }
                )
            elif ENABLE_SESSION_TOPLISTS and first_10k_date:
                lb_upsert_player(
                    sp_region,
                    sp_player,
                    {
                        "season":         current_profile_season,
                        "games":          int(total),
                        "first_10k_date": first_10k_date,
                        "cr":             int(current_mmr),
                        "peak_mmr":       int(peak_mmr),
                        "mmr_milestones": json.dumps(_mmr_milestones),
                        "updated_at":     _utc_now_iso_z(),
                    }
                )

            if DEBUG:
                with st.expander("Supabase upsert debug", expanded=False):
                    st.caption(f"TOPLIST_BACKEND={TOPLIST_BACKEND} | SUPABASE_ENABLED={SUPABASE_ENABLED}")
                    if "sb_upsert_status" in st.session_state:
                        code, txt = st.session_state["sb_upsert_status"]
                        st.caption(f"Upsert: {code} | {txt}")

            st.markdown(
                "<p style='color:#888;font-size:0.8rem;text-align:left;margin:0.6rem 0 0.8rem;'>"
                "See the <strong>Info &amp; Explanations</strong> tab for details."
                "</p>",
                unsafe_allow_html=True
            )
            # ── Head-to-Head ──────────────────────────────────────────────
            st.markdown("<hr style='border-color:#1e1e1e;margin:0.8rem 0;'>", unsafe_allow_html=True)
            st.markdown("<p style='color:#8a8a8a;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;'>Head-to-Head comparison ⚔️</p>", unsafe_allow_html=True)
            with st.form("h2h_form"):
                _h2h_cols = st.columns([3, 1, 1])
                _h2h_name = _h2h_cols[0].text_input("H2H player", placeholder="Compare stats with player…", label_visibility="collapsed", key="h2h_name_input")
                _h2h_region = _h2h_cols[1].selectbox("H2H region", VALID_REGIONS, key="h2h_region_input", label_visibility="collapsed")
                _h2h_submitted = _h2h_cols[2].form_submit_button("Compare", width="stretch")
            if _h2h_submitted and _h2h_name.strip():
                import time as _time
                _h2h_last  = st.session_state.get("h2h_last_time", 0)
                _h2h_count = st.session_state.get("h2h_count", 0)
                if _h2h_count >= 2 and _time.time() - _h2h_last < 15:
                    st.warning("Please wait a moment before comparing again.")
                else:
                    st.session_state["h2h_last_time"] = _time.time()
                    st.session_state["h2h_count"]     = _h2h_count + 1
                    with st.spinner(f"Fetching {_h2h_name.strip()}…"):
                        try:
                            _h2h_result = fetch_and_calculate(_h2h_name.strip().lower(), _h2h_region)
                            st.session_state["h2h_games"]  = _h2h_result[0]
                            st.session_state["h2h_label"]  = _h2h_name.strip()
                            st.session_state["h2h_region"] = _h2h_region
                            st.session_state.pop("h2h_error", None)
                        except Exception as _e:
                            st.session_state["h2h_games"] = None
                            st.session_state["h2h_error"] = str(_e)

            if st.session_state.get("h2h_error"):
                st.error(st.session_state["h2h_error"])
            elif st.session_state.get("h2h_games"):
                _h2h_games = st.session_state["h2h_games"]
                _h2h_label = st.session_state.get("h2h_label", "Opponent")
                _s1 = _compute_player_stats(games)
                _s2 = _compute_player_stats(_h2h_games)
                _n1 = sp_player.title()
                _n2 = _h2h_label.title()
                if _n1.lower() == _n2.lower():
                    _n1 = f"{_n1} ({sp_region.upper()})"
                    _n2 = f"{_n2} ({st.session_state.get('h2h_region', '?').upper()})"

                _WIN  = "color: #7ab87a"
                _LOSE = "color: #c47a75"

                def _fmt_diff(s1_val, s2_val, higher_is_better=True, fmt=None, template="{name} has {val} more"):
                    if s1_val is None or s2_val is None:
                        return "—"
                    d = s1_val - s2_val
                    if abs(d) < 0.01:
                        return "Similar"
                    better = _n1 if (d > 0) == higher_is_better else _n2
                    val    = fmt(abs(d)) if fmt else f"{abs(d):.2f}"
                    return template.format(name=better, val=val)

                def _winner(s1_val, s2_val, higher_is_better=True):
                    if s1_val is None or s2_val is None:
                        return None
                    d = s1_val - s2_val
                    if abs(d) < 0.01:
                        return None
                    return _n1 if (d > 0) == higher_is_better else _n2

                _stat_defs = [
                    ("Games",          str(_s1["total"]),                                                        str(_s2["total"]),                                                        _fmt_diff(_s1["total"],        _s2["total"],        higher_is_better=True,  fmt=lambda x: str(int(x)),   template="{name} has played {val} more games"),                  _winner(_s1["total"],        _s2["total"],        higher_is_better=True)),
                    ("Avg Placement",  f"{_s1['avg']:.2f}",                                                     f"{_s2['avg']:.2f}",                                                     _fmt_diff(round(_s1["avg"],2),round(_s2["avg"],2),higher_is_better=False, fmt=lambda x: f"{x:.2f}",    template="{name} has a {val} lower average placement"),            _winner(_s1["avg"],          _s2["avg"],          higher_is_better=False)),
                    ("1st %",          f"{_s1['first_pct']:.1f}%",                                              f"{_s2['first_pct']:.1f}%",                                              _fmt_diff(_s1["first_pct"],    _s2["first_pct"],    higher_is_better=True,  fmt=lambda x: f"{x:.1f}%",   template="{name} has a {val} higher 1st place rate"),                    _winner(_s1["first_pct"],    _s2["first_pct"],    higher_is_better=True)),
                    ("Top 4 %",        f"{_s1['top4_pct']:.1f}%",                                               f"{_s2['top4_pct']:.1f}%",                                               _fmt_diff(_s1["top4_pct"],     _s2["top4_pct"],     higher_is_better=True,  fmt=lambda x: f"{x:.1f}%",   template="{name} has a {val} higher top 4 rate"),                _winner(_s1["top4_pct"],     _s2["top4_pct"],     higher_is_better=True)),
                    ("Current MMR",    f"{_s1['current_mmr']:,}",                                                f"{_s2['current_mmr']:,}",                                                _fmt_diff(_s1["current_mmr"],  _s2["current_mmr"],  higher_is_better=True,  fmt=lambda x: f"{int(x):,}", template="{name} has {val} higher Current MMR"),                 _winner(_s1["current_mmr"],  _s2["current_mmr"],  higher_is_better=True)),
                    ("Peak MMR",       f"{_s1['peak_mmr']:,}",                                                   f"{_s2['peak_mmr']:,}",                                                   _fmt_diff(_s1["peak_mmr"],     _s2["peak_mmr"],     higher_is_better=True,  fmt=lambda x: f"{int(x):,}", template="{name} has {val} higher Peak MMR"),                    _winner(_s1["peak_mmr"],     _s2["peak_mmr"],     higher_is_better=True)),
                    ("Max MMR Drop",   f"-{_s1['max_drawdown']:,}",                                              f"-{_s2['max_drawdown']:,}",                                              _fmt_diff(_s1["max_drawdown"], _s2["max_drawdown"], higher_is_better=True,  fmt=lambda x: f"{int(x):,}", template="{name} has a {val} larger Max MMR Drop"),                _winner(_s1["max_drawdown"], _s2["max_drawdown"], higher_is_better=True)),
                    ("Hot Streak",     str(_s1["hot_streak"]),                                                   str(_s2["hot_streak"]),                                                   _fmt_diff(_s1["hot_streak"],   _s2["hot_streak"],   higher_is_better=True,  fmt=lambda x: str(int(x)),   template="{name} has a {val} game longer streak of 1st places"),  _winner(_s1["hot_streak"],   _s2["hot_streak"],   higher_is_better=True)),
                    ("Roach Streak",   str(_s1["roach_streak"]),                                                 str(_s2["roach_streak"]),                                                 _fmt_diff(_s1["roach_streak"], _s2["roach_streak"], higher_is_better=True,  fmt=lambda x: str(int(x)),   template="{name} has a {val} game longer top 4 streak"),              _winner(_s1["roach_streak"], _s2["roach_streak"], higher_is_better=True)),
                    ("Form (last 50)", f"{_s1['form_diff']:+.2f}" if _s1["form_diff"] is not None else "—",     f"{_s2['form_diff']:+.2f}" if _s2["form_diff"] is not None else "—",     _fmt_diff(_s1["form_diff"],    _s2["form_diff"],    higher_is_better=False, fmt=lambda x: f"{x:.2f}",    template="{name} has a {val} better current form"),                _winner(_s1["form_diff"],    _s2["form_diff"],    higher_is_better=False)),
                    ("Tilt Factor",    f"{_s1['tilt_factor']:.2f}" if _s1["tilt_factor"] is not None else "—", f"{_s2['tilt_factor']:.2f}" if _s2["tilt_factor"] is not None else "—", _fmt_diff(_s1["tilt_factor"],  _s2["tilt_factor"],  higher_is_better=False, fmt=lambda x: f"{x:.2f}",    template="{name} has a {val} lower tilt factor"),                _winner(_s1["tilt_factor"],  _s2["tilt_factor"],  higher_is_better=False)),
                    ("Aggression",     f"{_s1['u_score']:+.2f}",                                                f"{_s2['u_score']:+.2f}",                                                (f"{_n1} has a {'slightly ' if abs(_s1['u_score'] - _s2['u_score']) < 0.3 else ''}more aggressive style") if _s1["u_score"] > _s2["u_score"] else (f"{_n2} has a {'slightly ' if abs(_s1['u_score'] - _s2['u_score']) < 0.3 else ''}more aggressive style" if _s2["u_score"] > _s1["u_score"] else "Similar style"), _winner(_s1["u_score"], _s2["u_score"], higher_is_better=True)),
                    ("Farmer Factor",  f"{_s1['farmer_factor']:+.2f}" if _s1["farmer_factor"] is not None else "N/A (need 300+ games at 10k+)", f"{_s2['farmer_factor']:+.2f}" if _s2["farmer_factor"] is not None else "N/A (need 300+ games at 10k+)", _fmt_diff(_s1["farmer_factor"], _s2["farmer_factor"], higher_is_better=True, fmt=lambda x: f"{x:.2f}", template="{name} performs relatively better vs weaker opponents"), _winner(_s1["farmer_factor"], _s2["farmer_factor"], higher_is_better=True)),
                ]
                _rows    = [{"Stat": s, _n1: v1, _n2: v2, "Comparison": cmp} for s, v1, v2, cmp, _ in _stat_defs]
                _winners = [w for *_, w in _stat_defs]

                _df_h2h = pd.DataFrame(_rows).set_index("Stat")

                def _color_h2h(df):
                    styles = pd.DataFrame("", index=df.index, columns=df.columns)
                    for i, w in enumerate(_winners):
                        if w == _n1:
                            styles.iloc[i, df.columns.get_loc(_n1)] = _WIN
                            styles.iloc[i, df.columns.get_loc(_n2)] = _LOSE
                        elif w == _n2:
                            styles.iloc[i, df.columns.get_loc(_n1)] = _LOSE
                            styles.iloc[i, df.columns.get_loc(_n2)] = _WIN
                    return styles

                st.dataframe(
                    _df_h2h.style.apply(_color_h2h, axis=None),
                    width="stretch",
                    height=495,
                    column_config={
                        _n1:    st.column_config.TextColumn(width="small"),
                        _n2:    st.column_config.TextColumn(width="small"),
                    }
                )

            import altair as alt
            period = st.radio("Period", ["Season", "Week", "Day"], horizontal=True, label_visibility="collapsed", key="rg_period")
            now = datetime.now(timezone.utc)
            cutoff = {"Season": None, "Week": now - timedelta(days=7), "Day": now - timedelta(days=1)}[period]
            filtered = [
                g for g in games
                if cutoff is None or datetime.fromisoformat(g["time"].replace("Z", "+00:00")) >= cutoff
            ]
            if not filtered:
                st.caption("No games in this period.")
            else:
                df_mmr = pd.DataFrame([
                    {"Game": i + 1, "MMR": g["mmr_after"], "Date": g["time"][:10], "Placement": f"{g['placement']:.1f}".rstrip('0').rstrip('.') + f" ({g['mmr_after'] - g['mmr_before']:+d})"}
                    for i, g in enumerate(filtered)
                ])
                peak_idx = df_mmr["MMR"].idxmax()
                df_peak = df_mmr.loc[[peak_idx]]
                df_peak = df_peak.assign(PeakLabel=df_peak["MMR"].map(lambda v: f"Peak: {int(v):,}"))

                milestone_rows = []
                crossed = set()
                for i, g in enumerate(filtered):
                    threshold = (g["mmr_after"] // 1000) * 1000
                    if g["mmr_before"] < threshold <= g["mmr_after"] and threshold not in crossed:
                        crossed.add(threshold)
                        milestone_rows.append({"Game": i + 1, "MMR": g["mmr_after"], "Milestone": f"{threshold:,}", "Date": g["time"][:10]})
                df_milestones = pd.DataFrame(milestone_rows) if milestone_rows else pd.DataFrame(columns=["Game", "MMR", "Milestone", "Date"])

                nearest = alt.selection_point(nearest=True, on="mouseover", encodings=["x"], empty=False)
                line = (
                    alt.Chart(df_mmr)
                    .mark_line(color="#7ab87a")
                    .encode(
                        x=alt.X("Game:Q", title="Game"),
                        y=alt.Y("MMR:Q", title="MMR", scale=alt.Scale(zero=False, padding=20)),
                    )
                )
                hover_points = (
                    alt.Chart(df_mmr)
                    .mark_point(color="#7ab87a", size=60, filled=True)
                    .encode(
                        x="Game:Q",
                        y="MMR:Q",
                        opacity=alt.condition(nearest, alt.value(1), alt.value(0.15)),
                        tooltip=[alt.Tooltip("Game:Q", title="Game"), alt.Tooltip("MMR:Q", title="MMR"), alt.Tooltip("Placement:N", title="Placement"), alt.Tooltip("Date:N", title="Date")],
                    )
                    .add_params(nearest)
                )
                rule = (
                    alt.Chart(df_mmr)
                    .mark_rule(color="#444", strokeWidth=1)
                    .encode(x="Game:Q")
                    .transform_filter(nearest)
                )
                peak_dot = (
                    alt.Chart(df_peak)
                    .mark_point(color="#d4a843", size=80, filled=True)
                    .encode(
                        x="Game:Q",
                        y="MMR:Q",
                        tooltip=[alt.Tooltip("Game:Q", title="Game"), alt.Tooltip("MMR:Q", title="Peak MMR"), alt.Tooltip("Date:N", title="Date")],
                    )
                )
                peak_text = (
                    alt.Chart(df_peak)
                    .mark_text(color="#d4a843", fontSize=11, fontWeight=600, dx=10, dy=-10, align="left")
                    .encode(
                        x="Game:Q",
                        y="MMR:Q",
                        text="PeakLabel:N",
                    )
                )
                milestone_dots = (
                    alt.Chart(df_milestones)
                    .mark_point(color="#aaaaaa", size=40, filled=True, opacity=0.6)
                    .encode(
                        x="Game:Q",
                        y="MMR:Q",
                        tooltip=[alt.Tooltip("Milestone:N", title="First time crossing"), alt.Tooltip("Game:Q", title="Game"), alt.Tooltip("Date:N", title="Date")],
                    )
                )
                chart = line + milestone_dots + peak_dot + peak_text + rule + hover_points
                st.altair_chart(chart.properties(height=250).configure_view(strokeWidth=0), width='stretch')

            if st.button("Compare with leaderboard neighbors", width='stretch'):
                st.session_state["nb_result"] = None
                player_rank = st.session_state.get("sp_rank")

                if player_rank is None:
                    st.session_state["nb_result"] = {"error": "No rank found for this player - they may not be on the leaderboard."}
                else:
                    with st.spinner(f"Finding neighbors around rank {player_rank}..."):
                        names_above, names_below, ranks_above, ranks_below = fetch_neighbor_names(
                            player_rank, sp_region
                        )

                    _neighbor_season = st.session_state.get("sp_season", CURRENT_SEASON)
                    _neighbor_stats = _lb_stats_by_player(_neighbor_season)

                    all_names = names_above + names_below
                    all_ranks = ranks_above + ranks_below

                    if not all_names:
                        st.session_state["nb_result"] = {"error": "No neighbors found nearby."}
                    else:
                        all_pcts = []
                        failed   = []
                        progress = st.progress(0)
                        status   = st.empty()

                        for i, (name, rank) in enumerate(zip(all_names, all_ranks)):
                            status.markdown(
                                "<p style='color:#666;font-size:0.8rem;'>Loading stats for "
                                + name + " (rank " + str(rank) + ")...</p>",
                                unsafe_allow_html=True
                            )
                            try:
                                ng, _rank = _sb_get_cached_games(name, sp_region, season=_neighbor_season)
                                if ng is not None and len(ng) >= 1:
                                    all_pcts.append(norm_to_pct(ng))
                                else:
                                    failed.append(name)
                            except Exception:
                                failed.append(name)
                            progress.progress((i + 1) / len(all_names))

                        progress.empty()
                        status.empty()

                        if not all_pcts:
                            st.session_state["nb_result"] = {"error": "No Supabase neighbor histories could be loaded for comparison."}
                        else:
                            st.session_state["nb_result"] = {
                                "pcts":        all_pcts,
                                "names":       all_names,
                                "ranks":       all_ranks,
                                "failed":      failed,
                                "player_rank": player_rank,
                            }

            nb = st.session_state.get("nb_result")
            if nb:
                if "error" in nb:
                    st.warning(nb["error"])
                elif nb.get("pcts"):
                    if nb["failed"]:
                        st.caption("Could not fetch: " + ", ".join(nb["failed"]))
                    st.markdown(
                        "<p style='color:#666;font-size:0.8rem;margin-bottom:0.5rem;'>Neighbors: "
                        + ", ".join(nb["names"]) + "</p>",
                        unsafe_allow_html=True
                    )
                    st.pyplot(make_neighbor_chart(nb["pcts"], nb["names"], nb["ranks"], sp_player, nb["player_rank"]))

                    player_pct = norm_to_pct(games)
                    avg_pct    = {p: sum(d[p] for d in nb["pcts"]) / len(nb["pcts"]) for p in range(1, 9)}
                    st.caption(summarize_neighbor_differences(player_pct, avg_pct, player_name=sp_player))

                    cells = ""
                    for p in range(1, 9):
                        diff  = player_pct[p] - avg_pct[p]
                        color = diff_pct_color(diff)
                        cells += (
                            f"<div style='text-align:center;flex:1;'>"
                            f"<div style='color:#444;font-size:0.65rem;margin-bottom:0.2rem;'>{p}</div>"
                            f"<div style='color:{color};font-size:1.1rem;font-weight:600;'>{diff:+.1f}%</div>"
                            f"</div>"
                        )

                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;padding:0.3rem 4.5%;border-top:1px solid #1e1e1e;margin-top:0.2rem;'>"
                        f"{cells}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Could not retrieve enough data from neighbors.")

            with st.expander("View as table"):
                rows = [{"Place": p, "Count": norm[p], "%": f"{norm[p]/total*100:.1f}%"} for p in range(1, 9)]
                st.table(rows)

        except Exception as e:
            st.error(str(e))

        # ── Opponent MMR analysis ─────────────────────────────────────────
        st.divider()
        st.markdown("<p style='color:#8a8a8a;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;'>Performance by opponent MMR range</p>", unsafe_allow_html=True)

        _opp_buckets = {}
        for _g in games:
            _gp   = _g.get("placement")
            _gmmr = _g.get("mmr_before")
            _ggain = _g.get("gain")
            if _gp is None or _gmmr is None or _ggain is None:
                continue
            if _gmmr < 10000:
                continue
            _avg_opp = _gmmr - 148.1181435 * (100 - ((_gp - 1) * (200 / 7) + _ggain))
            _bucket  = int(_avg_opp // 1000) * 1000
            # expected placement for this player's MMR at this moment
            _exp = 1 + (7 / 200) * (100 - (_gmmr - _avg_opp) / 148.1181435)
            if _bucket not in _opp_buckets:
                _opp_buckets[_bucket] = {"placements": [], "expected": []}
            _opp_buckets[_bucket]["placements"].append(_gp)
            if _exp is not None:
                _opp_buckets[_bucket]["expected"].append(_exp)

        _MIN_GAMES_BUCKET = 5
        _games_above_10k = sum(1 for g in games if g.get("mmr_before", 0) >= 10000)
        _valid_buckets = {k: v for k, v in _opp_buckets.items() if len(v["placements"]) >= _MIN_GAMES_BUCKET and k >= 7000} if _games_above_10k >= 300 else {}
        if _valid_buckets:
            _opp_rows = ""
            for _bk in sorted(_valid_buckets.keys()):
                _bv   = _valid_buckets[_bk]
                _bavg = sum(_bv["placements"]) / len(_bv["placements"])
                _bexp = sum(_bv["expected"]) / len(_bv["expected"]) if _bv["expected"] else None
                _dev  = (_bavg - _bexp) if _bexp is not None else None
                if _dev is not None:
                    _dev_color = "#81c784" if _dev < -0.15 else "#e57373" if _dev > 0.15 else "#8a8a8a"
                    _dev_str   = f"{_dev:+.2f}"
                    _dev_html  = f"<span style='color:{_dev_color};font-size:0.78rem;min-width:44px;text-align:right;'>{_dev_str}</span>"
                else:
                    _dev_html = ""
                _bar_w     = max(4, int((8 - _bavg) / 7 * 100))
                _bar_color = "#81c784" if _bavg <= 3 else "#fff176" if _bavg <= 4.5 else "#e57373"
                _uncertain = _bk >= 11000
                _ast = "<span title='Data in this interval is unreliable - the placement estimation formula tends to systematically overestimate results at this opponent MMR level, as well as an issue of sample size.' style='color:#666;cursor:help;margin-left:2px;'>*</span>" if _uncertain else ""
                _top_margin = "1rem" if _bk == 11000 else "0"
                _opp_rows += (
                    f"<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.3rem;margin-top:{_top_margin};'>"
                    f"<span style='color:#666;font-size:0.78rem;min-width:90px;'>{_bk:,}–{_bk+1000:,}{_ast}</span>"
                    f"<div style='flex:1;background:#1e1e1e;border-radius:3px;height:8px;'>"
                    f"<div style='width:{_bar_w}%;background:{_bar_color};border-radius:3px;height:8px;'></div></div>"
                    f"<span style='color:#ccc;font-size:0.82rem;min-width:32px;text-align:right;'>{_bavg:.2f}</span>"
                    f"{_dev_html}"
                    f"<span style='color:#444;font-size:0.72rem;min-width:50px;'>({len(_bv['placements'])} games)</span>"
                    f"</div>"
                )
            st.markdown(_opp_rows, unsafe_allow_html=True)
            _legend = "Avg placement per 1k MMR interval. Only games played at 10k+ MMR (early season affects the stats a bit much otherwise)."
            st.caption(_legend)

            # ── Matchup scaling score ─────────────────────────────────────
            _slope = compute_matchup_scaling(games)
            if _slope is not None:
                _ff = -_slope  # flip: positive = good at beating weaker opponents
                _ff_color = "#81c784" if _ff > 0.2 else "#e57373" if _ff < -0.2 else "#8a8a8a"
                if _ff > 0.45:
                    _ff_label = "way better vs weaker opponents"
                elif _ff > 0.2:
                    _ff_label = "performs better vs weaker opponents"
                elif _ff > 0:
                    _ff_label = "slightly better vs weaker opponents"
                elif _ff > -0.2:
                    _ff_label = "slightly better vs stronger opponents"
                elif _ff > -0.45:
                    _ff_label = "performs better vs stronger opponents"
                else:
                    _ff_label = "way better vs stronger opponents"
                st.markdown(
                    f"<div style='margin-top:0.6rem;'>"
                    f"<span style='color:#666;font-size:0.78rem;'>Farmer factor: </span>"
                    f"<span style='color:{_ff_color};font-size:0.9rem;font-weight:700;'>{_ff:+.2f}</span>"
                    f"<span style='color:#444;font-size:0.75rem;margin-left:0.4rem;'>({_ff_label})</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("Not enough data to show opponent MMR breakdown.")

    st.markdown("<div style='height:3rem;'></div>", unsafe_allow_html=True)
