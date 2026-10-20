"""Administrative tools"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests
import streamlit as st

from wa2_config import CURRENT_SEASON, PLAYER_STATS_TABLE, SEASONS
from wa2_rating import binned_weighted_curve


@dataclass(frozen=True)
class AdminServices:
    supabase_headers: dict[str, str]
    supabase_url: str
    cache_bust_toplists: Callable[..., Any]
    post_player_links: Callable[..., Any]
    sb_fetch_all: Callable[..., Any]
    sb_fetch_player_links: Callable[..., Any]
    sb_get_cached_snapshots: Callable[..., Any]
    sb_load_regression: Callable[..., Any]
    sb_save_regression: Callable[..., Any]
    sb_top_n: Callable[..., Any]
    split_player_link_names: Callable[..., Any]
    twitch_get_live_streams: Callable[..., Any]
    yt_fetch_subscribers: Callable[..., Any]
    snapshots_to_games: Callable[..., Any]
    binned_weighted_curve: Callable[..., Any]
    compute_and_upsert: Callable[..., Any]
    compute_matchup_scaling: Callable[..., Any]
    fetch_and_calculate: Callable[..., Any]
    fetch_top_n_for_scan: Callable[..., Any]


def render_admin_panel(services: AdminServices):
    SUPABASE_HEADERS = services.supabase_headers
    SUPABASE_URL = services.supabase_url
    _cache_bust_toplists = services.cache_bust_toplists
    _post_player_links = services.post_player_links
    _sb_fetch_all = services.sb_fetch_all
    _sb_fetch_player_links = services.sb_fetch_player_links
    _sb_get_cached_snapshots = services.sb_get_cached_snapshots
    _sb_load_regression = services.sb_load_regression
    _sb_save_regression = services.sb_save_regression
    _sb_top_n = services.sb_top_n
    _split_player_link_names = services.split_player_link_names
    _twitch_get_live_streams = services.twitch_get_live_streams
    _yt_fetch_subscribers = services.yt_fetch_subscribers
    _snapshots_to_games = services.snapshots_to_games
    binned_weighted_curve = services.binned_weighted_curve
    compute_and_upsert = services.compute_and_upsert
    compute_matchup_scaling = services.compute_matchup_scaling
    fetch_and_calculate = services.fetch_and_calculate
    fetch_top_n_for_scan = services.fetch_top_n_for_scan
    with st.expander("Secret stuff"):
        pwd = st.text_input("Password", type="password", key="admin_pwd")
        if pwd == st.secrets.get("ADMIN_PASSWORD", ""):
            st.caption("Add or update Twitch/YouTube links for players.")
            _COUNTRIES = [("","- None -"),("AF","Afghanistan"),("AL","Albania"),("DZ","Algeria"),("AR","Argentina"),("AM","Armenia"),("AU","Australia"),("AT","Austria"),("AZ","Azerbaijan"),("BE","Belgium"),("BR","Brazil"),("BG","Bulgaria"),("BY","Belarus"),("CA","Canada"),("CL","Chile"),("CN","China"),("CO","Colombia"),("HR","Croatia"),("CZ","Czech Republic"),("DK","Denmark"),("EG","Egypt"),("EE","Estonia"),("FI","Finland"),("FR","France"),("GE","Georgia"),("DE","Germany"),("GR","Greece"),("HK","Hong Kong"),("HU","Hungary"),("IN","India"),("ID","Indonesia"),("IE","Ireland"),("IL","Israel"),("IT","Italy"),("JP","Japan"),("KZ","Kazakhstan"),("KR","South Korea"),("LV","Latvia"),("LT","Lithuania"),("LU","Luxembourg"),("MY","Malaysia"),("MX","Mexico"),("NL","Netherlands"),("NZ","New Zealand"),("NO","Norway"),("PH","Philippines"),("PL","Poland"),("PT","Portugal"),("RO","Romania"),("RU","Russia"),("SA","Saudi Arabia"),("RS","Serbia"),("SG","Singapore"),("SK","Slovakia"),("SI","Slovenia"),("ZA","South Africa"),("ES","Spain"),("SE","Sweden"),("CH","Switzerland"),("TW","Taiwan"),("TH","Thailand"),("TR","Turkey"),("UA","Ukraine"),("GB","United Kingdom"),("US","United States"),("UZ","Uzbekistan"),("VN","Vietnam")]
            _lnk_player  = st.text_input("Player name", key="lnk_player").strip().lower()
            _lnk_aliases = st.text_area("Alias/account names", key="lnk_aliases", height=70, placeholder="optional aliases separated by comma/new line").strip().lower()
            _lnk_twitch  = st.text_input("Twitch URL (leave blank to clear)", value="https://www.twitch.tv/", key="lnk_twitch").strip()
            _lnk_youtube = st.text_input("YouTube URL (leave blank to clear)", key="lnk_youtube").strip()
            _nat_options = [f"{code} - {name}" if code else f"- {name} -" for code, name in _COUNTRIES]
            _nat_sel     = st.selectbox("Nationality", _nat_options, key="lnk_nat_sel")
            _lnk_nat     = _nat_sel.split(" - ")[0].strip() if " - " in _nat_sel and not _nat_sel.startswith("- ") else ""
            if st.button("Save link", key="lnk_save", width='stretch'):
                _lnk_primary = _lnk_player.strip().lower()
                _lnk_players = _split_player_link_names("\n".join([_lnk_primary, _lnk_aliases]))
                if _lnk_primary and _lnk_players:
                    try:
                        _payload = [
                            {k: v for k, v in {
                                "player_name": player_name,
                                "display_name": _lnk_primary,
                                "twitch_url": (_lnk_twitch if _lnk_twitch not in ("https://www.twitch.tv/", "") else None),
                                "youtube_url": _lnk_youtube or None,
                                "nationality": _lnk_nat or None,
                            }.items() if k == "player_name" or v is not None}
                            for player_name in _lnk_players
                        ]
                        _saved_display_name = _post_player_links(_payload)
                        st.session_state.setdefault("player_link_display_names", {}).update({
                            player_name: _lnk_primary for player_name in _lnk_players
                        })
                        _sb_fetch_player_links.clear()
                        _twitch_get_live_streams.clear()
                        _yt_fetch_subscribers.clear()
                        st.success(f"Saved links for {', '.join(_lnk_players)}.")
                        if not _saved_display_name:
                            st.warning("Saved aliases, but Supabase is missing the optional display_name column. Add it for Twitch/YouTube lists to always prefer the primary name.")
                    except Exception as _le:
                        st.error(str(_le))
                else:
                    st.warning("Enter a player name.")

            st.divider()
            st.caption("Fetches all players in the leaderboard and recalculates their stats.")
            if st.button("Refresh all players", width='stretch'):
                try:
                    url = f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}"
                    resp = requests.get(url, headers=SUPABASE_HEADERS, params={"select": "player,region", "limit": "500"}, timeout=10)
                    resp.raise_for_status()
                    all_players = resp.json()
                except Exception as e:
                    st.error(f"Could not fetch player list: {e}")
                    all_players = []
                if all_players:
                    bar = st.progress(0, text="Starting...")
                    for i, row in enumerate(all_players):
                        name, region = row["player"], row["region"]
                        bar.progress((i + 1) / len(all_players), text=f"({i+1}/{len(all_players)}) {name} [{region}]")
                        try:
                            games_r, _, _ = fetch_and_calculate(name, region)
                            compute_and_upsert(name, region, games_r)
                        except Exception:
                            pass
                    bar.progress(1.0, text="Done!")
                    _cache_bust_toplists()
                    st.success(f"Done! {len(all_players)} players refreshed.")
                    st.rerun(scope="app")

            st.divider()
            st.caption("Fetches top N players per region from wallii.gg leaderboard and upserts their stats.")
            _scan_regions = st.multiselect("Regions to scan", ["EU", "NA", "AP", "CN"], default=["EU", "NA", "AP", "CN"], key="scan_regions")
            _scan_limit   = st.number_input("Players per region", min_value=10, max_value=500, value=100, step=10, key="scan_limit")
            if st.button("Scan leaderboard top N", width='stretch'):
                _scan_ok, _scan_err = 0, 0
                for _scan_rgn in _scan_regions:
                    try:
                        _scan_names = fetch_top_n_for_scan(_scan_rgn, int(_scan_limit))
                    except Exception as e:
                        st.warning(f"{_scan_rgn}: failed to fetch leaderboard - {e}")
                        continue
                    _bar = st.progress(0, text=f"{_scan_rgn}: starting...")
                    for _si, _sname in enumerate(_scan_names):
                        _bar.progress((_si + 1) / len(_scan_names), text=f"{_scan_rgn} ({_si+1}/{len(_scan_names)}) {_sname}")
                        try:
                            fetch_and_calculate.clear()
                            _sgames, _, _ = fetch_and_calculate(_sname, _scan_rgn)
                            compute_and_upsert(_sname, _scan_rgn, _sgames)
                            _scan_ok += 1
                        except Exception:
                            _scan_err += 1
                    _bar.progress(1.0, text=f"{_scan_rgn}: done!")
                _cache_bust_toplists()
                st.success(f"Scan complete - {_scan_ok} ok, {_scan_err} errors.")
                st.rerun(scope="app")

            st.divider()
            st.caption("Rebuild avg-placement regression curves from player_stats data (last 7 days, min 100 games).")
            _cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            _fresh_counts = {}
            for _reg in ["EU", "NA", "AP", "CN"]:
                try:
                    _rc = requests.get(
                        f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}",
                        headers=SUPABASE_HEADERS,
                        params={"region": f"eq.{_reg}", "updated_at": f"gte.{_cutoff_7d}", "games": "gte.100", "select": "player", "limit": "10000"},
                        timeout=10,
                    )
                    _fresh_counts[_reg] = len(_rc.json())
                except Exception:
                    _fresh_counts[_reg] = "?"
            _total_fresh = sum(v for v in _fresh_counts.values() if isinstance(v, int))
            st.caption("Fresh players (7d): " + "  |  ".join(f"{r}: {n}" for r, n in _fresh_counts.items()) + f"  |  **Total: {_total_fresh}**")
            if st.button("Rebuild regression curve", width='stretch'):
                try:
                    _all_rows = []
                    for _reg in ["EU", "NA", "AP", "CN"]:
                        _rd = requests.get(
                            f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}",
                            headers=SUPABASE_HEADERS,
                            params={"region": f"eq.{_reg}", "updated_at": f"gte.{_cutoff_7d}", "games": "gte.100", "select": "cr,avg_place,games", "limit": "10000"},
                            timeout=15,
                        )
                        _rd.raise_for_status()
                        _all_rows.extend(_rd.json())
                    if len(_all_rows) < 5:
                        st.warning(f"Not enough data ({len(_all_rows)} players total)")
                    else:
                        _df_all = pd.DataFrame(_all_rows).rename(columns={"cr": "current_mmr"})
                        _rbx, _rby, _ = binned_weighted_curve(_df_all, x_col="current_mmr", y_col="avg_place", w_col="games", bin_size=500, mode="wmean", min_games=0)
                        if len(_rbx) < 2:
                            st.warning("Regression failed (not enough bins)")
                        else:
                            _poly = np.polyfit(_rbx, _rby, deg=2)
                            _smooth_x = np.linspace(_rbx.min(), _rbx.max() + 5000, 100)
                            _smooth_y = np.clip(np.polyval(_poly, _smooth_x), 1.0, 8.0)
                            _sb_save_regression("ALL", _smooth_x, _smooth_y, len(_all_rows))
                            st.success(f"Curve updated ({len(_all_rows)} players, {len(_rbx)} bins)")
                except Exception as _re:
                    st.error(str(_re))

            st.divider()
            st.caption("Recalculates Form Rating for all players using existing player_stats data and the current regression curve. No wallii.gg calls.")
            if st.button("Rebuild Form Ratings", width='stretch'):
                try:
                    _bx_fr, _by_fr = _sb_load_regression("ALL")
                    if _bx_fr is None or len(_bx_fr) < 2:
                        st.error("No regression curve found. Rebuild regression curve first.")
                    else:
                        _ps_resp = requests.get(
                            f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}",
                            headers=SUPABASE_HEADERS,
                            params={"select": "player,region,avg_place,form_diff", "limit": "1000"},
                            timeout=15,
                        )
                        _ps_resp.raise_for_status()
                        _ps_rows = [r for r in _ps_resp.json() if r.get("avg_place") is not None and r.get("form_diff") is not None]
                        _ok, _fail = 0, 0
                        for _pr in _ps_rows:
                            try:
                                _ra = _pr["avg_place"] + _pr["form_diff"]
                                _fr = int(round(float(np.interp(_ra, _by_fr[::-1], _bx_fr[::-1]))))
                                requests.post(
                                    f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}?on_conflict=player,region",
                                    headers={**SUPABASE_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"},
                                    json={"player": _pr["player"], "region": _pr["region"], "form_rating": _fr},
                                    timeout=10,
                                ).raise_for_status()
                                _ok += 1
                            except Exception:
                                _fail += 1
                        st.success(f"Updated {_ok} players. Failed: {_fail}.")
                        _sb_fetch_all.clear()
                        _sb_top_n.clear()
                except Exception as _fre:
                    st.error(str(_fre))

            st.divider()
            st.caption("Recalculates Matchup Scaling for all players using existing player_cache snapshots. No wallii.gg calls.")
            if st.button("Rebuild Matchup Scaling", width='stretch'):
                try:
                    _pc_resp = requests.get(
                        f"{SUPABASE_URL}/rest/v1/player_cache",
                        headers=SUPABASE_HEADERS,
                        params={"select": "player_name,region", "limit": "2000"},
                        timeout=30,
                    )
                    _pc_resp.raise_for_status()
                    _pc_rows = _pc_resp.json()
                    _ok, _skip, _fail = 0, 0, 0
                    _bar = st.progress(0.0)
                    for _pi, _pc in enumerate(_pc_rows):
                        _bar.progress((_pi + 1) / max(len(_pc_rows), 1), text=f"{_pc.get('player_name','?')}...")
                        try:
                            _snaps, _, _ = _sb_get_cached_snapshots(_pc["player_name"], _pc["region"])
                            if not _snaps or len(_snaps) < 2:
                                # fetch directly without cache age check
                                _sr = requests.get(
                                    f"{SUPABASE_URL}/rest/v1/snapshots",
                                    headers=SUPABASE_HEADERS,
                                    params={"player_name": f"eq.{_pc['player_name'].lower()}", "region": f"eq.{_pc['region'].upper()}", "game_mode": "eq.0", "snapshot_time": f"gte.{SEASONS[CURRENT_SEASON]['start']}", "order": "snapshot_time.asc", "limit": "2000", "select": "snapshot_time,rating"},
                                    timeout=15,
                                )
                                _sr.raise_for_status()
                                _snaps = _sr.json()
                            if not _snaps or len(_snaps) < 2:
                                _skip += 1
                                continue
                            _ms_games = _snapshots_to_games(_snaps)
                            _ms_val = compute_matchup_scaling(_ms_games)
                            if _ms_val is None:
                                _skip += 1
                                continue
                            requests.patch(
                                f"{SUPABASE_URL}/rest/v1/{PLAYER_STATS_TABLE}?player=eq.{_pc['player_name'].lower()}&region=eq.{_pc['region'].upper()}",
                                headers={**SUPABASE_HEADERS, "Content-Type": "application/json", "Prefer": "return=minimal"},
                                json={"matchup_scaling": _ms_val},
                                timeout=10,
                            ).raise_for_status()
                            _ok += 1
                        except Exception:
                            _fail += 1
                    _bar.progress(1.0, text="Done!")
                    _cache_bust_toplists()
                    st.success(f"Updated {_ok} players. Skipped: {_skip}. Failed: {_fail}.")
                except Exception as _mse:
                    st.error(str(_mse))

        elif pwd:
            st.caption("Wrong password.")
