"""Developer-facing regression and CSV curve views."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from wa2_charts import style_dark_axes
from wa2_config import DEFAULT_CSV_NAME
from wa2_rating import interp_with_extrap


@dataclass(frozen=True)
class TestingViewServices:
    app_dir: Path
    cached_binned_curve: Callable[..., Any]
    get_csv_for_region: Callable[..., Any]
    read_curve_csv: Callable[..., Any]
    sb_fetch_all: Callable[..., Any]
    sb_load_regression: Callable[..., Any]


def render_testing_views(tabs, services: TestingViewServices):
    _cached_binned_curve = services.cached_binned_curve
    get_csv_for_region = services.get_csv_for_region
    _read_curve_csv = services.read_curve_csv
    _sb_fetch_all = services.sb_fetch_all
    _sb_load_regression = services.sb_load_regression
    __file__ = str(services.app_dir / "wa2_app.py")
    if tabs[3].open:
        with tabs[3]:
            st.info("Ignore this (if you want), just backend stuff for debug/testign. Used for estimating expected average placement at a given MMR based on currently uploaded CSV curves (regression between MMR and avgPlace) for some of the values.")

            st.markdown("<p style='color:#8a8a8a;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Supabase regression curve (all regions combined)</p>", unsafe_allow_html=True)
            _sb_bx, _sb_by = _sb_load_regression("ALL")
            if _sb_bx is None or len(_sb_bx) < 2:
                st.caption("No Supabase regression found. Run 'Rebuild regression curve' in admin.")
            else:
                _sb_q_mmr = st.number_input(
                    "MMR",
                    min_value=float(np.min(_sb_bx)) - 5000.0,
                    max_value=float(np.max(_sb_bx)) + 5000.0,
                    value=float(np.percentile(_sb_bx, 75)),
                    step=100.0,
                    label_visibility="collapsed",
                    key="sb_reg_mmr",
                )
                _sb_est = float(np.clip(interp_with_extrap(_sb_q_mmr, _sb_bx, _sb_by), 1.0, 8.0))
                st.markdown(
                    f"<div style='margin-top:0.35rem;margin-bottom:0.8rem;'>"
                    f"<span style='color:#eee;font-size:1.6rem;font-weight:700;'>{_sb_est:.2f}</span>"
                    f"<span style='color:#777;font-size:0.9rem;margin-left:0.6rem;'>at {_sb_q_mmr:,.0f} MMR (all regions)</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                _sb_fig, _sb_ax = plt.subplots(figsize=(10, 5))
                _sb_fig.patch.set_facecolor("#191715")
                _sb_ax.set_facecolor("#191715")
                _sb_ax.plot(_sb_bx, _sb_by, color="#d4a843", linewidth=3)
                _sb_ax.set_xlabel("Current MMR")
                _sb_ax.set_ylabel("Avg Place")
                style_dark_axes(_sb_ax)
                st.pyplot(_sb_fig)
                plt.close(_sb_fig)

            st.divider()

    if tabs[3].open:
        with tabs[3]:
            rr = st.selectbox("Curve region (CSV)", ["EU"], index=0)

            if rr == "Fallback export.csv":
                csv_path = Path(__file__).parent / DEFAULT_CSV_NAME
            else:
                csv_path = get_csv_for_region(rr)

            if not csv_path.exists():
                st.error(
                    f"Can't find CSV: {csv_path}\n\n"
                    f"Put it next to this file. Expected columns: lb_mmr, current_mmr, avg_place, games."
                )
                st.stop()

            _csv_mtime_ns = csv_path.stat().st_mtime_ns
            df = _read_curve_csv(str(csv_path), _csv_mtime_ns)

            required = {"lb_mmr", "current_mmr", "avg_place", "games"}
            missing  = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"CSV is missing columns: {', '.join(missing)}")
                st.stop()

            c1, c2, c3 = st.columns([1.2, 1.0, 1.2])
            with c1:
                x_choice = st.selectbox("MMR source", ["lb_mmr", "current_mmr"], index=0)
            with c2:
                bin_size = st.select_slider("Bin size", options=[250, 500, 750, 1000], value=1000)
            with c3:
                max_games         = int(pd.to_numeric(df["games"], errors="coerce").max() or 0)
                default_min_games = 300 if max_games >= 300 else max_games
                min_games         = st.slider("Min games", min_value=0, max_value=max_games, value=default_min_games, step=50)

            mode = st.selectbox(
                "Curve",
                [
                    ("Weighted 10th percentile", "wquant", 0.10),
                    ("Weighted 25th percentile", "wquant", 0.25),
                    ("Weighted median",          "wquant", 0.50),
                    ("Weighted mean",            "wmean",  None),
                ],
                format_func=lambda t: t[0],
                index=3,
            )
            mode_kind = mode[1]
            q         = mode[2] if mode_kind == "wquant" else 0.5

            bx, by, d = _cached_binned_curve(
                str(csv_path),
                _csv_mtime_ns,
                x_choice,
                int(bin_size),
                mode_kind,
                float(q) if mode_kind == "wquant" else 0.5,
                int(min_games),
            )

            if len(bx) < 2:
                st.warning("Too few bins after filtering. Lower min games or change bin size.")
                st.stop()

            st.markdown(
                "<div style='color:#666;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>"
                f"Estimate Avg Place at {x_choice}  ·  using {csv_path.name}</div>",
                unsafe_allow_html=True
            )

            q_mmr = st.number_input(
                label="MMR",
                min_value=float(np.min(bx)) - 5000.0,
                max_value=float(np.max(bx)) + 5000.0,
                value=float(np.percentile(bx, 75)),
                step=100.0,
                label_visibility="collapsed",
            )

            est = interp_with_extrap(q_mmr, bx, by)
            est = float(np.clip(est, 1.0, 8.0))

            st.markdown(
                f"<div style='margin-top:0.35rem;margin-bottom:0.8rem;'>"
                f"<span style='color:#eee;font-size:1.6rem;font-weight:700;'>{est:.2f}</span>"
                f"<span style='color:#777;font-size:0.9rem;margin-left:0.6rem;'>at {q_mmr:,.0f} {x_choice}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#191715")
            ax.set_facecolor("#191715")
            ax.plot(bx, by, linewidth=3)
            ax.set_xlabel(x_choice)
            ax.set_ylabel("Avg Place")
            style_dark_axes(ax)
            st.pyplot(fig)
            plt.close(fig)

            def _remove_outliers(xs, ys, z_thresh=2.5):
                mask = (np.abs((xs - xs.mean()) / xs.std()) < z_thresh) & (np.abs((ys - ys.mean()) / ys.std()) < z_thresh)
                return xs[mask], ys[mask], int((~mask).sum())

            st.divider()
            st.markdown("<h3 style='text-decoration:none;'>Farmer Factor vs MMR</h3>", unsafe_allow_html=True)
            _corr_mmr_rows = [r for r in _sb_fetch_all() if r.get("matchup_scaling") is not None and r.get("cr") is not None]
            if len(_corr_mmr_rows) < 5:
                st.caption("Not enough data yet.")
            else:
                _ff_mmr_vals = np.array([-r["matchup_scaling"] for r in _corr_mmr_rows])
                _cr_vals = np.array([r["cr"] for r in _corr_mmr_rows])
                _cr_vals, _ff_mmr_vals, _mmr_removed = _remove_outliers(_cr_vals, _ff_mmr_vals)
                _corr_mmr = float(np.corrcoef(_ff_mmr_vals, _cr_vals)[0, 1])
                _fig_m, _ax_m = plt.subplots(figsize=(8, 5))
                _fig_m.patch.set_facecolor("#191715")
                _ax_m.set_facecolor("#191715")
                _ax_m.scatter(_cr_vals, _ff_mmr_vals, color="#9146FF", alpha=0.6, s=30)
                _m_m, _b_m = np.polyfit(_cr_vals, _ff_mmr_vals, 1)
                _xs_line_m = np.linspace(_cr_vals.min(), _cr_vals.max(), 100)
                _ax_m.plot(_xs_line_m, _m_m * _xs_line_m + _b_m, color="#d4a843", linewidth=2)
                _ax_m.set_xlabel("MMR (cr)")
                _ax_m.set_ylabel("Farmer Factor")
                style_dark_axes(_ax_m)
                st.pyplot(_fig_m)
                plt.close(_fig_m)
                _removed_note = f", {_mmr_removed} outliers removed" if _mmr_removed else ""
                st.caption(f"Pearson correlation: **{_corr_mmr:.3f}** ({len(_cr_vals)} players{_removed_note})")

            st.divider()
            st.markdown("<h3 style='text-decoration:none;'>Farmer Factor vs Aggression Score</h3>", unsafe_allow_html=True)
            _corr_rows = [r for r in _sb_fetch_all() if r.get("matchup_scaling") is not None and r.get("u_score") is not None]
            if len(_corr_rows) < 5:
                st.caption("Not enough data yet.")
            else:
                _ff_vals = np.array([-r["matchup_scaling"] for r in _corr_rows])
                _us_vals = np.array([r["u_score"] for r in _corr_rows])
                _us_vals, _ff_vals, _us_removed = _remove_outliers(_us_vals, _ff_vals)
                _corr = float(np.corrcoef(_ff_vals, _us_vals)[0, 1])
                _fig_c, _ax_c = plt.subplots(figsize=(8, 5))
                _fig_c.patch.set_facecolor("#191715")
                _ax_c.set_facecolor("#191715")
                _ax_c.scatter(_us_vals, _ff_vals, color="#9146FF", alpha=0.6, s=30)
                _m, _b = np.polyfit(_us_vals, _ff_vals, 1)
                _xs_line = np.linspace(_us_vals.min(), _us_vals.max(), 100)
                _ax_c.plot(_xs_line, _m * _xs_line + _b, color="#d4a843", linewidth=2)
                _ax_c.set_xlabel("Aggression Score (u_score)")
                _ax_c.set_ylabel("Farmer Factor")
                style_dark_axes(_ax_c)
                st.pyplot(_fig_c)
                plt.close(_fig_c)
                _removed_note = f", {_us_removed} outliers removed" if _us_removed else ""
                st.caption(f"Pearson correlation: **{_corr:.3f}** ({len(_us_vals)} players{_removed_note}")

            st.divider()
            st.markdown("<h3 style='text-decoration:none;'>Aggression Score vs Tilt Factor</h3>", unsafe_allow_html=True)
            _corr_tilt_rows = [r for r in _sb_fetch_all() if r.get("u_score") is not None and r.get("tilt_factor") is not None]
            if len(_corr_tilt_rows) < 5:
                st.caption("Not enough data yet.")
            else:
                _us_tilt_vals = np.array([r["u_score"] for r in _corr_tilt_rows])
                _tf_vals = np.array([r["tilt_factor"] for r in _corr_tilt_rows])
                _us_tilt_vals, _tf_vals, _tilt_removed = _remove_outliers(_us_tilt_vals, _tf_vals)
                _corr_tilt = float(np.corrcoef(_us_tilt_vals, _tf_vals)[0, 1])
                _fig_t, _ax_t = plt.subplots(figsize=(8, 5))
                _fig_t.patch.set_facecolor("#191715")
                _ax_t.set_facecolor("#191715")
                _ax_t.scatter(_us_tilt_vals, _tf_vals, color="#9146FF", alpha=0.6, s=30)
                _m_t, _b_t = np.polyfit(_us_tilt_vals, _tf_vals, 1)
                _xs_line_t = np.linspace(_us_tilt_vals.min(), _us_tilt_vals.max(), 100)
                _ax_t.plot(_xs_line_t, _m_t * _xs_line_t + _b_t, color="#d4a843", linewidth=2)
                _ax_t.set_xlabel("Aggression Score (u_score)")
                _ax_t.set_ylabel("Tilt Factor")
                style_dark_axes(_ax_t)
                st.pyplot(_fig_t)
                plt.close(_fig_t)
                _removed_note = f", {_tilt_removed} outliers removed" if _tilt_removed else ""
                st.caption(f"Pearson correlation: **{_corr_tilt:.3f}** ({len(_us_tilt_vals)} players{_removed_note})")
