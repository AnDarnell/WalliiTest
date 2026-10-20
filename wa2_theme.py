"""Page configuration, branding, and global CSS styling."""

from pathlib import Path
import base64 as _b64

import streamlit as st


def apply_page_theme():
    # ── Page styling ──────────────────────────────────────────────────────────────

    _app_dir = Path(__file__).resolve().parent
    st.set_page_config(page_title="Placement Statistics", layout="wide", page_icon=str(_app_dir / "nerdbob2.png"))
    st.logo(str(_app_dir / "nerdbob.png"))
    _tavern_bg_b64 = _b64.b64encode((_app_dir / "assets" / "tavern.avif").read_bytes()).decode()

    st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Georgia', serif; }
    .stApp {
        background:
            radial-gradient(ellipse at 50% -12%, rgba(138, 78, 42, 0.22), transparent 58%),
            #211813;
        color: #e7d6ba;
    }

    div[data-testid="stMainBlockContainer"] {
        max-width: 1040px;
        margin-left: auto;
        margin-right: auto;
    }

    .stTextInput input, .stNumberInput input {
        background-color: #35261e !important;
        border: 1px solid #5a4432 !important;
        color: #f2e5cf !important;
        border-radius: 4px !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus { border-color: #d9aa57 !important; }
    .stTextInput label, .stSelectbox label, .stNumberInput label {
        color: #b69b79 !important; font-size: 0.75rem !important;
        text-transform: uppercase !important; letter-spacing: 0.07em !important;
    }
    .st-key-lb_inc_eu,
    .st-key-lb_inc_na,
    .st-key-lb_inc_ap,
    .st-key-lb_inc_cn {
        position: relative !important;
        left: 1.4rem !important;
    }
    .stSelectbox > div > div {
        background-color: #35261e !important;
        border: 1px solid #5a4432 !important;
        color: #f2e5cf !important;
        border-radius: 4px !important;
    }
    .stFormSubmitButton button, .stButton button, .stDownloadButton button {
        background-color: #161616 !important;
        color: #d4a843 !important;
        border: 1px solid #d4a843 !important;
        border-radius: 4px !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    .stFormSubmitButton button:hover, .stButton button:hover, .stDownloadButton button:hover {
        background-color: #d4a843 !important;
        color: #0e0e0e !important;
    }
    .stButton button[kind="primary"] {
        background-color: #d4a843 !important;
        color: #0e0e0e !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #e8bb55 !important;
    }
    .stButton button[kind="primary"] {
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
    }
    .streamlit-expanderHeader {
        background-color: #35261e !important;
        border: 1px solid #5a4432 !important;
        color: #c4a985 !important; font-size: 0.8rem !important;
        border-radius: 4px !important;
    }
    .st-key-updates_expander_wrap [data-testid="stExpander"] summary {
        background-color: #454b25 !important;
        border: 1px solid #a8b746 !important;
        color: #e2e99a !important;
        border-radius: 5px !important;
    }
    .st-key-updates_expander_wrap [data-testid="stExpander"] summary:hover {
        background-color: #59602d !important;
    }
    .stTable th { color: #b69b79 !important; font-size: 0.75rem !important; text-transform: uppercase !important; }
    .stTable td { color: #e0cfb3 !important; }
    hr { border-color: #503a2b !important; }
    .lb-row-divider {
        border: 0 !important;
        border-top: 1px solid #503a2b !important;
        margin: 0.55rem 0 0.65rem !important;
        opacity: 0.7;
    }
    .lb-row-divider-paired { margin-top: 1.55rem !important; }
    .lb-row-divider-top { margin-top: 0.65rem !important; }
    #MainMenu, footer, header { visibility: hidden; }
    h2 a[data-testid], h1 a[data-testid], h3 a[data-testid] { display: none !important; }

    /* Client-side show-more controls avoid a Streamlit rerun. */
    .lb-show-more {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.5rem;
        width: 100%;
        min-height: 2rem;
    }
    .lb-show-more summary,
    .st-key-lb_live_sort_btn button {
        display: block;
        width: max-content !important;
        cursor: pointer;
        list-style: none;
        background-color: #161616 !important;
        border: 1px solid #d4a843 !important;
        border-radius: 4px !important;
        color: #d4a843 !important;
        font-family: inherit !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.08em !important;
        padding: 0.25rem 0.75rem !important;
        height: auto !important;
        min-height: 0 !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    .st-key-lb_live_sort_btn {
        display: flex !important;
        width: 100% !important;
        justify-content: flex-end !important;
    }
    .st-key-lb_live_sort_btn button { margin-left: auto !important; }
    div[data-testid="stHorizontalBlock"]:has(.lb-show-more-live.lb-live-extra-1[open]) .st-key-lb_live_sort_btn { margin-top: 3.8rem !important; }
    div[data-testid="stHorizontalBlock"]:has(.lb-show-more-live.lb-live-extra-2[open]) .st-key-lb_live_sort_btn { margin-top: 6.2rem !important; }
    div[data-testid="stHorizontalBlock"]:has(.lb-show-more-live.lb-live-extra-3[open]) .st-key-lb_live_sort_btn { margin-top: 8.6rem !important; }
    div[data-testid="stHorizontalBlock"]:has(.lb-show-more-live.lb-live-extra-4[open]) .st-key-lb_live_sort_btn { margin-top: 11rem !important; }
    div[data-testid="stHorizontalBlock"]:has(.lb-show-more-live.lb-live-extra-5[open]) .st-key-lb_live_sort_btn { margin-top: 13.4rem !important; }
    div[data-testid="stHorizontalBlock"]:has(.lb-show-more-live[open]) .st-key-lb_live_sort_btn {
        position: relative !important;
        z-index: 10 !important;
    }
    .lb-show-more[open] {
        display: flex;
        flex-direction: column;
        align-items: stretch;
        width: 100%;
    }
    .lb-show-more[open] > summary { order: 2; }
    .lb-show-more[open] > .lb-more-rows { order: 1; }
    .lb-show-more-live[open] > summary { order: 1; }
    .lb-show-more-live[open] > .lb-more-rows { order: 2; }
    .lb-show-more .lb-less-label { display: none; }
    .lb-show-more[open] .lb-more-label { display: none; }
    .lb-show-more[open] .lb-less-label { display: inline; }
    .lb-more-rows {
        align-self: stretch;
        width: 100%;
        min-width: 100%;
        box-sizing: border-box;
    }
    .lb-more-rows > div { width: 100% !important; box-sizing: border-box; }
    .lb-show-more-live .lb-more-rows { width: calc(200% + 1rem); min-width: calc(200% + 1rem); position: relative; z-index: 1; }
    .lb-show-more summary::-webkit-details-marker { display: none; }
    .lb-show-more summary:hover,
    .st-key-lb_live_sort_btn button:hover {
        background-color: #d4a843 !important;
        color: #0e0e0e !important;
    }
    .lb-show-more-disabled {
        display: inline-block;
        border: 1px solid #604b24;
        border-radius: 4px;
        color: #8b712e;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        padding: 0.25rem 0.75rem;
        text-transform: uppercase;
    }

    .lb-hover-row {
        position: relative;
        overflow: visible !important;
    }
    .lb-hover-card {
        position: absolute;
        left: 0;
        top: calc(100% + 6px);
        min-width: 220px;
        max-width: 260px;
        background: rgba(14, 14, 14, 0.98);
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        padding: 0.6rem 0.75rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.45);
        opacity: 0;
        visibility: hidden;
        transform: translateY(-4px);
        transition: opacity 120ms ease, transform 120ms ease, visibility 120ms ease;
        pointer-events: none;
        z-index: 100;
    }
    .lb-hover-name {
        position: relative;
        display: inline-block;
    }
    .lb-hover-name:hover {
        z-index: 40;
    }
    .lb-hover-name:hover .lb-hover-card {
        opacity: 1;
        visibility: visible;
        transform: translateY(0);
    }
    .lb-hover-title {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 0.75rem;
        color: #eee;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .lb-hover-meta {
        color: #777;
        font-size: 0.72rem;
        font-weight: 500;
        white-space: nowrap;
    }
    .lb-hover-rank {
        color: #d4a843;
        font-size: 0.72rem;
        font-weight: 700;
        margin-left: 0.35rem;
    }
    .lb-hover-grid {
        display: grid;
        grid-template-columns: auto auto;
        column-gap: 0.8rem;
        row-gap: 0.22rem;
        align-items: baseline;
    }
    .lb-hover-label {
        color: #666;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .lb-hover-value {
        color: #ddd;
        font-size: 0.8rem;
        font-weight: 600;
        text-align: right;
    }

    .profile-achievement-hover {
        position: relative;
        display: inline-block;
        margin-right: 0.8rem;
        cursor: help;
        overflow: visible !important;
    }
    .profile-leaderboard-region-hover { margin-right: 0; }
    .profile-achievement-hover:hover { z-index: 1000; }
    .profile-achievement-tooltip {
        position: absolute;
        left: 0;
        top: calc(100% + 6px);
        min-width: 230px;
        max-width: 300px;
        padding: 0.6rem 0.75rem;
        border: 1px solid #3a3a3a;
        border-radius: 7px;
        background: rgba(14, 14, 14, 0.98);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.45);
        color: #ccc;
        font-size: 0.76rem;
        line-height: 1.5;
        opacity: 0;
        visibility: hidden;
        transform: translateY(-3px);
        transition: opacity 120ms ease, transform 120ms ease, visibility 120ms ease;
        pointer-events: none;
        z-index: 1001;
        white-space: normal;
    }
    .profile-achievement-hover:hover .profile-achievement-tooltip {
        opacity: 1;
        visibility: visible;
        transform: translateY(0);
    }
    .profile-achievement-tooltip-title {
        display: block;
        margin-bottom: 0.3rem;
        color: #eee;
        font-weight: 700;
    }
    .profile-achievement-counts {
        display: block;
        white-space: nowrap;
    }
    .profile-achievement-total {
        display: block;
        margin-top: 0.15rem;
        color: #777;
        font-size: 0.7rem;
    }
    .profile-achievement-seasons {
        display: block;
        margin-top: 0.25rem;
    }
    .profile-achievement-season {
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        white-space: nowrap;
    }

    /* Icon button wrapper (Home arrow) */
    .icon-btn button {
        background: transparent !important;
        border: 1px solid #2a2a2a !important;
        color: #999 !important;
        padding: 0.15rem 0.45rem !important;
        border-radius: 6px !important;
        font-size: 1.0rem !important;
        line-height: 1.1rem !important;
    }
    .icon-btn button:hover {
        border-color: #d4a843 !important;
        color: #d4a843 !important;
    }

    /* Warm up inline-styled leaderboard rows and stat cards as well. */
    [style*="background:#121212"], [style*="background:#111"],
    [style*="background:#161616"] { background-color: #141312 !important; }
    [style*="border:1px solid #1e1e1e"], [style*="border-top:1px solid #1e1e1e"] {
        border-color: #60462f !important;
    }
    [style*="color:#eee"] { color: #f2e5cf !important; }
    [style*="color:#ccc"], [style*="color:#bbb"], [style*="color:#ddd"] { color: #e0cfb3 !important; }
    [style*="color:#666"], [style*="color:#555"], [style*="color:#444"] { color: #b69b79 !important; }

    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(
                90deg,
                transparent calc(50% - 521px),
                #191715 calc(50% - 520px),
                #191715 calc(50% + 520px),
                transparent calc(50% + 521px)
            ),
            linear-gradient(rgba(20, 13, 9, 0.62), rgba(20, 13, 9, 0.68)),
            url("data:image/avif;base64,{_tavern_bg_b64}");
        background-size: auto, cover, cover;
        background-position: center center, center center, center center;
        background-attachment: fixed, fixed, fixed;
    }}
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background: transparent !important;
    }}
    div[data-testid="stMainBlockContainer"] {{
        box-sizing: border-box;
        min-height: 100vh;
        background: transparent;
    }}
    @media (max-width: 1080px) {{
        .stApp {{
            background-image:
                linear-gradient(#191715, #191715),
                linear-gradient(rgba(20, 13, 9, 0.62), rgba(20, 13, 9, 0.68)),
                url("data:image/avif;base64,{_tavern_bg_b64}");
        }}
        div[data-testid="stMainBlockContainer"] {{
            background: transparent;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

    _logo_b64 = _b64.b64encode((_app_dir / "nerdbob.png").read_bytes()).decode()
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:1rem; margin-bottom:1.0rem;'>
      <img src='data:image/png;base64,{_logo_b64}' style='height:72px; width:72px; object-fit:cover; border-radius:8px; flex-shrink:0;'>
      <div style='line-height:1.2;'>
        <div style='color:#eee; font-size:1.5rem; font-weight:normal; margin:0 0 0.2rem 0;'><a href='?goto_home=1' style='color:inherit;text-decoration:none;' onmouseover="this.style.opacity='0.7'" onmouseout="this.style.opacity='1'">Placement Statistics</a></div>
        <div style='color:#555; font-size:0.8rem; margin:0; text-transform:uppercase; letter-spacing:0.08em;'>Hearthstone Battlegrounds Leaderboard Stats</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
