"""
wa2_cards.py  –  Battlegrounds Card Library

Fetches live card data from HearthstoneJSON (no local image files needed).
Card art is served directly from art.hearthstonejson.com via CDN.

Cached for 1 hour with st.cache_data so repeated visits are instant.
Automatically picks up new/changed cards after each patch with zero maintenance.
"""

import html

import requests
import streamlit as st
import streamlit.components.v1 as components

# ── Constants ─────────────────────────────────────────────────────────────────

HSJSON_URL = "https://api.hearthstonejson.com/v1/latest/enUS/cards.json"

# BG card image CDN  –  256x is fast; swap to 512x for higher res
def _card_img_url(card_id: str, golden: bool = False) -> str:
    display_id = card_id if not golden else card_id + "_G"
    return f"https://art.hearthstonejson.com/v1/bgs/latest/enUS/256x/{display_id}.png"

# Map HearthstoneJSON race values → display labels (matches old TRIBE_LABELS)
RACE_TO_TRIBE = {
    "MURLOC":     "murloc",
    "BEAST":      "beast",
    "MECHANICAL": "mechs",
    "MECH":       "mechs",
    "DEMON":      "demons",
    "DRAGON":     "dragons",
    "ELEMENTAL":  "elementals",
    "NAGA":       "naga",
    "PIRATE":     "pirates",
    "QUILBOAR":   "quilboar",
    "UNDEAD":     "undead",
    "INVALID":    "neutral",
    "ALL":        "neutral",
}

TRIBES = [
    "beast", "demons", "dragons", "elementals", "mechs",
    "murloc", "naga", "neutral", "pirates", "quilboar", "undead",
]
TRIBE_LABELS = {t: t.capitalize() for t in TRIBES}
TRIBE_LABELS["mechs"] = "Mech"
TRIBE_LABELS["murloc"] = "Murloc"

TIERS = [1, 2, 3, 4, 5, 6]


def _associated_races_to_tribes(races: list[str] | None) -> list[str]:
    tribes = []
    for race in races or []:
        tribe = RACE_TO_TRIBE.get(race, "neutral")
        if tribe not in tribes:
            tribes.append(tribe)
    return tribes or ["neutral"]


def _is_duo_card(card: dict) -> bool:
    """Return True for card variants whose display name explicitly contains 'Duo'."""
    haystacks = [card.get("display_name", ""), card.get("name", "")]
    return any("duo" in text.lower() for text in haystacks if text)


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_all_cards() -> list[dict]:
    """Fetch the full HearthstoneJSON card list. Cached 1 hour."""
    try:
        r = requests.get(HSJSON_URL, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Could not fetch card data from HearthstoneJSON: {e}")
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def _get_all_minion_cards() -> list[dict]:
    raw = _fetch_all_cards()
    result = []
    for c in raw:
        if (
            c.get("set") == "BATTLEGROUNDS"
            and c.get("type") == "MINION"
            and c.get("techLevel")
            and not c.get("id", "").endswith("_G")
            and not c.get("id", "").lower().startswith("tb_")
            and c.get("name")
            and c.get("isBattlegroundsPoolMinion") == True
        ):
            if "battlegroundsAssociatedRaces" in c:
                card_tribes = _associated_races_to_tribes(c.get("battlegroundsAssociatedRaces"))
            else:
                race = c.get("race", "INVALID")
                card_tribes = [RACE_TO_TRIBE.get(race, "neutral")]

            result.append({
                "name":         c["id"],
                "display_name": c["name"],
                "tier":         int(c["techLevel"]),
                "tribes":       card_tribes,  # <── Skicka in hela listan med tribes här!
                "attack":       c.get("attack", 0),
                "health":       c.get("health", 0),
                "text":         c.get("text", ""),
                "card_id":      c["id"],
                "img_url":      _card_img_url(c["id"]),
                "img_url_gold": _card_img_url(c["id"], golden=True),
            })
    result.sort(key=lambda x: (x["tier"], x["display_name"]))
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def _get_all_trinket_cards() -> list[dict]:
    raw = _fetch_all_cards()
    result = []
    for c in raw:
        if c.get("type") != "BATTLEGROUND_TRINKET":
            continue
        if not c.get("name"):
            continue
        spell_school = c.get("spellSchool", "")
        if "LESSER" in spell_school:
            trinket_type = "trinket_lesser"
        elif "GREATER" in spell_school:
            trinket_type = "trinket_greater"
        else:
            continue
        tribes = _associated_races_to_tribes(c.get("battlegroundsAssociatedRaces"))
        result.append({
            "tribe":        tribes[0],
            "tribes":       tribes,
            "trinket_type": trinket_type,
            "name":         c["id"],
            "display_name": c["name"],
            "cost":         c.get("cost", 0),
            "text":         c.get("text", ""),
            "card_id":      c["id"],
            "img_url":      _card_img_url(c["id"]),
        })
    result.sort(key=lambda x: (x["cost"], x["display_name"]))
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def _get_all_spell_cards() -> list[dict]:
    raw = _fetch_all_cards()
    result = []
    for c in raw:
        if (
            c.get("type") == "BATTLEGROUND_SPELL"
            and c.get("isBattlegroundsPoolSpell") == True
            and c.get("techLevel")
            and not c.get("id", "").endswith("_G")
            and c.get("name")
        ):
            result.append({
                "name":         c["id"],
                "display_name": c["name"],
                "tier":         int(c["techLevel"]),
                "cost":         c.get("cost", 0),
                "text":         c.get("text", ""),
                "card_id":      c["id"],
                "img_url":      _card_img_url(c["id"]),
            })
    result.sort(key=lambda x: (x["tier"], x["cost"], x["display_name"]))
    return result


# ── Card grid renderer ────────────────────────────────────────────────────────

def _render_card_grid(cards: list[dict], cols_per_row: int = 6, golden: bool = False):
    """Render cards at a stable size in a horizontally scrollable row."""
    if not cards:
        st.info("No cards matched the filter.")
        return

    card_items = []
    for card in cards:
        url = card.get("img_url_gold") if golden else card.get("img_url")
        if not url:
            url = card.get("img_url", "")
        display_name = html.escape(card["display_name"])
        card_items.append(
            f"""
            <div class="wa-card-browser-card">
                <img src="{html.escape(url, quote=True)}" alt="{display_name}" loading="lazy">
                <div class="wa-card-browser-caption" title="{display_name}">{display_name}</div>
            </div>
            """
        )

    components.html(
        f"""
        <!doctype html>
        <html>
        <head>
        <style>
        html,
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .wa-card-browser-scroll {{
            box-sizing: border-box;
            display: flex;
            gap: 12px;
            width: 100vw;
            max-width: 100%;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 4px 2px 14px;
            scroll-snap-type: x proximity;
            scrollbar-gutter: stable;
            -webkit-overflow-scrolling: touch;
        }}
        .wa-card-browser-card {{
            flex: 0 0 156px;
            width: 156px;
            scroll-snap-align: start;
        }}
        .wa-card-browser-card img {{
            display: block;
            width: 156px;
            height: auto;
        }}
        .wa-card-browser-caption {{
            margin-top: 6px;
            font-size: 0.9rem;
            font-weight: 600;
            line-height: 1.2;
            text-align: center;
            color: #f2f4f8;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.75);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        @media (max-width: 640px) {{
            .wa-card-browser-card,
            .wa-card-browser-card img {{
                flex-basis: 132px;
                width: 132px;
            }}
            .wa-card-browser-scroll {{
                gap: 10px;
            }}
        }}
        </style>
        </head>
        <body>
        <div class="wa-card-browser-scroll">
            {''.join(card_items)}
        </div>
        </body>
        </html>
        """,
        height=286,
        scrolling=True,
    )


# ── Public entry point (called from wa2_app.py) ───────────────────────────────

def show_card_browser():
    st.markdown("## Card Library")

    card_type = st.radio(
        "Type",
        ["Minions", "Trinkets", "Spells"],
        horizontal=True,
        key="cb_type",
    )
    include_duos = st.checkbox("Include Duos?", value=False, key="cb_include_duos")
    filter_col_1, filter_col_2, filter_col_3 = st.columns([2, 2, 1])

    def _apply_duo_filter(cards: list[dict]) -> list[dict]:
        if include_duos:
            return cards
        return [card for card in cards if not _is_duo_card(card)]

    # ── Spells ──────────────────────────────────────────────────────────────
    if card_type == "Spells":
        with st.spinner("Loading spells..."):
            all_spells = _apply_duo_filter(_get_all_spell_cards())

        if not all_spells:
            st.warning("Spell data unavailable. Check your internet connection.")
            return

        tier_options = ["All"] + [f"Tier {t}" for t in TIERS]
        selected_tier_label = filter_col_1.selectbox(
            "Tier", tier_options, index=0, key="cb_spell_tier_select"
        )
        selected_tiers = (
            TIERS if selected_tier_label == "All"
            else [int(selected_tier_label.split(" ")[1])]
        )

        tier_filtered = [c for c in all_spells if c["tier"] in selected_tiers]

        cost_options = ["All"] + [
            str(cost) for cost in sorted({c["cost"] for c in tier_filtered})
        ]
        selected_cost_label = filter_col_2.selectbox(
            "Cost", cost_options, index=0, key="cb_spell_cost_select"
        )
        selected_cost = None if selected_cost_label == "All" else int(selected_cost_label)

        filtered = [
            c for c in tier_filtered
            if selected_cost is None or c["cost"] == selected_cost
        ]

        if not filtered:
            st.info("No spells found for this filter.")
            return

        for tier in selected_tiers:
            tier_cards = [c for c in filtered if c["tier"] == tier]
            if not tier_cards:
                continue
            st.markdown(f"### ⭐ Tier {tier}")
            _render_card_grid(tier_cards, cols_per_row=max(4, min(8, len(tier_cards))))
        return

    # ── Shared tribe filter ──────────────────────────────────────────────────
    tribe_options = ["All"] + [TRIBE_LABELS[t] for t in TRIBES]
    default_tribe_idx = tribe_options.index("Beast") if "Beast" in tribe_options else 0
    selected_label = filter_col_1.selectbox(
        "Tribe", tribe_options, index=default_tribe_idx, key="cb_tribe_select"
    )
    selected_tribes = (
        list(TRIBES)
        if selected_label == "All"
        else [t for t in TRIBES if TRIBE_LABELS[t] == selected_label]
    )
    selected_set = set(selected_tribes)

    # ── Minions ──────────────────────────────────────────────────────────────
    if card_type == "Minions":
        with st.spinner("Loading minions..."):
            all_minions = _apply_duo_filter(_get_all_minion_cards())

        if not all_minions:
            st.warning("Minion data unavailable. Check your internet connection.")
            return

        tier_options = ["All"] + [f"Tier {t}" for t in TIERS]
        selected_tier_label = filter_col_2.selectbox(
            "Tier", tier_options, index=0, key="cb_tier_select"
        )
        selected_tiers = (
            TIERS if selected_tier_label == "All"
            else [int(selected_tier_label.split(" ")[1])]
        )

        show_golden = filter_col_3.checkbox("Golden", key="cb_golden")

        filtered = [
            c for c in all_minions
            if c["tier"] in selected_tiers and selected_set.intersection(c["tribes"])
        ]

        if not filtered:
            st.info("No minions matched the filter.")
            return

        for tier in selected_tiers:
            tier_cards = [c for c in filtered if c["tier"] == tier]
            if not tier_cards:
                continue
            st.markdown(f"### ⭐ Tier {tier}")
            cols_count = max(4, min(8, len(tier_cards)))
            _render_card_grid(tier_cards, cols_per_row=cols_count, golden=show_golden)

    # ── Trinkets ─────────────────────────────────────────────────────────────
    elif card_type == "Trinkets":
        with st.spinner("Loading trinkets..."):
            all_trinkets = _apply_duo_filter(_get_all_trinket_cards())

        if not all_trinkets:
            st.warning("Trinket data unavailable. Check your internet connection.")
            return

        trinket_type_label = filter_col_2.radio(
            "Trinket type", ["Greater", "Lesser"], horizontal=True, key="cb_trinket_type"
        )
        folder_name = (
            "trinket_greater" if trinket_type_label == "Greater" else "trinket_lesser"
        )

        type_and_tribe_filtered = [
            t for t in all_trinkets
            if t["trinket_type"] == folder_name and selected_set.intersection(t["tribes"])
        ]
        cost_options = ["All"] + [
            str(cost) for cost in sorted({t["cost"] for t in type_and_tribe_filtered})
        ]
        selected_cost_label = filter_col_3.selectbox(
            "Cost", cost_options, index=0, key="cb_trinket_cost_select"
        )
        selected_cost = None if selected_cost_label == "All" else int(selected_cost_label)

        filtered = [
            t for t in type_and_tribe_filtered
            if selected_cost is None or t["cost"] == selected_cost
        ]

        if not filtered:
            st.info("No trinkets matched the filter.")
            return

        cols_count = max(4, min(8, len(filtered)))
        _render_card_grid(filtered, cols_per_row=cols_count)
