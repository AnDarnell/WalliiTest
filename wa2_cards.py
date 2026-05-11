from collections import defaultdict
from pathlib import Path

import streamlit as st


CARDS_ROOT = Path(__file__).parent / "cards"

TRIBES = [
    "beast",
    "demons",
    "dragons",
    "elementals",
    "mechs",
    "murloc",
    "naga",
    "neutral",
    "pirates",
    "quilboar",
    "undead",
]
TRIBE_LABELS = {t: t.capitalize() for t in TRIBES}
TIERS = [1, 2, 3, 4, 5, 6]


@st.cache_data(show_spinner=False)
def _get_minion_images(tribes, tiers):
    by_name = defaultdict(lambda: defaultdict(list))

    for tribe in TRIBES:
        folder = CARDS_ROOT / f"S13_{tribe}" / "minions"
        for tier in tiers:
            tier_folder = folder / f"tier {tier}"
            if tier_folder.exists():
                for img in sorted(tier_folder.glob("*.png")):
                    by_name[img.name][tier].append((tribe, img))

    result = []
    selected_set = set(tribes)
    for name, tiers_dict in by_name.items():
        for tier, entries in tiers_dict.items():
            tribes_list = [t for t, _ in entries]
            if not selected_set.intersection(tribes_list):
                continue
            path = entries[0][1]
            result.append((tribes_list, tier, path))

    result.sort(key=lambda x: (x[1], x[2].name))
    return result


@st.cache_data(show_spinner=False)
def _get_trinket_images(tribes, trinket_type):
    result = []
    for tribe in tribes:
        folder = CARDS_ROOT / f"S13_{tribe}" / trinket_type
        if folder.exists():
            for img in sorted(folder.glob("*.png")):
                result.append((tribe, img))
    return result


@st.cache_data(show_spinner=False)
def _get_spell_images():
    folder = CARDS_ROOT / "S13_spells"
    if not folder.exists():
        return []
    return list(sorted(folder.glob("*.png")))


@st.cache_data(show_spinner=False)
def _get_all_spell_cards():
    result = []
    folder = CARDS_ROOT / "S13_spells"
    if not folder.exists():
        return result
    category_folder = folder / "staying"
    if not category_folder.exists():
        return result
    for tier in TIERS:
        tier_folder = category_folder / f"tier {tier}"
        if not tier_folder.exists():
            continue
        for img in sorted(tier_folder.glob("*.png")):
            result.append(
                {
                    "name": img.stem,
                    "display_name": img.stem.replace("_", " "),
                    "tier": tier,
                    "path_str": str(img),
                }
            )
    result.sort(key=lambda x: (x["tier"], x["name"]))
    return result


@st.cache_data(show_spinner=False)
def _get_all_trinket_cards():
    result = []
    for trinket_type in ("trinket_greater", "trinket_lesser"):
        for tribe in TRIBES:
            folder = CARDS_ROOT / f"S13_{tribe}" / trinket_type
            if folder.exists():
                for img in sorted(folder.glob("*.png")):
                    result.append(
                        {
                            "tribe": tribe,
                            "trinket_type": trinket_type,
                            "name": img.stem,
                            "path_str": str(img),
                        }
                    )
    return result


@st.cache_data(show_spinner=False)
def _get_all_minion_cards():
    by_name = defaultdict(lambda: defaultdict(list))
    for tribe in TRIBES:
        folder = CARDS_ROOT / f"S13_{tribe}" / "minions"
        for tier in TIERS:
            tier_folder = folder / f"tier {tier}"
            if tier_folder.exists():
                for img in sorted(tier_folder.glob("*.png")):
                    by_name[img.stem][tier].append((tribe, img))
    result = []
    for name, tiers_dict in by_name.items():
        for tier, entries in tiers_dict.items():
            tribes_list = [t for t, _ in entries]
            path = entries[0][1]
            result.append(
                {
                    "name": name,
                    "display_name": name.replace("_", " "),
                    "tier": tier,
                    "tribes": tribes_list,
                    "path_str": str(path),
                }
            )
    result.sort(key=lambda x: (x["tier"], x["name"]))
    return result


def show_card_browser():
    st.markdown(
        """
    <style>
    div[data-testid="stCheckbox"] label p { font-size: 1rem !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("## Card Library – Season 13")

    card_type = st.radio(
        "Type",
        ["Minions", "Trinkets", "Spells*"],
        horizontal=True,
        key="cb_type",
    )

    filter_col_1, filter_col_2 = st.columns(2)

    if card_type == "Spells*":
        tier_options = ["All"] + [f"Tier {t}" for t in TIERS]
        selected_tier_label = filter_col_1.selectbox("Tier", tier_options, index=0, key="cb_spell_tier_select")
        selected_tiers = TIERS if selected_tier_label == "All" else [int(selected_tier_label.split(" ")[1])]

        all_cards = _get_all_spell_cards()
        if not all_cards:
            st.info("Updating soon...")
            return

        filtered = [c for c in all_cards if c["tier"] in selected_tiers]

        if not filtered:
            st.info("No cards matched the filter.")
            return

        for tier in selected_tiers:
            tier_cards = [c for c in filtered if c["tier"] == tier]
            if not tier_cards:
                continue
            st.markdown(f"### ⭐ Tier {tier}")
            cols_per_row = max(4, min(8, len(tier_cards)))
            cols = st.columns(cols_per_row)
            for i, card in enumerate(tier_cards):
                cols[i % cols_per_row].image(card["path_str"], width="stretch")
        return

    tribe_options = ["All"] + [TRIBE_LABELS[t] for t in TRIBES]
    selected_label = filter_col_1.selectbox("Tribe", tribe_options, index=tribe_options.index("Beast"), key="cb_tribe_select")
    if selected_label == "All":
        selected_tribes = list(TRIBES)
    else:
        selected_tribes = [t for t in TRIBES if TRIBE_LABELS[t] == selected_label]

    if card_type == "Minions":
        tier_options = ["All"] + [f"Tier {t}" for t in TIERS]
        selected_tier_label = filter_col_2.selectbox("Tier", tier_options, index=0, key="cb_tier_select")
        selected_tiers = TIERS if selected_tier_label == "All" else [int(selected_tier_label.split(" ")[1])]

        all_cards = _get_all_minion_cards()
        selected_set = set(selected_tribes)
        filtered = [
            c
            for c in all_cards
            if c["tier"] in selected_tiers and selected_set.intersection(c["tribes"])
        ]

        if not filtered:
            st.info("No cards matched the filter.")
            return

        for tier in selected_tiers:
            tier_cards = [c for c in filtered if c["tier"] == tier]
            if not tier_cards:
                continue
            st.markdown(f"### ⭐ Tier {tier}")
            cols_per_row = max(4, min(8, len(selected_tribes) * 2))
            cols = st.columns(cols_per_row)
            for i, card in enumerate(tier_cards):
                cols[i % cols_per_row].image(card["path_str"], width="stretch")

    elif card_type == "Trinkets":
        trinket_type = st.radio(
            "Trinket type", ["Greater", "Lesser"], horizontal=True, key="cb_trinket_type"
        )
        folder_name = "trinket_greater" if trinket_type == "Greater" else "trinket_lesser"

        all_trinkets = _get_all_trinket_cards()
        selected_set = set(selected_tribes)
        filtered = [
            t
            for t in all_trinkets
            if t["trinket_type"] == folder_name and t["tribe"] in selected_set
        ]

        if not filtered:
            st.info("No trinkets matched the filter.")
            return

        cols_per_row = max(4, min(8, len(selected_tribes) * 2))
        cols = st.columns(cols_per_row)
        for i, card in enumerate(filtered):
            cols[i % cols_per_row].image(card["path_str"], width="stretch")
