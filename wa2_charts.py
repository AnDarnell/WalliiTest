import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def style_dark_axes(ax):
    ax.xaxis.label.set_color("#aaa")
    ax.yaxis.label.set_color("#aaa")
    ax.tick_params(axis="x", colors="#aaa")
    ax.tick_params(axis="y", colors="#aaa")
    ax.grid(True, alpha=0.2, color="#444")
    for spine in ax.spines.values():
        spine.set_visible(False)


def normalized_counts(games):
    counts = {p: 0 for p in range(1, 9)}
    half_counts = {}
    for g in games:
        p = g["placement"]
        if p == int(p):
            counts[int(p)] += 1
        else:
            low, high = int(p), int(p) + 1
            half_counts[(low, high)] = half_counts.get((low, high), 0) + 1
    for (low, high), n in half_counts.items():
        counts[low] += n // 2
        counts[high] += n // 2 + n % 2
    return counts


def norm_to_pct(games):
    counts = normalized_counts(games)
    total = sum(counts.values())
    if total == 0:
        return {p: 0.0 for p in range(1, 9)}
    return {p: counts[p] / total * 100 for p in range(1, 9)}


@st.cache_data(show_spinner=False)
def make_chart(games):
    norm = normalized_counts(games)
    labels = [str(p) for p in range(1, 9)]
    values = [norm[p] for p in range(1, 9)]
    colors = ["#d4a843" if p == 1 else "#4a8c5c" if p <= 4 else "#8c3a2a" for p in range(1, 9)]
    total = len(games)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    bars = ax.bar(labels, values, color=colors, edgecolor="#222", linewidth=0.8, width=0.55)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val}\n{val/total*100:.1f}%",
                ha="center",
                va="bottom",
                color="#aaa",
                fontsize=13,
            )

    ax.set_ylim(0, max(values) * 1.45)
    ax.set_xlabel("Placement", fontsize=12, labelpad=10)
    ax.tick_params(labelsize=10)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    style_dark_axes(ax)

    avg_place = sum(g["placement"] for g in games) / total
    ax.text(
        0.02,
        0.97,
        f"Avg: {avg_place:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#aaa",
        fontsize=12,
    )

    ax.legend(
        handles=[
            mpatches.Patch(color="#d4a843", label="1st"),
            mpatches.Patch(color="#4a8c5c", label="Top 4"),
            mpatches.Patch(color="#8c3a2a", label="Bot 4"),
        ],
        facecolor="#161616",
        labelcolor="#aaa",
        fontsize=11,
        edgecolor="#555",
        framealpha=1,
        loc="upper right",
    )

    plt.tight_layout(pad=1.2)
    return fig


def make_neighbor_chart(all_pcts, names, ranks, player_name, player_rank):
    avg_pcts = {p: np.mean([d[p] for d in all_pcts]) for p in range(1, 9)}
    labels = [str(p) for p in range(1, 9)]
    values = [avg_pcts[p] for p in range(1, 9)]
    colors = ["#d4a843" if p == 1 else "#4a8c5c" if p <= 4 else "#8c3a2a" for p in range(1, 9)]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    bars = ax.bar(labels, values, color=colors, edgecolor="#222", linewidth=0.8, width=0.55)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                color="#aaa",
                fontsize=16,
            )

    n = len(all_pcts)
    above = sum(1 for r in ranks if r < player_rank)
    below = sum(1 for r in ranks if r > player_rank)
    ax.set_title(
        f"Neighbor average  ({above} above - {below} below - {n} players - rank {player_rank})",
        color="#666",
        fontsize=9,
        pad=10,
    )
    ax.set_xlabel("Placement", fontsize=12, labelpad=10)
    ax.tick_params(labelsize=10)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.set_ylim(0, max(values) * 1.45)
    style_dark_axes(ax)

    ax.legend(
        handles=[
            mpatches.Patch(color="#d4a843", label="1st"),
            mpatches.Patch(color="#4a8c5c", label="Top 4"),
            mpatches.Patch(color="#8c3a2a", label="Bot 4"),
        ],
        facecolor="#161616",
        labelcolor="#aaa",
        fontsize=11,
        edgecolor="#555",
        framealpha=1,
        loc="upper right",
    )

    plt.tight_layout(pad=1.2)
    return fig


def delta_color(delta):
    if delta <= -0.40:
        return "#4a8c5c"
    elif delta <= -0.20:
        return "#6aab6a"
    elif delta <= 0.02:
        return "#7ab87a"
    elif delta <= 0.20:
        return "#d4a843"
    elif delta <= 0.40:
        return "#c47a75"
    else:
        return "#8c3a2a"


def diff_pct_color(diff):
    if diff <= -10:
        return "#8c3a2a"
    elif diff < 0:
        return "#c47a75"
    elif diff == 0:
        return "#555"
    elif diff < 10:
        return "#7ab87a"
    else:
        return "#4a8c5c"


def summarize_neighbor_differences(player_pct, avg_pct, player_name="this player"):
    """Return a short third-person summary of how the player's placements differ from the neighbor average."""

    def ordinal(n):
        if 10 <= n % 100 <= 20:
            return f"{n}th"
        return {1: "1st", 2: "2nd", 3: "3rd"}.get(n % 10, f"{n}th")

    def label(diff):
        if diff >= 5.0:
            return "significantly more"
        if diff >= 2.5:
            return "noticably more"
        return "slightly more"

    def label_less(diff):
        if diff <= -5.0:
            return "significantly less"
        if diff <= -2.5:
            return "noticably less"
        return "slightly less"

    diffs = [(p, player_pct[p] - avg_pct[p]) for p in range(1, 9)]
    more = [(p, diff) for p, diff in diffs if diff >= 2.5]
    less = [(p, diff) for p, diff in diffs if diff <= -2.5]

    if not more and not less:
        return f"{player_name}'s placements look very similar to this neighbor group, and the rest is fairly consistent."

    def join_items(items):
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    more_phrase = join_items([f"{ordinal(p)}" for p, _ in more[:3]]) + " places"
    less_phrase = join_items([f"{ordinal(p)}" for p, _ in less[:3]]) + " places"

    if more and less:
        return (
            f"{player_name} has {label(more[0][1])} {more_phrase}, and {label_less(less[0][1])} {less_phrase}. "
            "Everything else is fairly consistent."
        )
    if more:
        return f"{player_name} has {label(more[0][1])} {more_phrase}. Everything else is fairly consistent."
    return f"{player_name} has {label_less(less[0][1])} {less_phrase}. Everything else is fairly consistent."
