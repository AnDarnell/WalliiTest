from pathlib import Path
import re

path = Path("/mnt/data/wallii_app.py")
txt = path.read_text(encoding="utf-8")

# Ensure pandas import
if "import pandas as pd" not in txt:
    txt = txt.replace("import numpy as np", "import numpy as np\nimport pandas as pd")

# Add weighted quantile + binned functions if not already
if "def weighted_quantile" not in txt:
    insert_point = txt.find("# ── Regression helpers")
    func_block = r'''
def weighted_quantile(values, weights, q):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)

    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[m], weights[m]
    if len(values) == 0:
        return np.nan

    idx = np.argsort(values)
    values, weights = values[idx], weights[idx]
    cum = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(q, cum, values))


def binned_weighted_curve(df, x_col="lb_mmr", y_col="avg_place", w_col="games", bin_size=500, mode="wmedian", q=0.5, min_games=0):
    """
    mode:
      - "wmean"   : weighted mean in each bin (weights = games)
      - "wquant"  : weighted quantile in each bin (q controls quantile; q=0.5 => weighted median)
    """
    d = df[[x_col, y_col, w_col]].dropna().copy()
    d = d[np.isfinite(d[x_col]) & np.isfinite(d[y_col]) & np.isfinite(d[w_col])]
    d = d[d[w_col] > 0]
    if min_games > 0:
        d = d[d[w_col] >= min_games]

    if d.empty:
        return np.array([]), np.array([])

    d["bin"] = (d[x_col] // bin_size) * bin_size

    xs, ys = [], []
    for b, g in d.groupby("bin"):
        y = g[y_col].to_numpy(float)
        w = g[w_col].to_numpy(float)
        if mode == "wmean":
            val = float((w * y).sum() / w.sum())
        else:
            val = weighted_quantile(y, w, q)
        if np.isfinite(val):
            xs.append(float(b))
            ys.append(val)

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    order = np.argsort(xs)
    return xs[order], ys[order]
'''
    txt = txt[:insert_point] + func_block + "\n" + txt[insert_point:]

# Update regression tab UI section: replace whole with tabs[1] block content
pattern = re.compile(r"# ── Regression tab ────────────────────────────────────────────────────────────.*?\n\s*except Exception as e:\n\s*st\.error\(str\(e\)\)\n", re.DOTALL)
m = pattern.search(txt)
if not m:
    raise RuntimeError("Could not find regression tab block to replace.")

new_block = r'''# ── Regression tab ────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown(
        "<p style='color:#bbb; font-size:0.9rem; margin:0 0 0.8rem;'>"
        "Load a previously exported CSV and build a <b>binned curve</b> (Option C) for "
        "<b>Avg Place vs MMR</b>. You can also weight each player by <b>games</b>."
        "</p>",
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("Upload exported CSV", type=["csv"])
    if not uploaded:
        st.info("Upload your CSV export to begin.")
    else:
        try:
            df = pd.read_csv(uploaded)

            # Basic validation
            required = {"lb_mmr", "current_mmr", "avg_place", "games"}
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"CSV is missing columns: {', '.join(missing)}")
            else:
                c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.1, 1.0])
                with c1:
                    x_choice = st.selectbox("X (MMR source)", ["lb_mmr", "current_mmr"], index=0)
                with c2:
                    bin_size = st.select_slider("Bin size", options=[250, 500, 750, 1000], value=500)
                with c3:
                    min_games = st.slider("Min games per player", min_value=0, max_value=int(max(0, df["games"].max() if "games" in df else 0)), value=0, step=10)
                with c4:
                    show_scatter = st.checkbox("Show scatter", value=True)

                mode = st.selectbox(
                    "Curve type (weighted by games)",
                    [
                        ("Weighted median (typical)", "wquant", 0.5),
                        ("Weighted 25th percentile (better-than-typical)", "wquant", 0.25),
                        ("Weighted 10th percentile (\"requirement\" feel)", "wquant", 0.10),
                        ("Weighted mean", "wmean", None),
                    ],
                    format_func=lambda t: t[0],
                    index=0
                )

                mode_kind = mode[1]
                q = mode[2] if mode_kind == "wquant" else 0.5

                # Clean & clip
                d = df.copy()
                d["avg_place"] = pd.to_numeric(d["avg_place"], errors="coerce")
                d["games"] = pd.to_numeric(d["games"], errors="coerce")
                d["lb_mmr"] = pd.to_numeric(d["lb_mmr"], errors="coerce")
                d["current_mmr"] = pd.to_numeric(d["current_mmr"], errors="coerce")

                # Drop weird values
                d = d.dropna(subset=[x_choice, "avg_place", "games"])
                d = d[(d["avg_place"] >= 1) & (d["avg_place"] <= 8)]
                d = d[d["games"] > 0]

                bx, by = binned_weighted_curve(
                    d,
                    x_col=x_choice,
                    y_col="avg_place",
                    w_col="games",
                    bin_size=int(bin_size),
                    mode=mode_kind,
                    q=float(q) if mode_kind == "wquant" else 0.5,
                    min_games=int(min_games),
                )

                if len(bx) < 2:
                    st.warning("Too few bins after filtering. Lower min games or change bin size.")
                else:
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    fig.patch.set_facecolor("#0e0e0e")
                    ax.set_facecolor("#0e0e0e")

                    if show_scatter:
                        ax.scatter(d[x_choice].to_numpy(float), d["avg_place"].to_numpy(float), alpha=0.35)

                    ax.plot(bx, by, linewidth=3)

                    ax.set_xlabel(f"{x_choice}")
                    ax.set_ylabel("Avg Place")
                    ax.grid(True, alpha=0.2)
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    st.pyplot(fig)

                    # Summary metrics in the high end
                    hi_cut = np.percentile(d[x_choice].to_numpy(float), 90)
                    hi = d[d[x_choice] >= hi_cut]
                    if len(hi) >= 5:
                        st.caption(f"High-end (>= 90th percentile of {x_choice} ≈ {hi_cut:.0f}): "
                                   f"mean avg_place = {hi['avg_place'].mean():.2f} (players={len(hi)})")

                    with st.expander("Data preview"):
                        st.dataframe(d.sort_values(x_choice, ascending=False).head(200), use_container_width=True)

        except Exception as e:
            st.error(str(e))
'''
txt = txt[:m.start()] + new_block + "\n" + txt[m.end():]

# Remove unused leaderboard fetch functions? keep for later; fine.

path.write_text(txt, encoding="utf-8")
str(path)