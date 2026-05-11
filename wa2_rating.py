import numpy as np
import pandas as pd
import streamlit as st


def interp_with_extrap(x, xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(xs) < 2:
        return float(ys[0]) if len(ys) else float("nan")

    x = float(x)
    if x <= xs[0]:
        m = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return float(ys[0] + (x - xs[0]) * m)
    if x >= xs[-1]:
        m = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return float(ys[-1] + (x - xs[-1]) * m)

    return float(np.interp(x, xs, ys))


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


def binned_weighted_curve(
    df,
    x_col,
    y_col="avg_place",
    w_col="games",
    bin_size=500,
    mode="wquant",
    q=0.5,
    min_games=0,
):
    d = df[[x_col, y_col, w_col]].dropna().copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d[w_col] = pd.to_numeric(d[w_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col, w_col])
    d = d[np.isfinite(d[x_col]) & np.isfinite(d[y_col]) & np.isfinite(d[w_col])]
    d = d[(d[y_col] >= 1) & (d[y_col] <= 8)]
    d = d[d[w_col] > 0]
    if min_games > 0:
        d = d[d[w_col] >= min_games]
    if d.empty:
        return np.array([]), np.array([]), d

    d["bin"] = (d[x_col] // bin_size) * bin_size
    xs, ys = [], []
    for b, g in d.groupby("bin"):
        y = g[y_col].to_numpy(float)
        w = g[w_col].to_numpy(float)
        val = float((w * y).sum() / w.sum()) if mode == "wmean" else weighted_quantile(y, w, q)
        if np.isfinite(val):
            xs.append(float(b))
            ys.append(val)

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    order = np.argsort(xs)
    return xs[order], ys[order], d


@st.cache_data(show_spinner=False)
def load_rating_curve(
    csv_path_str,
    file_mtime,
    x_choice="current_mmr",
    bin_size=1000,
    mode_kind="wmean",
    q=0.5,
    min_games=300,
):
    df = pd.read_csv(csv_path_str)
    bx, by, _ = binned_weighted_curve(
        df,
        x_col=x_choice,
        y_col="avg_place",
        w_col="games",
        bin_size=int(bin_size),
        mode=mode_kind,
        q=float(q) if mode_kind == "wquant" else 0.5,
        min_games=int(min_games),
    )
    return bx, by
