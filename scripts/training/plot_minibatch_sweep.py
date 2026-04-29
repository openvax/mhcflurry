"""Plot the full-ensemble minibatch sweep summary.

Reads the ``sweep_summary.csv`` produced by
``full_ensemble_minibatch_sweep.sh`` and emits, into ``--out``:

* ``per_variable/<col>_{linear,log}.png`` -- one chart per non-minibatch
  numeric column, x-axis = minibatch_size on linear and log scales,
  y-axis = the variable.
* ``scatter/<x>_vs_<y>_{linear,log,loglog}.png`` -- a scatter plot for
  every pair of numeric columns. Each pair is rendered three times so
  every axis-scale combination is available without re-running.
* ``index.html`` -- an offline gallery linking everything above so a
  reviewer can scroll through the whole sweep on one page.

Usage:
    python plot_minibatch_sweep.py \
        --sweep-csv path/to/sweep_summary.csv \
        --out plots/

If ``per_allele.csv`` files exist alongside each minibatch's
``eval_comparison/``, they're merged into a long-format frame so the
per-allele dotplots get a fourth axis (allele) -- only emitted when
``--per-allele-glob`` matches at least one file.
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# Columns whose values can sensibly be log-scaled (strictly positive).
NON_PLOTTABLE = {"minibatch"}


def _is_log_safe(series):
    """True if every value is finite and strictly positive."""
    s = pd.to_numeric(series, errors="coerce")
    return s.notna().all() and (s > 0).all()


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_per_variable(df, out_dir):
    """For every non-minibatch column, plot value vs minibatch on
    linear and log x-axes."""
    out_dir = os.path.join(out_dir, "per_variable")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    x = df["minibatch"].astype(float)
    cols = [c for c in df.columns if c not in NON_PLOTTABLE]
    for col in cols:
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().sum() < 2:
            continue
        for x_scale in ("linear", "log"):
            for y_scale in ("linear", "log"):
                if y_scale == "log" and not _is_log_safe(y):
                    continue
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(x, y, marker="o", linewidth=1.2)
                for xi, yi in zip(x, y):
                    if pd.notna(yi):
                        ax.annotate(
                            f"{int(xi)}",
                            (xi, yi),
                            textcoords="offset points",
                            xytext=(4, 4),
                            fontsize=7,
                        )
                ax.set_xscale(x_scale)
                ax.set_yscale(y_scale)
                ax.set_xlabel("minibatch_size")
                ax.set_ylabel(col)
                ax.set_title(f"{col} vs minibatch ({x_scale} x, {y_scale} y)")
                ax.grid(True, which="both", linestyle=":", alpha=0.5)
                fname = f"{col}__x{x_scale}_y{y_scale}.png"
                path = os.path.join(out_dir, fname)
                _save(fig, path)
                written.append(path)
    return written


def plot_pairs(df, out_dir):
    """Scatter every pair (x, y) of numeric columns. Three scale
    combos so every axis behavior is available."""
    out_dir = os.path.join(out_dir, "scatter")
    os.makedirs(out_dir, exist_ok=True)
    cols = [c for c in df.columns if c not in NON_PLOTTABLE]
    written = []
    for x_col, y_col in itertools.combinations(cols, 2):
        xv = pd.to_numeric(df[x_col], errors="coerce")
        yv = pd.to_numeric(df[y_col], errors="coerce")
        if xv.notna().sum() < 2 or yv.notna().sum() < 2:
            continue
        for x_scale, y_scale in [
            ("linear", "linear"),
            ("log",    "linear"),
            ("linear", "log"),
            ("log",    "log"),
        ]:
            if x_scale == "log" and not _is_log_safe(xv):
                continue
            if y_scale == "log" and not _is_log_safe(yv):
                continue
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter(xv, yv, c=df["minibatch"], cmap="viridis", s=80,
                       edgecolor="black", linewidth=0.5)
            for xi, yi, mi in zip(xv, yv, df["minibatch"]):
                if pd.notna(xi) and pd.notna(yi):
                    ax.annotate(
                        f"mb={int(mi)}",
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=7,
                    )
            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(
                f"{y_col} vs {x_col} (x={x_scale}, y={y_scale})"
            )
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            fname = f"{x_col}__VS__{y_col}__x{x_scale}_y{y_scale}.png"
            path = os.path.join(out_dir, fname)
            _save(fig, path)
            written.append(path)
    return written


def write_index(out_dir, per_var_paths, scatter_paths):
    """Tiny static HTML gallery -- groups per-variable charts and
    pair scatters so a reviewer can skim the whole sweep without
    opening a notebook."""
    rel = lambda p: os.path.relpath(p, out_dir)
    html = ["<!doctype html><meta charset='utf-8'><title>Minibatch sweep</title>"]
    html.append(
        "<style>body{font-family:sans-serif;margin:24px;}h2{margin-top:32px}"
        "img{max-width:520px;border:1px solid #ddd;margin:6px}</style>"
    )
    html.append("<h1>Minibatch sweep — full ensemble</h1>")
    html.append("<h2>Per-variable (vs minibatch)</h2>")
    for p in sorted(per_var_paths):
        html.append(
            f"<figure><img src='{rel(p)}'/>"
            f"<figcaption>{os.path.basename(p)}</figcaption></figure>"
        )
    html.append("<h2>Pairwise scatter (every axis-scale combo)</h2>")
    for p in sorted(scatter_paths):
        html.append(
            f"<figure><img src='{rel(p)}'/>"
            f"<figcaption>{os.path.basename(p)}</figcaption></figure>"
        )
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write("\n".join(html))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-csv", required=True,
                   help="Path to sweep_summary.csv")
    p.add_argument("--out", required=True, help="Output dir for plots")
    args = p.parse_args()

    df = pd.read_csv(args.sweep_csv)
    if "minibatch" not in df.columns:
        sys.exit("sweep_summary.csv missing 'minibatch' column")
    df = df.sort_values("minibatch").reset_index(drop=True)

    os.makedirs(args.out, exist_ok=True)

    per_var = plot_per_variable(df, args.out)
    scatter = plot_pairs(df, args.out)
    write_index(args.out, per_var, scatter)

    print(f"Wrote {len(per_var)} per-variable charts and {len(scatter)} "
          f"scatter plots to {args.out}")
    print(f"Open {os.path.join(args.out, 'index.html')} in a browser.")


if __name__ == "__main__":
    main()
