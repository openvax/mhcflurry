"""Plot the full-ensemble minibatch sweep summary.

Reads the ``sweep_summary.csv`` produced by
``full_ensemble_minibatch_sweep.sh`` and emits, into ``--out``:

* ``per_variable/<col>_{linlin,loglog}.png`` -- one chart per non-constant
  numeric column, x-axis = minibatch_size, y-axis = the variable.
* ``scatter/<x>_vs_<y>_{linlin,loglog}.png`` -- a scatter plot for every
  pair of non-constant numeric columns. Dots are colored along the
  viridis gradient by minibatch size, with a colorbar.
* ``index.html`` -- an offline gallery linking everything above so a
  reviewer can scroll through the whole sweep on one page.

Constant columns (``n_rows``, ``n_hits``, ``n_alleles_reported`` --
anything where every row has the same value) are dropped because they
yield uninformative flat lines. Each chart is rendered in just two
axis-scale combos: linear-linear and log-log.

Usage:
    python plot_minibatch_sweep.py \
        --sweep-csv path/to/sweep_summary.csv \
        --out plots/
"""
import argparse
import itertools
import os
import sys

import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator

try:
    # adjustText iteratively repositions text to avoid overlapping any
    # dots or other labels. Optional; we fall back to fixed offsets.
    from adjustText import adjust_text as _adjust_text
except ImportError:
    _adjust_text = None


# Columns excluded from plotting because they are not values we want
# to compare. ``minibatch`` is the primary axis. Constants are dropped
# automatically downstream.
NON_PLOTTABLE = {"minibatch"}

# Visual defaults tuned for the small-N sweep summaries.
FIGSIZE = (9.5, 6.2)
DPI = 130
DOT_SIZE = 160
DOT_EDGE = "#222"
DOT_EDGE_WIDTH = 0.8
LINE_COLOR = "#444"
LINE_ALPHA = 0.45
LABEL_PAD = 12
TITLE_PAD = 16
GRID_KW = dict(which="both", linestyle=":", alpha=0.4, linewidth=0.8)


def _setup_style():
    """Apply a clean, modern matplotlib style for all figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#444",
        "axes.linewidth":   0.9,
        "axes.titlesize":   14,
        "axes.titleweight": "bold",
        "axes.labelsize":   12,
        "axes.labelweight": "regular",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "xtick.color":      "#333",
        "ytick.color":      "#333",
        "font.family":      "DejaVu Sans",
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.25,
    })


def _is_log_safe(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.notna().all() and (s > 0).all()


# Minimum span (max/min) for a log scale to add visual information.
# Below this, log just stretches the axis linearly and the resulting
# plot is indistinguishable from lin-lin.
_LOG_SPAN_THRESHOLD = 1.5


def _wide_enough_for_log(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    if len(s) < 2:
        return False
    return float(s.max()) / float(s.min()) >= _LOG_SPAN_THRESHOLD


def _is_constant(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.nunique() <= 1


def _plain_formatter():
    fmt = ScalarFormatter(useMathText=False)
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    return fmt


def _force_plain_ticks(ax, x_scale, y_scale):
    if x_scale == "log":
        ax.xaxis.set_major_formatter(_plain_formatter())
        ax.xaxis.set_minor_formatter(_plain_formatter())
    if y_scale == "log":
        ax.yaxis.set_major_formatter(_plain_formatter())
        ax.yaxis.set_minor_formatter(_plain_formatter())


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # constrained_layout (set on the figure) handles colorbar padding
    # automatically; explicit tight_layout would fight with it.
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _mb_norm(mb_values, log=True):
    """Normalize minibatch values to a colormap range. Log-norm by
    default since mb sweeps are usually geometric."""
    vmin, vmax = float(min(mb_values)), float(max(mb_values))
    if log and vmin > 0:
        return LogNorm(vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def _add_mb_colorbar(fig, ax, mb_values, norm):
    """Append a colorbar showing the minibatch gradient and pin its
    ticks to the actual mb values in the sweep."""
    sm = ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("minibatch_size", fontsize=11)
    ticks = sorted(set(int(v) for v in mb_values))
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([str(t) for t in ticks])
    return cbar


def _annotate_points(ax, xs, ys, labels):
    """Place mb labels next to each dot, then (if available) iteratively
    nudge them to avoid overlapping each other and the dots. Falls back
    to a fixed NE offset if adjustText isn't installed."""
    texts = []
    for xi, yi, lbl in zip(xs, ys, labels):
        if pd.notna(xi) and pd.notna(yi):
            t = ax.text(
                xi, yi, str(lbl),
                fontsize=8.5,
                color="#222",
                ha="left", va="bottom",
            )
            texts.append(t)
    if not texts:
        return

    if _adjust_text is not None:
        # Treat the data points as obstacles too so labels don't sit on
        # top of any dot. expand_text/expand_points pad bounding boxes
        # so adjacent labels keep a small gap.
        valid = [(x, y) for x, y in zip(xs, ys)
                 if pd.notna(x) and pd.notna(y)]
        x_arr = [v[0] for v in valid]
        y_arr = [v[1] for v in valid]
        _adjust_text(
            texts,
            x=x_arr, y=y_arr, ax=ax,
            expand=(1.2, 1.4),
            arrowprops=dict(arrowstyle="-", color="#888",
                            lw=0.6, alpha=0.6),
            time_lim=2.0,
            max_move=20,
        )
    else:
        # Static NE offset; readers will sometimes see overlap, but
        # without adjustText there's no automated fix.
        for t, xi, yi in zip(texts, xs, ys):
            t.set_position((xi, yi))
            t.set_x(xi)
            t.set_y(yi)


def _filter_columns(df):
    """Drop columns that are non-numeric, constant, or in NON_PLOTTABLE."""
    keep = []
    for c in df.columns:
        if c in NON_PLOTTABLE:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 2:
            continue
        if _is_constant(s):
            continue
        keep.append(c)
    return keep


def plot_per_variable(df, out_dir):
    out_dir = os.path.join(out_dir, "per_variable")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    x = df["minibatch"].astype(float)
    mb_values = sorted(int(v) for v in df["minibatch"].unique())
    norm = _mb_norm(mb_values, log=True)
    cmap = plt.get_cmap("viridis")
    point_colors = [cmap(norm(v)) for v in x]

    cols = _filter_columns(df)
    for col in cols:
        y = pd.to_numeric(df[col], errors="coerce")
        for x_scale, y_scale in (("linear", "linear"), ("log", "log")):
            if x_scale == "log" and (not _is_log_safe(x)
                                     or not _wide_enough_for_log(x)):
                continue
            if y_scale == "log" and (not _is_log_safe(y)
                                     or not _wide_enough_for_log(y)):
                continue

            fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
            ax.plot(x, y, color=LINE_COLOR, linewidth=1.4,
                    alpha=LINE_ALPHA, zorder=1)
            ax.scatter(x, y, c=point_colors, s=DOT_SIZE,
                       edgecolor=DOT_EDGE, linewidth=DOT_EDGE_WIDTH,
                       zorder=3)
            _annotate_points(ax, x, y, [f"mb={int(v)}" for v in x])

            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            _force_plain_ticks(ax, x_scale, y_scale)
            ax.xaxis.set_major_locator(FixedLocator(mb_values))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel("minibatch_size", labelpad=LABEL_PAD)
            ax.set_ylabel(col, labelpad=LABEL_PAD)
            ax.set_title(f"{col}  vs  minibatch_size", pad=TITLE_PAD)
            ax.text(
                0.99, 1.02,
                f"axes: {x_scale}-{y_scale}",
                ha="right", va="bottom",
                transform=ax.transAxes,
                fontsize=9, color="#666",
            )
            ax.grid(True, **GRID_KW)

            tag = "loglog" if x_scale == "log" else "linlin"
            path = os.path.join(out_dir, f"{col}__{tag}.png")
            _save(fig, path)
            written.append(path)
    return written


def plot_pairs(df, out_dir):
    out_dir = os.path.join(out_dir, "scatter")
    os.makedirs(out_dir, exist_ok=True)
    cols = _filter_columns(df)
    written = []
    mb_values = sorted(int(v) for v in df["minibatch"].unique())
    norm = _mb_norm(mb_values, log=True)
    cmap = plt.get_cmap("viridis")
    point_colors = [cmap(norm(v)) for v in df["minibatch"].astype(float)]

    for x_col, y_col in itertools.combinations(cols, 2):
        xv = pd.to_numeric(df[x_col], errors="coerce")
        yv = pd.to_numeric(df[y_col], errors="coerce")
        for x_scale, y_scale in (("linear", "linear"), ("log", "log")):
            if x_scale == "log" and (not _is_log_safe(xv)
                                     or not _wide_enough_for_log(xv)):
                continue
            if y_scale == "log" and (not _is_log_safe(yv)
                                     or not _wide_enough_for_log(yv)):
                continue

            fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
            ax.scatter(xv, yv, c=point_colors, s=DOT_SIZE,
                       edgecolor=DOT_EDGE, linewidth=DOT_EDGE_WIDTH,
                       zorder=3)
            _annotate_points(
                ax, xv, yv,
                [f"mb={int(m)}" for m in df["minibatch"]],
            )

            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            _force_plain_ticks(ax, x_scale, y_scale)
            ax.set_xlabel(x_col, labelpad=LABEL_PAD)
            ax.set_ylabel(y_col, labelpad=LABEL_PAD)
            ax.set_title(f"{y_col}  vs  {x_col}", pad=TITLE_PAD)
            ax.text(
                0.99, 1.02,
                f"axes: {x_scale}-{y_scale}",
                ha="right", va="bottom",
                transform=ax.transAxes,
                fontsize=9, color="#666",
            )
            ax.grid(True, **GRID_KW)
            _add_mb_colorbar(fig, ax, mb_values, norm)

            tag = "loglog" if x_scale == "log" else "linlin"
            path = os.path.join(out_dir, f"{x_col}__VS__{y_col}__{tag}.png")
            _save(fig, path)
            written.append(path)
    return written


def write_index(out_dir, per_var_paths, scatter_paths):
    rel = lambda p: os.path.relpath(p, out_dir)
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Minibatch sweep</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;",
        "  margin:32px;background:#fafafa;color:#222}",
        "h1{font-weight:600;letter-spacing:-0.01em;margin-bottom:8px}",
        "h2{margin-top:48px;border-bottom:1px solid #ddd;padding-bottom:6px;",
        "  font-weight:500}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));",
        "  gap:16px;margin-top:16px}",
        "figure{margin:0;padding:6px;background:white;border:1px solid #e5e5e5;",
        "  border-radius:6px;box-shadow:0 1px 2px rgba(0,0,0,0.04)}",
        "img{width:100%;display:block;border-radius:4px}",
        "figcaption{font-size:11px;color:#666;margin-top:6px;text-align:center;",
        "  font-family:ui-monospace,Menlo,monospace}",
        "</style>",
        "<h1>Minibatch sweep — full ensemble</h1>",
    ]
    html.append("<h2>Per-variable (vs minibatch_size)</h2><div class='grid'>")
    for p in sorted(per_var_paths):
        html.append(
            f"<figure><img src='{rel(p)}'/>"
            f"<figcaption>{os.path.basename(p)}</figcaption></figure>"
        )
    html.append("</div>")
    html.append("<h2>Pairwise scatter (color = minibatch_size)</h2>"
                "<div class='grid'>")
    for p in sorted(scatter_paths):
        html.append(
            f"<figure><img src='{rel(p)}'/>"
            f"<figcaption>{os.path.basename(p)}</figcaption></figure>"
        )
    html.append("</div>")
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
    _setup_style()

    cols = _filter_columns(df)
    dropped = [c for c in df.columns
               if c not in NON_PLOTTABLE and c not in cols]
    if dropped:
        print(f"dropped uninformative cols: {dropped}")

    per_var = plot_per_variable(df, args.out)
    scatter = plot_pairs(df, args.out)
    write_index(args.out, per_var, scatter)

    print(f"Wrote {len(per_var)} per-variable charts and {len(scatter)} "
          f"scatter plots to {args.out}")
    print(f"Open {os.path.join(args.out, 'index.html')} in a browser.")


if __name__ == "__main__":
    main()
