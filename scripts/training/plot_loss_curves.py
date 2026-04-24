"""Plot loss curves for every network in a training run + highlight
which ones were picked by the selection step.

The fit_info arrays (``loss``, ``val_loss``, plus the optional Phase-0
timing breakdowns) are serialized inside each row's ``config_json`` in
the trained models' ``manifest.csv``. Before selection, that manifest
has ~140 rows (one per candidate); after selection it has ~11 rows —
the ones that entered the final ensemble.

Usage — aggregate curves over the full candidate pool, colored by
selection status:

    python plot_loss_curves.py \\
        --unselected-dir results/new_run/models.unselected.combined \\
        --selected-dir results/new_run/models.combined \\
        --out results/plots/loss_curves

Usage — just the selected ensemble:

    python plot_loss_curves.py \\
        --selected-dir results/new_run/models.combined \\
        --out results/plots/loss_curves

Produces (in ``--out``):
  - ``loss_curves_by_model.png`` — per-model train+val curves. Non-
    selected models in gray (if ``--unselected-dir`` is provided),
    selected in color.
  - ``loss_curves_by_arch.png`` — curves colored by layer_sizes.
  - ``per_fold_summary.png`` — one panel per fold showing val_loss
    convergence of selected vs non-selected.
  - ``summary.csv`` — tabular summary of final val_loss per model.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import defaultdict

import numpy
import pandas


def _parse_config_json(raw):
    """Handle both JSON and Python-repr config_json encodings."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(raw)


def _load_manifest_curves(manifest_path):
    """Load a manifest.csv and pull per-model fit_info curves.

    Returns a list of dicts: {model_name, arch, layer_sizes, l1, fold,
    loss_history, val_loss_history, final_val_loss, phase}.
    """
    df = pandas.read_csv(manifest_path)
    rows = []
    for _, r in df.iterrows():
        cfg = _parse_config_json(r.config_json)
        hp = cfg.get("hyperparameters", cfg)
        fit_info = cfg.get("fit_info") or []
        # Each fit_info record is one .fit() invocation; pan-allele has
        # a pretrain pass then a finetune pass. Both are useful to plot.
        model_row = {
            "model_name": r.model_name,
            "layer_sizes": tuple(hp.get("layer_sizes", [])),
            "l1": hp.get("dense_layer_l1_regularization"),
            "fold": None,
            "arch_num": None,
            "replicate": None,
            "phase_curves": [],
        }
        for fit_rec in fit_info:
            if not isinstance(fit_rec, dict):
                continue
            tinfo = fit_rec.get("training_info") or {}
            model_row["fold"] = tinfo.get("fold_num", model_row["fold"])
            model_row["arch_num"] = tinfo.get(
                "architecture_num", model_row["arch_num"])
            model_row["replicate"] = tinfo.get(
                "replicate_num", model_row["replicate"])
            losses = fit_rec.get("loss") or []
            val_losses = fit_rec.get("val_loss") or []
            model_row["phase_curves"].append({
                "phase": tinfo.get("phase", "unknown"),
                "loss": list(losses),
                "val_loss": list(val_losses),
                "final_val_loss": (
                    float(val_losses[-1]) if val_losses else float("nan")
                ),
            })
        rows.append(model_row)
    return rows


def _plot_all_curves(selected_names, all_models, out_path, title_suffix=""):
    """Plot train/val curves for every model, gray non-selected + colored selected.

    Non-selected models get a low-alpha gray; selected get distinguishable
    colors so the eye can follow them through the phase transitions.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = plt.get_cmap("tab10")
    selected_color_idx = 0
    for m in all_models:
        is_selected = m["model_name"] in selected_names
        label = None
        if is_selected:
            color = cmap(selected_color_idx % 10)
            alpha = 0.9
            lw = 1.4
            label = f"{m['layer_sizes']} f{m['fold']}"
            selected_color_idx += 1
        else:
            color = "#666666"
            alpha = 0.12
            lw = 0.7
        # Concatenate pretrain + finetune curves with a vertical
        # line at the handoff so phase transitions are visible.
        loss_curve = []
        val_curve = []
        phase_boundaries = []
        for ph in m["phase_curves"]:
            loss_curve.extend(ph["loss"])
            val_curve.extend(ph["val_loss"])
            phase_boundaries.append(len(loss_curve))
        x = numpy.arange(1, len(loss_curve) + 1)
        if len(loss_curve):
            axes[0].plot(
                x, loss_curve, color=color, alpha=alpha, lw=lw, label=label,
            )
        if len(val_curve):
            xv = numpy.arange(1, len(val_curve) + 1)
            axes[1].plot(
                xv, val_curve, color=color, alpha=alpha, lw=lw,
                label=label if not loss_curve else None,
            )

    axes[0].set_title("Train loss")
    axes[0].set_xlabel("epoch (pretrain + finetune concat)")
    axes[0].set_ylabel("loss")
    axes[0].set_yscale("log")
    axes[1].set_title("Val loss")
    axes[1].set_xlabel("epoch (pretrain + finetune concat)")
    axes[1].set_ylabel("val loss")
    axes[1].set_yscale("log")
    if selected_color_idx <= 12:
        # Only show legend when it won't swallow the plot. With >12
        # selected models the legend is more noise than signal.
        axes[1].legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle(
        f"Loss curves ({selected_color_idx} selected of {len(all_models)}) "
        f"{title_suffix}".strip(),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_by_arch(all_models, out_path):
    """Color curves by layer_sizes — groups see how different archs
    behave during training, independent of selection outcome."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    arch_groups = defaultdict(list)
    for m in all_models:
        arch_groups[m["layer_sizes"]].append(m)

    # Consistent color per arch; stable ordering by total params
    def arch_size(ls):
        # Rough param estimate for ordering — 1092 input × layer_sizes
        if not ls:
            return 0
        total = 1092 * ls[0]
        for a, b in zip(ls, ls[1:]):
            total += a * b
        return total
    sorted_archs = sorted(arch_groups.keys(), key=arch_size)
    cmap = plt.get_cmap("viridis")
    for i, arch in enumerate(sorted_archs):
        color = cmap(i / max(len(sorted_archs) - 1, 1))
        for j, m in enumerate(arch_groups[arch]):
            loss_curve = []
            val_curve = []
            for ph in m["phase_curves"]:
                loss_curve.extend(ph["loss"])
                val_curve.extend(ph["val_loss"])
            label = str(arch) if j == 0 else None
            if len(loss_curve):
                axes[0].plot(
                    numpy.arange(1, len(loss_curve) + 1), loss_curve,
                    color=color, alpha=0.4, lw=0.8, label=label,
                )
            if len(val_curve):
                axes[1].plot(
                    numpy.arange(1, len(val_curve) + 1), val_curve,
                    color=color, alpha=0.4, lw=0.8,
                )

    axes[0].set_title("Train loss by arch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[1].set_title("Val loss by arch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val loss")
    axes[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_per_fold(selected_names, all_models, out_path):
    """One panel per fold; selected vs non-selected val_loss curves."""
    import matplotlib.pyplot as plt
    folds = sorted({m["fold"] for m in all_models if m["fold"] is not None})
    if not folds:
        return
    fig, axes = plt.subplots(
        1, len(folds), figsize=(5 * len(folds), 5), sharey=True,
    )
    if len(folds) == 1:
        axes = [axes]
    for ax, fold in zip(axes, folds):
        fold_models = [m for m in all_models if m["fold"] == fold]
        for m in fold_models:
            val_curve = []
            for ph in m["phase_curves"]:
                val_curve.extend(ph["val_loss"])
            if not val_curve:
                continue
            if m["model_name"] in selected_names:
                color = "#d62728"
                alpha = 0.95
                lw = 1.4
            else:
                color = "#666666"
                alpha = 0.18
                lw = 0.7
            ax.plot(
                numpy.arange(1, len(val_curve) + 1), val_curve,
                color=color, alpha=alpha, lw=lw,
            )
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
    axes[0].set_ylabel("val loss")
    fig.suptitle(
        "Val loss per fold — red=selected, gray=not-selected", fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--unselected-dir",
        help=(
            "Path to models.unselected.combined/ (has the full candidate "
            "pool). When provided, all candidates are plotted in gray + "
            "the selected ones in color."
        ),
    )
    p.add_argument(
        "--selected-dir", required=True,
        help="Path to models.combined/ (the selected ensemble)",
    )
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    selected = _load_manifest_curves(
        os.path.join(args.selected_dir, "manifest.csv"),
    )
    selected_names = {m["model_name"] for m in selected}

    if args.unselected_dir:
        all_models = _load_manifest_curves(
            os.path.join(args.unselected_dir, "manifest.csv"),
        )
    else:
        all_models = selected

    print(
        f"Loaded {len(all_models)} candidate model(s); "
        f"{len(selected_names)} selected."
    )

    _plot_all_curves(
        selected_names, all_models,
        os.path.join(args.out, "loss_curves_by_model.png"),
        title_suffix="(selected in color)",
    )
    _plot_by_arch(
        all_models,
        os.path.join(args.out, "loss_curves_by_arch.png"),
    )
    _plot_per_fold(
        selected_names, all_models,
        os.path.join(args.out, "per_fold_summary.png"),
    )

    # Summary CSV
    rows = []
    for m in all_models:
        for ph in m["phase_curves"]:
            rows.append({
                "model_name": m["model_name"],
                "layer_sizes": str(m["layer_sizes"]),
                "l1": m["l1"],
                "fold": m["fold"],
                "replicate": m["replicate"],
                "selected": m["model_name"] in selected_names,
                "phase": ph["phase"],
                "n_epochs": len(ph["loss"]),
                "final_loss": (
                    float(ph["loss"][-1]) if ph["loss"] else float("nan")
                ),
                "final_val_loss": ph["final_val_loss"],
            })
    summary_df = pandas.DataFrame(rows)
    summary_df.to_csv(
        os.path.join(args.out, "summary.csv"), index=False,
    )

    print(f"Wrote plots + summary to {args.out}/")


if __name__ == "__main__":
    main()
