"""Evaluate a trained presentation predictor against the public release.

Uses the ``data_evaluation`` multiallelic hit/decoy benchmark and compares a
new ``Class1PresentationPredictor`` to the installed public presentation
predictor. Writes:

  - compressed per-row predictions for with-flanks and without-flanks modes
  - per-sample metrics
  - per-peptide-length micro/macro metrics
  - summary.json and summary_table.csv
  - ROC/PR/scatter/delta plots

Higher presentation score is better. Lower presentation percentile is better,
so percentile metrics are computed on ``-presentation_percentile``.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy
import pandas
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


_T0 = time.time()
METRICS = ("roc_auc", "pr_auc", "ppv_at_n")
SCORE_KINDS = ("presentation_score", "presentation_percentile")


def _stamp(msg):
    print(f"[+{time.time() - _T0:7.1f}s] {msg}", flush=True)


def _detect_gpu_count():
    """Probe GPU count in a subprocess so the parent does not init CUDA."""
    import subprocess
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.cuda.device_count() if "
            "torch.cuda.is_available() else 0)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        return int(out.stdout.strip())
    except ValueError:
        return 0


def _resolve_torchinductor_compile_threads(num_jobs):
    if os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") != "auto":
        return
    cap = int(os.environ.get(
        "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_CAP", "16"))
    cpu_count = os.cpu_count() or 1
    threads = max(1, min(cap, cpu_count // max(int(num_jobs), 1)))
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(threads)
    os.environ["MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO"] = "1"
    _stamp(
        f"resolved TORCHINDUCTOR_COMPILE_THREADS=auto -> {threads} "
        f"(cpu={cpu_count}, num_jobs={num_jobs}, cap={cap})"
    )


def _benchmark_files(data_dir, limit_files=None):
    files = sorted(glob.glob(os.path.join(
        data_dir,
        "benchmark.multiallelic.train_excluded.*.csv.bz2",
    )))
    if limit_files is not None:
        files = files[:int(limit_files)]
    return files


def _load_benchmark(data_dir, limit_files=None):
    files = _benchmark_files(data_dir, limit_files=limit_files)
    if not files:
        raise ValueError(
            "No benchmark files found under %s matching "
            "benchmark.multiallelic.train_excluded.*.csv.bz2" % data_dir
        )
    _stamp(f"loading {len(files)} multiallelic benchmark files")
    dfs = []
    for i, path in enumerate(files):
        df = pandas.read_csv(path)
        df["source_file"] = os.path.basename(path)
        dfs.append(df)
        if (i + 1) % 25 == 0:
            _stamp(f"  loaded {i + 1}/{len(files)} files")
    result = pandas.concat(dfs, ignore_index=True)
    required = {"peptide", "sample_id", "hla", "hit"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError("Benchmark data missing columns: %s" % missing)
    result = result.dropna(subset=["peptide", "sample_id", "hla", "hit"]).copy()
    result["hit"] = result["hit"].astype(int)
    result["peptide_len"] = result.peptide.str.len()
    result = result[
        (result.peptide_len >= 8) &
        (result.peptide_len <= 15)
    ].reset_index(drop=True)
    for col in ("n_flank", "c_flank"):
        if col not in result:
            result[col] = ""
        result[col] = result[col].fillna("")
    _stamp(
        f"benchmark rows after filtering: {len(result):,}; "
        f"samples={result.sample_id.nunique():,}; "
        f"hits={int(result.hit.sum()):,}"
    )
    return result


def _predict_chunk_worker(args):
    predictor_dir, rows, mode, gpu_id = args
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import pandas as _pandas
    from mhcflurry.common import configure_pytorch
    from mhcflurry import Class1PresentationPredictor

    configure_pytorch(backend="gpu" if gpu_id is not None else "auto")
    predictor = Class1PresentationPredictor.load(predictor_dir)

    df = _pandas.DataFrame(rows)
    sample_to_alleles = (
        df.drop_duplicates("sample_id")
        .set_index("sample_id")
        .hla.str.split()
        .to_dict()
    )
    kwargs = {
        "peptides": df.peptide.values,
        "sample_names": df.sample_id.values,
        "alleles": sample_to_alleles,
        "verbose": 0,
        "throw": False,
    }
    if mode == "with_flanks":
        kwargs["n_flanks"] = df.n_flank.values
        kwargs["c_flanks"] = df.c_flank.values
    elif mode != "without_flanks":
        raise ValueError("Unexpected mode: %s" % mode)

    pred = predictor.predict(**kwargs)
    if len(pred) != len(df):
        raise ValueError(
            "Predictor returned %d rows for %d inputs" % (len(pred), len(df))
        )

    cols = [
        "presentation_score",
        "presentation_percentile",
        "affinity",
        "processing_score",
    ]
    out = _pandas.DataFrame(index=df.index)
    for col in cols:
        out[col] = pred[col].values if col in pred else numpy.nan
    return out


def _parallel_predict(predictor_dir, df, mode, n_gpus, label):
    import multiprocessing as mp

    if n_gpus <= 1 or len(df) < 100_000:
        gpu_id = 0 if n_gpus == 1 else None
        _stamp(f"predicting {label} {mode} in one process (gpu={gpu_id})")
        return _predict_chunk_worker(
            (predictor_dir, df.to_dict("list"), mode, gpu_id)
        ).reset_index(drop=True)

    _stamp(f"predicting {label} {mode} across {n_gpus} GPU workers")
    idx_chunks = numpy.array_split(numpy.arange(len(df)), n_gpus)
    tasks = []
    for gpu_id, idxs in enumerate(idx_chunks):
        tasks.append((
            predictor_dir,
            df.iloc[idxs].to_dict("list"),
            mode,
            gpu_id,
        ))
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_gpus) as pool:
        pieces = pool.map(_predict_chunk_worker, tasks)
    out = pandas.concat(pieces, axis=0)
    return out.reset_index(drop=True)


def ppv_at_n(y_true, y_score, n):
    order = numpy.argsort(-y_score)
    top = order[:n]
    return float(y_true[top].sum()) / float(n) if n > 0 else numpy.nan


def metrics(y_true, y_score):
    y_true = numpy.asarray(y_true)
    y_score = numpy.asarray(y_score)
    mask = ~numpy.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {
            "n": int(len(y_true)),
            "n_pos": n_pos,
            "roc_auc": numpy.nan,
            "pr_auc": numpy.nan,
            "ppv_at_n": numpy.nan,
        }
    return {
        "n": int(len(y_true)),
        "n_pos": n_pos,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ppv_at_n": ppv_at_n(y_true, y_score, n_pos),
    }


def _score_values(df, prefix, score_kind):
    if score_kind == "presentation_score":
        return df[f"{prefix}_presentation_score"].values
    if score_kind == "presentation_percentile":
        return -df[f"{prefix}_presentation_percentile"].values
    raise ValueError("Unexpected score kind: %s" % score_kind)


def _metric_rows_by_group(df, group_cols, score_kind):
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        m_new = metrics(group.hit.values, _score_values(group, "new", score_kind))
        m_pub = metrics(group.hit.values, _score_values(group, "public", score_kind))
        row.update({"n": m_new["n"], "n_pos": m_new["n_pos"]})
        for metric in METRICS:
            row[f"new_{metric}"] = m_new[metric]
            row[f"public_{metric}"] = m_pub[metric]
            row[f"{metric}_diff"] = m_new[metric] - m_pub[metric]
        rows.append(row)
    return pandas.DataFrame(rows)


def _per_length_metrics(df, per_sample, score_kind):
    rows = []
    per_length_per_sample = []
    for length, group in df.groupby("peptide_len"):
        m_new = metrics(group.hit.values, _score_values(group, "new", score_kind))
        m_pub = metrics(group.hit.values, _score_values(group, "public", score_kind))
        row = {
            "length": int(length),
            "n": m_new["n"],
            "n_pos": m_new["n_pos"],
        }
        sub_sample = _metric_rows_by_group(group, ["sample_id"], score_kind)
        sub_sample["length"] = int(length)
        per_length_per_sample.append(sub_sample)
        for metric in METRICS:
            row[f"new_micro_{metric}"] = m_new[metric]
            row[f"public_micro_{metric}"] = m_pub[metric]
            row[f"micro_{metric}_diff"] = m_new[metric] - m_pub[metric]
            row[f"new_macro_{metric}"] = float(
                numpy.nanmean(sub_sample[f"new_{metric}"])
            )
            row[f"public_macro_{metric}"] = float(
                numpy.nanmean(sub_sample[f"public_{metric}"])
            )
            row[f"macro_{metric}_diff"] = (
                row[f"new_macro_{metric}"] - row[f"public_macro_{metric}"]
            )
        rows.append(row)
    per_length = pandas.DataFrame(rows).sort_values("length")
    if per_length_per_sample:
        per_length_per_sample = pandas.concat(
            per_length_per_sample, ignore_index=True
        )
    else:
        per_length_per_sample = pandas.DataFrame()
    return per_length, per_length_per_sample


def _summarize(df, per_sample, per_length, mode, score_kind):
    m_new = metrics(df.hit.values, _score_values(df, "new", score_kind))
    m_pub = metrics(df.hit.values, _score_values(df, "public", score_kind))
    return {
        "mode": mode,
        "score_kind": score_kind,
        "n_rows": int(m_new["n"]),
        "n_hits": int(m_new["n_pos"]),
        "n_samples_reported": int(len(per_sample)),
        "micro_pooled": {"new": m_new, "public": m_pub},
        "macro_mean_over_samples": {
            metric: {
                "new": float(numpy.nanmean(per_sample[f"new_{metric}"])),
                "public": float(numpy.nanmean(per_sample[f"public_{metric}"])),
            } for metric in METRICS
        },
        "sample_count": {
            f"new_better_{metric}": int((per_sample[f"{metric}_diff"] > 0).sum())
            for metric in METRICS
        } | {
            f"public_better_{metric}": int(
                (per_sample[f"{metric}_diff"] < 0).sum()
            )
            for metric in METRICS
        },
        "per_length": per_length.to_dict(orient="records"),
    }


def _save_plots(scored_by_mode, summary_rows, out_dir, max_points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for mode, df in scored_by_mode.items():
        for score_kind in SCORE_KINDS:
            y = df.hit.values
            new_score = _score_values(df, "new", score_kind)
            pub_score = _score_values(df, "public", score_kind)

            fig, ax = plt.subplots(figsize=(6, 5))
            for label, values in (("new", new_score), ("public", pub_score)):
                mask = ~numpy.isnan(values)
                fpr, tpr, _ = roc_curve(y[mask], values[mask])
                auc = roc_auc_score(y[mask], values[mask])
                ax.plot(fpr, tpr, label=f"{label} AUC={auc:.3f}")
            ax.plot([0, 1], [0, 1], color="0.6", linewidth=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(f"{mode} {score_kind} ROC")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"roc_{mode}_{score_kind}.png"))
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 5))
            for label, values in (("new", new_score), ("public", pub_score)):
                mask = ~numpy.isnan(values)
                precision, recall, _ = precision_recall_curve(y[mask], values[mask])
                ap = average_precision_score(y[mask], values[mask])
                ax.plot(recall, precision, label=f"{label} AP={ap:.3f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{mode} {score_kind} PR")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"pr_{mode}_{score_kind}.png"))
            plt.close(fig)

            mask = ~(numpy.isnan(new_score) | numpy.isnan(pub_score))
            idx = numpy.flatnonzero(mask)
            if len(idx) > max_points:
                rng = numpy.random.default_rng(17)
                idx = rng.choice(idx, size=max_points, replace=False)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(pub_score[idx], new_score[idx], s=4, alpha=0.25)
            ax.set_xlabel("public")
            ax.set_ylabel("new")
            ax.set_title(f"{mode} {score_kind}: new vs public")
            fig.tight_layout()
            fig.savefig(os.path.join(
                plot_dir, f"scatter_{mode}_{score_kind}.png"
            ))
            plt.close(fig)

    summary_df = pandas.DataFrame(summary_rows)
    if not summary_df.empty:
        x_labels = [
            f"{row.mode}\n{row.score_kind.replace('presentation_', '')}"
            for row in summary_df.itertuples()
        ]
        x = numpy.arange(len(summary_df))
        width = 0.38
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x - width / 2, summary_df[f"new_macro_{metric}"], width, label="new")
            ax.bar(
                x + width / 2,
                summary_df[f"public_macro_{metric}"],
                width,
                label="public",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=30, ha="right")
            ax.set_ylabel(metric)
            ax.set_title(f"Macro mean over samples: {metric}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"macro_{metric}.png"))
            plt.close(fig)


def _summary_table_row(summary):
    row = {
        "mode": summary["mode"],
        "score_kind": summary["score_kind"],
        "n_rows": summary["n_rows"],
        "n_hits": summary["n_hits"],
        "n_samples_reported": summary["n_samples_reported"],
    }
    for metric in METRICS:
        row[f"new_micro_{metric}"] = summary["micro_pooled"]["new"][metric]
        row[f"public_micro_{metric}"] = summary["micro_pooled"]["public"][metric]
        row[f"micro_{metric}_diff"] = (
            row[f"new_micro_{metric}"] - row[f"public_micro_{metric}"]
        )
        row[f"new_macro_{metric}"] = (
            summary["macro_mean_over_samples"][metric]["new"]
        )
        row[f"public_macro_{metric}"] = (
            summary["macro_mean_over_samples"][metric]["public"]
        )
        row[f"macro_{metric}_diff"] = (
            row[f"new_macro_{metric}"] - row[f"public_macro_{metric}"]
        )
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new-models-dir", required=True)
    p.add_argument("--public-models-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["with_flanks", "without_flanks"],
        default=["with_flanks", "without_flanks"],
    )
    p.add_argument("--max-plot-points", type=int, default=100000)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n_gpus = _detect_gpu_count()
    _stamp(f"detected {n_gpus} GPUs")
    _resolve_torchinductor_compile_threads(num_jobs=max(n_gpus, 1))

    benchmark = _load_benchmark(args.data_dir, limit_files=args.limit_files)
    scored_by_mode = {}
    summaries = {}
    summary_rows = []

    for mode in args.modes:
        _stamp(f"=== mode: {mode} ===")
        scored = benchmark.copy()
        new_pred = _parallel_predict(
            args.new_models_dir, benchmark, mode, n_gpus, "new"
        )
        public_pred = _parallel_predict(
            args.public_models_dir, benchmark, mode, n_gpus, "public"
        )
        for prefix, pred in (("new", new_pred), ("public", public_pred)):
            for col in [
                "presentation_score",
                "presentation_percentile",
                "affinity",
                "processing_score",
            ]:
                scored[f"{prefix}_{col}"] = pred[col].values

        pred_path = os.path.join(args.out, f"predictions_{mode}.csv.bz2")
        scored.to_csv(pred_path, index=False)
        _stamp(f"wrote {pred_path}")
        scored_by_mode[mode] = scored
        summaries[mode] = {}

        for score_kind in SCORE_KINDS:
            per_sample = _metric_rows_by_group(
                scored,
                ["sample_id", "hla"],
                score_kind,
            ).sort_values("n", ascending=False)
            per_sample_path = os.path.join(
                args.out, f"per_sample_{mode}_{score_kind}.csv"
            )
            per_sample.to_csv(per_sample_path, index=False)
            _stamp(f"wrote {per_sample_path} ({len(per_sample)} samples)")

            per_length, per_length_per_sample = _per_length_metrics(
                scored,
                per_sample,
                score_kind,
            )
            per_length_path = os.path.join(
                args.out, f"per_length_{mode}_{score_kind}.csv"
            )
            per_length.to_csv(per_length_path, index=False)
            _stamp(f"wrote {per_length_path}")
            if not per_length_per_sample.empty:
                per_length_per_sample.to_csv(
                    os.path.join(
                        args.out,
                        f"per_length_per_sample_{mode}_{score_kind}.csv",
                    ),
                    index=False,
                )

            summary = _summarize(
                scored,
                per_sample,
                per_length,
                mode,
                score_kind,
            )
            summaries[mode][score_kind] = summary
            summary_rows.append(_summary_table_row(summary))

    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as fd:
        json.dump(summaries, fd, indent=2, sort_keys=True)
    _stamp(f"wrote {summary_path}")

    summary_table = pandas.DataFrame(summary_rows)
    summary_table_path = os.path.join(args.out, "summary_table.csv")
    summary_table.to_csv(summary_table_path, index=False)
    _stamp(f"wrote {summary_table_path}")

    _save_plots(
        scored_by_mode,
        summary_rows,
        args.out,
        max_points=args.max_plot_points,
    )
    _stamp(f"wrote plots under {os.path.join(args.out, 'plots')}")
    print()
    print("=== SUMMARY TABLE ===")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
