"""Compare the new pan-allele training run against the public 2.2.0 release
on the data_evaluation mass-spec train-excluded benchmark.

This benchmark provides hit/decoy peptide sets per HLA allele. We score
peptides with both ensembles and report how well each ranks hits above
decoys. Lower predicted nM = stronger binder, so we negate the affinity
for AUC/AP calculations.

Metrics per allele:
  - ROC-AUC
  - Average-precision (PR-AUC) at the positive class
  - PPV at N (N = number of true hits in the set; fraction of top-N
    predictions that are actual hits)

Usage:
    python compare_new_vs_public.py \\
        --new-models-dir results/new_run/models.combined \\
        --public-models-dir "$HOME/Library/Application Support/mhcflurry/4/2.2.0/models_class1_pan/models.combined" \\
        --data-dir "$HOME/Library/Application Support/mhcflurry/4/2.2.0/data_evaluation" \\
        --out results/comparison \\
        --num-jobs auto
"""
import argparse
import glob
import json
import os
import time
from functools import partial

import numpy
import pandas
from sklearn.metrics import average_precision_score, roc_auc_score

from mhcflurry.local_parallelism import (
    add_prediction_parallelism_args,
    call_wrapped_kwargs,
    chunk_ranges_for_local_parallelism,
    worker_pool_with_gpu_assignments_from_args,
)
from mhcflurry.workload_planning import WORKLOAD_AFFINITY_INFERENCE


_T0 = time.time()


def _stamp(msg):
    print(f"[+{time.time() - _T0:6.1f}s] {msg}", flush=True)


def _predict_chunk(predictor_dir, peptides, alleles, chunk_num):
    """Load the predictor in the configured process and score one chunk."""
    from mhcflurry import Class1AffinityPredictor
    predictor = Class1AffinityPredictor.load(predictor_dir)
    return chunk_num, numpy.asarray(predictor.predict(
        peptides=peptides, alleles=alleles, throw=False))


def _read_supported_alleles(predictor_dir):
    """Enumerate supported alleles from the predictor's bundled
    allele_sequences.csv -- avoids importing the predictor in the parent
    process (which would initialize CUDA before workers can pin GPUs)."""
    from mhcflurry.pseudosequences import LEGACY_ALLELE_SEQUENCES_FILENAME
    path = os.path.join(predictor_dir, LEGACY_ALLELE_SEQUENCES_FILENAME)
    if not os.path.exists(path):
        # Fall back to system download path; layout matches predictor.
        return set()
    df = pandas.read_csv(path)
    col = "normalized_allele" if "normalized_allele" in df.columns else df.columns[0]
    return set(df[col].astype(str).tolist())


def parallel_predict(args, predictor_dir, peptides, alleles):
    """Score peptides using mhcflurry's shared prediction parallelism plan."""
    if len(peptides) == 0:
        return numpy.asarray([], dtype=float)

    worker_pool = worker_pool_with_gpu_assignments_from_args(
        args,
        workload_name=WORKLOAD_AFFINITY_INFERENCE,
        workload_hints={"prediction_rows": len(peptides)},
        start_method="spawn",
    )
    _stamp(
        "      prediction plan: jobs=%d, gpus=%d, workers/gpu=%d, backend=%s"
        % (
            int(args.num_jobs),
            int(args.gpus or 0),
            int(args.max_workers_per_gpu),
            args.backend,
        )
    )

    if worker_pool is None:
        _, predictions = _predict_chunk(
            predictor_dir=predictor_dir,
            peptides=peptides,
            alleles=alleles,
            chunk_num=0,
        )
        return predictions

    work_items = []
    for (chunk_num, start, end) in chunk_ranges_for_local_parallelism(
            len(peptides), args.num_jobs):
        work_items.append({
            "chunk_num": chunk_num,
            "predictor_dir": predictor_dir,
            "peptides": peptides[start:end],
            "alleles": alleles[start:end],
        })
    try:
        results = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, _predict_chunk),
            work_items,
            chunksize=1,
        )
        chunks = [result for result in results]
    finally:
        worker_pool.close()
        worker_pool.join()
    # chunk_num is unique per work item, so the int comparator wins and
    # sorted() never needs to compare the ndarray values.
    return numpy.concatenate([
        values for (_, values) in sorted(chunks)
    ])


def ppv_at_n(y_true, y_score, n):
    order = numpy.argsort(-y_score)
    top = order[:n]
    return float(y_true[top].sum()) / float(n) if n > 0 else numpy.nan


def metrics(y_true, y_score):
    y_true = numpy.asarray(y_true)
    y_score = numpy.asarray(y_score)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return dict(n=int(len(y_true)), n_pos=n_pos, roc_auc=numpy.nan,
                    pr_auc=numpy.nan, ppv_at_n=numpy.nan)
    return dict(
        n=int(len(y_true)),
        n_pos=n_pos,
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        ppv_at_n=ppv_at_n(y_true, y_score, n_pos),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new-models-dir", required=True)
    p.add_argument("--public-models-dir", required=True)
    p.add_argument("--data-dir", required=True,
                   help="data_evaluation/ directory with "
                        "benchmark.monoallelic.*.train_excluded.*.csv.bz2")
    p.add_argument("--source", choices=["mixmhcpred", "netmhcpan4", "both"],
                   default="mixmhcpred",
                   help="Which baseline-variant to pull benchmark files from")
    p.add_argument("--out", required=True)
    p.add_argument("--limit-alleles", type=int, default=None,
                   help="Debug/dry-run: only evaluate first N allele files")
    add_prediction_parallelism_args(p)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # We need supported_alleles for the new/public ensembles before we
    # know what rows to evaluate, but we don't want to import torch in
    # the parent. Read manifests directly via pandas to enumerate
    # supported alleles -- predict happens entirely in workers.
    new_alleles = _read_supported_alleles(args.new_models_dir)
    pub_alleles = _read_supported_alleles(args.public_models_dir)
    _stamp(f"[1/5] new ensemble: {args.new_models_dir} "
           f"({len(new_alleles)} alleles supported)")
    _stamp(f"[2/5] public ensemble: {args.public_models_dir} "
           f"({len(pub_alleles)} alleles supported)")

    _stamp("[3/5] Finding benchmark files...")
    if args.source == "both":
        patterns = ["mixmhcpred", "netmhcpan4"]
    else:
        patterns = [args.source]
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(
            os.path.join(args.data_dir,
                         f"benchmark.monoallelic.{pat}.train_excluded.*.csv.bz2")
        )))
    if args.limit_alleles:
        files = files[:args.limit_alleles]
    print(f"      {len(files)} benchmark files")

    all_dfs = []
    for i, f in enumerate(files):
        df = pandas.read_csv(f)
        df["source_file"] = os.path.basename(f)
        all_dfs.append(df)
        if (i + 1) % 50 == 0:
            print(f"      loaded {i + 1}/{len(files)}")
    test = pandas.concat(all_dfs, ignore_index=True)
    print(f"      total rows: {len(test):,}")

    both = new_alleles & pub_alleles
    print(f"      new alleles: {len(new_alleles)}, public: {len(pub_alleles)}, "
          f"intersect: {len(both)}")
    before = len(test)
    test = test[test.hla.isin(both)].copy()
    if len(test) < before:
        print(f"      dropped {before - len(test)} rows (alleles outside intersect)")
    # Filter to supported peptide lengths (8-15)
    test["peptide_len"] = test["peptide"].str.len()
    test = test[(test.peptide_len >= 8) & (test.peptide_len <= 15)].copy()
    print(f"      evaluable rows after filter: {len(test):,}")

    _stamp(f"[4a/5] Predicting NEW ensemble ({len(test):,} rows)...")
    test["new_pred"] = parallel_predict(
        args,
        args.new_models_dir,
        test.peptide.values,
        test.hla.values,
    )
    _stamp(f"[4b/5] Predicting PUBLIC ensemble ({len(test):,} rows)...")
    test["public_pred"] = parallel_predict(
        args,
        args.public_models_dir,
        test.peptide.values,
        test.hla.values,
    )
    test = test.dropna(subset=["new_pred", "public_pred"])
    _stamp(f"      rows with both predictions: {len(test):,}")
    # Score = -log10(predicted nM) (higher = binder-like)
    test["new_score"] = -numpy.log10(numpy.clip(test.new_pred, 1e-3, 1e8))
    test["public_score"] = -numpy.log10(numpy.clip(test.public_pred, 1e-3, 1e8))
    predictions_path = os.path.join(args.out, "predictions.csv.bz2")
    test.to_csv(predictions_path, index=False)
    _stamp(f"      wrote {predictions_path}")

    _stamp("[5/5] Computing metrics per allele...")
    rows = []
    for allele, group in test.groupby("hla"):
        if len(group) < 30 or group.hit.sum() == 0:
            continue
        m_new = metrics(group.hit.values, group.new_score.values)
        m_pub = metrics(group.hit.values, group.public_score.values)
        rows.append({
            "allele": allele,
            "n": m_new["n"], "n_pos": m_new["n_pos"],
            "new_roc_auc": m_new["roc_auc"],
            "public_roc_auc": m_pub["roc_auc"],
            "new_pr_auc": m_new["pr_auc"],
            "public_pr_auc": m_pub["pr_auc"],
            "new_ppv_at_n": m_new["ppv_at_n"],
            "public_ppv_at_n": m_pub["ppv_at_n"],
        })
    per_allele = pandas.DataFrame(rows)
    for col in ["roc_auc", "pr_auc", "ppv_at_n"]:
        per_allele[f"{col}_diff"] = per_allele[f"new_{col}"] - per_allele[f"public_{col}"]
    per_allele = per_allele.sort_values("n", ascending=False)
    per_allele_path = os.path.join(args.out, "per_allele.csv")
    per_allele.to_csv(per_allele_path, index=False)
    print(f"      wrote {per_allele_path} ({len(per_allele)} alleles)")

    m_new_all = metrics(test.hit.values, test.new_score.values)
    m_pub_all = metrics(test.hit.values, test.public_score.values)

    # Per-peptide-length breakdown (8-12mers). For each length we compute:
    #   - micro: pool all rows of that length across alleles
    #   - macro: mean of per-allele metrics (allele restricted to that length;
    #     skip alleles with <30 rows or 0 hits at this length)
    lengths = [8, 9, 10, 11, 12]
    per_length_rows = []
    per_length_per_allele_rows = []
    for L in lengths:
        sub = test[test.peptide_len == L]
        if len(sub) == 0:
            continue
        m_new_L = metrics(sub.hit.values, sub.new_score.values)
        m_pub_L = metrics(sub.hit.values, sub.public_score.values)
        per_allele_L = []
        for allele, group in sub.groupby("hla"):
            if len(group) < 30 or group.hit.sum() == 0:
                continue
            ma_new = metrics(group.hit.values, group.new_score.values)
            ma_pub = metrics(group.hit.values, group.public_score.values)
            per_allele_L.append({
                "allele": allele, "length": L,
                "n": ma_new["n"], "n_pos": ma_new["n_pos"],
                "new_roc_auc": ma_new["roc_auc"],
                "public_roc_auc": ma_pub["roc_auc"],
                "new_pr_auc": ma_new["pr_auc"],
                "public_pr_auc": ma_pub["pr_auc"],
                "new_ppv_at_n": ma_new["ppv_at_n"],
                "public_ppv_at_n": ma_pub["ppv_at_n"],
            })
        per_length_per_allele_rows.extend(per_allele_L)
        macro_new = {m: float(numpy.nanmean([r[f"new_{m}"] for r in per_allele_L]))
                     if per_allele_L else float("nan")
                     for m in ["roc_auc", "pr_auc", "ppv_at_n"]}
        macro_pub = {m: float(numpy.nanmean([r[f"public_{m}"] for r in per_allele_L]))
                     if per_allele_L else float("nan")
                     for m in ["roc_auc", "pr_auc", "ppv_at_n"]}
        per_length_rows.append({
            "length": L,
            "n": m_new_L["n"], "n_pos": m_new_L["n_pos"],
            "n_alleles_reported": len(per_allele_L),
            "new_micro_roc_auc": m_new_L["roc_auc"],
            "public_micro_roc_auc": m_pub_L["roc_auc"],
            "new_micro_pr_auc": m_new_L["pr_auc"],
            "public_micro_pr_auc": m_pub_L["pr_auc"],
            "new_micro_ppv_at_n": m_new_L["ppv_at_n"],
            "public_micro_ppv_at_n": m_pub_L["ppv_at_n"],
            "new_macro_roc_auc": macro_new["roc_auc"],
            "public_macro_roc_auc": macro_pub["roc_auc"],
            "new_macro_pr_auc": macro_new["pr_auc"],
            "public_macro_pr_auc": macro_pub["pr_auc"],
            "new_macro_ppv_at_n": macro_new["ppv_at_n"],
            "public_macro_ppv_at_n": macro_pub["ppv_at_n"],
        })
    per_length = pandas.DataFrame(per_length_rows)
    if not per_length.empty:
        for col in ["micro_roc_auc", "micro_pr_auc", "micro_ppv_at_n",
                    "macro_roc_auc", "macro_pr_auc", "macro_ppv_at_n"]:
            per_length[f"{col}_diff"] = (
                per_length[f"new_{col}"] - per_length[f"public_{col}"])
    per_length_path = os.path.join(args.out, "per_length.csv")
    per_length.to_csv(per_length_path, index=False)
    print(f"      wrote {per_length_path} ({len(per_length)} lengths)")

    if per_length_per_allele_rows:
        per_length_per_allele = pandas.DataFrame(per_length_per_allele_rows)
        for col in ["roc_auc", "pr_auc", "ppv_at_n"]:
            per_length_per_allele[f"{col}_diff"] = (
                per_length_per_allele[f"new_{col}"]
                - per_length_per_allele[f"public_{col}"])
        per_length_per_allele = per_length_per_allele.sort_values(
            ["length", "n"], ascending=[True, False])
        per_length_per_allele_path = os.path.join(
            args.out, "per_length_per_allele.csv")
        per_length_per_allele.to_csv(per_length_per_allele_path, index=False)
        print(f"      wrote {per_length_per_allele_path} "
              f"({len(per_length_per_allele)} (length, allele) rows)")

    summary = {
        "n_rows": int(len(test)),
        "n_hits": int(test.hit.sum()),
        "n_alleles_reported": int(len(per_allele)),
        "micro_pooled": {"new": m_new_all, "public": m_pub_all},
        "macro_mean_over_alleles": {
            metric: {
                "new": float(per_allele[f"new_{metric}"].mean()),
                "public": float(per_allele[f"public_{metric}"].mean()),
            } for metric in ["roc_auc", "pr_auc", "ppv_at_n"]
        },
        "per_length": {
            str(int(row["length"])): {
                "n": int(row["n"]),
                "n_pos": int(row["n_pos"]),
                "n_alleles_reported": int(row["n_alleles_reported"]),
                "micro": {
                    metric: {
                        "new": float(row[f"new_micro_{metric}"]),
                        "public": float(row[f"public_micro_{metric}"]),
                    } for metric in ["roc_auc", "pr_auc", "ppv_at_n"]
                },
                "macro": {
                    metric: {
                        "new": float(row[f"new_macro_{metric}"]),
                        "public": float(row[f"public_macro_{metric}"]),
                    } for metric in ["roc_auc", "pr_auc", "ppv_at_n"]
                },
            }
            for _, row in per_length.iterrows()
        },
        "allele_count": {
            "new_better_roc_auc": int((per_allele.roc_auc_diff > 0).sum()),
            "public_better_roc_auc": int((per_allele.roc_auc_diff < 0).sum()),
            "tie_roc_auc": int((per_allele.roc_auc_diff == 0).sum()),
            "new_better_pr_auc": int((per_allele.pr_auc_diff > 0).sum()),
            "public_better_pr_auc": int((per_allele.pr_auc_diff < 0).sum()),
            "new_better_ppv_at_n": int((per_allele.ppv_at_n_diff > 0).sum()),
            "public_better_ppv_at_n": int((per_allele.ppv_at_n_diff < 0).sum()),
        },
    }
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as fd:
        json.dump(summary, fd, indent=2, sort_keys=True)
    print(f"      wrote {summary_path}")
    print()
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
