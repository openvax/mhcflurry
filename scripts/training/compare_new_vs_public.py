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
        --out results/comparison
"""
import argparse
import glob
import json
import os
import sys

import numpy
import pandas
from sklearn.metrics import average_precision_score, roc_auc_score


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
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from mhcflurry import Class1AffinityPredictor

    print(f"[1/5] Loading new ensemble: {args.new_models_dir}")
    new = Class1AffinityPredictor.load(args.new_models_dir)
    print(f"      {len(new.neural_networks)} networks, "
          f"{len(new.supported_alleles)} alleles supported")

    print(f"[2/5] Loading public ensemble: {args.public_models_dir}")
    pub = Class1AffinityPredictor.load(args.public_models_dir)
    print(f"      {len(pub.neural_networks)} networks, "
          f"{len(pub.supported_alleles)} alleles supported")

    print(f"[3/5] Finding benchmark files...")
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

    new_alleles = set(new.supported_alleles)
    pub_alleles = set(pub.supported_alleles)
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

    print(f"[4/5] Predicting with both ensembles...")
    test["new_pred"] = new.predict(
        peptides=test.peptide.values,
        alleles=test.hla.values,
        throw=False,
    )
    test["public_pred"] = pub.predict(
        peptides=test.peptide.values,
        alleles=test.hla.values,
        throw=False,
    )
    test = test.dropna(subset=["new_pred", "public_pred"])
    print(f"      rows with both predictions: {len(test):,}")
    # Score = -log10(predicted nM) (higher = binder-like)
    test["new_score"] = -numpy.log10(numpy.clip(test.new_pred, 1e-3, 1e8))
    test["public_score"] = -numpy.log10(numpy.clip(test.public_pred, 1e-3, 1e8))

    print(f"[5/5] Computing metrics per allele...")
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
