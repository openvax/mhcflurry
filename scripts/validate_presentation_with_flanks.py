"""Compare trained ensemble vs public mhcflurry on 10 peptides with their
real source-protein flanks.

Per-user ask (phase-c-single-a100 run): "test the ensemble against the
public weights for accuracy on our example set of 10 peptides hopefully
including SLLQHLIGL (find its flanking region in PRAME), and random other
peptides + flanks and alleles including A0201 and A2402. Summarize rank
correlation of public mhcflurry with trained ensemble on all outputs."

What this script does:

  1. For each of 10 well-characterized epitopes (incl. SLLQHLIGL in PRAME,
     A*02:01 + A*24:02 coverage), look up the peptide in uniprot_proteins
     and extract real N- and C-terminal 10-AA flanks from the source
     protein. Falls back to empty flanks when a peptide isn't found.
  2. Predict presentation scores with both our ensemble and the public
     ensemble, on (peptide, allele, n_flank, c_flank) tuples for a small
     allele pool.
  3. Report Spearman rank correlation overall, per peptide, and the
     canonical-allele rank check (is the peptide's best-scoring allele
     the one the literature says binds?).

Run:
    python scripts/validate_presentation_with_flanks.py \\
        --ours /path/to/models.combined/ \\
        --public $(mhcflurry-downloads path models_class1_presentation)/models/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


# 10 peptides: A*02:01 (4), A*24:02 (3), B*07:02/B*35:01 (3).
# Canonical allele is the literature-cited binder; we still predict over
# the full ALLELE_POOL to check ranking behavior.
PEPTIDES = [
    # peptide,      canonical_allele, source_antigen_comment
    ("SLLQHLIGL",   "HLA-A*02:01",    "PRAME (melanoma preferentially expressed)"),
    ("GILGFVFTL",   "HLA-A*02:01",    "Influenza A M1 58-66"),
    ("SLYNTVATL",   "HLA-A*02:01",    "HIV-1 Gag p17 77-85"),
    ("NLVPMVATV",   "HLA-A*02:01",    "HCMV pp65 495-503"),
    ("NYNYLYRLF",   "HLA-A*24:02",    "HIV-1 Nef 134-142"),
    ("VYFFKTNKF",   "HLA-A*24:02",    "SARS-CoV-2 Spike"),
    ("RYPLTFGWCF",  "HLA-A*24:02",    "synthetic A*24 probe"),
    ("TPRVTGGGAM",  "HLA-B*07:02",    "HCMV pp65 265-274"),
    ("IPSINVHHY",   "HLA-B*35:01",    "EBV EBNA-1 407-417"),
    ("KAFSPEVIPMF", "HLA-B*57:01",    "HIV-1 Gag KF11"),
]

ALLELE_POOL = [
    "HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02", "HLA-B*35:01",
    "HLA-B*57:01", "HLA-A*03:01", "HLA-C*07:01",
]

FLANK_SIZE = 10  # AAs on each side


def lookup_flanks(peptides):
    """Return dict peptide -> (n_flank, c_flank, source_id) for each peptide.

    Uses mhcflurry's bundled uniprot_proteins.csv.bz2. Searches every
    protein for the peptide; takes the first protein that contains it.
    Returns empty flanks if not found.
    """
    try:
        from mhcflurry.downloads import get_path
        path = get_path("data_references", "uniprot_proteins.csv.bz2")
    except Exception as exc:
        print(f"[warn] uniprot_proteins reference not available: {exc}",
              file=sys.stderr)
        return {p: ("", "", "N/A") for p in peptides}

    df = pd.read_csv(path)
    seqs = df[["gene_name", "accession", "seq"]].dropna(subset=["seq"])
    result = {}
    for pep in peptides:
        found = False
        for _, row in seqs.iterrows():
            seq = str(row["seq"])
            idx = seq.find(pep)
            if idx >= 0:
                n = seq[max(0, idx - FLANK_SIZE):idx]
                c = seq[idx + len(pep): idx + len(pep) + FLANK_SIZE]
                source = f"{row['gene_name']}/{row['accession']}"
                result[pep] = (n, c, source)
                found = True
                break
        if not found:
            result[pep] = ("", "", "NOT_FOUND")
    return result


def load_predictor(path):
    from mhcflurry import Class1PresentationPredictor
    return Class1PresentationPredictor.load(str(path))


def predict_all(predictor, peptides, flanks, alleles):
    """Predict presentation with flanks for each (peptide, allele) pair.

    Uses Class1PresentationPredictor.predict with explicit n_flanks/c_flanks.
    Returns tall DataFrame: peptide, allele, n_flank, c_flank, score.
    """
    rows = []
    for allele in alleles:
        peps = [p for p, _, _ in PEPTIDES]
        n_flanks = [flanks[p][0] for p in peps]
        c_flanks = [flanks[p][1] for p in peps]
        try:
            df = predictor.predict(
                peptides=peps,
                alleles=[allele],
                n_flanks=n_flanks,
                c_flanks=c_flanks,
                verbose=0,
            )
        except Exception as exc:
            print(f"  [warn] {allele}: {exc!r}", file=sys.stderr)
            for p in peps:
                rows.append({
                    "peptide": p,
                    "allele": allele,
                    "n_flank": flanks[p][0],
                    "c_flank": flanks[p][1],
                    "score": float("nan"),
                })
            continue
        for p in peps:
            sub = df[df["peptide"] == p]
            score = float(sub.iloc[0]["presentation_score"]) if not sub.empty else float("nan")
            rows.append({
                "peptide": p,
                "allele": allele,
                "n_flank": flanks[p][0],
                "c_flank": flanks[p][1],
                "score": score,
            })
    return pd.DataFrame(rows)


def spearman_all(ours, public):
    """Overall Spearman on (peptide, allele, score)."""
    merged = ours.merge(
        public, on=["peptide", "allele"], suffixes=("_ours", "_public")
    )
    if merged.empty:
        return float("nan"), 0
    rho, _ = spearmanr(
        merged["score_ours"], merged["score_public"], nan_policy="omit"
    )
    return rho, len(merged)


def spearman_per_peptide(ours, public):
    """Spearman per peptide (across the 7 alleles). Captures within-peptide
    ranking agreement — the signal most relevant for ranking likely
    presenters of a given epitope."""
    merged = ours.merge(
        public, on=["peptide", "allele"], suffixes=("_ours", "_public")
    )
    rows = []
    for p, grp in merged.groupby("peptide"):
        if len(grp) < 3:
            rows.append({"peptide": p, "n_alleles": len(grp), "spearman": float("nan")})
            continue
        rho, _ = spearmanr(
            grp["score_ours"], grp["score_public"], nan_policy="omit"
        )
        rows.append({"peptide": p, "n_alleles": len(grp), "spearman": rho})
    return pd.DataFrame(rows)


def canonical_ranks(df, label):
    """For each peptide, is the canonical allele ranked #1 by score?"""
    wins = 0
    for pep, canonical, antigen in PEPTIDES:
        sub = df[df["peptide"] == pep].sort_values("score", ascending=False)
        if sub.empty:
            continue
        top_allele = sub.iloc[0]["allele"]
        rank = (
            sub["allele"].tolist().index(canonical) + 1
            if canonical in sub["allele"].values
            else -1
        )
        ok = "✓" if top_allele == canonical else " "
        score = sub.iloc[0]["score"]
        print(f"  {ok} {label:6s} {pep:12s} canonical={canonical:14s}"
              f"  top={top_allele:14s} (score={score:.3f})  rank={rank}"
              f"  [{antigen}]")
        if top_allele == canonical:
            wins += 1
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True,
                    help="Dir with trained Class1PresentationPredictor")
    ap.add_argument("--public", required=True,
                    help="Dir with public Class1PresentationPredictor")
    ap.add_argument("--out-dir", default="out/validation_with_flanks")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    peptides = [p for p, _, _ in PEPTIDES]
    print(f"[flanks] looking up source-protein flanks for {len(peptides)} peptides...")
    flanks = lookup_flanks(peptides)
    for p, canon, antigen in PEPTIDES:
        n, c, src = flanks[p]
        print(f"  {p:12s} N={n or '(empty)':10s}  C={c or '(empty)':10s}  src={src}")

    print(f"\n[load] ours    <- {args.ours}")
    ours = load_predictor(Path(args.ours))
    print(f"[load] public  <- {args.public}")
    public = load_predictor(Path(args.public))

    print(f"\n[predict] {len(peptides)} peptides × {len(ALLELE_POOL)} alleles = "
          f"{len(peptides) * len(ALLELE_POOL)} pairs, each model")
    df_ours = predict_all(ours, peptides, flanks, ALLELE_POOL)
    df_public = predict_all(public, peptides, flanks, ALLELE_POOL)

    df_ours.to_csv(out / "preds_ours.csv", index=False)
    df_public.to_csv(out / "preds_public.csv", index=False)

    # Overall Spearman
    rho, n = spearman_all(df_ours, df_public)
    print(f"\n[overall] Spearman rank correlation (peptide×allele, n={n}): {rho:.3f}")

    # Per-peptide Spearman
    per_pep = spearman_per_peptide(df_ours, df_public)
    per_pep.to_csv(out / "spearman_per_peptide.csv", index=False)
    print("\n[per-peptide] within-peptide Spearman (across alleles):")
    for _, row in per_pep.iterrows():
        rho_str = f"{row['spearman']:.3f}" if pd.notna(row["spearman"]) else "NaN"
        print(f"  {row['peptide']:12s}  n_alleles={row['n_alleles']}  rho={rho_str}")
    print(f"\n[per-peptide] mean Spearman: {per_pep['spearman'].mean():.3f}")

    # Canonical-allele ranking
    print(f"\n[canonical] is the canonical allele the top-scoring one?")
    w_ours = canonical_ranks(df_ours, "ours")
    print()
    w_public = canonical_ranks(df_public, "public")
    print(f"\n[canonical] ours:   {w_ours}/{len(PEPTIDES)}")
    print(f"[canonical] public: {w_public}/{len(PEPTIDES)}")

    print(f"\n[saved] predictions + spearman under {out}/")


if __name__ == "__main__":
    main()
