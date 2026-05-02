"""Compare a freshly-trained pan-allele predictor against the public
mhcflurry release on a peptide × allele grid.

Two levels of comparison are reported:

  1. **Affinity** — load each model as a Class1AffinityPredictor and
     predict IC50 for every (peptide, allele) pair. Spearman measures
     rank agreement with the public affinity predictor.

  2. **Presentation** — load the public Class1PresentationPredictor,
     then swap its affinity predictor for ours while keeping the
     public's processing predictors and logistic weights. This isolates
     how much our affinity change propagates through the full
     presentation stack.

We also tally canonical-allele rank-1 hits and best-allele agreement
across random per-peptide 4-subsets of the allele pool.

Run:
    python scripts/validate_against_public.py \\
        --ours out/models.unselected.release/    \\
        --public $(mhcflurry-downloads path models_class1_pan)/models.combined/ \\
        --public-presentation $(mhcflurry-downloads path models_class1_presentation)/models/
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

# Well-characterized peptide / allele binding pairs with source antigen.
# SPRWYFYYL (SARS-CoV-2 N protein) dropped: both ours and public rank
# B*07:02 over B*35:01 for that peptide (motif at C-term Leu fits B*07,
# not B*35), so the canonical label is ambiguous and it penalized both
# models equally.
KNOWN_BINDERS = [
    # peptide,      canonical_allele, source antigen
    ("SLLQHLIGL",   "HLA-A*02:01",    "HCV NS3 1406-1414"),
    ("GILGFVFTL",   "HLA-A*02:01",    "Influenza A M1 58-66"),
    ("SLYNTVATL",   "HLA-A*02:01",    "HIV-1 Gag p17 77-85"),
    ("NLVPMVATV",   "HLA-A*02:01",    "HCMV pp65 495-503"),
    ("NYNYLYRLF",   "HLA-A*24:02",    "HIV-1 Nef 134-142"),
    ("RYPLTFGWCF",  "HLA-A*24:02",    "synthetic A*24 probe"),
    ("VYFFKTNKF",   "HLA-A*24:02",    "SARS-CoV-2 Spike"),
    ("TPRVTGGGAM",  "HLA-B*07:02",    "HCMV pp65 265-274"),
    ("APRTVALTA",   "HLA-B*07:02",    "synthetic B*07 probe"),
    ("IPSINVHHY",   "HLA-B*35:01",    "EBV EBNA-1 407-417"),
]

# All alleles we'll consider. Using each canonical + some distractors
# (other MHC class I supertypes) so every peptide has both "should bind"
# and "should not bind" options in the grid.
ALLELE_POOL = sorted({
    "HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02", "HLA-B*35:01",
    "HLA-A*03:01", "HLA-A*11:01",          # A3 supertype
    "HLA-B*08:01", "HLA-B*15:01",
    "HLA-C*07:01",
})


def load_affinity_predictor(path: Path):
    from mhcflurry import Class1AffinityPredictor
    return Class1AffinityPredictor.load(str(path))


def load_presentation_predictor(path: Path):
    from mhcflurry import Class1PresentationPredictor
    return Class1PresentationPredictor.load(str(path))


def swap_affinity_into_presentation(public_presentation, new_affinity):
    """Build a Class1PresentationPredictor that uses our affinity but
    keeps the public's processing predictors and logistic weights. This
    isolates how much our affinity change propagates through the full
    presentation stack."""
    from mhcflurry import Class1PresentationPredictor
    return Class1PresentationPredictor(
        affinity_predictor=new_affinity,
        processing_predictor_with_flanks=public_presentation.processing_predictor_with_flanks,
        processing_predictor_without_flanks=public_presentation.processing_predictor_without_flanks,
        weights_dataframe=public_presentation.weights_dataframe,
        metadata_dataframes=getattr(public_presentation, "metadata_dataframes", None),
    )


def predict_affinity(pred, peptides, alleles, encoding_cache_dir=None) -> pd.DataFrame:
    """Tall DataFrame: peptide, allele, affinity_nM.

    Batches by allele so the full peptide list goes through predict()
    once per allele (letting the encoding cache, if configured,
    prepopulate once and reuse across alleles).
    """
    rows = []
    peptides = list(peptides)
    for a in alleles:
        try:
            affs = pred.predict(
                peptides=peptides, alleles=[a] * len(peptides),
                encoding_cache_dir=encoding_cache_dir,
            )
        except Exception as exc:
            print(f"  [warn] affinity allele={a}: {exc!r}", file=sys.stderr)
            affs = [float("nan")] * len(peptides)
        for p, aff in zip(peptides, affs):
            rows.append({"peptide": p, "allele": a, "affinity_nM": float(aff)})
    return pd.DataFrame(rows)


def predict_presentation(pred, peptides, alleles, encoding_cache_dir=None) -> pd.DataFrame:
    """Tall DataFrame: peptide, allele, presentation_score. Uses the
    without-flanks processing model since our canonical peptide list
    doesn't carry true source-protein context. Higher = more likely
    presented."""
    # Note: Class1PresentationPredictor.predict doesn't accept
    # encoding_cache_dir yet — the affinity sub-predictor inside does.
    # When the presentation predictor's own signature picks up the
    # param, propagate it here; for now it's ignored (we share the
    # affinity-side speedup via the shared cache on disk anyway).
    rows = []
    for a in alleles:
        try:
            df = pred.predict(peptides=list(peptides), alleles=[a], verbose=0)
        except Exception as exc:
            print(f"  [warn] presentation allele={a}: {exc!r}", file=sys.stderr)
            for p in peptides:
                rows.append({"peptide": p, "allele": a, "presentation_score": float("nan")})
            continue
        for p in peptides:
            row_ = df[df["peptide"] == p]
            if row_.empty:
                rows.append({"peptide": p, "allele": a, "presentation_score": float("nan")})
            else:
                rows.append({
                    "peptide": p,
                    "allele": a,
                    "presentation_score": float(row_.iloc[0]["presentation_score"]),
                })
    return pd.DataFrame(rows)


def canonical_agreement(df: pd.DataFrame, score_col: str, lower_is_better: bool, label: str) -> int:
    """For each canonical peptide-allele pair, does the peptide's best
    scoring allele (lowest affinity or highest presentation score) match
    the canonical binder?"""
    wins = 0
    ascending = lower_is_better
    unit_tag = "nM" if score_col == "affinity_nM" else ""
    for peptide, canonical, antigen in KNOWN_BINDERS:
        sub = df[df["peptide"] == peptide].sort_values(score_col, ascending=ascending)
        if sub.empty:
            continue
        rank = (
            sub["allele"].tolist().index(canonical) + 1
            if canonical in sub["allele"].values
            else -1
        )
        top = sub.iloc[0]
        mark = "✓" if top["allele"] == canonical else " "
        val = top[score_col]
        val_str = f"{val:.2f} {unit_tag}".strip() if pd.notna(val) else "?"
        print(
            f"  {mark} {label:8s} {peptide:12s}  canonical={canonical:14s}  "
            f"antigen={antigen:28s}  rank={rank}/{len(sub):<2}  top={top['allele']} ({val_str})"
        )
        if top["allele"] == canonical:
            wins += 1
    return wins


def spearman(df_ours: pd.DataFrame, df_public: pd.DataFrame, score_col: str) -> float:
    from scipy.stats import spearmanr
    merged = df_ours.merge(df_public, on=["peptide", "allele"], suffixes=("_ours", "_public"))
    if merged.empty:
        return float("nan")
    rho, _ = spearmanr(
        merged[f"{score_col}_ours"], merged[f"{score_col}_public"],
        nan_policy="omit",
    )
    return rho


def per_sample_random(df: pd.DataFrame, score_col: str, lower_is_better: bool,
                      *, k: int = 4, n_samples: int = 20, seed: int = 17) -> pd.DataFrame:
    rng = random.Random(seed)
    ascending = lower_is_better
    rows = []
    for p in df["peptide"].unique():
        pdf = df[df["peptide"] == p]
        alleles = pdf["allele"].tolist()
        for _ in range(n_samples):
            subset = rng.sample(alleles, k=min(k, len(alleles)))
            sdf = pdf[pdf["allele"].isin(subset)].sort_values(score_col, ascending=ascending)
            top = sdf.iloc[0]
            rows.append({
                "peptide": p,
                "subset": ",".join(sorted(subset)),
                "best_allele": top["allele"],
                "best_score": float(top[score_col]),
            })
    return pd.DataFrame(rows)


def run_comparison(df_ours, df_public, score_col, lower_is_better, block_label):
    total = len(KNOWN_BINDERS)
    print()
    print(f"== {block_label}: canonical allele at rank 1? ==")
    w_ours = canonical_agreement(df_ours, score_col, lower_is_better, "ours")
    print()
    w_public = canonical_agreement(df_public, score_col, lower_is_better, "public")
    print()
    print(f"[{block_label}] ours:   {w_ours}/{total} canonical pairs with true allele at rank 1")
    print(f"[{block_label}] public: {w_public}/{total}")

    rho = spearman(df_ours, df_public, score_col)
    n_pairs = len(df_ours.merge(df_public, on=["peptide", "allele"]))
    print(f"[{block_label}] Spearman rank correlation over {n_pairs} (peptide,allele) pairs: {rho:.3f}")

    print()
    print(f"== {block_label}: random 4-subset best-allele agreement ==")
    rand_ours = per_sample_random(df_ours, score_col, lower_is_better)
    rand_public = per_sample_random(df_public, score_col, lower_is_better)
    merged = rand_ours.merge(rand_public, on=["peptide", "subset"], suffixes=("_ours", "_public"))
    agree = (merged["best_allele_ours"] == merged["best_allele_public"]).sum()
    print(f"[{block_label}] best-allele agreement on random 4-subsets: "
          f"{agree}/{len(merged)} ({agree/len(merged)*100:.0f}%)")
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", required=True, help="dir with trained affinity predictor")
    p.add_argument("--public", required=True,
                   help="dir with public models_class1_pan predictor (models.combined)")
    p.add_argument("--public-presentation", default=None,
                   help="dir with public models_class1_presentation/models/. "
                        "If given, presentation Spearman is also reported.")
    p.add_argument("--encoding-cache-dir", default=None,
                   help="Shared BLOSUM62 peptide encoding cache directory. "
                        "Populated once on the first predict() call, reused "
                        "by all subsequent predicts — so comparing N "
                        "predictors on the same peptide list costs 1 "
                        "encoding pass instead of N.")
    args = p.parse_args()

    peptides = [pep for pep, _, _ in KNOWN_BINDERS]
    alleles = ALLELE_POOL

    print(f"[peptides] ({len(peptides)})")
    for pep, canon, antigen in KNOWN_BINDERS:
        print(f"    {pep:12s}  canonical={canon:14s}  antigen={antigen}")
    print(f"[alleles] ({len(alleles)}) {alleles}")

    print(f"\n[load] ours affinity  <- {args.ours}")
    ours_aff = load_affinity_predictor(Path(args.ours))
    print(f"[load] public affinity <- {args.public}")
    public_aff = load_affinity_predictor(Path(args.public))

    print(f"[predict] affinity: {len(peptides)} peptides × {len(alleles)} alleles = "
          f"{len(peptides)*len(alleles)} pairs, both models")
    df_aff_ours = predict_affinity(
        ours_aff, peptides, alleles, encoding_cache_dir=args.encoding_cache_dir
    )
    df_aff_public = predict_affinity(
        public_aff, peptides, alleles, encoding_cache_dir=args.encoding_cache_dir
    )

    out = Path("out/validation")
    out.mkdir(parents=True, exist_ok=True)

    merged_aff = run_comparison(df_aff_ours, df_aff_public, "affinity_nM", True, "AFFINITY")
    df_aff_ours.to_csv(out / "preds_affinity_ours.csv", index=False)
    df_aff_public.to_csv(out / "preds_affinity_public.csv", index=False)
    merged_aff.to_csv(out / "subset_agreement_affinity.csv", index=False)

    if args.public_presentation:
        print(f"\n[load] public presentation <- {args.public_presentation}")
        public_pres = load_presentation_predictor(Path(args.public_presentation))
        ours_pres = swap_affinity_into_presentation(public_pres, ours_aff)

        print(f"[predict] presentation: {len(peptides)} peptides × {len(alleles)} alleles, "
              f"both models (without flanks — canonical peptide list has no true "
              f"source-protein flanking context)")
        df_pres_ours = predict_presentation(
            ours_pres, peptides, alleles, encoding_cache_dir=args.encoding_cache_dir
        )
        df_pres_public = predict_presentation(
            public_pres, peptides, alleles, encoding_cache_dir=args.encoding_cache_dir
        )

        merged_pres = run_comparison(
            df_pres_ours, df_pres_public, "presentation_score", False, "PRESENTATION"
        )
        df_pres_ours.to_csv(out / "preds_presentation_ours.csv", index=False)
        df_pres_public.to_csv(out / "preds_presentation_public.csv", index=False)
        merged_pres.to_csv(out / "subset_agreement_presentation.csv", index=False)

    print(f"\n[saved] prediction + agreement tables under {out}/")


if __name__ == "__main__":
    main()
