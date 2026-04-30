"""Compare a freshly-trained pan-allele predictor against the public
mhcflurry release on a peptide × allele grid.

For each peptide we pick a canonical allele it's known to bind, then
mix in "distractor" alleles known to bind different peptide motifs.
Our trained predictor should:
  1. Rank canonical peptide-allele pairs above distractor pairs
     consistently with the public weights.
  2. Produce affinities in similar range (IC50 nM).

We also sample random per-peptide allele subsets (Alex's "per sample
allele lists sampled at random") and compare the relative ranking of
alleles our model vs the public one produces.

Run:
    python scripts/validate_against_public.py \\
        --ours out/models.single/    \\
        --public $(mhcflurry-downloads path models_class1_pan)/models.combined/
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

# Well-characterized peptide / allele binding pairs. Pulled from commonly
# cited MHC I epitopes — the canonical allele is what the peptide is
# known to bind; the distractor is a different canonical binder for a
# different peptide.
KNOWN_BINDERS = [
    # peptide            canonical_allele        source / context
    ("SLLQHLIGL",        "HLA-A*02:01"),      # classic A*02:01 test peptide (HCV NS3)
    ("GILGFVFTL",        "HLA-A*02:01"),      # Flu M1 58-66, strong A*02 binder
    ("SLYNTVATL",        "HLA-A*02:01"),      # HIV Gag p17
    ("NLVPMVATV",        "HLA-A*02:01"),      # CMV pp65
    ("NYNYLYRLF",        "HLA-A*24:02"),      # HIV Nef
    ("RYPLTFGWCF",       "HLA-A*24:02"),      # classic A*24:02 binder
    ("VYFFKTNKF",        "HLA-A*24:02"),      # SARS-CoV-2 S
    ("TPRVTGGGAM",       "HLA-B*07:02"),      # CMV pp65
    ("APRTVALTA",        "HLA-B*07:02"),      # P at P2 anchor (B*07 family)
    ("SPRWYFYYL",        "HLA-B*35:01"),      # SARS-CoV-2
    ("IPSINVHHY",        "HLA-B*35:01"),      # EBV EBNA-1
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


def load_predictor(path: Path):
    from mhcflurry import Class1AffinityPredictor
    return Class1AffinityPredictor.load(str(path))


def predict(pred, peptides, alleles) -> pd.DataFrame:
    """Return a tall DataFrame with columns peptide, allele, affinity_nM."""
    rows = []
    for p in peptides:
        for a in alleles:
            try:
                aff = float(pred.predict(peptides=[p], alleles=[a])[0])
            except Exception as exc:
                aff = float("nan")
                print(f"  [warn] {p}+{a}: {exc!r}", file=sys.stderr)
            rows.append({"peptide": p, "allele": a, "affinity_nM": aff})
    return pd.DataFrame(rows)


def canonical_agreement(df: pd.DataFrame, label: str) -> int:
    """For each peptide, check that its canonical allele is ranked at or
    near the top (best affinity = lowest nM)."""
    wins = 0
    total = 0
    for peptide, canonical in KNOWN_BINDERS:
        sub = df[df["peptide"] == peptide].sort_values("affinity_nM")
        if sub.empty:
            continue
        total += 1
        rank = sub["allele"].tolist().index(canonical) + 1 if canonical in sub["allele"].values else -1
        top = sub.iloc[0]
        mark = "✓" if top["allele"] == canonical else " "
        print(
            f"  {mark} {label:8s} {peptide:12s} canonical={canonical:14s} "
            f"rank={rank}/{len(sub):<2} top={top['allele']} ({top['affinity_nM']:.1f} nM)"
        )
        if top["allele"] == canonical:
            wins += 1
    return wins


def rank_correlation(df_ours: pd.DataFrame, df_public: pd.DataFrame) -> float:
    from scipy.stats import spearmanr
    merged = df_ours.merge(df_public, on=["peptide", "allele"],
                           suffixes=("_ours", "_public"))
    if merged.empty:
        return float("nan")
    rho, _ = spearmanr(merged["affinity_nM_ours"], merged["affinity_nM_public"])
    return rho


def per_sample_random(df: pd.DataFrame, *, k: int = 4,
                      n_samples: int = 20, seed: int = 17) -> pd.DataFrame:
    """For each peptide, N random k-subsets of alleles. Returns a per-
    sample dataframe with peptide + allele_subset + predicted best."""
    rng = random.Random(seed)
    rows = []
    for p in df["peptide"].unique():
        pdf = df[df["peptide"] == p]
        alleles = pdf["allele"].tolist()
        for _ in range(n_samples):
            subset = rng.sample(alleles, k=min(k, len(alleles)))
            sdf = pdf[pdf["allele"].isin(subset)].sort_values("affinity_nM")
            top = sdf.iloc[0]
            rows.append({
                "peptide": p,
                "subset": ",".join(sorted(subset)),
                "best_allele": top["allele"],
                "best_affinity_nM": top["affinity_nM"],
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", required=True, help="dir with trained predictor")
    p.add_argument("--public", required=True, help="dir with public models_class1_pan predictor")
    args = p.parse_args()

    peptides = [pep for pep, _ in KNOWN_BINDERS]
    alleles = ALLELE_POOL

    print(f"[load] ours <- {args.ours}")
    ours = load_predictor(Path(args.ours))
    print(f"[load] public <- {args.public}")
    public = load_predictor(Path(args.public))

    print(f"[predict] {len(peptides)} peptides × {len(alleles)} alleles = "
          f"{len(peptides)*len(alleles)} pairs, both models")
    df_ours = predict(ours, peptides, alleles)
    df_public = predict(public, peptides, alleles)

    print()
    print("== Canonical allele at top of ranking? ==")
    w_ours = canonical_agreement(df_ours, "ours")
    print()
    w_public = canonical_agreement(df_public, "public")
    total = len(KNOWN_BINDERS)
    print()
    print(f"[score] ours:   {w_ours}/{total} canonical peptides with canonical allele at rank 1")
    print(f"[score] public: {w_public}/{total}")

    rho = rank_correlation(df_ours, df_public)
    print(f"[score] Spearman rank correlation (all pairs): {rho:.3f}")

    print()
    print("== Random per-peptide 4-subset agreement ==")
    rand_ours = per_sample_random(df_ours)
    rand_public = per_sample_random(df_public)
    merged = rand_ours.merge(
        rand_public, on=["peptide", "subset"], suffixes=("_ours", "_public")
    )
    agree = (merged["best_allele_ours"] == merged["best_allele_public"]).sum()
    print(f"[score] best-allele agreement on random 4-subsets: "
          f"{agree}/{len(merged)} ({agree/len(merged)*100:.0f}%)")

    print()
    print("== Example disagreement (first 5) ==")
    disagree = merged[merged["best_allele_ours"] != merged["best_allele_public"]]
    if not disagree.empty:
        print(disagree.head(5).to_string(index=False))
    else:
        print("(none)")

    out = Path("out/validation")
    out.mkdir(parents=True, exist_ok=True)
    df_ours.to_csv(out / "preds_ours.csv", index=False)
    df_public.to_csv(out / "preds_public.csv", index=False)
    merged.to_csv(out / "random_subset_agreement.csv", index=False)
    print(f"\n[saved] full predictions + agreement tables to {out}/")


if __name__ == "__main__":
    main()
