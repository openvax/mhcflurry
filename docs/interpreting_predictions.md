# Interpreting predictions

MHCflurry's default presentation predictor outputs several columns for each
peptide. This guide explains what each number means, which column to rank by,
and what cutoffs work well in practice.

## What each column means

In the command-line tool (`mhcflurry predict` and `mhcflurry predict-scan`),
prediction columns are prefixed with `mhcflurry_`. In the Python API
(`Class1PresentationPredictor.predict()`), the same columns appear without the
`mhcflurry_` prefix.

| Column | Scale | Direction | What it measures |
| :--- | :--- | :--- | :--- |
| `presentation_score` | `0` to `1` | **Higher is stronger** | Overall probability-like score combining MHC binding affinity and antigen processing. |
| `presentation_percentile` | `0` to `100` | **Lower is stronger** | Rank of `presentation_score` compared against random natural peptides (for example, `0.5` = top 0.5%). |
| `affinity` | `nM` (IC50) | **Lower is stronger** | Predicted peptide–MHC binding affinity in nanomolar. Values below `500 nM` are typical binders; values below `50 nM` are strong binders. |
| `affinity_percentile` | `0` to `100` | **Lower is stronger** | Allele-specific rank of `affinity` against random peptides for that allele. |
| `processing_score` | `0` to `1` | **Higher is stronger** | Allele-independent estimate of how likely cellular cleavage and transport are to generate this peptide. |
| `best_allele` | Allele name | — | When scoring a multi-allele genotype (sample), the single allele with the strongest predicted binding for that peptide. |

## Which score should I use?

- **Prioritizing candidate epitopes or neoantigens (most users):**
  Rank by **`presentation_score`** (highest first) or **`presentation_percentile`** (lowest first).
  Combining binding affinity with antigen processing consistently improves
  enrichment of real mass-spectrometry ligands over binding affinity alone,
  especially when source-protein flanking sequences are available.
- **Predicting or comparing in-vitro binding assays:**
  Use **`affinity`** (nM) or **`affinity_percentile`**. Antigen processing does
  not occur in a purified peptide–MHC binding assay.
- **Filtering peptides when you do not know the MHC allele:**
  Use **`processing_score`** alone (or `Class1ProcessingPredictor`).

## Practical cutoffs and percentile ranks

### Why percentile rank is often better than a fixed nM cutoff

Different MHC alleles naturally have different binding-strength distributions —
some alleles bind many peptides below `100 nM`, while others bind their best
ligands around `300–1,000 nM`. Comparing raw nM values across alleles can
unfairly favor tighter-binding alleles.

**`affinity_percentile`** normalizes each allele against a panel of random
peptides:

- **`<= 2.0` (top 2%):** standard inclusive screening threshold (also the default
  filter for `mhcflurry predict-scan`).
- **`<= 0.5` (top 0.5%):** stricter strong-binder threshold.

If you prefer a fixed affinity cutoff:

- **`<= 500 nM`:** common moderate-to-strong binder cutoff.
- **`<= 50 nM`:** strong binder cutoff.

### Presentation score cutoffs

There is **no single universal cutoff** for `presentation_score` because the
fraction of true ligands depends on your peptide pool size, allele set, and
whether flanking sequences were supplied. In practice:

- Sort candidates by `presentation_score` descending and take the top *K*
  candidates your downstream assay can test, or
- Filter by **`presentation_percentile`** (for example, `<= 2.0` or `<= 1.0`)
  when you need a consistent rank cutoff across different samples.

## Single alleles vs. multi-allele genotypes

When you pass **one allele per query** (for example, `--alleles 'HLA-A*02:01' 'HLA-B*07:02'`),
MHCflurry outputs one row per peptide *for each allele separately*.

When you pass a **genotype or allele set** (for example, `--alleles 'HLA-A*02:01;HLA-B*07:02'`
on the CLI, or `alleles=["HLA-A*02:01", "HLA-B*07:02"]` in Python):

- MHCflurry scores the peptide against every allele in the set.
- It keeps the **strongest binder** (lowest nM `affinity`), records which allele
  won in `best_allele`, and computes `presentation_score` from that strongest
  affinity plus the peptide's `processing_score`.

A single patient's 6-allele HLA class I genotype is a natural allele set. A
large population panel of 20+ alleles is *not* — score population alleles
separately so one promiscuous allele does not mask the others.

## With flanks vs. without flanks

MHCflurry's presentation bundle contains **two** processing models:

- **With flanks** (used automatically by `predict-scan`, and by `predict` when
  `n_flank` and `c_flank` columns are provided): uses up to 15 amino acids
  upstream and downstream of the peptide in its source protein to model
  proteasomal cleavage.
- **Without flanks** (used automatically when flank columns are omitted or
  `--no-flanking` is passed): scores processing from the peptide sequence alone.

Whenever you know the source protein sequence, include flanking sequences (or
use `predict-scan` directly on the protein FASTA) for the strongest presentation
ranking.
