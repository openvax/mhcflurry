# Scanning proteins and proteomes

MHCflurry has two main prediction commands:

- **`mhcflurry predict`** — when you already have a list or CSV of specific
  peptides to score.
- **`mhcflurry predict-scan`** — when you have full protein sequences (FASTA or
  CSV) and want MHCflurry to generate peptide windows and source-protein flanks
  automatically.

This guide covers common protein-scanning recipes, memory-safe whole-proteome
scans, and everyday command-line pitfalls.

## Basic protein scan

Given a FASTA file (`proteins.fasta`), run:

```shell
mhcflurry predict-scan proteins.fasta \
    --alleles 'HLA-A*02:01;HLA-B*07:02' \
    --out scan_results.csv
```

By default, `predict-scan`:

- slices every **8–11mer** window from each protein (`--peptide-lengths 8-11`),
- automatically extracts N- and C-terminal **flanking sequences** from each
  protein for the antigen-processing model, and
- keeps only strong candidates with **`affinity_percentile <= 2.0`**
  (`--results-filtered`).

## Choosing which peptides to keep

Use one of the three result modes depending on your downstream workflow:

```shell
# 1. Keep the single strongest peptide per protein and sample
mhcflurry predict-scan proteins.fasta \
    --alleles 'HLA-A*02:01;HLA-B*07:02' \
    --results-best \
    --out top_per_protein.csv

# 2. Keep all peptides with affinity <= 500 nM
mhcflurry predict-scan proteins.fasta \
    --alleles 'HLA-A*02:01;HLA-B*07:02' \
    --threshold-affinity 500 \
    --out under_500nm.csv

# 3. Return every scored window (can be large on big FASTA files)
mhcflurry predict-scan proteins.fasta \
    --alleles 'HLA-A*02:01' \
    --results-all \
    --out all_windows.csv
```

To change peptide lengths (for example, 9-mers only, or 8–14mers):

```shell
mhcflurry predict-scan proteins.fasta \
    --alleles 'HLA-A*02:01' \
    --peptide-lengths 9 \
    --out nine_mers.csv
```

## Scanning large proteomes without running out of memory

A full vertebrate proteome (20,000+ proteins) generates tens of millions of
8–11mer windows. By default, `predict-scan` holds the full result table in
memory before writing the CSV.

Pass **`--chunk-size`** to process proteins in fixed batches and stream each
chunk's results directly to the output CSV with flat memory usage:

```shell
mhcflurry predict-scan human_proteome.fasta \
    --alleles 'HLA-A*02:01;HLA-A*03:01;HLA-B*07:02' \
    --chunk-size 500 \
    --out proteome_hits.csv
```

Output rows are bit-for-bit identical to an unchunked run, and the CSV header
is written once at the start.

In Python, you can achieve the same streaming pattern by calling
`predictor.predict_sequences(...)` over slices of your `sequences` dictionary.

## Scanning from Python

The Python equivalent of `predict-scan` is
{meth}`~mhcflurry.Class1PresentationPredictor.predict_sequences`:

```python
from mhcflurry import Class1PresentationPredictor

predictor = Class1PresentationPredictor.load()

df = predictor.predict_sequences(
    sequences={
        "protein_a": "MDSKGSSQKGSRLLLLLVVSNLL",
        "protein_b": "SSLPTPEDKEQAQQTHH",
    },
    alleles={
        "patient_1": ["HLA-A*02:01", "HLA-B*07:02"],
    },
    peptide_lengths=(8, 9, 10, 11),
    result="filtered",
    comparison_quantity="affinity_percentile",
    filter_value=2.0,
)
```

## Common pitfalls

- **Always quote allele names containing `*` in your shell.**
  In `zsh` and `bash`, an unquoted `HLA-A*02:01` is treated as a filename
  wildcard and can fail with `no matches found`. Write `'HLA-A*02:01'` (or
  shorthand `HLA-A0201`).
- **Know the difference between spaces and semicolons in `--alleles`.**
  - `--alleles 'HLA-A*02:01' 'HLA-B*07:02'` treats the two alleles as **two
    separate samples** (separate output rows for each).
  - `--alleles 'HLA-A*02:01;HLA-B*07:02'` treats them as **one genotype** and
    reports the strongest binder (`best_allele`).
- **Handling mixed or unrecognized allele names.**
  MHCflurry normalizes allele names with
  [`mhcgnomes`](https://github.com/pirl-unc/mhcgnomes) and raises a clear error
  if an allele name is invalid, class II, or unsupported. On messy batch tables,
  pass `--no-throw` (CLI) or `throw=False` (Python) to keep valid rows and
  return `NaN` for unrecognized alleles instead of stopping the run.
- **Skipping flanks when scanning synthetic constructs.**
  If your FASTA contains artificial concatenated peptides rather than real
  proteins, pass `--no-flanking` so artificial neighbor residues are not fed
  into the cleavage model.
