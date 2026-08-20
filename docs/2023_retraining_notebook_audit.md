---
orphan: true
---

# 2023 retraining notebook audit

> Maintainer history, not a current user workflow. Use {doc}`evaluation` for
> current evaluation commands and the release-script README for current remote
> training and deployment behavior.

This audits the copied notebook bundle in `notebooks/2023-retraining/` against
the maintained release scripts and 2.3.x evaluation commands. The notebooks are
valuable source material, but they include generated data, checkpoint notebooks,
old imports, and local paths, so they should be ported into scripts rather than
committed as the workflow.

## What The Notebooks Cover

- Benchmark assembly: joins `data_evaluation` sample shards into wide
  monoallelic and multiallelic benchmark tables with MHCflurry, NetMHCpan, and
  MixMHCpred columns.
- Accuracy scoring: per-sample and per-length AUC / PPV tables, percent-change
  columns against NetMHCpan BA, NetMHCpan EL, MixMHCpred, and older MHCflurry.
- Figure generation: paper-style scatter panels, score bars, processing motifs,
  processing-vs-affinity correlation plots, and presentation coefficient plots.
- Supplemental outputs: sample tables, benchmark data files, model-selection
  accuracy workbook, antigen-processing motif workbook, and supplementary data
  CSV/XLSX files.
- Exploratory analyses: novel-allele analysis, proteasome mass-spec analysis,
  and an immunogenicity experiment depending on external local files.

## Differences From Maintained Workflows

- Data curation is split differently. Current maintained scripts use
  `downloads-generation/` to build raw downloads and `scripts/training/` to build
  release models. The notebooks start from already-generated downloads and create
  analysis artifacts under `notebooks/2023-retraining/artifacts/`.
- The current `data_evaluation` download generates train-excluded benchmark
  files and sample shards. The notebooks additionally join those shards into
  wide analysis tables and include an abandoned no-exclude-train path.
- The notebooks label recent multiallelic samples using PMIDs `31844290` and
  `31154438`. Current presentation training excludes `31844290`, `31495665`, and
  `31154438`; this needs one named holdout policy before new public weights are
  trained and evaluated.
- The notebooks use old notebook-era dependencies and names (`keras`,
  `tensorflow`, `mhcnames` in places). Ports should use the current PyTorch code,
  `mhcgnomes` / existing allele normalization helpers, and maintained CLI
  commands.
- Current evaluation commands cover affinity, processing, and presentation
  metrics, diagnostic plots, reusable AUC/PPV score caches, and paper-style
  figures. The optional `external-predictors` adapter can invoke locally
  installed NetMHCpan/MixMHCpred runners through `mhctools`; binaries and
  licenses remain outside MHCflurry. Exact curated split tables, external
  supplemental sources, and hand-authored architecture artwork remain separate
  inputs.
- The processing-variant ambiguity has been resolved. The full release trains
  `with_flanks`, `no_flank`, and `short_flanks`, and presentation uses the true
  `with_flanks` predictor by default.
- Historical runplz files in `jobs/` patched scripts in place for one experiment
  and called deleted comparison scripts. Remote execution is now represented by
  maintained scripts instead of patching a remote copy.

## Current Coverage And Remaining Inputs

- Predictor metadata / aesthetics: `paper-figures render` accepts
  `predictor_info.csv` and otherwise uses the maintained `figure_style.py`
  labels and palette for MHCflurry, NetMHCpan 4.0 / 4.2, and MixMHCpred.
  Saved-prediction scoring also uses `predictor_info.csv` for custom score
  orientation through a `higher_is_better` column.
- Wide benchmark tables: saved monoallelic and multiallelic prediction tables
  can be passed directly with `--monoallelic-predictions` and
  `--multiallelic-predictions`. The optional `external-predictors` adapter can
  populate canonical columns through a locally installed runner; tables
  produced by other workflows remain valid inputs.
- Accuracy score tables: `eval paper-figures score-predictions` derives the
  notebook-style AUC / PPV / percent-change tables from canonical saved
  prediction tables. `paper-figures render` can also derive them in-process when
  a cache table is absent.
- Monoallelic figures: current-vs-public scatter panels are generated from
  `compare-models`; external-predictor and novel-allele panels are generated
  when `accuracy_scores.monoallelic.csv` and
  `accuracy_scores.monoallelic.novel_alleles.csv` are supplied.
- Multiallelic figures: PPV/AUC scatter panels, percent-change bars, and
  presentation-vs-baseline panels are generated from saved multiallelic scores
  or prediction tables. Recent-vs-old sample grouping is enabled by
  `--sample-table`.
- Model-selection evidence: locus score bars are generated from
  `model_selection_accuracy.csv`, or a current comparison fallback is generated
  from `compare-models` when that workbook is absent.
- Processing evidence: cysteine-removed AP panels, AP motif/logo panels, and AP
  correlation panels are generated when their 2023 source tables exist; motif
  and correlation fallbacks are generated from current training / saved
  prediction artifacts where possible.
- Proteasome and model-info figures: proteasome plots are generated from
  `proteasome_mass_spec.csv`, `Additional File 8.csv`, or a run's
  `processing/hits_with_tpm.csv.bz2`. Architecture/model-info figures copy
  supplied artwork or generate a run-manifest fallback.
- Still deferred: the immunogenicity experiment and any exact supplemental
  source that depends on external files not present in this repository.

## Decisions and remaining work

1. Notebooks remain references rather than the maintained workflow.
2. Benchmark prediction tables use one canonical layout: one row per evaluated pMHC, a
   `hit` label, `sample_id` or allele columns, optional `sample_group`, and one
   numeric column per predictor. Allele parsing / normalization should use
   `mhcgnomes` or existing MHCflurry helpers, not ad hoc string matching.
3. External predictor binaries remain outside MHCflurry's core dependencies.
   The maintained adapter invokes an optional local runner and imports canonical
   saved columns.
4. The 2.3.0 holdout policy uses component-appropriate identities. Affinity is
   evaluated on all 103 monoallelic benchmark samples plus the multiallelic
   final holdout; every current-training `(allele, peptide)` found in those
   benchmark rows (hits or decoys) is removed, expanding multiallelic genotypes
   to every listed allele. Processing and presentation are evaluated on the 10
   multiallelic samples from PMID 31154438, a source study absent from
   monoallelic processing training. Presentation excludes those whole samples
   from training. This avoids the invalid alternative of excluding all 179
   `data_evaluation` samples, which are exactly the full annotated MS input and
   would leave no processing or presentation training data. The workflow
   persists/checksums the manifests, filters final evaluation to them, and
   fails if any specified training overlap remains.
5. Processing variant naming is resolved: the release trains a real
   `models.selected.with_flanks` artifact and uses it for presentation.
6. Release retraining is available as one command:
   `mhcflurry train pan-allele-release`. It trains, evaluates, plots, syncs
   remote artifacts, and leaves deployment opt-in through `--deploy-mode`.

## Current Single-Command Entry Point

Local:

```bash
mhcflurry train pan-allele-release \
    --run-dir /path/to/release-run \
    --release 2.3.0 \
    --backend local
```

Brev through existing runplz capacity:

```bash
mhcflurry train pan-allele-release \
    --run-dir /path/to/release-run \
    --release 2.3.0 \
    --backend brev-existing
```

Generic SSH machine:

```bash
mhcflurry train pan-allele-release \
    --run-dir /path/to/local-copy \
    --release 2.3.0 \
    --backend ssh \
    --remote user@host \
    --remote-repo /path/to/mhcflurry \
    --remote-run-dir /path/to/remote-release-run
```
