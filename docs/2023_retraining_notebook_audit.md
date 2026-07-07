# 2023 Retraining Notebook Audit

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
- Current `mhcflurry eval compare-models` covers affinity, processing, and
  presentation regression metrics, and `mhcflurry eval plot-comparison` renders
  diagnostic plots. `mhcflurry eval paper-figures render` can render paper-style
  panels from current compare output, saved score tables, and saved prediction
  tables. `mhcflurry eval paper-figures score-predictions` materializes the
  derived AUC / PPV cache tables so repeated figure runs do not recompute them.
  The remaining non-MHCflurry prerequisites are external-predictor execution
  (NetMHCpan / MixMHCpred binaries and licenses), exact curated split tables,
  external supplemental source files, and hand-authored architecture artwork.
- The current full training script trains processing `no_flank` and
  `short_flanks` variants, then uses `short_flanks` as the with-flank processing
  component for presentation. Older download-generation trained a separate
  `models.selected.with_flanks` variant. That semantic difference should be
  resolved explicitly before publishing 2.3.0 weights.
- Historical runplz files in `jobs/` patched scripts in place for one experiment
  and called deleted comparison scripts. Remote execution is now represented by
  maintained scripts instead of patching a remote copy.

## Current Coverage And Remaining Inputs

- Predictor metadata / aesthetics: `paper-figures render` accepts
  `predictor_info.csv` and otherwise uses the maintained `figure_style.py`
  labels and palette for MHCflurry, NetMHCpan 4.0 / 4.2, and MixMHCpred.
- Wide benchmark tables: saved monoallelic and multiallelic prediction tables
  can be passed directly with `--monoallelic-predictions` and
  `--multiallelic-predictions`. The command does not yet run NetMHCpan /
  MixMHCpred itself; those tools should write canonical saved prediction columns
  first.
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

## Porting Plan

1. Keep notebooks out of the maintained path; use them only as references.
2. Keep benchmark prediction tables canonical: one row per evaluated pMHC, a
   `hit` label, `sample_id` or allele columns, optional `sample_group`, and one
   numeric column per predictor. Allele parsing / normalization should use
   `mhcgnomes` or existing MHCflurry helpers, not ad hoc string matching.
3. Run external predictor binaries outside MHCflurry's core package, then import
   their saved prediction columns into the canonical tables. This keeps licensed
   tools and local paths out of the release code.
4. Decide the 2.3.0 holdout policy in one place: PMIDs, pMHC overlap removal,
   and whether evaluation data is filtered out of affinity / processing /
   presentation training. Move any useful logic from `jobs/filter_training...`
   into maintained scripts after that decision.
5. Resolve the processing variant naming: either train a real
   `models.selected.with_flanks` variant again or document and test that
   `short_flanks` is the canonical with-flank presentation input for 2.3.x.
6. Make the release gate one command:
   `scripts/release/retrain_evaluate_deploy.sh` trains, evaluates, plots, and
   runs deployment validation. It supports local execution, existing
   Brev/runplz capacity, and SSH-backed remote machines.

## Current Single-Command Entry Point

Local:

```bash
scripts/release/retrain_evaluate_deploy.sh \
    --run-dir /path/to/release-run \
    --release 2.3.0 \
    --backend local \
    --deploy-mode dry-run
```

Brev through existing runplz capacity:

```bash
scripts/release/retrain_evaluate_deploy.sh \
    --run-dir /path/to/release-run \
    --release 2.3.0 \
    --backend brev-existing \
    --deploy-mode dry-run
```

Generic SSH machine:

```bash
scripts/release/retrain_evaluate_deploy.sh \
    --run-dir /path/to/local-copy \
    --release 2.3.0 \
    --backend ssh \
    --remote user@host \
    --remote-repo /path/to/mhcflurry \
    --remote-run-dir /path/to/remote-release-run \
    --deploy-mode dry-run
```
