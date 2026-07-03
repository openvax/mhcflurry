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
- Current `mhcflurry compare-models` covers affinity and presentation regression
  metrics, and `mhcflurry plot-model-comparison` renders generic plots. It does
  not yet recreate the paper-style benchmark tables, sample-group comparisons,
  model-selection workbook, motif workbook, or supplemental figures.
- The current full training script trains processing `no_flank` and
  `short_flanks` variants, then uses `short_flanks` as the with-flank processing
  component for presentation. Older download-generation trained a separate
  `models.selected.with_flanks` variant. That semantic difference should be
  resolved explicitly before publishing 2.3.0 weights.
- Historical runplz files in `jobs/` patched scripts in place for one experiment
  and called deleted comparison scripts. Remote execution is now represented by
  maintained scripts instead of patching a remote copy.

## Missing Figure And Table Ports

- Predictor metadata / aesthetics: convert `0 aesthetics.ipynb` into a small
  `predictor_info.csv` generator with stable labels, colors, and descriptions.
- Wide benchmark tables: port the monoallelic and multiallelic shard-join logic
  from `1 prepare benchmark dataset*.ipynb`.
- Accuracy score tables: port the AUC / PPV / percent-change tables from
  `2 monoallelic accuracy plots.ipynb` and `2 multiallelic accuracy.ipynb`.
- Monoallelic figures: BA/EL/MixMHCpred scatter panels, training-count vs AUC,
  and novel-allele comparison tables / plots.
- Multiallelic figures: PPV/AUC scatter panels comparing NetMHCpan BA,
  NetMHCpan EL, MixMHCpred, older MHCflurry, and presentation score variants,
  including recent-vs-old sample grouping.
- Model-selection evidence: port the unselected-model held-out decoy AUC workbook
  and HLA locus score bars.
- Processing evidence: port antigen-processing motif count/PWM workbook,
  processing logo figure, and processing-vs-affinity correlation plots.
- Supplemental outputs: port sample table, supplemental sample table with
  accuracies, benchmark supplemental CSVs, and proteasome mass-spec additional
  file generation.
- Immunogenicity experiment: defer until external source files are located and
  licensed; do not make this a release gate yet.

## Porting Plan

1. Add `scripts/analysis/` for reusable release-analysis scripts. Keep notebooks
   out of the maintained path; use them only as references.
2. Extract shared metric helpers: PPV@N, sign normalization, bootstrap intervals,
   percent-change calculations, and sample/length grouping.
3. Add a benchmark-assembly script that consumes a `data_evaluation` directory
   and writes wide monoallelic/multiallelic analysis CSVs from the group files.
4. Add an accuracy-table script that writes the notebook-style
   `accuracy_scores.*.csv` tables from those wide benchmarks.
5. Add figure scripts that consume the generated tables and produce stable PNG /
   PDF outputs under a release-analysis directory.
6. Add processing-motif generation from the presentation benchmark and trained
   processing predictors. This should be scriptable without `logomaker` unless
   the logo plot is explicitly requested; the workbook can be generated first.
7. Decide the 2.3.0 holdout policy in one place: PMIDs, pMHC overlap removal,
   and whether evaluation data is filtered out of affinity / processing /
   presentation training. Move any useful logic from `jobs/filter_training...`
   into maintained scripts after that decision.
8. Resolve the processing variant naming: either train a real
   `models.selected.with_flanks` variant again or document and test that
   `short_flanks` is the canonical with-flank presentation input for 2.3.x.
9. Make the release gate one command:
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
