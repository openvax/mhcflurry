(evaluation)=

# Evaluating trained models

Use the evaluation commands after training a model to compare it with a public
release or another run. The normal workflow separates reusable metrics from
plot rendering, so you can change figures without rerunning predictions.

## Quick evaluation

Compare a candidate run with the installed public models:

```shell
mhcflurry eval compare-models \
    --a results/new_run \
    --b public \
    --out results/new_run/eval_comparison
```

Render the diagnostic plots and combined PDF:

```shell
mhcflurry eval plot-comparison \
    --input results/new_run/eval_comparison \
    --summary-pdf results/new_run/eval_comparison/plots/model_comparison_figures.pdf
```

Use `public:<release_name>` instead of `public` when the baseline version must
be fixed. Components are compared only when both sides contain the corresponding
model artifacts; explicitly requested processing modes must exist on both sides.

## Outputs

For processing pools that cannot supply ten distinct negatives per hit, expand
and freeze a single cohort before comparing models:

```shell
mhcflurry eval prepare-processing-cohort \
    --data-dir DATA_EVALUATION \
    --release-holdout-dir results/new_run/release_holdout \
    --affinity-predictor PUBLIC_MODELS_CLASS1_PAN/models.combined \
    --proteome-reference-csv HUMAN_PROTEOME.csv.bz2 \
    --out results/processing_cohort

mhcflurry eval compare-models \
    --a results/new_run --b public:2.2.0 \
    --include processing --data-dir DATA_EVALUATION \
    --release-holdout-dir results/new_run/release_holdout \
    --processing-matched-cohort results/processing_cohort \
    --out results/new_run/processing_comparison
```

The preparation command retains every held-out hit and the original cached
public affinities, samples additional candidates only for unresolved lengths,
and scores them with the frozen reference. It first checks that this reference
reproduces 256 spread-out cached predictions per sample within 0.0001 log10
units. Input, model and output checksums, the checked rows, seeds, scored rounds
and unique assignments are saved. `--resume` reuses verified rounds; insufficient
capacity remains an error. This processing-only expansion does not alter the
presentation benchmark. The remote launcher accepts the resulting directory
through `PROCESSING_EVALUATION_COHORT`.

For external comparisons, pass `--coverage common` to
`mhcflurry eval presentation-external-predictors`. Every table, paired interval
and figure then uses identical rows across all requested models. The coverage
table retains original missing-score counts and common exclusions; provenance
records original and scored hit/row counts. Every sample must remain represented
with both classes. The default `available` mode preserves per-model coverage
and pairwise intersections for diagnostic use. All requested NetMHCpan versions,
both BA and EL, receive paired comparisons.

| Stage | Command | Main outputs |
|---|---|---|
| Metrics | `eval compare-models` | Component CSV/JSON files, `release_summary.csv`, and `release_summary.md` |
| Diagnostics | `eval plot-comparison` | ROC, precision-recall, scatter, and delta plots plus an optional combined PDF |
| Paper figures | `eval paper-figures render` | SVG/PDF/PNG panels, `paper_figures.pdf`, `manifest.csv`, and `missing_inputs.md` |

The metrics directory is the reusable contract between evaluation and plotting.
Keep it when iterating on figure style or assembling a review packet.

Report raw-score and percentile metrics separately, recording the calibration
method and background. Preserve public baselines unchanged; see
{doc}`shared_percent_rank_transforms` for controlled recalibration comparisons.

Affinity comparison summaries include a `benchmark_identity` hash calculated
after holdout selection, allele intersection, peptide-length filtering, and
training-overlap exclusion. A saved prediction column can be reused without
rerunning a baseline only when that identity matches exactly:

```shell
mhcflurry eval compare-models \
    --a results/candidate/models.combined \
    --b public \
    --b-affinity-predictions results/public-comparison/affinity/predictions.csv.bz2 \
    --b-affinity-prediction-column b_pred \
    --out results/candidate/comparison-vs-public
```

For affinity-factorial finalists, `mhcflurry eval affinity-candidate-figures`
combines the direct, row-identical candidate/public prediction tables into one
monoallelic score table and figure suite. Its paginated AUROC, AUPRC, and PPV
grids show every requested candidate against every available baseline, and its
overview ranks all predictors by macro allele-level metrics. Canonical
NetMHCpan BA/EL and MixMHCpred columns are retained when present.

Use `--external-predictions` to add a benchmark-aligned table containing
NetMHCpan BA/EL or MixMHCpred scores. The table can be built from official
per-sample `data_evaluation` groups with `mhcflurry eval
merge-external-predictions`; both commands validate stable row identity and
record input hashes and coverage in figure provenance.

Keep paper figures in their own directory (the default is
`<out>/plots/paper_figures`). The combined diagnostic `--summary-pdf` may be a
top-level file under `<out>/plots` or live outside the plot tree, but it cannot
be placed inside the paper-figure, affinity, processing, presentation, or
diagnostic-paper subdirectories. Commands reject overlapping output paths
before clearing or rendering anything.

## Count-matched processing ensembles

To compare a four-network candidate with an eight-network public ensemble,
evaluate all four-of-eight public subsets rather than choosing a subset using
release-holdout performance:

```shell
mhcflurry eval processing-ensemble-subsets \
    --input results/matched/matched_predictions.csv.bz2 \
    --models-dir /path/to/public/models.selected.short_flanks \
    --subset-size 4 --reference-score public_5aa \
    --comparison-score new_legacy_cnn --comparison-score new_boundary_5x5 \
    --out results/public_four_network_subsets
```

This source-checkout command requires strict length/affinity-matched risk
sets, caches each network's predictions, verifies reconstruction against the
named full-ensemble score, and preserves all subset scores and memberships.
It reports median, range and quartiles across the complete subset set. This
spread measures ensemble composition sensitivity; it is not a confidence
interval or a validation-based model selection procedure. Matching network
count does not match the historical training or architecture-search budget.

Use a fresh output directory. `--member-cache-dir` can reuse an earlier run's
predictions after checking input, model and execution identities. Cross-device
score tolerances must be explicitly justified and recorded; do not increase
`--verification-atol` to hide changed weights or a misaligned prediction table.

## Paper-style figures

For an already-trained local model, compose comparison, diagnostics, and any
available paper panels with one command:

```shell
mhcflurry eval paper-figures run \
    --a results/new_run \
    --b public \
    --out results/new_run/eval_comparison
```

Paper panels can also use cached benchmark predictions or score tables. A saved
prediction table has one row per evaluated peptide–MHC example and contains:

- `hit`;
- `sample_id` for multiallelic data, or `allele`/`hla` for monoallelic data;
- optional peptide and flank metadata; and
- canonical predictor score columns, or custom score columns declared in
  `predictor_info.csv`.

Derive reusable AUC and PPV score tables with:

```shell
mhcflurry eval paper-figures score-predictions \
    --kind multiallelic \
    --input benchmark.multiallelic.csv.bz2 \
    --out accuracy_scores.multiallelic.csv
```

Score direction must be explicit. Common MHCflurry, NetMHCpan, and MixMHCpred
column names have built-in orientation. Describe custom columns in
`predictor_info.csv` using `predictor` and `higher_is_better`; those rows also
select the custom columns. `--predictor-columns` can restrict scoring to an
explicit subset, but custom names still need their direction in
`predictor_info.csv`. Other numeric columns are treated as metadata rather than
guessed to be predictor scores.

Optional licensed predictors run outside the MHCflurry core package. The
`paper-figures external-predictors` adapter can invoke a locally installed
runner such as `mhctools` and join its output into the canonical benchmark
table. Missing optional inputs are listed in `missing_inputs.md`; they are not
silently replaced with synthetic panels.

## External predictor comparison

`mhcflurry eval presentation-external-predictors` compares saved compare-models
scores with the NetMHCpan 4.0 BA, NetMHCpan 4.0 EL and MixMHCpred columns
distributed in `data_evaluation`. It runs no predictor:

```shell
mhcflurry eval presentation-external-predictors \
    --comparison-dir results/new_run/eval_comparison \
    --data-dir "$(mhcflurry-downloads path data_evaluation)" \
    --cohort multiallelic \
    --a-label "MHCflurry 2.3.0" --b-label "MHCflurry 2.2" \
    --out results/new_run/external_comparison
```

Rows join by benchmark source file and row identity, with genotypes
canonicalized as compare-models saves them; any unmatched row fails the command.
`data_evaluation` ships NetMHCpan 4.0 BA/EL and MixMHCpred columns. Pass
`--external-dir` once per directory of locally generated scores, such as
NetMHCpan 4.2, whose files may be plain CSV. Read any NetMHCpan 4.1 or 4.2
result as an optimistic bound: both postdate this holdout's 2019 source study
and are not train-excluded against it, unlike NetMHCpan 4.0. `multiallelic` uses the saved presentation scores with
and without flanks; `monoallelic` uses saved affinity predictions (pass
`--skip-joined-table` for that large cohort). Metrics use the compare-models
definitions. Each predictor is scored on the rows it covers and each paired
comparison on rows both predictors score, so MHCflurry-only comparisons match
compare-models exactly; `coverage.csv` counts unscored rows. Paired intervals resample whole samples (10000 draws, seed 42 by
default) and are exploratory. Two reference baselines are
included by default: seeded random scores, and a logistic regression on the
one-hot first and last four residues with no MHC or flank input, fitted
leave-one-sample-out so no sample's labels reach its own scores. Pass
`--baselines none` to skip them. Outputs include per-sample, macro and pooled
metrics, paired differences, a joined score table, `external_comparison.pdf`
with PNG pages, `summary.md` and a provenance manifest.

## Release and remote runs

`mhcflurry train pan-allele-release` runs comparison and diagnostic plotting on
the training machine before copying results home. Paper inputs that depend on
locally licensed tools can remain on the control machine and be rendered after
the remote artifacts arrive.

Release maintainers should use the
[release workflow guide](https://github.com/openvax/mhcflurry/tree/master/scripts/release)
for lifecycle, synchronization, and deployment options. The complete argument
reference is in {doc}`commandline_tools` and `mhcflurry eval --help`.
