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

| Stage | Command | Main outputs |
|---|---|---|
| Metrics | `eval compare-models` | Component CSV/JSON files, `release_summary.csv`, and `release_summary.md` |
| Diagnostics | `eval plot-comparison` | ROC, precision-recall, scatter, and delta plots plus an optional combined PDF |
| Paper figures | `eval paper-figures render` | SVG/PDF/PNG panels, `paper_figures.pdf`, `manifest.csv`, and `missing_inputs.md` |

The metrics directory is the reusable contract between evaluation and plotting.
Keep it when iterating on figure style or assembling a review packet.

Keep paper figures in their own directory (the default is
`<out>/plots/paper_figures`). The combined diagnostic `--summary-pdf` may be a
top-level file under `<out>/plots` or live outside the plot tree, but it cannot
be placed inside the paper-figure, affinity, processing, presentation, or
diagnostic-paper subdirectories. Commands reject overlapping output paths
before clearing or rendering anything.

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

## Release and remote runs

`mhcflurry train pan-allele-release` runs comparison and diagnostic plotting on
the training machine before copying results home. Paper inputs that depend on
locally licensed tools can remain on the control machine and be rendered after
the remote artifacts arrive.

Release maintainers should use the
[release workflow guide](https://github.com/openvax/mhcflurry/tree/master/scripts/release)
for lifecycle, synchronization, and deployment options. The complete argument
reference is in {doc}`commandline_tools` and `mhcflurry eval --help`.
