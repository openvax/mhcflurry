# scripts/training/

Production training pipeline for the pan-allele release. Every file
here has an enduring role; transient sweep cells, smoketests, and
one-off tuning runs do not belong here.

## Release pipeline (run in order, or use `pan_allele_release_full.sh`)

- **`pan_allele_release_affinity.sh`** — Stage 1. Trains the affinity
  ensemble end-to-end (data fetch → train → select → calibrate). Carries
  the heartbeat / write_snapshot / log_release_event instrumentation,
  `--continue-incomplete` resume, and the eval-against-public step.
- **`presentation_from_affinity.sh`** — Stages 2–3. Takes an existing
  affinity `models.combined/` and trains the with-flanks, no-flank, and short-flanks
  processing predictors, then fits + calibrates the presentation
  predictor on top. Use this as a tail-on after a sweep.
- **`pan_allele_release_full.sh`** — Composition wrapper that runs Stage
  1 then inlines Stages 2–3. The full release in one invocation.
- **`launch_pan_allele_training_remote.py`** — remote/cloud launcher for
  the same pan-allele training pipeline. It currently uses runplz, with
  Brev configuration when runplz is pointed at Brev instances. Use this
  when training remotely; keep machine-specific patching or interrupted-
  run recovery in ignored `jobs/`.

For Brev/runplz execution:

```bash
runplz --outputs-dir /path/to/output scripts/training/launch_pan_allele_training_remote.py
```

For the full release gate (training, comparison, plots, and deployment
validation), prefer `scripts/release/retrain_evaluate_deploy.sh`.

## Hyperparameter generation (consumed by the release scripts)

- **`release_exact/generate_hyperparameters.py`** — The 35-architecture
  pan-allele recipe. It defaults to minibatch 1024 and accepts
  `--minibatch-size` so release scripts and sweeps can change the value
  without patching the file.
- **`release_exact/generate_hyperparameters.base.py`** /
  **`generate_hyperparameters.variants.py`** — Processing-network
  hyperparameter base + with_flanks / no_flank / short_flanks variant emitters.
  The base generator accepts `--minibatch-size` and also defaults to 1024.
- **`release_exact/make_train_data.processing.py`** /
  **`make_train_data.presentation.py`** — Per-stage train-data
  preparation (annotated mass-spec hits, decoy generation, format
  filters). Run by the release scripts.
- **`release_exact/reassign_mass_spec_training_data.py`** — One-time
  remapping kept in tree because rerunning the release sometimes
  surfaces stale assignments and we'd want it again.
- **`release_exact/additional_alleles.txt`** — Curated allele list
  augmenting the auto-derived set; baked into the release.

## Sweep + analysis tooling

- **`full_ensemble_minibatch_sweep.sh`** — Production minibatch sweep.
  Phase-idempotent (`.train.done` / `.select.done` / `.calibrate.done` /
  `.eval.done` sentinels) and supports `MHCFLURRY_SCALE_LR`,
  `MHCFLURRY_SKIP_CALIBRATE` for the variants we routinely run.
- **`mhcflurry compare-models`** — Unified two-side comparator covering
  training stats (per-task wall-time / epoch / loss deltas), affinity
  (per-allele + per-length ROC/PR/PPV on `data_evaluation` monoallelic),
  processing, and presentation (per-sample + per-length on multiallelic
  flank modes). Each side can be a training-run directory, `public`
  (current install), or `public:<release_name>`. Default `--b public`.
  Writes detailed component artifacts plus `release_summary.csv` and
  `release_summary.md`.
- **`mhcflurry plot-model-comparison`** — Renders ROC/PR/scatter/delta
  plots from a `compare-models` output directory. Separate subcommand so
  the metric pipeline doesn't pay the matplotlib import cost.
- **`plot_minibatch_sweep.py`** — Stylized plots from a `sweep_summary.csv`
  (gradient-color dots by mb, lin-lin + log-log only, adjustText
  de-overlap). Invoked by the sweep wrapper after completion.
- **`plot_loss_curves.py`** — Per-architecture loss curves from a
  trained ensemble's `manifest.csv` + `weights_*.npz` series.

## Performance helpers (sourced, not invoked directly)

- **`set_cpu_threads.sh`** — Auto-computes the per-training-worker BLAS
  thread budget and uniformly sets `OMP_NUM_THREADS` /
  `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS`. Sourced by
  the release-stage scripts before they fork training workers.

## Profiling

- **`benchmark_training_profile.py`** — Thin CLI wrapper around
  `mhcflurry.training_benchmark`. Emits per-phase timings (data load,
  encode, fit, save) for any architecture. Used during perf
  regressions; the long-lived value is that it's the documented entry
  point if/when someone needs to repeat the analysis.

## Candidates for `downloads-generation/`

Keep scripts here while they are maintainer tooling. Move or wrap them
with a `downloads-generation/<download_name>/GENERATE.sh` once their
outputs are release artifacts that should be reproducible and
downloadable.

- **`mhcflurry compare-models`** (affinity + presentation components) —
  if the summary tables, plots, or row-level new-vs-public predictions
  are used as release evidence, make a generated analysis download that
  pins the new model paths, public download versions, data-evaluation
  version, git SHA, and command arguments.
- **`pan_allele_release_full.sh`** /
  **`pan_allele_release_affinity.sh`** /
  **`presentation_from_affinity.sh`** — once the 2.3.x recipe is final,
  fold the canonical recipe back into the relevant model
  `downloads-generation/` directories rather than relying only on this
  maintainer pipeline.
- **`full_ensemble_minibatch_sweep.sh`** and
  **`plot_minibatch_sweep.py`** — keep as scripts unless the sweep CSV,
  plots, or conclusions are published as a downloadable analysis
  artifact.

## What used to live here (deleted)

- `pan_allele_smoketest.sh`, `pan_allele_omp_smoketest.sh` — smoketests.
- `minibatch_sweep_experiment.sh`, `sweep_workers.sh`,
  `sweep_workers_cpu_extension.sh` — exploratory sweeps superseded by
  `full_ensemble_minibatch_sweep.sh`.
- `pan_allele_ensemble.sh`, `pan_allele_single.sh` — older single-
  ensemble + single-network runners superseded by the release pipeline.
- `pan_allele_presentation_subset.sh` — subset variant superseded by
  `presentation_from_affinity.sh`.
