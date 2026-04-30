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
  affinity `models.combined/` and trains the no-flank + short-flanks
  processing predictors, then fits + calibrates the presentation
  predictor on top. Use this as a tail-on after a sweep.
- **`pan_allele_release_full.sh`** — Composition wrapper that runs Stage
  1 then inlines Stages 2–3. The full release in one invocation.

## Hyperparameter generation (consumed by the release scripts)

- **`release_exact/generate_hyperparameters.py`** — The 35-architecture
  pan-allele recipe (lr=1e-3, mb=128, 1024×512 dense, with-skip-
  connections). Pinned bit-for-bit with the 2.2.0 release.
- **`release_exact/generate_hyperparameters.base.py`** /
  **`generate_hyperparameters.variants.py`** — Processing-network
  hyperparameter base + no_flank / short_flanks variant emitters.
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
- **`compare_new_vs_public.py`** — Multi-GPU eval against the public 2.2.0
  release on the `data_evaluation` benchmark. Per-allele + per-peptide-
  length micro/macro ROC AUC, PR AUC, PPV@N. Used by the sweep eval
  phase and as a standalone tool.
- **`compare_runs.py`** — Two-run comparator (read each run's
  `manifest.csv` + `eval_comparison/` outputs, emit markdown table +
  CSV). Useful any time you train a new ensemble and want a side-by-
  side against an older one.
- **`plot_minibatch_sweep.py`** — Stylized plots from a `sweep_summary.csv`
  (gradient-color dots by mb, lin-lin + log-log only, adjustText
  de-overlap). Invoked by the sweep wrapper after completion.
- **`plot_loss_curves.py`** — Per-architecture loss curves from a
  trained ensemble's `manifest.csv` + `weights_*.npz` series.

## Performance helpers (sourced, not invoked directly)

- **`set_cpu_threads.sh`** — Auto-computes the per-training-worker BLAS
  thread budget and uniformly sets `OMP_NUM_THREADS` /
  `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS`. Sourced by
  `pan_allele_release_affinity.sh`.

## Profiling

- **`benchmark_training_profile.py`** — Thin CLI wrapper around
  `mhcflurry.training_benchmark`. Emits per-phase timings (data load,
  encode, fit, save) for any architecture. Used during perf
  regressions; the long-lived value is that it's the documented entry
  point if/when someone needs to repeat the analysis.

## What used to live here (deleted)

- `pan_allele_smoketest.sh`, `pan_allele_omp_smoketest.sh` — smoketests.
- `minibatch_sweep_experiment.sh`, `sweep_workers.sh`,
  `sweep_workers_cpu_extension.sh` — exploratory sweeps superseded by
  `full_ensemble_minibatch_sweep.sh`.
- `pan_allele_ensemble.sh`, `pan_allele_single.sh` — older single-
  ensemble + single-network runners superseded by the release pipeline.
- `pan_allele_presentation_subset.sh` — subset variant superseded by
  `presentation_from_affinity.sh`.
