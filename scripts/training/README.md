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
  1 then inlines Stages 2–3. The full release in one invocation. Its affinity
  stage leaves `AFFINITY_MAX_WORKERS_PER_GPU=auto` so the affinity training
  command can choose worker packing from the hyperparameter grid, row count,
  minibatch, and detected VRAM. Set `AFFINITY_MAX_WORKERS_PER_GPU` to a number
  only when deliberately pinning a known-good count for a specific machine.
- **`launch_pan_allele_training_remote.py`** — remote/cloud launcher for
  the same pan-allele training pipeline. It uses runplz as the transport,
  with Brev provisioning behavior controlled by `RUNPLZ_BREV_*`
  environment variables. When invoked by the release workflow wrapper, it also
  runs `mhcflurry compare-models` and `mhcflurry plot-model-comparison` on the
  remote GPU machine before cleanup so release-scale inference does not fall
  back to the local laptop. Prefer the release workflow wrapper below unless
  you are debugging the remote launcher itself.

For Brev/runplz execution:

```bash
RUNPLZ_BREV_AUTO_CREATE=0 \
runplz brev \
    --outputs-dir /path/to/output \
    --instance existing-brev-instance \
    scripts/training/launch_pan_allele_training_remote.py
```

For the full release gate (training, comparison, plots, and deployment
validation), prefer `scripts/release/retrain_evaluate_deploy.sh`.

## Hyperparameter generation (consumed by the release scripts)

- **`mhcflurry class1-generate-training-hyperparameters`** — package-owned
  generator for the 35-architecture pan-allele affinity recipe and the
  processing-network base / with_flanks / no_flank / short_flanks variants.
  It defaults to minibatch 1024 and accepts `--minibatch-size` for the base
  grids so release scripts and sweeps can change the value without patching
  files. The `release_exact/generate_hyperparameters*.py` files are thin
  compatibility shims for older direct-script workflows.
- **`release_exact/make_train_data.processing.py`** /
  **`make_train_data.presentation.py`** — Per-stage train-data
  preparation (annotated mass-spec hits, decoy generation, format
  filters). Run by the release scripts.
- **`mhcflurry class1-reassign-mass-spec-training-data`** — one-time
  mass-spec affinity-value remapping kept as a real CLI because rerunning
  the release sometimes surfaces stale assignments and we'd want it again.
  `release_exact/reassign_mass_spec_training_data.py` remains as a
  compatibility shim.
- **`release_exact/additional_alleles.txt`** — archived curated allele list
  from the older release recipe. The maintained rc14 release scripts do not
  currently read it.

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
  (current install), or `public:<release_name>`. The release wrapper defaults
  to comparing new weights against `public:2.0.0`, while the bare command still
  defaults `--b public`.
  Prediction uses the same GPU-aware parallel worker planner as the inference
  CLIs; metric aggregation remains in pandas / scikit-learn for exact release
  summary compatibility. Writes detailed component artifacts plus
  `release_summary.csv` and `release_summary.md`.
- **`mhcflurry plot-model-comparison`** — Renders ROC/PR/scatter/delta
  diagnostics plus paper-style per-allele, per-sample, per-length, and
  release-summary panels from a `compare-models` output directory. It writes
  per-plot PDFs next to PNGs and can collect the vector outputs into a single
  PDF via `--summary-pdf`. Separate subcommand so the metric pipeline doesn't
  pay the matplotlib import cost.
- **`mhcflurry paper-figures`** — Ports the 2023 retraining-notebook figure
  suite into a reproducible CLI. It consumes notebook-style artifacts such as
  `accuracy_scores.multiallelic.csv`, `predictor_info.csv`, monoallelic/AP
  score tables, motif workbooks, and architecture artwork; writes SVG/PDF/PNG
  panels plus a vector multi-page PDF; and records unavailable figure families
  in `manifest.csv` / `missing_inputs.md` instead of silently fabricating them.
  Its comparator predictors are configurable (`--candidate-predictor`,
  `--external-baselines`, `--preferred-predictors`) with the 2023 predictor set
  as the default.
- **`plot_minibatch_sweep.py`** — Stylized plots from a `sweep_summary.csv`
  (gradient-color dots by mb, lin-lin + log-log only, adjustText
  de-overlap). Invoked by the sweep wrapper after completion.
- **`plot_loss_curves.py`** — Per-architecture loss curves from a
  trained ensemble's `manifest.csv` + `weights_*.npz` series.

## Performance helpers (sourced, not invoked directly)

- **`set_cpu_threads.sh`** — Auto-computes the per-training-worker BLAS
  thread budget and uniformly sets `OMP_NUM_THREADS` /
  `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS`. It also pins
  `MKL_THREADING_LAYER=GNU` before Python starts so numpy/MKL and
  PyTorch/Inductor workers share GNU OpenMP instead of aborting on mixed
  runtimes. Sourced by the release-stage scripts before they fork training
  workers.

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
