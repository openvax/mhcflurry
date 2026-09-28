# Training scripts

Maintained scripts for building the pan-allele release models. For a complete
retrain, evaluation, synchronization, and optional deployment, start with:

```shell
mhcflurry train pan-allele-release --help
```

The files in this directory are the lower-level stages and profiling tools used
by that command. One-off experiments and machine-specific launchers belong in
the ignored `jobs/` directory, not here.

## Release training stages

- **`pan_allele_release_affinity.sh`** trains, selects, and calibrates the
  affinity ensemble. It supports incomplete-run continuation and writes
  heartbeat, snapshot, event, and evaluation records.
- **`presentation_from_affinity.sh`** starts from an affinity
  `models.combined/`, trains the configured processing variants, then fits and
  calibrates presentation models.
- **`pan_allele_release_full.sh`** runs both stages in order for a complete
  local training pass.
- **`run_release_affinity_ablations.sh`** and
  **`run_release_processing_ablations.sh`** run the small, paired parity panels
  described in the neural hyperparameter audit before a full release retrain.
- **`launch_pan_allele_training_remote.py`** transports the same stages through
  runplz. Use it directly only when debugging transport; normal remote releases
  should use `mhcflurry train pan-allele-release`.

Affinity, processing, and presentation stages write persistent GPU telemetry.
Worker packing defaults to workload-aware `auto`; pin a count only for a measured
machine-specific benchmark.

## Reproduce the 2.3.0 weights

Use the frozen `final-2.3.0-candidate-v2` recipe identifier. It selects the
released settings; the identifier is retained for reproducibility and does not
indicate the package release status. See the
[recipe](../../docs/release_training_recipe.md) for data and component settings
and the [launch guide](../../docs/final_230_candidate_experiment.md) for local and
Modal execution, resumption and artifact collection. The original v1 preset is
retained for reproducing earlier experiments and is not the 2.3.0 weight recipe.

For transport debugging, `launch_pan_allele_training_remote.py` accepts
`MHCFLURRY_REMOTE_WORKFLOW=full` (default), `affinity-ablations` or
`processing-ablations`. Use a persistent remote output directory and preserve
the recorded source commit and generated configuration.

## Training data and hyperparameters

- **`mhcflurry class1-generate-training-hyperparameters`** generates the
  maintained affinity and processing grids. The affinity minibatch defaults to
  the published value 128; processing defaults to 512. Both can be changed with
  `--minibatch-size`.
- **`release_exact/generate_hyperparameters*.py`** are compatibility shims for
  historical direct-script workflows.
- **`release_exact/make_train_data.processing.py`** and
  **`make_train_data.presentation.py`** prepare annotated hits, decoys, and
  model-family input tables.
- **`mhcflurry class1-reassign-mass-spec-training-data`** reruns the maintained
  mass-spec affinity remapping step. Its file under `release_exact/` is a
  compatibility shim.

## Evaluation

Model comparison and plotting are package commands, not training scripts. Use
`mhcflurry eval ...` after a standalone stage run, or let
`mhcflurry train pan-allele-release` invoke them automatically.

See the [evaluation guide](../../docs/evaluation.md) for comparison outputs,
paper figures, saved-prediction tables, and external predictors.

## Sweeps and profiling

- **`full_ensemble_minibatch_sweep.sh`** runs resumable minibatch and validation
  batch experiments. It writes per-cell completion sentinels, summaries, and GPU
  occupancy data. Defaults stay on the automatic worker and validation-batch
  paths; explicit values are for controlled comparisons.
- **`plot_minibatch_sweep.py`** renders throughput and loss plots from
  `sweep_summary.csv`.
- **`run_affinity_factorial.sh`** trains controlled affinity-recipe conditions,
  saves per-epoch histories and held-out predictions, and compares every
  condition directly to an explicitly supplied public
  `models.no_additional_ms` predictor. The direct comparisons exclude union
  training overlap, assert one common cohort identity, and render one review
  PDF per condition. **`evaluate_affinity_factorial_public.sh`** applies the
  same evaluation/figure gate to an already-trained factorial directory.
- **`mhcflurry eval affinity-candidate-figures`** combines shortlisted
  conditions, public 2.2, and any available canonical NetMHCpan/MixMHCpred
  columns into one reusable held-out prediction table and paper-figure suite.
- **`mhcflurry train compose-processing-ensemble`** builds a fixed processing
  ensemble while hashing every source predictor.
- **`mhcflurry train processing-kernel-sweep`** trains widths 5/7/9/11/13/15
  for legacy 5-aa and boundary 5x5 families (48 fits, four shared folds).
  Requires verified matched training negatives, restores best checkpoints,
  uses explicit X context padding and preserves per-fold/per-member predictions.
  See `docs/processing_kernel_sweep.md`; the bounded single-A100 launcher is
  `launch_processing_kernel_sweep_modal.py` (`runplz modal --detach`).
- **`mhcflurry train processing-hyperparameter-sweep --design training-recipe`**
  runs the 64-fit optimizer/initialization/batch factorial with explicit
  processing LSUV, independent best/terminal checkpoints, paired frozen folds
  and per-member state predictions. Use `--evaluation none` for development
  screening. Width recovery imports complete conditions without resampling.
  See `docs/processing_hyperparameter_campaign.md` for recovery, budget controls
  and the distinction between screening and compact-ensemble selection.
- **`mhcflurry train processing-hyperparameter-sweep --design ranking-confirmation`**
  runs the six-condition, 24-fit width/optimizer/checkpoint panel with
  sample-disjoint inner ranking monitoring and retained best/best-AP/terminal
  states. **`mhcflurry eval processing-confirmation-analysis`** applies the
  recorded paired gate to a collected snapshot and exports the candidate
  recipe; `generate_processing_recipe.confirmed_processing_candidate_hyperparameters()`
  defines the short-flank settings included in the frozen 2.3.0 v2 recipe.
- **`mhcflurry train processing-data`** now supports `--resume` and
  `--resume-matching-dir`: verify cached inputs/scores and expand only unmatched
  peptide-length pools, with unchanged affinity calipers and bounded rounds.
  Per-round score hashes, seeds, timings and final matched rows are durable.
  `--preparation-pipeline-depth 3` (default) overlaps bounded CPU preparation and
  atomic writes around caller-owned inference; use depth 1 as the serial control.
  Protein-reference inputs stay numeric through device-side gathering and
  scoring. Strings are materialized when exporting artifacts. Independent sample
  seeds preserve draws across pipeline depths.
- **`mhcflurry train benchmark-processing-preparation`** checks exact numeric
  sampling/export and matching parity, records timing repetitions and source/input
  hashes, and optionally checks real ensemble predictions with
  `--affinity-predictor`. See `docs/processing_preparation_acceleration.md`.
- **`mhcflurry train benchmark-processing-sampler`** compares the numeric-position
  and historical reservoir samplers on a reproducible synthetic CPU workload.
- **`mhcflurry eval processing-ensemble`** and
  **`mhcflurry eval presentation-affinity-ensemble`** score preserved
  prediction tables without rerunning inference.
- **`mhcflurry eval processing-ensemble-subsets`** caches individual network
  scores on strict matched risk sets and evaluates every fixed-size subset.
  It verifies reconstruction of the full cached ensemble, preserves subset
  membership and predictions, and reports composition sensitivity without
  selecting a subset on held-out metrics.
- **`mhcflurry eval release-experiment-figures`** regenerates the release
  training figures from archived experiment inputs.
- **`mhcflurry train plot-loss-curves`** renders per-architecture loss curves
  from a trained ensemble. The historical `plot_loss_curves.py` path remains a
  compatibility shim.
- **`benchmark_training_profile.py`** reports data-load, encoding, fit, and save
  timings for one architecture.

When an experiment becomes published release evidence, move or wrap it under
`downloads-generation/<download_name>/GENERATE.sh` so the artifact records its
inputs, command, version, and Git commit.

## Shared performance helpers

- **`set_cpu_threads.sh`** sets automatic BLAS/OpenMP thread budgets before
  Python imports native runtimes. Explicit caller settings remain authoritative,
  and serial execution applies an automatic runtime limit in-process.
- **`gpu_telemetry.sh`** records `nvidia-smi` samples for training stages. Set
  `MHCFLURRY_GPU_TELEMETRY=0` to disable it or
  `MHCFLURRY_GPU_TELEMETRY_SECONDS=N` to change the interval.
