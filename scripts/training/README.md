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

For a direct runplz debugging session:

```shell
RUNPLZ_BREV_AUTO_CREATE=0 runplz brev \
    --outputs-dir /path/to/output \
    --instance existing-brev-instance \
    scripts/training/launch_pan_allele_training_remote.py
```

Set `MHCFLURRY_REMOTE_WORKFLOW=affinity-ablations` or
`MHCFLURRY_REMOTE_WORKFLOW=processing-ablations` to run the corresponding
committed parity panel through the same image and transport. The default is
`full`.

## Training data and hyperparameters

- **`mhcflurry class1-generate-training-hyperparameters`** generates the
  maintained affinity and processing grids. The base minibatch defaults to 1024
  and can be changed with `--minibatch-size`.
- **`release_exact/generate_hyperparameters*.py`** are compatibility shims for
  historical direct-script workflows.
- **`release_exact/make_train_data.processing.py`** and
  **`make_train_data.presentation.py`** prepare annotated hits, decoys, and
  model-family input tables.
- **`mhcflurry class1-reassign-mass-spec-training-data`** reruns the maintained
  mass-spec affinity remapping step. Its file under `release_exact/` is a
  compatibility shim.
- **`release_exact/additional_alleles.txt`** is an archived input from an older
  recipe and is not read by the current release stages.

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
