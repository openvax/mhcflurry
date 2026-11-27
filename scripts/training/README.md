# Training scripts

Maintained stages and helpers for building pan-allele release models. For
complete end-to-end orchestration, start with:

```shell
scripts/release/retrain_evaluate_deploy.sh --help
```

Packaged training and evaluation CLI workflows live under `mhcflurry train
--help` and `mhcflurry eval --help`. Recorded historical experiment campaigns
(such as the `2.3.0` affinity factorial, processing regularization/activation,
flank ablation, and subset diagnostics) live separately under
[`../experiments/README.md`](../experiments/README.md). Transient one-off
experiments belong in gitignored `jobs/`.

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
  should use `scripts/release/retrain_evaluate_deploy.sh`.

Affinity, processing, and presentation stages write persistent GPU telemetry.
Worker packing defaults to workload-aware `auto`; pin a count only for a measured
machine-specific benchmark.

Processing source annotations use `RELEASE_RANDOM_SEED` (default 42) for ties
between maximum-expression sources. Canonical row order makes the selection
independent of input row order. `processing/hits_with_tpm.provenance.json`
records the policy, seed, input and output hashes, generator hash and versions.
Resuming the full workflow validates this record before reusing annotated hits.
Old runs without that evidence must remain historical artifacts; use a new run
directory for preparation with the reproducible policy.

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
- **`release_exact/make_train_data.processing.py`** and
  **`make_train_data.presentation.py`** prepare annotated hits, decoys, and
  model-family input tables.
- **`mhcflurry class1-reassign-mass-spec-training-data`** reruns the maintained
  mass-spec affinity remapping step.

## Evaluation

Model comparison and plotting are package commands, not training scripts. Use
`mhcflurry eval --help` (for example `compare-models`, `plot-comparison`,
`paper-figures`, `affinity-candidate-figures`, `processing-affinity-control`)
after a standalone stage run, or let `scripts/release/retrain_evaluate_deploy.sh`
invoke them automatically.

See the [evaluation guide](../../docs/evaluation.md) for comparison outputs,
paper figures, saved-prediction tables, and external predictors.

## Additional training & diagnostic scripts

- **`compose_processing_ensemble.py`** builds a fixed processing ensemble while
  hashing every source predictor.
- **`generate_processing_cleavage_boundaries.py`** and
  **`generate_processing_recipe.py`** define the cleavage-boundary and
  confirmed short-flank processing candidate hyperparameters used by the `2.3.0`
  release recipe (`final_230_candidate_v2_recipe.json`).
- **`run_processing_kernel_sweep.py`**, **`run_exact_public_processing.py`**,
  **`analyze_processing_recipe.py`**, and
  **`analyze_processing_confirmation.py`** support exact-public-data processing
  replays and confirmation analyses (see `notes/processing_kernel_sweep.md`,
  `notes/processing_hyperparameter_campaign.md`, and
  `notes/exact_public_data_experiment.md`).
- **`audit_training_data_identity.py`**, **`prepare_processing_evaluation.py`**,
  **`evaluate_saved_candidate.py`**, **`paired_sample_metrics.py`**,
  **`compare_presentation_external_predictors.py`**,
  **`compare_presentation_percentiles.py`**, and
  **`summarize_presentation_affinity_ensemble.py`** provide reusable training
  data audits, saved-candidate evaluation, and metric comparisons.
- **`benchmark_processing_preparation.py`** and **`training_benchmark.py`**
  profile processing preparation parity/speed and single-architecture training
  throughput.

Completed standalone campaign scripts (affinity factorial, processing
regularization/activation, flank ablation, fold-ensemble and subset analyses)
are quarantined under [`../experiments/release_2_3_0/`](../experiments/README.md).

## Shared performance helpers

- **`set_cpu_threads.sh`** sets automatic BLAS/OpenMP thread budgets before
  Python imports native runtimes. Explicit caller settings remain authoritative,
  and serial execution applies an automatic runtime limit in-process.
- **`gpu_telemetry.sh`** records `nvidia-smi` samples for training stages. Set
  `MHCFLURRY_GPU_TELEMETRY=0` to disable it or
  `MHCFLURRY_GPU_TELEMETRY_SECONDS=N` to change the interval.
