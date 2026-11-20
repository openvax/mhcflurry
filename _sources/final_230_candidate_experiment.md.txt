# Reproduce the 2.3.0 model training

Use a clean source checkout, explicit input snapshots and the recorded recipe.
The `final-2.3.0-candidate-v2` identifier names the frozen settings used for
2.3.0; see {doc}`release_training_recipe` for the component definitions.

The published weights retain their original training commit
`ac253f8c859dc687f7fbce9f4840fb55d5acf2c5` in their provenance. The commands below
run the recipe with the stable 2.3.0 code. That code corrects streaming
pretraining validation of inequality bounds, so a new fit need not reproduce
the original training trajectory. The released selected weights are unchanged
by that correction.

## Launch on Modal

The maintained launcher uses runplz and a persistent Modal volume. Substitute
a new run directory and workflow identifier for each independent experiment:

```shell
MHCFLURRY_RELEASE_RECIPE=final-2.3.0-candidate-v2 \
MHCFLURRY_RELEASE_DATA_VINTAGE=current \
RUNPLZ_OUTPUT_VOLUME=mhcflurry-model-training \
MHCFLURRY_RELEASE_OUT=/out/runs/mhcflurry-2.3.0 \
RUNPLZ_TIMEOUT_SECONDS=86400 \
MHCFLURRY_RELEASE_VERSION=2.3.0 \
MHCFLURRY_RELEASE_GIT_COMMIT="$(git rev-parse HEAD)" \
MHCFLURRY_RELEASE_WORKFLOW_ID=mhcflurry-2.3.0 \
RELEASE_RANDOM_SEED=42 \
RUN_RELEASE_EVAL=1 RUN_RELEASE_PLOTS=1 \
runplz modal scripts/training/launch_pan_allele_training_remote.py --detach
```

For detached launches, use `MHCFLURRY_RELEASE_OUT`; runplz owns `RUNPLZ_OUT`.
The receipt's `mhcflurry_release_out.txt` records the durable volume path.
One Modal invocation is limited to 24 hours. Resume only after verifying that
the previous worker has exited, using the same source, configuration and run
path. A failed evaluation is not a reason to retrain completed fits.

Processing preparation verifies cached input and score hashes before reusing
completed samples. Training resumes from saved manifests. Do not modify
hyperparameters, input tables or seeds within an existing run directory.

## Collect and verify

Create the local destination parent before `modal volume get`; the CLI
recreates the remote directory basename beneath it. Check the downloaded
manifests and checksums rather than treating a successful copy command as
proof of completeness.

Keep the source commit, architecture decision JSON, data hashes, training and
selection records, fitted presentation coefficients, calibration, and final
row-level evaluation scores together. The training software version remains
part of model provenance even when the weights are distributed in a later
or differently numbered download release.

A completed run can be archived with:

```shell
mhcflurry train snapshot-experiment \
  --source-dir /path/to/collected-run \
  --name mhcflurry-2.3.0 \
  --source-commit TRAINING_COMMIT
```

Use {doc}`evaluation` for independent comparisons. Keep the test rows fixed
across predictors, audit overlap against all compared MHCflurry training
sources, and record missing external-model training provenance explicitly.
