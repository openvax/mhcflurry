# Maintainer workflows

## Contributing and verification

- Read the [contribution guide](https://github.com/openvax/mhcflurry/blob/master/CONTRIBUTING.md)
  before opening a pull request.
- Use {doc}`testing` for the fast local loop and full verification commands.
- Use {doc}`development` for internal API and compatibility-shim conventions.
- Use {doc}`orchestrator` when changing worker processes, hardware planning, or
  training data residency.

## Training and model releases

Maintained operational documentation lives next to the scripts it describes:

- [Training pipeline](https://github.com/openvax/mhcflurry/tree/master/scripts/training)
- [Release, synchronization, and deployment](https://github.com/openvax/mhcflurry/tree/master/scripts/release)
- [Generated model and data bundles](https://github.com/openvax/mhcflurry/tree/master/downloads-generation)
- {doc}`release_training_recipe` records the data and component settings used
  for the published 2.3.0 weights.
- {doc}`release_neural_hyperparameter_audit` explains framework equations and compatibility choices.
- {doc}`final_230_candidate_experiment` reruns the recorded 2.3.0 recipe from a
  clean checkout. A new fit need not reproduce the original training
  trajectory.

The public release entry point is:

```shell
mhcflurry train pan-allele-release --help
```

It coordinates training, evaluation, plots, remote artifact synchronization,
and optional deployment. Deployment is never enabled by default.

## Training internals

These pages specify current behavior of the training and evaluation pipeline.
Read {doc}`release_training_recipe` first for the component definitions the
others reference.

- {doc}`affinity_dual_checkpoint_workflow` covers `--save-all-checkpoints` and
  comparing terminal against minimum-validation-loss weights.
- {doc}`processing_preparation_acceleration` describes how `mhcflurry train
  processing-data` builds affinity-matched negatives with resumable artifacts.
- {doc}`probabilistic_processing_matching` specifies the seeded
  without-replacement negative-matching contract.
- {doc}`presentation_percentile_calibration` documents the shared
  `CompactPercentRankTransform` budget for presentation calibration.
- {doc}`saved_candidate_evaluation` evaluates a completed training run with
  `mhcflurry eval saved-candidate`, without retraining.

## Experiment notes

Controlled-experiment write-ups (cleavage-boundary models, kernel-width sweeps,
the processing hyperparameter campaign, and the historical public-data replay)
are kept as working notes in the repository's
[notes directory](https://github.com/openvax/mhcflurry/tree/master/notes), outside the
published docs.
