# Affinity dual-checkpoint workflow

## Problem

Affinity early stopping currently leaves only one recoverable state: either the
terminal state or the state with minimum internal validation loss. Comparing
those policies therefore requires independent training runs, and a completed
full ensemble cannot be re-evaluated under the other policy.

Tracked as [openvax/mhcflurry#386](https://github.com/openvax/mhcflurry/issues/386).

## Interface

- Add `--save-all-checkpoints` to
  `mhcflurry-class1-train-pan-allele-models`. It retains both the terminal and
  minimum-validation weights from each training trajectory while leaving
  `restore_best_weights` responsible for which state is the predictor's primary
  weight file.
- Store the two states as ordinary NPZ weight files under
  `checkpoints/terminal/` and `checkpoints/best/`, with relative paths recorded
  in `manifest.csv`.
- Add the reusable command:

      mhcflurry train materialize-affinity-checkpoint \
          --models-dir models.unselected.combined \
          --policy best \
          --out-models-dir models.unselected.best

  The command emits a normal loadable predictor directory whose primary weight
  files use the requested checkpoint policy. It records hashes and selection
  provenance and removes stale percentile-rank or optimization metadata.

## Compatibility and safety

- Existing training and model directories are unchanged unless the opt-in flag
  is supplied.
- A missing best checkpoint is an error rather than a silent terminal fallback.
- Weight arrays are copied when captured so the terminal and best states cannot
  alias one another.
- Incremental, multiprocessing, continuation, and cluster predictor
  serialization retain the checkpoint sidecars.

## Verification

- Prove terminal and best arrays differ and survive predictor save/load.
- Materialize both policies and prove their predictions match the corresponding
  stored weights.
- Cover CLI validation, missing checkpoints, and pre-existing output paths.
- Run focused tests, the full test suite, and lint before the change is called
  complete.
