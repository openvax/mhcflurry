# Affinity checkpoint comparison

Pass `--save-all-checkpoints` to `mhcflurry class1-train-pan-allele-models`
to retain both terminal and minimum-validation-loss weights from each training
trajectory. `restore_best_weights` still selects the primary prediction state.
Checkpoint NPZ files are stored under `checkpoints/terminal/` and
`checkpoints/best/`, with relative paths in the training manifest.

Materialize an alternative state as a separate predictor:

```shell
mhcflurry train materialize-affinity-checkpoint \
    --models-dir models.unselected.combined \
    --policy best \
    --out-models-dir models.unselected.best
```

The command records source hashes and selection provenance and removes stale
calibration and optimization metadata. A missing requested checkpoint is an
error. Repeat ensemble selection and calibration before treating the new
predictor as a replacement for the original; compare both on identical rows.

Training save/load, multiprocessing and continuation preserve checkpoint
sidecars. Release inference exports omit them and load only the selected primary
state. Keep the original training directory to revisit checkpoint selection.
The 2.3.0 affinity weights use terminal checkpoints; retaining best-loss states
does not imply that they improved the released ensemble.
