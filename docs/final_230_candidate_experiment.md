# Final 2.3.0 candidate training experiment

## Objective

Train one full release candidate after the frozen affinity and processing
screens, then evaluate affinity, processing, and end-to-end presentation
against the exact public release and external predictors.

## Frozen component recipes

### Affinity

- Full 35-architecture grid x 4 folds.
- Minibatch 1024, native PyTorch RMSprop, Glorot initialization, tanh where
  specified by the release grid, pre-activation LSUV, 50% dropout, patience 20,
  and terminal weights (`restore_best_weights: false`).
- Preserve architecture and fold diversity; do not add another optimizer,
  activation, dropout, or stopping-policy screen.
- Terminal weights remain the primary policy because the separate restore-best
  conditions were worse than the otherwise matched terminal control on macro
  AUPRC and PPV@N. Every training trajectory nevertheless retains both terminal
  and best checkpoints, so the exact same fits can be audited without retraining.
- The 0.25-dropout condition improved the global metrics but regressed the
  HLA-G locus; it therefore failed the predeclared locus safeguard. Keep 0.50
  dropout for this candidate rather than selecting on the global aggregate.

### Processing

- Minibatch 512 and four folds throughout.
- Train the full 128-architecture legacy grids for 5-aa short flanks and no
  flanks with Keras-compatible Adam and Glorot initialization.
- Train the full 15-aa compatibility grid once with native PyTorch Adam and
  Kaiming initialization; do not tune it further.
- Train one additional large-ReLU radius-5 cleavage-boundary architecture over
  four folds using the same processing rows as the legacy grids.
- The presentation with-flank predictor is a fixed eight-network ensemble:
  one selected legacy 5-aa network per fold plus one radius-5 boundary network
  per fold, equally weighted. This preserves the 50/50 family balance that was
  tested end to end. The no-flank predictor uses the ordinary selected
  no-flank ensemble.

## Presentation and evaluation

- Build the release holdout before training and exclude it from every component
  training table.
- Fit the public/new 2 x 2 presentation component factorial from shared cached
  features, then select one final stack by the predeclared macro AUPRC and
  PPV@N gate with micro, per-sample, per-length, allele, and locus safeguards.
- Preserve per-epoch histories, all candidate manifests and weights, row-level
  validation predictions, fitted presentation coefficients, telemetry, input
  hashes, source snapshot, and plotting tables.
- After internal selection, compare the chosen candidate with the public
  weights, NetMHCpan BA/EL, and MixMHCpred on the frozen affinity, processing,
  and presentation cohorts. Missing external-tool support is a tracked blocker,
  not a reason to change the internal holdout.

## Compute policy

- Use persistent Modal storage and resumable training directories.
- Separate GPU training/inference from CPU compression and figure assembly.
- Do not run another broad hyperparameter screen unless the full candidate
  reveals a large, reproducible failure that the existing screens did not
  cover.

## Reusable launch and collection commands

Launch the frozen recipe from a clean commit. The launcher rejects a commit
label that does not equal local `HEAD`, and rejects modified or untracked files
under the executable source directories.

```shell
MHCFLURRY_RELEASE_RECIPE=final-2.3.0-candidate \
RUNPLZ_OUTPUT_VOLUME=mhcflurry-230-final-weights \
RUNPLZ_OUT=/out/runs/final-2.3.0-candidate \
RUNPLZ_TIMEOUT_SECONDS=86400 \
MHCFLURRY_RELEASE_VERSION=2.3.0 \
MHCFLURRY_RELEASE_GIT_COMMIT="$(git rev-parse HEAD)" \
MHCFLURRY_RELEASE_WORKFLOW_ID=final-2.3.0-candidate \
RUN_RELEASE_EVAL=1 RUN_RELEASE_PLOTS=1 \
runplz modal scripts/training/launch_pan_allele_training_remote.py
```

Modal limits a single function invocation to 24 hours. If this complete run
does not finish within that window, rerun the identical command: the persistent
volume and `--continue-incomplete` manifests resume completed processing fits.

After collection, archive the immutable experiment record with the semantic
command (use the commit recorded by the remote run):

```shell
mhcflurry train snapshot-experiment \
  --source-dir output/final-2.3.0-candidate \
  --name final-2.3.0-candidate \
  --source-commit "$(git rev-parse HEAD)"
```

The snapshot must contain the exact architecture decision JSON, manifests,
per-epoch fit histories, model-selection tables, row-level benchmark and
validation predictions emitted by each experimental condition, public and
external score columns, fitted presentation coefficients, telemetry, logs,
source archive, hashes, and the generated PDF/SVG/PNG figures. Large model
weights may remain inventory-only when their immutable remote-volume location
and hashes are recorded; predictions and plotting tables must be copied.
