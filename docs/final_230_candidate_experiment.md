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

## Paired sample follow-up (2026-09-06)

A paired percentile bootstrap (10,000 replicates, seed 42) resamples the ten
evaluation samples together across models. These exploratory intervals describe
sample variability conditional on the fitted models; they do not account for
training-seed variability, repeated model selection, or multiple comparisons.

| Comparison | Macro AUPRC difference (95% interval) | Macro PPV@N difference (95% interval) |
| --- | --- | --- |
| Processing large 5x5 vs matched large legacy 5-aa | +0.00816 (-0.00146, +0.01822) | +0.00474 (-0.00157, +0.01121) |
| Processing large 5x5 vs selected public | -0.00596 (-0.01203, +0.00036) | -0.00123 (-0.00661, +0.00333) |
| Presentation public affinity + new processing vs public | +0.00549 (+0.00208, +0.00817) | +0.00563 (+0.00239, +0.00899) |
| Presentation new affinity + new processing vs public | +0.00454 (-0.00507, +0.01377) | +0.00633 (-0.00008, +0.01274) |

The 5x5 architecture remains the leading new processing branch candidate, but
its advantage over the matched legacy architecture is uncertain across samples.
Keep architecture/fold diversity and test the frozen hybrid at full scale.
Public affinity plus new processing is the most consistent presentation fallback
in the existing experiment (AUPRC improves in 9/10 samples, PPV@N in 8/10).
None of the no-flank presentation alternatives establishes a meaningful joint
improvement. Additional broad architecture searches are not justified by these
results; full-ensemble evaluation is the next decision experiment.

Reproduce from saved tables, without GPU inference or retraining:

```shell
mhcflurry eval paired-sample-metrics \
  --metrics output/processing-cleavage-boundary-radius-modal-c1928c1bb/processing-cleavage-boundary-radius/affinity_controlled/metrics.csv \
  --unit-columns group --condition-column score \
  --metric-columns pr_auc ppv_at_n --scope sample \
  --baseline large_relu__legacy_5aa \
  --out output/paired-sample-decision-230/processing-vs-legacy
```

Outputs include per-sample differences, bootstrap replicates, summary intervals,
input hashes, SVG and PNG plots. `mhcflurry eval collate-figures` combines these
panels with prior PDFs and annotated conclusions; its manifest hashes every input.

The boundary-padding review fix was also checked against the previous source
(`93734a09e`) using the trained four-network 5x5 ensemble. All 203,577 matched
held-out predictions are bit-for-bit identical (maximum difference 0). The
reusable check is `scripts/training/verify_processing_boundary_compatibility.py`;
its output includes both prediction columns and per-sample metrics.

### Active-run provenance exception

The in-flight `20260905T022747Z-final-230-candidate-c1928c1bb` run predates the
maintained launch command below. Its exact source archive hash is authoritative.
Its affinity checkpoints are terminal-only, and its completed 15-aa grid uses
Glorot/Keras Adam instead of the frozen Kaiming/native-Adam recipe. Preserve the
valid completed affinity/no-flank/short-flank fits and evaluate them; do not
describe this run as a clean implementation of the entire frozen recipe. Any
15-aa correction must use a separate model directory and explicit provenance.

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

When collecting a Modal directory, create its **local parent** first and pass
that parent to `modal volume get`; the CLI recreates the remote directory's
basename inside it. Verify the resulting manifest and weights before evaluation.
Passing a nonexistent destination can collapse multiple downloads into one file
with some Modal CLI versions, despite a success message.

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
