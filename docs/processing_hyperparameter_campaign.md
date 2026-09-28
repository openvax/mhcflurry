# Processing hyperparameter campaign

The maintained sweep compares processing-specific recipes; it does
not assume that affinity's winning optimizer transfers. All new fits use a
frozen affinity/length-matched table and preserve fold and reference identities.
Development results and the repeatedly inspected release benchmark are distinct.

## Maintained commands

Use `mhcflurry train processing-hyperparameter-sweep --design kernel-width`
for the six-width, two-family, four-fold comparison (48 networks). The existing
`processing-kernel-sweep` command remains an alias for the width workflow.

Use `mhcflurry train processing-hyperparameter-sweep --design training-recipe`
for the optimizer/initialization/batch factorial (64 networks):

- Keras-compatible Adam versus native PyTorch RMSprop.
- Glorot, orthogonal only, pre-LSUV and post-LSUV.
- Minibatches 512/1024, two architecture families and two paired folds.

The recipe screen holds width 11, ReLU, 512 filters, dropout 0.5, learning rate
0.001 and L1/L2 zero fixed. The width screen retains its original L1=1e-6.
Thus its width-11 control is not silently reused as a zero-L1 recipe fit.
The families are an external-flank CNN and a whole-peptide CNN plus 5x5
boundary branches, not a fixed mixture of networks.

Required arguments include `--out`, `--train-data`, `--public-root`,
`--release-holdout-dir` and `--source-commit`. Use `--evaluation none` for
development screening. `--prepare-only` writes the exact design without training.
`--folds-from` inherits the first two folds for the recipe screen; pass the same
frozen table as `--train-data` to preserve all numerical metadata exactly.

## Checkpoints and initialization

`save_all_checkpoints=True` retains independent best-loss and terminal processing
weights, plus best-AP when ranking monitoring is enabled.
`restore_best_weights=True` chooses the best state for `checkpoint_metric`
(`val_loss` by default; opt-in `val_macro_ap`). Checkpoints use NPZ sidecars; they do not enter manifest
JSON as weight arrays. Loading a referenced missing checkpoint fails. Refitting
without retention clears old references. Older predictors still load without
checkpoint sidecars, and missing historical terminal states are not synthesized.

`monitor_validation_ranking=True` records equal-sample AP and PPV@N after every
epoch. It requires explicit sample-disjoint training/stopping-validation masks
and sample IDs. Each validation sample must contain both labels. AP uses grouped
score ties; PPV breaks ties by original input-row order. Exact AP ties retain
the earliest epoch. Inner-best AP is saved as `best_ap`, separately from `best`
(loss) and `terminal`. `checkpoint_metric=val_macro_ap` enables this monitoring
and chooses that state when restoration is enabled. Patience remains based on
validation loss, so retained-state comparisons share one training trajectory.
Historical defaults and predictions remain unchanged. Fit metadata explicitly
records the restored policy, best loss/ranking epochs, and inner sample identity.

## Focused width / optimizer / checkpoint confirmation

Use `mhcflurry train processing-hyperparameter-sweep --design ranking-confirmation`
with `--folds-from FROZEN_TABLE --train-data FROZEN_TABLE --evaluation none`.
This creates 24 fits: legacy 5-aa CNN, widths 11/13/15, Adam/Keras versus
RMSprop/PyTorch and four paired folds. Glorot, batch 512, ReLU, 512 filters,
dropout 0.5, no normalization, fixed BLOSUM62 and zero L1/L2 are held constant.
All three checkpoint states have joinable outer-fold predictions and separate
per-sample and pooled-per-fold metric tables. Primary inference uses inner-best
AP; outer evaluation cannot select epochs. The source generator exposes
`processing_candidate_hyperparameters` for this explicit shortlist; it is not
a global default change or a final release recipe.

Set `PROCESSING_RANKING_CONFIRMATION=1` in the runplz Modal launcher, together
with the frozen table, absolute deadline and timeout at most four hours.
This mode skips both the width and full recipe factorials. It is incompatible
with `PROCESSING_RECIPE_AFTER_WIDTHS=1`. Training tables and independent
checkpoint files survive completed conditions; the budget is never extended
by retries. New experiments require compact selection and a separate full-presentation
evaluation before release, regardless of development improvements.

Analyze collected condition-level checkpoint tables with
`mhcflurry eval processing-confirmation-analysis --experiment RUN_DIR --out NEW_DIR`.
The command checks identical fold/sample/count identities across every state
and condition, writes paired sample-bootstrap intervals and figures, and only
exports `candidate_hyperparameters.yaml` after all six conditions are complete
and the declared development gate passes. The gate compares inner-best-AP
candidates against Adam/width-11 best-loss control: positive AP and PPV changes,
95% paired AP lower bound above zero, PPV lower bound above -0.002, and every
fold's pooled AUROC/AP/PPV changes at least -0.002. These are exploratory
intervals, not corrected for repeated screening or study dependence. A recipe
export is not a model release or a substitute for presentation validation.

`initialization_method` has explicit `none`, `orthogonal`, `lsuv_pre` and
`lsuv_post` values. None preserves legacy initialization. Non-default methods
apply to fresh fits; continued fitting does not reinitialize a trained network.
Calibration uses at most `initialization_batch_size` training rows (default
512), never stopping-validation rows. Their fit-input row indices are saved.

Eligible layers are boundary hidden layers, the main convolution, and hidden
pointwise convolutional heads, in forward order. Scalar heads and output/gating
weights are protected. Pre/post refer to before/after the configured activation,
before normalization or dropout. Variance excludes positions beyond actual
configured input extent; missing-context X rows within that extent remain.
Dropout is disabled during initialization. Non-convergence fails explicitly
and restores the pre-initialization parameters. Diagnostics include each layer's
variance and iteration count. See the [LSUV paper](https://arxiv.org/abs/1511.06422).

## Confirmation result

The complete panel (`processing-ranking-confirmation-20260909`; runplz run
`a8889cc8dfbb4f4390d4ceb153ad81e1`; source `9778c625`; 24 fits in 65 minutes
on one A100-40GB, about 1.1 of the 4 authorized GPU-hours) promoted
`legacy_5aa__rmsprop_pytorch__k13` with inner-best-AP weights.
`confirmed_processing_candidate_hyperparameters()` in
`scripts/training/generate_processing_recipe.py` names that recipe together
with the run, source and decision hashes; it equals the exported
`candidate_hyperparameters.yaml`. Macro metrics average four paired folds
within each of 37 held-out samples. Differences are against Adam/Keras width
11 best-loss weights (macro AUROC 0.7717, AUPRC 0.7555, PPV@N 0.7001), using
10000 paired sample-bootstrap draws with seed 42.

| Inner-best-AP condition | AUROC | AUPRC | PPV@N | AUPRC difference | PPV@N difference | Gate |
|---|---:|---:|---:|---:|---:|---|
| Adam/Keras, width 11 | 0.7810 | 0.7641 | 0.7128 | +0.0086 [0.0043, 0.0126] | +0.0127 [0.0085, 0.0169] | pass |
| Adam/Keras, width 13 | 0.7857 | 0.7671 | 0.7162 | +0.0116 [0.0048, 0.0182] | +0.0162 [0.0105, 0.0219] | pass |
| Adam/Keras, width 15 | 0.7838 | 0.7666 | 0.7153 | +0.0111 [0.0043, 0.0176] | +0.0153 [0.0085, 0.0216] | pass |
| RMSprop/PyTorch, width 11 | 0.7798 | 0.7629 | 0.7090 | +0.0074 [0.0021, 0.0128] | +0.0089 [0.0035, 0.0144] | fail: pooled fold AUROC -0.0039, AUPRC -0.0074 |
| RMSprop/PyTorch, width 13 | 0.7870 | 0.7701 | 0.7155 | +0.0146 [0.0085, 0.0207] | +0.0154 [0.0101, 0.0206] | pass, promoted |
| RMSprop/PyTorch, width 15 | 0.7869 | 0.7688 | 0.7146 | +0.0133 [0.0076, 0.0191] | +0.0145 [0.0088, 0.0203] | pass |

The retained-state choice is the largest single effect. In all six conditions
inner-best-AP and terminal weights outscore best-loss weights on AUPRC and
PPV@N; best-loss epochs ranged from 4 to 34 and inner-best-AP epochs from 14
to 44. Within the control, switching states alone gains +0.0086 AUPRC.
Terminal weights scored similarly but were never eligible, by design. Widths
13 and 15 lead width 11 with either optimizer, but the promoted recipe's
AUPRC margin over the other passing conditions (0.001 to 0.006) lies inside
every paired interval; it is the recorded point-estimate tie-break, not a
resolved difference.

These are single-fit, four-fold development scores on one training trajectory
per fold, without multiple-screening or study-cluster correction. They do not
change public downloads, default hyperparameters or predictions, and they say
nothing about ensembles or presentation. Compact selection on a common
held-out panel and the separate presentation gate remain required before any
release use; family and width diversity is preserved for that step.

## Recovery and outputs

`--resume-from PRIOR_WIDTH_RUN` copies complete conditions and frozen folds to
a separate run after checking input/design identity. Incomplete conditions stay
in the old run and are reinitialized in the new one. Every copied file is hashed.
Existing recovery artifacts must match those hashes on resume. A lossless CSV
parser preserves saved reference scores; strict fold checks are not relaxed.

Every new condition saves epoch losses, optimizer steps, epoch timing, stop
reason, initialization diagnostics, best/terminal weights, and context-joinable
per-member predictions on each model's held-out fold. Root summaries give each
sample equal weight after averaging repeated folds within sample. These
one-decoy validation AP values are not comparable to ten-decoy processing or
full-presentation AP levels. Four/eight-member selection still requires a
common held-out panel unseen by every member; this screen does not perform it.

The runplz Modal launcher accepts `PROCESSING_KERNEL_TRAIN_DATA` and
`PROCESSING_KERNEL_PRIOR_SWEEP` to reuse completed preparation and fits.
`PROCESSING_KERNEL_EVALUATION=none` is the default.
`PROCESSING_RECIPE_AFTER_WIDTHS=1` queues the factorial behind width completion
on the same single GPU. It requires an absolute
`MHCFLURRY_EXPERIMENT_DEADLINE_EPOCH`; command descendants are terminated at
that deadline, without marking the stage complete. Set an absolute deadline
at most 15.5 hours after submission for the authorized 16-GPU-hour allocation,
leaving margin for teardown. The per-function timeout cannot exceed 15.5 hours.
No retries or subsequent allocation may exceed the remaining campaign budget.

The launcher writes width figures first, then a combined
`campaign-all-figures.pdf` after the recipe screen. Raw inputs and prediction
caches remain independently collectible even if a later stage fails. A generated
PDF is not a substitute for inspected figures, paired uncertainty, four-fold
confirmation, compact selection and an end-to-end presentation gate.

## Analyze an incomplete or completed recipe screen

Run the maintained read-only analysis against a collected snapshot from one run:

```bash
mhcflurry eval processing-recipe-analysis \
  --experiment collected/recipe_sweep/experiment.json \
  --metrics collected/recipe_sweep/validation_per_sample.csv \
  --out experiments/recipe-analysis-snapshot
```

The output directory must be new or empty. The command preserves unreported
conditions as pending and rejects incomplete folds, mismatched sample cohorts
or counts, nonfinite metrics, and uncontrolled hyperparameter differences.
It averages folds within each sample, then bootstraps samples jointly for each
one-factor contrast. It does not average across unmatched factorial cells to
claim an overall optimizer or initialization effect. Optimizer comparisons
include their implementation (for example, Adam/Keras versus RMSprop/PyTorch).

Outputs include a PDF and page PNGs, condition means, design completion status,
paired contrasts, per-sample differences, and a provenance manifest with input
and analysis-source hashes. Gray plot cells mean pending, not poor performance.
Pareto flags describe AUPRC/PPV point estimates only; they are not significance
tests, ensemble selections, or release acceptance. Intervals are exploratory,
conditional on trained fits, and uncorrected for screening or study clustering.

The shared `mhcflurry train plot-loss-curves` command recognizes processing CNN
architecture identities and convolutional L1/L2 regularization. Its curves and
`final_loss`/`final_val_loss` columns describe final training epochs, even when
inference restores an earlier checkpoint. Separate best-epoch, best-validation-
loss, and checkpoint-policy columns record that distinction. Historical fits
without explicit restoration metadata report an unknown checkpoint policy.

## Cached within-fold ensemble diagnostics

`mhcflurry eval processing-fold-ensembles --ensembles ensembles.json --out NEW_DIR`
accepts a JSON object mapping ensemble names to ordered checkpoint prediction
cache paths (relative to that JSON). It verifies each cache checksum/sidecar,
training-table fingerprint, sample/row/context/label identity, and one distinct
member per condition per fold. Every mean uses only the models from the row's
held-out fold. Best and terminal are reported separately by default; use
`--checkpoint-policy best` to request only the primary policy.

Outputs preserve averaged predictions, fold-to-member identities, per-sample
and pooled-per-fold metrics, sample means, provenance and source. Pass the
sample means to `mhcflurry eval paired-sample-metrics` for paired plots. This
is a no-training, no-inference development diagnostic of multiple fold-specific
ensembles. It does not measure one fixed ensemble spanning different folds,
select members, or establish release acceptance. A final fixed ensemble still
requires a common panel unseen by every one of its members.
