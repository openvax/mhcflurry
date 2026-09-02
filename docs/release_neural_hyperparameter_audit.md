# Neural training parity audit: 2.1.x to 2.3.0

This audit compares the latest 2.1.x tag (`v2.1.5`) and its published release
generators with the 2.3.0 release branch. The 2.1.x and 2.2.x public downloads
use the same release grids. It distinguishes three things that are easy to
conflate:

1. the values written into the public release hyperparameter YAML;
2. framework-level training equations behind names such as `RMSprop`, `Adam`,
   `Glorot`, and `LSUV`; and
3. execution controls that should not alter the scientific recipe.

The compatibility rule is: restore 2.1.x behavior unless a held-out comparison
supports a change. A rejected full candidate trained with affinity minibatch
1024 outperformed an older public ensemble, but that comparison changed many
other settings and could not identify batch size as the cause. In a direct
latest-code comparison on the frozen release holdout, batch 1024 was worse than
128. Batch 128 therefore remains the release baseline. The ablations below test
optimizer/minibatch interactions explicitly rather than coupling them.

## Release-grid audit

### Affinity

| Category | Published 2.1.x | 2.3.0 decision |
|---|---|---|
| Architectures | 20 feedforward networks from `[512,256]`, `[512,512]`, `[1024,512]`, `[1024,1024]`; 15 skip-connected networks from `[256,512]`, `[256,256,512]`, `[256,512,512]`; each crossed with L1 `1e-7`, `1e-8`, `1e-9`, `1e-10`, `0` | Exact match: 35 architectures × 4 folds |
| Input | BLOSUM62, maximum length 15, `left_pad_centered_right_pad`; concatenated peptide and allele representations | Exact mathematical encoding; fixed BLOSUM62 expansion now happens on-device |
| Hidden/output activations | `tanh` / `sigmoid` | Exact match |
| Dropout | `dropout_probability=0.5` means keep probability 0.5, hence dropout rate 0.5 | Exact match; legacy keep-probability naming retained |
| Batch normalization / local convolution | Off / none | Exact match |
| Base initializer | Glorot uniform, zero bias | Exact match for release Dense layers |
| Data-dependent initializer | LSUV, measuring the post-activation Keras Dense output | Restored; `data_dependent_initialization_target=post_activation` is explicit |
| Loss | Inequality-aware mean squared error | Exact objective and reduction |
| Optimizer | Keras RMSprop: pretrain LR `0.001`, fine-tune LR `0.0001`, rho `0.9`, momentum `0`, centered false, epsilon `1e-7` inside `sqrt(v + epsilon)` | Restored with public `KerasRMSprop`; `optimizer_implementation=keras` is explicit |
| Minibatch | 128 | Restored. Under Keras RMSprop, 1024 reduced frozen-holdout macro AUROC 0.40%, AUPRC 5.20%, and PPV@N 4.31% relative to the paired batch-128 control. |
| Validation | Tail 10%; every epoch | Restored, including Keras split rounding |
| Early stopping | patience 20, `min_delta=0`, ceiling 5000 epochs | Restored |
| Pretraining | 64 peptides/chunk, 256 steps/epoch, min 5/max 50 epochs, patience 2, `min_delta=1e-4`, maximum accepted validation loss 0.10 | Exact match |
| Random negatives | Per-allele nonbinder equalization, rate 1, constant 1, 30,000–50,000 nM, binder threshold 500 nM, fresh pool each epoch | Exact distribution and pool lifetime; seeded identities differ |
| Randomness | Historically entropy-derived | One explicit release master seed (42), recorded and threaded through folds, fits, random negatives, and calibration |
| Selection | 4 folds, minimum 2 / maximum 8 models per fold | Exact match |

### Processing

| Category | Published 2.1.x | 2.3.0 decision |
|---|---|---|
| Architectures | `tanh`/`relu` × 256/512 filters × kernels 11/13/15/17 × L1 0/`1e-6` × dense `[8]`/`[16]` × dropout 0.3/0.5 | Exact match: 128 architectures × 4 folds per flank variant |
| Input | BLOSUM62; peptide maximum 15; 15/15, 0/0, and 5/5 flank variants | Exact encoding and variants; expansion now happens on-device |
| Pooling/topology | Spatial dropout; N/C cleavage features; internal peptide maxima; flank averages; sigmoid output | Exact formula and masks |
| Initializer | Every Conv1D/Dense kernel Glorot uniform, every bias zero; final output kernel ones | Retained for 0/0 and 5/5 flanks. The independently trained 15/15-flank variant uses Kaiming fan-in with uniform random biases after the paired held-out panel favored it in both representative architecture families. The generated YAML records `init` explicitly. |
| Loss | Binary cross-entropy | Same objective; PyTorch and Keras use algebraically equivalent stable/clipped kernels with possible last-bit differences near saturation |
| Optimizer | Keras Adam: LR `0.001`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-7` using Keras' non-adaptive epsilon placement | Retained for 0/0 and 5/5 flanks. The 15/15-flank variant uses native PyTorch Adam after the same paired panel. `optimizer_implementation` is explicit in every generated variant YAML. |
| Minibatch | 512 in the published generator | Exact match |
| Validation / early stop | Tail 10%, patience 20, `min_delta=0`, ceiling 500 epochs | Exact match, including Keras split rounding |
| Holdout / selection | 10 samples per fold; minimum 1 / maximum 2 models per fold | Restored |
| Decoys | Proteome-derived; PPV multiplier 100; retain 2 candidates per hit | Exact recipe |
| Randomness | Historically entropy-derived | One explicit release master seed (42), with stable per-sample decoy streams and deterministic output order |

### Presentation combiner

Presentation is logistic regression rather than a neural network, but it is
part of the end-to-end result. The compatibility recipe is two proteome decoys
per hit, sample fraction 0.1, the three study exclusions, `short_flanks` (5 aa
on each side), and L-BFGS with 100 iterations. L-BFGS itself is supported; an
upstream SciPy warning concerns deprecated display options, not the optimizer.
Presentation decoy selection and the 0.1 subsample now derive from the same
explicit release seed (42), rather than ambient process state.

## Framework-semantic discrepancies found

| Discrepancy | Release relevance | Resolution / justification |
|---|---|---|
| Processing layers silently used PyTorch Kaiming/fan-in initialization and nonzero random biases | Direct: every processing candidate | Made explicit and equation-tested. Glorot/zero bias remains the compatibility baseline and the 0/0- and 5/5-flank recipe; paired held-out evidence selects Kaiming/fan-in only for the independently trained 15/15-flank variant. |
| Affinity LSUV observed raw Linear output instead of the activated Dense output | Direct: every affinity candidate uses LSUV | Fixed. Post-activation is historical parity; pre-activation remains selectable for an ablation. |
| Native PyTorch RMSprop places epsilon outside the square root; Keras places it inside | Direct: every affinity optimizer step | Fixed with a public, tested Keras equation and an explicit implementation switch. PyTorch documents this framework difference. |
| Native PyTorch and Keras Adam place epsilon differently relative to bias correction | Direct: every processing optimizer step | Both equations are public and tested, and the selected implementation is serialized. Keras-compatible Adam remains the 0/0- and 5/5-flank recipe; native Adam is selected for 15/15 flanks by the paired panel. |
| Both PyTorch trainers rounded validation rows as `floor(N * split)` instead of using Keras' split boundary `floor(N * (1 - split))` | Direct but usually one row per network | Fixed centrally and regression-tested. This is deterministic parity, not an optimization question. |
| PyTorch BatchNorm updates running variance with an unbiased estimate; Keras uses population variance | Inactive in the release affinity grid; processing has no BN | Fixed in `KerasBatchNorm1d` and tested against the Keras equation, so non-release configurations are not left divergent. |
| Generic PyTorch Xavier fan calculation was wrong for the transposed 3-D `LocallyConnected1D` storage | Inactive: release affinity grid has no local layers | Fixed using Keras' `(output_length, flattened_input, filters)` fans. |
| Native SGD fallback used LR 0.001 and unknown optimizer names silently became Adam | Inactive: release uses RMSprop/Adam | Fixed: Keras SGD default LR is 0.01 and unknown names now fail. |
| Keras Glorot/He *normal* initializers use variance-corrected truncated normals; PyTorch's normal initializers are untruncated | Inactive: release uses Glorot uniform | Follow-up only if those non-release initializer values are to remain supported for new training. Loaded historical weights are unaffected. |
| TensorFlow and PyTorch dropout RNGs, shuffles, reduction order, and GPU kernels differ | Direct stochastic trajectory, but not a hyperparameter drift | Irreducible framework difference. Compare distributions and held-out metrics, not byte-identical weights. |
| Fixed master/per-fit seeds replace entropy-derived seeds | Direct identities/trajectory | Intentional reproducibility improvement. It does not change the sampled distributions. |
| Device-side encoding, compact cartesian batches, validation batching, lazy proteome sampling, and prediction chunking | Execution only | Algebra/prediction parity is covered by tests. Release training fails if autosizing would shrink a configured minibatch. |
| `torch.compile` and reduced float32 matmul precision | Could alter trajectory | Off for release; eager + `highest` precision is pinned. |

[PyTorch's RMSprop documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)
explicitly notes that it takes the square root before adding epsilon and that
TensorFlow interchanges those operations. The
[Keras RMSprop documentation](https://keras.io/api/optimizers/rmsprop/) records
its defaults and epsilon convention. For Adam, both implementations are valid
variants of the algorithm in the
[Kingma and Ba paper](https://arxiv.org/abs/1412.6980); compatibility, not a
claim of universal optimizer superiority, determines the affinity and 0/0- and
5/5-flank compatibility settings. The 15/15-flank processing departure is based
on the controlled held-out comparison below.

The public predictor class defaults are independent of the release YAML.
Current release generators explicitly use affinity minibatch 128 and processing
minibatch 512, and the generated YAML records the actual values.

## Controlled experiment matrix

Experiments use the same curated data, frozen release holdout, folds, seeds,
precision, and eager execution. Only the named field changes. Report paired
results by architecture/fold/seed plus aggregate medians and confidence
intervals; do not compare two dataframes after merging ambiguous metric
columns.

### Affinity initialization/optimizer/minibatch panel

Use one representative feedforward architecture (`[512,512]`) and one
skip-connected architecture (`[256,512,512]`), both at L1 `1e-8`, across all
four folds. This is 8 networks per condition rather than the full 140.

| Condition | Batch | LSUV | Optimizer equations | Purpose |
|---|---:|---|---|---|
| A: published parity | 128 | post-activation | Keras RMSprop | Historical control |
| B: proposed release | 1024 | post-activation | Keras RMSprop | Confirm the only retained change after optimizer restoration |
| C: rejected LSUV port | 1024 | pre-activation | Keras RMSprop | Isolate LSUV boundary |
| D: no LSUV | 1024 | disabled | Keras RMSprop | Test whether LSUV itself helps |
| E: native optimizer | 1024 | post-activation | PyTorch RMSprop | Isolate optimizer equation |
| F: native optimizer at parity batch | 128 | post-activation | PyTorch RMSprop | Isolate optimizer equation without the batch-size confounder |

Primary metrics are frozen-holdout affinity AUROC, AUPRC, and PPV@N. Secondary
diagnostics are pretrain failure rate, inequality-aware validation loss,
early-stop epoch, gradient/weight norms, and wall time. Each alternative is
scored directly against the published-parity condition so the saved tables use
unambiguous A/B columns. Retain a non-baseline setting only for a consistent
paired improvement without a locus/allele subgroup regression.

#### Affinity interim results and expanded factorial

The first controlled panel established that batch size, RMSprop equations, and
LSUV cannot be ranked independently. Changes below are relative to the same
batch-128, post-activation LSUV, Keras-compatible RMSprop control. Each cell is
macro `AUROC / AUPRC / PPV@N` percent change followed by the corresponding
micro changes.

| Condition | Macro change | Micro change | Interpretation |
|---|---|---|---|
| Keras RMSprop, batch 1024, post-activation LSUV | -0.40% / -5.20% / -4.31% | -0.34% / -2.58% / -5.34% | Reject as a global batch-size change |
| Keras RMSprop, batch 1024, pre-activation LSUV | -0.38% / -6.57% / -5.53% | -0.34% / -5.79% / -7.57% | Pre-activation port does not restore the loss |
| Keras RMSprop, batch 1024, no LSUV | -0.09% / -1.17% / -0.93% | -0.00% / +3.89% / +0.19% | Mixed macro/micro result; batch and initialization remain confounded |
| Native RMSprop, batch 1024, post-activation LSUV | +0.09% / -0.22% / +0.04% | +0.06% / -3.08% / -1.09% | Nearly flat macro result but a micro-AUPRC regression |
| Native RMSprop, batch 128, post-activation LSUV | -0.29% / -4.79% / -3.33% | -0.36% / -9.53% / -4.40% | Native RMSprop interacts strongly with batch size |

These paired comparisons explain why an earlier batch-1024 candidate could
beat an older public model without proving that 1024 was the cause. Under the
historical Keras recipe, the direct latest-code comparison favors batch 128.
Under native RMSprop, 1024 is much less damaging than 128 on the same panel.
No tested alternative improves both macro and micro AUPRC/PPV, so the release
control remains batch 128, post-activation LSUV, and Keras RMSprop.

The follow-up runner expands this to a true 40-condition factorial:
minibatch 128/256/512/1024; Keras/native RMSprop; post- and pre-activation LSUV;
and no-LSUV Glorot, He, and orthogonal initialization. LSUV is not an
alternative name for Glorot or He: for eligible hidden layers it first replaces
the kernel with an orthogonal draw and then rescales it from observed
activations, so the nominal base initializer is overwritten there. The
representative phase trains two architecture families across four folds per
condition; only promoted conditions should be expanded to all 35 architectures.

### Processing initializer/optimizer panel

Use two deliberately different architectures across four folds: a smaller
tanh/256/kernel-11/dense-8/dropout-0.3/L1-0 network and a larger
relu/512/kernel-17/dense-16/dropout-0.5/L1-`1e-6` network. Cross:

1. Glorot + Keras Adam (published parity),
2. Kaiming/fan-in + Keras Adam,
3. Glorot + PyTorch Adam, and
4. Kaiming/fan-in + PyTorch Adam (the framework port's coupled behavior).

Run the 15/15 and no-flank variants first; add 5/5 only if the direction differs
by variant. Primary metrics on the frozen 10 samples are AUPRC and PPV, with
AUROC, calibration, early-stop epoch, stability, and peak memory secondary.
The reproducible runner is
`scripts/training/run_release_processing_ablations.sh`; it holds the public
affinity predictor fixed while generating one shared processing-training table,
then compares each alternative directly with Glorot + Keras Adam on identical
folds and frozen rows.

#### Processing panel results

The panel used one shared 399,392-row training table from 100 samples, seed 42,
four folds, 10 held-out samples per fold, and the public affinity predictor for
decoy selection. Training used BLOSUM62, peptide maximum length 15, minibatch
512, learning rate 0.001, validation split 0.1, maximum 500 epochs, early-stop
patience 20, and flank averages. The two representative networks were:

| Network | Activation | Conv filters | Kernel | Dense | Dropout | Conv L1/L2 |
|---|---|---:|---:|---:|---:|---:|
| Small | tanh | 256 | 11 | 8 | 0.3 | 0 / 0 |
| Large | ReLU | 512 | 17 | 16 | 0.5 | 1e-6 / 0 |

The initialization and optimizer settings were crossed; no other field changed.
Glorot draws weights uniformly with bound
`sqrt(6 / (fan_in + fan_out))` and sets biases to zero. The rejected PyTorch
port behavior draws both weights and biases uniformly with bound
`1 / sqrt(fan_in)`. Both Adam implementations used learning rate 0.001,
`beta_1=0.9`, `beta_2=0.999`, epsilon `1e-7`, no weight decay, and no AMSGrad.
Keras-compatible Adam applies the bias-corrected step to
`m / (sqrt(v) + epsilon)`; native PyTorch Adam applies epsilon after expressing
the second moment in bias-corrected form. Thus the two update rules are not
identical despite equal numeric settings.

The table combines the crossed network/optimizer fields with primary held-out
results. Changes are relative to Glorot + Keras Adam. Each result cell is
macro `AUPRC / PPV@N` percent change followed by per-sample
`AUPRC wins-losses-ties; PPV@N wins-losses-ties` on the same 10 samples.

| Initializer and bias | Adam equations | 15-aa flanks | No flanks | 5-aa flanks | Decision |
|---|---|---|---|---|---|
| Glorot uniform; zero bias | Keras-compatible | reference 0.1699 / 0.2600 | reference 0.2063 / 0.2930 | reference 0.1833 / 0.2718 | Release baseline |
| Kaiming fan-in; uniform random bias | Keras-compatible | -0.88% / -0.08%; 3-7-0, 4-6-0 | +2.03% / +1.31%; 8-2-0, 7-2-1 | -2.16% / -3.18%; 0-10-0, 0-9-1 | Reject |
| Glorot uniform; zero bias | Native PyTorch | +0.59% / +0.83%; 8-2-0, 6-4-0 | -0.80% / -0.55%; 4-6-0, 4-6-0 | -0.49% / -0.17%; 3-7-0, 5-5-0 | Reject |
| Kaiming fan-in; uniform random bias | Native PyTorch | +1.99% / +0.65%; 8-2-0, 6-4-0 | +0.43% / +1.30%; 6-4-0, 6-4-0 | -2.15% / -1.43%; 3-7-0, 3-7-0 | 15-aa variant only; reject for 0/5 aa |

Macro and micro AUROC stayed nearly flat. On the production-relevant 5-aa
condition, every non-baseline setting reduced both macro and micro AUPRC and
PPV@N. The architecture-stratified results show why a single pooled rule is
unsafe:

| Flanks | Representative-architecture result | Decision |
|---|---|---|
| 15 aa | Kaiming + native Adam improved small/tanh AUPRC 4.10% and large/ReLU 1.97%. Kaiming + Keras Adam improved large/ReLU more (3.22%) but reduced small/tanh 3.34%. | Keep Kaiming + native Adam only as a low-priority diagnostic; do not give 15-aa the same search budget as 5-aa or no-flank models. |
| No flanks | Every alternative improved small/tanh (best: Kaiming + Keras Adam, +9.70%) but reduced large/ReLU (worst: Glorot + native Adam, -6.08%). | Search Kaiming + Keras for small/tanh while retaining historical Glorot + Keras for large/ReLU; select the mixture using fold-internal data. |
| 5 aa | Every alternative reduced small/tanh. Large/ReLU improved under every alternative (best: Glorot + native Adam, +2.19%). | Primary search: retain historical Glorot + Keras for small/tanh and compare historical, Glorot + native, and Kaiming + native for large/ReLU using fold-internal selection. |

These rows compare uniform recipes. They do not imply that one
initializer/optimizer pair should be fixed across every architecture. An
equal-weight exploratory mixture kept the historical Glorot/Keras small model
and substituted the Kaiming/native large model in the 5-aa ensemble. Against
an equal-weight historical control it changed macro AUPRC by +0.16%, PPV@N by
+0.22%, and AUROC by +0.02%; per-sample signs were 6-4, 5-5, and 6-4. The
effect is positive but too small and inconsistent to bypass normal model
selection. The already-selected seven-model historical ensemble also remains
better than either equal-weight eight-model mixture. Initializer and optimizer
therefore remain architecture-level search axes; mixed-recipe candidates must
be selected using fold-internal data rather than assembled from release-
benchmark means.

#### Direct 5-aa versus 15-aa comparison

The best uniform-recipe 5-aa and 15-aa ensembles were also compared directly,
not inferred from two separately rounded tables. Row identity was verified
over all 2,054,263 benchmark rows before taking the 5-aa score from the
Glorot/Keras prediction and the 15-aa score from the Kaiming/native prediction.
The 5-aa selector retained the large/ReLU network in all four folds and the
small/tanh network in three folds (seven models); the 15-aa selector retained
both architectures in all four folds (eight models). Both used batch 512 and
learning rate 0.001.

| Metric | 5-aa Glorot/Keras | 15-aa Kaiming/native | Change for 5 aa | Per-sample wins |
|---|---:|---:|---:|---:|
| Macro AUPRC | 0.18333 | 0.17330 | +5.79% | 10-0 |
| Macro PPV@N | 0.27177 | 0.26165 | +3.87% | 10-0 |
| Macro AUROC | 0.86761 | 0.86734 | +0.03% | 6-4 |
| Micro AUPRC | 0.18368 | 0.17355 | +5.84% | n/a |
| Micro PPV@N | 0.27260 | 0.26552 | +2.67% | n/a |

This is aggregate and sample-level dominance on the primary metrics. Apparent
15-aa advantages in the pooled peptide-length breakdown were checked by
bootstrapping the 10 held-out samples rather than treating millions of rows as
independent observations:

| Length | Positive rows (% of positives) | Macro AUPRC, 5-aa minus 15-aa (95% CI) | Macro PPV@N, 5-aa minus 15-aa (95% CI) | Interpretation |
|---:|---:|---:|---:|---|
| 8 | 852 (4.60%) | -0.00068 (-0.00145, +0.00009) | -0.00917 (-0.01877, -0.00046) | AUPRC is inconclusive; this sparse stratum is the only credible 15-aa PPV@N advantage. |
| 9 | 12,753 (68.91%) | +0.00720 (+0.00302, +0.01107) | +0.00955 (+0.00518, +0.01446) | Clear 5-aa advantage on both primary metrics. |
| 10 | 2,735 (14.78%) | +0.00505 (+0.00125, +0.00945) | -0.00081 (-0.00934, +0.01024) | Clear 5-aa AUPRC advantage; PPV@N is inconclusive. |
| 11 | 2,167 (11.71%) | +0.00112 (-0.00192, +0.00416) | +0.00302 (-0.00464, +0.00994) | Neither primary difference is distinguishable from sample-level noise. |

Thus most of the apparent length-specific exceptions are either sparse or
inconclusive. The release and search priority is explicitly: optimize 5-aa
performance first, no-flank performance second, and 15-aa performance third.
The 15-aa recipe is retained only as a targeted diagnostic for 8-mer PPV@N or
for confirmation with additional independent positives, not as a co-equal
search arm. This prioritization must still use fold-internal selection; the
release benchmark cannot be used to choose individual ensemble members.

Batch 512 was held fixed throughout these processing comparisons because it
is the historical processing recipe. This experiment did not establish that
512 is better than 128 or 1024 for processing; the separate affinity batch
sweep must not be used to make that claim. A focused processing release gate
therefore compares minibatches 128, 256, 512, and 1024 while holding training
rows, folds, seeds, learning rate, and architecture candidates fixed. The
primary 5-aa panel retains historical small/tanh and compares historical,
Glorot/native-Adam, and Kaiming/native-Adam large/ReLU candidates. The
secondary no-flank panel compares historical and Kaiming/Keras-Adam small/tanh
while retaining historical large/ReLU. Each batch is selected using
fold-internal data before the frozen prediction rows are scored. A broad 15-aa
batch sweep is excluded; only the targeted 8-mer diagnostic remains. The
release recipe cannot label a processing batch size as preferred until this
gate completes.

The Glorot + Keras Adam control itself still underperformed the public model:
macro AUPRC changed by -17.20%, -9.51%, and -16.74% for 15-aa, no-flank, and
5-aa models respectively; PPV@N changed by -10.06%, -7.66%, and -9.27%, while
AUROC was nearly flat. This panel held the public affinity predictor fixed for
decoy selection, so its processing regression cannot be attributed to a newer
affinity model producing harder negatives. Restoring framework parity is
necessary, but the eventual full retrain still has to demonstrate that its
weights are competitive with the public release.

#### Activation-by-architecture follow-up

The representative panel coupled small geometry with tanh and large geometry
with ReLU. A follow-up swapped those activations while holding flank mode,
initializer, optimizer, data, folds, and all other geometry fixed. Across the
four initializer/optimizer conditions and two geometries:

| Flanks | Mean AUPRC change after activation swap | Wins | Interpretation |
|---|---:|---:|---|
| No flanks | +5.15% | 7 / 8 | The reversed activation assignment often helps when only peptide positions are present. |
| 5 aa | -4.34% | 0 / 8 | The original activation assignment is consistently better. |
| 15 aa | -7.08% | 0 / 8 | The original activation assignment is consistently better. |

The same swap helps no-flank models and hurts both flanked variants in both
directions (small tanh to ReLU and large ReLU to tanh). Parameter counts and
initialized parameter norms are unchanged between 5-aa and 15-aa models
because convolutional weights are shared across sequence positions; the longer
input changes activation statistics and pooling opportunities, not the number
of learned weights. The general lesson is that activation, geometry, flank
context, initialization, and optimizer interact. The full release grid already
crosses activation with the other geometry fields, so this result argues for
full-grid selection rather than another universal activation rule.

#### Convolution topology ablation

The topology experiment held data, folds, initializer, Keras-compatible Adam,
downstream cleavage/pooling/flank-average heads, and the two representative
architectures fixed. It replaced Conv1D with either one shared position-wise
Dense transform or a two-layer position-wise MLP whose parameter count is
within 0.2% of the corresponding convolutional network. Neither control mixes
information between neighboring residues.

Fourteen paired comparisons completed on the same 10 frozen held-out samples.
Every completed non-convolutional comparison lost AUPRC and PPV in all 10
samples. Percent changes below are macro changes against the paired
convolutional model:

| Transform | Flanks | Architecture | AUPRC | PPV@N | AUROC |
|---|---|---|---:|---:|---:|
| Position-wise Dense | 15 aa | Small/tanh | -84.62% | -83.54% | -19.65% |
| Position-wise Dense | 15 aa | Large/ReLU | -89.41% | -87.67% | -17.97% |
| Position-wise Dense | 15 aa | Selected ensemble | -88.05% | -84.44% | -18.94% |
| Position-wise Dense | 5 aa | Small/tanh | -83.79% | -85.27% | -18.18% |
| Position-wise Dense | 5 aa | Large/ReLU | -89.76% | -89.58% | -17.07% |
| Position-wise Dense | 5 aa | Selected ensemble | -88.28% | -86.05% | -17.84% |
| Position-wise Dense | No flanks | Small/tanh | -91.60% | -97.77% | -22.35% |
| Position-wise Dense | No flanks | Large/ReLU | -92.29% | -94.27% | -21.19% |
| Position-wise Dense | No flanks | Selected ensemble | -92.80% | -94.30% | -21.85% |
| Parameter-matched position-wise MLP | 15 aa | Small/tanh | -81.24% | -77.53% | -15.30% |
| Parameter-matched position-wise MLP | 15 aa | Large/ReLU | -89.09% | -84.29% | -18.57% |
| Parameter-matched position-wise MLP | 15 aa | Selected ensemble | -87.40% | -83.36% | -16.99% |
| Parameter-matched position-wise MLP | 5 aa | Small/tanh | -81.73% | -82.60% | -15.23% |
| Parameter-matched position-wise MLP | 5 aa | Large/ReLU | -90.77% | -89.65% | -20.55% |

The no-flank result is not evidence that spatial structure is absent: the model
still sees the peptide sequence, and convolution can recognize local peptide
motifs and cleavage context. Matching parameter count does not rescue the
position-wise model, so capacity alone does not explain the gap. The release
decision is to retain convolutional processing. The 5-aa matched-MLP ensemble
and all no-flank matched-MLP comparisons were intentionally stopped after this
decision; their absence is recorded rather than treated as a successful run.

### Decision before the full retrain

The frozen benchmark used above has now been queried repeatedly to choose
training settings. It remains valuable as a fixed development benchmark with
zero training overlap, but it is no longer an untouched confirmatory test.
Hyperparameter pruning must use fold-internal selection plus this explicitly
labelled development evidence. Before the final candidate is trained, reserve
a second source-study holdout without inspecting candidate performance, remove
its samples/pMHCs from training, and access it once for the release gate. If no
such data are available, the release report must state that its benchmark
estimates are post-selection rather than independent validation.

The full 35×4 affinity and 128×4-per-processing-variant run starts only after:

1. equation/initialization parity tests pass on CPU and an accelerator;
2. the small panels finish and their paired tables are preserved;
3. the release YAML explicitly records every selected implementation; and
4. any proposed departure from 2.1.x wins its isolated held-out comparison.

The eventual full candidate still has to beat or match the public model on the
fixed development benchmarks and pass the untouched confirmatory gate. A
plausible recipe is not evidence that the resulting weights are good.
