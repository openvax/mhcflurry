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
supports a change. The only supported training change so far is affinity
minibatch 128 to 1024. The ablations below recheck that result after restoring
the Keras optimizer equations, because optimizer/minibatch interactions are
possible.

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
| Minibatch | 128 | 1024 retained from held-out improvement; recheck in restored-optimizer ablation |
| Validation | Tail 10%; every epoch | Restored, including Keras split rounding |
| Early stopping | patience 20, `min_delta=0`, ceiling 5000 epochs | Restored |
| Pretraining | 64 peptides/chunk, 256 steps/epoch, min 5/max 50 epochs, patience 2, `min_delta=1e-4`, maximum accepted validation loss 0.10 | Exact match |
| Random negatives | Per-allele nonbinder equalization, rate 1, constant 1, 30,000–50,000 nM, binder threshold 500 nM, fresh pool each epoch | Exact distribution and pool lifetime; seeded identities differ |
| Selection | 4 folds, minimum 2 / maximum 8 models per fold | Exact match |

### Processing

| Category | Published 2.1.x | 2.3.0 decision |
|---|---|---|
| Architectures | `tanh`/`relu` × 256/512 filters × kernels 11/13/15/17 × L1 0/`1e-6` × dense `[8]`/`[16]` × dropout 0.3/0.5 | Exact match: 128 architectures × 4 folds per flank variant |
| Input | BLOSUM62; peptide maximum 15; 15/15, 0/0, and 5/5 flank variants | Exact encoding and variants; expansion now happens on-device |
| Pooling/topology | Spatial dropout; N/C cleavage features; internal peptide maxima; flank averages; sigmoid output | Exact formula and masks |
| Initializer | Every Conv1D/Dense kernel Glorot uniform, every bias zero; final output kernel ones | Restored and equation-tested; `init=glorot_uniform` is explicit |
| Loss | Binary cross-entropy | Same objective; PyTorch and Keras use algebraically equivalent stable/clipped kernels with possible last-bit differences near saturation |
| Optimizer | Keras Adam: LR `0.001`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-7` using Keras' non-adaptive epsilon placement | Restored with public `KerasAdam`; `optimizer_implementation=keras` is explicit |
| Minibatch | 512 in the published generator | Exact match |
| Validation / early stop | Tail 10%, patience 20, `min_delta=0`, ceiling 500 epochs | Exact match, including Keras split rounding |
| Holdout / selection | 10 samples per fold; minimum 1 / maximum 2 models per fold | Restored |
| Decoys | Proteome-derived; PPV multiplier 100; retain 2 candidates per hit | Exact recipe |

### Presentation combiner

Presentation is logistic regression rather than a neural network, but it is
part of the end-to-end result. The compatibility recipe is two proteome decoys
per hit, sample fraction 0.1, the three study exclusions, `short_flanks` (5 aa
on each side), and L-BFGS with 100 iterations. L-BFGS itself is supported; an
upstream SciPy warning concerns deprecated display options, not the optimizer.

## Framework-semantic discrepancies found

| Discrepancy | Release relevance | Resolution / justification |
|---|---|---|
| Processing layers silently used PyTorch Kaiming/fan-in initialization and nonzero random biases | Direct: every processing candidate | Fixed. Glorot/zero bias is the baseline; former behavior is an explicit ablation value, not a default. |
| Affinity LSUV observed raw Linear output instead of the activated Dense output | Direct: every affinity candidate uses LSUV | Fixed. Post-activation is historical parity; pre-activation remains selectable for an ablation. |
| Native PyTorch RMSprop places epsilon outside the square root; Keras places it inside | Direct: every affinity optimizer step | Fixed with a public, tested Keras equation and an explicit implementation switch. PyTorch documents this framework difference. |
| Native PyTorch and Keras Adam place epsilon differently relative to bias correction | Direct: every processing optimizer step | Fixed with a public, tested Keras equation and an explicit implementation switch. |
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
claim of universal optimizer superiority, determines the 2.3.0 default.

The public class defaults also changed independently of the release YAML:
affinity defaults from minibatch 128 to 512, while the release explicitly uses
1024; processing defaults from 256 to 512, while both published and current
release generators explicitly use 512. These default changes cannot silently
alter the release because the generated YAML records the actual values.

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

Primary metrics are frozen-holdout affinity AUROC, AUPRC, and PPV@N. Secondary
diagnostics are pretrain failure rate, inequality-aware validation loss,
early-stop epoch, gradient/weight norms, and wall time. Each alternative is
scored directly against the published-parity condition so the saved tables use
unambiguous A/B columns. Retain a non-baseline setting only for a consistent
paired improvement without a locus/allele subgroup regression.

### Processing initializer/optimizer panel

Use two deliberately different architectures across four folds: a smaller
tanh/256/kernel-11/dense-8/dropout-0.3/L1-0 network and a larger
relu/512/kernel-17/dense-16/dropout-0.5/L1-`1e-6` network. Cross:

1. Glorot + Keras Adam (published parity),
2. Kaiming/fan-in + Keras Adam,
3. Glorot + PyTorch Adam, and
4. Kaiming/fan-in + PyTorch Adam (the rejected port's coupled behavior).

Run the 15/15 and no-flank variants first; add 5/5 only if the direction differs
by variant. Primary metrics on the frozen 10 samples are AUPRC and PPV, with
AUROC, calibration, early-stop epoch, stability, and peak memory secondary.

### Decision before the full retrain

The full 35×4 affinity and 128×4-per-processing-variant run starts only after:

1. equation/initialization parity tests pass on CPU and an accelerator;
2. the small panels finish and their paired tables are preserved;
3. the release YAML explicitly records every selected implementation; and
4. any proposed departure from 2.1.x wins its isolated held-out comparison.

The eventual full candidate still has to beat or match the public model on the
frozen component and end-to-end presentation benchmarks. A plausible recipe is
not evidence that the resulting weights are good.
