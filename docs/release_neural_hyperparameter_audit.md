# Neural training semantics

MHCflurry 2.3.0 uses PyTorch while retaining support for historical model
weights. Matching a hyperparameter name does not always match the training
equation across frameworks. This reference describes those distinctions;
{doc}`release_training_recipe` is the authoritative recipe for the released
weights. The smaller controlled experiments are described separately in
{doc}`release_2_3_training_experiments`.

## Framework-semantic discrepancies found

| Discrepancy | Release relevance | Resolution / justification |
|---|---|---|
| Processing layers silently used PyTorch Kaiming/fan-in initialization and nonzero random biases | Direct: every processing candidate | Made explicit and equation-tested. Initialization is explicit in every generated recipe; see the final recipe for each processing family. |
| Affinity LSUV observed raw Linear output instead of the activated Dense output | Direct: every affinity candidate uses LSUV | Both targets are explicit. Historical parity uses post-activation; the 2.3.0 affinity recipe selects pre-activation. |
| Native PyTorch RMSprop places epsilon outside the square root; Keras places it inside | Direct: every affinity optimizer step | Fixed with a public, tested Keras equation and an explicit implementation switch. PyTorch documents this framework difference. |
| Native PyTorch and Keras Adam place epsilon differently relative to bias correction | Direct: every processing optimizer step | Both equations are public and tested, and the selected implementation is serialized. The final no-flank and boundary families use Keras-compatible Adam; the short-flank family uses native RMSprop. |
| Both PyTorch trainers rounded validation rows as `floor(N * split)` instead of using Keras' split boundary `floor(N * (1 - split))` | Direct but usually one row per network | Fixed centrally and regression-tested. This is deterministic parity, not an optimization question. |
| PyTorch BatchNorm updates running variance with an unbiased estimate; Keras uses population variance | Inactive in the release affinity grid; processing has no BN | Fixed in `KerasBatchNorm1d` and tested against the Keras equation, so non-release configurations are not left divergent. |
| Generic PyTorch Xavier fan calculation was wrong for the transposed 3-D `LocallyConnected1D` storage | Inactive: release affinity grid has no local layers | Fixed using Keras' `(output_length, flattened_input, filters)` fans. |
| Native SGD fallback used LR 0.001 and unknown optimizer names silently became Adam | Inactive: release uses RMSprop/Adam | Fixed: Keras SGD default LR is 0.01 and unknown names now fail. |
| Keras Glorot/He *normal* initializers use variance-corrected truncated normals; PyTorch's normal initializers are untruncated | Inactive: release uses Glorot uniform | Follow-up only if those non-release initializer values are to remain supported for new training. Loaded historical weights are unaffected. |
| TensorFlow and PyTorch dropout RNGs, shuffles, reduction order, and GPU kernels differ | Direct stochastic trajectory, but not a hyperparameter drift | Irreducible framework difference. Compare distributions and held-out metrics, not byte-identical weights. |
| Fixed master/per-fit seeds replace entropy-derived seeds | Direct identities/trajectory | Intentional reproducibility improvement. It does not change the sampled distributions. |
| Device-side encoding, compact cartesian batches, validation batching, lazy proteome sampling, and prediction chunking | Execution only | Algebra/prediction parity is covered by tests. Release training fails if autosizing would shrink a configured minibatch. |
| `torch.compile` and reduced float32 matmul precision | Could alter trajectory | Off for release; eager + `highest` precision is pinned. |


Native RMSprop adds epsilon outside the square root; Keras-compatible RMSprop
adds it inside. Adam implementations also differ in epsilon placement relative
to bias correction. These are explicit optimizer choices, not transparent
performance switches. See the [PyTorch RMSprop documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html),
[Keras RMSprop documentation](https://keras.io/api/optimizers/rmsprop/), and
[Adam paper](https://arxiv.org/abs/1412.6980).

The public class defaults, compatibility generators, and frozen release recipe
serve different purposes. The compatibility affinity generator defaults to
minibatch 128; the 2.3.0 release recipe explicitly selects 1024 with native
RMSprop and pre-activation LSUV. Always retain the generated hyperparameter YAML
with a training run. Framework RNGs and reduction order preclude byte-identical
retraining; validate prediction quality on a common held-out cohort.
