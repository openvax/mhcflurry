# Resource auto-sizing audit

This audit was triggered by the 2.3.0rc20 affinity run on four A100-40GB GPUs.
The initial plan packed 12 workers per GPU from a 2.5 GB analytic estimate. A
one-minibatch compile warmup measured only 0.1 GB and confirmed that plan because
it deliberately omitted full data residency and validation. The production pool
then reached 2.7–4.1 GiB per process and one validation forward failed with only
607 MiB free.

## Findings

| Decision | Previous evidence source | Generalization problem | Disposition |
|---|---|---|---|
| GPU workers per device | Workload constants plus analytic model/data formulas | A formula for steady training tensors was treated as the peak for every phase. Historical constants were repeatedly retuned to individual release runs. | Keep formulas only as launch fallbacks; tighten from a real peak-phase probe before training pools start. |
| Validation and prediction batch | Live free VRAM divided by declared co-resident workers | Live free VRAM already contains allocations from earlier-starting peers. Dividing it again creates startup-order-dependent batches and does not prevent simultaneous claims. | Snapshot launch-time free capacity, allocate a fixed per-worker share, subtract the current process working set, then cap by live headroom. |
| Compile warmup | One minibatch, validation disabled | It tested compilation, not the production resource envelope, yet its allocator peak was used to validate concurrency. It also disappeared when compilation was disabled. | Replace it with a compilation-independent full-residency resource probe; compilation cache warming is an optional side effect. |
| Calibration batch | A second cache/scratch formula over live free VRAM | It duplicated the global memory partition and inherited the same live-free race. | Route it through the shared per-worker entitlement, retaining only calibration-specific future cache and cartesian-forward terms. |
| Host worker count | Static workload RSS plus loaded spawn-context size | The spawn-context refinement is useful, but dependent DataLoader and random-negative decisions are made before late refinements and are only safe because refinements can tighten. | Retain for 2.3.0; recompute dependent auto knobs to a fixed point in the follow-up planner cleanup. |
| DataLoader children | 2 cores and 0.5 GB per child, capped at 4 | These are throughput heuristics, not measured memory requirements. They are documented and conservative but hardware-specific. | Keep as a bounded performance policy; separate it from correctness capacity in the follow-up API. |
| Random-negative pool epochs | 1 GB per worker/epoch, capped at 10 | The function accepts peptide/count inputs but discards them; the estimate is an empirical release-run constant. | Keep the conservative value for 2.3.0 and replace it with measured/shape-derived host allocation accounting later. |
| Training minibatch guard | Hand-written model-width multipliers and live memory | Silent shrinking changes batch-normalization and optimization behavior, so resource pressure can change trained weights. | Continue warning for the public API, but make an unexpected shrink a release-provenance failure. |

The resource code is large because it mixes three different concerns: correctness
capacity, elastic batch sizing, and throughput tuning. A single from-scratch
replacement in the 2.3.0 release branch would have excessive blast radius. The
right holistic rewrite is staged around a smaller shared contract rather than a
new collection of workload constants.

## Planner invariants

1. The sum of automatic worker entitlements never exceeds launch-time free
   device memory after one shared reserve, including when unrelated processes
   already occupy the GPU.
2. A worker's elastic allocation is bounded by both its remaining entitlement
   and current global headroom; worker startup order cannot increase it.
3. A measurement can only tighten an automatic plan. Explicit user concurrency
   remains authoritative and receives diagnostics instead of mutation.
4. A training probe exercises every phase that can own the peak: real resident
   inputs, the configured minibatch and optimizer, and validation.
5. Process-level measurement includes CUDA context and non-PyTorch allocations;
   allocator counters alone are insufficient for multi-process packing. A
   before-context driver baseline covers containers whose host and container
   process IDs differ.
6. Resource decisions are visible in the run log, and release workflows turn
   any training-affecting shrink into a hard provenance failure.

## Rewrite boundary

The 2.3.0 release-blocking change implements invariants 1–5 for affinity and
processing training and shares the same entitlement with prediction and
calibration batches. Hardware discovery, cgroup-aware host-memory detection,
explicit overrides, and the pure shared-reserve arithmetic remain useful and do
not need replacement.

After 2.3.0, consolidate the remaining throughput heuristics behind a structured
resource-envelope API, recompute dependent auto knobs after every late
refinement, persist the final plan as JSON, and add measured probes for the
remaining calibration/inference workloads. Track that follow-up in
[issue #363](https://github.com/openvax/mhcflurry/issues/363).

## Release gate

The next A100-40GB run must show that the resource probe tightens the original
analytic plan, completes affinity training with no automatic minibatch shrink or
OOM, and records a stable per-worker device entitlement. The end-to-end release
run remains the empirical validation for prediction-affecting training changes.
