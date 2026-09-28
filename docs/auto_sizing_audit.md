# Resource planning guarantees

The planner combines conservative capacity estimates with real-workload probes.
Use {doc}`configuration` for user controls and {doc}`orchestrator` for process
ownership and implementation boundaries.

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
7. Automatic CUDA processing batches are calibrated after the real model and
   encoded inputs are resident. The measured peak may only shrink the analytic
   batch, and the selected value cannot exceed the successfully exercised probe
   shape. If a probe OOMs, calibration halves the probe until one succeeds; if
   allocator telemetry is unavailable, it uses the successful probe instead of
   restoring an unverified analytic value. Explicit batches are unchanged, and
   elastic halving remains the final allocator-specific safety net.

## Hardware validation matrix

The autosizer tests machine *characteristics*, not accelerator names. A label
such as ``4xH100`` makes a test case readable, but the planner receives only
GPU count and free memory per GPU, available host RAM, available CPU units, and
the workload envelope. This keeps a new card or cloud shape from requiring a
model-name lookup table.

Each matrix row validates two plans:

1. The provisional launch plan packs the analytic worker estimate into the
   shared-reserve GPU budget, then clamps total jobs by host RAM, CPU count,
   and available work items. DataLoader children are resolved from the
   resulting CPU/RAM share and host capacity is checked again.
2. The measured plan replaces optimistic worker estimates with the maximum
   full-residency probe peaks plus safety margins. It may only tighten an
   automatic plan. Explicit concurrency is never mutated.

The test matrix includes CPU-only, single- and multiple-GPU hosts and CPU/RAM
constrained variants. These are tests of capacity arithmetic, not runtime
benchmarks of every accelerator. Real planning uses detected free capacity,
not a hardware-name lookup table.

## Limits

Host worker, DataLoader and random-negative pool estimates include conservative
throughput heuristics. They do not establish identical runtime peaks on all
hardware. Explicit user concurrency remains authoritative; release workflows
reject unexpected training minibatch reductions because those can alter the
trained model. Further planner consolidation is tracked in
[issue #363](https://github.com/openvax/mhcflurry/issues/363).
