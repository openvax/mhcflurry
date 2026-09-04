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
| Processing prediction batch | Analytic sequence-span and filter-width estimate | It did not observe CUDA's kernel-dependent convolution workspace or masking/pooling temporaries. Thirteen release-evaluation forwards recovered by halving otherwise valid auto batches. A first measured implementation still extrapolated from 4,096 successful rows to about 12,000 and recovered twice under a different valid worker startup order. | Keep the analytic estimate as a portable ceiling, then let a safe real-input CUDA forward probe only tighten automatic batches. Never select above the largest batch exercised successfully: CUDA workspace choice is not reliably linear. CPU and MPS retain the analytic path and elastic retry. |
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
7. Automatic CUDA processing batches are calibrated after the real model and
   encoded inputs are resident. The measured peak may only shrink the analytic
   batch, and the selected value cannot exceed the successfully exercised probe
   shape. If a probe OOMs, calibration halves the probe until one succeeds; if
   allocator telemetry is unavailable, it uses the successful probe instead of
   restoring an unverified analytic value. Explicit batches are unchanged, and
   elastic halving remains the final allocator-specific safety net.

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

## Helper API audit

Private-helper status follows responsibility rather than function size:

| Decision | Helpers | Rationale |
|---|---|---|
| Public API | ``normalize_workload_hints``, ``is_auto_value`` | Both define planner input semantics, are reused throughout resolution, and have direct contract tests. |
| Public API | ``free_vram_per_gpu_from_nvidia_smi_gb`` | Per-device discovery is a reusable hardware boundary. The existing minimum and override-aware APIs build on it. |
| Public API | ``resolve_cpu_thread_budget`` | The numeric budget and whether MHCflurry owns the thread environment are both required by orchestration callers; returning only the number hid half of that contract. |
| Folded | Workload-specific environment-variable name construction | It had one caller and no independent policy, so keeping a named private function obscured the estimate path. |
| Remains private | OS/cgroup parsers, PID-namespace discovery, environment byte parsing, and CUDA-baseline combination | These are platform-specific implementation details behind public memory-discovery and peak-measurement APIs. Their direct tests protect edge cases without making them compatibility promises. |
| Remains private | Host-memory clipping and finite-hint parsing | These are cohesive validation/planning steps whose intermediate forms are not useful to callers; they are covered through the public planner. |

The audit also found a behavioral bug rather than an API-shape problem: the
planner changed a CPU-only serial run from zero fit workers to one before
DataLoader sizing. That contradicted ``auto_dataloader_num_workers`` and
started four loader children. CPU-only serial plans now retain zero for that
decision while still budgeting one main process for resident host memory.

## Release gate

The next A100-40GB run must show that the resource probe tightens the original
analytic plan, completes affinity training with no automatic minibatch shrink or
OOM, and records a stable per-worker device entitlement. The end-to-end release
run remains the empirical validation for prediction-affecting training changes.

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

The representative matrix covers CPU-only hosts, a 32 GB RTX 5090 workstation,
single- and four-GPU A100-40GB hosts, four H100-80GB GPUs, eight H200-141GB
GPUs, and deliberately CPU- and RAM-constrained variants. Nominal memory sizes
come from NVIDIA's published [RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/),
[A100/H100 support table](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html),
and [H200 specifications](https://www.nvidia.com/en-us/data-center/h200/).
Tests pass an explicit free-memory value, since real capacity must use current
free memory rather than nominal card memory.

Only the four-A100 row below is an empirical result. The 5090, H100, and H200
rows are unit tests of capacity arithmetic: they map the observed affinity
workload envelope onto explicit published hardware characteristics, but do not
claim that an unmeasured accelerator generation has the identical runtime
peak.

The empirical golden row is the GCP ``a2-highgpu-4g`` release host: four
A100-40GB GPUs, 48 vCPUs, and 340 GB system memory (the published shape is
documented by [Google Cloud](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines)).
Run ``20260821T234311Z-rc14-gcp-provision-train-release-full-db33e184`` on
commit ``fdf746158`` observed 39.49 GiB free per GPU and 329.3 GiB available
host RAM. Its affinity plan changed as follows:

| Phase | Jobs | Workers/GPU | Device worker | Host worker | Device entitlement | CPU threads/worker |
|---|---:|---:|---:|---:|---:|---:|
| Analytic launch | 48 | 12 | 2.5 GiB | 3.0 GiB | 3.0 GiB | 1 |
| Full-residency probe | 4 | 1 | 40.7 GiB | 12.3 GiB | 35.5 GiB | 11 |

The seven probe architectures completed at minibatch 1024 with a maximum
35.4 GiB CUDA process peak and 11.2 GiB host RSS peak. After explicit
task-boundary cleanup, later architectures began near 2.7 GiB instead of
inheriting the preceding 32--40 GiB allocator state. The production pool then
started exactly four workers, one per GPU, without a training-batch shrink.
