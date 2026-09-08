---
orphan: true
---

# Orchestrator architecture

> Internals doc — for contributors. End users running
> `mhcflurry class1-train-pan-allele-models` don't need to read this.

## The one-line summary

The orchestrator owns the **run-wide resources** (parallel workers,
env tuning); workers consume those and own their per-fit device state.
Workers never build shared state and never set env knobs that affect
other workers.

If you find yourself adding orchestrator-shaped logic inside
`fit()`, `train_model()`, or any worker function, stop and think
whether it belongs in the orchestrator instead.

## Locus of control

```
orchestrator process               training worker process
─────────────────────              ───────────────────────

parse args                         (forked / spawned)
load training data                  receive WORKER_CONTEXT once
build WORKER_CONTEXT        ─┐
                             │
hoist env knobs              │
  TORCHINDUCTOR_COMPILE_THREADS
                             │
start worker pool            ─┘
                                    fit() builds device-resident
                                      AffinityDeviceTrainingData
                                      (peptide/allele/y/weights and
                                      random-negative slice live on
                                      the active torch device)
collect results  ◀──────────────────  return predictor
```

The orchestrator does the **expensive, single-threaded preparation**
ONCE. Workers do the **parallel, per-task** work N times.

## Tensor residency: device-resident affinity fit

Affinity `fit()` is **device-resident**. One container,
`mhcflurry.class1_affinity_training_data.AffinityDeviceTrainingData`, holds
the row space `[random negatives | real training rows]` as torch tensors on
the active device for the lifetime of one `fit()` call. The inner loop
forms batches via `index_select` against the combined buffer — no per-batch
host-to-device copies, no DataLoader workers feeding the affinity loop.

Random-negative peptides are refilled in-place once per epoch into the
top slice of the combined buffer, so the real-data block is never
recopied.

The pretrain-streaming path (`fit_streaming_batches`) is a separate code
path that *does* use a PyTorch DataLoader with optional prefetch workers.
That's controlled by `dataloader_num_workers`; 0 means in-process. It
feeds the pretrain generator only and does not affect the affinity row
layout above.

## Random negatives

Random negatives are worker-local. `RandomNegativesPool` amortizes
generation over `random_negative_pool_epochs`, but it does not try to
share encoded pools between model fits. This preserves the historical
behavior where independently trained workers sample independent negative
peptides.

For the default torch peptide-encoding path, the pool samples amino-acid
indices directly on the worker's active torch device and writes fixed-length
int8 rows. The model's embedding layer expands those indices to BLOSUM62 /
PMBEC / physchem features during the forward pass. Legacy host-vector
models fall back to the host encoder so their input shape remains unchanged.

## Parallelism backends

Two backends, one CLI surface:

- **Local** (`mhcflurry.parallelism`): `multiprocessing.Pool` of
  non-daemon workers. Workers can spawn DataLoader children for the
  pretrain streaming path. `resolve_local_parallelism_args` is the
  single pre-fork normalization point: it resolves
  `--max-workers-per-gpu=auto` and `--dataloader-num-workers=auto`
  without touching CUDA, caps local `--num-jobs` to GPU capacity when
  auto was requested, and hoists torch.compile's thread cap before the
  Pool starts. Fork workers inherit `WORKER_CONTEXT` through copy-on-write;
  spawn workers receive it once in their initializer, never once per queued
  task. Before a spawn pool starts, the planner deep-sizes that loaded context
  and rechecks current host RAM, so worker copies are capped even when the
  source CSV was compressed or torch compilation is disabled. Explicit numeric
  `--max-workers-per-gpu` keeps the
  historical CPU-overflow behavior.
- **Cluster** (`cluster_parallelism.py`): one job per work-item
  submitted via `bsub` / `sbatch` / `sh`. Workers serialize
  WORKER_CONTEXT to NFS and deserialize on the worker side.

Worker-side code (`train_model()`, etc.) is identical between
backends. Only the orchestrator branches on `args.cluster_parallelism`.

## Coverage matrix

The 2.3.0 modernization concentrates on the pan-allele affinity
training + percentile calibration paths. Other components inherit
shared backend selection and worker orchestration, but their data
paths remain intentionally smaller.

| | pretrain | finetune | select | calibrate |
|---|---|---|---|---|
| **affinity (pan-allele)** | streaming DataLoader + compact torch-index peptide batches | device-resident tensors + worker-local RN pool | filter ✓; pool/cache n/a (no fit) | filter ✓; pool/cache n/a |
| **affinity (allele-specific)** | n/a | device-resident tensors + worker-local RN pool | filter ✓ | shares calibrate command |
| **processing** | n/a | local+cluster worker pool | local+cluster worker pool | n/a (allele-independent) |
| **presentation** | n/a | parallel auto-sized feature generation + deterministic fit | n/a | filter ✓ (shares calibrate command) |

## Auto-tuned parallelism knobs

Three knobs auto-derive from the box's hardware so the orchestrator
keeps working when the recipe lands on a different tier. Every auto
resolver lives in `mhcflurry.parallelism` and is exercised by
the unit-test matrix in `test/test_orchestrator_helpers.py`. The
production recipes pass `auto` for each; pin a literal int only when
intentionally re-benchmarking.

The design boundaries and remaining follow-up work are recorded in
{doc}`auto_sizing_audit`.

### `--max-workers-per-gpu auto` → `auto_max_workers_per_gpu`

Picks per-GPU worker concurrency from the number of complete estimated worker
working sets that fit in detected free VRAM after one shared reserve (10%, at
least 1 GiB). There is no default worker hard cap. The working set is derived
from model artifacts/parameters, gradients and optimizer state for training,
resident encoded data, and the configured training minibatch where those facts
are available; otherwise the workload profile supplies a conservative fallback.
Free VRAM is read from `nvidia-smi` (no torch import — the parent process must
not initialize CUDA before forking). The final job count is also bounded by
available host RAM and vCPUs. In containers, host-memory sizing honors the
current cgroup limit and usage rather than the physical host's `/proc/meminfo`.
Spawn pools then refine the host cap from the loaded per-worker context and
current available RAM; fork pools retain copy-on-write sharing and skip that
copy multiplier. If no additional spawn copy fits safely, automatic local
execution falls back to serial work in the parent instead of forcing one child.

Before affinity or processing training starts, a one-worker resource probe runs
one bounded epoch for each resource-distinct architecture with production-shaped
data residency, minibatching, and validation. It records PyTorch allocator peaks,
whole-process CUDA memory (including the CUDA context), and process peak RSS. A
measured peak can tighten an automatic plan before the production pool starts,
but can never change explicit CLI values. When torch compilation is enabled the
same pass also primes its on-disk cache.

Runtime validation, calibration, and inference do not divide live free VRAM by
the worker count. That value already reflects whichever peer workers happened to
initialize first and made batch sizing depend on startup order. Instead, every
worker receives a fixed share of launch-time free device capacity after one
shared reserve, subtracts its own resident process working set, and caps the
remainder by live global headroom. The launch snapshot accounts for unrelated
processes that already occupy a card. This keeps launch packing and runtime
batches within the same resource envelope.

Release training sets ``MHCFLURRY_FAIL_ON_TRAINING_BATCH_SHRINK=1``. The public
API continues to warn and shrink an oversized affinity-training batch, but a
release run fails because changing the effective minibatch changes the
optimization problem and therefore the resulting weights.

For the generic 4 GiB fallback, 80 GiB free fits 18 workers and 40 GiB fits 9;
workload-specific estimates, host RAM, vCPUs, and the number of work items can
reduce those capacities. CPU-only execution resolves to one device worker.

### `--dataloader-num-workers auto` → `auto_dataloader_num_workers`

Picks the per-fit-worker DataLoader prefetch child count from box
capacity. This applies to the pretrain streaming path
(`fit_streaming_batches`); the affinity `fit()` path is device-resident
and ignores it. Inputs: total vCPUs, total RAM in GB, the post-cap
fit-worker count (`--num-jobs` resolved by
`resolve_local_parallelism_args`), and a hard cap (default 4 — the
empirical SM-scheduler-style wall, env-overridable via
`MHCFLURRY_AUTO_DATALOADER_HARD_CAP`).

Heuristic:

1. `cpu_per_fit = vcpus // num_fit_workers`
2. `cpu_cap = cpu_per_fit // 2` — each DL child needs ~2 effective
   cores (1 for the main loop, 1 for queue/collate/copy).
3. RAM cap (when `ram_gb` is provided): assume ~2 GB baseline per
   fit-worker plus ~0.5 GB per DL child;
   `ram_cap = max(0, (ram_gb / num_fit_workers - 2.0) / 0.5)`.
4. Result: `min(hard_cap, cpu_cap, ram_cap)` clamped to ≥1, except
   when any input cap is 0 (oversubscribed mains, RAM exhaustion, or
   user override `MHCFLURRY_AUTO_DATALOADER_HARD_CAP=0`) in which case
   the result is 0 — i.e. in-process batching, no children.
5. Serial / no-fit-worker case: returns 0.

Cross-checked configurations (see
`test_auto_dataloader_num_workers_hardware_tiers`):

| Box | vCPU | RAM | fit | resolved |
|---|---|---|---|---|
| 8×A100-80GB Verda | 176 | 400 G | 16 | **4** |
| 8×A100-40GB | 176 | 400 G | 8 | 4 |
| 8×L40S sweep box | 96 | 200 G | 16 | 3 |
| Single A100-80G Lambda | 30 | 200 G | 2 | 4 |
| Single A100-80G tight | 16 | 64 G | 2 | 4 |
| Single T4 / RTX | 8 | 16 G | 1 | 4 |
| Tight cluster node | 32 | 64 G | 16 | 1 |
| Very tight (8v / 8fit) | 8 | 16 G | 8 | **0** |
| RAM-starved (32G / 16fit) | 176 | 32 G | 16 | **0** |
| Serial / CPU | — | — | 0 | **0** |

**The heuristic is hardware-only by design** — `train_row_count` and
`random_negative_rate` do not enter the formula. Per-batch CPU work is
bounded by `minibatch_size`, not total dataset rows.

### `--num-jobs` (auto-derives from MWPG × GPUs)

`--num-jobs` defaults to `auto`, which resolves to `gpus × max_workers_per_gpu`
via `mhcflurry.parallelism.auto_num_jobs(num_gpus, max_workers_per_gpu)`
once `auto_max_workers_per_gpu` has resolved. Pass an integer to pin it.

### Cross-model coverage

| Model | `--max-workers-per-gpu auto` | `--dataloader-num-workers auto` | Notes |
|---|---|---|---|
| Pan-allele affinity | ✓ | ✓ (pretrain only) | Default in release recipe; affinity `fit()` is device-resident |
| Allele-specific affinity | ✓ | ✓ | Same `Class1NeuralNetwork` codebase; auto already wired |
| Processing | ✓ | n/a | Parallel model fits; the processing network has no DataLoader pretraining path. |
| Presentation | ✓ | n/a | Parallel feature generation followed by deterministic fitting. |

### Env overrides

| Env var | Default | Effect |
|---|---|---|
| `MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB` | 4.0 generic fallback | Expert per-worker VRAM override; workload-specific estimates normally replace the fallback. |
| `MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP` | unset | Optional expert ceiling for workers per GPU. |
| `MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB` | (auto-detect) | Pin free VRAM (CSV per GPU); for tests / hidden-`nvidia-smi` launchers |
| `MHCFLURRY_AUTO_DATALOADER_HARD_CAP` | 4 | DL child cap for `auto_dataloader_num_workers` |

## Env knobs vs CLI flags

A persistent question: when does a setting belong on `argparse` vs
the environment?

**Rule of thumb:**

- **CLI flag** when the orchestrator owns it and propagates it.
  (Examples: `--max-workers-per-gpu`, `--random-negative-pool-epochs`,
  `--num-jobs`.)
- **Env var with optional CLI relay** when the consumer is inside
  `fit()` or another worker-private code path, and the orchestrator
  only centralizes policy. (Examples: `MHCFLURRY_TORCH_COMPILE`,
  `MHCFLURRY_TORCH_COMPILE_LOSS`, `TORCHINDUCTOR_COMPILE_THREADS`.)

The orchestrator may **hoist** env vars: read its own args, compute a
sensible default, and put a concrete value in `os.environ` so workers
inherit it. `resolve_local_parallelism_args` calls
`hoist_torchinductor_compile_threads` for local runs — it sizes the
inductor compile pool against the resolved `--num-jobs` so N workers
don't each spawn `cpu_count()` compile threads. If
`TORCHINDUCTOR_COMPILE_THREADS=auto`, the orchestrator replaces it
with a numeric value before any worker sees it.

## Torch Compile Warmup

`torch.compile` is worker-local: the Python wrapper, graph guards, and
CUDA module handles cannot be shared across processes. What can be
shared on one machine is the Inductor/Triton on-disk cache. For local
Pool training, the orchestrator therefore runs one real work item
first in a one-worker warmup pool with a larger compile-thread budget,
saves that result, then restores production compile-thread sizing and
launches the full worker pool.

Cluster workers are different: they may land on different nodes, so
mhcflurry does not try to share Inductor cache across a cluster. Each
cluster worker process auto-sizes `TORCHINDUCTOR_COMPILE_THREADS`
locally when the env is unset or `auto`. If a scheduler packs multiple
mhcflurry work items onto one node, set
`MHCFLURRY_CLUSTER_WORKERS_PER_NODE` so each process uses a fair share
of cores.

Compiled losses are enabled by default when `MHCFLURRY_TORCH_COMPILE=1`
and can be disabled with `MHCFLURRY_TORCH_COMPILE_LOSS=0` or
`--torch-compile-loss 0`. CUDA workers run a one-op autograd warmup
before compiling losses to avoid the PyTorch 2.4 / Triton
`invalid device context` failure in the first compiled backward kernel.

## What is NOT the orchestrator's job

- Compiling models (`torch.compile`). Compilation is per-network and
  happens inside `fit()`. The orchestrator only sizes the compile
  worker pool via env.
- Calling `predict()` on individual alleles. The orchestrator
  builds work items; workers iterate.
- Validating data shape consistency between work items beyond what
  the worker-side fit/data constructors already check.

## Recipes

### Adding a new pre-fork resource

1. Add a `_initialize_<name>(args, all_work_items)` helper near the
   existing ones in `train_pan_allele_models_command.py`.
2. Stash the result in `WORKER_CONTEXT["<name>"]`.
3. Document the lookup key. Workers retrieve via
   `constant_data["<name>"]` (forked workers inherit; spawned workers receive
   one initializer payload per process; cluster workers receive the serialized
   context through the shared filesystem).
4. Add a `getattr(args, "<flag>", None)` gate so the helper is opt-in
   on the CLI side.

### Adding a new env knob

1. Decide: orchestrator-set or worker-private?
2. If orchestrator-set: add a `_hoist_<knob>(args)` helper that reads
   args + system info and `os.environ.setdefault(...)`s the value.
   Document the rule the orchestrator applies.
3. If worker-private: just read it from `os.environ` inside `fit()`
   or wherever it's consumed. Document it in
   `RELEASE_NOTES_<version>.md`.

### Adding a new worker-side filter

1. Drop into `mhcflurry/common.py` (e.g.,
   `filter_canonicalizable_alleles`).
2. Apply at every iteration site that could trip on the bad input.
   The 2.3.0 application sites are calibrate (affinity + presentation
   paths) and select (pan-allele + allele-specific paths).

## Fit-time data flow

Affinity training has two configuration knobs that are deliberately
configured separately. Do not collapse these into one.

### 1. Peptide amino-acid vector lookup

Controlled by the model hyperparameter
`peptide_amino_acid_encoding_torch`.

When enabled (the default), peptide strings are encoded as `(N, L)`
integer amino-acid indices. The model owns a frozen torch buffer for
the configured vector encoding (`BLOSUM62`, `PMBEC`, `simons1999_contact`,
or combinations), moves that buffer with `.to(device)`, and widens
indices to `(N, L, V)` inside `forward()` using
`torch.nn.functional.embedding`.

This path is device agnostic: CUDA, MPS, and CPU all execute the same
torch embedding operation on the active device. It is model semantics,
not DataLoader behavior.

Disable it only with `peptide_amino_acid_encoding_torch=False`, which
restores the old path where numpy expands peptide strings into
`(N, L, V)` vectors before they are moved to the device.

### 2. Pretrain DataLoader process parallelism

Controlled by the model hyperparameter `dataloader_num_workers`.

This applies to the pretrain streaming path
(`fit_streaming_batches`) only. It is pure process-count policy
for the pretrain data pipeline:

| value | meaning |
|---|---|
| `0` | build batches in the training worker process |
| `>0` | spawn that many DataLoader child processes per training worker |

Release recipes set only this knob. On a local 8-GPU run with
`NUM_JOBS=16`, `dataloader_num_workers=1` means up to 16 extra
fit-local DataLoader children while pretrain epochs are active; `2`
means up to 32. The thread-budget helper accounts for this when sizing
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS`.
Caller-provided values remain authoritative and are not replaced by the
planner's uniform runtime limit. For serial (`num_jobs=0`) execution, an
auto-owned budget is applied directly to the current native and PyTorch pools
because there is no worker initializer.

The affinity `fit()` path does not use a DataLoader and therefore
ignores this hyperparameter.

### Component consistency

The `Class1NeuralNetwork` affinity component model owns both
hyperparameters above, so allele-specific affinity models, pan-allele
affinity models, and affinity ensembles resolve the same internal rules
after config load. Missing keys in old component configs are filled
from defaults (`peptide_amino_acid_encoding_torch=True`,
`dataloader_num_workers=0`).

Processing models do not use the affinity `fit()` row layout and
therefore have no `AffinityDeviceTrainingData`.

## rsync hygiene (laptop ↔ remote training box)

`runplz` stages the working tree on the box for every launch, and the run's
`out/` directory returns to the laptop on completion. Two asymmetries to know
about:

- **Up direction:** runplz 3.24.31 uses Git-aware staging, HUP-safe detached
  bootstraps, shared backend retries, and retryable idempotent SSH preparation.
  Tracked files and
  intentional untracked inputs are staged, while ignored directories such as
  `brev_runs/`, `results/`, `.venv/`, and `out/` are excluded without a
  workstation-specific relocation or symlink workaround. The release wrapper
  requires exactly 3.24.31 and rejects a dirty editable runplz checkout or a
  source/distribution version mismatch.
- **Down direction:** the post-run rsync has NO excludes — everything
  under `out/` returns. Keep large throwaway run artifacts outside
  `out/` unless they are meant to ship back.

## Pointers to code

- Random-negative planning and pooling: `mhcflurry/random_negative_peptides.py`
- Affinity device-resident row space:
  `mhcflurry/class1_affinity_training_data.py` (`AffinityDeviceTrainingData`)
- Pseudogene/null filter: `mhcflurry/common.py`
  (`filter_canonicalizable_alleles`)
- Worker pool sizing: `mhcflurry/parallelism`
  (`auto_max_workers_per_gpu`, `resolve_max_workers_per_gpu`)
- Compile-thread hoist:
  `mhcflurry.parallelism.hoist_torchinductor_compile_threads`
- Cluster fork point:
  `mhcflurry/cluster_parallelism.cluster_results`

## Known asymmetries (deliberate)

These show up to readers as "why is this only done in pan-allele?"
The answer in each case is "the other components don't yet need it
and adding it would be feature work, not a fix":

- **`torch.compile`** is off by default everywhere; opt-in via
  `MHCFLURRY_TORCH_COMPILE=1` or `--torch-compile 1`. When enabled,
  the shared local-parallelism layer owns compile-thread sizing and
  local one-worker cache warmup for affinity and processing trainers.
  Presentation fitting is a separate model family and does not enter
  this central torch-training path.

## Future tightening (not in 2.3.0)

- **Presentation-training orchestration.** Today
  `train_presentation_models_command` runs single-process; mirror the
  pan-allele orchestration shape if presentation retraining becomes
  GPU-bound.
