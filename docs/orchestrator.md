# Orchestrator architecture

> Internals doc — for contributors. End users running
> `mhcflurry-class1-train-pan-allele-models` don't need to read this.

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
load training data                  inherit GLOBAL_DATA
build GLOBAL_DATA           ─┐
                             │
hoist env knobs              │
  TORCHINDUCTOR_COMPILE_THREADS
                             │
fork worker pool             ─┘
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

- **Local** (`local_parallelism.py`): `multiprocessing.Pool` of
  non-daemon workers. Workers can spawn DataLoader children for the
  pretrain streaming path. `resolve_local_parallelism_args` is the
  single pre-fork normalization point: it resolves
  `--max-workers-per-gpu=auto` and `--dataloader-num-workers=auto`
  without touching CUDA, caps local `--num-jobs` to GPU capacity when
  auto was requested, and hoists torch.compile's thread cap before the
  Pool forks. Explicit numeric `--max-workers-per-gpu` keeps the
  historical CPU-overflow behavior.
- **Cluster** (`cluster_parallelism.py`): one job per work-item
  submitted via `bsub` / `sbatch` / `sh`. Workers serialize
  GLOBAL_DATA to NFS and deserialize on the worker side.

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
| **presentation** | n/a | serial only (single-process; no orchestration story today) | n/a | filter ✓ (shares calibrate command) |

## Auto-tuned parallelism knobs

Three knobs auto-derive from the box's hardware so the orchestrator
keeps working when the recipe lands on a different tier. Every auto
resolver lives in `mhcflurry.local_parallelism` and is exercised by
the unit-test matrix in `test/test_orchestrator_helpers.py`. The
production recipes pass `auto` for each; pin a literal int only when
intentionally re-benchmarking.

### `--max-workers-per-gpu auto` → `auto_max_workers_per_gpu`

Picks the per-GPU worker concurrency from `min(num_jobs / num_gpus,
floor(0.6 × free_vram_gb / per_worker_gb), hard_cap=4)`. Free VRAM is
read from `nvidia-smi` (no torch import — the parent process must not
initialize CUDA before forking). Per-worker VRAM upper bound is
conservative (16 GB) and tunable via
`MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB`.

| Box | num_gpus | free_vram | resolved |
|---|---|---|---|
| 8×A100-80GB | 8 | ~80 GB | **2** |
| 8×A100-40GB | 8 | ~40 GB | **1** |
| 1×A100-80GB | 1 | ~80 GB | **2** |
| CPU-only | 0 | — | **1** |

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

Today the recipe explicitly sets `--num-jobs $((GPUS * MAX_WORKERS_PER_GPU))`
in the shell. `mhcflurry.local_parallelism.auto_num_jobs(num_gpus,
max_workers_per_gpu)` is the in-Python equivalent for callers that
want to derive it after `auto_max_workers_per_gpu` has resolved.

### Cross-model coverage

| Model | `--max-workers-per-gpu auto` | `--dataloader-num-workers auto` | Notes |
|---|---|---|---|
| Pan-allele affinity | ✓ | ✓ (pretrain only) | Default in release recipe; affinity `fit()` is device-resident |
| Allele-specific affinity | ✓ | ✓ | Same `Class1NeuralNetwork` codebase; auto already wired |
| Processing | ✓ | (no-op for now) | `Class1ProcessingNeuralNetwork` does not yet expose `dataloader_num_workers`; flag is accepted via shared `add_local_parallelism_args` so argv stays uniform across train_*_command, but `apply_dataloader_num_workers_to_work_items` won't change processing behavior until that hyperparameter is added. |
| Presentation | n/a | n/a | Single-process today |

### Env overrides

| Env var | Default | Effect |
|---|---|---|
| `MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB` | 16.0 | Per-worker VRAM upper bound for the MWPG resolver |
| `MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP` | 4 | SM-scheduler ceiling for MWPG |
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
2. Stash the result in `GLOBAL_DATA["<name>"]`.
3. Document the lookup key. Workers retrieve via
   `constant_data["<name>"]` (forked workers inherit; spawned/cluster
   workers receive via pickle).
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

`runplz` rsyncs the working tree up to the box on every launch, and
the run's `out/` directory back to the laptop on completion. Two
asymmetries to know about:

- **Up direction:** runplz's hardcoded exclude list (`.git`, `.venv`,
  `__pycache__`, etc.) doesn't anticipate per-project output dirs.
  In mhcflurry, `brev_runs/` accumulates 7-15 GB of past-run
  artifacts that ride along on every launch unless relocated. Run
  `bash scripts/dev/relocate_run_outputs.sh --apply` once to move
  `brev_runs/` and `results/` to `~/mhcflurry-brev-runs/` and
  `~/mhcflurry-results/`, with symlinks back into the repo. After
  that, rsync ships ~tiny symlinks instead of multi-GB directories.
- **Down direction:** the post-run rsync has NO excludes — everything
  under `out/` returns. Keep large throwaway run artifacts outside
  `out/` unless they are meant to ship back.

## Pointers to code

- Random-negative planning and pooling: `mhcflurry/random_negative_peptides.py`
- Affinity device-resident row space:
  `mhcflurry/class1_affinity_training_data.py` (`AffinityDeviceTrainingData`)
- Pseudogene/null filter: `mhcflurry/common.py`
  (`filter_canonicalizable_alleles`)
- Worker pool sizing: `mhcflurry/local_parallelism.py`
  (`auto_max_workers_per_gpu`, `resolve_max_workers_per_gpu`)
- Compile-thread hoist:
  `mhcflurry/local_parallelism.hoist_torchinductor_compile_threads`
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
