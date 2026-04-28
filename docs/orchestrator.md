# Orchestrator architecture

> Internals doc — for contributors. End users running
> `mhcflurry-class1-train-pan-allele-models` don't need to read this.

## The one-line summary

The orchestrator owns the **shared resources** (parallel workers,
shared-memory layers, encoding caches, env tuning); workers consume
them. Workers never build shared state and never set env knobs that
affect other workers. The one intentional child-process exception is
fit-local DataLoader prefetch workers, which are bounded by the
orchestrator's CPU-thread plan.

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
prebuild encoding cache      │     fit():
  (run mmap cache)           │       look up encoding cache
                             │       (mmap hit, no rebuild)
prebuild random-neg pool     │
  (run mmap cache)           │       look up random-neg pool
                             │       (mmap hit, no replan)
fork worker pool             ─┘
                                    fit() materializes
                                      fit DataLoader SHM (per-fit)
                                      DataLoader workers see
                                      shared tensor handles
collect results  ◀──────────────────  return predictor
```

The orchestrator does the **expensive, single-threaded preparation**
ONCE. Workers do the **parallel, per-task** work N times.

## Layered shared memory

mhcflurry uses two SHM layers. They share the "build once, share with
many readers" idea but use different OS mechanisms because lifetimes
differ.

| | run mmap cache | fit DataLoader SHM |
|---|---|---|
| **Lifetime** | per-run (persists to disk; survives across runs) | per-`fit()` call |
| **Mechanism** | `numpy.memmap` of files | `torch.Tensor.share_memory_()` (POSIX shm or file descriptors) |
| **Built by** | orchestrator | the `fit()` call inside a worker |
| **Read by** | every training worker | DataLoader spawn workers inside one `fit()` |
| **Mutability** | read-only | mutable (random-neg buffer refilled per epoch) |
| **Holds** | encoded peptides, random-neg peptides | x_peptide, x_allele, y_encoded, sample_weights, random_negative_x_peptide buffer, random_negative_x_allele |

**Why two layers?** The run mmap cache's persist-across-runs property
fits file mmap; the fit DataLoader SHM's mutate-per-epoch property fits
torch shm. The split is intentional, but the API surface is uniform:
each has a `setup_*` factory and a small set of generic helpers — see
`mhcflurry/shared_memory.py`.

**Why not "always fit DataLoader SHM"?** It doesn't survive process
exit; encoding the ~1 M training peptides takes ~30 sec, and we'd pay
it every run. The run mmap cache's mmap files live on disk, so a re-run
hits cache.

**Why not "always run mmap cache"?** It's read-only; the per-epoch
random-negative buffer needs in-place updates. The fit DataLoader SHM
is the right tool for that.

## Parallelism backends

Two backends, one CLI surface:

- **Local** (`local_parallelism.py`): `multiprocessing.Pool` of
  non-daemon workers. Workers can spawn DataLoader children. SHM
  layers both work — workers inherit GLOBAL_DATA via fork, mmap pool
  paths are local-process-visible. `resolve_local_parallelism_args`
  is the single pre-fork normalization point: it resolves
  `--max-workers-per-gpu=auto` and `--dataloader-num-workers=auto`
  without touching CUDA, caps local `--num-jobs` to GPU capacity when
  auto was requested, and hoists torch.compile's thread cap before the
  Pool forks. Explicit numeric `--max-workers-per-gpu` keeps the
  historical CPU-overflow behavior.
- **Cluster** (`cluster_parallelism.py`): one job per work-item
  submitted via `bsub` / `sbatch` / `sh`. Workers serialize
  GLOBAL_DATA to NFS, deserialize on the worker side. The run mmap
  cache works ONLY when the pool dir is on a shared filesystem
  reachable from every worker node — orchestrator emits a loud warning
  when both flags are set so the user can verify.

Worker-side code (`train_model()`, etc.) is identical between
backends. Only the orchestrator branches on `args.cluster_parallelism`.

## Coverage matrix

The 2.3.0 modernization concentrates on the pan-allele affinity
training + percentile calibration paths. Other components inherit
parts of the stack (fit DataLoader SHM is automatic for affinity
`fit()` calls that use `fit_dataloader_backing="auto"` with
`dataloader_num_workers > 0`; the pseudogene filter is now shared) but
their orchestrators don't yet drive the run mmap cache or
encoding-cache prefetch — those are opt-in, not on by default.

| | pretrain | finetune | select | calibrate |
|---|---|---|---|---|
| **affinity (pan-allele)** | full stack ✓ | full stack ✓ | filter ✓; SHM/cache n/a (no fit) | filter ✓; SHM/cache n/a |
| **affinity (allele-specific)** | n/a | local pool only; cache+L1 are opt-in via `prebuild_encoding_caches` | filter ✓ | shares calibrate command |
| **processing** | n/a | local+cluster pool; cache+L1 are opt-in | local+cluster pool | n/a (allele-independent) |
| **presentation** | n/a | serial only (single-process; no orchestration story today) | n/a | filter ✓ (shares calibrate command) |

"Opt-in" cells aren't *broken* — they just inherit the worker-side
fit DataLoader SHM (free with `dataloader_num_workers > 0`) and don't
yet have orchestrator-side run mmap cache prebuild. When their
datasets grow to where prebuild matters, call `prebuild_encoding_caches`
from the relevant `train_*_command` after
`add_local_parallelism_args(...)`.

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
capacity. Inputs: total vCPUs, total RAM in GB, the post-cap
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
bounded by `minibatch_size`, not total dataset rows. Bigger data
scales fit-time SHM bytes (handled by `estimate_fit_dataloader_shm_gb`)
and per-epoch random-negative regeneration cost (handled by
`random_negative_pool_epochs > 1` if you need it), but DL child count
stays the same.

### `--num-jobs` (deprecated default of 0)

Today the recipe explicitly sets `--num-jobs $((GPUS * MAX_WORKERS_PER_GPU))`
in the shell. `mhcflurry.local_parallelism.auto_num_jobs(num_gpus,
max_workers_per_gpu)` is the in-Python equivalent for callers that
want to derive it after `auto_max_workers_per_gpu` has resolved.

### Cross-model coverage

| Model | `--max-workers-per-gpu auto` | `--dataloader-num-workers auto` | Notes |
|---|---|---|---|
| Pan-allele affinity | ✓ | ✓ | Default in release recipe |
| Allele-specific affinity | ✓ | ✓ | Same `Class1NeuralNetwork` codebase; auto already wired |
| Processing | ✓ | (no-op for now) | `Class1ProcessingNeuralNetwork` does not yet expose `dataloader_num_workers`; flag is accepted via shared `add_local_parallelism_args` so argv stays uniform across train_*_command, but `apply_dataloader_num_workers_to_work_items` won't change processing behavior until that hyperparameter is added. |
| Presentation | n/a | n/a | Single-process today |

When processing's fit() grows the same prefetch hyperparameter, the
orchestrator hookup is one line: call
`apply_dataloader_num_workers_to_work_items` from
`train_processing_models_command.run()` after
`resolve_local_parallelism_args(args)`. The auto resolver itself is
already model-agnostic.

### Env overrides

| Env var | Default | Effect |
|---|---|---|
| `MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB` | 16.0 | Per-worker VRAM upper bound for the MWPG resolver |
| `MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP` | 4 | SM-scheduler ceiling for MWPG |
| `MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB` | (auto-detect) | Pin free VRAM (CSV per GPU); for tests / hidden-`nvidia-smi` launchers |
| `MHCFLURRY_AUTO_DATALOADER_HARD_CAP` | 4 | DL child cap for `auto_dataloader_num_workers` |
| `MHCFLURRY_PER_FIT_SHM_FOOTPRINT_GB` | (live estimate) | Override per-fit SHM footprint constant for the capacity check |
| `MHCFLURRY_FIT_DATALOADER_SHM` | unset | Force-pin fit DataLoader SHM on/off (`1`/`0`) regardless of preflight |

## Env knobs vs CLI flags

A persistent question: when does a setting belong on `argparse` vs
the environment?

**Rule of thumb:**

- **CLI flag** when the orchestrator owns it and propagates it.
  (Examples: `--max-workers-per-gpu`, `--random-negative-shared-pool-dir`,
  `--num-jobs`.)
- **Env var** when the consumer is inside `fit()` or another
  worker-private code path, and the orchestrator is just a relay.
  (Examples: `MHCFLURRY_TORCH_COMPILE`,
  `MHCFLURRY_FIT_DATALOADER_SHM`, `TORCHINDUCTOR_COMPILE_THREADS`.)

The orchestrator may **hoist** env vars: read its own args, compute a
sensible default, and `os.environ.setdefault(...)` so workers
inherit. `resolve_local_parallelism_args` calls
`hoist_torchinductor_compile_threads` for local runs — it sizes the
inductor compile pool against the resolved `--num-jobs` so N workers
don't each spawn `cpu_count()` compile threads.

## What is NOT the orchestrator's job

- Compiling models (`torch.compile`). Compilation is per-network and
  happens inside `fit()`. The orchestrator only sizes the compile
  worker pool via env.
- Calling `predict()` on individual alleles. The orchestrator
  builds work items; workers iterate.
- Validating data shape consistency between work items beyond what
  the SHM helpers themselves require (uniform `pool_epochs`, uniform
  `peptide_encoding`).

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

## /dev/shm capacity and fit DataLoader SHM auto-fallback

The fit DataLoader SHM path allocates per-fit() backing tensors in
POSIX shm (`/dev/shm`). With small `/dev/shm` (the Docker default is
8 GB) and many fit() workers, the allocator runs out mid-fit and the
worker crashes with `OSError: [Errno 28] No space left on device`.

The orchestrator handles this in **four** layers, ordered from "best
recovery" to "conservative fallback":

1. **Pre-fork advisory** — `_preflight_shm_capacity` runs before
   workers fork. It computes
   `num_workers × per_fit_gb × 1.5` (50% margin), compares to
   `df /dev/shm`, and prints a one-line summary every run.
2. **Prefer torch's file-descriptor handle strategy** — when capacity
   is tight, the orchestrator first tries
   `torch.multiprocessing.set_sharing_strategy('file_descriptor')`.
   This changes how torch passes shared-memory handles and avoids some
   `torch_shm_manager` cleanup/permission failures. It does **not**
   bypass `/dev/shm` capacity: both torch CPU sharing strategies still
   allocate POSIX shared memory. Bumps `RLIMIT_NOFILE` automatically.
   Triggered by default; disable with `MHCFLURRY_TORCH_SHM_AUTO=0`.
   Also fires at `mhcflurry.class1_neural_network` import time so paths
   that bypass the orchestrator (tests, allele-specific train) inherit it.
3. **Auto-disable fit DataLoader SHM** — when the strategy switch
   fails (some platforms only ship `file_system`) OR the tmpfs remains
   too small, and the user hasn't force-pinned
   `MHCFLURRY_FIT_DATALOADER_SHM`, the orchestrator sets it to `"0"`
   so auto-backed workers use the numpy DataLoader path instead.
   Throughput hit is ~10–30% vs the SHM path; better than crashing.
4. **Per-fit defensive catch** — even if the pre-flight estimate
   underran the actual need, `fit()` catches resource-exhaustion
   errors from both torch tensor sharing and DataLoader setup
   (`ENOSPC`, `ENOMEM`, FD exhaustion, sandbox/permission failures, and
   torch RuntimeError variants)
   and falls back to the numpy / single-process path for that fit()
   call, with a loud warning.

For `fit_dataloader_backing="auto"`, force-on despite tight `/dev/shm`
with `MHCFLURRY_FIT_DATALOADER_SHM=1`. The pre-flight will warn but
respect the pin; workers will OOM or fall back inside `fit()` if the
estimate is correct.

For `fit_dataloader_backing="auto"`, force-off with
`MHCFLURRY_FIT_DATALOADER_SHM=0`. This skips the fit DataLoader SHM
path. An explicit component-model `fit_dataloader_backing`
hyperparameter wins over this diagnostic env var.

To resize `/dev/shm` on a Docker-based runtime:
* relaunch the container with `--shm-size=64g` (or more); remount
  in-place requires `CAP_SYS_ADMIN` which most container runtimes
  drop.
* On a bare host: `sudo mount -o remount,size=64g /dev/shm`.

The per-fit footprint estimate is computed live in
`_preflight_shm_capacity` from the actual work-item bundle:
`estimate_fit_dataloader_shm_gb` reads `peptide_encoding`
(alignment, max_length), `peptide_amino_acid_encoding_torch`
(int8 indices vs fp32 expansion), `random_negative_rate`, and the
loaded train-row count, then prints the breakdown alongside the
capacity headline. Override with `MHCFLURRY_PER_FIT_SHM_FOOTPRINT_GB`
when invoking the helper outside the orchestrator (tests, ad-hoc
scripts); it falls back to a 0.25 GB constant when no work items are
visible.

## Fit-time data flow

Affinity training has three distinct mechanisms that are deliberately
configured separately. Do not collapse these into one knob.

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
not DataLoader behavior. `dataloader_num_workers=0` does **not** turn
this off.

Disable it only with `peptide_amino_acid_encoding_torch=False`, which
restores the old path where numpy expands peptide strings into
`(N, L, V)` vectors before they are moved to the device.

### 2. Fit DataLoader process parallelism

Controlled by the model hyperparameter `dataloader_num_workers`.

This is pure process-count policy for the `fit()` inner batch loop:

| value | meaning |
|---|---|
| `0` | build batches in the training worker process |
| `>0` | spawn that many DataLoader child processes per training worker |

Release recipes set only this knob. On a local 8-GPU run with
`NUM_JOBS=16`, `dataloader_num_workers=1` means up to 16 extra
fit-local DataLoader children while epochs are active; `2` means up to
32. The thread-budget helper accounts for this when sizing
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS`.

### 3. Fit DataLoader backing storage

Controlled by the model hyperparameter
`fit_dataloader_backing="auto|numpy|shared_tensor"`.

This is data transport/storage for one affinity `fit()` call. It does
not alter model architecture, peptide vector lookup, loss, or training
semantics.

| value | behavior |
|---|---|
| `auto` | default; use `shared_tensor` when `dataloader_num_workers > 0`, otherwise `numpy` |
| `numpy` | keep `FitBacking` arrays as numpy arrays; no fit DataLoader SHM |
| `shared_tensor` | clone `FitBacking` arrays into torch shared-memory tensors and use tensor-backed batches |

`FitBacking` is a small internal bundle for one `fit()` call:
`x_peptide`, `x_allele`, `y_encoded`, optional weights, and
random-negative buffers. It exists so the fit loop can treat numpy and
shared-tensor backing uniformly.

The legacy diagnostic env var `MHCFLURRY_FIT_DATALOADER_SHM=0|1` still
works, but only for `fit_dataloader_backing="auto"`. An explicit
component-model hyperparameter wins. This keeps component models
self-describing: configs serialize the backing policy with each
`Class1NeuralNetwork`, while release orchestrators can stay simple and
only choose `dataloader_num_workers`.

### Component consistency

The `Class1NeuralNetwork` affinity component model owns all three
hyperparameters above, so allele-specific affinity models,
pan-allele affinity models, and affinity ensembles resolve the same
internal rules after config load. Missing keys in old component
configs are filled from defaults (`peptide_amino_acid_encoding_torch=True`,
`dataloader_num_workers=0`, `fit_dataloader_backing="auto"`).

Processing models do not use this affinity `fit()` DataLoader path and
therefore do not have `FitBacking` or `fit_dataloader_backing`.

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
  under `out/` returns. The orchestrator script
  (`scripts/training/pan_allele_release_affinity.sh`) places the
  encoding cache OUTSIDE `$MHCFLURRY_OUT` (default
  `$HOME/runplz-cache/encoding_cache/`) so the ~7 GB fixed-encoding mmap
  doesn't ride back. Override with `MHCFLURRY_ENCODING_CACHE_DIR`.
  Bonus: cache persists across runs on the same box, so the second
  run hits a warm cache.

## Pointers to code

- Run mmap cache + fit DataLoader SHM helpers: `mhcflurry/shared_memory.py`
- Encoding cache: `mhcflurry/encoding_cache.py` (generic
  `prebuild_encoding_caches`; pan-allele wrapper in
  `train_pan_allele_models_command._initialize_encoding_cache`)
- Pseudogene/null filter: `mhcflurry/common.py`
  (`filter_canonicalizable_alleles`)
- Worker pool sizing: `mhcflurry/local_parallelism.py`
  (`auto_max_workers_per_gpu`, `resolve_max_workers_per_gpu`)
- Compile-thread hoist:
  `mhcflurry/local_parallelism.hoist_torchinductor_compile_threads`
- /dev/shm capacity: `mhcflurry/local_parallelism.fit_shm_capacity_check`
  + `mhcflurry/train_pan_allele_models_command._preflight_shm_capacity`
- Cluster fork point:
  `mhcflurry/cluster_parallelism.cluster_results`

## Known asymmetries (deliberate)

These show up to readers as "why is this only done in pan-allele?"
The answer in each case is "the other components don't yet need it
and adding it would be feature work, not a fix":

- **Encoding-cache prebuild** is pan-allele-only because no other
  command sweeps enough architectures × folds × peptides for the
  prebuild to matter at current scale. The shared helper
  (`prebuild_encoding_caches`) makes adoption a one-line call when
  this changes.
- **Run mmap cache for the random-negative pool** is pan-allele-only
  because only pan-allele training does the per-epoch random-negative
  regeneration at the scale where mmap-share is meaningful.
  Allele-specific training does similar work but at smaller per-allele
  scale.
- **`torch.compile`** is off by default everywhere; opt-in via
  `MHCFLURRY_TORCH_COMPILE=1`. The thread-count hoist is in the
  pan-allele orchestrator only because that's where worker oversub
  was observed. If processing/allele-specific runs ever turn on
  `torch.compile` at scale, lift `_hoist_torchinductor_compile_threads`
  to a shared helper.

## Future tightening (not in 2.3.0)

- **Run mmap cache for allele-specific training.** Wire
  `--random-negative-shared-pool-dir` through
  `train_allele_specific_models_command`.
- **Encoding-cache prebuild for processing/allele-specific.** When
  datasets grow large enough to make the encoding pass dominate
  fit() time, call `prebuild_encoding_caches` from each command's
  `run()` before forking workers.
- **Presentation-training orchestration.** Today
  `train_presentation_models_command` runs single-process; mirror the
  pan-allele orchestration shape if presentation retraining becomes
  GPU-bound.
