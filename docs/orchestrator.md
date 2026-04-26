# Orchestrator architecture

> Internals doc — for contributors. End users running
> `mhcflurry-class1-train-pan-allele-models` don't need to read this.

## The one-line summary

The orchestrator owns the **shared resources** (parallel workers,
shared-memory layers, encoding caches, env tuning); workers consume
them. Workers never spawn other workers, never build shared state,
never set env knobs that affect other workers.

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
  (Layer-1, mmap)            │       look up encoding cache
                             │       (mmap hit, no rebuild)
prebuild random-neg pool     │
  (Layer-1, mmap)            │       look up random-neg pool
                             │       (mmap hit, no replan)
fork worker pool             ─┘
                                    fit() materializes
                                      Layer-2 SHM (per-fit)
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

| | Layer 1 | Layer 2 |
|---|---|---|
| **Lifetime** | per-run (persists to disk; survives across runs) | per-`fit()` call |
| **Mechanism** | `numpy.memmap` of files | `torch.Tensor.share_memory_()` (POSIX shm in `/dev/shm`) |
| **Built by** | orchestrator | the `fit()` call inside a worker |
| **Read by** | every training worker | DataLoader spawn workers inside one `fit()` |
| **Mutability** | read-only | mutable (random-neg buffer refilled per epoch) |
| **Holds** | encoded peptides, random-neg peptides | x_peptide, x_allele, y_encoded, sample_weights, random_negative_x_peptide buffer, random_negative_x_allele |

**Why two layers?** Layer 1's persist-across-runs property fits file
mmap; Layer 2's mutate-per-epoch property fits torch shm. The split is
intentional, but the API surface is uniform: each has a `setup_*`
factory and a small set of generic helpers — see `mhcflurry/shared_memory.py`.

**Why not "always Layer 2"?** Layer 2 doesn't survive process exit;
encoding the ~1 M training peptides takes ~30 sec, and we'd pay it
every run. Layer 1's mmap files live on disk, so a re-run hits cache.

**Why not "always Layer 1"?** Layer 1 is read-only; the per-epoch
random-negative buffer needs in-place updates. Layer 2 is the right
tool for that.

## Parallelism backends

Two backends, one CLI surface:

- **Local** (`local_parallelism.py`): `multiprocessing.Pool` of
  non-daemon workers. Workers can spawn DataLoader children. SHM
  layers both work — workers inherit GLOBAL_DATA via fork, mmap pool
  paths are local-process-visible.
- **Cluster** (`cluster_parallelism.py`): one job per work-item
  submitted via `bsub` / `sbatch` / `sh`. Workers serialize
  GLOBAL_DATA to NFS, deserialize on the worker side. Layer-1 SHM
  works ONLY when the pool dir is on a shared filesystem reachable
  from every worker node — orchestrator emits a loud warning when both
  flags are set so the user can verify.

Worker-side code (`train_model()`, etc.) is identical between
backends. Only the orchestrator branches on `args.cluster_parallelism`.

## Coverage matrix

The 2.3.0 modernization concentrates on the pan-allele affinity
training + percentile calibration paths. Other components inherit
parts of the stack (Layer-2 SHM is automatic for any `fit()` call;
the pseudogene filter is now shared) but their orchestrators don't
yet drive Layer-1 SHM or encoding-cache prefetch — those are
opt-in, not on by default.

| | pretrain | finetune | select | calibrate |
|---|---|---|---|---|
| **affinity (pan-allele)** | full stack ✓ | full stack ✓ | filter ✓; SHM/cache n/a (no fit) | filter ✓; SHM/cache n/a |
| **affinity (allele-specific)** | n/a | local pool only; cache+L1 are opt-in via `prebuild_encoding_caches` | filter ✓ | shares calibrate command |
| **processing** | n/a | local+cluster pool; cache+L1 are opt-in | local+cluster pool | n/a (allele-independent) |
| **presentation** | n/a | serial only (single-process; no orchestration story today) | n/a | filter ✓ (shares calibrate command) |

"Opt-in" cells aren't *broken* — they just inherit the worker-side
Layer-2 SHM (free with `dataloader_num_workers > 0`) and don't yet
have orchestrator-side Layer-1 prebuild. When their datasets grow
to where prebuild matters, call `prebuild_encoding_caches` from the
relevant `train_*_command` after `add_local_parallelism_args(...)`.

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
inherit. `_hoist_torchinductor_compile_threads` is the canonical
example — it sizes the inductor compile pool against `--num-jobs` so
N workers don't each spawn `cpu_count()` compile threads.

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

## /dev/shm capacity and Layer-2 SHM auto-fallback

Layer-2 SHM allocates per-fit() backing tensors in POSIX shm
(`/dev/shm`). With small `/dev/shm` (the Docker default is 8 GB) and
many fit() workers, the allocator runs out mid-fit and the worker
crashes with `OSError: [Errno 28] No space left on device`.

The orchestrator handles this in three layers:

1. **Pre-fork advisory** — `_preflight_shm_capacity` runs before
   workers fork. It computes
   `num_workers × per_fit_gb × 1.5` (50% margin), compares to
   `df /dev/shm`, and prints a one-line summary every run.
2. **Auto-fallback** — when capacity is tight AND the user hasn't
   force-pinned `MHCFLURRY_FIT_DATALOADER_SHM`, the orchestrator sets
   it to `"0"` so workers use the numpy DataLoader path instead.
   Throughput hit is ~10–30% vs the SHM path; better than crashing.
3. **Per-fit defensive catch** — even if the orchestrator estimate
   underran the actual need, `fit()` catches `OSError(ENOSPC)` and
   falls back to numpy mode for that one fit() call, with a loud
   warning.

To force-on (despite tight `/dev/shm`):
`MHCFLURRY_FIT_DATALOADER_SHM=1`. The pre-flight will warn but
respect the pin; workers will OOM if the estimate is correct.

To force-off:
`MHCFLURRY_FIT_DATALOADER_SHM=0`. Skips both Layer-2 SHM and the
capacity check.

To resize `/dev/shm` on a Docker-based runtime:
* relaunch the container with `--shm-size=64g` (or more) — this is
  the recommended fix; remount in-place requires `CAP_SYS_ADMIN`
  which most container runtimes drop.
* On a bare host: `sudo mount -o remount,size=64g /dev/shm`.

Per-fit footprint estimate is tuned via
`MHCFLURRY_PER_FIT_SHM_FOOTPRINT_GB` (default 4.0 GB, sized for
pan-allele MLP at standard data scale).

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
  (`scripts/training/pan_allele_release_exact.sh`) places the
  encoding cache OUTSIDE `$MHCFLURRY_OUT` (default
  `$HOME/runplz-cache/encoding_cache/`) so the ~7 GB BLOSUM62 mmap
  doesn't ride back. Override with `MHCFLURRY_ENCODING_CACHE_DIR`.
  Bonus: cache persists across runs on the same box, so the second
  run hits a warm cache.

## Pointers to code

- Layer 1 + Layer 2 helpers: `mhcflurry/shared_memory.py`
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
- **Layer-1 random-negative pool** is pan-allele-only because only
  pan-allele training does the per-epoch random-negative regeneration
  at the scale where mmap-share is meaningful. Allele-specific
  training does similar work but at smaller per-allele scale.
- **`torch.compile`** is off by default everywhere; opt-in via
  `MHCFLURRY_TORCH_COMPILE=1`. The thread-count hoist is in the
  pan-allele orchestrator only because that's where worker oversub
  was observed. If processing/allele-specific runs ever turn on
  `torch.compile` at scale, lift `_hoist_torchinductor_compile_threads`
  to a shared helper.

## Future tightening (not in 2.3.0)

- **Layer-1 SHM for allele-specific training.** Wire
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
