# Pan-allele OOM bisect — 2026-04-25

Reference notes on the OOM blocking the perf PR (#270 / Phase 0–4 + #272).
Workers grow to 60–76 GB host RSS each during pretrain; on a 16-worker
8×A100 box (944 GB total) this OOM-kills the run before any models save.

## Bisect runs (5 × ~12 min × $20 = ~$60 of remote time)

| Code state | t=8 min sum_worker | t=11 min | Result |
|---|---|---|---|
| HEAD `f0c86e77` (baseline attempt 7) | 932 GB sys | OOM @ ~7:30 | FAIL |
| HEAD + `MALLOC_TRIM_THRESHOLD_=131072` env | 528 GB sys | 1022 GB sys | FAIL |
| HEAD + `_try_malloc_trim()` per epoch | 629 GB sys | OOM | FAIL |
| HEAD + Codex `ensure_built()` fix | 859 GB sys | OOM | FAIL |
| HEAD + Codex + per-step `del` + per-epoch `gc.collect` + pool early-null | 875 GB sys | OOM | FAIL |
| `df0ff25d3` (HEAD−1) | 935 GB sys | OOM | FAIL |
| `2731ed510` (HEAD−2,3,4) | 921 GB sys | OOM | FAIL |
| `bff0a71af` (HEAD−5) | crashed at startup (`ImportError: canonicalize_encoding_name`) | — | NOT TESTABLE in isolation |

**Common trajectory across every failing run:**
- t=242s (workers spawn): 2.9 GB max worker
- t=242–484s: steady **~2.4 GB/min/worker growth** during pretrain. Matches per-step 2 MB pretrain chunk allocation × ~16 chunks/sec exactly.
- t=545s: jump to ~30 GB max worker (workers transition pretrain → fit())
- t=605s: jump to ~60 GB max worker (fit() first epochs)
- t=666s: OOM-killed at ~940 GB system used.

## What was ruled out

- **Not in any of the 5 fixup commits between `f5cf99b21` and HEAD.** `2731ed510` (which has only no-op changes vs `f5cf99b21`) shows identical OOM trajectory.
- **Not torch.compile recompile storm.** `MHCFLURRY_TORCH_COMPILE=0` did not help.
- **Not `max_tasks_per_worker` accumulation.** Workers OOM before finishing 1 task; recycle frequency irrelevant.
- **Not the `peptide_to_idx` dict in `EncodingCache._load()`** (Codex's first hypothesis). The fix is correct (saves ~250 MB orchestrator-side) but didn't move the needle on production OOM.
- **Not glibc heap fragmentation.** `MALLOC_TRIM_THRESHOLD_=131072` + `MALLOC_MMAP_THRESHOLD_=131072` + `MALLOC_ARENA_MAX=2` slowed growth ~30% but didn't stop it.
- **Not Python-level cyclic references.** Per-epoch `gc.collect()` did not help.
- **Not held references.** Explicit `del batch, inputs, y_tensor, weights_batch, loss` per step in the hot loop did not help.
- **Not `RandomNegativesPool._build_cycle` 2× transition peak.** Setting `_current_encoded = None` before allocating the new array did not help.

## What we know about the leak

Worker `/proc/<pid>/smaps_rollup` near OOM:
```
RssAnon:        76066652 kB    (76 GB anonymous heap)
RssFile:           75300 kB    (75 MB file-backed)
RssShmem:          19216 kB    (19 MB shared)
AnonHugePages:  34236416 kB    (35 GB transparent huge pages)
```

- **76 GB is anonymous heap**, not file mmap (so not the encoding cache).
- **35 GB is THP-backed.** Transparent Huge Pages allocate 2 MB pages; small allocations on a THP page can pin the whole 2 MB.
- The growth rate (2.4 GB/min/worker during pretrain) matches the per-step pretrain chunk allocation exactly (~2 MB × 16 chunks/sec = 32 MB/sec = 1.9 GB/min, observed 2.4).
- gc.collect() does not reclaim. malloc_trim does not reclaim. This points to **C-level allocator territory that Python `gc` can't see** — most likely PyTorch's CPU allocator (`c10::alloc::CPUAllocator`) caching staging buffers from `torch.from_numpy(...).to(device)` paths, or libtorch's pinned-memory pool retaining pages.

## What's in HEAD now (kept)

- **Codex's `EncodingCache.ensure_built()`** + orchestrator + `_get_or_build_pretrain_batch_cache` row-offset shortcut. Correct fix; saves ~250 MB orchestrator memory and avoids the full peptide-to-idx dict load. Just not the dominant leak source.
- **Codex's `RandomNegativesPool.write_shared_pool()` streaming refactor.** Defensive fix for the Phase 3 mmap pool primitive (not currently wired into production training).
- **7 regression tests** (locally passing, 105/105 in cache + random-negative + NN suites):
  - `test_ensure_built_does_not_load_peptides_txt`
  - `test_ensure_built_returns_entry_dir`
  - `test_ensure_built_after_get_or_build_returns_same_entry_dir`
  - `test_ensure_built_idempotent_on_cache_hit`
  - `test_pretrain_batch_cache_row_offsets_match_index_map_lookup`
  - `test_pretrain_batch_cache_assertion_fires_on_row_count_mismatch`
  - `test_initialize_encoding_cache_orchestrator_path_does_not_call_get_or_build`

## Recommended next steps (offline, not on the live $19.30/hr box)

1. **Reproduce on a Linux dev box with `ptrace` permissions** (any EC2 m6i.4xlarge or larger). The Brev verda container has `/proc/sys/kernel/yama/ptrace_scope` read-only, blocking py-spy / memray / heaptrack — that's why bisecting on the box is blind.
2. **Run a 1-worker, 1-architecture, 1-fold training under `memray`:**
   ```bash
   memray run -o /tmp/profile.bin python -m mhcflurry.train_pan_allele_models_command \
       --num-jobs 1 --gpus 1 --max-tasks-per-worker 1 \
       --hyperparameters single_arch.yaml \
       --num-folds 1 \
       ...
   memray flamegraph /tmp/profile.bin
   ```
   The pretrain hot-loop allocation source will be the dominant flame in the per-step body.
3. **Two strong candidates to investigate first:**
   - `_move_fit_batch_to_device` line 829: `torch.from_numpy(value)` shares numpy memory with a torch tensor; if the tensor isn't released before the next `torch.from_numpy` call, libtorch's allocator may keep referencing the numpy buffer.
   - Default `torch.utils.data._utils.collate.default_collate` (used by `_make_fit_dataloader` for fit()): allocates a new CPU tensor per batch via PyTorch's allocator; if that allocator caches, ~5 MB/batch × 5,000 batches per fit() epoch × 30 epochs = 750 MB cached per fit() (6 GB across 8 fits per worker).
4. **Once root cause is pinned**, the fix is likely:
   - Either swap default collate for `_numpy_batch_collate` (already exists for Darwin) so PyTorch never allocates per-batch CPU tensors, OR
   - Add explicit `torch.cuda.empty_cache()` and CPU-side allocator reset between epochs.

## Brev box state at handoff

- 0 mhcflurry processes alive
- 3 GB / 944 GB used (clean)
- **STILL RUNNING** — `brev stop` CLI silently no-ops; needs to be stopped via the dashboard at console.brev.dev. Cost meter ticking at $19.30/hr.

## What did NOT work locally on Mac/MPS

I tried to repro on Mac with a focused script (`/tmp/bisect_fit_repro.py`) at scales up to 100K rows, 200 alleles, 10 epochs, minibatch 2048, no pretrain. RSS stayed flat at ~1 GB. The regression is Linux + CUDA + 16-worker specific, likely driven by interaction between torch CUDA pinned memory, glibc malloc, and THP. Not reproducible on macOS unified memory + MPS.
