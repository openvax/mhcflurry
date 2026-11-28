# Orchestrator architecture

*Contributor internals guide — end users running MHCflurry do not need to read this.*

## Ownership boundary

MHCflurry splits multi-model training into two roles:

- **Orchestrator (parent process):** parses CLI arguments, loads shared inputs once into `WORKER_CONTEXT`, resolves automatic concurrency and thread budgets before starting workers, and collects returned predictors.
- **Worker (child or cluster process):** receives `WORKER_CONTEXT` once, builds its own device-resident training buffers, runs `fit()`, and returns the trained predictor. Workers never mutate shared state or set process-wide policy for peers.

## Training data residency

- **Affinity `fit()`** is device-resident. `AffinityDeviceTrainingData` (`mhcflurry/class1_affinity_training_data.py`) holds `[random negatives | real training rows]` as integer-index tensors on the active PyTorch device for the lifetime of a `fit()` call. Batches are indexed directly on-device, and the model's embedding layer expands amino-acid indices to BLOSUM62 / PMBEC / physicochemical vectors inside `forward()`. No DataLoader workers are used here.
- **Random negatives** (`mhcflurry/random_negative_peptides.py`) are sampled independently per worker on its active device and refilled in-place each epoch so real rows are never recopied.
- **Pretrain streaming (`fit_streaming_batches`)** is the only path that uses a PyTorch `DataLoader` (`dataloader_num_workers`, where `0` runs in-process).
- **Processing and presentation models** share the same worker-pool orchestration, without the affinity `AffinityDeviceTrainingData` row layout.

## Parallelism backends

Both backends execute the same worker entry points (`train_model()`, etc.); only the orchestrator branches on `--cluster-parallelism`:

| Backend | Implementation | How `WORKER_CONTEXT` is delivered |
| :--- | :--- | :--- |
| **Local** | `mhcflurry/parallelism/` (`multiprocessing.Pool`) | Inherited copy-on-write under `fork`; passed once to the worker initializer under `spawn` (capped by deep-sized RAM headroom). |
| **Cluster** | `mhcflurry/cluster_parallelism.py` (`bsub` / `sbatch` / `sh`) | Serialized to shared storage and deserialized once per cluster job. |

`resolve_local_parallelism_args` is the single pre-fork normalization point: it detects free GPU VRAM via `nvidia-smi` (without initializing CUDA in the parent), honors container cgroup RAM limits, caps `--num-jobs`, and hoists `TORCHINDUCTOR_COMPILE_THREADS` before spawning workers.

## Auto-tuned hardware knobs

Pass `auto` (the default in release recipes) to size concurrency from detected hardware; pass an explicit integer to pin concurrency (automatic probes never override explicit CLI values):

- **`--max-workers-per-gpu auto`** (`auto_max_workers_per_gpu`): packs estimated per-worker VRAM working sets into free VRAM after a 10% (min 1 GiB) shared reserve, bounded by host RAM and vCPUs. Before affinity or processing training starts, a one-worker resource probe runs a bounded epoch per distinct architecture to measure peak CUDA + RSS usage and tighten the automatic plan if needed (also warming the local `torch.compile` cache when enabled). Automatic CPU/MPS runs execute serially in the parent process.
- **`--num-jobs auto`** (`auto_num_jobs`): resolves to `gpus × max_workers_per_gpu`.
- **`--dataloader-num-workers auto`** (`auto_dataloader_num_workers`): applies only to pretrain streaming (`fit_streaming_batches`), sizing DataLoader children from per-fit vCPUs and RAM (capped at `4` by default, `0` when serial or tight on CPU/RAM).

See {doc}`auto_sizing_audit` for the formal planner invariants and hardware test matrix.

### Expert environment overrides

| Environment variable | Default | Effect |
| :--- | :--- | :--- |
| `MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB` | `4.0` (fallback) | Override fallback per-worker VRAM estimate (GiB). |
| `MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP` | unset | Ceiling on auto-selected workers per GPU. |
| `MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB` | auto-detect | Comma-separated free VRAM override per GPU (for tests or hidden `nvidia-smi`). |
| `MHCFLURRY_AUTO_DATALOADER_HARD_CAP` | `4` | Ceiling on auto-selected pretrain DataLoader children per fit worker. |
| `MHCFLURRY_FAIL_ON_TRAINING_BATCH_SHRINK` | `0` (`1` in release) | Fail instead of shrinking an oversized affinity minibatch. |
| `MHCFLURRY_TORCH_COMPILE` / `MHCFLURRY_TORCH_COMPILE_LOSS` | `0` / `1` | Opt into `torch.compile` (and optionally disable compiled losses). |
| `TORCHINDUCTOR_COMPILE_THREADS` | `auto` | Hoisted by the orchestrator to avoid `N × cpu_count()` compile threads. |
| `MHCFLURRY_CLUSTER_WORKERS_PER_NODE` | unset | Fair-share CPU divisor when a cluster scheduler packs multiple jobs per node. |

## Key modules

- `mhcflurry/parallelism/` — local worker pool, VRAM/RAM/CPU sizing, and compile-thread hoisting
- `mhcflurry/workload_planning.py` — analytic and probe-based memory envelopes
- `mhcflurry/cluster_parallelism.py` — cluster submission and result collection
- `mhcflurry/class1_affinity_training_data.py` — `AffinityDeviceTrainingData` device-resident buffers
- `mhcflurry/random_negative_peptides.py` — worker-local random-negative pooling
- `mhcflurry/common.py` — `filter_canonicalizable_alleles` shared across selection and calibration
