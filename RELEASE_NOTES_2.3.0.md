# MHCflurry 2.3.0

Release-candidate notes for 2.3.0. Held in this file (vs upstream into a
single `CHANGELOG.md`) until after the validation training run completes
and any last revisions land; will move to `CHANGELOG.md` at tag time.

## Headline

Pan-allele training pipeline modernization: layered shared-memory
infrastructure closes the GPU-starvation gap, recipe tightening kills
the patience-reset noise tail, and the post-training pipeline
(select → calibrate → eval → plot) is unified into a single
resumable script. Calibration is ~10–20× faster.

No changes to the prediction interface. **Saved 2.2.x model bundles
load and predict identically — the changes are entirely in how new
models are trained.**

## Performance

- **~2–3× per-task training speedup** from Layer-2 shared-memory
  DataLoader workers (closes 0–30% GPU utilization observed on the
  2026-04-25 8×A100 baseline run).
- **~10–20× calibration speedup** from `--gpu-batched`, larger work
  chunks, and 50 K-peptides-per-length default (was 100 K).
- **30–40% fewer wasted training epochs** from the recipe changes
  (`min_delta=1e-7`, `max_epochs=500`) terminating noise-floor
  patience-reset trajectories.

## New public API

- `mhcflurry/shared_memory.py` — layered SHM primitives. Two layers
  with different OS mechanisms but a uniform "build once, share with
  many readers" pattern:
  - **Layer 1** — per-run, file-mmap (orchestrator builds, workers
    read). `setup_shared_random_negative_pools(...)`,
    `lookup_pool_dir(...)`.
  - **Layer 2** — per-fit() POSIX-shm via `Tensor.share_memory_()`.
    `share_tensor`, `share_like`, `update_shared`, `array_nbytes`,
    `numpy_batch_collate`, `tensor_batch_collate`, `FitBacking`.
- `mhcflurry/training_benchmark.py` — micro-benchmarks for the
  training inner loop (used for sweep_workers analysis).

## Recipe changes (`scripts/training/release_exact/generate_hyperparameters.py`)

These produce **different model weights** from the published 2.2.x
release. Quantitative deltas vs the 2.2.0 ensemble on the
data_evaluation hit/decoy benchmark are reported in
[validation results](#validation-results) below once the 2.3.0
validation run completes.

| Hyperparameter | 2.2.x | 2.3.0 | Why |
|---|---|---|---|
| `max_epochs` | 5000 | 500 | Median observed was 67; max 174. The 5000 ceiling was theatrical and let pathological patience-reset tasks burn unbounded compute. 500 leaves comfortable headroom. |
| `min_delta` | 0.0 | 1e-7 | With `min_delta=0`, a 1e-9 RMSprop noise-floor improvement resets the 20-epoch patience counter, stretching some tasks to 174+ epochs at val_loss ~0.28 with no real signal. 1e-7 is two orders of magnitude above the observed noise rate; preserves real escape trajectories (typically ≥1e-3/epoch). |
| `validation_interval` | 1 (always validated) | 5 | Skip the validation forward pass on 4 of 5 epochs; saves ~150 ms/epoch + a GPU sync barrier. The final epoch and any patience-trigger epoch are always measured (the saved model reflects an up-to-date val_loss). |
| `dataloader_num_workers` (job-env default) | 0 | 4 | Was 0 because spawn workers OOM'd on the 600 MB per-fit dataset. Layer-2 SHM eliminated that cost; auto-enables when `dataloader_num_workers > 0`. |

`patience` stays at 20.

## CLI changes

- **`mhcflurry-class1-train-pan-allele-models --max-workers-per-gpu`**
  default changed from `1000` (effectively unlimited per-GPU) to
  `auto`. Auto-detect picks `min(num_jobs/num_gpus,
  0.6×free_vram/8GB, hard_cap=4)`.

  Cross-checks: 8×A100-80GB + 16 jobs → 2 (matches old production
  setting); 8×A100-40GB + 16 jobs → 2; 1×A100-80GB + 8 jobs → 4;
  CPU-only → 1.

  `MAX_WORKERS_PER_GPU=N` env var still pins explicitly.
- **`mhcflurry-calibrate-percentile-ranks`** wrapper-default now
  passes `--gpu-batched` and uses larger chunk sizes. Bit-identical
  on CUDA per the existing flag's behavior (issue #272).

## Behavioral changes worth knowing

### Calibrate silently filters unsupported alleles

`mhcflurry-calibrate-percentile-ranks` now drops alleles from
`predictor.supported_alleles` that fail `mhcgnomes.parse` annotation
checks (pseudogenes, null, questionable) before iterating, with a
logged sample. Previously these would crash the calibration partway
through with `ValueError("Unsupported annotation on MHC allele: ...")`.

User-visible asymmetry: the percent-rank table now lacks rows for
those alleles. Runtime `predict()` on a dropped allele still raises
the same `ValueError` it always did. To list the dropped alleles for
a specific predictor:

```python
from mhcflurry import Class1AffinityPredictor
from mhcflurry.calibrate_percentile_ranks_command import (
    _filter_canonicalizable_alleles,
)
predictor = Class1AffinityPredictor.load(models_dir)
all_alleles = predictor.supported_alleles
kept = _filter_canonicalizable_alleles(all_alleles)
dropped = sorted(set(all_alleles) - set(kept))
print(f"{len(dropped)} dropped:", dropped[:10])
```

### `validation_interval > 1` and the saved val_loss

When `validation_interval > 1`, `fit_info["val_loss"]` is still one
entry per epoch (the on-interval values get carried forward into the
intervening rows for plotting compatibility). Three triggers force a
real measurement:

1. on the cadence (`epoch % interval == 0`),
2. on the final epoch of the loop,
3. when patience would trigger this epoch (so the saved val_loss
   reflects the actual stop state, not a stale carried-forward value).

### Layered SHM is auto-on with workers

When `hyperparameters["dataloader_num_workers"] > 0`, fit() materializes
the dataset's static backing arrays (x_peptide, x_allele, y_encoded,
sample_weights, random_negative_x_allele) and a per-epoch
random-negative buffer as CPU torch tensors via
`Tensor.share_memory_()`. The DataLoader's spawn workers receive
storage handles instead of byte copies. Force-on / force-off via
`MHCFLURRY_FIT_DATALOADER_SHM=1/0`. Default detection is correct for
all common configurations.

## New tools

| Tool | Purpose |
|---|---|
| `scripts/training/compare_runs.py` | Compare two training runs side-by-side. Reads each run's `models.unselected.combined/manifest.csv` to compute per-task wall-time / epoch-count / final-loss aggregates; compares `eval_comparison/` outputs for per-allele ROC-AUC / PR-AUC / PPV@N deltas. Markdown to stdout, CSV to `--out`. |
| `scripts/training/compare_new_vs_public.py` | Compare a freshly-trained ensemble vs the published 2.2.0 release on the data_evaluation hit/decoy benchmark. Per-allele metrics, used as the in-pipeline eval stage. |
| `scripts/training/plot_loss_curves.py` | Per-model train + val loss curves from manifest (no weight files needed). Three PNGs + summary CSV. |

When to use which:
- **`compare_new_vs_public.py`** — single run vs the published 2.2.0
  baseline. The eval stage of `pan_allele_release_exact.sh` runs this
  by default.
- **`compare_runs.py`** — any two runs against each other. Use when
  comparing recipe variants, hyperparameter sweeps, or 2.3.0
  candidates against each other.
- **`plot_loss_curves.py`** — diagnostic. Doesn't need a baseline.

## Pipeline orchestration

`scripts/training/pan_allele_release_exact.sh` is now end-to-end:

```
fetch_pretrain_data   → fetch_data_curated   → train_combined
  → select_combined   → calibrate_combined   → fetch_eval_data
  → eval_compare_new_vs_public                → plot_loss_curves
```

Each stage runs through `run_logged_step` with its own log file under
`$MHCFLURRY_OUT/`. Both new stages (eval + plot) skip cleanly via
`SKIP_EVAL=1` / `SKIP_PLOTS=1` env knobs for incremental reruns. CI
now runs `bash -n` over every `scripts/**/*.sh` to catch syntax
regressions before a multi-hour training run discovers them.

## Validation results

> **TODO: filled in after the 2.3.0 validation training run completes.**
>
> Will include `compare_runs.py` output comparing the 2.3.0 candidate
> vs the 2026-04-25 baseline run:
>
> - End-to-end wall time delta.
> - Per-task training time distribution shift.
> - Per-allele eval metric deltas (mean + p25) on the data_evaluation
>   benchmark.
>
> Acceptance: existing-allele PR-AUC / ROC-AUC mean delta ≥ 0,
> p25 ≥ −0.005. Not shipped to master until this passes.

## Dependencies

No required dependency version changes vs 2.2.x. PyTorch 2.0+ (already
required) is needed for the `torch.compile` + `Tensor.share_memory_()`
paths.

## Known limitations / follow-ups

- `persistent_workers=True` is not supported with the in-place
  `update_shared` random-negative-buffer refill (would race). Comment
  in `_make_fit_dataloader` documents this; tracked as a future-PR
  item.
- The shared mmap random-negative pool (Layer 1) requires
  `random_negative_pool_epochs >= max_epochs`; validated upfront in
  `setup_shared_random_negative_pools`. To use the shared pool path,
  set both `--random-negative-shared-pool-dir` and
  `random_negative_pool_epochs >= max_epochs` in the recipe.

## Migration notes

- **Models trained with 2.3.0** will produce different weights from
  2.2.x even on identical seeds. Predictions on the same `(peptide,
  allele)` pair will differ — quantified in
  [validation results](#validation-results).
- **Saved 2.2.x model bundles still work unchanged** in 2.3.0 for
  prediction; no migration needed for downstream users running
  inference on existing bundles.
- The pan-allele release training pipeline is the
  primary thing that's changed. Allele-specific and processing
  training paths inherit the perf primitives (encoding cache,
  SHM dataloader) but their wrapper scripts are unaffected.
