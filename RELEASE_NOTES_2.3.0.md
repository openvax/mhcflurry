# MHCflurry 2.3.0

Release-candidate notes for 2.3.0. Held in this file (vs upstream into a
single `CHANGELOG.md`) until after the validation training run completes
and any last revisions land; will move to `CHANGELOG.md` at tag time.

## Headline

Pan-allele training pipeline modernization. ``fit()`` now defaults to
**device-resident** tensors on CUDA — the inner training loop, random-
negative pool, and validation forward pass all stay on-GPU, closing the
GPU-starvation gap that the host-side batching path was working
around. The post-training pipeline (select → calibrate → eval → plot)
is unified into a single resumable script, and calibration runs on
device end-to-end (`PercentRankTransform.fit_batch_torch`,
`motif_summary.motif_summary_chunk_gpu`) for an additional per-worker speedup on
top of the legacy `--gpu-batched` allele batching. Recipe tightening
(``min_delta=1e-7``, ``max_epochs=500``) kills the patience-reset
noise tail. Auto-resolvers pick workers/dataloader/compile settings
from hardware so per-box tuning stops being manual.

The orchestrator-as-locus-of-control architecture is documented in
[docs/orchestrator.md](docs/orchestrator.md) — read that for the
"who owns what" picture across parallelism, tensor residency, and env
knobs.

No changes to the prediction interface. **Saved 2.2.x model bundles
load and predict identically — the changes are entirely in how new
models are trained.**

## Performance

- **~2–3× per-task training speedup** from device-resident affinity
  training tensors (closes 0–30% GPU utilization observed on the
  2026-04-25 8×A100 baseline run).
- **~10–20× calibration speedup** from `--gpu-batched`, larger work
  chunks, and the affinity release wrapper calibrating at 50 K peptides
  per length (the `--num-peptides-per-length` CLI default is unchanged at
  100 K).
- **30–40% fewer wasted training epochs** from the recipe changes
  (`min_delta=1e-7`, `max_epochs=500`) terminating noise-floor
  patience-reset trajectories.

## New public API

- `mhcflurry/class1_affinity_training_data.py` — device-resident affinity
  training row space. `AffinityDeviceTrainingData` keeps real examples and
  random negatives as torch tensors on the active device for one `fit()` call.
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
| `dataloader_num_workers` (job-env default) | 0 | 1 | Applies to streaming pretraining batches only. Affinity fine-tuning no longer uses a per-fit DataLoader; it batches from device-resident tensors. One streaming worker per fit is the release wrapper default; tune upward only when CPU headroom and measurements justify it. |
| `peptide_amino_acid_encoding_torch` | n/a | `true` | Renamed replacement for the legacy `peptide_amino_acid_encoding_gpu` key, which is still accepted as an alias. Fixed peptide vector expansion moves from a numpy lookup at encode time to a frozen torch embedding table in the network's forward pass. `peptides_to_network_input` now returns int8 amino-acid indices by default; CUDA/MPS/CPU widens to the configured fixed vector encoding (`BLOSUM62`, `one-hot`, `PMBEC`, `contact`, `physchem` explicit descriptors, `atchley` factors, or composites such as `BLOSUM62+physchem`). Encodings may use a `:minmax` suffix, e.g. `PMBEC:minmax+contact:minmax`, to scale non-X values to [-1, 1] while preserving X as zero. Eliminates the ~17 sec/epoch CPU bottleneck in random-negative regeneration with `random_negative_pool_epochs=1`. Forward parity vs numpy path verified by `test_peptide_amino_acid_encoding_torch_forward_parity`. |

`patience` stays at 20.

## CLI changes

- **Unified `mhcflurry` parent command.** Every tool is now reachable as
  `mhcflurry <subcommand>` (`mhcflurry predict`, `mhcflurry downloads fetch`,
  `mhcflurry class1-train-pan-allele-models`, …) under one `mhcflurry --help`
  surface. The historical `mhcflurry-<subcommand>` console scripts still work
  as compat shims (same entry points). Two tools are new and unified-only:
  `mhcflurry compare-models` and `mhcflurry plot-model-comparison`.
- **`mhcflurry-class1-train-pan-allele-models --max-workers-per-gpu`**
  default changed from `1000` (effectively unlimited per-GPU) to
  `auto`. Auto-detect picks `min(num_jobs/num_gpus,
  0.6×free_vram/per_worker_gb, hard_cap=4)` without importing torch or
  initializing CUDA in the parent process. `per_worker_gb` defaults to
  4 GB (the affinity-fit footprint).

  Cross-checks: 8 GPUs + 16 jobs → 2 (num-jobs-limited); 8 GPUs +
  32 jobs → 4 (hard cap, ample VRAM); CPU-only → 1.

  Pass `--max-workers-per-gpu N` to pin explicitly.
- **`mhcflurry-class1-train-pan-allele-models --dataloader-num-workers`**
  new flag, default `auto`. Orchestrator derives the per-fit-worker
  DataLoader prefetch child count from the box's vCPUs / RAM /
  resolved fit-worker plan via
  `auto_dataloader_num_workers`, capped at 4. The resolved value
  overrides any `dataloader_num_workers` set in component-model
  hyperparameters at planning time, so saved configs reflect the
  actual choice. On 8×A100-80GB Verda (176v / 16 fit / 400 G) this
  resolves to 4 — the 2026-04-26 production benchmark — and steps
  down on tighter boxes (3 on 8×L40S, 1 on tight cluster nodes, 0 on
  RAM-starved or CPU-oversubscribed configs). The release recipe
  passes `DATALOADER_NUM_WORKERS=auto` by default; pin a literal int
  only when re-benchmarking.

  The flag is added via shared `add_local_parallelism_args` so every
  `train_*_command` accepts it. Affinity (pan-allele, allele-specific)
  applies it via `apply_dataloader_num_workers_to_work_items`.
  Processing accepts the flag for argv uniformity but is a no-op
  until `Class1ProcessingNeuralNetwork` grows the same prefetch
  hyperparameter; presentation runs single-process and ignores it.
- **`mhcflurry-calibrate-percentile-ranks`** wrapper-default now
  passes `--gpu-batched` and uses larger chunk sizes. Bit-identical
  on CUDA per the existing flag's behavior (issue #272).

## Behavioral changes worth knowing

### Training and calibration are reproducible by default (`--random-seed`)

Every CLI command that involves randomness — `mhcflurry-class1-train-pan-allele-models`,
`-train-allele-specific-models`, `-train-processing-models`,
`-select-allele-specific-models`, and `mhcflurry-calibrate-percentile-ranks` —
now takes a single `--random-seed` that controls **all** of its randomness:
fold/held-out assignment, weight initialization, example/batch shuffles,
random-negative sampling, random peptide universes, and genotype sampling.
The master seed is logged and, for the two-phase pan-allele/processing
pipelines, persisted into `training_init_info.pkl` so it survives an
`--only-initialize` / `--continue-incomplete` split.

**The default is `42`, not entropy** — so a run reproduces bit-for-bit out of
the box (same data, folds, replicates, hyperparameters → identical models).
This is a change from 2.2.x, where each fit drew independent OS entropy and
runs were not reproducible. Pass `--random-seed N` for a different, still
reproducible run. Ensemble members and per-fit work stay decorrelated (each
derives a distinct sub-seed from the master), so seeding does not reduce
diversity. The neural-network `fit()` / `fit_streaming_batches()` and
`Class1AffinityPredictor.fit_allele_specific_predictors()` APIs gained a
matching `seed=` keyword (defaults to `None` = the prior stochastic behavior
for direct API callers).

**Reproducibility caveats.** "Bit-for-bit" is exact on CPU and for the default
(Linear/RMSprop) affinity/processing architecture. Two scope conditions are
worth knowing:

- **Fixed effective minibatch size.** `fit()` may shrink the minibatch to fit
  available VRAM, and that shrink depends on free GPU memory and how many
  workers share the card — so the *same* seed on a busier or smaller GPU can
  produce a different model. A warning is logged whenever the shrink fires
  under an explicit seed, and `fit_info["effective_minibatch_size"]` records
  the value actually used. Pin the minibatch (or run on matching hardware) for
  cross-machine bit-for-bit reproduction.
- **CUDA kernel determinism.** Seeding covers the RNGs, but mhcflurry does not
  force `torch.use_deterministic_algorithms(True)`, and opting into
  `MHCFLURRY_MATMUL_PRECISION` enables `cudnn.benchmark` autotuning. The
  default MLP triggers no cuDNN kernels so it stays deterministic;
  convolutional `locally_connected_layers` variants are not guaranteed
  bit-identical run-to-run on CUDA.

`mhcflurry-class1-train-presentation-models` also accepts `--random-seed` for
uniformity (and logs the resolved value), though it has no stochastic step
today (the logistic-regression fit is deterministic and the parallel feature
path is pure inference).

Because the framework moved from TF/Keras to a Torch-resident loop, 2.3.0 does
not reproduce *2.2.x* outputs at an equal seed even on CPU: the per-epoch
training shuffle moved from NumPy to `torch.randperm`, and scan/presentation
`result="best"` ties now break deterministically by peptide (a stable
secondary sort key), so the specific tied peptide reported can differ from
2.2.x. These changes are intentional; only exact-tie outputs and cross-version
seed-equality are affected.

### `--held-out-fraction-seed` default is now `None` (allele-specific)

In `mhcflurry-class1-train-allele-specific-models`, the
`--held-out-fraction-seed` default changed from `0` to `None`. With no flag,
the held-out split is now derived from `--random-seed` (so the whole run
reproduces from one value) instead of the implicit `seed=0` split 2.2.0 used.
The no-flag held-out partition therefore differs from 2.2.0; pass
`--held-out-fraction-seed 0` to recover the previous split exactly.

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
    filter_canonicalizable_alleles,
)
predictor = Class1AffinityPredictor.load(models_dir)
all_alleles = predictor.supported_alleles
kept = filter_canonicalizable_alleles(all_alleles)
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

### Affinity fit is device-resident

Affinity `fit()` no longer routes minibatches through a per-fit
DataLoader. `AffinityDeviceTrainingData` owns the row space for one
fit call as torch tensors on the active backend, and the training loop
forms batches by index-selecting from those resident tensors. Random
negatives are refilled into the top slice of that row space each epoch.

## New tools

| Tool | Purpose |
|---|---|
| `mhcflurry compare-models` | Compare two ensembles (run-vs-run or run-vs-public) across affinity, presentation, and training-stats components. Markdown to stdout, CSVs to `--out`. Each component runs only when both sides have the matching artifact. |
| `mhcflurry plot-model-comparison` | Render ROC/PR/scatter/delta plots from a `compare-models` output directory. |
| `scripts/training/plot_loss_curves.py` | Per-model train + val loss curves from manifest (no weight files needed). Three PNGs + summary CSV. |

When to use which:
- **`compare-models --b public`** — a single run vs the published 2.2.0
  baseline (`--b` defaults to `public`). The eval stage of
  `pan_allele_release_affinity.sh` runs this by default.
- **`compare-models --a run1 --b run2`** — any two runs against each other.
  Use when comparing recipe variants, hyperparameter sweeps, or 2.3.0
  candidates against each other.
- **`plot_loss_curves.py`** — diagnostic. Doesn't need a baseline.

Dev-workstation helper: `scripts/dev/relocate_run_outputs.sh` moves
`brev_runs/` and `results/` outside the repo (with symlinks) so runplz's
rsync_up doesn't ship 15+ GB of stale prior-run artifacts to the box on every
launch. Run with `--apply` once per workstation.

## Pipeline orchestration

`scripts/training/pan_allele_release_affinity.sh` is now end-to-end:

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
> Will include `mhcflurry compare-models` output comparing the 2.3.0
> candidate vs the 2026-04-25 baseline run:
>
> - End-to-end wall time delta.
> - Per-task training time distribution shift.
> - Per-allele eval metric deltas (mean + p25) on the data_evaluation
>   benchmark.
>
> Acceptance: existing-allele PR-AUC / ROC-AUC mean delta ≥ 0,
> p25 ≥ −0.005. Not shipped to master until this passes.

## Dependencies

No required dependency version changes vs 2.2.x. PyTorch 2.0+ is already
required and is used for device-resident training and optional
`torch.compile`.

## Migration notes

- **Models trained with 2.3.0** will produce different weights from
  2.2.x even on identical seeds. Predictions on the same `(peptide,
  allele)` pair will differ — quantified in
  [validation results](#validation-results).
  - Two contributing factors beyond the obvious framework switch:
    1. `RandomNegativesPool` with `random_negative_pool_epochs > 1`
       generates one batch of random negatives and slices it across N
       epochs, rather than re-sampling fresh negatives every epoch as
       2.2.x did. Within a pool cycle consecutive epochs see distinct
       slices of the same pool; a new pool is drawn at each
       `epoch // pool_epochs` boundary. Set `random_negative_pool_epochs=1`
       to recover the pre-2.3.0 "fresh negatives every epoch" semantics
       (at the ~17 s/epoch encode cost).
    2. The 1-batch-per-architecture warmup primes torch.compile's
       on-disk cache with one synthetic forward+backward; the
       compiled-graph cache it writes does not affect weights, but
       running it does advance the global RNG before training proper
       starts. Pin a per-arch seed if you need bit-equivalence across
       runs.
    3. Device-resident random-negative sampling
       (`encode_random_negatives_on_device`) draws negative peptides as
       amino-acid indices via `torch.multinomial` rather than the host
       numpy `random_peptides` stream. Because this is a different RNG
       stream than 2.2.x used, even at an identical `--random-seed` the
       actual random-negative *peptides* differ (not just their row
       layout) — an additional contributor, beyond the framework switch
       and the `random_negative_pool_epochs` slicing above, to why 2.3.0
       models differ from 2.2.x.
- **Training ingestion now canonicalizes allele names**, so retraining on
  data that contained aliased / retired / alternative spellings can change
  which rows are included and therefore the resulting weights. Previously the
  training commands exact-string-matched the `allele` column and assumed it was
  pre-normalized: non-canonical rows were silently dropped (pan-allele, no
  matching pseudosequence key) or fragmented into separate models
  (allele-specific). 2.3.0 maps each name to its canonical key no-alias-first —
  an allele keeps its own pseudosequence when it has one, otherwise its alias
  target — matching how prediction already resolves names. If your training
  CSVs were already fully normalized this is a no-op; otherwise expect more
  rows retained and previously-fragmented alleles merged. (Prediction and
  calibration behavior is unchanged.)
- **Saved 2.2.x model bundles still work unchanged** in 2.3.0 for
  prediction; no migration needed for downstream users running
  inference on existing bundles.
- **`Class1PresentationPredictor.save()` keyword `write_metdata` renamed to
  `write_metadata`** (the prior spelling was a typo). The misspelled form would
  have raised `TypeError` for in-tree callers, so this is a no-op for code that
  used the correct spelling; any external caller passing `write_metdata=` must
  update to `write_metadata=`.
- **Deprecated: the dense-vector amino-acid encoding path.** Peptides and
  processing-model sequences are now always index-encoded (`(N, L)` int8) and
  embedded on device. The `peptide_amino_acid_encoding_torch=False` /
  `amino_acid_encoding_torch=False` hyperparameters (and the
  `peptide_amino_acid_encoding_gpu` alias) no longer select a dense `(N, L, V)`
  path — they are accepted but coerced to index encoding with a one-time
  deprecation warning, so existing configs still load and predict identically.
  `EncodableSequences.variable_length_to_fixed_length_vector_encoding` and the
  network's defensive dense-input branch are retained only for tests and are
  marked for removal (grep `DEPRECATED (scheduled for removal)`). The shared
  vector-encoding table machinery stays — it backs the index embedding and the
  allele encoder.
- The pan-allele release training pipeline is the
  primary thing that's changed. Allele-specific and processing
  training paths inherit shared backend selection and worker sizing,
  but their wrapper scripts are unaffected.
