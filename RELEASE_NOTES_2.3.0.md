# MHCflurry 2.3.0

Release notes for 2.3.0. Held in this file until the final tag so the measured
model-validation record stays distinct from the package changelog.

## rc20

- Restore framework-level 2.1.x training semantics that numeric
  hyperparameters alone did not capture: post-activation affinity LSUV,
  Keras-equation RMSprop/Adam, Keras validation-split rounding, processing
  Glorot/zero-bias initialization, and Keras BatchNorm moving variance.
  Optimizer equation, LSUV boundary, and processing initializer are serialized
  switches with paired frozen-holdout ablations before the full retrain.
- Require runplz 3.16.0 for remote release provenance and use its updated
  staging/bootstrap behavior.
- Restore the published 2.1.x/2.2.x scientific recipe after auditing the first
  full candidate. Retain affinity minibatch 1024 because its held-out sweep was
  better, but restore affinity early stopping/calibration, processing batch and
  fold holdout, and presentation decoy/sampling/flank/solver/calibration
  settings. Pin fresh affinity negatives each epoch, eager execution, and full
  float32 matmul precision for release provenance. Processing layers now pin
  Keras-compatible Glorot initialization instead of inheriting PyTorch's
  Kaiming default. See
  [the recipe audit](docs/release_training_recipe.md).
- Freeze a provenance-recorded final holdout before training. Affinity uses all
  103 monoallelic benchmark samples plus the final multiallelic holdout and
  removes every current-training pMHC found in those benchmark rows, including
  decoys. Processing and presentation final metrics use the 10 multiallelic
  samples from PMID 31154438; presentation excludes those complete samples
  from training. Persist, checksum, and validate the generated manifests with
  the release run.
- Pin Ruff 0.16.0 in CI, commit an explicit correctness-focused lint rule set,
  and run `./lint.sh` on every pull request. Mutable defaults, environment
  defaults, and loop-closure findings uncovered during lint triage are fixed.
- Interpret any processing-metric change alongside affinity-driven decoy
  difficulty. The working hypothesis is that a stronger affinity predictor
  selects higher-quality, harder processing negatives; final acceptance still
  reports processing metrics on the strictly held-out evaluation set.

## rc19

- Reorganize the user documentation around progressive disclosure: explain
  prediction concepts and common workflows first, then introduce score
  interpretation, schemas, performance overrides, and exhaustive references.

## rc18

- Make README and installation-guide prerelease instructions independent of
  the current release-candidate number so they cannot become stale on later
  releases.

## rc17

- Move first-party GitHub Actions to their Node.js 24 releases, removing
  deprecated Node.js 20 runtime warnings from CI, documentation, and package
  publishing workflows.

## rc16

- Credit all MHCflurry contributors in generated documentation metadata and
  the site footer.

## rc15

- Make the unified `mhcflurry` command primary throughout user documentation.
- Clarify allele, genotype, and sample semantics in CLI help and tutorials;
  `predict-scan` now accepts semicolon-delimited genotypes like `predict`.
- Move detailed training configuration out of the beginner tutorial into a
  dedicated training guide.
- Remove the unused `--proteome-peptides` argument from model-selection decoy
  generation.

## Summary

MHCflurry 2.3 modernizes pan-allele training and release evaluation:

- affinity training keeps its working tensors on the active device;
- one resumable workflow covers selection, calibration, evaluation, and plots;
- automatic resource planning replaces machine-specific worker overrides; and
- prediction-affecting release settings remain anchored to the published
  2.1.x/2.2.x recipe unless held-out evidence supports a change.

The sections below describe measured performance, behavioral changes, and
compatibility. Maintainer-level parallelism and tensor-residency details are in
[docs/orchestrator.md](docs/orchestrator.md).

No changes to the prediction interface. **Saved 2.2.x model bundles remain
compatible, and predictions for valid class-I alleles are unchanged.** Loading
now omits incomplete pseudosequences that mhcgnomes identifies as pseudogene,
class-II, or non-MHC records; these rows were never valid prediction targets.

## Performance

- **~2–3× per-task training speedup** from device-resident affinity
  training tensors (closes 0–30% GPU utilization observed on the
  2026-04-25 8×A100 baseline run).
- **Calibration throughput improvements** from `--gpu-batched` and larger work
  chunks while retaining the published 100 K affinity calibration peptides per
  length.

## New public API

- `mhcflurry/class1_affinity_training_data.py` — device-resident affinity
  training row space. `AffinityDeviceTrainingData` keeps real examples and
  random negatives as torch tensors on the active device for one `fit()` call.
- `mhcflurry/training_benchmark.py` — micro-benchmarks for the
  training inner loop (used for sweep_workers analysis).

## Release recipe

The release recipe matches the published 2.1.x/2.2.x scientific settings
except for affinity minibatch 1024, which improved held-out affinity metrics in
the controlled batch sweep. Affinity `max_epochs=5000`, `min_delta=0`,
validation every epoch, and fresh negatives every epoch remain unchanged.
Processing uses minibatch 512 and 10 held-out samples. Presentation uses 2
proteome decoys per hit, a 0.1 sample fraction, `short_flanks` (5 aa per side),
L-BFGS, and 10 K calibration peptides per length. The full audit distinguishes
these settings from execution-only implementation changes in
[docs/release_training_recipe.md](docs/release_training_recipe.md).

## CLI changes

- **Unified `mhcflurry` parent command.** Every tool is now reachable as
  `mhcflurry <subcommand>` (`mhcflurry predict`, `mhcflurry downloads fetch`,
  `mhcflurry class1-train-pan-allele-models`, …) under one `mhcflurry --help`
  surface. The historical `mhcflurry-<subcommand>` console scripts still work
  as compat shims (same entry points). Two tools are new and unified-only:
  `mhcflurry compare-models` and `mhcflurry plot-model-comparison`.
- **`mhcflurry-class1-train-pan-allele-models --max-workers-per-gpu`**
  default changed from `1000` (effectively unlimited per-GPU) to
  `auto`. The planner reads free memory on every visible GPU without
  initializing CUDA in the parent, sizes to the least-free card, subtracts the
  shared allocator/context reserve, and divides the remainder by a
  workload-specific complete-worker estimate. Affinity training estimates
  include the configured encoding width, network topology, training rows,
  random-negative residency, and minibatch. Host RAM, CPUs, available work
  items, and an explicit job count can further reduce concurrency. There is no
  default fixed worker cap; an expert hard-cap environment override remains.

  Before a parallel affinity or processing pool starts, one isolated worker
  runs a bounded full-residency train-and-validation pass for every
  resource-distinct architecture. Process-level CUDA and host peaks may only
  tighten the analytic plan. Each production worker then receives a fixed
  launch-time device-memory entitlement, so startup order cannot inflate or
  double-discount its batch budget. Prediction and validation batches use an
  architecture-aware activation estimate plus a real CUDA forward probe;
  elastic OOM halving remains a last-resort recovery path. CPU-only resolves
  to one worker per accelerator slot, and explicit sizing flags remain
  authoritative.

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

## Behavioral changes

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
(Linear/RMSprop) affinity/processing architecture. Two limits apply:

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
from mhcflurry.cli.calibrate_percentile_ranks_command import (
    filter_canonicalizable_alleles,
)
predictor = Class1AffinityPredictor.load(models_dir)
all_alleles = predictor.supported_alleles
kept = filter_canonicalizable_alleles(all_alleles)
dropped = sorted(set(all_alleles) - set(kept))
print(f"{len(dropped)} dropped:", dropped[:10])
```

### MHC classification comes from mhcgnomes 3.33

MHCflurry now requires mhcgnomes 3.33 and uses its ontology-backed
`is_pseudogene` and gene-family classification directly. This removes the
local HLA pseudogene table and name-based TAP/`PS` regular expression, while
also recognizing pseudogene loci in macaque and orangutan species. One exact
compatibility entry remains for the malformed `Caja-PS*02:01` key shipped in
the public 2.2.0 pseudosequence artifact; upstream ontology/alias support is
tracked in [pirl-unc/mhcgnomes#88](https://github.com/pirl-unc/mhcgnomes/issues/88).

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

`matplotlib` is now an installed package dependency, so the documented
evaluation and paper-figure commands work in a clean MHCflurry installation;
remote launchers no longer install it as an undeclared workaround.
Percent-change paper figures now use finite axis limits for degenerate
all-equal metrics, restoring compatibility with Matplotlib 3.11.
Paper and diagnostic output paths are validated before comparison, cleanup, or
rendering, preventing a custom summary path or paper directory from deleting or
overwriting command-owned PDFs.

When to use which:
- **`compare-models --b public`** — a single run vs the published 2.2.0
  baseline (`--b` defaults to `public`). The eval stage of
  `pan_allele_release_affinity.sh` runs this by default.
- **`compare-models --a run1 --b run2`** — any two runs against each other.
  Use when comparing recipe variants, hyperparameter sweeps, or 2.3.0
  candidates against each other.
- **`plot_loss_curves.py`** — diagnostic. Doesn't need a baseline.

Remote release launches require runplz 3.15.3. Its Git-aware staging excludes
ignored output directories such as `brev_runs/` and `results/`, so the former
workstation-specific relocation and symlink workaround is no longer needed.
The wrapper records the runplz module path and, for an editable checkout, its
clean Git commit before launching.

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

## Rejected candidate results (not release results)

The following measurements are retained as diagnostic evidence only. These
weights are rejected because the run predated the recipe audit and changed
processing/presentation training data and several optimizer settings at once.
They must not be packaged or published as 2.3.0. Replacement results will be
filled in after a clean retrain from the corrected recipe.

The rejected weights were trained from commit
`121ed667b770d27b395fb92da1eca5f5c3f0e339` in workflow
`20260823T043004Z-26013` on 4×A100 40 GB. Final evaluation used commit
`ad8e0f1d22265928a62f4fd6c7ea3ffff1658940` in workflow
`20260825T054920Z-43847`. Training completed all 140 affinity candidates and
all 512 candidates for each processing mode; the selected release contains 8
affinity models, 8 models for each processing mode, and 2 presentation models.

The frozen holdout contains 103 monoallelic samples (132,049 unique pMHCs) and
10 multiallelic samples from PMID 31154438. The final artifacts have zero
affinity, processing, or presentation training overlap with their respective
holdout manifests. Evaluation scored 15,027,952 monoallelic rows with 135,388
hits and 2,054,263 multiallelic rows with 18,507 hits.

### Fair affinity comparison

The published 2.2.0 ensemble contains 135,451 rows from the frozen affinity
benchmark in its training data, including all 135,388 hits. Its direct score
against the cleanly held-out 2.3.0 candidate is therefore descriptive, not a
valid release gate. The decision-grade comparison uses the 2.2.0
`models.no_additional_ms` ensemble and excludes the union of both sides'
training pMHCs. That union removes only 2 rows and 1 hit; the 2.3.0 candidate
itself overlaps zero rows.

| Average | Metric | 2.3.0 | train-excluded 2.2.0 | Absolute delta | Relative delta |
|---|---|---:|---:|---:|---:|
| Macro | AUROC | 0.967238 | 0.958759 | +0.008480 | +0.88% |
| Macro | AUPRC | 0.606905 | 0.569811 | +0.037094 | +6.51% |
| Macro | PPV@N | 0.610859 | 0.578031 | +0.032829 | +5.68% |
| Micro | AUROC | 0.974173 | 0.962068 | +0.012105 | +1.26% |
| Micro | AUPRC | 0.554248 | 0.456353 | +0.097895 | +21.45% |
| Micro | PPV@N | 0.588949 | 0.516815 | +0.072134 | +13.96% |

Across the 95 reported alleles, the p25 deltas are +0.002005 AUROC,
+0.001586 AUPRC, and 0.000000 PPV@N. The predeclared affinity acceptance gate
(mean delta non-negative and p25 at least -0.005) passes.

### Processing and presentation

Processing AUROC is essentially flat to slightly improved, while AUPRC and
PPV@N regress systematically:

| Processing mode | Macro AUROC | Micro AUROC | Macro AUPRC | Micro AUPRC | Macro PPV@N | Micro PPV@N |
|---|---:|---:|---:|---:|---:|---:|
| with flanks | +0.28% | +0.20% | -17.86% | -18.06% | -9.68% | -9.43% |
| no flank | +0.20% | +0.07% | -10.51% | -11.62% | -7.49% | -7.35% |
| short flanks | +0.06% | -0.01% | -12.93% | -13.75% | -8.05% | -7.32% |

End-to-end presentation regresses less than the processing-only ranking:

| Presentation mode | Macro AUROC | Micro AUROC | Macro AUPRC | Micro AUPRC | Macro PPV@N | Micro PPV@N |
|---|---:|---:|---:|---:|---:|---:|
| with flanks | -0.10% | -0.20% | -4.55% | -5.75% | -3.54% | -3.76% |
| without flanks | -0.29% | -0.38% | -4.92% | -5.76% | -3.01% | -3.93% |

The corrected presentation-percentile calculation agrees closely with raw
score AUROC (candidate micro AUROC 0.898903 vs 0.898975 with flanks, and
0.892751 vs 0.892767 without flanks) but has lower candidate micro AUPRC
(0.262658 vs 0.287274, and 0.244795 vs 0.265738). Raw presentation score
therefore remains the preferred ranking output.

One plausible interpretation, established before the final run, is that the
stronger affinity model selects higher-quality decoys and consequently makes
the processing subproblem harder. The holdout results are consistent with
that hypothesis—processing AUROC is retained while precision-focused metrics
fall—but do not prove causality. The regressions are not accepted as a release
tradeoff because this was not a controlled 2.1.x-compatible comparison.

All six processing inference passes and all four presentation passes completed
on 4×A100 without an OOM, batch retry, or calibration warning after the final
autosizing changes. The complete exact-source evaluation and training-loss
figure set is archived as one 65-page PDF for release review.

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
    2. A single-worker resource probe runs one bounded, full-residency training
       epoch with validation for each resource-distinct architecture before the
       production pool starts. It tightens automatic concurrency from measured
       process-level CUDA and host peaks; when compilation is enabled, the same
       pass primes torch.compile's on-disk cache. The probe runs in a separate
       process with a fixed probe seed and does not contribute weights or advance
       any production worker's RNG stream.
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
