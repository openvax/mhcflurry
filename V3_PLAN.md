# MHCflurry 3.0 Plan

## Goals

MHCflurry 3.0 retrains all models on updated data with broader allele
coverage, using mhcseqs for allele sequences instead of the legacy
alignment-derived pseudosequences. It builds on the training perf,
recipe, and pipeline modernization landed in the 2.3.0 candidate.

## Foundation: 2.3.0 baseline

The 2.3.0 release candidate (`codex/release-2.3.0`) ships training
infrastructure and recipe changes that 3.0 inherits as its starting
point. They aren't reopened in 3.0 unless explicitly noted.

**Already shipped on 2.3.0:**

* **Layered shared-memory training infra** (`mhcflurry/shared_memory.py`):
  * Layer 1 — per-run, file-mmap random-negative pool. Orchestrator
    pre-builds; workers consume read-only via `numpy.memmap`.
  * Layer 2 — per-fit() POSIX-shm torch tensors via
    `Tensor.share_memory_()`. Auto-enabled when
    `dataloader_num_workers > 0`. Closes the GPU starvation gap by
    letting workers prefetch CPU batch prep in parallel with GPU
    compute. Reduces dataset duplication 16× across worker processes.
* **Recipe tightening:** `max_epochs=500`, `min_delta=1e-7`,
  `validation_interval=5`, patience=20. Kills the noise-floor
  patience-reset tail (val_loss improving by 4×10⁻⁹/epoch was
  stretching tasks to 500+ epochs against a median of 67) without
  dropping late-escape trajectories.
* **`--max-workers-per-gpu auto`:** detects GPU memory + job count to
  pick the right concurrency. Replaces the hand-tuned `1000` default.
* **`--gpu-batched` calibration path** (issue #272), enabled by
  default in the release wrapper. Bit-identical on CUDA, ~10–20×
  faster than the per-allele `predict()` loop.
* **Calibrate filter** (`_filter_canonicalizable_alleles`) drops
  pseudogene / null / questionable annotations from
  `predictor.supported_alleles` before iteration. Makes calibration
  robust to mhcgnomes-rejected entries — directly relevant when
  expanding allele coverage in 3.0.
* **Unified train→select→calibrate→eval→plot pipeline** in
  `pan_allele_release_exact.sh` with `run_logged_step` per stage,
  `SKIP_EVAL` / `SKIP_PLOTS` env knobs, and CI lint over all bash
  scripts.
* **Cross-run comparison harness** (`scripts/training/compare_runs.py`)
  for side-by-side training time + per-allele eval metric deltas.
  This is the tool 3.0 will use to compare against 2.x baselines.

3.0 picks all of this up by branching from `codex/release-2.3.0` once
2.3.0 is tagged.

## Allele sequences: switch to mhcseqs

Replace the current pseudosequence pipeline (`downloads-generation/allele_sequences/`)
with [mhcseqs](https://github.com/pirl-unc/mhcseqs), which provides
alignment-free structural groove decomposition for 54,000+ MHC alleles
across 500+ species.

**Current state (2.2.0 / 2.3.0):** 20,252 alleles with 34–39 aa
pseudosequences derived from ClustalO multiple sequence alignment of
full-length proteins, then position selection to match a reference
pseudosequence set.

**With mhcseqs:** 26,000+ class I alleles with groove-ok status,
including ~2,000 new human classical HLA-A/B/C alleles, plus major
gains in fish (+657), bird (+518), murine (+422), and other species.

### Sequence representation — must decide before training

Three candidates, in order of model input size:

| Option | Length | Input tensor | Allele embedding |
|---|---|---|---|
| Pseudosequence (status quo) | 34 aa | (N, 34, 21) | small |
| Contact-position subset | ~50 aa | (N, 50, 21) | ~1.5× pseudoseq |
| Full groove (g1 + g2) | ~180 aa | (N, 180, 21) | **~5.3× pseudoseq** |

Implications of going to full groove:
* Per-fit dataset memory grows from ~600 MB → ~3 GB. Layer-2 SHM
  still works; the legacy 128 MB spawn-pickle ceiling becomes
  irrelevant. (The 2.3.0 SHM path was sized for this kind of
  growth.)
* Per-epoch wall time: probably 1.5–2× from the larger forward
  pass. Combined with 2.3.0 perf gains the absolute wall time
  may still beat the 2.2.0 baseline.
* Capacity / overfit risk scales with input dim. May want a
  contact-position prefilter to land between pseudoseq and full
  groove without paying full groove cost.

Recommendation: prototype a **contact-position subset** (~50 aa)
first; only escalate to full groove if a small ablation shows
meaningful PR-AUC gains on the data_evaluation benchmark.

### Tasks

* [ ] Add `mhcseqs` as a dependency.
* [ ] Replace pseudosequence generation with mhcseqs groove sequences.
* [ ] **Decide** on sequence representation (pseudoseq / contact /
  full groove) — recommend contact-position subset by default;
  ablate vs full groove on a small fold sweep.
* [ ] Handle the 223 alleles in the current file but not in mhcseqs
  (tracked in [pirl-unc/mhcseqs#23](https://github.com/pirl-unc/mhcseqs/issues/23) — mostly macaque genes,
  horse, rat).
* [ ] Include non-classical genes (HLA-E, F, G, MICA, MICB) where
  grooves are available.
* [ ] Validate that models trained on mhcseqs groove sequences
  reproduce or improve upon 2.2.0 / 2.3.0 accuracy on the
  data_evaluation hit/decoy benchmark (using
  `scripts/training/compare_runs.py`).

## Data curation

Curate updated training data for both binding affinity and mass-spec
identified ligands.

Tasks:
* [ ] Update IEDB affinity data pull (current curated data was last
  regenerated for the 2.0 release cycle).
* [ ] Update mass-spec ligand data pull.
* [ ] Integrate any new public MS datasets published since last
  curation.
* [ ] Review and update data filtering / QC pipeline.
* [ ] Check for allele name normalization issues with new mhcgnomes
  + mhcseqs (especially for non-classical genes — the
  `_filter_canonicalizable_alleles` helper in calibrate may need
  to expand to other call sites if pseudogenes leak elsewhere).

## Model training

Retrain pan-allele models on the updated data with the new allele
representation. Reuse the 2.3.0 training stack — no recipe
re-derivation needed unless the input-size change forces it.

Tasks:
* [ ] Decide whether to change network architecture or keep current
  (convolutional + pan-allele embedding). Default: keep current;
  the input-size change alone is the variable to test.
* [ ] Verify the 2.3.0 SHM dataloader handles the larger groove
  input without per-worker memory regressions (smoke test before
  the full sweep).
* [ ] Train pan-allele binding affinity models with the chosen
  groove representation.
* [ ] Train antigen processing models.
* [ ] Train presentation models.
* [ ] Run benchmark evaluation (see "Benchmark plan" below).
* [ ] Calibrate percentile ranks for the expanded allele set
  (`--gpu-batched` already on; expanded allele count is what
  this will stress-test).

## Benchmark plan

Quantitative success criteria, evaluated via
`scripts/training/compare_runs.py` and
`scripts/training/compare_new_vs_public.py`:

| Slice | Metric | Threshold |
|---|---|---|
| Existing alleles (overlap with 2.2.0 coverage) | Per-allele PR-AUC delta | mean ≥ 0, p25 ≥ −0.005 |
| Existing alleles | ROC-AUC delta | mean ≥ 0, p25 ≥ −0.005 |
| Existing alleles | PPV@N delta | mean ≥ 0 |
| New human alleles (HLA-A/B/C added by mhcseqs) | Per-allele PR-AUC | report only — no baseline to compare |
| Non-classical (HLA-E/F/G/MICA/B) | Per-allele PR-AUC if held-out data exists | report only |
| Wall time (full pipeline) | end-to-end | ≤ 2.3.0 baseline + 30% (groove headroom) |

Cross-tool comparison (NetMHCpan / NetMHCIIpan IEDB benchmark) is
optional — useful if results suggest meaningful improvement but not
gating.

## Compatibility / migration

* **Old saved models:** 2.2.0 and 2.3.0 model bundles will NOT load
  in 3.0 because the allele-representation input shape differs.
  Document explicitly in the 3.0 release notes.
* **Predictions:** the same `(peptide, allele)` pair will return a
  different value in 3.0 vs 2.x. Quantify mean / p95 absolute and
  rank delta on a representative held-out set in the release notes.
* **Deprecation:** 2.x bundles remain downloadable via
  `mhcflurry-downloads` for at least 12 months after 3.0 ships, with
  a deprecation warning in `mhcflurry-predict` when 2.x is loaded.
* **Allele aliases:** mhcgnomes alias coverage may shift between
  2.x and 3.0 (mhcgnomes >= 3.14 dependency). Audit a sample of
  long-lived user-facing allele names (`B*44:01`, `Cw*0201`, etc.)
  for round-trip stability.

## Fixes from 2.2.0

* **H2-D*q missing from model bundle `allele_sequences.csv`**
  (sequence exists in global file but was dropped during model
  generation). **Decision: include in the 2.3.0 patch via the
  data_curated regen, not 3.0.** Cheap, isolated, and stops being
  relevant once 3.0's mhcseqs pipeline replaces the source of
  allele_sequences.csv anyway.

## Out of scope for 3.0

* **Class II support.** Heterodimer α/β networks, peptide-length
  variability up to ~22 aa, different binding-groove geometry — a
  parallel-architecture project worth a dedicated track. Re-evaluate
  for 4.0; tracked in a follow-up issue.

## Dependency changes

| Dependency | 2.3.0 | 3.0 (proposed) |
|------------|-------|----------------|
| mhcseqs | — | new, ≥ TBD (post pirl-unc/mhcseqs#23 resolution) |
| mhcgnomes | ≥ 3.0.1 | ≥ 3.14.0 (mhcseqs requirement) |
| torch | ≥ 2.0.0 | ≥ 2.0.0 (no change expected) |
| numpy | ≥ 1.22.4 | ≥ 1.22.4 (no change expected) |
| pandas | ≥ 2.0 | ≥ 2.0 (no change expected) |
| Python | ≥ 3.10 | ≥ 3.10 (no change expected) |

## Open questions (still need owner + decision date)

* **Sequence representation choice** (pseudoseq / contact subset /
  full groove) — central architectural decision; blocks all model
  training. Recommended default: contact subset.
* **Non-classical inclusion in default predictor** — coverage is
  available via mhcseqs but binding data is sparse. Default-include
  with a low-confidence flag, or default-exclude and ship as opt-in?

## Sequencing

```
2.3.0 candidate (codex/release-2.3.0)
    ├── PR to master  ← when ready, separate sign-off
    │
    └── tag v2.3.0
            │
            └── branch v3 from v2.3.0
                    ├── mhcseqs integration
                    ├── data refresh
                    ├── retrain
                    ├── benchmark
                    └── tag v3.0.0
```

3.0 cannot start full training until 2.3.0 is tagged (so the
foundation is locked).
