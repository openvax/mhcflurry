# Processing kernel-width experiment

## Specification (2026-09-07)

Train six widths (5, 7, 9, 11, 13, 15) in each of two unmixed families:
the legacy 5-aa-flank CNN and a peptide-only CNN with separate 5-outside/
5-inside N/C boundary branches. Four paired sample folds per condition give
48 networks. Keep 512 filters, ReLU, dropout 0.5, Glorot initialization,
Keras-compatible Adam, batch 512, patience 20 and seed 42 fixed. Restore the
best validation weights. This is not the final broad-grid ensemble selection.

Use one frozen, length/affinity-matched training table, excluding release
holdout samples. Do not reuse the earlier random-negative training table.
Retain candidate-pool scores, matching assignments, data/model/source hashes,
fold membership, all epoch losses, and per-fold validation predictions.

Known flanks are adjacent to the actual peptide, including short peptides;
there is no gap to a maximum-length peptide slot. Missing flank residues and
convolution context beyond the supplied five residues are encoded as X.
For the boundary family, external residues enter only the boundary branches;
the central CNN sees X outside the peptide. Missing context is not evidence
of a true protein terminus. No fabricated BOS/EOS token is introduced.

Implement this outer-padding choice as an opt-in serialized hyperparameter;
old configs retain zero padding and unchanged predictions. Verify every
requested width, short/long peptides, full/partial/missing flanks, convolution
alignment, and save/load compatibility before cloud training.

Rank widths within each family using paired sample-held-out predictions
(macro AUPRC and PPV@N, with AUROC/micro safeguards). Preserve all conditions;
do not label the release holdout an unbiased final test after using it to
choose an architecture. Report actual public weights as a reference, not a
matched-data control: these new fits also change negative policy and checkpoint
restoration relative to historical runs. Plot metrics against width, per-sample
deltas and epoch traces; keep prediction rows joinable to external predictors.

Run on one Modal A100 through runplz, using persistent volumes and a bounded
timeout. Keep an exact source archive and durable commands/logs. Never stop
unrelated Modal jobs. No release/publishing is part of this experiment.

## Maintained command

```bash
mhcflurry train processing-kernel-sweep \
  --out experiments/processing-kernels \
  --train-data matched/train_data.csv \
  --public-root /path/to/downloads/2.2.0 \
  --release-holdout-dir /path/to/release_holdout \
  --source-commit COMMIT --gpus 1 --num-jobs 1
```

The Modal launcher additionally generates the fresh matched table from the
cached hit annotations and frozen public affinity predictor. It preserves
scored candidate pools even if matching fails; failure never selects legacy
random negatives. Set `PROCESSING_KERNEL_RUN_ID`,
`PROCESSING_KERNEL_SOURCE_COMMIT`, and `PROCESSING_KERNEL_SOURCE_SHA256`, upload
the exact source tarball to `/inputs/RUN_ID/source.tar.gz` in the
`mhcflurry-230-final-weights` volume, then launch from that tarball's extraction:

```bash
runplz modal scripts/training/launch_processing_kernel_sweep_modal.py \
  --detach --outputs-dir /path/to/local/experiment
runplz status --outputs-dir /path/to/local/experiment
runplz collect --outputs-dir /path/to/local/experiment
```

The receipt identifies the run-specific remote output path; no whole-volume
download is needed. Inside it, `processing.shared` retains matching inputs and
outputs; `kernel_sweep` holds models, epoch-loss figures, member predictions,
sample/fold metrics, and width plots. Training validation uses one matched
negative per hit; release evaluation uses ten. Do not compare their AUPRC levels
directly. The four folds can overlap in held-out samples: summaries average
folds within sample before averaging samples, not independent-fold significance.

Padding issue: [#404](https://github.com/openvax/mhcflurry/issues/404).
The zero-padding behavior follows the [PyTorch Conv1d API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv1d.html).

## Preparation recovery specification (2026-09-07)

The initial run stopped after 27 samples: four KESKIN_A3303 8-mer hits had
no sampled negative inside the unchanged 0.25-log10-affinity caliper. Retain
that run and all its candidate scores. Recovery must:

1. Accept an explicit prior matching-artifacts directory, verifying the input
   and frozen affinity-reference hashes, seed and matching policy. Reconstruct
   successful old samples from their saved scores without GPU prediction.
   New runs save checksummed matched outputs and immutable per-round scored
   pools, with atomic completion markers. Resume rejects changed/corrupt inputs.
2. Expose incomplete matching as a typed error with all unresolved hits. Expand
   only their peptide-length pools, with deterministic per-sample/round/length
   seeds, excluding all observed and previously scored sequences. Score only
   additions. Bound expansion rounds and candidate counts; exhaustion remains
   an explicit error, never hit dropping or relaxed matching.
3. Sample numeric protein positions before creating peptide/flank strings.
   Preserve the existing candidate population, including terminal-window
   conventions, amino-acid validity, exclusion and no-replacement semantics.
   RNG draws change and are versioned; explicit historical sampling remains
   available for legacy replay. Construct flank strings only for retained rows.
4. Load the affinity ensemble once per worker. Save elapsed sampling, scoring,
   matching and artifact-I/O timings separately. Test serial/worker seed
   independence, cache corruption, interrupted runs, expansion and exact
   sampler population equivalence before recovery on Modal.

No processing architecture or matching acceptance threshold changes in this
repair. The same final matched table is frozen for every sweep condition.

Recovery uses the maintained data command with either `--resume` for the same
output or `--resume-matching-dir PRIOR/train_data.csv.matching` for a new output.
The default bound is eight additional rounds of up to 100,000 sampled positions
per unresolved length. Exhaustion remains a hard error. Each round records its
sample/length seed, candidate score hash, unresolved hits and sampling/scoring/
write timings. Successful samples get checksummed matched tables and completion
markers. Old pools without per-round hashes are explicitly marked as such on
import; their source hashes and observed-row identities are recorded and checked.

For the Modal launcher, set `PROCESSING_KERNEL_RESUME_MATCHING_DIR` to the prior
directory's path under `/out`. It is read-only; recovery writes a new run directory.
The previous failure used
`/out/runplz/48c85d3639284d25bd1720e3e1d6f515/processing.shared/train_data.csv.matching`.

`mhcflurry train benchmark-processing-sampler --out benchmark.json` reproduces a
synthetic CPU timing comparison without training models. The September 7 local
run (1,000 proteins of length 1,000; 25,000 eight-mer draws) measured 1.99 seconds
for reservoir sampling and 0.091 seconds for position sampling (21.9x). This is
a sampling microbenchmark, not a measured end-to-end preparation speedup.
