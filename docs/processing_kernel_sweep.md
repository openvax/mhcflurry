# Processing kernel-width experiment

## Workflow

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

The outer-padding choice is an opt-in serialized hyperparameter;
old configs retain zero padding and unchanged predictions. Validation covers every
requested width, short/long peptides, full/partial/missing flanks, convolution
alignment, and save/load compatibility before training.

Rank widths within each family using paired sample-held-out predictions
(macro AUPRC and PPV@N, with AUROC/micro safeguards). Preserve all conditions;
do not label the release holdout an unbiased final test after using it to
choose an architecture. Report actual public weights as a reference, not a
matched-data control: these new fits also change negative policy and checkpoint
restoration relative to historical runs. Plot metrics against width, per-sample
deltas and epoch traces; keep prediction rows joinable to external predictors.

Run on one Modal A100 through runplz, using persistent volumes and a bounded
timeout. Keep an exact source archive and durable commands/logs. This experiment does
not publish weights.

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

## Resume preparation

Use `mhcflurry train processing-data --resume` for the same output or
`--resume-matching-dir PRIOR/train_data.csv.matching` for a new output. Resume
verifies input/reference hashes, seed and matching policy. Completed samples
are reconstructed from saved outputs; only unresolved peptide-length pools
expand. Matching keeps the original affinity caliper and never silently drops
hits. Exhausting the bounded expansion raises an error.

Each round records score hashes, deterministic seeds, unresolved hits and
sampling/scoring/write timings. Older pools without per-round hashes are marked
on import; their source hashes and observed-row identities are still checked.
For the Modal launcher, `PROCESSING_KERNEL_RESUME_MATCHING_DIR` names a read-only
prior directory under `/out`; the resumed experiment writes a new run directory.

`mhcflurry train benchmark-processing-sampler --out benchmark.json` measures the
numeric-position and reservoir samplers on a synthetic CPU workload. A sampling
microbenchmark does not establish an end-to-end preparation speedup.
