# Exact-public-data architecture gate for 2.3.0

## Question and controls

Does the frozen eight-network processing ensemble (four large-ReLU legacy
5-aa networks plus four large-ReLU 5x5 boundary networks), paired with the
**actual public affinity weights**, improve presentation without changing
training examples? This is the primary contrast, chosen before this replay.
Legacy-only and boundary-only ensembles are diagnostic contrasts, not a new
architecture search. Public/public with a refitted combiner separates combiner
implementation effects from changes to processing.

The 2.1.5 download manifest points at the same June 2020 model archives as the
current public `2.2.0` download label. Do not use the October 2023 curated data
bundle as a proxy for the data inside those weights.

- Processing: use the archived 399,392-row `train_data.csv.bz2` byte-for-byte,
  including its original four fold assignments. No regenerated decoys.
- Presentation: use the original 75,378-row training table, unchanged.
- Affinity and no-flank processing: use actual public ensembles, unchanged.
- Two processing architectures, four folds each, seed 42; batch 512,
  Glorot/Keras Adam and all remaining settings from the frozen boundary panel.
  The panel's terminal-weight policy is unchanged in both architectures;
  epoch losses and best/stop information remain saved.
- Original folds are model-selection holdouts. The fit's internal stopping
  split is seeded, but cannot be claimed identical to the historical random
  stopping split, which was not archived as row assignments.

Earlier affinity screens used 811,134 rows versus 713,069 in public weights.
Earlier processing screens used 399,392 rows but a different row multiset.
Those results are not exact-data architecture controls. Raw affinity row
differences also include representation changes; they are not a count of
biologically new measurements.

## Decision and budget

Use the frozen release holdout for every comparison. Primary endpoints are
presentation-with-flanks macro AUPRC and PPV@N. Before authorizing a new
extra-data campaign, require positive paired sample-bootstrap intervals for
both, inspect micro changes (flag losses greater than 0.002 absolute), and
sample/length/allele-locus breakdowns (flag sample losses greater than 0.02).
These thresholds are decision rules for this replay, not retrospectively
claimed as the original screen's preregistration. Multiple exploratory tests
and only ten evaluation samples limit certainty. No automatic extra-data
launch is allowed just because a point estimate improves.

First run: eight processing networks on one Modal A100-40GB, one training
worker, eight-hour invocation limit, persistent volume. This resolves the
processing/presentation question economically; it does **not** establish a
new exact-data affinity-training result. Affinity performance in the primary
contrast is unchanged by construction. The already-running changed-data full
candidate is retained separately, not relabeled as this experiment.

## Reproduction and artifacts

`mhcflurry train exact-public-processing --help` exposes the maintained driver.
It fails closed on public input hash mismatch and evaluation sample overlap,
preserves original folds with `--reuse-folds`, resumes incomplete training,
and saves per-condition validation predictions and loss plots. Presentation
component scores are cached with predictor/input hashes and row identity;
the same combiner is fitted from those scores for all conditions.

Outputs include commands/events, input copies and hashes, trained manifests
with complete epoch histories, OOF predictions, fitted combiner coefficients,
component-score caches, held-out row predictions, sample/length metrics, and
comparison PDFs. Snapshot the output with `mhcflurry train snapshot-experiment`
and append the final comparison PDFs with `mhcflurry eval collate-figures`.
Results and gates belong in the experiment directory, not solely in chat.

The runplz entry point is
`scripts/training/launch_exact_public_processing_modal.py`. Set
`EXACT_PUBLIC_RUN_ID`, `EXACT_PUBLIC_SOURCE_COMMIT`, and
`EXACT_PUBLIC_SOURCE_SHA256`; upload the corresponding clean source archive to
`mhcflurry-230-final-weights:/inputs/<run-id>/source.tar.gz`; then run
`runplz modal scripts/training/launch_exact_public_processing_modal.py` from
its clean extraction. The launcher records exact source identity, package
versions and GPU telemetry. A documented adapter keeps the Modal job detached
pending upstream runplz issue 165. Collect only the experiment subtree; create
the destination parent before `modal volume get` of a directory.
