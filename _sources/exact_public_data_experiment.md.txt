# Replaying historical processing data

## Controlled comparison

Does the frozen eight-network processing ensemble (four large-ReLU legacy
5-aa networks plus four large-ReLU 5x5 boundary networks), paired with the
**actual public affinity weights**, improve presentation without changing
training examples? This is the primary contrast, chosen before this replay.
Legacy-only and boundary-only ensembles are diagnostic contrasts, not a new
architecture search. Public/public with a refitted combiner separates combiner
implementation effects from changes to processing.

The 2.1.5 download manifest points at the same June 2020 model archives as the
historical public `2.2.0` download label. Do not use the October 2023 curated data
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

## Interpretation

This workflow isolates processing architecture and combiner changes while
keeping public affinity weights and original processing/presentation training
tables fixed. It is distinct from the final 2.3.0 model, which uses the 2023
training snapshot. Affinity performance in this replay is unchanged by
construction; it does not validate new affinity training.

Compare full presentation AP, PPV@N and AUROC on the same held-out rows and use
paired patient intervals. Original model-selection folds and an archived
training table do not prove disjointness from every comparator. Audit all
inventoried training data separately and retain external-overlap caveats.

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
versions and GPU telemetry. Collect only the experiment subtree; create
the destination parent before `modal volume get` of a directory.
