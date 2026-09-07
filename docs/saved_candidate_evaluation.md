# Evaluation-only completion of saved 2.3.0 candidates

## Spec and acceptance criteria

Evaluate the completed full train-and-select candidate before spending more
training budget. Do not retrain or modify any source run. Use a separate,
resumable evaluation directory bound to source-model fingerprints, training
provenance, evaluator commit, and frozen holdout. Score presentation first,
then all processing modes, then the train-excluded and descriptive affinity
comparisons. Preserve row-level predictions, per-sample/allele/length metrics,
commands, logs and figure PDFs. No comparison may change the heldout negatives
or silently omit an unavailable component.

The primary with-flank processing model is the actual eight-network hybrid
inside the presentation bundle, not the independently selected legacy model.
Also score the selected legacy ensemble separately. Keep the historical 15-aa
Glorot/Keras and terminal-only affinity provenance exceptions visible.

Optionally finish the already-trained exact-public-data eight-network replay
in a separate output subtree. Verify original data hashes, original fold/row
identity checks, completed training/selection markers and four networks per
family before assembly. Refit identical presentation combiners on the archived
public table. This is a small controlled replay, NOT a full exact-data
train-and-select experiment. Never relabel it as one or use this workflow to
authorize additional-data training automatically.

Commands:

```shell
mhcflurry eval saved-candidate \
  --candidate /persist/runs/COMPLETED_FULL_RUN \
  --exact-processing-run /persist/runs/COMPLETED_EIGHT_NETWORK_RUN \
  --public-root /persist/downloads/2.2.0 \
  --release-holdout-dir /persist/inputs/release_holdout \
  --source-commit EVALUATOR_COMMIT \
  --out /persist/runs/NEW_EVALUATION_RUN
```

The output `candidate/presentation/models` is a portable copy of the saved
candidate bundle. Processing evaluation views use that bundle's actual
components. Nothing is published or automatically accepted for release.
Joint macro AUPRC/PPV@N improvements need micro, sample, length, allele and
locus safeguards; repeated consultation of this holdout makes exploratory
selection intervals optimistic. Independent confirmation and the full
exact-data comparison remain distinct gates.

Run remotely from a clean source archive with the maintained
`launch_saved_candidate_evaluation_modal.py` runplz adapter. It allocates one
A100-40GB, requests 8 CPU cores/64 GiB and has an eight-hour timeout. It has no
training command. Every output stays on the persistent Modal volume even if
the client disconnects. Collect to an existing local parent directory, then
use `mhcflurry train snapshot-experiment` and `mhcflurry eval collate-figures`.

Assembly parent-directory failures are tracked in
[issue #399](https://github.com/openvax/mhcflurry/issues/399).
