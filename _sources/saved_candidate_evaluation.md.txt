# Evaluate saved model runs

`mhcflurry eval saved-candidate` evaluates a completed training run without
retraining. Write its results to a new directory and preserve the source models,
training provenance, evaluator commit, and frozen holdout identities.

```shell
mhcflurry eval saved-candidate \
  --candidate /persist/runs/COMPLETED_FULL_RUN \
  --exact-processing-run /persist/runs/COMPLETED_EIGHT_NETWORK_RUN \
  --public-root /persist/downloads/2.2.0 \
  --release-holdout-dir /persist/inputs/release_holdout \
  --source-commit EVALUATOR_COMMIT \
  --out /persist/runs/NEW_EVALUATION_RUN
```

`--exact-processing-run` supplies an optional historical-data processing replay;
omit it when evaluating only the full run. The replay is a separate condition,
not an exact-data retrain of the entire model.

The copied `candidate/presentation/models` bundle supplies the actual affinity
and processing components for the end-to-end comparison. Report the with-flanks
hybrid separately from an independently trained long-flank diagnostic ensemble.
No model is published by this evaluation command.

## Comparable cohorts

All models within a comparison must receive the same retained positive and
negative rows. Exclude the union of known training overlaps from every model,
preserve row identifiers, and record revised counts and cohort hashes. A model
must not choose its own evaluation negatives. Preserve the overlap caveat for
external predictors whose complete training records are unavailable.

Processing-specific evaluation uses sample/length/affinity-matched risk sets.
End-to-end presentation uses its separately frozen presentation cohort. Do not
average their AP values or compare absolute AP across different prevalences.
See {doc}`evaluation` for matching, score orientation, saved tables, and figures.

Report AP, PPV@N and AUROC per patient, their equal-patient means, and paired
patient-bootstrap intervals for differences. Full-model gains do not establish
that each component improved. Repeated selection on a held-out cohort also
limits the interpretation of its intervals.

## Remote execution

`scripts/training/launch_saved_candidate_evaluation_modal.py` runs the same
command on Modal through runplz. It contains no training stage and keeps outputs
on the configured persistent volume. Collect the completed output, inspect its
status and provenance, then use `mhcflurry train snapshot-experiment` to preserve
a hashed snapshot. A disconnected client is not evidence of job completion.
