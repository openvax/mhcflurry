# MHCflurry 2.3.7

This maintenance release corrects preparation reproducibility, comparison
metrics and partial processing-model serialization. Existing model weights and
their prediction calculations are unchanged; default weights remain 2.3.0.

- Processing source annotations use explicit seeded maximum-expression ties,
  canonical input order and stable output order. Release workflows pass the
  master seed and record input, generator and output hashes. Resuming preparation
  checks that record; historical unrecorded annotations must not be relabeled as
  a reproducible new run.
- Model comparisons calculate expected PPV@N for ties crossing the cutoff,
  independent of row order, and record the metric policy. Existing reports are
  unchanged; newly generated metrics can differ for tied scores.
- Saving selected processing weights into a fresh directory creates a loadable
  subset in the requested order. Incremental saves retain previously written
  members. Subset exports omit the full ensemble's percentile calibration;
  calibrate the subset separately before using percentiles.
