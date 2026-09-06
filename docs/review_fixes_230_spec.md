# 2.3.0 review fixes and remaining decision work

The release candidate must use the same held-out evaluation policy locally and
remotely, preserve checkpoint identity through persistence and architecture
subsetting, and support every accepted boundary-window configuration.

Implementation and regression checks:

- Resolve explicit download releases consistently for status and extraction,
  preserving custom download-directory overrides.
- Apply the release holdout and overlap audit to remote comparisons, and run the
  separate train-excluded affinity gate before reporting evaluation complete.
- Remove initializer-payload pipe preloading and preserve worker assignment on
  process replacement. Test large payloads and shutdown with bounded timeouts.
- Clear discarded checkpoint manifest references; copy retained sidecars when
  splitting architectures and omit full-ensemble percentile calibration.
- Pad peptide-side boundary context with X beyond each actual peptide boundary;
  verify ordinary windows and predictions remain identical.

Experiment decision work: inspect the live Modal run and archived evaluations
before allocating more compute. Reuse completed fits and cached predictions.
Compare the full candidate's processing families and presentation combinations
on matched rows, preserve predictions and epoch histories, and report primary
macro AUPRC/PPV@N with micro and sample/allele/locus safeguards. Run an additional
training experiment only to resolve an identified decision gap. The 15-aa run's
recipe mismatch and historical terminal-only affinity checkpoints must remain
explicit in provenance.
