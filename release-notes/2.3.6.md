# MHCflurry 2.3.6

This release clarifies model selection and aligns documentation, examples and
API docstrings with current behavior. Model weights and prediction calculations
are unchanged; the default weights remain 2.3.0.

- Put the recommended presentation bundle first in `mhcflurry downloads info`.
  It includes both affinity and processing components.
- Correct scan thresholds, model-path precedence, parallelism defaults,
  calibration options, return shapes and reproducibility descriptions.
- Refresh the tutorials and notebooks with N/C flanks and a comparison using
  the same peptides without flanks. Remove obsolete training instructions and
  development-specific prose while retaining historical provenance.
- Add Docker builds for Intel/AMD and ARM Linux, with bundled presentation
  weights and offline prediction checks. Successful stable-release builds
  publish matching Docker Hub tags and update `latest`. The CPU image runs
  as a non-root user; Jupyter uses its default access-token authentication.
- Document Docker publication checks and the separate Bioconda update process.
