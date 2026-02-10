# TODO

## In Progress

- [ ] Optional follow-up: run broader/full test suite before merge
  - Status: targeted parity and regression suites pass.
  - Next: run additional suites as needed for merge confidence.

## DONE

- [x] Localize parity mismatch component.
  - Affinity and processing-without-flanks parity confirmed.
  - Processing-with-flanks identified as source of presentation divergence.

- [x] Create development tracking docs.
  - Added `NOTES.md` and `TODO.md`.

- [x] Fix with-flanks processing parity vs TF in `mhcflurry/class1_processing_neural_network.py`.
  - Changed N/C flank-average pooling to match TF masked `reduce_mean` semantics.
  - Verified by comparing intermediate feature vectors and outputs against TF.

- [x] Validate end-to-end parity after fix.
  - Targeted TF-vs-PyTorch comparisons now match to near float precision.
  - `test/test_class1_presentation_predictor.py::test_downloaded_predictor` now passes.
  - Parity-focused tests pass:
    - `test/test_master_compat_predictions.py`
    - `test/test_released_master_predictions.py`
    - `test/test_pytorch_regressions.py`

- [x] Add regression coverage for with-flanks average behavior.
  - Added `test_processing_flank_averages_use_tf_masked_mean_semantics`.

- [x] Add Modal training script for larger jobs.
  - Added `scripts/modal_train_mhcflurry.py` with:
    - GPU worker function
    - shared artifacts volume
    - command-template based parallel launch

- [x] Speed up TF-vs-PyTorch random comparison harness.
  - Added curated default allele panel (`iedb_plus_animals`) to reduce per-run
    affinity-group fragmentation.
  - Removed redundant direct processing passes in `predict-backend`; processing
    outputs now reused from presentation predictions.
  - Verified end-to-end run succeeds with expected parity metrics and faster runtime.

- [x] Add cross-product parity analysis + plots for fixed peptide panel across alleles.
  - Added `scripts/cross_allele_parity_analysis.py`.
  - Ran `1000` random peptides (uniform lengths `7-15`) across curated panel (`35` alleles).
  - Generated summaries and plots in `/tmp/mhcflurry-cross-allele-1000-panel`.

- [x] Extend cross-product analysis to random flanks + strict sanity requirements.
  - Added unique random flank generation per peptide.
  - Added pre-run duplicate checks for peptide/flank fields.
  - Added post-run presentation score checks:
    - >=1% rows with score >0.2
    - at least one row with score >0.9
  - Ran and validated in `/tmp/mhcflurry-cross-allele-1000-randflanks`.

- [x] Build a high-score TF fixture for presentation regression tests.
  - Added `scripts/extract_high_presentation_fixture.py`.
  - Extracted contexts with any presentation score > 0.9 and retained all
    alleles per context (including low-score alleles).
  - Added fixture files under `test/data/` and new test
    `test/test_released_presentation_highscore_rows.py`.
