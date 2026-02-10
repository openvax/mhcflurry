# Notes

## 2026-02-10

- Goal: match PyTorch branch behavior to TensorFlow master for class I presentation prediction.
- Confirmed mismatch is isolated to `processing_predictor_with_flanks` path:
  - Affinity outputs match TF nearly exactly.
  - Processing without flanks matches TF nearly exactly.
  - Processing with flanks differs materially.
- Intermediate feature comparison (single processing model) shows:
  - `n_flank_cleaved`, `n_flank_internal_cleaved`, `c_flank_cleaved`, `c_flank_internal_cleaved` match TF.
  - Only `n_flank_avg_dense` and `c_flank_avg_dense` inputs differ.
- Root cause identified:
  - TF computes masked flank averages with `reduce_mean(..., axis=1)` over full sequence length.
  - Current PyTorch computes average over flank positions only.
  - This changes the two flank-average scalar features and can change top peptide ranking in presentation mode.
- Fix implemented:
  - Updated `Class1ProcessingModel` N/C flank-average pooling math to mirror TF exactly:
    - `mean((x + 1) * mask, axis=sequence_axis) - 1`
    - denominator is full sequence length.
- Validation after fix:
  - Single-model intermediate features now match TF to float noise.
  - With-flanks processing predictions now match TF to float noise.
  - End-to-end presentation predictions for test sequences now match TF best-peptide selection.
  - `test/test_class1_presentation_predictor.py::test_downloaded_predictor` passes.
  - Parity test subset passes:
    - `test/test_master_compat_predictions.py`
    - `test/test_released_master_predictions.py`
    - `test/test_pytorch_regressions.py`
- Regression coverage:
  - Added `test_processing_flank_averages_use_tf_masked_mean_semantics` in
    `test/test_pytorch_regressions.py`.
- Tooling add-on:
  - Added `scripts/modal_train_mhcflurry.py` for running parallel training jobs on Modal.
- Random TF-vs-PyTorch comparison harness improvements:
  - Added curated default allele panel in `scripts/compare_tf_pytorch_random_outputs.py`:
    - ~30 common HLA alleles plus a few animal alleles (`--allele-panel iedb_plus_animals`).
  - Reduced duplicate work in backend prediction:
    - Reused `Class1PresentationPredictor.predict(...)` processing outputs for
      `processing_with_score` and `processing_without_score` columns.
    - Removed separate direct processing predictor passes.
  - Runtime sanity:
    - Full `run --num-examples 5000` dropped from ~142s to ~80s on this machine.

- Added cross-product parity analysis workflow:
  - New script: `scripts/cross_allele_parity_analysis.py`
  - Generates random peptides uniformly across supported lengths (requested 7-15).
  - Crosses peptides against curated allele panel and predicts PT vs TF.
  - Produces:
    - prediction tables
    - numeric parity summaries
    - break analysis tables/report
    - plots under `plots/`
- Executed full run:
  - `1000` peptides x `35` alleles = `35000` pMHC rows
  - lengths: `7..15`
  - key result: no thresholded break events observed; differences remained at
    expected floating-point noise scale for score outputs and tiny absolute nM
    differences for affinity outputs.
- Follow-up experiment with random flanks:
  - Updated `scripts/cross_allele_parity_analysis.py` to:
    - generate random N/C flanks per peptide (length 5/5 from model support),
    - enforce pre-run uniqueness checks on peptide entries:
      - no repeated `peptide`, `n_flank`, or `c_flank`,
      - no duplicate `(peptide, n_flank, c_flank)` rows,
      - no duplicate `(peptide, allele, n_flank, c_flank)` in full dataset,
    - enforce post-run presentation sanity checks on both PT and TF:
      - at least 1% rows with score > 0.2,
      - at least one row with score > 0.9.
  - Run output dir: `/tmp/mhcflurry-cross-allele-1000-randflanks`
    - `1000` peptides x `35` alleles = `35000` rows.
  - Sanity thresholds passed:
    - PT with-flanks: 1.28% > 0.2, max 0.973
    - TF with-flanks: 1.28% > 0.2, max 0.973
    - PT without-flanks: 1.32% > 0.2, max 0.970
    - TF without-flanks: 1.32% > 0.2, max 0.970
- High-score fixture extraction for unit tests:
  - Added `scripts/extract_high_presentation_fixture.py`.
  - Extracted TF fixture rows from
    `/tmp/mhcflurry-cross-allele-1000-randflanks/tf_predictions.csv.gz`:
    - selected peptide+flank contexts where any allele had presentation score > 0.9,
    - retained all alleles for each selected context (including low scorers),
    - produced `315` rows (`9` contexts x `35` alleles).
  - Added fixture files:
    - `test/data/master_released_class1_presentation_highscore_rows.csv.gz`
    - `test/data/master_released_class1_presentation_highscore_rows_metadata.json`
  - Added regression test:
    - `test/test_released_presentation_highscore_rows.py`
    - validates fixture high/low context properties and compares released
      PyTorch predictions against TF fixture outputs.
