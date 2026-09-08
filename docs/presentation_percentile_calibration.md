# Tail-adaptive presentation percentiles

New presentation calibrations retain the 10,000 uniform-quantile bin grid and
add up to 10,000 bins in the top 1% of calibration scores. Additional quantiles
are spaced logarithmically in upper-tail probability, down to `1 / N` for `N`
finite calibration scores. Their count is also capped by the number of
calibration observations in that tail. Duplicate score edges are removed.

This addresses [issue #402](https://github.com/openvax/mhcflurry/issues/402):
uniform quantiles can place thousands of informative high-scoring evaluation
peptides in one percentile bin even though their raw scores are distinct.
Preserving the original broad grid also retains the resolution needed when
the presentation combiner's score range is compressed near zero.

The change is limited to choosing the default bins in
`Class1PresentationPredictor.calibrate_percentile_ranks`. It does not change
the shared percentile transform, affinity calibration, network predictions,
explicit caller-supplied bins, or any existing saved calibration table.
No binning rule is fitted to evaluation labels.

## Applying the fix

The existing `mhcflurry-calibrate-percentile-ranks` command uses the new default
automatically for `--predictor-kind class1_presentation`. Re-run the original
calibration recipe on a separate copy of the candidate predictor, retaining
the same calibration reference policy, random seed, and source provenance.
For full candidates, that recipe is in
`scripts/training/pan_allele_release_full.sh` and the saved `GENERATE.sh`.
The calibration command writes into `--models-dir`; do not point it at the
public baseline or the only preserved copy of the old candidate.

An old `percent_ranks.csv` does not contain the raw calibration scores inside
each bin. Merely subdividing that table would invent within-bin CDF detail.
Regenerate calibration predictions when those scores were not retained, then
repeat both with-flanks and without-flanks held-out comparisons. More bins
cannot resolve missing calibration-tail support, true raw-score ties, or
clipping beyond the calibration maximum.

## Verification

The regression test uses an independent million-score reference and 4,000
evaluation scores in a rare upper-tail interval. Its evaluation AUPRC
is 0.93023 for raw scores, 0.50025 with the old uniform-quantile grid, and
0.92849 with the new tail-adaptive grid. This is a synthetic regression test,
not a recalibrated release-candidate performance claim.

Tests also cover preservation of all base-grid edges, bounded extra-bin count,
small reference sets, repeated/nonfinite values, monotonicity and endpoint
behavior, compressed scores, explicit bin overrides, and saved-table loading.

The previously observed real-candidate raw presentation metrics remain about
0.3373 macro AUPRC and 0.3879 macro PPV@N. This binning change cannot turn those
raw rankings into both metrics above 0.4. PPV@N's row-order tie-breaking is a
[separate evaluation issue](https://github.com/openvax/mhcflurry/issues/405);
it is not changed by this fix.
