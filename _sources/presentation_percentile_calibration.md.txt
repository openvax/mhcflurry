# Presentation percentile calibration

New calibration now uses the shared `CompactPercentRankTransform`, also used
for affinity and processing. See [shared APIs and compatibility](shared_percent_rank_transforms.md).
The automatic budget starts at 64 and permits 128 only after a meaningful
label-free background-validation improvement. Historical calibrations still
load as `HistogramPercentRankTransform`, without conversion or refitting.

## Experimental compact-curve comparison

`mhcflurry eval presentation-percentiles` compares histogram mappings with
continuous curves, without loading or retraining networks. The original
experiment included 64/128/256 knots; the current production policy permits
only 64/128. Specify those two budgets explicitly for a new comparison:

```shell
mhcflurry eval presentation-percentiles fit \
    --reference-dir /path/to/cached-reference --out /path/to/new-comparison \
    --knots 64 128
mhcflurry eval presentation-percentiles evaluate \
    --calibration-dir /path/to/new-comparison \
    --predictions /path/to/predictions_with_flanks.csv.bz2 --mode with-flanks
mhcflurry eval presentation-percentiles evaluate \
    --calibration-dir /path/to/new-comparison \
    --predictions /path/to/predictions_without_flanks.csv.bz2 --mode without-flanks
mhcflurry eval presentation-percentiles plot \
    --calibration-dir /path/to/new-comparison
```

The reference directory contains `reference_peptides.csv.bz2` and one aligned
`reference_scores_*.npy` vector per allele. Prediction tables contain
`sample_id`, `hla`, `hit`, and `a_`/`b_` presentation score/percentile columns;
side `a` is the candidate and side `b` is the contextual public baseline.

This experimental command splits the reference 60/20/20 by unique peptide,
keeping all allele queries for a peptide together. Knot count is selected using
background validation accuracy, not presentation labels. Its separate test
split checks calibration. This is distinct from the production helper's
80/20 selection/refit policy and minimum-count safeguards; see
{doc}`shared_percent_rank_transforms` for that rule. The experiment command's
historical default still includes 256, so use `--knots 64 128` as shown above.
Full-reference refits are then compared on the frozen presentation rows.
Outputs retain the splits, counts at percentile cutoffs, versioned curves,
micro/per-sample metrics, tie-neutral PPV, ranking audits, transformation timings,
and joinable prediction arrays, plus a report and PNG/SVG figures.

The experimental curves interpolate in logit-score/logit-survival coordinates,
using weighted midranks at distinct reference scores. End slopes estimated from
up to 32 distinct reference scores provide explicitly modeled extrapolation.
Extreme extrapolated percentiles are not empirically established probabilities.
Existing raw-score ties and floating-point saturation cannot be recovered by
this mapping. The original experiment was isolated; its generic compact
implementation is now used by new predictor calibration and versioned loading.

## Historical tail-adaptive histogram method

With no explicit `bins`, `method="histogram"` calibration retains up to 10,000
uniform-quantile bins and adds up to 10,000 bins in the top 1% of calibration
scores. The base grid is bounded by the finite reference count. Additional quantiles
are spaced logarithmically in upper-tail probability, down to `1 / N` for `N`
finite calibration scores. Their count is also capped by the number of
calibration observations in that tail. Duplicate score edges are removed.

Uniform quantiles can place thousands of informative high-scoring evaluation
peptides in one percentile bin even though their raw scores are distinct.
Preserving the original broad grid also retains the resolution needed when
the presentation combiner's score range is compressed near zero.

This remains available for reproducing the earlier binning experiment.
Neither method changes network predictions or any existing saved calibration
merely by loading it. No calibration rule is fitted to evaluation labels.

## Calibrating a predictor

The existing `mhcflurry-calibrate-percentile-ranks` command uses the new default
automatically for `--predictor-kind class1_presentation` (compact by default;
`--percentile-method histogram` reproduces the earlier tail-bin method). Re-run the original
calibration recipe on a separate copy of the predictor, retaining
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

## Historical histogram experiment verification

The earlier histogram regression experiment used an independent million-score
reference and 4,000 evaluation scores in a rare upper-tail interval. Its AUPRC
was 0.93023 for raw scores, 0.50025 with the old uniform-quantile grid, and
0.92849 with the tail-adaptive grid. These are historical synthetic results,
not a recalibrated release-candidate performance claim. The current default
calibration regression tests exercise compact interpolation instead.

Tests also cover preservation of all base-grid edges, bounded extra-bin count,
small reference sets, repeated/nonfinite values, monotonicity and endpoint
behavior, compressed scores, explicit bin overrides, and saved-table loading.

The archived real-candidate comparison found that 64 compact knots recovered
the raw presentation ranking on the frozen with-flanks cohort: macro AUPRC
0.337292 and macro PPV@N 0.387948 (using tie-neutral PPV), with the same ranking
metrics at 128 and 256 knots. This used the experiment's independent cached
background, not a fresh calibration on the complete release-policy reference.
The artifact directory is
`experiments/20260909T023206Z-compact-presentation-percentiles-comparison-9778c6256b7e`.

Neither compact nor histogram calibration can turn those raw rankings into
both metrics above 0.4. PPV@N's row-order tie-breaking is a
[separate evaluation issue](https://github.com/openvax/mhcflurry/issues/405);
it is not changed by this fix.
