# Percentile calibration for all predictors

Affinity, processing, and presentation share two implementations:
{class}`~mhcflurry.HistogramPercentRankTransform` preserves the historical
histogram lookup; {class}`~mhcflurry.CompactPercentRankTransform` uses a
continuous, compact approximation to the reference distribution.

New predictor calibration defaults to compact, starting with **64 knots** and
allowing **128 only when background validation justifies it**. Loading an
existing predictor preserves its saved calibration, including public models.
Calibration changes neither neural-network weights nor raw prediction scores.

## What a percentile means

Predictor `percentile_ranks(...)` methods return values from 0 to 100, with
**lower always meaning stronger**. A percentile estimates how frequently the
chosen background produces an equally strong or stronger score. It is not a
probability that a peptide binds, is processed, or is presented.

| Predictor | Raw score and direction | Compact input coordinate | Low-level transform call |
|---|---|---|---|
| Affinity | IC50 in nM; lower is stronger | `log(IC50)` | `transform(ic50)` |
| Processing | Score in [0, 1]; higher is stronger | `logit(score)` | `transform(scores, survival=True)` |
| Presentation | Score in [0, 1]; higher is stronger | `logit(score)` | `transform(scores, survival=True)` |

Affinity calibration is per allele, with existing sequence-equivalent allele
lookup retained. Processing and presentation each store one score distribution
per predictor. A processing percentile is not an affinity percentile, and a
presentation percentile is not an average of its component percentiles.

## Functional form: what the knots store

For each distinct background score `v`, the compact fit computes the midrank
survival probability, assigning half the tied reference mass to either side:

```{math}
S(v) = \frac{\#\{v_j > v\} + \tfrac12\#\{v_j = v\}}{N}.
```

Each knot stores a pair `(x_i, y_i)`: the transformed raw score and
`logit(S(v_i))`. Logarithms are natural logarithms; `logit(p) = log(p / (1-p))`.
Between adjacent knots the ordinate is **linear**:

```{math}
y(x) = y_i + \frac{x-x_i}{x_{i+1}-x_i}(y_{i+1}-y_i),
\qquad x_i \leq x \leq x_{i+1}.
```

With `sigmoid(z) = 1 / (1 + exp(-z))`, the two directions are:

```{math}
P_{\mathrm{processing/presentation}}(v) = 100\,\operatorname{sigmoid}(y(x(v))),
\qquad
P_{\mathrm{affinity}}(v) = 100\,\operatorname{sigmoid}(-y(x(v))).
```

The implementation evaluates each direction directly; it does not obtain a
tiny upper-tail percentile by subtracting an already rounded CDF from 100.
This is a continuous, piecewise-linear curve in transformed coordinates, not
a histogram, a cubic spline, or a new neural network. Its slope can change at
a knot, so it is not generally differentiable there.

The fit keeps both endpoints and repeatedly adds the reference point with the
largest absolute interpolation error in `y`. Equal errors choose the earliest
source index. It stops at the budget, or earlier if there are too few distinct
coordinates or all remaining errors are at most `1e-12`. Thus 64 is a knot
**budget**, not a requirement to store 64 redundant points. Storage is two
coordinates per knot, two extrapolation slopes, and reference/selection metadata.

Outside the reference range, a line anchored at the corresponding endpoint
continues the curve. Its slope is fitted using up to 32 distinct reference
coordinates nearest that endpoint, rather than just the last knot interval.
Those extrapolated percentiles are model-based estimates: continuity does not
establish the accuracy of frequencies in a sparsely observed tail.

## Choosing 64 versus 128

There are two distinct APIs:

- Low-level `CompactPercentRankTransform(...).fit(scores, num_knots=64)` fits
  the specified budget directly, without automatic selection.
- Predictor `calibrate_percentile_ranks(..., max_knots=128)` and the shared
  `fit_percent_rank_transform` helper start with 64 and may select 128.
  **The default value 128 is a ceiling, not a starting size.**

The automatic rule uses background scores only, never evaluation labels:

1. With at least 1,000 groups, hold out one fifth of the groups using a fixed
   selection seed of **403**. Peptide identities define groups in the affinity
   APIs and calibration commands. Processing/presentation score-only APIs use
   rows unless the caller supplies `groups`; pass identities when rows repeat.
2. Compare 64- and 128-knot fits on the remaining groups. Evaluate percentile
   cutoffs **0.03, 0.1, 0.3, 1, 3, and 10 percent**, retaining only cutoffs with
   at least ten expected validation observations. At least two cutoffs and
   two distinct fitting scores are required; otherwise retain 64.
3. For a cutoff `q`, let `c_q` count validation percentiles at or below `q` and
   `M` be the validation row count. The smoothed observed percentage is
   `100 * (c_q + 0.5) / (M + 1)`. For each budget, `E` is the root-mean-square
   of `log10(observed_percentage / q)` over the eligible cutoffs.
4. Select 128 only if `E64 > E128 + max(0.01, 0.1 * min(E64, E128))`.
   Fit the selected budget on the complete background.

Setting `max_knots=64` disables escalation. Fewer than 1,000 groups also retain
64. This is a predeclared practical tolerance, not a statistical significance
test or a guarantee of accuracy at every percentile.

The compact transform's `selection` dictionary records the selected budget,
actual knot count, reason, seed, reference count/hash, and—when comparison is
possible—validation counts, split-mask hash, cutoffs, errors, and tolerance.
The fixed selection seed is separate from the CLI `--random-seed` (default 42),
which governs generation of calibration peptides and MHC allele sets.

## Python examples

These small synthetic arrays illustrate the API, not a suitable release
calibration background. Both classes use instance `fit`, return `self`, and
support `to_dict`/`from_dict` serialization.

```{doctest}
>>> import json
>>> import numpy as np
>>> from mhcflurry import HistogramPercentRankTransform, CompactPercentRankTransform
>>> from mhcflurry.percent_rank_transform import PercentRankTransform
>>> PercentRankTransform is HistogramPercentRankTransform
True
>>> background_ic50 = np.array([1., 10., 100., 1000.])
>>> affinity_curve = CompactPercentRankTransform(score_transform="log").fit(background_ic50)
>>> np.round(affinity_curve.transform(background_ic50), 2).tolist()
[12.5, 37.5, 62.5, 87.5]
>>> probability_curve = CompactPercentRankTransform(score_transform="logit").fit([.1, .1, .2, .9])
>>> np.round(probability_curve.transform([.1, .2, .9], survival=True), 2).tolist()
[75.0, 37.5, 12.5]
>>> restored = CompactPercentRankTransform.from_dict(json.loads(json.dumps(probability_curve.to_dict())))
>>> np.array_equal(restored.transform([.2, .8]), probability_curve.transform([.2, .8]))
True
>>> histogram = HistogramPercentRankTransform().fit(background_ic50, bins=[1., 10., 100., 1000.])
>>> np.array_equal(histogram.transform([5., 50.], survival=True), 100 - histogram.transform([5., 50.]))
True
```

For predictor calibration, affinity takes background **peptides** and performs
inference; processing and presentation take already computed background
**scores**. Their `percentile_ranks` methods all take raw scores:

```python
# Assume independently prepared backgrounds and loaded predictor instances.
affinity_predictor.calibrate_percentile_ranks(
    peptides=background_peptides, alleles=calibration_alleles,
    method="compact", max_knots=128)
affinity_percentiles = affinity_predictor.percentile_ranks(query_ic50, allele=allele)

processing_predictor.calibrate_percentile_ranks(
    background_processing_scores, groups=background_peptides,
    method="compact", max_knots=128)
processing_percentiles = processing_predictor.percentile_ranks(query_processing_scores)

presentation_predictor.calibrate_percentile_ranks(
    background_presentation_scores, groups=background_peptide_per_score,
    method="compact", max_knots=128)
presentation_percentiles = presentation_predictor.percentile_ranks(query_presentation_scores)
```

Processing's raw `predict`/`predict_to_dataframe` outputs are unchanged; call
`percentile_ranks` explicitly. An uncalibrated processing predictor raises by
default; `throw=False` warns and returns NaNs. Do not assume historical public
processing models include a calibration merely because the API now exists.

Use `method="histogram"` to explicitly request histogram calibration. Supplying
`bins` also selects it when `method` is omitted; `method="compact"` plus `bins`
is rejected. Exact historical reproduction requires the original reference
and bin edges, not just the histogram class name. In particular, the current
presentation histogram default uses the later tail-adaptive bin policy, not
necessarily the grid in an older public artifact.

`CompactPresentationPercentiles` is only an adapter for the original experiment:
it retains its historical classmethod fit and upper-tail default. New code
should use the generic class and specify `survival=True` where appropriate.

## Calibration commands and reference policy

The command writes **in place**. Point it at a separately preserved candidate
copy, not a public baseline or the only copy of an earlier calibration.
These are command templates; retain the original release's reference policy,
sample count, amino-acid distribution, seed, and MHC allele set sampling for a
controlled recalibration. Omitted affinity/presentation reference-generation
options use CLI defaults, which need not match a release's recipe.

```shell
mhcflurry calibrate-percentile-ranks \
    --models-dir candidate-affinity-copy \
    --predictor-kind class1_affinity \
    --percentile-method compact --max-percentile-knots 128 \
    --random-seed 42 --num-jobs 0

mhcflurry calibrate-percentile-ranks \
    --models-dir candidate-processing-copy \
    --predictor-kind class1_processing \
    --processing-reference-data independent_background.csv \
    --percentile-method compact --max-percentile-knots 128 \
    --num-jobs 0

mhcflurry calibrate-percentile-ranks \
    --models-dir candidate-presentation-copy \
    --predictor-kind class1_presentation \
    --percentile-method compact --max-percentile-knots 128 \
    --random-seed 42 --num-jobs 0
```

Processing requires an explicit independent background CSV with `peptide` and,
for flanked ensembles, `n_flank` and `c_flank`. Empty flank cells mean missing
context. Use the same ensemble and context policy as at prediction time; the
command does not fabricate flanks or generate a processing reference from
`--num-peptides-per-length`. It currently supports `--num-jobs 0` only and
retains inputs and raw scores in `percent_rank_reference.npz`.

Affinity/presentation commands generate background predictions from their
configured reference policy; unlike the processing command, they do **not**
automatically retain a raw reference-score archive. Preserve those predictions
separately when needed for calibration plots/reanalysis. Retain model hashes,
reference inputs, the command, transform files, and held-out prediction tables
with joinable sample/peptide/allele identifiers for all comparisons.

Affinity `--only-missing` fills gaps, including sequence-equivalent lookup;
it does not convert existing histogram calibrations to compact. For an
intentional conversion of all requested alleles, omit that flag on a copy.

## Saved formats and compatibility

- Historical `percent_ranks.csv` loads through the histogram implementation,
  without refitting. `PercentRankTransform` remains an import/pickle alias.
- `percent_ranks.json` stores versioned compact or mixed collections, including
  different knots per allele. All-histogram collections use CSV when their
  edges match; different histogram grids also require JSON.
- Saving replaces the target calibration file and removes its alternate
  format only after successful replacement. Saving an empty collection removes
  saved calibration. Loading prefers JSON if both files are present; it does
  not fall back to CSV if the JSON is invalid.
- Older software cannot interpret compact JSON calibration. Do not expect
  percentile-dependent prediction to work there merely because the network
  weight files remain readable. Preserve original artifacts for old clients.

Changing ensemble members or weights requires new calibration. Keep raw-score
performance comparisons separate from percentile-mapping comparisons.

## Numerical limits and interpretation

Compact fitting requires a finite, nonconstant reference. Log coordinates
require positive scores; logit coordinates require scores in [0, 1]. Exact
zero and one use the nearest interior float64 values for taking logits.
Transform calls propagate NaNs but reject infinities and out-of-domain scores.
The histogram implementation retains its historical input/boundary behavior.

Compact interpolation is monotone within one distribution, avoiding the
deliberate step ties introduced by histograms. It cannot recover ties already
present in raw scores or remove floating-point rounding. Extreme percentiles
can still underflow to zero or approach 100 without distinguishable digits;
`log_percentiles(scores, survival=...)` retains tiny-tail information before
exponentiation, but does not validate extrapolated frequencies.

Monotonic mapping preserves ordering within a shared calibration distribution,
subject to those numerical limits. Combining allele-specific calibrations can
change ordering **across** alleles. Better percentile resolution is not an
improvement in the neural network's raw ranking ability, and calibration is
not a substitute for held-out affinity/processing/presentation evaluation.

See {doc}`presentation_percentile_calibration` for the archived presentation
comparison and its distinct 60/20/20 experimental validation design.
