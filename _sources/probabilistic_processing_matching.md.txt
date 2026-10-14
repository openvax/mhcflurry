# Processing negative matching

The maintained processing-data workflow uses seeded random matching without
replacement (`sample-length-affinity-random-without-replacement-v2`). A negative
peptide can be selected only once within a sample, including when the same sequence
maps to multiple proteins. It may be used in another sample with a different MHC.

## Matching contract

Hits and negatives share a sample and peptide length. The absolute difference in
predicted log10 affinity must not exceed the configured caliper (at most 0.25).
Eligible negatives from the same source protein are preferred. Matching draws
uniformly from the currently available eligible preferred pool, then from the
remaining sample/length pool, with a seeded random assignment order.

Augmenting paths repair competing assignments so an early draw cannot make a
feasible pool appear insufficient. This is randomized sequential matching;
it does not sample uniformly from all possible complete matchings. If no complete
assignment exists, preparation saves the failure and expands the scored pool.
It never reuses negatives, widens the caliper, or drops hits to finish.

Saved metadata identifies the affinity reference, policy, seed, caliper and ratio.
Resume validates these inputs and rejects assignments from another policy. See
{doc}`processing_preparation_acceleration` for artifacts and execution.

## Interpretation and evaluation

The original [MHCflurry processing model](https://doi.org/10.1016/j.cels.2020.06.010)
used strong-binder filtering. Affinity matching instead controls predicted binding
more locally. Neither procedure guarantees removal of every binding-associated
signal, nor turns presentation labels into direct measurements of processing.

Training defaults to one matched negative per hit. Processing-specific evaluation
can use the same matcher at another ratio. The public release presentation
benchmark has its own frozen negative cohort; this matcher does not redefine it.
Every comparator in a comparison must receive identical saved rows and labels.
AP values from different negative policies or ratios are not directly comparable.

## Validation

Regression tests cover uniqueness within samples, reuse across samples, seeded
reproducibility, caliper boundaries, matching feasibility against independent
small-graph solutions, metadata validation and adaptive expansion. These checks
establish the sampler's contract, not improved trained-model accuracy. Changes to
the training policy require separate held-out component and full-model evaluation.
