# Probabilistic processing matching without replacement

## Scope and contract

Replace nearest-neighbor reuse with seeded random matching for newly generated
processing training and evaluation cohorts. A negative peptide may be selected
only once within a sample, even when it maps to multiple proteins. The same
sequence can be used in a different sample, where its allele context differs.
Keep all hits, sample and peptide-length matching, the hard log10-affinity
caliper (at most 0.25), and preference for eligible same-protein negatives.

Draw uniformly among available eligible negatives in the preferred pool,
then the remaining sample/length pool. Randomize assignment order using a
recorded seed. Repair competing assignments through augmenting paths so a
greedy early draw cannot make a feasible pool appear insufficient. This is
randomized sequential matching, not a claim of uniform sampling over all
possible complete matchings. If no complete assignment exists, preserve the
failure evidence and expand the scored pool through the existing preparation
workflow; never reuse a negative or widen the caliper to finish.

Version the matching policy and record the seed, replacement rule, and
uniqueness diagnostics. Reject old matched caches for new training/resumption
instead of relabeling them. Existing v2 experiments retain their pinned source,
assignments, and results. New evaluation cohorts need the new policy too;
comparators must share saved assignments.

## Validation

- Regression: competing hits cannot reuse a negative, including duplicate
  source rows/protein mappings; reuse remains legal across samples.
- Reproducibility and randomness: repeat a seed exactly, vary assignments
  across seeds, and retain assignments when processing samples separately.
- Feasibility: test adversarial greedy traps and compare small random graphs
  with an independent maximum-matching implementation.
- Invariants: ratios, labels, sample/length, affinity boundaries, and saved
  metadata/cache checks. Exercise adaptive pool expansion and resumption.
- Real-data replay: compare old and new assignments from saved scored pools;
  report uniqueness, calipers, distributions, completion, and runtime. Keep
  frozen held-out model metrics descriptive; the sampler alone does not
  establish improved trained-model accuracy.
- Run Ruff, focused tests, and the full test suite before proposing the PR.

## Presentation features to evaluate separately

Start with categorical peptide length and interactions of length with affinity
and processing. Test a clipped log-odds transform of the processing probability
(a representation change, not additional information). A later model can add
allele-dependent length preferences or multiple per-allele binding scores;
expression/protein abundance should be an optional sample-aware extension.
Use training/validation samples for feature selection, regularize the small
combiner, and preserve the final holdout. No presentation feature or released
prediction changes are part of the matching fix.

The original MHCflurry processing model uses strong-binder filtering:
https://pubmed.ncbi.nlm.nih.gov/32711842/. Allele-specific length preferences
are modeled in NetMHCpan 4.0: https://pmc.ncbi.nlm.nih.gov/articles/PMC5679736/.
HLAthena integrates transcript abundance and processing:
https://www.nature.com/articles/s41587-019-0322-9.

## Initial real-data verification (2026-09-16)

Replay of the saved initial scored pools from v2-2020:

| Sample | Hits | Old distinct negatives | Old maximum reuse | Additional assignments needed without replacement |
| --- | ---: | ---: | ---: | ---: |
| KESKIN_B1510 | 1,624 | 440 | 152 | 849 |
| KESKIN_B4901 | 4,037 | 1,913 | 49 | 614 |
| KESKIN_C0401 | 2,075 | 704 | 148 | 754 |

The new matcher correctly refuses these insufficient pools, preserving all
hits for adaptive expansion. The exact interval-based capacity check takes
0.15–0.51 seconds per pool on the local machine. On seeded 100-hit subsets
of the same pools (all candidate negatives retained), all three produce
exactly 100 distinct negatives and satisfy the caliper, in 0.14–0.33 seconds.
Those subsets test the algorithm, not trained-model accuracy. Full-size data
preparation will need more candidates; no improved model accuracy is claimed.
Artifacts: `output/probabilistic-matching-20260916/real-data-replay.json` and
`real-pool-capacity-smoke.json`, including input hashes and diagnostics.
