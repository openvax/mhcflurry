# MHCflurry 2.3.1rc1

This training-code prerelease fixes repeated negative peptides in processing
data matching. It does not publish new trained model weights; the default
downloaded models remain the public 2.2.0 models.

## Processing data matching

- Sample eligible negatives randomly with a recorded seed, without replacement
  within each sample. Duplicate protein mappings cannot increase a peptide's
  training weight.
- Retain all hits, exact sample and peptide-length matching, the hard affinity
  caliper, and preference for available eligible same-protein negatives.
- Repair competing assignments, and expand insufficient candidate pools through
  the existing preparation workflow. Never reuse a negative to fill a shortage.
- Record the new matching policy and reject old matched caches for new training.
  Use a fresh experiment output directory; historical experiments need their
  original pinned code and artifacts.

The sampling fix has algorithmic and real-pool checks. Its effect on trained
model accuracy requires a separate controlled experiment. Presentation feature
recommendations are documented, with no combiner change in this prerelease.

Fixes #432. Release-version test fixtures also follow the package version (#434).
