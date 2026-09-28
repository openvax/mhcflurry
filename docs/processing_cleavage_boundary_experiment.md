# Cleavage-boundary processing models

## Boundary inputs

Does sequence context spanning the two peptide boundaries add antigen-processing
signal after controlling for peptide--MHC binding affinity, and can an explicit
boundary-aligned architecture recover that signal more reliably than the legacy
full-sequence convolution?

The boundary inputs intentionally contain residues on both sides of each cut,
using standard protease cleavage-site orientation. The supported window variants are:

- Compact N: `n_flank[-5:] + peptide[:2]` (`P5...P1 | P1'P2'`)
- Compact C: `peptide[-2:] + c_flank[:5]` (`P2P1 | P1'...P5'`)
- Extended N: `n_flank[-5:] + peptide[:5]` (`P5...P1 | P1'...P5'`)
- Extended C: `peptide[-5:] + c_flank[:5]` (`P5...P1 | P1'...P5'`)

The phrase "flank branch" must not be used for this transform because the
within-peptide residues are part of the cleavage context.

Unavailable external context is encoded with the existing `X` unknown token.
This covers source-protein termini and prediction calls without supplied
flanks. During training only, independently replace the complete five-residue
external N or C segment with `X` with a configurable probability. Within-peptide residues are preserved. This context dropout is distinct from hidden
unit/channel dropout and must be recorded separately.

## Architecture and evaluation

The opt-in boundary architecture retains a full-peptide base path and adds two
separate boundary-aligned residual paths. Each residual path receives its
crossed-boundary window. Its contribution to the final logit is initialized to
zero, so initial predictions come from the base path. See
{doc}`release_training_recipe` for the 5x5 boundary family selected for 2.3.0.

For processing-specific comparisons, use the same sample/length/affinity-matched
risk sets for all predictors and retain the frozen public affinity reference.
`mhcflurry eval processing-flank-ablation` can compare real, masked and shuffled
contexts with unchanged weights. These diagnostics differ from end-to-end
presentation ranking and do not establish that independently selected
flank/no-flank ensembles constitute a controlled ablation.

Preserve input and model hashes, row identities, matching assignments, seeds and
per-patient predictions. Report AP, PPV@N and AUROC with paired patient intervals,
and evaluate the complete presentation model separately before promoting a
processing change.
