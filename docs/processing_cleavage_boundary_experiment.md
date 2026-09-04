# Processing cleavage-boundary experiment

## Question

Does sequence context spanning the two peptide boundaries add antigen-processing
signal after controlling for peptide--MHC binding affinity, and can an explicit
boundary-aligned architecture recover that signal more reliably than the legacy
full-sequence convolution?

The boundary inputs intentionally contain residues on both sides of each cut,
using standard protease cleavage-site orientation. Compare two window variants:

- Compact N: `n_flank[-5:] + peptide[:2]` (`P5...P1 | P1'P2'`)
- Compact C: `peptide[-2:] + c_flank[:5]` (`P2P1 | P1'...P5'`)
- Extended N: `n_flank[-5:] + peptide[:5]` (`P5...P1 | P1'...P5'`)
- Extended C: `peptide[-5:] + c_flank[:5]` (`P5...P1 | P1'...P5'`)

The phrase "flank branch" must not be used for this transform because the
within-peptide residues are part of the cleavage context.

Unavailable external context is encoded with the existing `X` unknown token.
This covers source-protein termini and prediction calls without supplied
flanks. During training only, independently replace the complete five-residue
external N or C segment with `X` with a configurable probability. Never replace
the two within-peptide residues. This context dropout is distinct from hidden
unit/channel dropout and must be recorded separately.

## Phase A: evaluation without retraining

1. Join the frozen processing benchmark to its cached public MHCflurry affinity
   predictions using an asserted one-to-one row identity.
2. Preserve the existing unfiltered proteome-retrieval metrics.
3. Add affinity-controlled risk sets matched within sample and peptide length
   on best-genotype `log10(affinity)`. Prefer same-source-protein matches when
   the source identifier is available. Record unmatched hits and matching
   distances.
4. Report paired macro AUPRC, PPV@N, ROC AUC, and hit-versus-decoy concordance,
   with per-sample and per-length tables.
5. Re-score existing 5-aa models after independently replacing the N context,
   C context, or both with shuffled context. Shuffling is within sample and
   peptide length and uses fixed recorded seeds.

All joined rows, scores, match assignments, seeds, metrics, configuration, and
input hashes must be retained for plots and external-predictor joins.

## Phase B: crossed-boundary model

Add an opt-in architecture that retains a full-peptide base path and adds two
separate, cleavage-aligned residual paths. Each residual path receives its
crossed-boundary window defined above. Its contribution to the final logit is
initialized to zero, so the model initially represents the peptide-only base
exactly. The full-peptide path is retained because ERAP/TAP and other processing
effects can depend on peptide length, termini, or distal internal residues that
do not belong in an arbitrarily enlarged cleavage window.

First screen the compact and extended windows with independent external-context
dropout probability 0.25. Use the two previously selected representative
processing architectures, four folds, batch 512, Glorot initialization,
Keras-compatible Adam, and the same training rows and fold assignments as the
prior processing screen. Only if a boundary variant passes the initial gate,
train its matched context-dropout-0 ablation. Do not expand the production grid
during screening.

Every comparison is architecture- and fold-matched against both the legacy
5-aa model and the no-flank model. Retrain these controls in the same job rather
than importing prior model artifacts, for a total screen of 32 networks (16
boundary and 16 legacy-control networks). Training rows, folds, seeds,
optimizer, minibatch, early-stopping/checkpoint policy, and held-out prediction
rows remain identical. Report paired differences rather than comparing only
aggregate ensembles.

## Decision rule

Advance the crossed-boundary model only if it improves both sample-macro AUPRC
and PPV@N on affinity-controlled risk sets, the improvement is directionally
consistent across samples and lengths, and it causes no meaningful regression
on the unfiltered benchmark. The final release decision additionally requires
improvement or parity in the cached affinity/processing 2x2 presentation test.

The experiment snapshot must contain per-epoch histories, terminal and best
checkpoints, all validation and held-out predictions, model manifests, input
hashes, software provenance, comparison tables, and plot-ready figure inputs.
