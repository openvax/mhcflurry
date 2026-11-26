# Experiment log

Dated empirical studies, cross-species validation campaigns, and controlled
ablation experiments supporting MHCflurry releases.

## 2.5 series

- {doc}`experiment_25x_pan_class_ii` *(2026-11-25)* — Speculative zero-shot
  Class I → Class II transfer test: mapping Class II `DR`/`DQ`/`DP`
  heterodimers into MHCflurry's 39-aa groove representation via `mhcseqs`
  pan-class IMGT G-DOMAIN alignments and benchmarking frozen Class I `2.3.0`
  weights against `NetMHCIIpan-4.3` and an `HLA-A*02:01` negative control
  across 15 alleles (`7,500` 15-mer IC50 pairs).
- {doc}`experiment_25x_cross_species_hitlist` *(2026-11-25)* — Zero-shot
  Class I Affinity and Presentation (+ real species proteome flanks) validation
  across 11 previously unseen vertebrate species and 3 binding-pocket mutants
  using [`hitlist`](https://github.com/pirl-unc/hitlist) + IEDB, taxonomic
  expansion census (`+184` net new species in `mhcseqs`), and controlled A/B
  pan-allele training parity (`Max |Δ| = 0.0`, `r = 1.0`) validating the
  `GENERATE.sh --lock-existing` migration.

## 2.3 series

- {doc}`release_model_evaluation` & {doc}`release_model_evaluation_details` —
  Head-to-head benchmarks of the released `2.3.0` PyTorch weights against
  historical `2.2.0` weights and external predictors.
- {doc}`dla_validation` — Exploratory canine DLA-88 evaluation and training
  evidence audit.
- {doc}`final_230_candidate_experiment` — Clean-checkout replay of the `2.3.0`
  pan-allele release training recipe.

Additional working ablation notes (cleavage-boundary sweeps, kernel-width
sweeps, and historical public-data replays) live in the repository's
[`notes/` directory](https://github.com/openvax/mhcflurry/tree/master/notes).

```{toctree}
:maxdepth: 1
:hidden:

experiment_25x_pan_class_ii
experiment_25x_cross_species_hitlist
```
