# Experiment log and whitepapers

Technical whitepapers, dated validation studies, cross-species benchmarks, and
controlled ablations supporting MHCflurry releases.

## 2.5 series — cross-species & pan-class studies

- **{doc}`whitepaper_25x_cross_species`** ([Download PDF](_static/whitepaper/mhcflurry_25x_cross_species_whitepaper.pdf)) —
  Unified technical whitepaper synthesizing all three `2.5.x` studies below
  with 15 standalone single-panel vector figures, summary tables, a
  plain-English glossary, and practical guidance.
- {doc}`experiment_25x_cattle_dog_hitlist` *(2026-11-26)* —
  Training-overlap audit and strictly unseen evaluation of Cattle (`BoLA-I`,
  `21,332` flanked MS ligands) and Dog (`DLA-88`) cohorts from
  [`hitlist`](https://github.com/pirl-unc/hitlist).
- {doc}`experiment_25x_pan_class_ii` *(2026-11-25)* —
  Exploratory zero-shot test of frozen Class I `2.3.0` weights on 15 Class II
  `HLA-DR`/`DQ`/`DP` alleles using
  [`mhcseqs`](https://github.com/pirl-unc/mhcseqs) alignments, benchmarked
  against `NetMHCIIpan-4.3`.
- {doc}`experiment_25x_cross_species_hitlist` *(2026-11-25)* —
  Zero-shot Class I affinity and presentation benchmarks across 11 unseen
  vertebrate species and 3 pocket mutants, plus bit-identical `--lock-existing`
  training parity for [`mhcseqs`](https://github.com/pirl-unc/mhcseqs).

## 2.3 series — model release evaluation & audits

- {doc}`release_model_evaluation` & {doc}`release_model_evaluation_details` —
  Head-to-head benchmarks of the `2.3.0` PyTorch weights against `2.2.0`,
  NetMHCpan, and MixMHCpred.
- {doc}`dla_validation` —
  Canine (`DLA`) allele capability report, training-data overlap audit, and
  evaluation limits.
- {doc}`final_230_candidate_experiment` —
  Clean-checkout replay of the `2.3.0` pan-allele training recipe.

Additional working ablation notes (cleavage-boundary sweeps, kernel-width
sweeps, and historical replays) live in the repository's
[`notes/` directory](https://github.com/openvax/mhcflurry/tree/master/notes).

```{toctree}
:maxdepth: 1
:hidden:

whitepaper_25x_cross_species
dla_validation
experiment_25x_cattle_dog_hitlist
experiment_25x_pan_class_ii
experiment_25x_cross_species_hitlist
```
