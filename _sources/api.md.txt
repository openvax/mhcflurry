(api-documentation)=

# API Documentation

For worked examples, see {doc}`python_tutorial`; for calibration-specific
examples and compatibility, see {doc}`shared_percent_rank_transforms`.

## Public API overview

Most Python callers only need the top-level predictors and download helpers:

| Symbol | Role |
|---|---|
| {class}`~mhcflurry.Class1PresentationPredictor` | Recommended entry point: predicts binding affinity, antigen processing, and combined presentation scores. |
| {class}`~mhcflurry.Class1AffinityPredictor` | Pan-allele or allele-specific peptide–MHC class I binding affinity predictor. |
| {class}`~mhcflurry.Class1ProcessingPredictor` | Antigen processing predictor (with or without N/C flanking sequences). |
| {class}`~mhcflurry.CompactPercentRankTransform` | Compact knot-based percentile rank transform for affinity, processing, and presentation calibration. |
| {mod}`mhcflurry.downloads` | Locates downloaded model weights and catalogue releases (`get_path`, `get_downloads_dir`, `resolve_data_location`). |
| {mod}`mhcflurry.pseudosequences` | Extracts `mhcseqs`-backed Class I binding-groove pseudosequences from arbitrary heavy-chain sequences or `mhcseqs` tables (`extract_class1_pseudosequence`, `build_pseudosequences_from_mhcseqs`). |

Parallelism helpers live in {mod}`mhcflurry.parallelism` and CLI modules live
under `mhcflurry.cli.*`.

## Full module reference

```{eval-rst}
.. include:: _build/mhcflurry.rst
   :start-line: 2
```
