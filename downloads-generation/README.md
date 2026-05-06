# Downloads generation

This directory contains code and instructions needed to *generate* the datasets and trained models published with MHCflurry.

If you are only looking to download datasets and trained models, you do not need to use any of this. Just run `mhcflurry-downloads fetch` to download the standard models and datasets.

## Class I Pseudosequence Files

Class I pan-allele and allele-specific model artifacts should ship the
pseudosequence CSV they were trained with. New artifacts use explicit
pseudosequence filenames and keep the historical `allele_sequences.csv` or
`class1_pseudosequences.csv` aliases only for compatibility with older
MHCflurry releases and scripts:

- `pseudosequences.netmhcpan.34aa.csv` — NetMHCpan-derived 34 amino acid
  pseudosequences.
- `pseudosequences.mhcflurry.37aa.csv` — MHCflurry-generated 37 amino acid
  pseudosequences used by older public pan-allele model bundles. This is a
  model-artifact compatibility table, not the output of the current
  `downloads-generation/allele_sequences` recipe.
- `pseudosequences.mhcflurry.39aa.csv` — MHCflurry-generated 39 amino acid
  pseudosequences from the aligned full-sequence pipeline.

Do not substitute the standalone `allele_sequences` download for the
pseudosequence CSV inside a trained model directory. The saved weights depend on
the representation width and position definition used during training.
