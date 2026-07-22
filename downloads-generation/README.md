# Downloads generation

This directory contains reproducible generators for datasets and trained models
published with MHCflurry.

Prediction users do not need these generators. Use `mhcflurry downloads fetch`
to install published models and datasets.

## Class I Pseudosequence Files

The canonical pseudosequence filename registry is
`mhcflurry/pseudosequences.py`. Use it from Python or shell scripts instead of
hardcoding pseudosequence artifact names:

```bash
mhcflurry pseudosequences list
mhcflurry pseudosequences filename --length 39
mhcflurry pseudosequences path \
    --directory "$(mhcflurry downloads path allele_sequences)" \
    --length 39 \
    --fallback-legacy
```

Do not substitute the standalone `allele_sequences` download for the
pseudosequence CSV inside a trained model directory. The saved weights depend on
the representation width and position definition used during training.
