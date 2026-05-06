# Downloads generation

This directory contains code and instructions needed to *generate* the datasets and trained models published with MHCflurry.

If you are only looking to download datasets and trained models, you do not need to use any of this. Just run `mhcflurry-downloads fetch` to download the standard models and datasets.

## Class I Pseudosequence Files

The canonical pseudosequence filename registry is
`mhcflurry/pseudosequences.py`. Use it from Python or shell scripts instead of
hardcoding pseudosequence artifact names:

```bash
mhcflurry-pseudosequences list
mhcflurry-pseudosequences filename --length 39
mhcflurry-pseudosequences path \
    --directory "$(mhcflurry-downloads path allele_sequences)" \
    --length 39 \
    --fallback-legacy
```

Do not substitute the standalone `allele_sequences` download for the
pseudosequence CSV inside a trained model directory. The saved weights depend on
the representation width and position definition used during training.
