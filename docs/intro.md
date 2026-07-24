# Introduction and installation

MHCflurry predicts which peptides are likely to be displayed by MHC class I
molecules. It includes pretrained models and tools for three related tasks:

| Prediction | What it answers | Output direction |
|---|---|---|
| Binding affinity | How strongly does this peptide bind this MHC allele? | Lower affinity in nM is stronger. |
| Antigen processing | Does cellular processing favor this peptide? | Higher score is stronger. |
| Presentation | Is this peptide likely to be presented, considering binding and processing? | Higher score is stronger. |

For most epitope-prioritization work, start with the **presentation** predictor.
Use binding affinity when you specifically need peptide–MHC binding estimates,
or processing alone when you do not have an allele or genotype.

The default pan-allele models support most sequenced human MHC I alleles and
several other species. GPUs and Apple Silicon (MPS) are optional and are
detected automatically.

## Install MHCflurry

Install MHCflurry, including prereleases, with:

```shell
pip install --upgrade --pre mhcflurry
```

Omit `--pre` to install the latest stable release. Older releases may use the
historical `mhcflurry-*` command names shown in the command reference.

Download the pretrained presentation models:

```shell
mhcflurry downloads fetch models_class1_presentation
```

This bundle includes the binding-affinity and antigen-processing components
needed for presentation prediction.

## Make a first prediction

```shell
mhcflurry predict \
    --alleles HLA-A0201 HLA-A0301 \
    --peptides SIINFEKL SIINFEKD SIINFEKQ \
    --out predictions.csv
```

The output contains one row per peptide and allele or genotype query. The main
columns are:

- `mhcflurry_affinity`: predicted binding affinity in nM; lower is stronger.
- `mhcflurry_affinity_percentile`: allele-specific rank from 0–100; lower is
  stronger.
- `mhcflurry_processing_score`: processing score from 0–1; higher is stronger.
- `mhcflurry_presentation_score`: combined presentation score from 0–1; higher
  is stronger.

Separate allele arguments request separate predictions. A delimited allele
list represents one genotype and reports its strongest-binding allele. See
{ref}`allele-input-semantics` for examples. Historical `mhcflurry-*` command
names remain supported for existing scripts.

## Where to go next

- {doc}`commandline_tutorial`: predict peptides and scan proteins.
- {doc}`python_tutorial`: use predictors from Python.
- {doc}`training`: fit and select custom models.
- {doc}`evaluation`: compare trained models and generate evaluation figures.
- {doc}`commandline_tools`: complete generated command reference.
- {doc}`configuration`: runtime defaults, hardware autosizing, and
  reproducibility.

## Using conda

You can install into a conda environment and then use pip normally:

```shell
conda create -q -n mhcflurry-env python=3.10
conda activate mhcflurry-env
pip install --pre mhcflurry
mhcflurry downloads fetch models_class1_presentation
```

MHCflurry supports Python 3.10+ on Linux and macOS. Windows may work but is not
currently part of the supported test matrix.

## Getting help and citing MHCflurry

For questions and bug reports, use the
[GitHub issue tracker](https://github.com/openvax/mhcflurry/issues).

If you use MHCflurry in research, cite the MHCflurry 2.0 presentation-model
paper and the original binding-affinity paper listed in the
[project README](https://github.com/openvax/mhcflurry#citing-mhcflurry).
