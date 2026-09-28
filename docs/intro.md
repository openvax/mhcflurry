# Introduction and installation

MHCflurry predicts which peptides are likely to be displayed by MHC class I
molecules. Its pretrained models answer three related questions:

- **Binding affinity:** can this peptide bind a particular MHC allele?
- **Antigen processing:** is cellular processing likely to produce this
  peptide?
- **Presentation:** is the peptide likely to reach the cell surface, considering
  both binding and processing?

For most epitope-prioritization work, start with the **presentation** predictor.
Use binding affinity when you specifically need peptide–MHC binding estimates,
or processing alone when you do not have an allele or MHC allele set.

The default pan-allele models support most sequenced human MHC I alleles and
several other species. GPUs and Apple Silicon (MPS) are optional and are
detected automatically.

## Install MHCflurry

Install the latest stable MHCflurry 2.3 patch release with:

```shell
pip install --upgrade "mhcflurry>=2.3,<2.4"
```

Download the pretrained presentation models:

```shell
mhcflurry downloads fetch models_class1_presentation
```

This bundle includes the binding-affinity and antigen-processing components
needed for presentation prediction.

## Make a first prediction

Provide each peptide, its MHC alleles, and its N- and C-terminal source-protein
flanks in a CSV. This small example uses windows from the sequences in
[example.fasta](example.fasta):

```text
allele,peptide,n_flank,c_flank
HLA-A*02:01;HLA-A*03:01,TPVCPNGPG,MSSSS,NCQV
HLA-A*02:01;HLA-A*03:01,RLLEGMEMI,MVENK,FGQVI
```

Save it as `peptides.csv`, then run:

```shell
mhcflurry predict peptides.csv --out predictions.csv
```

Flanks are used by default when both columns are present. To compare the same
rows without that context:

```shell
mhcflurry predict peptides.csv --no-flanking --out predictions-no-flanks.csv
```

If the source context is unavailable, omit the flank columns; MHCflurry uses
its no-flank predictor. See {doc}`commandline_tutorial` for direct peptide
arguments and protein scanning.

## Understand the results

The output contains one row per peptide and allele or MHC allele set query. These are
the main prediction columns:

| Column | Interpretation |
|---|---|
| `mhcflurry_presentation_score` | Combined binding and processing score from 0–1; higher is stronger. |
| `mhcflurry_affinity` | Predicted binding affinity in nM; lower is stronger. |
| `mhcflurry_affinity_percentile` | Allele-specific rank from 0–100; lower is stronger. |
| `mhcflurry_processing_score` | Processing score from 0–1; higher is stronger. |

Separate allele arguments request separate predictions. A delimited allele
list represents one MHC allele set and reports its strongest-binding allele. See
{ref}`allele-input-semantics` for examples. Historical `mhcflurry-*` command
names remain supported for existing scripts.

## Where to go next

- {doc}`model_downloads` lists available weight releases and shows how to select older weights.
- {doc}`release_model_evaluation` compares the released models with figures and a PDF.

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
pip install --upgrade "mhcflurry>=2.3,<2.4"
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
