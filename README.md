[![Build Status](https://github.com/openvax/mhcflurry/actions/workflows/ci.yml/badge.svg)](https://github.com/openvax/mhcflurry/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/openvax/mhcflurry/badge.svg?branch=master)](https://coveralls.io/github/openvax/mhcflurry?branch=master)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvax/mhcflurry/blob/master/notebooks/mhcflurry-colab.ipynb)

# MHCflurry

MHCflurry predicts which peptides are likely to be displayed by MHC class I
molecules. It provides pretrained models for three related tasks:

- **Binding affinity:** how strongly a peptide binds an MHC allele.
- **Antigen processing:** whether cellular processing favors the peptide.
- **Presentation:** a combined score using binding and processing predictions.

You can use the released models from the command line or Python, scan proteins
for candidate epitopes, or train models on your own data.

## Quick start

Install MHCflurry and download the pretrained presentation models:

```shell
pip install --pre mhcflurry
mhcflurry downloads fetch models_class1_presentation
```

Predict a few peptides:

```shell
mhcflurry predict \
    --alleles HLA-A0201 HLA-A0301 \
    --peptides SIINFEKL SIINFEKD SIINFEKQ \
    --out predictions.csv
```

Or scan a protein sequence for candidate ligands:

```shell
mhcflurry predict-scan \
    --sequences MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHS \
    --alleles 'HLA-A*02:01' \
    --out scan.csv
```

To try MHCflurry without installing anything, open the
[Colab notebook](https://colab.research.google.com/github/openvax/mhcflurry/blob/master/notebooks/mhcflurry-colab.ipynb).

> [!IMPORTANT]
> This source tree documents `2.3.0rc15`. Use
> `pip install mhcflurry==2.3.0rc15` for that exact version. Omitting `--pre`
> installs the latest stable 2.2.x release until 2.3.0 is final.

The historical `mhcflurry-*` command names remain supported for existing
scripts. See the [2.3.0 release notes](RELEASE_NOTES_2.3.0.md) for details.

## Documentation

- [Introduction and installation](https://openvax.github.io/mhcflurry/intro.html)
- [Command-line tutorial](https://openvax.github.io/mhcflurry/commandline_tutorial.html)
- [Python tutorial](https://openvax.github.io/mhcflurry/python_tutorial.html)
- [Training models](https://openvax.github.io/mhcflurry/training.html)
- [Command reference](https://openvax.github.io/mhcflurry/commandline_tools.html)
- [API reference](https://openvax.github.io/mhcflurry/api.html)

Please [file an issue](https://github.com/openvax/mhcflurry/issues) if you have
questions or encounter problems.

## Citing MHCflurry

If you use MHCflurry in your research, please cite:

> T. O'Donnell, A. Rubinsteyn, U. Laserson. "MHCflurry 2.0: Improved
> pan-allele prediction of MHC I-presented peptides by incorporating antigen
> processing," *Cell Systems*, 2020.
> <https://doi.org/10.1016/j.cels.2020.06.010>

> T. O'Donnell, A. Rubinsteyn, M. Bonsack, A. B. Riemer, U. Laserson, and
> J. Hammerbacher, "MHCflurry: Open-Source Class I MHC Binding Affinity
> Prediction," *Cell Systems*, 2018.
> <https://doi.org/10.1016/j.cels.2018.05.014>

## Development

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md); the
[testing guide](https://openvax.github.io/mhcflurry/testing.html) describes the
fast local checks and full suite.

## Docker

Run the latest image from Docker Hub:

```shell
docker run -p 9999:9999 --rm openvax/mhcflurry:latest
```

Then open `http://localhost:9999` to use the included Jupyter environment. To
build the image from a checkout:

```shell
docker build -t mhcflurry:latest .
docker run -p 9999:9999 --rm mhcflurry:latest
```

## More resources

- [Predicted binding motifs](https://openvax.github.io/mhcflurry-motifs/)
- [Manual download instructions](https://openvax.github.io/mhcflurry/commandline_tutorial.html#downloading-models)
