[![Build Status](https://github.com/openvax/mhcflurry/actions/workflows/ci.yml/badge.svg)](https://github.com/openvax/mhcflurry/actions/workflows/ci.yml)
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

Install MHCflurry 2.3 and download the pretrained
presentation models:

```shell
pip install --upgrade "mhcflurry>=2.3,<2.4"
mhcflurry downloads fetch models_class1_presentation
```

The presentation bundle includes affinity and processing components. Browse
current and historical weights with `mhcflurry downloads info`; select a weight
release for prediction with `--model-release`.

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

The Docker image includes the full presentation weights, the command-line
tools and Jupyter notebooks. It runs predictions on the CPU and supports
Intel/AMD and ARM Linux. Check the image's version with:

```shell
docker pull openvax/mhcflurry:latest
docker run --rm openvax/mhcflurry:latest mhcflurry --version
```

Stable releases publish a matching version tag and update `latest` after
both architecture builds pass an offline prediction check. Publication status
is visible in the [Docker workflow](https://github.com/openvax/mhcflurry/actions/workflows/docker.yml);
an incomplete publication can leave `latest` on the previous version.

Run predictions against files in the current directory:

```shell
docker run --rm -v "$PWD:/work" openvax/mhcflurry:latest \
    mhcflurry predict input.csv --out predictions.csv
```

To start Jupyter, run the image without a command:

```shell
docker run --rm -p 127.0.0.1:9999:9999 -v "$PWD:/work" openvax/mhcflurry:latest
```

Open the localhost URL printed in the logs, including its access token.
Without the volume mount, the image starts in a directory containing the
example notebooks. Mounted directories must be writable by the container's
user (UID 1000). To build from a checkout, use
`docker build -t mhcflurry:local .`. For CUDA training, a separate image can be
built with `docker build -f docker/Dockerfile.train -t mhcflurry:train .`.

## More resources

- [Predicted binding motifs](https://openvax.github.io/mhcflurry-motifs/)
- [Manual download instructions](https://openvax.github.io/mhcflurry/commandline_tutorial.html#downloading-models)
