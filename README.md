[![Build Status](https://github.com/openvax/mhcflurry/actions/workflows/ci.yml/badge.svg)](https://github.com/openvax/mhcflurry/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/openvax/mhcflurry/badge.svg?branch=master)](https://coveralls.io/github/openvax/mhcflurry?branch=master)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvax/mhcflurry/blob/master/notebooks/mhcflurry-colab.ipynb)

# mhcflurry
[MHC I](https://en.wikipedia.org/wiki/MHC_class_I) ligand
prediction package with competitive accuracy and a fast and
[documented](http://openvax.github.io/mhcflurry/) implementation.

> [!IMPORTANT]
> **Version 2.2.0** is the first release to use [PyTorch](https://pytorch.org/) as its neural network backend, replacing TensorFlow/Keras used in previous versions. It loads the same published weights and produces equivalent predictions, so existing workflows should continue to work with no changes.
>
> Key changes in 2.2.0:
> - **Backend**: TensorFlow/Keras replaced by PyTorch (>= 2.0)
> - **Python**: Requires Python 3.10+ (previously 3.9+)
> - **Dependencies**: `pandas >= 2.0` is now required; `tensorflow` and `keras` are no longer needed
> - **Hardware**: Automatic GPU detection; Apple Silicon (MPS) is now supported
>
> If you are upgrading from 2.1.x, simply `pip install --upgrade mhcflurry`. The published pre-trained models are unchanged and will be loaded and converted automatically.

MHCflurry implements class I peptide/MHC binding affinity prediction.
The current version provides pan-MHC I predictors supporting any MHC
allele of known sequence. MHCflurry runs on Python 3.10+ using the
[PyTorch](https://pytorch.org/) neural network library.
It exposes [command-line](http://openvax.github.io/mhcflurry/commandline_tutorial.html)
and [Python library](http://openvax.github.io/mhcflurry/python_tutorial.html)
interfaces.

MHCflurry also includes two experimental predictors,
an "antigen processing" predictor that attempts to model MHC allele-independent
effects such as proteosomal cleavage and a "presentation" predictor that
integrates processing predictions with binding affinity predictions to give a
composite "presentation score." Both models are trained on mass spec-identified
MHC ligands.

If you find MHCflurry useful in your research please cite:

> T. O'Donnell, A. Rubinsteyn, U. Laserson. "MHCflurry 2.0: Improved pan-allele prediction of MHC I-presented peptides by incorporating antigen processing," *Cell Systems*, 2020. https://doi.org/10.1016/j.cels.2020.06.010

> T. O'Donnell, A. Rubinsteyn, M. Bonsack, A. B. Riemer, U. Laserson, and J. Hammerbacher, "MHCflurry: Open-Source Class I MHC Binding Affinity Prediction," *Cell Systems*, 2018. https://doi.org/10.1016/j.cels.2018.05.014

Please file an issue if you have questions or encounter problems.

Have a bugfix or other contribution? We would love your help. See our [contributing guidelines](CONTRIBUTING.md).

## Try it now

You can generate MHCflurry predictions without any setup by running our Google colaboratory [notebook](https://colab.research.google.com/github/openvax/mhcflurry/blob/master/notebooks/mhcflurry-colab.ipynb).

## Installation (pip)

Install the package:

```
$ pip install mhcflurry
```

Download our datasets and trained models:

```
$ mhcflurry-downloads fetch
```

You can now generate predictions:

```
$ mhcflurry-predict \
       --alleles HLA-A0201 HLA-A0301 \
       --peptides SIINFEKL SIINFEKD SIINFEKQ \
       --out /tmp/predictions.csv

Wrote: /tmp/predictions.csv
```

Or scan protein sequences for potential epitopes:

```
$ mhcflurry-predict-scan \
        --sequences MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHS \
        --alleles HLA-A*02:01 \
        --out /tmp/predictions.csv

Wrote: /tmp/predictions.csv
```


See the [documentation](http://openvax.github.io/mhcflurry/) for more details.


## Docker
You can also try the latest (GitHub master) version of MHCflurry using the Docker
image hosted on [Dockerhub](https://hub.docker.com/r/openvax/mhcflurry) by
running:

```
$ docker run -p 9999:9999 --rm openvax/mhcflurry:latest
```

This will start a [jupyter](https://jupyter.org/) notebook server in an
environment that has MHCflurry installed. Go to `http://localhost:9999` in a
browser to use it.

To build the Docker image yourself, from a checkout run:

```
$ docker build -t mhcflurry:latest .
$ docker run -p 9999:9999 --rm mhcflurry:latest
```
## Predicted sequence motifs
Sequence logos for the binding motifs learned by MHCflurry BA are available [here](https://openvax.github.io/mhcflurry-motifs/).

## Common issues and fixes

### Problems downloading data and models
Some users have reported HTTP connection issues when using `mhcflurry-downloads fetch`. As a workaround, you can download the data manually (e.g. using `wget`) and then use `mhcflurry-downloads` just to copy the data to the right place.

To do this, first get the URL(s) of the downloads you need using `mhcflurry-downloads url`:

```
$ mhcflurry-downloads url models_class1_presentation
https://github.com/openvax/mhcflurry/releases/download/1.6.0/models_class1_presentation.20200205.tar.bz2```
```

Then make a directory and download the needed files to this directory:

```
$ mkdir downloads
$ wget  --directory-prefix downloads https://github.com/openvax/mhcflurry/releases/download/1.6.0/models_class1_presentation.20200205.tar.bz2```

HTTP request sent, awaiting response... 200 OK
Length: 72616448 (69M) [application/octet-stream]
Saving to: 'downloads/models_class1_presentation.20200205.tar.bz2'
```

Now call `mhcflurry-downloads fetch` with the `--already-downloaded-dir` option to indicate that the downloads should be retrived from the specified directory:

```
$ mhcflurry-downloads fetch models_class1_presentation --already-downloaded-dir downloads
```
