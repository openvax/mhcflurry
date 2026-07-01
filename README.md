[![Build Status](https://github.com/openvax/mhcflurry/actions/workflows/ci.yml/badge.svg)](https://github.com/openvax/mhcflurry/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/openvax/mhcflurry/badge.svg?branch=master)](https://coveralls.io/github/openvax/mhcflurry?branch=master)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvax/mhcflurry/blob/master/notebooks/mhcflurry-colab.ipynb)

# mhcflurry
[MHC I](https://en.wikipedia.org/wiki/MHC_class_I) ligand
prediction package with competitive accuracy and a fast and
[documented](http://openvax.github.io/mhcflurry/) implementation.

If you find MHCflurry useful in your research please cite:

> T. O'Donnell, A. Rubinsteyn, U. Laserson. "MHCflurry 2.0: Improved pan-allele prediction of MHC I-presented peptides by incorporating antigen processing," *Cell Systems*, 2020. https://doi.org/10.1016/j.cels.2020.06.010

> T. O'Donnell, A. Rubinsteyn, M. Bonsack, A. B. Riemer, U. Laserson, and J. Hammerbacher, "MHCflurry: Open-Source Class I MHC Binding Affinity Prediction," *Cell Systems*, 2018. https://doi.org/10.1016/j.cels.2018.05.014

Please file an issue if you have questions or encounter problems.

Have a bugfix or other contribution? We would love your help. See our [contributing guidelines](CONTRIBUTING.md).

## 2.3.0 release candidate

> [!IMPORTANT]
> 2.3.0 is currently a release candidate (`2.3.0rc10`), not yet a final release.
> It keeps the same public API and pre-trained models as 2.2.x. Install it with
> `pip install --pre mhcflurry`, or pin the version with
> `pip install mhcflurry==2.3.0rc10`. A plain `pip install --upgrade mhcflurry`
> stays on the latest stable release (2.2.x) until 2.3.0 is final, since pip
> skips pre-releases.

2.3.0 adds speed and tooling for people who train their own models or run large
prediction jobs:

- Training keeps data on the GPU for the whole fit, avoiding per-batch host/device copies.
- `mhcflurry-predict`, `mhcflurry-predict-scan`, and `mhcflurry-calibrate-percentile-ranks` use all visible GPUs by default.
- `mhcflurry-class1-train-pan-allele-models` auto-tunes job and worker counts from the hardware, so the same command runs on a laptop, a single GPU, or an 8×A100 host.
- `torch.compile` and matmul precision (including TF32) are available as flags on the training commands.

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
        --alleles 'HLA-A*02:01' \
        --out /tmp/predictions.csv

Wrote: /tmp/predictions.csv
```

### Unified `mhcflurry` parent command

Starting in 2.3.0 there is also a single `mhcflurry` command that dispatches
to every subcommand:

```
$ mhcflurry predict \
        --alleles HLA-A0201 HLA-A0301 \
        --peptides SIINFEKL SIINFEKD SIINFEKQ \
        --out /tmp/predictions.csv
```

Every historical command is reachable as a subcommand
(`mhcflurry-predict` ↔ `mhcflurry predict`, `mhcflurry-downloads` ↔
`mhcflurry downloads`, `mhcflurry-class1-train-pan-allele-models` ↔
`mhcflurry class1-train-pan-allele-models`, etc.). Both forms run the
same underlying entry point; the legacy `mhcflurry-*` scripts remain
installed as compat shims and are not changing. `mhcflurry --help`
lists every available subcommand.

See the [documentation](http://openvax.github.io/mhcflurry/) for more details.

## Development and tests

From a checkout, source `develop.sh` to create and activate the editable
environment:

```
$ source develop.sh
```

For quick feedback, run lint plus a focused unit subset:

```
$ ./lint.sh
$ pytest -q test/test_amino_acid.py test/test_random_negative_peptides.py
```

`pytest test/` is the full test suite, not a fast unit-only loop. It includes
small end-to-end training runs, command subprocess tests, public-model smoke
tests that require cached MHCflurry download bundles, and speed/regression
checks, so it can take many minutes. Use
`pytest -q test -m "not slow and not downloads"` for the broad fast tier, and
`pytest -q test --durations=25` when auditing slow tests. See the
[testing documentation](http://openvax.github.io/mhcflurry/testing.html) for
the current test tiers.

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
