[![Build Status](https://travis-ci.org/openvax/mhcflurry.svg?branch=master)](https://travis-ci.org/openvax/mhcflurry)

# mhcflurry
[MHC I](https://en.wikipedia.org/wiki/MHC_class_I) ligand
prediction package with competitive accuracy and a fast and 
[documented](http://openvax.github.io/mhcflurry/) implementation.

MHCflurry implements class I peptide/MHC binding affinity prediction. By default
it supports 112 MHC alleles using ensembles of allele-specific models.
Pan-allele predictors supporting virtually any MHC allele of known sequence
are available for testing (see below). MHCflurry runs on Python 2.7 and 3.4+ using the
[keras](https://keras.io) neural network library.
It exposes [command-line](http://openvax.github.io/mhcflurry/commandline_tutorial.html)
and [Python library](http://openvax.github.io/mhcflurry/python_tutorial.html)
interfaces.

If you find MHCflurry useful in your research please cite:

> T. J. O’Donnell, A. Rubinsteyn, M. Bonsack, A. B. Riemer, U. Laserson, and J. Hammerbacher, "MHCflurry: Open-Source Class I MHC Binding Affinity Prediction," *Cell Systems*, 2018. Available at: https://www.cell.com/cell-systems/fulltext/S2405-4712(18)30232-1.

Have a bugfix or other contribution? We would love your help. See our [contributing guidelines](CONTRIBUTING.md) for more information.

## Installation (pip)

Install the package:

```
$ pip install mhcflurry
```

Then download our datasets and trained models:

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

See the [documentation](http://openvax.github.io/mhcflurry/) for more details.

## Pan-allele models (experimental)

We are testing new models that support prediction for any MHC I allele of known
sequence (as opposed to the 112 alleles supported by the allele-specific
predictors). These models are trained on both affinity measurements and mass spec.

To try the pan-allele models, first download them:

```
$ mhcflurry-downloads fetch models_class1_pan
```

then set this environment variable to use them by default:

```
$ export MHCFLURRY_DEFAULT_CLASS1_MODELS="$(mhcflurry-downloads path models_class1_pan)/models.with_mass_spec"
```

You can now generate predictions for about 14,000 MHC I alleles. For example:

```
$ mhcflurry-predict --alleles HLA-A*02:04 --peptides SIINFEKL
```

If you use these models please let us know how it goes.


## Other allele-specific models

The default MHCflurry models are trained on affinity measurements, one allele
per model (i.e. allele-specific). Mass spec datasets are incorporated in the
model selection step.

We also release experimental allele-specific predictors whose training data
directly includes mass spec. To download these predictors, run:

```
$ mhcflurry-downloads fetch models_class1_trained_with_mass_spec
```

and then to make them used by default:

```
$ export MHCFLURRY_DEFAULT_CLASS1_MODELS="$(mhcflurry-downloads path models_class1_trained_with_mass_spec)/models"
```

We also release predictors that do not use mass spec datasets at all. To use
these predictors, run:

```
$ mhcflurry-downloads fetch models_class1_selected_no_mass_spec
export MHCFLURRY_DEFAULT_CLASS1_MODELS="$(mhcflurry-downloads path models_class1_selected_no_mass_spec)/models"
```

## Common issues and fixes

### Problems downloading data and models
Some users have reported HTTP connection issues when using `mhcflurry-downloads fetch`. As a workaround, you can download the data manually (e.g. using `wget`) and then use `mhcflurry-downloads` just to copy the data to the right place.

To do this, first get the URL(s) of the downloads you need using `mhcflurry-downloads url`:

```
$ mhcflurry-downloads url models_class1
http://github.com/openvax/mhcflurry/releases/download/pre-1.2/models_class1.20180225.tar.bz2
```

Then make a directory and download the needed files to this directory:

```
$ mkdir downloads
$ wget  --directory-prefix downloads http://github.com/openvax/mhcflurry/releases/download/pre-1.2/models_class1.20180225.tar.bz2 
HTTP request sent, awaiting response... 200 OK
Length: 72616448 (69M) [application/octet-stream]
Saving to: 'downloads/models_class1.20180225.tar.bz2'
```

Now call `mhcflurry-downloads fetch` with the `--already-downloaded-dir` option to indicate that the downloads should be retrived from the specified directory:

```
$ mhcflurry-downloads fetch models_class1 --already-downloaded-dir downloads
```

### Problems deserializing models
If you encounter errors loading the MHCflurry models, such as:

```
...
  File "/usr/local/lib/python3.6/site-packages/keras/engine/topology.py", line 293, in __init__
    raise TypeError('Keyword argument not understood:', kwarg)
TypeError: ('Keyword argument not understood:', 'data_format')
```

You may need to upgrade Keras:

```
pip install --upgrade Keras
```


