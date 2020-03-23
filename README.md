[![Build Status](https://travis-ci.org/openvax/mhcflurry.svg?branch=master)](https://travis-ci.org/openvax/mhcflurry)

# mhcflurry
[MHC I](https://en.wikipedia.org/wiki/MHC_class_I) ligand
prediction package with competitive accuracy and a fast and 
[documented](http://openvax.github.io/mhcflurry/) implementation.

MHCflurry implements class I peptide/MHC binding affinity prediction. 
The current version provides pan-MHC I predictors supporting any MHC
allele of known sequence. MHCflurry runs on Python 3.4+ using the
[keras](https://keras.io) neural network library.
It exposes [command-line](http://openvax.github.io/mhcflurry/commandline_tutorial.html)
and [Python library](http://openvax.github.io/mhcflurry/python_tutorial.html)
interfaces.

Starting in version 1.6.0, MHCflurry also includes two expermental predictors,
an "antigen processing" predictor that attempts to model MHC allele-independent
effects such as proteosomal cleavage and a "presentation" predictor that
integrates processing predictions with binding affinity predictions to give a
composite "presentation score." Both models are trained on mass spec-identified
MHC ligands.

If you find MHCflurry useful in your research please cite:

> T. J. O’Donnell, A. Rubinsteyn, M. Bonsack, A. B. Riemer, U. Laserson, and J. Hammerbacher, "MHCflurry: Open-Source Class I MHC Binding Affinity Prediction," *Cell Systems*, 2018. https://www.cell.com/cell-systems/fulltext/S2405-4712(18)30232-1.

Please file an issue if you have questions or encounter problems.

Have a bugfix or other contribution? We would love your help. See our [contributing guidelines](CONTRIBUTING.md).

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

Or scan protein sequences for potential epitopes:

```
$ mhcflurry-predict-scan \
        --sequences MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHS \
        --alleles HLA-A*02:01 \
        --out /tmp/predictions.csv
        
Wrote: /tmp/predictions.csv  
```


See the [documentation](http://openvax.github.io/mhcflurry/) for more details.


## Older allele-specific models

Previous versions of MHCflurry used models trained on affinity measurements, one allele
per model (i.e. allele-specific). Mass spec datasets were incorporated in the
model selection step.

These models are still available to use with the latest version of MHCflurry.
To download these predictors, run:

```
$ mhcflurry-downloads fetch models_class1
```

and specify `--models` when you call `mhcflurry-predict`:

```
$ mhcflurry-predict \
       --alleles HLA-A0201 HLA-A0301 \
       --peptides SIINFEKL SIINFEKD SIINFEKQ \
       --models "$(mhcflurry-downloads path models_class1)/models"
       --out /tmp/predictions.csv
       
Wrote: /tmp/predictions.csv
```


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


