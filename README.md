[![Build Status](https://travis-ci.org/hammerlab/mhcflurry.svg?branch=master)](https://travis-ci.org/hammerlab/mhcflurry) [![Coverage Status](https://coveralls.io/repos/github/hammerlab/mhcflurry/badge.svg?branch=master)](https://coveralls.io/github/hammerlab/mhcflurry?branch=master)

# mhcflurry
Open source neural network models for peptide-MHC binding affinity prediction

The [adaptive immune system](https://en.wikipedia.org/wiki/Adaptive_immune_system) depends on the presentation of protein fragments by [MHC](https://en.wikipedia.org/wiki/Major_histocompatibility_complex) molecules. Machine learning models of this interaction are used in studies of infectious diseases, autoimmune diseases, vaccine development, and cancer immunotherapy.

MHCflurry currently supports allele-specific peptide / [MHC class I](https://en.wikipedia.org/wiki/MHC_class_I) affinity prediction using two approaches:

 * Ensembles of predictors trained on random halves of the training data (the default)
 * Single-model predictors for each allele trained on all data

For both kinds of predictors, you can fit models to your own data or download
trained models that we provide.

The downloadable models were trained on data from
[IEDB](http://www.iedb.org/home_v3.php) and [Kim 2014](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-241).
The ensemble predictors include models trained on data that has been
augmented with values imputed from other alleles (see
[Rubinsteyn 2016](http://biorxiv.org/content/early/2016/06/07/054775)).

In validation experiments using presented peptides identified by mass-spec,
the ensemble models perform best. We are working on a performance comparison of
these models with other predictors such as netMHCpan, which we hope to make
available soon.

We anticipate adding additional models, including pan-allele and class II predictors.


## Setup

The MHCflurry predictors are implemented in Python using [keras](https://keras.io).
To configure keras you'll need to set an environment variable in your shell:

```
export KERAS_BACKEND=theano
```

If you're familiar with keras, you may also try using the tensorflow backend. MHCflurry is currently tested using theano, however.
 

Now install the package:

```
pip install mhcflurry
```

Then download our datasets and trained models:

```
mhcflurry-downloads fetch
```

From a checkout you can run the unit tests with:

```
nosetests .
```

## Making predictions from the command-line

```shell
$ mhcflurry-predict --alleles HLA-A0201 HLA-A0301 --peptides SIINFEKL SIINFEKD SIINFEKQ
Predicting for 2 alleles and 3 peptides = 6 predictions
allele,peptide,mhcflurry_prediction
HLA-A0201,SIINFEKL,10672.34765625
HLA-A0201,SIINFEKD,26042.716796875
HLA-A0201,SIINFEKQ,26375.794921875
HLA-A0301,SIINFEKL,25532.703125
HLA-A0301,SIINFEKD,24997.876953125
HLA-A0301,SIINFEKQ,28262.828125
```

You can also specify the input and output as CSV files. Run `mhcflurry-predict -h` for details.


## Making predictions from Python

```python
from mhcflurry import predict
predict(alleles=['A0201'], peptides=['SIINFEKL'])
```

```
  Allele   Peptide  Prediction
0  A0201  SIINFEKL  10672.347656
```

The predictions returned by `predict` are affinities (KD) in nM.

## Training your own models

See the [class1_allele_specific_models.ipynb](https://github.com/hammerlab/mhcflurry/blob/master/examples/class1_allele_specific_models.ipynb) notebook for an overview of the Python API, including predicting, fitting, and scoring single-model predictors. There is also a script called `mhcflurry-class1-allele-specific-cv-and-train` that will perform cross validation and model selection given a CSV file of training data. Try `mhcflurry-class1-allele-specific-cv-and-train --help` for details.

The ensemble predictors are trained similarly using the `mhcflurry-class1-allele-specific-ensemble-train` command.

## Details on the downloadable models

The scripts we use to train predictors, including hyperparameter selection
using cross validation, are
[here](downloads-generation/models_class1_allele_specific_ensemble)
for the ensemble predictors and [here](downloads-generation/models_class1_allele_specific_single)
for the single-model predictors.

For the ensemble predictors, we also generate a [report](http://htmlpreview.github.io/?https://github.com/hammerlab/mhcflurry/blob/master/downloads-generation/models_class1_allele_specific_ensemble/models-summary/report.html)
that describes the hyperparameters selected and the test performance of each
model.

Besides the model weights, the data downloaded when you run
`mhcflurry-downloads  fetch` also includes a CSV file giving the
hyperparameters used for each predictor. Run `mhcflurry-downloads path
models_class1_allele_specific_ensemble` or `mhcflurry-downloads path
models_class1_allele_specific_single` to get the directory where these files are stored.

## Problems and Solutions

###  undefined symbol
If you get an error like:

```
ImportError: _CVXcanon.cpython-35m-x86_64-linux-gnu.so: undefined symbol: _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev
```

Try installing cvxpy using conda instead of pip.


## Environment variables

The path where MHCflurry looks for model weights and data can be set with the `MHCFLURRY_DOWNLOADS_DIR` environment variable. This directory should contain subdirectories like "models_class1_allele_specific_single". Setting this variable overrides the other environment variables described below.

If you only want to change the version of the released data used, you can set `MHCFLURRY_DOWNLOADS_CURRENT_RELEASE`. If you want to change the base directory used for all releases, set `MHCFLURRY_DATA_DIR`.

By default, `MHCFLURRY_DOWNLOADS_DIR` is a platform specific application storage directory, `MHCFLURRY_DOWNLOADS_CURRENT_RELEASE` is the latest release, and `MHCFLURRY_DOWNLOADS_DIR` is set to `$MHCFLURRY_DATA_DIR/$MHCFLURRY_DOWNLOADS_CURRENT_RELEASE`.
