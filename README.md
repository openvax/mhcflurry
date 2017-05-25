[![Build Status](https://travis-ci.org/hammerlab/mhcflurry.svg?branch=master)](https://travis-ci.org/hammerlab/mhcflurry) [![Coverage Status](https://coveralls.io/repos/github/hammerlab/mhcflurry/badge.svg?branch=master)](https://coveralls.io/github/hammerlab/mhcflurry?branch=master)

# mhcflurry
Open source neural network models for peptide-MHC binding affinity prediction

The [adaptive immune system](https://en.wikipedia.org/wiki/Adaptive_immune_system)
depends on the presentation of protein fragments by [MHC](https://en.wikipedia.org/wiki/Major_histocompatibility_complex)
molecules. Machine learning models of this interaction are used in studies of
infectious diseases, autoimmune diseases, vaccine development, and cancer
immunotherapy.

MHCflurry supports Class I peptide/MHC binding affinity prediction using
ensembles of allele-specific models. You can fit MHCflurry models to your own data or download models that we fit to data from
[IEDB](http://www.iedb.org/home_v3.php) and [Kim 2014](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-241).
Our combined dataset is available for download [here](https://github.com/hammerlab/mhcflurry/releases/download/pre-1.0.0-alpha/data_curated.tar.bz2).

We are working on a performance comparison of these models with other predictors
such as netMHCpan, which we plan to make available soon.

Pan-allelic prediction is supported in principle but is not yet performing
accurately. Infrastructure for modeling other aspects of antigen
processing is also implemented but experimental.


## Setup

Install the package:

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

The MHCflurry predictors are implemented in Python using [keras](https://keras.io).

MHCflurry works with both the tensorflow and theano keras backends. The
tensorflow backend gives faster model-loading time but is undergoing more
rapid development and sometimes hits issues. If you encounter tensorflow errors
running MHCflurry, try setting this environment variable to switch to the theano
backend:

```
export KERAS_BACKEND=theano
```

You may also needs to `pip install theano`.


## Making predictions from the command-line

```shell
$ mhcflurry-predict --alleles HLA-A0201 HLA-A0301 --peptides SIINFEKL SIINFEKD SIINFEKQ
allele,peptide,mhcflurry_prediction,mhcflurry_prediction_low,mhcflurry_prediction_high
HLA-A0201,SIINFEKL,6029.079749556217,4474.10333152741,7771.2922076773575
HLA-A0201,SIINFEKD,18950.310303704624,15317.127851792027,22490.05728778504
HLA-A0201,SIINFEKQ,18776.978315260818,14899.359763218705,22314.737180384865
HLA-A0301,SIINFEKL,25589.66470369661,22962.4956808368,29395.86949262485
HLA-A0301,SIINFEKD,25753.619337400796,22851.89399578629,29347.659901990868
HLA-A0301,SIINFEKQ,26870.51318688641,24198.39885651102,30364.15208364084
```

The predictions returned are affinities (KD) in nM. The `prediction_low` and
`prediction_high` fields give the 5-95 percentile predictions across the models 
in the ensemble.

You can also specify the input and output as CSV files.
Run `mhcflurry-predict -h` for details.


## Making predictions from Python

```python
>>> from mhcflurry import Class1AffinityPredictor
>>> predictor = Class1AffinityPredictor.load()
>>> predictor.predict_to_dataframe(peptides=['SIINFEKL'], allele='A0201')


  allele   peptide   prediction  prediction_low  prediction_high
  A0201  SIINFEKL  6029.084473     4474.103253      7771.297702
```

See the [class1_allele_specific_models.ipynb](https://github.com/hammerlab/mhcflurry/blob/master/examples/class1_allele_specific_models.ipynb)
notebook for an overview of the Python API, including fitting your own predictors.


## Details on the downloadable models

An ensemble of eight single-allele models was trained for each allele with at least
100 measurements in the training set (118 alleles). The models were trained on a
random 80% sample of the data for the allele and the remaining 20% was used for
early stopping. All models use the same [architecture](downloads-generation/models_class1/hyperparameters.json). The
predictions are taken to be the geometric mean of the nM binding affinity
predictions of the individual models. The training script is [here](downloads-generation/models_class1/GENERATE.sh).

## Environment variables

The path where MHCflurry looks for model weights and data can be set with the `MHCFLURRY_DOWNLOADS_DIR` environment variable. This directory should contain subdirectories like "models_class1".