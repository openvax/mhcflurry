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
>>> from mhcflurry import Class1AffinityPredictor
>>> predictor = Class1AffinityPredictor.load()
>>> predictor.predict_to_dataframe(peptides=['SIINFEKL'], allele='A0201')


  allele   peptide   prediction  prediction_low  prediction_high
  A0201  SIINFEKL  6029.084473     4474.103253      7771.297702
```

The predictions returned are affinities (KD) in nM. The `prediction_low` and
`prediction_high` fields give the 5-95 percentile predictions across the models 
in the ensemble.

## Training your own models

See the [class1_allele_specific_models.ipynb](https://github.com/hammerlab/mhcflurry/blob/master/examples/class1_allele_specific_models.ipynb)
notebook for an overview of the Python API.


## Details on the downloadable models

An ensemble of eight single-allele models was trained for each allele with at least
100 measurements in the training set (118 alleles). The models were trained on a
random 80% sample of the data for the allele and the remaining 20% was used for
early stopping. All models use the same [architecture](downloads-generation/models_class1/hyperparameters.json). The
predictions are taken to be the geometric mean of the nM binding affinity
predictions of the individual models. The training script is [here](downloads-generation/models_class1/GENERATE.sh).


## Problems and Solutions

###  undefined symbol
If you get an error like:

```
ImportError: _CVXcanon.cpython-35m-x86_64-linux-gnu.so: undefined symbol: _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev
```

Try installing cvxpy using conda instead of pip.


## Environment variables

The path where MHCflurry looks for model weights and data can be set with the `MHCFLURRY_DOWNLOADS_DIR` environment variable. This directory should contain subdirectories like "models_class1".