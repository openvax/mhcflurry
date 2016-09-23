[![Build Status](https://travis-ci.org/hammerlab/mhcflurry.svg?branch=master)](https://travis-ci.org/hammerlab/mhcflurry) [![Coverage Status](https://coveralls.io/repos/github/hammerlab/mhcflurry/badge.svg?branch=master)](https://coveralls.io/github/hammerlab/mhcflurry?branch=master)

# mhcflurry
Open source neural network models for peptide-MHC binding affinity prediction

The [adaptive immune system](https://en.wikipedia.org/wiki/Adaptive_immune_system) depends on the presentation of protein fragments by [MHC](https://en.wikipedia.org/wiki/Major_histocompatibility_complex) molecules. Machine learning models of this interaction are used in studies of infectious diseases, autoimmune diseases, vaccine development, and cancer immunotherapy.

MHCflurry currently supports peptide / [MHC class I](https://en.wikipedia.org/wiki/MHC_class_I) affinity prediction using one model per MHC allele. The predictors may be trained on data that has been augmented with data imputed based on other alleles (see [Rubinsteyn 2016](http://biorxiv.org/content/early/2016/06/07/054775)). We anticipate adding additional models, including pan-allele and class II predictors.

You can fit MHCflurry models to your own data or download trained models that we provide. Our models are trained on data from [IEDB](http://www.iedb.org/home_v3.php) and [Kim 2014](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-241). See [here](https://github.com/hammerlab/mhcflurry/tree/master/downloads-generation/data_combined_iedb_kim2014) for details on the training data preparation. The steps we use to train predictors on this data, including hyperparameter selection using cross validation, are [here](https://github.com/hammerlab/mhcflurry/tree/master/downloads-generation/models_class1_allele_specific_single).

The MHCflurry predictors are implemented in Python using [keras](https://keras.io).

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

This [unit test](https://github.com/hammerlab/mhcflurry/blob/master/test/test_class1_binding_predictor_A0205.py) gives a simple example of how to train a predictor in Python. There is also a script called `mhcflurry-class1-allele-specific-cv-and-train` that will perform cross validation and model selection given a CSV file of training data. Try `mhcflurry-class1-allele-specific-cv-and-train --help` for details.

## Details on the downloaded class I allele-specific models

Besides the actual model weights, the data downloaded with `mhcflurry-downloads fetch` also includes a CSV file giving the hyperparameters used for each predictor. Another CSV gives the cross validation results used to select these hyperparameters.

To see the hyperparameters for the production models, run:

```
open "$(mhcflurry-downloads path models_class1_allele_specific_single)/production.csv"
```

To see the cross validation results:

```
open "$(mhcflurry-downloads path models_class1_allele_specific_single)/cv.csv"
```

## Environment variables

The path where MHCflurry looks for model weights and data can be set with the `MHCFLURRY_DOWNLOADS_DIR` environment variable. This directory should contain subdirectories like "models_class1_allele_specific_single". Setting this variable overrides the other environment variables described below.

If you only want to change the version of the released data used, you can set `MHCFLURRY_DOWNLOADS_CURRENT_RELEASE`. If you want to change the base directory used for all releases, set `MHCFLURRY_DATA_DIR`.

By default, `MHCFLURRY_DOWNLOADS_DIR` is a platform specific application storage directory, `MHCFLURRY_DOWNLOADS_CURRENT_RELEASE` is the latest release, and `MHCFLURRY_DOWNLOADS_DIR` is set to `$MHCFLURRY_DATA_DIR/$MHCFLURRY_DOWNLOADS_CURRENT_RELEASE`.
