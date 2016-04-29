[![Build Status](https://travis-ci.org/hammerlab/mhcflurry.svg?branch=master)](https://travis-ci.org/hammerlab/mhcflurry) [![Coverage Status](https://coveralls.io/repos/github/hammerlab/mhcflurry/badge.svg?branch=master)](https://coveralls.io/github/hammerlab/mhcflurry?branch=master)

# mhcflurry
Peptide-MHC binding affinity prediction

## Quickstart

Set up the Python environment:

```
# (set up environment)
pip install scipy Cython
pip install h5py
python setup.py develop
```

Download, Normalize, and Combine Training Data:

(make sure you have `wget` available, e.g. `brew install wget` on Mac OS X)

```
script/download-iedb.sh
script/download-peters-2013-dataset.sh
script/create-iedb-class1-dataset.py
script/create-combined-class1-dataset.py
```

## Train Neural Network Models

```
mhcflurry-train-class1-allele-specific-models.py
```

This will train separate models for each HLA type.

## Making predictions

```python
from mhcflurry import predict
predict(alleles=['A0201'], peptides=['SIINFEKL'])
```

```
  Allele   Peptide  Prediction
0  A0201  SIINFEKL  586.730529
```
