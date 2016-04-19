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
scripts/download-iedb.sh
scripts/download-peters-2013-dataset.sh
scripts/create-iedb-class1-dataset.py
scripts/create-combined-class1-dataset.py
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
