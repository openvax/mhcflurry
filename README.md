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

## Getting Started: Train Neural Network Models

```
scripts/train-class1-allele-specific-models.py
```

