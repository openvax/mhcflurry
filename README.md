# mhcflurry
Peptide-MHC binding affinity prediction

## Getting Started: Download, Normalize, and Combine Training Data

```
scripts/download-iedb.sh
scripts/download-peters-2013-dataset.sh
python scripts/create-iedb-class1-dataset.py
python scripts/create-combined-class1-dataset.py
```