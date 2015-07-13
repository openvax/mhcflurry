import pandas as pd
import os

from .paths import CLASS1_MODEL_DIRECTORY
from .mhc1_binding_predictor import Mhc1BindingPredictor

def predict(alleles, peptides):
    allele_dataframes = []
    for allele in alleles:
        model = Mhc1BindingPredictor(allele=allele)
        df = model.predict_peptides(peptides)
        allele_dataframes.append(df)
    return pd.concat(allele_dataframes)

def supported_alleles():
    alleles = []
    for filename in os.listdir(CLASS1_MODEL_DIRECTORY):
        allele = filename.replace(".hdf", "")
        if len(allele) < 5:
            # skipping serotype names like A2 or B7
            continue
        allele = "HLA-%s*%s:%s" % (allele[0], allele[1:3], allele[3:])
        alleles.append(allele)
    return alleles
