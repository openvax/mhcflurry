import pandas as pd

from .mhc1_binding_predictor import Mhc1BindingPredictor

def predict(alleles, peptides):
    allele_dataframes = []
    for allele in alleles:
        model = Mhc1BindingPredictor(allele=allele)
        df = model.predict_peptides(peptides)
        allele_dataframes.append(df)
    return pd.concat(allele_dataframes)
