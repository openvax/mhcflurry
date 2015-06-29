"""
Allele specific MHC Class I binding affinity predictor
"""

from os.path import exists, join
from itertools import groupby
import pickle

import pandas as pd

from .feedforward import make_network
from .class1_allele_specific_hyperparameters import (
    EMBEDDING_DIM,
    HIDDEN_LAYER_SIZE,
    ACTIVATION,
    INITIALIZATION_METHOD,
    DROPOUT_PROBABILITY,
)
from .data_helpers import index_encoding, normalize_allele_name

_allele_model_cache = {}

class Mhc1BindingPredictor(object):
    def __init__(
            self,
            allele,
            model_directory="class1_predictors"):
        original_allele_name = allele
        self.allele = normalize_allele_name(allele)
        if self.allele in _allele_model_cache:
            self.model = _allele_model_cache[self.allele]
        else:
            filename = self.allele + ".hdf"
            path = join(model_directory, filename)
            if not exists(path):
                raise ValueError("Unsupported allele: %s" % (
                    original_allele_name,))
            self.model = make_network(
                input_size=9,
                embedding_input_dim=20,
                embedding_output_dim=EMBEDDING_DIM,
                layer_sizes=(HIDDEN_LAYER_SIZE,),
                activation=ACTIVATION,
                init=INITIALIZATION_METHOD,
                dropout_probability=DROPOUT_PROBABILITY)
            pickle.dumps(self.model)
            self.model.load_weights(path)
            _allele_model_cache[self.allele] = self.model

    def predict_peptides(self, peptides):
        column_names = [
            "allele",
            "peptide",
            "ic50",
        ]
        results = {}
        for column_name in column_names:
            results[column_name] = []

        for length, group in groupby(peptides, lambda x: len(x)):
            group_list = list(group)
            if length != 9:
                raise ValueError(
                    "Invalid peptide length %d: %s" % (
                        length, group_list[0]))
            X = index_encoding(group_list, peptide_length=9)
            y = self.model.predict(X)
            ic50 = 5000.0 ** (1.0 - y)
            results["allele"].extend([self.allele] * len(X))
            results["peptide"].extend(group_list)
            results["ic50"].extend(ic50.flatten())
        return pd.DataFrame(results, columns=column_names)
