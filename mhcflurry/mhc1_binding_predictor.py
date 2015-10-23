# Copyright (c) 2015. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Allele specific MHC Class I binding affinity predictor
"""
from __future__ import (
    print_function,
    division,
    absolute_import,
)
from os import listdir
from os.path import exists, join
from itertools import groupby

import numpy as np
import pandas as pd
from keras.models import model_from_json

from .class1_allele_specific_hyperparameters import MAX_IC50
from .data_helpers import index_encoding, normalize_allele_name
from .paths import CLASS1_MODEL_DIRECTORY
from .common import expand_9mer_peptides

_allele_model_cache = {}


class Mhc1BindingPredictor(object):
    def __init__(
            self,
            allele,
            model_directory=CLASS1_MODEL_DIRECTORY,
            max_ic50=MAX_IC50):
        self.max_ic50 = max_ic50
        if not exists(model_directory) or len(listdir(model_directory)) == 0:
            raise ValueError(
                "No MHC prediction models found in %s" % (model_directory,))
        original_allele_name = allele
        self.allele = normalize_allele_name(allele)
        if self.allele not in _allele_model_cache:
            json_filename = self.allele + ".json"
            json_path = join(model_directory, json_filename)
            if not exists(json_path):
                raise ValueError("Unsupported allele: %s" % (
                    original_allele_name,))

            hdf_filename = self.allele + ".hdf"
            hdf_path = join(model_directory, hdf_filename)

            if not exists(hdf_path):
                raise ValueError("Missing model weights for allele %s" % (
                    original_allele_name,))

            with open(hdf_path, "r") as f:
                self.model = model_from_json(f.read())

            self.model.load_weights(hdf_path)
            _allele_model_cache[self.allele] = self.model
        else:
            self.model = _allele_model_cache[self.allele]

    def __repr__(self):
        return "Mhc1BindingPredictor(allele=%s, model_directory=%s)" % (
            self.allele,
            self.model_directory)

    def __str__(self):
        return repr(self)

    def _log_to_ic50(self, log_value):
        """
        Convert neural network output to IC50 values between 0.0 and
        self.max_ic50 (typically 5000, 20000 or w0)
        """
        return self.max_ic50 ** (1.0 - log_value)

    def _predict_9mer_peptides(self, peptides):
        """
        Predict binding affinity for 9mer peptides
        """
        if any(len(peptide) != 9 for peptide in peptides):
            raise ValueError("Can only predict 9mer peptides")
        X = index_encoding(peptides, peptide_length=9)
        return self.model.predict(X, verbose=False).flatten()

    def _predict_9mer_peptides_ic50(self, peptides):
        log_y = self._predict_9mer_peptides(peptides)
        return self._log_to_ic50(log_y)

    def predict_peptides(self, peptides):
        column_names = [
            "Allele",
            "Peptide",
            "Prediction",
        ]
        results = {}
        for column_name in column_names:
            results[column_name] = []

        for length, group_peptides in groupby(peptides, lambda x: len(x)):
            group_peptides = list(group_peptides)
            expanded_peptides = expand_9mer_peptides(group_peptides, length)
            n_group = len(group_peptides)
            n_expanded = len(expanded_peptides)
            expansion_factor = int(n_expanded / n_group)
            raw_y = self._predict_9mer_peptides(expanded_peptides)
            median_y = np.zeros(n_group)
            # take the median of each group of log(IC50) values
            for i in range(n_group):
                start = i * expansion_factor
                end = (i + 1) * expansion_factor
                median_y[i] = np.median(raw_y[start:end])
            ic50 = self._log_to_ic50(median_y)
            results["Allele"].extend([self.allele] * n_group)
            results["Peptide"].extend(group_peptides)
            results["Prediction"].extend(ic50)
        return pd.DataFrame(results, columns=column_names)
