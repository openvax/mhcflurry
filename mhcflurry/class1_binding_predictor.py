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
import logging
from os import listdir, remove
from os.path import exists, join
from itertools import groupby
import json
from collections import OrderedDict

import numpy as np
from keras.models import model_from_config

from .class1_allele_specific_hyperparameters import MAX_IC50
from .common import normalize_allele_name
from .peptide_encoding import index_encoding
from .paths import CLASS1_MODEL_DIRECTORY
from .fixed_length_peptides import fixed_length_from_many_peptides

_allele_predictor_cache = {}


class Class1BindingPredictor(object):
    def __init__(self, model, name=None, max_ic50=MAX_IC50):
        self.model = model
        self.max_ic50 = max_ic50
        self.name = name

    @classmethod
    def from_disk(
            cls,
            model_json_path,
            weights_hdf_path=None,
            name=None,
            max_ic50=MAX_IC50):
        """
        Load model from stored JSON representation of network and
        (optionally) load weights from HDF5 file.
        """
        if not exists(model_json_path):
            raise ValueError("Model file %s (name = %s) not found" % (
                model_json_path, name,))

        with open(model_json_path, "r") as f:
            config_dict = json.load(f)

        model = model_from_config(config_dict)

        if weights_hdf_path:
            if not exists(weights_hdf_path):
                raise ValueError(
                    "Missing model weights file %s (name = %s)" % (
                        weights_hdf_path, name))

            model.load_weights(weights_hdf_path)

        return cls.__init__(
            model=model,
            max_ic50=max_ic50,
            name=name)

    """
    def fit(self, affinity_dict, synthetic_data=None):
        model.set_weights(old_weights)
        model.fit(
            allele_data.X,
            allele_data.Y,
            nb_epoch=N_EPOCHS,
            show_accuracy=True)
    """

    def to_disk(self, model_json_path, weights_hdf_path, overwrite=False):
        if exists(model_json_path) and overwrite:
            logging.info(
                "Removing existing model JSON file '%s'" % (
                    model_json_path,))
            remove(model_json_path)

        if exists(model_json_path):
            logging.warn(
                "Model JSON file '%s' already exists" % (model_json_path,))
        else:
            logging.info(
                "Saving model file %s (name=%s)" % (model_json_path, self.name))
            with open(model_json_path, "w") as f:
                f.write(self.model.to_json())

        if exists(weights_hdf_path) and overwrite:
            logging.info(
                "Removing existing model weights HDF5 file '%s'" % (
                    weights_hdf_path,))
            remove(weights_hdf_path)

        if exists(weights_hdf_path):
            logging.warn(
                "Model weights HDF5 file '%s' already exists" % (
                    weights_hdf_path,))
        else:
            logging.info(
                "Saving model weights HDF5 file %s (name=%s)" % (
                    weights_hdf_path, self.name))
            self.model.save_weights(weights_hdf_path)

    @classmethod
    def from_allele_name(
            cls,
            allele_name,
            model_directory=CLASS1_MODEL_DIRECTORY,
            max_ic50=MAX_IC50):
        if not exists(model_directory) or len(listdir(model_directory)) == 0:
            raise ValueError(
                "No MHC prediction models found in %s" % (model_directory,))

        allele_name = normalize_allele_name(allele_name)
        key = (allele_name, model_directory, max_ic50)
        if key in _allele_predictor_cache:
            return _allele_predictor_cache[key]

        if not exists(model_directory) or len(listdir(model_directory)) == 0:
            raise ValueError(
                "No MHC prediction models found in %s" % (model_directory,))

        model_json_filename = allele_name + ".json"
        model_json_path = join(model_directory, model_json_filename)

        weights_hdf_filename = allele_name + ".hdf"
        weights_hdf_path = join(model_directory, weights_hdf_filename)

        predictor = cls.from_disk(
            model_json_path=model_json_path,
            weights_hdf_path=weights_hdf_path,
            name=allele_name,
            max_ic50=max_ic50)

        _allele_predictor_cache[key] = predictor
        return predictor

    @classmethod
    def supported_alleles(cls, model_directory=CLASS1_MODEL_DIRECTORY):
        alleles_with_weights = set([])
        alleles_with_models = set([])
        for filename in listdir(model_directory):
            if filename.endswith(".hdf"):
                alleles_with_weights.add(filename.replace(".hdf", ""))
            elif filename.endswith(".json"):
                alleles_with_models.add(filename.replace(".json", ""))
        alleles = alleles_with_models.intersection(alleles_with_weights)
        return list(sorted(alleles))

    def __repr__(self):
        return "Class1BindingPredictor(allele=%s, model=%s, max_ic50=%f)" % (
            self.allele,
            self.model,
            self.max_ic50)

    def __str__(self):
        return repr(self)

    def _log_to_ic50(self, log_value):
        """
        Convert neural network output to IC50 values between 0.0 and
        self.max_ic50 (typically 5000, 20000 or 50000)
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
        """
        Given a list of peptides, returns a dictionary mapping
        """
        results = OrderedDict([])
        for length, group_peptides in groupby(peptides, lambda x: len(x)):
            expanded_peptides, _, _ = fixed_length_from_many_peptides(
                peptides=list(group_peptides),
                desired_length=length)
            n_group = len(group_peptides)
            n_expanded = len(expanded_peptides)
            expansion_factor = int(n_expanded / n_group)
            raw_y = self._predict_9mer_peptides(expanded_peptides)
            if expansion_factor == 1:
                ic50s = self._log_to_ic50(raw_y)
            else:
                # if peptides were a different length than the predictor's
                # expected input length, then let's take the median prediction
                # of each expanded peptide set
                median_y = np.zeros(n_group)
                # take the median of each group of log(IC50) values
                for i in range(n_group):
                    start = i * expansion_factor
                    end = (i + 1) * expansion_factor
                    median_y[i] = np.median(raw_y[start:end])
                ic50s = self._log_to_ic50(median_y)
            assert len(group_peptides) == len(ic50s)
            for peptide, ic50 in zip(group_peptides, ic50s):
                results[peptide] = ic50
        return results
