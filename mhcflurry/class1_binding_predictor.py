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

import numpy as np
from keras.models import model_from_config

from .class1_allele_specific_hyperparameters import MAX_IC50
from .common import normalize_allele_name
from .peptide_encoding import index_encoding
from .paths import CLASS1_MODEL_DIRECTORY
from .feedforward import make_embedding_network
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

    @classmethod
    def from_hyperparameters(
            cls,
            name=None,
            max_ic50=MAX_IC50,
            peptide_length=9,
            embedding_input_dim=20,
            embedding_output_dim=20,
            layer_sizes=[50],
            activation="tanh",
            init="lecun_uniform",
            loss="mse",
            output_activation="sigmoid",
            dropout_probability=0,
            learning_rate=0.001):
        """
        Create untrained predictor with the given hyperparameters.
        """
        model = make_embedding_network(
            peptide_length=peptide_length,
            embedding_input_dim=embedding_input_dim,
            embedding_output_dim=embedding_output_dim,
            layer_sizes=layer_sizes,
            activation=activation,
            init=init,
            loss=loss,
            output_activation=output_activation,
            dropout_probability=dropout_probability,
            learning_rate=learning_rate)
        return cls(
            name=name,
            max_ic50=max_ic50,
            model=model)

    def _combine_training_data(
            self,
            X,
            Y,
            sample_weights,
            X_pretrain,
            Y_pretrain,
            pretrain_sample_weights,
            verbose=False):
        """
        Make sure the shapes of given training and pre-training data
        conform with each other. Then concatenate the pre-training and the
        training data.

        Returns (X_combined, Y_combined, weights_combined, n_pretrain_samples)
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        if verbose:
            print("X.shape = %s" % (X.shape,))

        n_samples, n_dims = X.shape

        if len(Y) != n_samples:
            raise ValueError("Mismatch between len(X) = %d and len(Y) = %d" % (
                n_samples, len(Y)))

        if Y.min() < 0:
            raise ValueError("Minimum value of Y can't be negative, got %f" % (
                Y.min()))
        if Y.max() > 1:
            raise ValueError("Maximum value of Y can't be greater than 1, got %f" % (
                Y.max()))

        if sample_weights is None:
            sample_weights = np.ones_like(Y)
        else:
            sample_weights = np.asarray(sample_weights)

        if len(sample_weights) != n_samples:
            raise ValueError(
                "Length of sample_weights (%d) doesn't match number of samples (%d)" % (
                    len(sample_weights),
                    n_samples))

        if X_pretrain is None or Y_pretrain is None:
            X_pretrain = np.empty((0, n_dims), dtype=X.dtype)
            Y_pretrain = np.empty((0,), dtype=Y.dtype)
        else:
            X_pretrain = np.asarray(X_pretrain)
            Y_pretrain = np.asarray(Y_pretrain)

        if verbose:
            print("X_pretrain.shape = %s" % (X_pretrain.shape,))
        n_pretrain_samples, n_pretrain_dims = X_pretrain.shape
        if n_pretrain_dims != n_dims:
            raise ValueError(
                "# of dims for pretraining data (%d) doesn't match X.shape[1] = %d" % (
                    n_pretrain_dims, n_dims))

        if len(Y_pretrain) != n_pretrain_samples:
            raise ValueError(
                "length of Y_pretrain (%d) != length of X_pretrain (%d)" % (
                    len(Y_pretrain),
                    len(X_pretrain)))

        if len(Y_pretrain) > 0 and Y_pretrain.min() < 0:
            raise ValueError("Minimum value of Y_pretrain can't be negative, got %f" % (
                Y.min()))
        if len(Y_pretrain) > 0 and Y_pretrain.max() > 1:
            raise ValueError("Maximum value of Y_pretrain can't be greater than 1, got %f" % (
                Y.max()))

        if pretrain_sample_weights is None:
            pretrain_sample_weights = np.ones_like(Y_pretrain)
        else:
            pretrain_sample_weights = np.asarray(pretrain_sample_weights)
        if verbose:
            print("sample weights mean = %f, pretrain weights mean = %f" % (
                sample_weights.mean(),
                pretrain_sample_weights.mean()))
        X_combined = np.vstack([X_pretrain, X])
        Y_combined = np.concatenate([Y_pretrain, Y])
        combined_weights = np.concatenate([
            sample_weights,
            pretrain_sample_weights
        ])
        return X_combined, Y_combined, combined_weights, n_pretrain_samples

    def fit(
            self,
            X,
            Y,
            sample_weights=None,
            X_pretrain=None,
            Y_pretrain=None,
            pretrain_sample_weights=None,
            n_training_epochs=200,
            verbose=False):
        """
        Train predictive model from index encoding of fixed length 9mer peptides.

        Parameters
        ----------
        X : array
            Training data with shape (n_samples, n_dims)

        Y : array
            Training labels with shape (n_samples,)

        sample_weights : array
            Weight of each training sample with shape (n_samples,)

        X_pretrain : array
            Extra samples used for soft pretraining of the predictor,
            should have same number of dimensions as X. During training the weights
            of these samples will decay after each epoch.

        Y_pretrain : array
            Labels for extra samples, shape

        pretrain_sample_weights : array
            Initial weights for the rows of X_pretrain. If not specified then
            initialized to ones.

        n_training_epochs : int

        verbose : bool
        """
        X_combined, Y_combined, combined_weights, n_pretrain = \
            self._combine_training_data(
                X, Y, sample_weights, X_pretrain, Y_pretrain, pretrain_sample_weights,
                verbose=verbose)

        total_pretrain_sample_weight = combined_weights[:n_pretrain].sum()
        total_train_sample_weight = combined_weights[n_pretrain:].sum()
        total_combined_sample_weight = (
            total_pretrain_sample_weight + total_train_sample_weight)

        for epoch in range(n_training_epochs):
            # weights for synthetic points can be shrunk as:
            #  ~ 1 / (1+epoch)**2
            # or
            # e ** -epoch
            decay_factor = np.exp(-epoch)
            # if the contribution of synthetic samples is less than a
            # thousandth of the actual data, then stop using it
            pretrain_contribution = total_pretrain_sample_weight * decay_factor
            pretrain_fraction_contribution = (
                pretrain_contribution / total_combined_sample_weight)

            use_pretrain_data = pretrain_fraction_contribution > 0.001

            # only use synthetic data if it contributes at least 1/1000th of
            # sample weight
            if verbose:

                real_data_percent = (
                    ((1.0 - pretrain_fraction_contribution) * 100)
                    if use_pretrain_data
                    else 100
                )
                pretrain_data_percent = (
                    (pretrain_fraction_contribution * 100)
                    if use_pretrain_data else 0
                )
                print(
                    ("-- Epoch %d/%d decay=%f, real data weight=%0.2f%%,"
                     " synth data weight=%0.2f%% (use_pretrain=%s)") % (
                        epoch + 1,
                        n_training_epochs,
                        decay_factor,
                        real_data_percent,
                        pretrain_data_percent,
                        use_pretrain_data))

            if use_pretrain_data:
                combined_weights[:n_pretrain] *= decay_factor
                self.model.fit(
                    X_combined,
                    Y_combined,
                    sample_weight=combined_weights,
                    nb_epoch=1,
                    verbose=0)
            else:
                self.model.fit(
                    X_combined[n_pretrain:],
                    Y_combined[n_pretrain:],
                    sample_weight=combined_weights[n_pretrain:],
                    nb_epoch=1,
                    verbose=0)

    """
    def fit_peptides(
            self,
            peptides,
            affinity_values,
            sample_weights=None,
            pretrain_peptides=None,
            pretrain_affinity_values=None,
            pretrain_sample_weights=None,
            n_training_epochs=200,
            verbose=False):
        '''
        Train model from peptide sequences, expanding shorter or longer
        peptides to make 9mers.
        '''
        X, original_peptides, counts = \
                fixed_length_index_encoding(
                    peptides=peptides,
                    desired_length=9)
        lookup = {k: v for (k, v) in zip}
        Y = np.asarray(Y)
        if sample_weights is None:
            sample_weights = np.ones_like(Y)
        else:
            sample_weights = np.asarray(Y)

        if pretrain_peptides is None:

        if Y_pretrain
        Y_pretrain = np.asarray(Y_pretrain)
        if pretrain_sample_weights is None:
            pretrain_sample_weights = np.ones_like(Y_pretrain)

        train_weights = 1.0 / np.array(expanded_train_counts)
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

    def predict_peptides_log_ic50(self, peptides):
        """
        Given a list of peptides, returns an array of predicted values
        """
        results_dict = {}
        for length, group_peptides in groupby(peptides, lambda x: len(x)):
            group_peptides = list(group_peptides)
            expanded_peptides, _, _ = fixed_length_from_many_peptides(
                peptides=group_peptides,
                desired_length=9)
            n_group = len(group_peptides)
            n_expanded = len(expanded_peptides)
            expansion_factor = int(n_expanded / n_group)
            raw_y = self._predict_9mer_peptides(expanded_peptides)
            if expansion_factor == 1:
                log_ic50s = raw_y
            else:
                # if peptides were a different length than the predictor's
                # expected input length, then let's take the median prediction
                # of each expanded peptide set
                log_ic50s = np.zeros(n_group)
                # take the median of each group of log(IC50) values
                for i in range(n_group):
                    start = i * expansion_factor
                    end = (i + 1) * expansion_factor
                    log_ic50s[i] = np.mean(raw_y[start:end])
            assert len(group_peptides) == len(log_ic50s)
            for peptide, log_ic50 in zip(group_peptides, log_ic50s):
                results_dict[peptide] = log_ic50
        return np.array([results_dict[p] for p in peptides])

    def predict_peptides(self, peptides):
        log_affinities = self.predict_peptides_log_ic50(peptides)
        return self._log_to_ic50(log_affinities)
