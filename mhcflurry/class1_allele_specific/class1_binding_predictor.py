# Copyright (c) 2016. Mount Sinai School of Medicine
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

import tempfile
import os

import numpy as np

import keras.models

from ..feedforward import make_embedding_network
from .class1_allele_specific_kmer_ic50_predictor_base import (
    Class1AlleleSpecificKmerIC50PredictorBase,
)
from ..peptide_encoding import check_valid_index_encoding_array
from ..regression_target import MAX_IC50, ic50_to_regression_target
from ..training_helpers import (
    combine_training_arrays,
    extend_with_negative_random_samples,
)
from ..regression_target import regression_target_to_ic50
from ..hyperparameters import HyperparameterDefaults


class Class1BindingPredictor(Class1AlleleSpecificKmerIC50PredictorBase):
    """
    Allele-specific Class I MHC binding predictor which uses
    fixed-length (k-mer) index encoding for inputs and outputs
    a value between 0 and 1 (where 1 is the strongest binder).
    """

    network_hyperparameter_defaults = HyperparameterDefaults(
        embedding_output_dim=32,
        layer_sizes=[64],
        init="glorot_uniform",
        loss="mse",
        optimizer="rmsprop",
        output_activation="sigmoid",
        activation="tanh",
        dropout_probability=0.0)

    fit_hyperparameter_defaults = HyperparameterDefaults(
        n_training_epochs=250,
        batch_size=128,
        pretrain_decay="numpy.exp(-epoch)",
        fraction_negative=0.0,
        batch_normalization=True)

    hyperparameter_defaults = (
        Class1AlleleSpecificKmerIC50PredictorBase.hyperparameter_defaults
        .extend(network_hyperparameter_defaults)
        .extend(fit_hyperparameter_defaults))

    def __init__(
            self,
            model=None,
            name=None,
            max_ic50=MAX_IC50,
            allow_unknown_amino_acids=True,
            kmer_size=9,
            n_amino_acids=20,
            verbose=False,
            **hyperparameters):
        Class1AlleleSpecificKmerIC50PredictorBase.__init__(
            self,
            name=name,
            max_ic50=max_ic50,
            allow_unknown_amino_acids=allow_unknown_amino_acids,
            verbose=verbose,
            kmer_size=kmer_size)

        specified_network_hyperparameters = (
            self.network_hyperparameter_defaults.subselect(hyperparameters))

        effective_hyperparameters = (
            self.hyperparameter_defaults.with_defaults(hyperparameters))

        if model is None:
            model = make_embedding_network(
                peptide_length=kmer_size,
                n_amino_acids=n_amino_acids + int(allow_unknown_amino_acids),
                **self.network_hyperparameter_defaults.subselect(
                    effective_hyperparameters))
        elif specified_network_hyperparameters:
            raise ValueError(
                "Do not specify network hyperparameters when passing a model. "
                "Network hyperparameters specified: %s"
                % " ".join(specified_network_hyperparameters))

        self.hyperparameters = effective_hyperparameters
        self.name = name
        self.model = model

    def __getstate__(self):
        result = dict(self.__dict__)
        del result['model']
        result['model_json'] = self.model.to_json()
        result['model_weights'] = self.get_weights()
        return result

    def __setstate__(self, state):
        model_bytes = model_json = model_weights = None
        try:
            model_bytes = state.pop('model_bytes')
        except KeyError:
            model_json = state.pop('model_json')
            model_weights = state.pop('model_weights')
        self.__dict__.update(state)

        if model_bytes is not None:
            # Old format
            fd = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False)
            try:
                fd.write(model_bytes)

                # HDF5 has issues when the file is open multiple times, so we close
                # it here before loading it into keras.
                fd.close()
                self.model = keras.models.load_model(fd.name)
            finally:
                os.unlink(fd.name)
        else:
            self.model = keras.models.model_from_json(model_json)
            self.set_weights(model_weights)

    def get_weights(self):
        """
        Returns weights, which can be passed to set_weights later.
        """
        return [x.copy() for x in self.model.get_weights()]

    def set_weights(self, weights):
        """
        Reset the model weights.
        """
        self.model.set_weights(weights)

    def fit_kmer_encoded_arrays(
            self,
            X,
            ic50,
            sample_weights=None,
            right_censoring_mask=None,
            X_pretrain=None,
            ic50_pretrain=None,
            sample_weights_pretrain=None,
            n_random_negative_samples=None,
            pretrain_decay=None,
            n_training_epochs=None,
            batch_size=None,
            verbose=False):
        """
        Train predictive model from index encoding of fixed length k-mer
        peptides.

        Parameters
        ----------
        X : array
            Training data with shape (n_samples, n_dims)

        ic50 : array
            Training IC50 values with shape (n_samples,)

        sample_weights : array
            Weight of each training sample with shape (n_samples,)

        right_censoring_mask : array, optional
            Boolean array which indicates whether each IC50 value is actually
            right censored (a lower bound on the true value). Censored values
            are transformed during training by sampling between the observed
            and maximum values on each iteration.

        X_pretrain : array
            Extra samples used for soft pretraining of the predictor,
            should have same number of dimensions as X.
            During training the weights of these samples will decay after
            each epoch.

        ic50_pretrain : array
            IC50 values for extra samples, shape

        pretrain_decay : int -> float function
            decay function for pretraining, mapping epoch number to decay
            factor

        sample_weights_pretrain : array
            Initial weights for the rows of X_pretrain. If not specified then
            initialized to ones.

        n_random_negative_samples : int
            Number of random samples to generate as negative examples.

        n_training_epochs : int

        verbose : bool

        batch_size : int
        """

        # Apply defaults from hyperparameters
        if n_random_negative_samples is None:
            n_random_negative_samples = (
                int(self.hyperparameters["fraction_negative"] * len(ic50)))

        if pretrain_decay is None:
            pretrain_decay = (
                lambda epoch:
                eval(
                    self.hyperparameters["pretrain_decay"],
                    {'epoch': epoch, 'numpy': np}))

        if n_training_epochs is None:
            n_training_epochs = self.hyperparameters["n_training_epochs"]

        if batch_size is None:
            batch_size = self.hyperparameters["batch_size"]

        X_combined, ic50_combined, combined_weights, n_pretrain = \
            combine_training_arrays(
                X, ic50, sample_weights,
                X_pretrain, ic50_pretrain, sample_weights_pretrain)

        Y_combined = ic50_to_regression_target(
            ic50_combined, max_ic50=self.max_ic50)

        # create a censored IC50 mask for all combined samples and then fill
        # in the training censoring mask if it's given
        right_censoring_mask_combined = np.zeros(len(Y_combined), dtype=bool)
        if right_censoring_mask is not None:
            right_censoring_mask = np.asarray(right_censoring_mask)
            if len(right_censoring_mask.shape) != 1:
                raise ValueError("Expected 1D censor mask, got shape %s" % (
                    right_censoring_mask.shape,))
            if len(right_censoring_mask) != len(ic50):
                raise ValueError(
                    "Wrong length for censoring mask, expected %d not %d" % (
                        len(ic50),
                        len(right_censoring_mask)))
            right_censoring_mask_combined[n_pretrain:] = right_censoring_mask

        n_censored = right_censoring_mask_combined.sum()

        total_pretrain_sample_weight = combined_weights[:n_pretrain].sum()
        total_train_sample_weight = combined_weights[n_pretrain:].sum()
        total_combined_sample_weight = (
            total_pretrain_sample_weight + total_train_sample_weight)

        for epoch in range(n_training_epochs):
            decay_factor = pretrain_decay(epoch)

            # if the contribution of synthetic samples is less than a
            # thousandth of the actual data, then stop using it
            pretrain_contribution = total_pretrain_sample_weight * decay_factor
            pretrain_fraction_contribution = (
                pretrain_contribution / total_combined_sample_weight)

            if n_censored > 0:
                # shrink the output values by a uniform amount to some value
                # between the lowest representable affinity and the observed
                # censored value
                Y_adjusted_for_censoring = Y_combined.copy()
                Y_adjusted_for_censoring[right_censoring_mask_combined] *= (
                    np.random.rand(n_censored))
            else:
                Y_adjusted_for_censoring = Y_combined

            # only use synthetic data if it contributes at least 1/1000th of
            # sample weight
            if pretrain_fraction_contribution > 0.001:
                combined_weights[:n_pretrain] *= decay_factor
                X_curr_iter = X_combined
                Y_curr_iter = Y_adjusted_for_censoring
                weights_curr_iter = combined_weights
            else:
                X_curr_iter = X_combined[n_pretrain:]
                Y_curr_iter = Y_adjusted_for_censoring[n_pretrain:]
                weights_curr_iter = combined_weights[n_pretrain:]

            if n_random_negative_samples > 0:
                X_curr_iter, Y_curr_iter, weights_curr_iter = \
                    extend_with_negative_random_samples(
                        X_curr_iter,
                        Y_curr_iter,
                        weights_curr_iter,
                        n_random_negative_samples,
                        max_amino_acid_encoding_value=(
                            self.max_amino_acid_encoding_value))

            self.model.fit(
                X_curr_iter,
                Y_curr_iter,
                sample_weight=weights_curr_iter,
                nb_epoch=1,
                verbose=0,
                batch_size=batch_size,
                shuffle=True)

    def predict_scores_for_kmer_encoded_array(self, X):
        """
        Given an encoded array of amino acid indices, returns a vector
        of affinity scores (values between 0 and 1).
        """
        X = check_valid_index_encoding_array(
            X,
            allow_unknown_amino_acids=self.allow_unknown_amino_acids)
        return self.model.predict(X, verbose=False).flatten()

    def predict_ic50_for_kmer_encoded_array(self, X):
        """
        Given an encoded array of amino acid indices,
        returns a vector of IC50 predictions.
        """
        scores = self.predict_scores_for_kmer_encoded_array(X)
        return regression_target_to_ic50(scores, max_ic50=self.max_ic50)
