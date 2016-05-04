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

from os import listdir
from os.path import exists, join

import numpy as np

from .common import normalize_allele_name
from .paths import CLASS1_MODEL_DIRECTORY
from .feedforward import make_embedding_network
from .predictor_base import PredictorBase
from .serialization_helpers import (
    load_keras_model_from_disk,
    save_keras_model_to_disk
)
from .peptide_encoding import check_valid_index_encoding_array
from .class1_allele_specific_hyperparameters import MAX_IC50

_allele_predictor_cache = {}

class Class1BindingPredictor(PredictorBase):
    """
    Allele-specific Class I MHC binding predictor which uses
    fixed-length (9mer) index encoding for inputs and outputs
    a value between 0 and 1 (where 1 is the strongest binder).
    """
    def __init__(
            self,
            model,
            name=None,
            max_ic50=MAX_IC50,
            allow_unknown_amino_acids=True,
            verbose=False):
        PredictorBase.__init__(
            self,
            name=name,
            max_ic50=max_ic50,
            allow_unknown_amino_acids=allow_unknown_amino_acids,
            verbose=verbose)
        self.name = name
        self.model = model

    @classmethod
    def from_disk(
            cls,
            model_json_path,
            weights_hdf_path=None,
            **kwargs):
        """
        Load model from stored JSON representation of network and
        (optionally) load weights from HDF5 file.
        """
        model = load_keras_model_from_disk(
            model_json_path,
            weights_hdf_path,
            name=None)
        return cls(model=model, **kwargs)

    def to_disk(self, model_json_path, weights_hdf_path, overwrite=False):
        save_keras_model_to_disk(
            self.model,
            model_json_path,
            weights_hdf_path,
            overwrite=overwrite,
            name=self.name)

    @classmethod
    def from_hyperparameters(
            cls,
            name=None,
            max_ic50=MAX_IC50,
            peptide_length=9,
            n_amino_acids=20,
            allow_unknown_amino_acids=True,
            embedding_output_dim=20,
            layer_sizes=[50],
            activation="tanh",
            init="lecun_uniform",
            loss="mse",
            output_activation="sigmoid",
            dropout_probability=0,
            learning_rate=0.001,
            **kwargs):
        """
        Create untrained predictor with the given hyperparameters.
        """
        embedding_input_dim = n_amino_acids + int(allow_unknown_amino_acids)
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
            model=model,
            allow_unknown_amino_acids=allow_unknown_amino_acids,
            **kwargs)

    def _combine_training_data(
            self,
            X,
            Y,
            sample_weights,
            X_pretrain,
            Y_pretrain,
            sample_weights_pretrain,
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

        if sample_weights_pretrain is None:
            sample_weights_pretrain = np.ones_like(Y_pretrain)
        else:
            sample_weights_pretrain = np.asarray(sample_weights_pretrain)
        if verbose:
            print("sample weights mean = %f, pretrain weights mean = %f" % (
                sample_weights.mean(),
                sample_weights_pretrain.mean()))
        X_combined = np.vstack([X_pretrain, X])
        Y_combined = np.concatenate([Y_pretrain, Y])
        combined_weights = np.concatenate([
            sample_weights_pretrain,
            sample_weights,
        ])
        return X_combined, Y_combined, combined_weights, n_pretrain_samples

    def _extend_with_negative_random_samples(
            self, X, Y, weights, n_random_negative_samples):
        """
        Extend training data with randomly generated negative samples.
        This only works since most random peptides will not bind to a
        particular allele.
        """
        assert len(X) == len(Y) == len(weights)
        if n_random_negative_samples == 0:
            return X, Y, weights
        min_value = X.min()
        max_value = X.max()
        n_cols = X.shape[1]
        X_random = np.random.randint(
            low=min_value,
            high=max_value + 1,
            shape=(n_random_negative_samples, n_cols)).astype(X.dtype)
        Y_random = np.zeros(n_random_negative_samples, dtype=float)
        weights_random = np.ones(n_random_negative_samples, dtype=float)
        X_with_negative = np.vstack([X, X_random])
        Y_with_negative = np.concatenate([Y, Y_random])
        weights_with_negative = np.concatenate([
            weights,
            weights_random])
        assert len(X_with_negative) == len(X) + n_random_negative_samples
        assert len(Y_with_negative) == len(Y) + n_random_negative_samples
        assert len(weights_with_negative) == len(weights) + n_random_negative_samples
        return X_with_negative, Y_with_negative, weights_with_negative

    def fit(
            self,
            X,
            Y,
            sample_weights=None,
            X_pretrain=None,
            Y_pretrain=None,
            sample_weights_pretrain=None,
            n_random_negative_samples=0,
            n_training_epochs=200,
            verbose=False,
            batch_size=128):
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

        sample_weights_pretrain : array
            Initial weights for the rows of X_pretrain. If not specified then
            initialized to ones.

        n_random_negative_samples : int
            Number of random samples to generate as negative examples.

        n_training_epochs : int

        verbose : bool

        batch_size : int
        """
        X_combined, Y_combined, combined_weights, n_pretrain = \
            self._combine_training_data(
                X, Y, sample_weights,
                X_pretrain, Y_pretrain, sample_weights_pretrain,
                verbose=verbose)

        total_pretrain_sample_weight = combined_weights[:n_pretrain].sum()
        total_train_sample_weight = combined_weights[n_pretrain:].sum()
        total_combined_sample_weight = (
            total_pretrain_sample_weight + total_train_sample_weight)

        if self.verbose:
            print("-- Total pretrain weight = %f (%f%%), sample weight = %f (%f%%)" % (
                total_pretrain_sample_weight,
                100 * total_pretrain_sample_weight / total_combined_sample_weight,
                total_train_sample_weight,
                100 * total_train_sample_weight / total_combined_sample_weight))

        for epoch in range(n_training_epochs):
            # weights for synthetic points can be shrunk as:
            #  ~ 1 / (1+epoch)**2
            # or
            # 2 ** -epoch
            # or
            # e ** -epoch
            #
            # TODO: explore the best scheme for shrinking imputation weight
            #
            decay_factor = 2.0 ** -epoch
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
                X_curr_iter = X_combined
                Y_curr_iter = Y_combined
                weights_curr_iter = combined_weights
            else:
                X_curr_iter = X_combined[n_pretrain:]
                Y_curr_iter = Y_combined[n_pretrain:]
                weights_curr_iter = combined_weights[n_pretrain:]

            X_curr_iter, Y_curr_iter, weights_curr_iter = \
                self._extend_with_negative_random_samples(
                    X_curr_iter,
                    Y_curr_iter,
                    weights_curr_iter)

            self.model.fit(
                X_curr_iter,
                Y_curr_iter,
                sample_weight=weights_curr_iter,
                nb_epoch=1,
                verbose=0,
                batch_size=batch_size,
                shuffle=True)

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
        return "Class1BindingPredictor(name=%s, model=%s, max_ic50=%f)" % (
            self.name,
            self.model,
            self.max_ic50)

    def __str__(self):
        return repr(self)

    def predict(self, X):
        """
        Given an encoded array of amino acid indices, returns a vector
        of predicted log IC50 values.
        """
        X = check_valid_index_encoding_array(
            X,
            allow_unknown_amino_acids=self.allow_unknown_amino_acids)
        return self.model.predict(X, verbose=False).flatten()
