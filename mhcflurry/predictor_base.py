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

from __future__ import (
    print_function,
    division,
    absolute_import,
)
from collections import defaultdict

import numpy as np
from six import string_types

from .peptide_encoding import fixed_length_index_encoding
from .amino_acid import (
    amino_acids_with_unknown,
    common_amino_acids
)
from .regression_target import regression_target_to_ic50, MAX_IC50
from .dataset import Dataset


class PredictorBase(object):
    """
    Base class for all mhcflurry predictors which used fixed-length
    k-mer representation of peptides. Eventually will need to move this code
    to something like FixedLengthPredictor to fit RNN-based sequence
    predictors into the inheritance hierarchy.
    """
    def __init__(
            self,
            name,
            allow_unknown_amino_acids,
            verbose,
            max_ic50=MAX_IC50,
            kmer_size=9):
        self.name = name
        self.max_ic50 = max_ic50
        self.allow_unknown_amino_acids = allow_unknown_amino_acids
        self.verbose = verbose
        self.kmer_size = kmer_size

    @property
    def amino_acids(self):
        """
        Amino acid alphabet used for encoding peptides, may include
        "X" if allow_unknown_amino_acids is True.
        """
        if self.allow_unknown_amino_acids:
            return amino_acids_with_unknown
        else:
            return common_amino_acids

    @property
    def max_amino_acid_encoding_value(self):
        return len(self.amino_acids)

    def encode_peptides(self, peptides):
        """
        Parameters
        ----------
        peptides : str list
            Peptide strings of any length

        Encode peptides of any length into fixed length vectors.
        Returns 2d array of encoded peptides and 1d array indicating the
        original peptide index for each row.
        """
        indices = []
        encoded_matrices = []
        for i, peptide in enumerate(peptides):
            matrix, _, _, _ = fixed_length_index_encoding(
                peptides=[peptide],
                desired_length=self.kmer_size,
                allow_unknown_amino_acids=self.allow_unknown_amino_acids)
            encoded_matrices.append(matrix)
            indices.extend([i] * len(matrix))
        combined_matrix = np.concatenate(encoded_matrices)
        index_array = np.array(indices)
        expected_shape = (len(index_array), self.kmer_size)
        assert combined_matrix.shape == expected_shape, \
            "Expected shape %s but got %s" % (expected_shape, combined_matrix.shape)
        return combined_matrix, index_array

    def _predict_kmer_peptides(self, peptides):
        """
        Predict binding affinity for 9mer peptides
        """
        if any(len(peptide) != self.kmer_size for peptide in peptides):
            raise ValueError("Can only predict 9mer peptides")
        X, _ = self.encode_peptides(peptides)
        return self.predict(X)

    def _predict_kmer_peptides_ic50(self, peptides):
        scores = self.predict_kmer_peptides(peptides)
        return regression_target_to_ic50(scores, max_ic50=self.max_ic50)

    def predict_scores(self, peptides, combine_fn=np.mean):
        """
        Given a list of peptides of any length, returns an array of predicted
        normalized affinity values. Unlike IC50, a higher value here
        means a stronger affinity. Peptides of lengths other than 9 are
        transformed into a set of k-mers either by deleting or inserting
        amino acid characters. The prediction for a single peptide will be
        the average of expanded k-mers.
        """
        if isinstance(peptides, string_types):
            raise TypeError("Input must be a list of peptides, not %s : %s" % (
                peptides, type(peptides)))

        input_matrix, original_peptide_indices = self.encode_peptides(peptides)
        # peptides of lengths other than self.kmer_size get multiple predictions,
        # which are then combined with the combine_fn argument
        multiple_predictions_dict = defaultdict(list)
        fixed_length_predictions = self.predict(input_matrix)
        for i, yi in enumerate(fixed_length_predictions):
            original_peptide_index = original_peptide_indices[i]
            original_peptide = peptides[original_peptide_index]
            multiple_predictions_dict[original_peptide].append(yi)
        combined_predictions_dict = {
            p: combine_fn(ys) if len(ys) > 1 else ys[0]
            for (p, ys) in multiple_predictions_dict.items()
        }
        return np.array([combined_predictions_dict[p] for p in peptides])

    def predict(self, peptides):
        """
        Predict IC50 affinities for peptides of any length
        """
        scores = self.predict_peptides(peptides)
        return regression_target_to_ic50(scores, max_ic50=self.max_ic50)

    def fit_dictionary(self, peptide_to_ic50_dict, **kwargs):
        """
        Fit the model parameters using the given peptide->IC50 dictionary,
        all samples are given the same weight.

        Parameters
        ----------
        peptide_to_ic50_dict : dict
            Dictionary that maps peptides to IC50 values.
        """
        dataset = Dataset.from_peptide_to_affinity_dictionary(
            allele_name=self.name,
            peptide_to_affinity_dict=peptide_to_ic50_dict)
        return self.fit_dataset(dataset, **kwargs)

    def fit_dataset(self, dataset, pretraining_dataset=None, *args, **kwargs):
        """
        Fit the model parameters on the given training data.

        Parameters
        ----------
        dataset : Dataset

        pretraining_dataset : Dataset

        **kwargs : dict
            Extra arguments are passed on to the fit_encoded_kmer_arrays()
            method.
        """
        X, Y, sample_weights = dataset.encode()
        X_pretrain, Y_pretrain, sample_weights_pretrain = pretraining_dataset.encode()
        return self.fit_arrays(
            X=X,
            Y=Y,
            sample_weights=sample_weights,
            X_pretrain=X_pretrain,
            Y_pretrain=Y_pretrain,
            sample_weights_pretrain=sample_weights,
            **kwargs)

    def fit_sequences(self, peptides, affinities, sample_weights=None, **kwargs):
        alleles = [self.name] * len(peptides)
        dataset = Dataset.from_sequences(
            alleles=alleles,
            peptides=peptides,
            affinities=affinities,
            sample_weights=sample_weights)
        return self.fit_dataset(dataset, **kwargs)
