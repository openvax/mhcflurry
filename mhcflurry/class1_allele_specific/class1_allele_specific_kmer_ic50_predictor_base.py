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

from six import string_types

from ..peptide_encoding import encode_peptides
from ..amino_acid import (
    amino_acids_with_unknown,
    common_amino_acids
)
from ..ic50_predictor_base import IC50PredictorBase
from ..hyperparameters import HyperparameterDefaults


class Class1AlleleSpecificKmerIC50PredictorBase(IC50PredictorBase):
    """
    Base class for all mhcflurry predictors which used fixed-length
    k-mer representation of peptides and don't require scanning over
    a longer sequence to find a binding core (like you might for Class II).
    """
    hyperparameter_defaults = (HyperparameterDefaults(
        kmer_size=9)
        .extend(IC50PredictorBase.hyperparameter_defaults))

    def __init__(
            self,
            name,
            allow_unknown_amino_acids,
            verbose,
            **hyperparameters):
        effective_hyperparameters = (
            self.hyperparameter_defaults.with_defaults(hyperparameters))
        IC50PredictorBase.__init__(
            self,
            name=name,
            verbose=verbose,
            **IC50PredictorBase.hyperparameter_defaults.subselect(
                effective_hyperparameters))
        self.allow_unknown_amino_acids = allow_unknown_amino_acids
        self.kmer_size = effective_hyperparameters["kmer_size"]

    def __repr__(self):
        return (
            "%s(name=%s, max_ic50=%f, allow_unknown_amino_acids=%s, "
            "kmer_size=%d)" % (
                self.__class__.__name__,
                self.name,
                self.max_ic50,
                self.allow_unknown_amino_acids,
                self.kmer_size))

    def __str__(self):
        return repr(self)

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

    def predict_scores(self, peptides):
        """
        Given a list of peptides of any length, returns an array of predicted
        normalized affinity values. Unlike IC50, a higher value here
        means a stronger affinity. Peptides of lengths other than 9 are
        transformed into a set of k-mers either by deleting or inserting
        amino acid characters. The prediction for a single peptides will be
        the average of expanded k-mers.
        """
        if isinstance(peptides, string_types):
            raise TypeError("Input must be a list of peptides, not %s : %s" % (
                peptides, type(peptides)))

        encoded_peptides = encode_peptides(
            peptides, kmer_size=self.kmer_size, allow_unknown_amino_acids=self.allow_unknown_amino_acids)
        return encoded_peptides.combine_predictions(
            self.predict_scores_for_kmer_encoded_array(encoded_peptides.encoded_matrix))

    def fit_dataset(
            self,
            dataset,
            pretraining_dataset=None,
            sample_censored_affinities=False,
            **kwargs):
        """
        Fit the model parameters on the given training data.

        Parameters
        ----------
        dataset : AffinityMeasurementDataset

        pretraining_dataset : AffinityMeasurementDataset

        sample_censored_affinities : bool
            If a column named 'inequality' is in the AffinityMeasurementDataset then every
            peptide with a value of '>' on each training epoch, gets a
            randomly sampled IC50 between its observed value and the
            max_ic50 of the predictor. Default is False.

        **kwargs : dict
            Extra arguments are passed on to the fit_encoded_kmer_arrays()
            method.
        """
        if len(dataset.unique_alleles()) > 1:
            raise ValueError(
                "Allele-specific predictor can't be trained on multi-allele "
                "data: %s" % dataset)

        if pretraining_dataset and len(pretraining_dataset.unique_alleles()) > 1:
            raise ValueError(
                "Allele-specific predictor can't pretrain on data from multiple alleles: %s" %
                (pretraining_dataset,))

        X, ic50, sample_weights, original_peptide_indices = \
            dataset.kmer_index_encoding(
                kmer_size=self.kmer_size,
                allow_unknown_amino_acids=self.allow_unknown_amino_acids)
        if pretraining_dataset is None:
            X_pretrain = ic50_pretrain = sample_weights_pretrain = None
        else:
            X_pretrain, ic50_pretrain, sample_weights_pretrain, _ = \
                pretraining_dataset.kmer_index_encoding(
                    kmer_size=self.kmer_size,
                    allow_unknown_amino_acids=self.allow_unknown_amino_acids)

        if sample_censored_affinities and 'inequality' in dataset.columns:
            df = dataset.to_dataframe()
            inequalities = df["inequality"]
            censored_mask_for_variable_length_peptides = (inequalities == ">")
            censored_mask_for_kmers = censored_mask_for_variable_length_peptides[
                original_peptide_indices]
        else:
            censored_mask_for_kmers = None

        return self.fit_kmer_encoded_arrays(
            X=X,
            ic50=ic50,
            sample_weights=sample_weights,
            right_censoring_mask=censored_mask_for_kmers,
            X_pretrain=X_pretrain,
            ic50_pretrain=ic50_pretrain,
            sample_weights_pretrain=sample_weights_pretrain,
            **kwargs)
