from __future__ import print_function

import time
import collections
from six import string_types

import numpy
import pandas
import mhcnames
import hashlib

from .hyperparameters import HyperparameterDefaults
from .class1_neural_network import Class1NeuralNetwork, DEFAULT_PREDICT_BATCH_SIZE
from .encodable_sequences import EncodableSequences
from .regression_target import from_ic50, to_ic50
from .random_negative_peptides import RandomNegativePeptides
from .allele_encoding import MultipleAlleleEncoding, AlleleEncoding
from .auxiliary_input import AuxiliaryInputEncoder
from .batch_generator import MultiallelicMassSpecBatchGenerator
from .custom_loss import (
    MSEWithInequalities,
    MultiallelicMassSpecLoss,
    ZeroLoss)


class Class1LigandomePredictor(object):
    def __init__(self, class1_ligandome_neural_networks, allele_to_sequence):
        self.networks = class1_ligandome_neural_networks
        self.allele_to_sequence = allele_to_sequence

    @property
    def max_alleles(self):
        max_alleles = self.networks[0].hyperparameters['max_alleles']
        assert all(
            n.hyperparameters['max_alleles'] == self.max_alleles
            for n in self.networks)
        return max_alleles

    def predict(self, peptides, alleles, batch_size=DEFAULT_PREDICT_BATCH_SIZE):
        return self.predict_to_dataframe(
            peptides=peptides,
            alleles=alleles,
            batch_size=batch_size).score.values

    def predict_to_dataframe(
            self,
            peptides,
            alleles,
            include_details=False,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):

        if isinstance(peptides, string_types):
            raise TypeError("peptides must be a list or array, not a string")
        if isinstance(alleles, string_types):
            raise TypeError(
                "alleles must be an iterable or MultipleAlleleEncoding")

        peptides = EncodableSequences.create(peptides)

        if not isinstance(alleles, MultipleAlleleEncoding):
            if len(alleles) > self.max_alleles:
                raise ValueError(
                    "When alleles is a list, it must have at most %d elements. "
                    "These alleles are taken to be a genotype for an "
                    "individual, and the strongest prediction across alleles "
                    "will be taken for each peptide. Note that this differs "
                    "from Class1AffinityPredictor.predict(), where alleles "
                    "is expected to be the same length as peptides."
                    % (
                        self.max_alleles))
            alleles = MultipleAlleleEncoding(
                experiment_names=numpy.tile("experiment", len(peptides)),
                experiment_to_allele_list={
                    "experiment": alleles,
                },
                allele_to_sequence=self.allele_to_sequence,
                max_alleles_per_experiment=self.max_alleles)

        score_array = []
        affinity_array = []

        for (i, network) in enumerate(self.networks):
            predictions = network.predict(
                peptides=peptides,
                allele_encoding=alleles,
                batch_size=batch_size)
            score_array.append(predictions.score)
            affinity_array.append(predictions.affinity)

        score_array = numpy.array(score_array)
        affinity_array = numpy.array(affinity_array)

        ensemble_scores = numpy.mean(score_array, axis=0)
        ensemble_affinity = numpy.mean(affinity_array, axis=0)
        top_allele_index = numpy.argmax(ensemble_scores, axis=-1)
        top_score = ensemble_scores[top_allele_index]
        top_affinity = ensemble_affinity[top_allele_index]

        result_df = pandas.DataFrame({"peptide": peptides.sequences})
        result_df["allele"] = alleles.alleles[top_allele_index]
        result_df["score"] = top_score
        result_df["affinity"] = to_ic50(top_affinity)

        if include_details:
            for i in range(self.max_alleles):
                result_df["allele%d" % (i + 1)] = alleles.allele[:, i]
                result_df["allele%d score" % (i + 1)] = ensemble_scores[:, i]
                result_df["allele%d score low" % (i + 1)] = numpy.percentile(
                    score_array[:, :, i], 5.0, axis=0)
                result_df["allele%d score high" % (i + 1)] = numpy.percentile(
                    score_array[:, :, i], 95.0, axis=0)
                result_df["allele%d affinity" % (i + 1)] = to_ic50(
                    ensemble_affinity[:, i])
                result_df["allele%d affinity low" % (i + 1)] = numpy.percentile(
                    affinity_array[:, :, i], 5.0, axis=0)
                result_df["allele%d affinity high" % (i + 1)] = numpy.percentile(
                    affinity_array[:, :, i], 95.0, axis=0)
        return result_df


    # TODO: implement saving and loading