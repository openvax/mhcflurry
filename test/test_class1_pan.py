"""
Tests for training and predicting using Class1 pan-allele models.
"""

import json
import os
import shutil
import tempfile
import subprocess
from copy import deepcopy

import pandas

from numpy.testing import assert_array_less, assert_equal

from mhcflurry import Class1AffinityPredictor,Class1NeuralNetwork
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.downloads import get_path


HYPERPARAMETERS = {
    'activation': 'tanh',
    'allele_dense_layer_sizes': [],
    'batch_normalization': False,
    'dense_layer_l1_regularization': 0.0,
    'dense_layer_l2_regularization': 0.0,
    'dropout_probability': 0.5,
    'early_stopping': True,
    'init': 'glorot_uniform',
    'layer_sizes': [64],
    'learning_rate': None,
    'locally_connected_layers': [],
    'loss': 'custom:mse_with_inequalities',
    'max_epochs': 5000,
    'minibatch_size': 128,
    'optimizer': 'rmsprop',
    'output_activation': 'sigmoid',
    'patience': 20,
    'peptide_allele_merge_activation': '',
    'peptide_allele_merge_method': 'concatenate',
    'peptide_amino_acid_encoding': 'BLOSUM62',
    'peptide_dense_layer_sizes': [],
    'peptide_encoding': {
        'alignment_method': 'left_pad_centered_right_pad',
        'max_length': 15,
        'vector_encoding_name': 'BLOSUM62',
    },
    'random_negative_affinity_max': 50000.0,
    'random_negative_affinity_min': 20000.0,
    'random_negative_constant': 25,
    'random_negative_distribution_smoothing': 0.0,
    'random_negative_match_distribution': True,
    'random_negative_rate': 0.2,
    'train_data': {},
    'validation_split': 0.1,
}


ALLELE_TO_SEQUENCE = pandas.read_csv(
    get_path(
        "allele_sequences", "allele_sequences.csv"),
    index_col=0).sequence.to_dict()


TRAIN_DF = pandas.read_csv(
    get_path(
        "data_curated", "curated_training_data.no_mass_spec.csv.bz2"))

TRAIN_DF = TRAIN_DF.loc[TRAIN_DF.allele.isin(ALLELE_TO_SEQUENCE)]
TRAIN_DF = TRAIN_DF.loc[TRAIN_DF.peptide.str.len() >= 8]
TRAIN_DF = TRAIN_DF.loc[TRAIN_DF.peptide.str.len() <= 15]


def test_train_simple():
    network = Class1NeuralNetwork(**HYPERPARAMETERS)
    allele_encoding = AlleleEncoding(
        TRAIN_DF.allele.values,
        allele_to_sequence=ALLELE_TO_SEQUENCE)
    network.fit(
        TRAIN_DF.peptide.values,
        affinities=TRAIN_DF.measurement_value.values,
        allele_encoding=allele_encoding,
        inequalities=TRAIN_DF.measurement_inequality.values)

    predictions = network.predict(
        peptides=TRAIN_DF.peptide.values,
        allele_encoding=allele_encoding)

    print(pandas.Series(predictions).describe())
