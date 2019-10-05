import logging
logging.getLogger('matplotlib').disabled = True
logging.getLogger('tensorflow').disabled = True

import time
import pandas

from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.amino_acid import BLOSUM62_MATRIX
from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor
from mhcflurry.downloads import get_path

from numpy.testing import assert_equal

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup

ALLELE_TO_SEQUENCE = pandas.read_csv(
    get_path(
        "allele_sequences", "allele_sequences.csv"),
    index_col=0).sequence.to_dict()

HYPERPARAMETERS = {
    'activation': 'tanh',
    'allele_dense_layer_sizes': [],
    'batch_normalization': False,
    'dense_layer_l1_regularization': 0.0,
    'dense_layer_l2_regularization': 0.0,
    'dropout_probability': 0.5,
    'early_stopping': True,
    'init': 'glorot_uniform',
    'layer_sizes': [4],
    'learning_rate': None,
    'locally_connected_layers': [],
    'loss': 'custom:mse_with_inequalities',
    'max_epochs': 40,
    'minibatch_size': 128,
    'optimizer': 'rmsprop',
    'output_activation': 'sigmoid',
    'patience': 2,
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
    'random_negative_constant': 0,
    'random_negative_distribution_smoothing': 0.0,
    'random_negative_match_distribution': True,
    'random_negative_rate': 0.0,
    'train_data': {},
    'validation_split': 0.1,
}


def test_changing_allele_representations():
    allele1 = "HLA-A*02:01"
    allele2 = "HLA-C*03:04"
    allele3 = "HLA-B*07:01"

    peptide = "SIINFEKL"

    allele_to_sequence = {}
    for allele in [allele1, allele2]:
        allele_to_sequence[allele] = ALLELE_TO_SEQUENCE[allele]

    data1 = []
    for i in range(5000):
        data1.append((allele1, peptide, 0, "="))
        data1.append((allele2, peptide, 50000, "="))
    data1 = pandas.DataFrame(
        data1, columns=["allele", "peptide", "affinity", "inequality"])

    predictor = Class1AffinityPredictor(allele_to_sequence=allele_to_sequence)
    predictor.fit_class1_pan_allele_models(
        n_models=1,
        architecture_hyperparameters=HYPERPARAMETERS,
        alleles=data1.allele.values,
        peptides=data1.peptide.values,
        affinities=data1.affinity.values,
        inequalities=data1.inequality.values)

    (value1, value2) = predictor.predict([peptide, peptide], alleles=[allele1, allele2])
    assert value1 < 100, value1
    assert value2 > 4000, value2

    allele_to_sequence[allele3] = ALLELE_TO_SEQUENCE[allele3]
    predictor.allele_to_sequence = allele_to_sequence
    predictor.clear_cache()

    (value1, value2, value3) = predictor.predict(
        [peptide, peptide, peptide],
        alleles=[allele1, allele2, allele3])
    assert value1 < 100, value1
    assert value2 > 4000, value2







