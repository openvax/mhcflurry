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
Tests for training and predicting using Class1 pan-allele models.
"""

from sklearn.metrics import roc_auc_score
import pandas
import pytest

from numpy.testing import assert_

from mhcflurry import Class1NeuralNetwork
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.downloads import get_path
from mhcflurry.pseudosequences import LEGACY_ALLELE_SEQUENCES_FILENAME

from mhcflurry.testing_utils import cleanup, startup

pytestmark = pytest.mark.downloads

@pytest.fixture(autouse=True, scope="module")
def setup_module():
    startup()
    yield
    cleanup()


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
    'minibatch_size': 256,
    'optimizer': 'rmsprop',
    'output_activation': 'sigmoid',
    'patience': 5,
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
    'random_negative_method': 'by_allele',
    'train_data': {},
    'validation_split': 0.1,
}


@pytest.fixture(scope="module")
def training_data():
    """Load bundles on first use so collecting this module needs no downloads."""
    allele_to_sequence = pandas.read_csv(
        get_path(
            "allele_sequences", LEGACY_ALLELE_SEQUENCES_FILENAME),
        index_col=0).sequence.to_dict()

    train_df = pandas.read_csv(
        get_path(
            "data_curated", "curated_training_data.affinity.csv.bz2"))

    train_df = train_df.loc[train_df.allele.isin(allele_to_sequence)]
    train_df = train_df.loc[train_df.peptide.str.len() >= 8]
    train_df = train_df.loc[train_df.peptide.str.len() <= 15]

    train_df = train_df.loc[
        train_df.allele.isin(train_df.allele.value_counts().iloc[:3].index)
    ]

    ms_hits_df = pandas.read_csv(
        get_path(
            "data_curated", "curated_training_data.csv.bz2"))
    ms_hits_df = ms_hits_df.loc[ms_hits_df.allele.isin(train_df.allele.unique())]
    ms_hits_df = ms_hits_df.loc[ms_hits_df.peptide.str.len() >= 8]
    ms_hits_df = ms_hits_df.loc[ms_hits_df.peptide.str.len() <= 15]
    ms_hits_df = ms_hits_df.loc[~ms_hits_df.peptide.isin(train_df.peptide)]

    print("Loaded %d training and %d ms hits" % (
        len(train_df), len(ms_hits_df)))
    return allele_to_sequence, train_df, ms_hits_df


@pytest.mark.slow
@pytest.mark.integration
def test_train_simple(training_data):
    allele_to_sequence, train_df, ms_hits_df = training_data

    # Reset random seeds to ensure reproducibility regardless of test order
    import numpy
    import random
    import torch
    numpy.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)

    network = Class1NeuralNetwork(**HYPERPARAMETERS)
    allele_encoding = AlleleEncoding(
        train_df.allele.values,
        allele_to_sequence=allele_to_sequence)
    network.fit(
        train_df.peptide.values,
        affinities=train_df.measurement_value.values,
        allele_encoding=allele_encoding,
        inequalities=train_df.measurement_inequality.values)

    validation_df = ms_hits_df.copy()
    validation_df["hit"] = 1

    decoys_df = ms_hits_df.copy()
    decoys_df["hit"] = 0
    decoys_df["allele"] = decoys_df.allele.sample(frac=1.0).values

    validation_df = pandas.concat([validation_df, decoys_df], ignore_index=True)

    predictions = network.predict(
        peptides=validation_df.peptide.values,
        allele_encoding=AlleleEncoding(
            validation_df.allele.values, borrow_from=allele_encoding))

    print(pandas.Series(predictions).describe())

    score = roc_auc_score(validation_df.hit, -1 * predictions)
    print("AUC", score)

    assert_(score > 0.6)
