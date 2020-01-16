import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True
import re
import numpy
from numpy import testing
numpy.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(2)

from sklearn.metrics import roc_auc_score

from nose.tools import eq_, assert_less, assert_greater, assert_almost_equal

import pandas

from mhcflurry.class1_cleavage_neural_network import Class1CleavageNeuralNetwork
from mhcflurry.common import random_peptides

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup


def test_big():
    train_basic_network(num=100000)


def test_small():
    train_basic_network(num=10000)


def test_more():
    train_basic_network(
        num=10000,
        flanking_averages=False,
        convolutional_kernel_size=3,
        c_flank_length=0,
        n_flank_length=3,
        post_convolutional_dense_layer_sizes=[8])


def train_basic_network(num, do_assertions=True, **hyperparameters):
    use_hyperparameters = {
        "n_flank_length": 5,
        "c_flank_length": 5,
        "convolutional_kernel_size": 3,
    }
    use_hyperparameters.update(hyperparameters)

    df = pandas.DataFrame({
        "n_flank": random_peptides(num / 2, 10) + random_peptides(num / 2, 1),
        "c_flank": random_peptides(num, 10),
        "peptide": random_peptides(num / 2, 11) + random_peptides(num / 2, 8),
    }).sample(frac=1.0)

    n_cleavage_regex = "[AILQSV][SINFEKLH][MNPQYK]"

    def is_hit(n_flank, c_flank, peptide):
        if re.search(n_cleavage_regex, peptide):
            return False  # peptide is cleaved
        return bool(re.match(n_cleavage_regex, n_flank[-1:] + peptide))

    df["hit"] = [
        is_hit(row.n_flank, row.c_flank, row.peptide)
        for (_, row) in df.iterrows()
    ]

    train_df = df.sample(frac=0.9)
    test_df = df.loc[~df.index.isin(train_df.index)]

    print(
        "Generated dataset",
        len(df),
        "hits: ",
        df.hit.sum(),
        "frac:",
        df.hit.mean())

    network = Class1CleavageNeuralNetwork(**use_hyperparameters)
    network.fit(
        train_df.peptide.values,
        train_df.n_flank.values,
        train_df.c_flank.values,
        train_df.hit.values)

    for df in [train_df, test_df]:
        df["predictions"] = network.predict(
            df.peptide.values,
            df.n_flank.values,
            df.c_flank.values)

    train_auc = roc_auc_score(train_df.hit.values, train_df.predictions.values)
    test_auc = roc_auc_score(test_df.hit.values, test_df.predictions.values)

    print("Train auc", train_auc)
    print("Test auc", test_auc)

    if do_assertions:
        assert_greater(train_auc, 0.9)
        assert_greater(test_auc, 0.85)

    return network
