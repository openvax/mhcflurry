import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True

import numpy
from numpy import testing
numpy.random.seed(0)
from tensorflow.random import set_seed
set_seed(2)

from nose.tools import eq_, assert_less, assert_greater, assert_almost_equal

import pandas
from sklearn.metrics import roc_auc_score

from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.downloads import get_path
from mhcflurry.common import random_peptides, normalize_allele_name, peptide_length_series

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup


def test_allele_specific_network_with_phosphopeptides():
    # Memorize the dataset.
    hyperparameters = dict(
        activation="tanh",
        layer_sizes=[16],
        max_epochs=500,
        early_stopping=False,
        validation_split=0.0,
        locally_connected_layers=[
            {
                "filters": 8,
                "activation": "tanh",
                "kernel_size": 3
            }
        ],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0)

    allele = "HLA-C*12:02"

    df = pandas.read_csv(
        get_path(
            "data_curated", "curated_training_data.csv.bz2"))
    df_allele = df.loc[
        df.allele == allele
    ]
    df_allele = df_allele.loc[df_allele.measurement_inequality.isin(["<", "="])]
    df_allele = df_allele.copy()
    df_allele["phospho"] = False

    df_negative = df.loc[
        df.allele == "HLA-A*02:01"
    ].sample(n=len(df_allele)).copy()
    df_negative["measurement_value"] = 40000
    df_negative["phospho"] = False

    phospho_df = pandas.read_csv("test/data/phospho.csv")
    phospho_df["measurement_value"] = 20000
    phospho_df["measurement_inequality"] = ">"
    phospho_df["phospho"] = True
    phospho_df["allele"] = phospho_df["allele"].map(normalize_allele_name)
    phospho_df.loc[phospho_df.allele == allele, "measurement_value"] = 100.0
    phospho_df.loc[phospho_df.allele == allele, "measurement_inequality"] = "<"

    print(phospho_df.groupby("allele").measurement_value.count())
    print(phospho_df.groupby("allele").measurement_value.mean())

    train_df = pandas.concat(
        [df_allele, df_negative, phospho_df], ignore_index=True)
    train_df["binder"] = train_df.measurement_value <= 500

    train_df = train_df.loc[
        (peptide_length_series(train_df.peptide) >= 8) &
        (peptide_length_series(train_df.peptide) <= 15)
    ]
    print(train_df)

    model1 = Class1NeuralNetwork(**hyperparameters)
    model1.fit(
        train_df.peptide.values,
        train_df.measurement_value.values)
    train_df["model1_prediction"] = model1.predict(train_df.peptide.values)

    overall_auc = roc_auc_score(train_df.binder, -train_df.model1_prediction)

    phospho_auc = roc_auc_score(
        train_df.loc[train_df.phospho].binder,
        -train_df.loc[train_df.phospho].model1_prediction)

    model2 = Class1NeuralNetwork(**hyperparameters)
    model2.fit(
        train_df.loc[~train_df.phospho].peptide.values,
        train_df.loc[~train_df.phospho].measurement_value.values)
    train_df["model2_prediction"] = model2.predict(train_df.peptide.values)

    exclude_phospho_overall_auc = roc_auc_score(
        train_df.binder, -train_df.model2_prediction)

    excluding_phospho_phospho_auc = roc_auc_score(
        train_df.loc[train_df.phospho].binder,
        -train_df.loc[train_df.phospho].model2_prediction)

    print("Train including phospho, overall AUC: ", overall_auc)
    print("Train including phospho, phospho AUC: ", phospho_auc)
    print("Train excluding phospho, overall AUC: ", exclude_phospho_overall_auc)
    print("Train excluding phospho, phospho AUC: ", excluding_phospho_phospho_auc)

    assert_greater(phospho_auc, excluding_phospho_phospho_auc)
    assert_greater(overall_auc, 0.8)
    assert_greater(phospho_auc, 0.8)
    assert_greater(exclude_phospho_overall_auc, 0.8)

