import tempfile
import shutil
import logging
import warnings
import traceback
import sys

import numpy
import pandas
numpy.random.seed(0)

from mhcflurry import Class1AffinityPredictor

from nose.tools import eq_, assert_raises
from numpy import testing

from mhcflurry.downloads import get_path
from mhcflurry.testing_utils import cleanup, startup

DOWNLOADED_PREDICTOR = Class1AffinityPredictor.load()


def setup():
    global DOWNLOADED_PREDICTOR
    startup()
    DOWNLOADED_PREDICTOR = Class1AffinityPredictor.load()
    logging.basicConfig(level=logging.DEBUG)


def teardown():
    global DOWNLOADED_PREDICTOR
    DOWNLOADED_PREDICTOR = None
    cleanup()


# To hunt down a weird warning we were seeing in pandas.
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
warnings.showwarning = warn_with_traceback


def predict_and_check(
        allele,
        peptide,
        predictor=DOWNLOADED_PREDICTOR,
        expected_range=(0, 500)):

    def debug():
        print("\n%s" % (
            predictor.predict_to_dataframe(
                peptides=[peptide],
                allele=allele,
                include_individual_model_predictions=True)))

        (prediction,) = predictor.predict(allele=allele, peptides=[peptide])
        assert prediction >= expected_range[0], (predictor, prediction, debug())
        assert prediction <= expected_range[1], (predictor, prediction, debug())


def test_a1_known_epitopes_in_newly_trained_model():
    allele = "HLA-A*01:01"
    df = pandas.read_csv(
        get_path(
            "data_curated", "curated_training_data.affinity.csv.bz2"))
    df = df.loc[
        (df.allele == allele) &
        (df.peptide.str.len() >= 8) &
        (df.peptide.str.len() <= 15)
    ]

    hyperparameters = {
        "max_epochs": 100,
        "patience": 10,
        "early_stopping": True,
        "validation_split": 0.2,

        "random_negative_rate": 0.0,
        "random_negative_constant": 25,

        "peptide_amino_acid_encoding": "BLOSUM62",
        "use_embedding": False,
        "kmer_size": 15,
        "batch_normalization": False,
        "locally_connected_layers": [
            {
                "filters": 8,
                "activation": "tanh",
                "kernel_size": 3
            }
        ],
        "activation": "relu",
        "output_activation": "sigmoid",
        "layer_sizes": [
            32
        ],
        "random_negative_affinity_min": 20000.0,
        "random_negative_affinity_max": 50000.0,
        "dense_layer_l1_regularization": 0.001,
        "dropout_probability": 0.0
    }

    predictor = Class1AffinityPredictor()
    predictor.fit_allele_specific_predictors(
        n_models=2,
        architecture_hyperparameters_list=[hyperparameters],
        allele=allele,
        peptides=df.peptide.values,
        affinities=df.measurement_value.values,
        verbose=0,
    )

    predict_and_check("HLA-A*01:01", "EVDPIGHLY", predictor=predictor)

    models_dir = tempfile.mkdtemp("_models")
    print(models_dir)
    predictor.save(models_dir)
    predictor2 = Class1AffinityPredictor.load(models_dir)
    predict_and_check("HLA-A*01:01", "EVDPIGHLY", predictor=predictor2)
    shutil.rmtree(models_dir)

    predictor3 = Class1AffinityPredictor(
        allele_to_allele_specific_models={
            allele: [predictor.allele_to_allele_specific_models[allele][0]]
        })
    predict_and_check("HLA-A*01:01", "EVDPIGHLY", predictor=predictor3)
    models_dir = tempfile.mkdtemp("_models")
    print(models_dir)
    predictor3.save(models_dir)
    predictor4 = Class1AffinityPredictor.load(models_dir)
    predict_and_check("HLA-A*01:01", "EVDPIGHLY", predictor=predictor4)
    shutil.rmtree(models_dir)


def test_class1_affinity_predictor_a0205_memorize_training_data():
    # Memorize the dataset.
    hyperparameters = dict(
        activation="tanh",
        layer_sizes=[64],
        max_epochs=100,
        early_stopping=False,
        validation_split=0.0,
        locally_connected_layers=[],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0)

    allele = "HLA-A*02:05"

    df = pandas.read_csv(
        get_path(
            "data_curated", "curated_training_data.affinity.csv.bz2"))
    df = df.loc[
        df.allele == allele
    ]
    df = df.loc[
        df.peptide.str.len() == 9
    ]
    df = df.loc[
        df.measurement_type == "quantitative"
    ]
    df = df.loc[
        df.measurement_source == "kim2014"
    ]

    predictor = Class1AffinityPredictor()
    predictor.fit_allele_specific_predictors(
        n_models=2,
        architecture_hyperparameters_list=[hyperparameters],
        allele=allele,
        peptides=df.peptide.values,
        affinities=df.measurement_value.values,
        verbose=0,
    )
    predictor.calibrate_percentile_ranks(num_peptides_per_length=1000)
    ic50_pred = predictor.predict(df.peptide.values, allele=allele)
    ic50_true = df.measurement_value.values
    eq_(len(ic50_pred), len(ic50_true))
    testing.assert_allclose(
        numpy.log(ic50_pred),
        numpy.log(ic50_true),
        rtol=0.2,
        atol=0.2)

    ic50_pred_df = predictor.predict_to_dataframe(
        df.peptide.values, allele=allele)
    print(ic50_pred_df)
    assert 'prediction_percentile' in ic50_pred_df.columns
    assert ic50_pred_df.prediction_percentile.isnull().sum() == 0

    ic50_pred_df2 = predictor.predict_to_dataframe(
        df.peptide.values,
        allele=allele,
        include_individual_model_predictions=True)
    print(ic50_pred_df2)

    # Test an unknown allele
    print("Starting unknown allele check")
    eq_(predictor.supported_alleles, [allele])
    ic50_pred = predictor.predict(
        df.peptide.values,
        allele="HLA-A*02:01",
        throw=False)
    assert numpy.isnan(ic50_pred).all()

    assert_raises(
        ValueError,
        predictor.predict,
        df.peptide.values,
        allele="HLA-A*02:01")


    eq_(predictor.supported_alleles, [allele])
    assert_raises(
        ValueError,
        predictor.predict,
        ["AAAAA"],  # too short
        allele=allele)
    assert_raises(
        ValueError,
        predictor.predict,
        ["AAAAAAAAAAAAAAAAAAAA"],  # too long
        allele=allele)
    ic50_pred = predictor.predict(
        ["AAAAA", "AAAAAAAAA", "AAAAAAAAAAAAAAAAAAAA"],
        allele=allele,
        throw=False)
    assert numpy.isnan(ic50_pred[0])
    assert not numpy.isnan(ic50_pred[1])
    assert numpy.isnan(ic50_pred[2])


def test_no_nans():
    df = DOWNLOADED_PREDICTOR.predict_to_dataframe(
            alleles=["A02:01", "A02:02"],
            peptides=["SIINFEKL", "SIINFEKLL"])
    print(df)
    assert not df.isnull().any().any()


def test_predict_implementations_equivalent():
    for allele in ["HLA-A02:01", "A02:02"]:
        for centrality_measure in ["mean", "robust_mean"]:
            peptides = ["SIINFEKL", "SYYNFIIIKL", "SIINKFELQY"]

            pred1 = DOWNLOADED_PREDICTOR.predict(
                allele=allele,
                peptides=peptides + ["SSSN"],
                throw=False,
                centrality_measure=centrality_measure)
            pred2 = DOWNLOADED_PREDICTOR.predict_to_dataframe(
                allele=allele,
                peptides=peptides + ["SSSN"],
                throw=False,
                centrality_measure=centrality_measure).prediction.values
            testing.assert_almost_equal(pred1, pred2, decimal=2)

            pred1 = DOWNLOADED_PREDICTOR.predict(
                allele=allele,
                peptides=peptides,
                centrality_measure=centrality_measure)
            pred2 = DOWNLOADED_PREDICTOR.predict_to_dataframe(
                allele=allele,
                peptides=peptides,
                centrality_measure=centrality_measure).prediction.values
            testing.assert_almost_equal(pred1, pred2, decimal=2)

