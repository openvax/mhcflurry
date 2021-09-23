import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True


import numpy
numpy.random.seed(0)

import pickle
import tempfile

from numpy.testing import assert_equal

from mhcflurry import Class1AffinityPredictor, Class1NeuralNetwork

from mhcflurry.testing_utils import cleanup, startup

DOWNLOADED_PREDICTOR = None


def setup():
    global DOWNLOADED_PREDICTOR
    startup()
    DOWNLOADED_PREDICTOR = Class1AffinityPredictor.load()


def teardown():
    global DOWNLOADED_PREDICTOR
    DOWNLOADED_PREDICTOR = None
    cleanup()


def predict_and_check(
        allele,
        peptide,
        expected_range=(0, 500)):

    print("\n%s" % (
        DOWNLOADED_PREDICTOR.predict_to_dataframe(
            peptides=[peptide],
            allele=allele,
            include_individual_model_predictions=True)))

    (prediction,) = DOWNLOADED_PREDICTOR.predict(allele=allele, peptides=[peptide])
    assert prediction >= expected_range[0], (DOWNLOADED_PREDICTOR, prediction)
    assert prediction <= expected_range[1], (DOWNLOADED_PREDICTOR, prediction)


def test_a1_titin_epitope_downloaded_models():
    # Test the A1 Titin epitope ESDPIVAQY from
    #   Identification of a Titin-Derived HLA-A1-Presented Peptide
    #   as a Cross-Reactive Target for Engineered MAGE A3-Directed
    #   T Cells
    predict_and_check("HLA-A*01:01", "ESDPIVAQY")


def test_a1_mage_epitope_downloaded_models():
    # Test the A1 MAGE epitope EVDPIGHLY from
    #   Identification of a Titin-Derived HLA-A1-Presented Peptide
    #   as a Cross-Reactive Target for Engineered MAGE A3-Directed
    #   T Cells
    predict_and_check("HLA-A*01:01", "EVDPIGHLY")


def test_a2_hiv_epitope_downloaded_models():
    # Test the A2 HIV epitope SLYNTVATL from
    #    The HIV-1 HLA-A2-SLYNTVATL Is a Help-Independent CTL Epitope
    predict_and_check("HLA-A*02:01", "SLYNTVATL")


def test_caching():
    if not DOWNLOADED_PREDICTOR.allele_to_sequence:
        # Only run this test on allele-specific predictors.
        Class1NeuralNetwork.KERAS_MODELS_CACHE.clear()
        DOWNLOADED_PREDICTOR.predict(
            peptides=["SIINFEKL"],
            allele="HLA-A*02:01")
        num_cached = len(Class1NeuralNetwork.KERAS_MODELS_CACHE)
        assert num_cached > 0


def test_downloaded_predictor_is_serializable():
    predictor_copy = pickle.loads(pickle.dumps(DOWNLOADED_PREDICTOR))
    numpy.testing.assert_equal(
        DOWNLOADED_PREDICTOR.predict(
            ["RSKERAVVVAW"], allele="HLA-A*01:01")[0],
        predictor_copy.predict(
            ["RSKERAVVVAW"], allele="HLA-A*01:01")[0])


def test_downloaded_predictor_is_savable():
    models_dir = tempfile.mkdtemp("_models")
    print(models_dir)
    DOWNLOADED_PREDICTOR.save(models_dir)
    predictor_copy = Class1AffinityPredictor.load(models_dir)

    numpy.testing.assert_equal(
        DOWNLOADED_PREDICTOR.predict(
            ["RSKERAVVVAW"], allele="HLA-A*01:01")[0],
        predictor_copy.predict(
            ["RSKERAVVVAW"], allele="HLA-A*01:01")[0])


def test_downloaded_predictor_gives_percentile_ranks():
    predictions = DOWNLOADED_PREDICTOR.predict_to_dataframe(
        peptides=["SAQGQFSAV", "SAQGQFSAV"],
        alleles=["HLA-A*03:01", "HLA-C*01:02"])

    print(predictions)
    assert not predictions.prediction.isnull().any()
    assert not predictions.prediction_percentile.isnull().any()


