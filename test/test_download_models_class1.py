from . import initialize
initialize()

import numpy
numpy.random.seed(0)

import pickle
import tempfile
import pytest

from mhcflurry import Class1AffinityPredictor, Class1NeuralNetwork

from mhcflurry.testing_utils import cleanup, startup

# Define a fixture to initialize and clean up predictors
@pytest.fixture(scope="module")
def downloaded_predictor():
    startup()
    yield Class1AffinityPredictor.load()
    cleanup()


def predict_and_check(
        downloaded_predictor,
        allele,
        peptide,
        expected_range=(0, 500)):

    print("\n%s" % (
        downloaded_predictor.predict_to_dataframe(
            peptides=[peptide],
            allele=allele,
            include_individual_model_predictions=True)))

    (prediction,) = downloaded_predictor.predict(allele=allele, peptides=[peptide])
    assert prediction >= expected_range[0], (downloaded_predictor, prediction)
    assert prediction <= expected_range[1], (downloaded_predictor, prediction)


def test_a1_titin_epitope_downloaded_models(downloaded_predictor):
    # Test the A1 Titin epitope ESDPIVAQY from
    #   Identification of a Titin-Derived HLA-A1-Presented Peptide
    #   as a Cross-Reactive Target for Engineered MAGE A3-Directed
    #   T Cells
    predict_and_check(downloaded_predictor, "HLA-A*01:01", "ESDPIVAQY")


def test_a1_mage_epitope_downloaded_models(downloaded_predictor):
    # Test the A1 MAGE epitope EVDPIGHLY from
    #   Identification of a Titin-Derived HLA-A1-Presented Peptide
    #   as a Cross-Reactive Target for Engineered MAGE A3-Directed
    #   T Cells
    predict_and_check(downloaded_predictor, "HLA-A*01:01", "EVDPIGHLY")


def test_a2_hiv_epitope_downloaded_models(downloaded_predictor):
    # Test the A2 HIV epitope SLYNTVATL from
    #    The HIV-1 HLA-A2-SLYNTVATL Is a Help-Independent CTL Epitope
    predict_and_check(downloaded_predictor, "HLA-A*02:01", "SLYNTVATL")


def test_caching(downloaded_predictor):
    if not downloaded_predictor.allele_to_sequence:
        # Only run this test on allele-specific predictors.
        Class1NeuralNetwork.KERAS_MODELS_CACHE.clear()
        downloaded_predictor.predict(
            peptides=["SIINFEKL"],
            allele="HLA-A*02:01")
        num_cached = len(Class1NeuralNetwork.KERAS_MODELS_CACHE)
        assert num_cached > 0


def test_downloaded_predictor_is_serializable(downloaded_predictor):
    predictor_copy = pickle.loads(pickle.dumps(downloaded_predictor))
    numpy.testing.assert_equal(
        downloaded_predictor.predict(
            ["RSKERAVVVAW"], allele="HLA-A*01:01")[0],
        predictor_copy.predict(
            ["RSKERAVVVAW"], allele="HLA-A*01:01")[0])


def test_downloaded_predictor_is_savable(downloaded_predictor):
    models_dir = tempfile.mkdtemp("_models")
    print(models_dir)
    downloaded_predictor.save(models_dir)
    predictor_copy = Class1AffinityPredictor.load(models_dir)

    numpy.testing.assert_equal(
        downloaded_predictor.predict(
            ["RSKERAVVVAW"], allele="HLA-A*01:01")[0],
        predictor_copy.predict(
            ["RSKERAVVVAW"], allele="HLA-A*01:01")[0])


def test_downloaded_predictor_gives_percentile_ranks(downloaded_predictor):
    predictions = downloaded_predictor.predict_to_dataframe(
        peptides=["SAQGQFSAV", "SAQGQFSAV"],
        alleles=["HLA-A*03:01", "HLA-C*01:02"])

    print(predictions)
    assert not predictions.prediction.isnull().any()
    assert not predictions.prediction_percentile.isnull().any()


