import numpy
numpy.random.seed(0)

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
