from . import initialize
initialize()

import numpy
import pandas
import pytest

from mhcflurry import Class1AffinityPredictor, Class1NeuralNetwork
from mhcflurry.common import random_peptides
from mhcflurry.downloads import get_path

from mhcflurry.testing_utils import cleanup, startup



def setup_module():
    global PAN_ALLELE_PREDICTOR
    startup()
    PAN_ALLELE_PREDICTOR = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.combined"),
        optimization_level=0,)


def teardown_module():
    global PAN_ALLELE_PREDICTOR
    PAN_ALLELE_PREDICTOR = None
    cleanup()


def test_merge(predictors):
    pan_allele_predictor = predictors['pan-allele']

    assert len(pan_allele_predictor.class1_pan_allele_models) > 1
    peptides = random_peptides(100, length=9)
    peptides.extend(random_peptides(100, length=10))
    peptides = pandas.Series(peptides).sample(frac=1.0)

    alleles = pandas.Series(
        ["HLA-A*03:01", "HLA-B*57:01", "HLA-C*02:01"]
    ).sample(n=len(peptides), replace=True)

    predictions1 = pan_allele_predictor.predict(
        peptides=peptides, alleles=alleles)

    merged = Class1NeuralNetwork.merge(
        pan_allele_predictor.class1_pan_allele_models)
    merged_predictor = Class1AffinityPredictor(
        allele_to_sequence=pan_allele_predictor.allele_to_sequence,
        class1_pan_allele_models=[merged],
    )
    predictions2 = merged_predictor.predict(peptides=peptides, alleles=alleles)
    numpy.testing.assert_allclose(predictions1, predictions2, atol=0.1)
