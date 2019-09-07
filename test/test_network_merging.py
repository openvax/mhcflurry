from nose.tools import eq_, assert_less, assert_greater, assert_almost_equal, assert_equal

import numpy
import pandas
from numpy import testing

numpy.random.seed(0)

import logging
logging.getLogger('tensorflow').disabled = True

from mhcflurry import Class1AffinityPredictor, Class1NeuralNetwork
from mhcflurry.common import random_peptides
from mhcflurry.downloads import get_path

ALLELE_SPECIFIC_PREDICTOR = Class1AffinityPredictor.load(
    get_path("models_class1", "models"), optimization_level=0)

PAN_ALLELE_PREDICTOR = Class1AffinityPredictor.load(
    get_path("models_class1_pan", "models.with_mass_spec"),
    optimization_level=0)


def test_merge():
    assert len(PAN_ALLELE_PREDICTOR.class1_pan_allele_models) > 1

    peptides = random_peptides(100, length=9)
    peptides.extend(random_peptides(100, length=10))
    peptides = pandas.Series(peptides).sample(frac=1.0)

    alleles = pandas.Series(
        ["HLA-A*03:01", "HLA-B*57:01", "HLA-C*02:01"]
    ).sample(n=len(peptides), replace=True)

    predictions1 = PAN_ALLELE_PREDICTOR.predict(
        peptides=peptides, alleles=alleles)

    merged = Class1NeuralNetwork.merge(
        PAN_ALLELE_PREDICTOR.class1_pan_allele_models)
    merged_predictor = Class1AffinityPredictor(
        allele_to_sequence=PAN_ALLELE_PREDICTOR.allele_to_sequence,
        class1_pan_allele_models=[merged],
    )
    predictions2 = merged_predictor.predict(peptides=peptides, alleles=alleles)
    numpy.testing.assert_allclose(predictions1, predictions2, atol=0.1)

