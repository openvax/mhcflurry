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


@pytest.fixture(scope="module")
def predictors():
    return {"pan-allele": PAN_ALLELE_PREDICTOR}


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
