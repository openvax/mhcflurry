# Copyright (c) 2016. Mount Sinai School of Medicine
#
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
'''
Load predictors

'''

import pickle

import pandas

from ..downloads import get_path
from ..common import normalize_allele_name


def from_allele_name(allele_name):
    """
    Load a predictor for an allele.

    Parameters
    ----------
    allele_name : class I allele name

    Returns
    ----------
    Class1BindingPredictor
    """
    global _ALLELE_PREDICTOR_CACHE
    allele_name = normalize_allele_name(allele_name)
    if allele_name in _ALLELE_PREDICTOR_CACHE:
        return _ALLELE_PREDICTOR_CACHE[allele_name]

    models_df = production_models_dataframe()
    predictor_name = models_df.ix[allele_name].predictor_name
    model_path = get_path(
        "models_class1_allele_specific_single",
        "models/%s.pickle" % predictor_name)

    with open(model_path, 'rb') as fd:
        predictor = pickle.load(fd)

    _ALLELE_PREDICTOR_CACHE[allele_name] = predictor
    return predictor
_ALLELE_PREDICTOR_CACHE = {}


def supported_alleles():
    """
    Return a list of the names of the alleles for which there are trained
    predictors.
    """
    return list(sorted(production_models_dataframe().allele))


def production_models_dataframe():
    """
    Return a pandas.DataFrame describing the currently available trained
    predictors.
    """
    global _PRODUCTION_MODELS_DATAFRAME
    if _PRODUCTION_MODELS_DATAFRAME is None:
        _PRODUCTION_MODELS_DATAFRAME = pandas.read_csv(
            get_path("models_class1_allele_specific_single", "production.csv"))
        _PRODUCTION_MODELS_DATAFRAME.index = (
            _PRODUCTION_MODELS_DATAFRAME.allele)
    return _PRODUCTION_MODELS_DATAFRAME
_PRODUCTION_MODELS_DATAFRAME = None
