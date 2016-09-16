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
from os.path import join

import pandas

from ..downloads import get_path
from ..common import normalize_allele_name

CACHED_LOADER = None


def from_allele_name(allele_name):
    """
    Load a predictor for an allele using the default loader.

    Parameters
    ----------
    allele_name : class I allele name

    Returns
    ----------
    Class1BindingPredictor
    """
    return get_loader_for_downloaded_models().from_allele_name(allele_name)


def supported_alleles():
    """
    Return a list of the names of the alleles for which there are trained
    predictors in the default laoder.
    """
    return get_loader_for_downloaded_models().supported_alleles


def get_loader_for_downloaded_models():
    """
    Return a Class1AlleleSpecificPredictorLoader that uses downloaded models.
    """
    global CACHED_LOADER

    # Some of the unit tests manipulate the downloads directory configuration
    # so get_path here may return different results in the same Python process.
    # For this reason we check the path and invalidate the loader if it's
    # different.
    path = get_path("models_class1_allele_specific_single")
    if CACHED_LOADER is None or path != CACHED_LOADER.path:
        CACHED_LOADER = Class1AlleleSpecificPredictorLoader(path)
    return CACHED_LOADER


class Class1AlleleSpecificPredictorLoader(object):
    """
    Factory for Class1BindingPredictor instances that are stored on disk
    using this directory structure:

        production.csv - Manifest file giving information on all models

        models/ - directory of models with names given in the manifest file
            MODEL-BAR.pickle
            MODEL-FOO.pickle
            ...
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path : string
            Path to directory containing manifest and models
        """
        self.path = path
        self.path_to_models_csv = join(path, "production.csv")
        self.df = pandas.read_csv(self.path_to_models_csv)
        self.df.index = self.df["allele"]
        self.supported_alleles = list(sorted(self.df.allele))
        self.predictors_cache = {}

    def from_allele_name(self, allele_name):
        """
        Load a predictor for an allele.

        Parameters
        ----------
        allele_name : class I allele name

        Returns
        ----------
        Class1BindingPredictor
        """
        allele_name = normalize_allele_name(allele_name)
        if allele_name not in self.predictors_cache:
            try:
                predictor_name = self.df.ix[allele_name].predictor_name
            except KeyError:
                raise ValueError(
                    "No models for allele '%s'. Alleles with models: %s"
                    " in models file: %s" % (
                        allele_name,
                        ' '.join(self.supported_alleles),
                        self.path_to_models_csv))

            model_path = join(self.path, "models", predictor_name + ".pickle")
            with open(model_path, 'rb') as fd:
                self.predictors_cache[allele_name] = pickle.load(fd)
        return self.predictors_cache[allele_name]
