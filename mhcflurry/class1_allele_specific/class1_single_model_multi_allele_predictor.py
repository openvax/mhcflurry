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
from __future__ import (
    print_function,
    division,
    absolute_import,
)
import pickle
from os.path import join

import pandas

from ..downloads import get_path
from ..common import normalize_allele_name, UnsupportedAllele

CACHED_PREDICTOR = None


def from_allele_name(allele_name):
    """
    Load a single-allele predictor.

    Parameters
    ----------
    allele_name : class I allele name

    Returns
    ----------
    Class1BindingPredictor
    """
    return get_downloaded_predictor().predictor_for_allele(allele_name)


def supported_alleles():
    """
    Return a list of the names of the alleles for which there are trained
    predictors in the default laoder.
    """
    return get_downloaded_predictor().supported_alleles


def get_downloaded_predictor():
    """
    Return a Class1AlleleSpecificPredictorLoader that uses downloaded models.
    """
    global CACHED_PREDICTOR

    # Some of the unit tests manipulate the downloads directory configuration
    # so get_path here may return different results in the same Python process.
    # For this reason we check the path and invalidate the loader if it's
    # different.
    path = get_path("models_class1_allele_specific_single")
    if CACHED_PREDICTOR is None or path != CACHED_PREDICTOR.path:
        CACHED_PREDICTOR = (
            Class1SingleModelMultiAllelePredictor
                .load_from_download_directory(path))
    return CACHED_PREDICTOR


class Class1SingleModelMultiAllelePredictor(object):
    """
    Factory for Class1BindingPredictor instances that are stored on disk
    using this directory structure:

        production.csv - Manifest file giving information on all models

        models/ - directory of models with names given in the manifest file
            MODEL-BAR.pickle
            MODEL-FOO.pickle
            ...
    """

    @staticmethod
    def load_from_download_directory(directory):
        return Class1SingleModelMultiAllelePredictor(directory)

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

    def predictor_for_allele(self, allele):
        """
        Load a predictor for an allele.

        Parameters
        ----------
        allele : class I allele name

        Returns
        ----------
        Class1BindingPredictor
        """
        allele = normalize_allele_name(allele)
        if allele not in self.predictors_cache:
            try:
                predictor_name = self.df.ix[allele].predictor_name
            except KeyError:
                raise UnsupportedAllele(
                    "No models for allele '%s'. Alleles with models: %s"
                    " in models file: %s" % (
                        allele,
                        ' '.join(self.supported_alleles),
                        self.path_to_models_csv))

            model_path = join(self.path, "models", predictor_name + ".pickle")
            with open(model_path, 'rb') as fd:
                self.predictors_cache[allele] = pickle.load(fd)
        return self.predictors_cache[allele]

    def predict(self, measurement_collection):
        if (measurement_collection.df.measurement_type != "affinity").any():
            raise ValueError("Only affinity measurements supported")

        result = pandas.Series(
            index=measurement_collection.df.index)
        for (allele, sub_df) in measurement_collection.df.groupby("allele"):
            result.loc[sub_df.index] = self.predict_for_allele(
                allele, sub_df.peptide.values)
        assert not result.isnull().any()
        return result

    def predict_for_allele(self, allele, peptides):
        predictor = self.predictor_for_allele(allele)
        result = predictor.predict(peptides)
        assert len(result) == len(peptides)
        return result
