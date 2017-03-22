# Copyright (c) 2015. Mount Sinai School of Medicine
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

from collections import OrderedDict

import pandas
import numpy

from .class1_allele_specific_ensemble import class1_ensemble_multi_allele_predictor
from .common import normalize_allele_name, UnsupportedAllele
from .peptide_encoding import encode_peptides


def predict(alleles, peptides, predictor=None):
    """
    Make predictions across all combinations of the specified alleles and
    peptides.

    Parameters
    ----------
    alleles : list of str
        Names of alleles to make predictions for.

    peptides : list of str
        Peptide amino acid sequences.

    predictor : Predictor to use. Defaults to downloaded Class1SingleModelMultiAllelePredictor.

    Returns DataFrame with columns "Allele", "Peptide", and "Prediction"
    """
    if predictor is None:
        predictor = class1_ensemble_multi_allele_predictor.get_downloaded_predictor()

    if len(peptides) == 0 or len(alleles) == 0:
        return pandas.DataFrame(columns=["Peptide", "Allele", "Prediction"])

    peptides = numpy.unique(peptides)
    encoded_peptides = encode_peptides(peptides)
    result_dfs = []
    result_df = pandas.DataFrame()
    result_df["Peptide"] = peptides
    for allele in alleles:
        allele = normalize_allele_name(allele)
        predictions = predictor.predict_for_allele(allele, encoded_peptides)
        result_df = result_df.copy()
        result_df["Allele"] = allele
        result_df["Prediction"] = predictions
        result_dfs.append(result_df)
    return pandas.concat(result_dfs, ignore_index=True)
