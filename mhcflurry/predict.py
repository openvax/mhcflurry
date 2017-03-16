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

import pandas as pandas

from .class1_allele_specific import class1_single_model_multi_allele_predictor
from .common import normalize_allele_name, UnsupportedAllele


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
        predictor = class1_single_model_multi_allele_predictor.get_downloaded_predictor()
    result_dict = OrderedDict([
        ("Allele", []),
        ("Peptide", []),
        ("Prediction", []),
    ])
    for allele in alleles:
        allele = normalize_allele_name(allele)
        for i, ic50 in enumerate(predictor.predict_for_allele(allele, peptides)):
            result_dict["Allele"].append(allele)
            result_dict["Peptide"].append(peptides[i])
            result_dict["Prediction"].append(ic50)
    return pandas.DataFrame(result_dict)
