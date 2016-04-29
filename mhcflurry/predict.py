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

import pandas as pd

from .class1_binding_predictor import Class1BindingPredictor
from .common import normalize_allele_name

def predict(alleles, peptides):
    result_dict = OrderedDict([
        ("allele", []),
        ("peptide", []),
        ("ic50", []),
    ])
    for allele in alleles:
        allele = normalize_allele_name(allele)
        model = Class1BindingPredictor.from_allele_name(allele)
        for i, ic50 in enumerate(model.predict_peptides_ic50(peptides)):
            result_dict["allele"].append(allele)
            result_dict["peptide"].append(peptides[i])
            result_dict["ic50"].append(ic50)
    return pd.DataFrame(result_dict)
