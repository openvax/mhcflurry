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

import os
from collections import OrderedDict

import pandas as pd

from .paths import CLASS1_MODEL_DIRECTORY
from .mhc1_binding_predictor import Mhc1BindingPredictor

def predict(alleles, peptides):
    allele_dataframes = OrderedDict([])
    for allele in alleles:
        model = Mhc1BindingPredictor(allele=allele)
        result_dictionary = model.predict_peptides(peptides)
        allele_dataframes.append(df)
    return pd.concat(allele_dataframes)
