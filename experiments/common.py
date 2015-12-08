#!/usr/bin/env python
#
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


from collections import defaultdict
import pandas as pd


def curry_dictionary(key_pair_dict, default_value=0.0):
    """
    Transform dictionary from pairs of keys to dict -> dict -> float
    """
    result = defaultdict(dict)
    for (a, b), value in key_pair_dict.items():
        result[a][b] = value
    return result


def uncurry_dictionary(curried_dict):
    """
    Transform dictionary from (key_a -> key_b -> float) to
    (key_a, key_b) -> float
    """
    result = {}
    for a, a_dict in curried_dict.items():
        for b, value in a_dict.items():
            result[(a, b)] = value
    return result


def matrix_to_dictionary(sims, allele_list):
    sims_dict = {}
    for i in range(sims.shape[0]):
        a = allele_list[i]
        for j in range(sims.shape[1]):
            b = allele_list[j]
            sims_dict[a, b] = sims[i, j]
    return sims_dict


def load_csv_binding_data_as_dict(
        csv_path,
        mhc_column_name="mhc",
        peptide_column_name="sequence",
        ic50_column_name="ic50"):
    """
    Given a path to a CSV file containing peptide-MHC binding data,
    load it as a dictionary mapping alleles to a dictionary peptide->IC50
    """
    df = pd.read_csv(csv_path)
    print("-- Read %d rows from %s" % (len(df), csv_path))
    return {
        allele: {
            peptide: ic50
            for (peptide, ic50)
            in zip(group[peptide_column_name], group[ic50_column_name])
        }
        for allele, group in df.groupby(mhc_column_name)
    }
