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

"""
Testing dataframe loading from CSV:

    load_dataframe
    Input:
        filename,
    Outputs:
        Tuple with following elements:
            - dataframe
            - name of allele column
            - name of peptide column
            - name of affinity column
"""


from mhcflurry.dataset_helpers import load_dataframe
import numpy as np
from tempfile import NamedTemporaryFile
from pandas import DataFrame
from nose.tools import eq_

def make_dummy_dataframe():
    dummy_ic50_values = np.array([
        50000.0,
        500.0,
        1.0,
    ])
    dummy_epitope_sequences = ["A" * 8, "A" * 9, "A" * 10]
    # make sure we prepared the test data correctly
    assert len(dummy_ic50_values) == len(dummy_epitope_sequences)
    dummy_binding_data = {
        "species": ["mouse", "human", "human"],
        "mhc": ["H-2-Kb", "HLA-A*02:01", "HLA-A*01:01"],
        "peptide_length": [len(s) for s in dummy_epitope_sequences],
        "sequence": dummy_epitope_sequences,
        "meas": dummy_ic50_values,
    }
    return DataFrame(dummy_binding_data)

def test_load_dataframe():
    df_expected = make_dummy_dataframe()
    n_expected = len(df_expected)
    with NamedTemporaryFile(mode="w") as f:
        binding_data_path = f.name
        df_expected.to_csv(binding_data_path, index=False)

        df, allele_column_name, peptide_column_name, affinity_column_name = \
            load_dataframe(filename=binding_data_path)
        eq_(allele_column_name, "mhc")
        eq_(peptide_column_name, "sequence")
        eq_(affinity_column_name, "meas")
        assert len(df) == n_expected, \
            "Expected %d entries in DataFrame but got %d in %s" % (
                n_expected,
                len(df),
                df)
