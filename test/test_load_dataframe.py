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
        peptide_length=None,
        max_ic50=MAX_IC50,
        sep=None,
        species_column_name="species",
        allele_column_name="mhc",
        peptide_column_name=None,
        peptide_length_column_name="peptide_length",
        ic50_column_name="meas",
        only_human=True
    Outputs:
        Tuple with dataframe and name of peptide column
"""


from mhcflurry.data import load_dataframe
import numpy as np
from tempfile import NamedTemporaryFile
from pandas import DataFrame

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

        for max_ic50 in [500.0, 50000.0]:
            df, peptide_column_name = load_dataframe(
                filename=binding_data_path,
                max_ic50=max_ic50,
                only_human=False)
            assert peptide_column_name == "sequence"
            assert len(df) == n_expected, \
                "Expected %d entries in DataFrame but got %d in %s" % (
                    n_expected,
                    len(df),
                    df)
            assert "regression_output" in df, df.columns
            expected_values = np.minimum(
                0,
                np.log(df["meas"]) / np.log(max_ic50)
            )
            np.allclose(df["regression_output"], expected_values)

def test_load_dataframe_human():
    df_expected = make_dummy_dataframe()
    human_mask = df_expected["species"] == "human"
    df_expected = df_expected[human_mask]
    n_expected = len(df_expected)

    with NamedTemporaryFile(mode="w") as f:
        binding_data_path = f.name
        df_expected.to_csv(binding_data_path, index=False)

        for max_ic50 in [500.0, 50000.0]:
            df, peptide_column_name = load_dataframe(
                filename=binding_data_path,
                max_ic50=max_ic50,
                only_human=True)
            assert peptide_column_name == "sequence"
            assert len(df) == n_expected, \
                "Expected %d entries in DataFrame but got %d in %s" % (
                    n_expected,
                    len(df),
                    df)
            assert "regression_output" in df, df.columns
            expected_values = np.minimum(
                0,
                np.log(df["meas"]) / np.log(max_ic50)
            )
            np.allclose(df["regression_output"], expected_values)
