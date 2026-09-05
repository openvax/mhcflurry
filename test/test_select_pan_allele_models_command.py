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

"""Unit tests for ``mhcflurry.select_pan_allele_models_command``."""
import inspect

import numpy
import pandas


def test_fold_col_parser_rejects_pandas_merge_suffixes():
    """Regression: when train_data has been pandas-merged with stale
    fold_* columns still attached, the saved metadata DataFrame can carry
    ``fold_0_x``/``fold_0_y`` etc. The select fold-col parser must
    surface that as a clean error rather than crashing later in
    ``int(col.split("_")[-1])``."""
    from mhcflurry import select_pan_allele_models_command as mod

    src = inspect.getsource(mod.run)
    assert r'r"^fold_\d+$"' in src or r"r'^fold_\d+$'" in src, (
        "fold col parser must use ^fold_<int>$ regex; otherwise "
        "fold_0_x slips through and crashes int() later"
    )


def test_train_peptide_hash_computed_before_allele_filter():
    """Regression: the per-fold train_peptide_hash must be computed on
    the raw saved ``train_data.csv.bz2`` *before* the canonicalizable-
    allele filter is applied. The saved file is exactly the data
    ``train_pan_allele_models_command`` trained on; if select hashes
    a strictly narrower subset (because ``filter_canonicalizable_alleles``
    drops a pseudogene/null annotation that train kept), the
    ``numpy.testing.assert_equal`` against ``training_info['train_peptide_hash']``
    fires a false-positive and aborts model selection.

    Verifies (via source inspection) that the hash is built over
    ``df.loc[df[col] == 1]`` while ``df`` is still the raw read, i.e.
    before the ``df = df.loc[df.allele.isin(alleles)]`` reassignment."""
    from mhcflurry import select_pan_allele_models_command as mod

    src = inspect.getsource(mod.run)
    hash_idx = src.index("make_train_peptide_hash(df.loc[df[col] == 1])")
    allele_filter_idx = src.index("df = df.loc[df.allele.isin(alleles)]")
    assert hash_idx < allele_filter_idx, (
        "train_peptide_hash must be computed before the canonicalizable-"
        "allele filter narrows df; otherwise hashes mismatch when any "
        "training allele is non-canonicalizable"
    )


def test_model_select_can_return_row_level_validation_predictions():
    from mhcflurry import Class1AffinityPredictor
    from mhcflurry import select_pan_allele_models_command as mod
    from mhcflurry.regression_target import to_ic50

    class FakeModel:
        def predict(self, peptides, alleles):
            assert len(peptides) == 4
            assert len(alleles.indices) == 4
            return numpy.array([50000.0, 10000.0, 100.0, 10.0])

        def clear_allele_representations(self):
            pass

    data = pandas.DataFrame({
        "allele": ["HLA-A*02:01"] * 4,
        "peptide": ["AAAAAAAAA", "AAAAAAAAC", "AAAAAAAAD", "AAAAAAAAE"],
        "measurement_value": [50000.0, 10000.0, 100.0, 10.0],
        "measurement_source": ["binding"] * 4,
        "fold_0": [0] * 4,
    })
    predictor = Class1AffinityPredictor(
        allele_to_sequence={"HLA-A*02:01": "A" * 34})

    result = mod.model_select(
        fold_num=0,
        models=[FakeModel()],
        min_models=1,
        max_models=1,
        save_validation_predictions=True,
        constant_data={"data": data, "input_predictor": predictor},
    )

    predictions = result["validation_predictions"]
    assert predictions["validation_row_index"].tolist() == [0, 1, 2, 3]
    assert predictions["fold_num"].tolist() == [0] * 4
    assert predictions["selected_model_indices"].tolist() == ["0"] * 4
    expected_01 = mod.from_ic50(
        numpy.array([50000.0, 10000.0, 100.0, 10.0]))
    numpy.testing.assert_allclose(
        predictions["affinity_prediction_01"], expected_01)
    numpy.testing.assert_allclose(
        predictions["affinity_prediction"],
        to_ic50(expected_01),
    )
    assert "fold_0" not in predictions
