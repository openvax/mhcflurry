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

"""Unit tests for ``mhcflurry.select_processing_models_command``."""
import inspect

import numpy
import pandas


def test_fold_col_parser_rejects_pandas_merge_suffixes():
    """Regression: ``fold_0_x`` must not be parsed as a real fold column."""
    from mhcflurry import select_processing_models_command as mod

    src = inspect.getsource(mod.run)
    assert r'r"^fold_\d+$"' in src or r"r'^fold_\d+$'" in src, (
        "fold col parser must use ^fold_<int>$ regex; otherwise "
        "fold_0_x slips through and crashes int() later"
    )


def test_model_select_uses_numpy_prediction_matrix():
    """Greedy processing selection should not depend on pandas model columns."""
    from mhcflurry import select_processing_models_command as mod

    class FakeModel:
        def __init__(self, predictions):
            self.predictions = numpy.asarray(predictions)

        def predict_encoded(self, sequences):
            assert len(sequences) == len(self.predictions)
            return self.predictions

    data = pandas.DataFrame({
        "peptide": ["AAAA"] * 6,
        "n_flank": [""] * 6,
        "c_flank": [""] * 6,
        "hit": [0, 0, 0, 1, 1, 1],
        "fold_0": [0] * 6,
    })
    models = [
        FakeModel([0.1, 0.2, 0.9, 0.4, 0.8, 0.7]),
        FakeModel([0.1, 0.8, 0.2, 0.9, 0.4, 0.7]),
        FakeModel([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]),
    ]

    result = mod.model_select(
        fold_num=0,
        models=models,
        min_models=1,
        max_models=3,
        constant_data={"data": data})

    assert result["selected_indices"] == [1, 0]
    summary = result["summary"]
    assert summary.loc[0, "selected_in_round"] == 2
    assert summary.loc[1, "selected_in_round"] == 1
    assert pandas.isna(summary.loc[2, "selected_in_round"])
