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

from dummy_predictors import (
    always_zero_predictor_with_unknown_AAs,
    always_one_predictor_with_unknown_AAs,
)
from mhcflurry import Ensemble

def test_ensemble_of_dummy_predictors():
    ensemble = Ensemble([
        always_one_predictor_with_unknown_AAs,
        always_zero_predictor_with_unknown_AAs])
    peptides = ["SYYFFYLLY"]
    y = ensemble.predict_peptides(peptides)
    assert len(y) == len(peptides)
    assert all(yi == 0.5 for yi in y)
