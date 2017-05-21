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

import numpy as np
from mhcflurry import Class1NeuralNetwork

class Dummy9merIndexEncodingModel(object):
    """
    Dummy molde used for testing the pMHC binding predictor.
    """
    def __init__(self, constant_output_value=0):
        self.constant_output_value = constant_output_value

    def predict(self, X, verbose=False):
        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2
        n_rows, n_cols = X.shape
        n_cols == 9, "Expected 9mer index input input, got %d columns" % (
            n_cols,)
        return np.ones(n_rows, dtype=float) * self.constant_output_value

always_zero_predictor_with_unknown_AAs = Class1NeuralNetwork(
    model=Dummy9merIndexEncodingModel(0),
    allow_unknown_amino_acids=True)

always_zero_predictor_without_unknown_AAs = Class1NeuralNetwork(
    model=Dummy9merIndexEncodingModel(0),
    allow_unknown_amino_acids=False)


always_one_predictor_with_unknown_AAs = Class1NeuralNetwork(
    model=Dummy9merIndexEncodingModel(1),
    allow_unknown_amino_acids=True)

always_one_predictor_without_unknown_AAs = Class1NeuralNetwork(
    model=Dummy9merIndexEncodingModel(1),
    allow_unknown_amino_acids=False)
