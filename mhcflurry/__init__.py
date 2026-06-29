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
Class I MHC ligand prediction package
"""

from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_neural_network import Class1NeuralNetwork
from .class1_processing_predictor import Class1ProcessingPredictor
from .class1_processing_neural_network import Class1ProcessingNeuralNetwork
from .class1_presentation_predictor import Class1PresentationPredictor

from .version import __version__

__all__ = [
    "__version__",
    "Class1AffinityPredictor",
    "Class1NeuralNetwork",
    "Class1ProcessingPredictor",
    "Class1ProcessingNeuralNetwork",
    "Class1PresentationPredictor",
]
