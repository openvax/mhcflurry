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
import os
import sys
from importlib import import_module


# Must run before importing modules that transitively import numpy/MKL. Some
# Linux conda images default MKL to INTEL threading, which aborts after PyTorch
# has loaded GNU libgomp. The GNU MKL threading layer is not available on every
# supported platform, so only select it on Linux and respect an explicit user
# choice everywhere.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

from .version import __version__

_PUBLIC_CLASSES = {
    "Class1AffinityPredictor": "class1_affinity_predictor",
    "Class1NeuralNetwork": "class1_neural_network",
    "Class1ProcessingPredictor": "class1_processing_predictor",
    "Class1ProcessingNeuralNetwork": "class1_processing_neural_network",
    "Class1PresentationPredictor": "class1_presentation_predictor",
    "HistogramPercentRankTransform": "histogram_percent_rank_transform",
    "CompactPercentRankTransform": "compact_percent_rank_transform",
}


def __getattr__(name):
    """Load public classes when requested, keeping CLI discovery lightweight."""
    if name not in _PUBLIC_CLASSES:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    value = getattr(import_module("." + _PUBLIC_CLASSES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "__version__",
    "Class1AffinityPredictor",
    "Class1NeuralNetwork",
    "Class1ProcessingPredictor",
    "Class1ProcessingNeuralNetwork",
    "Class1PresentationPredictor",
    "HistogramPercentRankTransform",
    "CompactPercentRankTransform",
]
