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
Utilities used in MHCflurry unit tests.
"""
from . import Class1NeuralNetwork
from .common import configure_pytorch


def startup():
    """
    Configure PyTorch for running unit tests.
    """
    configure_pytorch(num_threads=2)


def cleanup():
    """
    Clear PyTorch session and other process-wide resources.
    """
    Class1NeuralNetwork.clear_model_cache()
