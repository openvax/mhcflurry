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

import importlib
import inspect

import pytest

from mhcflurry import common, data_dependent_weights_initialization
from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.parallelism import worker_init


def test_class1_neural_network_public_helpers_remain_importable():
    import mhcflurry.class1_encoding as encoding
    import mhcflurry.class1_neural_network as old_module
    import mhcflurry.pytorch_sizing as sizing

    assert (
        old_module.compute_prediction_batch_size
        is sizing.compute_prediction_batch_size
    )
    assert (
        old_module.resolve_prediction_batch_size
        is sizing.resolve_prediction_batch_size
    )
    assert old_module.check_training_batch_fits is sizing.check_training_batch_fits
    assert (
        old_module.peptide_sequences_to_network_input
        is encoding.peptide_sequences_to_network_input
    )


@pytest.mark.usefixtures("preserve_global_random_state")
def test_worker_init_preserves_empty_gpu_assignment(monkeypatch):
    calls = []

    def fake_configure_pytorch(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "mhcflurry.parallelism.worker_runtime.configure_pytorch",
        fake_configure_pytorch,
    )

    worker_init(backend="auto", gpu_device_nums=[])

    assert calls == [{"backend": "auto", "gpu_device_nums": []}]


def test_removed_keras_era_aliases_are_absent():
    assert not hasattr(common, "configure_tensorflow")
    assert not hasattr(Class1NeuralNetwork, "keras_network_cache_key")
    assert not hasattr(Class1NeuralNetwork, "fit_generator")
    assert not hasattr(data_dependent_weights_initialization, "get_activations")
    assert "keras_backend" not in inspect.signature(worker_init).parameters
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mhcflurry.local_parallelism")
