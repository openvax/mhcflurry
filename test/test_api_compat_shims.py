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

import numpy
import pytest
import torch

from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.common import configure_tensorflow
from mhcflurry.custom_loss import MSEWithInequalities, get_loss
from mhcflurry.data_dependent_weights_initialization import (
    get_activations,
    get_activations_pytorch,
)
from mhcflurry.parallelism import worker_init


def test_old_local_parallelism_import_warns_and_matches_new_parallelism_package(
        import_deprecated_module):
    import mhcflurry.parallelism as new_package

    old_module = import_deprecated_module(
        "mhcflurry.local_parallelism", "mhcflurry.parallelism")
    from mhcflurry.parallelism import (
        cli_args,
        planning,
        torch_compile,
        worker_pool,
        worker_runtime,
    )

    assert old_module.worker_init is worker_runtime.worker_init
    assert old_module.add_local_parallelism_args is cli_args.add_local_parallelism_args
    assert old_module.resolve_local_parallelism_args is planning.resolve_local_parallelism_args
    assert (
        old_module.worker_pool_with_gpu_assignments
        is worker_pool.worker_pool_with_gpu_assignments
    )
    assert (
        old_module.resolve_torchinductor_compile_threads_env
        is torch_compile.resolve_torchinductor_compile_threads_env
    )
    assert old_module.__all__ == new_package.__all__


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


def test_legacy_configure_tensorflow_entry_point():
    with pytest.warns(FutureWarning, match="configure_tensorflow"):
        configure_tensorflow(backend="tensorflow", gpu_device_nums=None, num_threads=1)


def test_configure_tensorflow_warns_without_a_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mhcflurry.common.configure_pytorch", lambda **kwargs: calls.append(kwargs)
    )
    with pytest.warns(FutureWarning, match="configure_pytorch"):
        configure_tensorflow()
    assert calls == [
        {"backend": None, "gpu_device_nums": None, "num_threads": None}
    ]


@pytest.mark.usefixtures("preserve_global_random_state")
def test_legacy_worker_init_keras_backend_warns_and_is_ignored(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mhcflurry.parallelism.worker_runtime.configure_pytorch",
        lambda **kwargs: calls.append(kwargs),
    )
    with pytest.warns(DeprecationWarning, match="keras_backend"):
        worker_init(
            keras_backend="tensorflow-cpu", backend="auto", gpu_device_nums=[]
        )
    assert calls == [{"backend": "auto", "gpu_device_nums": []}]


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


def test_legacy_cache_key_alias():
    network_json = (
        '{"dense_layer_l1_regularization": 0.1, '
        '"dense_layer_l2_regularization": 0.2, "layer_sizes": [8]}'
    )
    with pytest.warns(DeprecationWarning, match="model_cache_key"):
        legacy = Class1NeuralNetwork.keras_network_cache_key(network_json)
    assert legacy == Class1NeuralNetwork.model_cache_key(network_json)


def test_legacy_fit_generator_warns_and_forwards(monkeypatch):
    calls = []
    monkeypatch.setattr(
        Class1NeuralNetwork,
        "fit_streaming_batches",
        lambda self, **kwargs: calls.append(kwargs) or "result",
    )
    network = Class1NeuralNetwork()
    with pytest.warns(DeprecationWarning, match="fit_streaming_batches"):
        result = network.fit_generator("generator", "peptides", "affinities", epochs=3)
    assert result == "result"
    assert calls[0]["generator"] == "generator"
    assert calls[0]["validation_peptide_encoding"] == "peptides"
    assert calls[0]["validation_affinities"] == "affinities"
    assert calls[0]["epochs"] == 3


def test_legacy_get_keras_loss_accessor():
    standard = get_loss("mse")
    with pytest.warns(DeprecationWarning, match="loss attribute"):
        assert standard.get_keras_loss() == standard.loss

    custom = MSEWithInequalities()
    with pytest.warns(DeprecationWarning, match="loss attribute"):
        assert custom.get_keras_loss(reduction="sum") == custom.loss


def test_legacy_get_activations_warns_and_matches_pytorch_helper():
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Linear(3, 2)

        def forward(self, inputs):
            return self.dense(inputs["peptide"])

    model = Network()
    batch = {"peptide": numpy.ones((4, 3), dtype="float32")}
    expected = get_activations_pytorch(model, "dense", batch)
    with pytest.warns(DeprecationWarning, match="get_activations_pytorch"):
        legacy = get_activations(model, "dense", batch)
    numpy.testing.assert_array_equal(legacy, expected)
