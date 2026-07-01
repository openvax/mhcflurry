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

import inspect
import pytest

from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.common import configure_tensorflow
from mhcflurry.custom_loss import MSEWithInequalities, get_loss
from mhcflurry.data_dependent_weights_initialization import get_activations
from mhcflurry.local_parallelism import worker_init


def test_old_local_parallelism_import_matches_new_parallelism_package():
    import mhcflurry.local_parallelism as old_module
    import mhcflurry.parallelism as new_package
    from mhcflurry.parallelism import (
        cli_args,
        planning,
        torch_compile,
        worker_pool,
        worker_runtime,
    )

    assert old_module is new_package
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


def test_class1_affinity_predictor_helper_imports_remain_compatible():
    import mhcflurry.class1_affinity_predictor as old_module
    from mhcflurry.affinity import calibration_sizing

    assert (
        old_module._peptide_sequences_fingerprint
        is calibration_sizing.peptide_sequences_fingerprint
    )
    assert old_module._CalibrationFastCache is calibration_sizing.CalibrationFastCache


def test_legacy_configure_tensorflow_entry_point():
    with pytest.warns(FutureWarning, match="configure_tensorflow"):
        configure_tensorflow(backend="tensorflow", gpu_device_nums=None, num_threads=1)


def test_legacy_worker_init_signature_kept():
    params = inspect.signature(worker_init).parameters
    assert "keras_backend" in params


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
    assert (
        Class1NeuralNetwork.keras_network_cache_key(network_json)
        == Class1NeuralNetwork.model_cache_key(network_json)
    )


def test_legacy_get_keras_loss_accessor():
    standard = get_loss("mse")
    assert standard.get_keras_loss() == standard.loss

    custom = MSEWithInequalities()
    assert callable(custom.get_keras_loss())


def test_legacy_get_activations_symbol_kept():
    params = inspect.signature(get_activations).parameters
    assert tuple(params) == ("model", "layer", "X_batch")
