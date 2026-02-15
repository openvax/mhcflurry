import inspect

from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.common import configure_tensorflow
from mhcflurry.custom_loss import MSEWithInequalities, get_loss
from mhcflurry.data_dependent_weights_initialization import get_activations
from mhcflurry.local_parallelism import worker_init


def test_legacy_configure_tensorflow_entry_point():
    configure_tensorflow(backend="tensorflow", gpu_device_nums=None, num_threads=1)


def test_legacy_worker_init_signature_kept():
    params = inspect.signature(worker_init).parameters
    assert "keras_backend" in params


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
