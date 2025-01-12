import numpy as np
import torch
from nose.tools import eq_
from numpy.testing import assert_array_almost_equal

from ..torch_implementations import (
    Class1AffinityPredictor,
    to_torch,
    to_numpy
)
from ..common import configure_tensorflow

def test_affinity_predictor_matches_keras():
    """Test that PyTorch affinity predictor gives identical results to Keras."""
    configure_tensorflow()
    from tf_keras.models import Sequential
    from tf_keras.layers import Dense

    # Create a simple test network in Keras
    keras_model = Sequential([
        Dense(64, activation='tanh', input_shape=(128,)),
        Dense(32, activation='tanh'),
        Dense(1, activation='sigmoid')
    ])

    # Create matching PyTorch network
    torch_model = Class1AffinityPredictor(
        input_size=128,
        peptide_dense_layer_sizes=[],
        layer_sizes=[64, 32],
        activation='tanh',
        output_activation='sigmoid'
    )

    # Load Keras weights into PyTorch model
    torch_model.load_weights_from_keras(keras_model)

    # Test on random input
    test_input = np.random.rand(10, 128).astype('float32')
    
    keras_output = keras_model.predict(test_input)
    torch_output = to_numpy(torch_model(test_input))

    assert_array_almost_equal(
        keras_output, 
        torch_output,
        decimal=4)

def test_to_torch():
    """Test numpy to torch conversion."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = to_torch(x)
    assert isinstance(t, torch.Tensor)
    assert_array_almost_equal(x, to_numpy(t))

def test_to_numpy():
    """Test torch to numpy conversion."""
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = to_numpy(t)
    assert isinstance(x, np.ndarray)
    assert_array_almost_equal(x, t.numpy())
