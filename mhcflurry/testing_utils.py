"""
Utilities used in MHCflurry unit tests.
"""
from . import Class1NeuralNetwork
from .common import set_keras_backend


def startup():
    """
    Configure Keras backend for running unit tests.
    """
    set_keras_backend("tensorflow-cpu", num_threads=2)


def cleanup():
    """
    Clear tensorflow session and other process-wide resources.
    """
    import keras.backend as K
    Class1NeuralNetwork.clear_model_cache()
    K.clear_session()
