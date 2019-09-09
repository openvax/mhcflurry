"""
Utilities used in MHCflurry unit tests.
"""
from . import Class1NeuralNetwork


def cleanup():
    """
    Clear tensorflow session and other process-wide resources.
    """
    import keras.backend as K
    Class1NeuralNetwork.clear_model_cache()
    K.clear_session()
