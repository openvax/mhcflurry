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
