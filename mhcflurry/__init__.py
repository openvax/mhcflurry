"""
Class I MHC ligand prediction package
"""

from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_neural_network import Class1NeuralNetwork
from .class1_processing_predictor import Class1ProcessingPredictor
from .class1_processing_neural_network import Class1ProcessingNeuralNetwork
from .class1_presentation_predictor import Class1PresentationPredictor

from .version import __version__

__all__ = [
    "__version__",
    "Class1AffinityPredictor",
    "Class1NeuralNetwork",
    "Class1ProcessingPredictor",
    "Class1ProcessingNeuralNetwork",
    "Class1PresentationPredictor",
]
