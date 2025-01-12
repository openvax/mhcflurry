import torch
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from mhcflurry.predict_command import run as predict_run
from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor
from mhcflurry.torch_implementations import Class1AffinityPredictor as TorchPredictor
from mhcflurry.class1_presentation_predictor import Class1PresentationPredictor

logging.basicConfig(level=logging.INFO)

def analyze_network_architectures():
    """Compare network architectures and weights between Keras and PyTorch models"""
    logging.info("Starting network architecture analysis")
    
    try:
        # Load models
        from mhcflurry.downloads import get_default_class1_presentation_models_dir
        models_dir = get_default_class1_presentation_models_dir()
        logging.info(f"Using models directory: {models_dir}")

        presentation_predictor = Class1PresentationPredictor.load(models_dir)
        tf_predictor = presentation_predictor.affinity_predictor
        torch_predictor = TorchPredictor.load(models_dir)
        
        # Get first Keras model from ensemble
        tf_model = tf_predictor.neural_networks[0]
        network = tf_model.network()
        
        logging.info("\n=== Model Architecture Comparison ===")
        network.summary()
        
        logging.info("\n=== PyTorch Model ===")
        logging.info(str(torch_predictor))
        
        # Compare architectures layer by layer
        logging.info("\n=== Detailed Layer Comparison ===")
        
        keras_layers = []
        for layer in network.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                keras_layers.append({
                    'name': layer.name,
                    'type': type(layer).__name__,
                    'weight_shape': weights[0].shape,
                    'bias_shape': weights[1].shape,
                    'weights': weights[0],
                    'bias': weights[1]
                })
                
        torch_layers = []
        for name, module in torch_predictor.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.detach().cpu().numpy()
                bias = module.bias.detach().cpu().numpy()
                torch_layers.append({
                    'name': name,
                    'type': type(module).__name__,
                    'weight_shape': weight.shape,
                    'bias_shape': bias.shape,
                    'weights': weight,
                    'bias': bias
                })

        # Compare layer shapes and weights
        logging.info("\nLayer shape comparison:")
        for k, t in zip(keras_layers, torch_layers):
            logging.info(f"\nKeras layer: {k['name']}, PyTorch layer: {t['name']}")
            logging.info(f"Weight shapes - Keras: {k['weight_shape']}, PyTorch: {t['weight_shape']}")
            logging.info(f"Bias shapes - Keras: {k['bias_shape']}, PyTorch: {t['bias_shape']}")
            
            # Log shapes and stats separately since architectures differ
            logging.info(f"\nKeras weights shape: {k['weight_shape']}")
            logging.info(f"Keras weights stats - min: {k['weights'].min():.4f}, max: {k['weights'].max():.4f}, mean: {k['weights'].mean():.4f}")
            logging.info(f"Keras bias shape: {k['bias_shape']}")
            logging.info(f"Keras bias stats - min: {k['bias'].min():.4f}, max: {k['bias'].max():.4f}, mean: {k['bias'].mean():.4f}")
            
            logging.info(f"\nPyTorch weights shape: {t['weight_shape']}")
            logging.info(f"PyTorch weights stats - min: {t['weights'].min():.4f}, max: {t['weights'].max():.4f}, mean: {t['weights'].mean():.4f}")
            logging.info(f"PyTorch bias shape: {t['bias_shape']}")
            logging.info(f"PyTorch bias stats - min: {t['bias'].min():.4f}, max: {t['bias'].max():.4f}, mean: {t['bias'].mean():.4f}")

        # Test predictions
        test_peptides = ["SIINFEKL", "SIINFEKD"]
        test_alleles = ["HLA-A0201", "HLA-A0301"]
        
        logging.info("\n=== Prediction Comparison ===")
        
        keras_pred = tf_predictor.predict(test_peptides, alleles=test_alleles)
        torch_pred = torch_predictor.predict(test_peptides, alleles=test_alleles)
        
        logging.info("\nPrediction comparison:")
        for i, (k, t) in enumerate(zip(keras_pred, torch_pred)):
            logging.info(f"Peptide {test_peptides[i]}, Allele {test_alleles[i]}")
            logging.info(f"Keras: {k:.4f}, PyTorch: {t:.4f}, Diff: {abs(k-t):.4f}")

    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}", exc_info=True)

if __name__ == "__main__":
    analyze_network_architectures()
