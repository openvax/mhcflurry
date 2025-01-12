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
        
        logging.info("\n=== Keras Model Architecture ===")
        network.summary()
        
        logging.info("\n=== PyTorch Model Architecture ===")
        logging.info(str(torch_predictor))
        
        # Analyze layer dimensions
        logging.info("\n=== Layer-by-layer Analysis ===")
        
        for layer in network.layers:
            logging.info(f"\nKeras layer: {layer.name}")
            logging.info(f"Type: {type(layer).__name__}")
            logging.info(f"Input shape: {layer.input_shape}")
            logging.info(f"Output shape: {layer.output_shape}")
            
            if isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                logging.info(f"Weight shape: {weights[0].shape}")
                logging.info(f"Bias shape: {weights[1].shape}")
                logging.info(f"Weight stats - min: {weights[0].min():.4f}, max: {weights[0].max():.4f}, mean: {weights[0].mean():.4f}")
                logging.info(f"Bias stats - min: {weights[1].min():.4f}, max: {weights[1].max():.4f}, mean: {weights[1].mean():.4f}")

        logging.info("\n=== PyTorch Layer Analysis ===")
        for name, module in torch_predictor.named_modules():
            if len(name) > 0:  # Skip the root module
                logging.info(f"\nPyTorch layer: {name}")
                logging.info(f"Type: {type(module).__name__}")
                
                if isinstance(module, torch.nn.Linear):
                    logging.info(f"Weight shape: {module.weight.shape}")
                    logging.info(f"Bias shape: {module.bias.shape}")
                    weight = module.weight.detach().cpu().numpy()
                    bias = module.bias.detach().cpu().numpy()
                    logging.info(f"Weight stats - min: {weight.min():.4f}, max: {weight.max():.4f}, mean: {weight.mean():.4f}")
                    logging.info(f"Bias stats - min: {bias.min():.4f}, max: {bias.max():.4f}, mean: {bias.mean():.4f}")

        # Compare input/output dimensions
        test_peptides = ["SIINFEKL", "SIINFEKD"]
        test_alleles = ["HLA-A0201", "HLA-A0301"]
        
        from mhcflurry.encodable_sequences import EncodableSequences
        peptides_obj = EncodableSequences.create(test_peptides)
        
        logging.info("\n=== Input/Output Dimension Analysis ===")
        
        # Analyze Keras input dimensions
        encoded_peptides_keras = tf_predictor.predict(test_peptides, alleles=test_alleles)
        logging.info(f"Keras prediction shape: {encoded_peptides_keras.shape}")
        
        # Analyze PyTorch input dimensions  
        encoded_peptides_torch = torch_predictor.predict(test_peptides, alleles=test_alleles)
        logging.info(f"PyTorch prediction shape: {encoded_peptides_torch.shape}")

    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}", exc_info=True)

if __name__ == "__main__":
    analyze_network_architectures()
