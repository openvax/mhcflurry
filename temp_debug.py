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

def compare_layer_outputs():
    logging.info("Starting layer output comparison test")
    try:
        # Test data
        alleles = ["HLA-A0201", "HLA-A0301"]
        peptides = ["SIINFEKL", "SIINFEKD", "SIINFEKQ"]

        # Get models directory and load predictors
        from mhcflurry.downloads import get_default_class1_presentation_models_dir
        models_dir = get_default_class1_presentation_models_dir()
        logging.info(f"Using models directory: {models_dir}")

        logging.info("Loading predictors...")
        presentation_predictor = Class1PresentationPredictor.load(models_dir)
        tf_predictor = presentation_predictor.affinity_predictor
        torch_predictor = TorchPredictor.load(models_dir)
        logging.info("Predictors loaded successfully")

    # Add hooks to capture intermediate outputs in PyTorch
        torch_activations = {}
        def get_activation(name):
            def hook(model, input, output):
                torch_activations[name] = output.detach().cpu().numpy()
                logging.info(f"PyTorch layer {name} output shape: {output.shape}")
                logging.info(f"PyTorch layer {name} output first few values: {output.flatten()[:5]}")
            return hook

        # Register hooks for each layer
        logging.info("Registering PyTorch hooks...")
        for name, layer in torch_predictor.named_modules():
            if isinstance(layer, (torch.nn.Linear, torch.nn.BatchNorm1d)):
                layer.register_forward_hook(get_activation(name))
                logging.info(f"Registered hook for layer: {name}")

        # Get TensorFlow predictions and intermediate outputs
        logging.info("Setting up TensorFlow model...")
        import tensorflow as tf
        tf_model = tf_predictor.neural_networks[0]  # Get first model in ensemble
        network = tf_model.network()

        # Create intermediate model that maintains graph connectivity
        peptide_input = network.get_layer('peptide').input
        allele_input = network.get_layer('allele').input
        
        logging.info("Successfully got input layers")
        logging.info(f"Peptide input shape: {peptide_input.shape}")
        logging.info(f"Allele input shape: {allele_input.shape}")
        
        # Build list of layer outputs while maintaining connections
        layer_outputs = []
        tensor_dict = {}
        
        logging.info("Starting layer traversal...")
        
        # Process all layers
        for layer in network.layers:
            # Skip input layers since we handle them separately
            if layer.name in ['peptide', 'allele']:
                continue
                
            # Get layer inputs
            layer_inputs = []
            for node in layer._inbound_nodes:
                try:
                    inbound_layers = node.inbound_layers
                    if not isinstance(inbound_layers, list):
                        inbound_layers = [inbound_layers]
                except AttributeError:
                    continue
                
                for inp in inbound_layers:
                    if isinstance(inp, str):
                        continue
                    if inp.name == 'peptide':
                        layer_inputs.append(peptide_input)
                    elif inp.name == 'allele':
                        layer_inputs.append(allele_input) 
                    else:
                        layer_inputs.append(tensor_dict[inp.name])
            
            # Compute layer output
            if isinstance(layer, tf.keras.layers.Embedding):
                # Special handling for Embedding layers - they expect a single input
                if len(layer_inputs) > 0:
                    tensor_dict[layer.name] = layer(layer_inputs[0])
            elif isinstance(layer, tf.keras.layers.Flatten):
                # Special handling for Flatten layers - they expect a single input
                if len(layer_inputs) > 0:
                    tensor_dict[layer.name] = layer(layer_inputs[0])
            else:
                # For other layers, pass inputs as list or single tensor
                if len(layer_inputs) == 1:
                    tensor_dict[layer.name] = layer(layer_inputs[0])
                elif len(layer_inputs) > 1:
                    tensor_dict[layer.name] = layer(layer_inputs)
            
            layer_outputs.append(tensor_dict[layer.name])
        
        # Create the intermediate model
        try:
            tf_intermediate_model = tf.keras.Model(
                inputs={'peptide': peptide_input, 'allele': allele_input},
                outputs=layer_outputs
            )
            logging.info("Successfully created intermediate model")
            logging.info(f"Number of outputs: {len(layer_outputs)}")
        except Exception as e:
            logging.error(f"Failed to create intermediate model: {str(e)}")
            raise
        logging.info("TensorFlow model ready")

        # Prepare input data
        logging.info("Preparing input data...")
        from mhcflurry.encodable_sequences import EncodableSequences
        from mhcflurry.allele_encoding import AlleleEncoding
        
        # Prepare peptide input
        peptides_obj = EncodableSequences.create(peptides)
        encoded_peptides = torch_predictor.peptides_to_network_input(peptides_obj)
        
        # Prepare allele input
        allele_encoding = AlleleEncoding(alleles)
        allele_input, allele_representations = tf_predictor.allele_encoding_to_network_input(allele_encoding)

        logging.info(f"Peptide input shape: {encoded_peptides.shape}")
        logging.info(f"Allele input shape: {allele_input.shape}")

        # Get TensorFlow intermediate outputs with both inputs
        logging.info("Getting TensorFlow outputs...")
        tf_outputs = tf_intermediate_model.predict({
            'peptide': encoded_peptides,
            'allele': allele_input
        })
        for i, output in enumerate(tf_outputs):
            logging.info(f"TF layer {i} output shape: {output.shape}")
            logging.info(f"TF layer {i} output first few values: {output.flatten()[:5]}")
        
        # Get PyTorch intermediate outputs
        logging.info("Getting PyTorch outputs...")
        torch_predictor.eval()
        with torch.no_grad():
            encoded_tensor = torch.from_numpy(encoded_peptides).float().to(torch_predictor.device)
            torch_output = torch_predictor(encoded_tensor)

        # Compare outputs layer by layer
        logging.info("\n=== Layer-by-layer comparison ===")
        
        # Compare input encoding
        logging.info("\nInput encoding comparison:")
        logging.info(f"TF input shape: {encoded_peptides.shape}")
        logging.info(f"Torch input shape: {encoded_tensor.shape}")
        input_diff = np.abs(encoded_peptides - encoded_tensor.cpu().numpy()).max()
        logging.info(f"Input max diff: {input_diff}")

        # Compare each layer's output
        for i, (name, activation) in enumerate(torch_activations.items()):
            if i < len(tf_outputs):  # Make sure we have corresponding TF output
                tf_out = tf_outputs[i]
                torch_out = activation
                
                logging.info(f"\nLayer {name}:")
                logging.info(f"TF output shape: {tf_out.shape}")
                logging.info(f"Torch output shape: {torch_out.shape}")
                max_diff = np.abs(tf_out - torch_out).max()
                mean_diff = np.abs(tf_out - torch_out).mean()
                logging.info(f"Max difference: {max_diff}")
                logging.info(f"Mean difference: {mean_diff}")
                logging.info(f"TF first few values: {tf_out[0, :5]}")
                logging.info(f"Torch first few values: {torch_out[0, :5]}")

        # Compare final predictions
        logging.info("\n=== Final Predictions ===")
        tf_preds = tf_predictor.predict(peptides, alleles=alleles)
        torch_preds = torch_predictor.predict(peptides, alleles=alleles)

        logging.info(f"TF predictions: {tf_preds}")
        logging.info(f"Torch predictions: {torch_preds}")
        pred_diff = np.abs(tf_preds - torch_preds).max()
        logging.info(f"Max prediction difference: {pred_diff}")

        # Print model architectures
        logging.info("\n=== Model Architectures ===")
        logging.info("TensorFlow model:")
        tf_model.network().summary()
        logging.info("\nPyTorch model:")
        logging.info(str(torch_predictor))

    except Exception as e:
        logging.error(f"Error during comparison: {str(e)}", exc_info=True)

if __name__ == "__main__":
    compare_layer_outputs()
