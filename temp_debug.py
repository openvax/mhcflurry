import torch
import numpy as np
import pandas as pd
from mhcflurry.predict_command import run as predict_run
from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor
from mhcflurry.torch_implementations import Class1AffinityPredictor as TorchPredictor

def compare_layer_outputs():
    # Test data
    alleles = ["HLA-A0201", "HLA-A0301"]
    peptides = ["SIINFEKL", "SIINFEKD", "SIINFEKQ"]

    # Load both predictors
    tf_predictor = Class1AffinityPredictor.load()
    torch_predictor = TorchPredictor.load()

    # Add hooks to capture intermediate outputs in PyTorch
    torch_activations = {}
    def get_activation(name):
        def hook(model, input, output):
            torch_activations[name] = output.detach().cpu().numpy()
        return hook

    # Register hooks for each layer
    for name, layer in torch_predictor.named_modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.BatchNorm1d)):
            layer.register_forward_hook(get_activation(name))

    # Get TensorFlow predictions and intermediate outputs
    import tensorflow as tf
    tf_model = tf_predictor.neural_networks[0]  # Get first model in ensemble
    tf_intermediate_model = tf.keras.Model(
        inputs=tf_model.network().inputs,
        outputs=[layer.output for layer in tf_model.network().layers]
    )

    # Prepare input data
    from mhcflurry.encodable_sequences import EncodableSequences
    peptides_obj = EncodableSequences.create(peptides)
    encoded_peptides = torch_predictor.peptides_to_network_input(peptides_obj)

    print("\nInput shape:", encoded_peptides.shape)
    print("Input first few values:", encoded_peptides[0, :10])

    # Get TensorFlow intermediate outputs
    tf_outputs = tf_intermediate_model.predict(encoded_peptides)
    
    # Get PyTorch intermediate outputs
    torch_predictor.eval()
    with torch.no_grad():
        encoded_tensor = torch.from_numpy(encoded_peptides).float().to(torch_predictor.device)
        torch_output = torch_predictor(encoded_tensor)

    # Compare outputs layer by layer
    print("\n=== Layer-by-layer comparison ===")
    
    # Compare input encoding
    print("\nInput encoding comparison:")
    print("TF input shape:", encoded_peptides.shape)
    print("Torch input shape:", encoded_tensor.shape)
    print("Input max diff:", np.abs(encoded_peptides - encoded_tensor.cpu().numpy()).max())

    # Compare each layer's output
    for i, (name, activation) in enumerate(torch_activations.items()):
        if i < len(tf_outputs):  # Make sure we have corresponding TF output
            tf_out = tf_outputs[i]
            torch_out = activation
            
            print(f"\nLayer {name}:")
            print("TF output shape:", tf_out.shape)
            print("Torch output shape:", torch_out.shape)
            print("Max difference:", np.abs(tf_out - torch_out).max())
            print("Mean difference:", np.abs(tf_out - torch_out).mean())
            print("TF first few values:", tf_out[0, :5])
            print("Torch first few values:", torch_out[0, :5])

    # Compare final predictions
    tf_preds = tf_predictor.predict(peptides, alleles=alleles)
    torch_preds = torch_predictor.predict(peptides, alleles=alleles)

    print("\n=== Final Predictions ===")
    print("TF predictions:", tf_preds)
    print("Torch predictions:", torch_preds)
    print("Max difference:", np.abs(tf_preds - torch_preds).max())

    # Print model architectures
    print("\n=== Model Architectures ===")
    print("TensorFlow model:")
    tf_model.network().summary()
    print("\nPyTorch model:")
    print(torch_predictor)

if __name__ == "__main__":
    compare_layer_outputs()
