import pickle
import numpy as np

from mhcflurry.class1_affinity_prediction import Class1NeuralNetwork


def test_predict_after_saving_model_to_disk():
    # don't even bother fitting the model, just save its random weights
    # and check we get the same predictions back afterward
    model = Class1NeuralNetwork(name="rando")
    peptides = ["A" * 9, "C" * 9]
    original_predictions = model.predict(peptides)

    depickled_model = pickle.loads(pickle.dumps(model))
    assert depickled_model.model is not None

    depickled_predictions = depickled_model.predict(peptides)

    assert np.allclose(original_predictions, depickled_predictions), (
        peptides, original_predictions, depickled_predictions)
