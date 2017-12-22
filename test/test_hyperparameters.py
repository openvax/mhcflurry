from numpy.testing import assert_equal

from mhcflurry.class1_neural_network import Class1NeuralNetwork


def test_all_combinations_of_hyperparameters():
    combinations_dict = dict(
        activation=["tanh", "sigmoid"],
        random_negative_constant=[0, 20])
    results = (
        Class1NeuralNetwork
        .hyperparameter_defaults
        .models_grid(**combinations_dict))
    assert_equal(len(results), 4)

if __name__ == "__main__":
    test_all_combinations_of_hyperparameters()
