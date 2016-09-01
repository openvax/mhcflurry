from numpy.testing import assert_equal

from mhcflurry.class1_allele_specific import Class1BindingPredictor


def test_all_combinations_of_hyperparameters():
    combinations_dict = dict(
        activation=["tanh", "sigmoid"],
        fraction_negative=[0, 0.2])
    results = (
        Class1BindingPredictor
        .hyperparameter_defaults
        .models_grid(**combinations_dict))
    assert_equal(len(results), 4)

if __name__ == "__main__":
    test_all_combinations_of_hyperparameters()
