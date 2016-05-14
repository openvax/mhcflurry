from mhcflurry.feedforward_hyperparameters import (
    all_combinations_of_hyperparameters,
    default_hyperparameters,
)
from nose.tools import eq_

def test_all_combinations_of_hyperparameters():
    combinations_dict = dict(
        learning_rate=[0.1, 0.5],
        hidden_layer_size=[10, 20])

    results = list(all_combinations_of_hyperparameters(**combinations_dict))
    eq_(len(results), 4)
    for params in results:
        for name, default_value in default_hyperparameters.__dict__.items():
            curr_value = params.__dict__[name]
            if name not in combinations_dict:
                eq_(curr_value, default_value)
            else:
                assert curr_value in combinations_dict[name], curr_value

if __name__ == "__main__":
    test_all_combinations_of_hyperparameters()
