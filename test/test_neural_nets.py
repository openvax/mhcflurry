from mhcflurry.feedforward import (
    make_embedding_network,
    make_hotshot_network,
)
from keras.optimizers import RMSprop
from keras.objectives import mse
import numpy as np
from nose.tools import eq_

def test_make_embedding_network_properties():
    layer_sizes = [3, 4]
    nn = make_embedding_network(
        peptide_length=3,
        n_amino_acids=3,
        layer_sizes=layer_sizes,
        loss=mse,
        optimizer=RMSprop(lr=0.7, rho=0.9, epsilon=1e-6),
        batch_normalization=False,)
    eq_(nn.layers[0].input_dim, 3)
    eq_(nn.loss, mse)
    assert np.allclose(nn.optimizer.lr.eval(), 0.7)
    print(nn.layers)
    # embedding + flatten + (dense->activation) * hidden layers and last layer
    eq_(len(nn.layers), 2 + 2 * (1 + len(layer_sizes)))

def test_make_hotshot_network_properties():
    layer_sizes = [3, 4]
    nn = make_hotshot_network(
        peptide_length=3,
        n_amino_acids=2,
        activation="relu",
        init="lecun_uniform",
        loss=mse,
        layer_sizes=layer_sizes,
        batch_normalization=False,
        optimizer=RMSprop(lr=0.7, rho=0.9, epsilon=1e-6))
    eq_(nn.layers[0].input_dim, 6)
    eq_(nn.loss, mse)
    assert np.allclose(nn.optimizer.lr.eval(), 0.7)
    print(nn.layers)
    # since the hotshot network doesn't have an embedding layer + flatten
    # we expect two fewer total layers than the embedding network.
    eq_(len(nn.layers), 2 * (1 + len(layer_sizes)))

def test_make_embedding_network_small_dataset():
    nn = make_embedding_network(
        peptide_length=3,
        n_amino_acids=3,
        layer_sizes=[3],
        activation="tanh",
        init="lecun_uniform",
        loss="mse",
        embedding_output_dim=20,
        optimizer=RMSprop(lr=0.05, rho=0.9, epsilon=1e-6))
    X_negative = np.array([
        [0] * 3,
        [1] * 3,
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
    ])
    X_positive = np.array([
        [0, 2, 0],
        [1, 2, 0],
        [1, 2, 1],
        [0, 2, 1],
        [2, 2, 0],
        [2, 2, 1],
        [2, 2, 2],
    ])
    X_index = np.vstack([X_negative, X_positive])
    Y = np.array([0.0] * len(X_negative) + [1.0] * len(X_positive))
    nn.fit(X_index, Y, nb_epoch=20)
    Y_pred = nn.predict(X_index)
    print(Y)
    print(Y_pred)
    for (Y_i, Y_pred_i) in zip(Y, Y_pred):
        assert abs(Y_i - Y_pred_i) <= 0.25, (Y_i, Y_pred_i)

def test_make_hotshot_network_small_dataset():
    nn = make_hotshot_network(
        peptide_length=3,
        n_amino_acids=2,
        activation="relu",
        init="lecun_uniform",
        loss="mse",
        layer_sizes=[4],
        optimizer=RMSprop(lr=0.05, rho=0.9, epsilon=1e-6))
    X_binary = np.array([
        [True, False, True, False, True, False],
        [True, False, True, False, False, True],
        [True, False, False, True, True, False],
        [True, False, False, True, False, True],
        [False, True, True, False, True, False],
        [False, True, True, False, False, True],
    ], dtype=bool)
    Y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    nn.fit(X_binary, Y, nb_epoch=20)
    Y_pred = nn.predict(X_binary)
    print(Y)
    print(Y_pred)
    for (Y_i, Y_pred_i) in zip(Y, Y_pred):
        if Y_i:
            assert Y_pred_i >= 0.6, "Expected higher value than %f" % Y_pred_i
        else:
            assert Y_pred_i <= 0.4, "Expected lower value than %f" % Y_pred_i

if __name__ == "__main__":
    test_make_embedding_network_properties()
    test_make_hotshot_network_properties()
    test_make_embedding_network_small_dataset()
    test_make_hotshot_network_small_dataset()
