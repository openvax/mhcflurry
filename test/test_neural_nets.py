from mhcflurry.feedforward import (
    make_embedding_network,
    make_hotshot_network,
)
import numpy as np

def test_make_embedding_network():
    nn = make_embedding_network(
        peptide_length=3,
        layer_sizes=[3],
        activation="tanh",
        embedding_input_dim=3,
        embedding_output_dim=20,
        learning_rate=0.05)

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

def test_make_hotshot_network():
    nn = make_hotshot_network(
        peptide_length=3,
        activation="relu",
        layer_sizes=[4],
        n_amino_acids=2,
        learning_rate=0.05)
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
    test_make_hotshot_network()
    test_make_embedding_network()
