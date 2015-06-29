import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.embeddings import Embedding

import theano
theano.config.exception_verbosity = 'high'

def make_network(
        input_size,
        embedding_input_dim=None,
        embedding_output_dim=None,
        layer_sizes=[100],
        activation="relu",
        init="lecun_uniform",
        loss="mse",
        output_activation="sigmoid",
        dropout_probability=0.0,
        model=None,
        optimizer=None):

    if model is None:
        model = Sequential()

    if optimizer is None:
        optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)

    if embedding_input_dim:
        if not embedding_output_dim:
            raise ValueError(
                "Both embedding_input_dim and embedding_output_dim must be set")

        model.add(Embedding(
            input_dim=embedding_input_dim,
            output_dim=embedding_output_dim,
            init=init))
        model.add(Flatten())
        input_size = input_size * embedding_output_dim

    layer_sizes = (input_size,) + tuple(layer_sizes)
    for i, dim in enumerate(layer_sizes):
        if i == 0:
            # input is only conceptually a layer of the network,
            # don't need to actually do anything
            continue

        previous_dim = layer_sizes[i - 1]

        # hidden layer fully connected layer
        model.add(
            Dense(
                input_dim=previous_dim,
                output_dim=dim,
                init=init))
        model.add(Activation(activation))
        if dropout_probability > 0:
            model.add(Dropout(dropout_probability))

    # output
    model.add(Dense(
        input_dim=layer_sizes[-1],
        output_dim=1,
        init=init))
    model.add(Activation(output_activation))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def make_hotshot_network(
        peptide_length=9,
        layer_sizes=[500],
        activation="relu",
        init="lecun_uniform",
        loss="mse",
        output_activation="sigmoid",
        dropout_probability=0.0,
        optimizer=None):
    return make_network(
        input_size=peptide_length * 20,
        layer_sizes=layer_sizes,
        activation=activation,
        init=init,
        loss=loss,
        output_activation=output_activation,
        dropout_probability=dropout_probability,
        optimizer=optimizer)

def make_embedding_network(
        peptide_length=9,
        embedding_input_dim=20,
        embedding_output_dim=20,
        layer_sizes=[500],
        activation="relu",
        init="lecun_uniform",
        loss="mse",
        output_activation="sigmoid",
        dropout_probability=0.0,
        optimizer=None):
    return make_network(
        input_size=peptide_length,
        embedding_input_dim=embedding_input_dim,
        embedding_output_dim=embedding_output_dim,
        layer_sizes=layer_sizes,
        activation=activation,
        init=init,
        loss=loss,
        output_activation=output_activation,
        dropout_probability=dropout_probability,
        optimizer=optimizer)