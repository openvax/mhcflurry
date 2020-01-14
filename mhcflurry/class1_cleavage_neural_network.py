"""
Idea:

Fully convolutional network with softmax output. Let it take a 35mer:
- N flank [10 aa]
- peptide [7-15 aa]
- C flank [10 aa]

Train on monoallelic mass spec. Match positive examples (hits) to negatives
from the same sample by finding unobserved peptides with as close as possible
a match for predicted binding affinity.

In final layer, take max cleavage score over peptide and the individual score
for the position right before the peptide terminus. Compute the ratio of these.
Or actually reverse of that. Hits get label 1, decoys get 0.

For a hit with sequence

NNNNNNNNNNPPPPPPPPPCCCCCCCCCC

penalize on:

[----------1000000000---------]

For a decoy with same sequence, penalize it on:

[----------0-----------------]

Train separate models for N- and C-terminal cleavage.

Issue:
- it'll learn mass spec biases in the peptide

Possible fix:
- Also provide the amino acid counts of the peptide as auxiliary inputs. After
training, set the cysteine value to 0.

Architecture:
architecture (for N terminal: for C terminal reverse the sequences):

input of length S=25 [flank + left-aligned peptide]
*** conv [vector of length S] ***
*** [more convs and local pools] ***
*** output [vector of length S] ***
*** extract: position 10 and max of peptide positions [2-vector]
*** concat:[position 10, max of peptide positions, number of Alananine, ..., number of Y in peptide]
*** single dense node, softmax activation [1-vector]

Train on monoallelic.

Decoys are length-matched to hits and sampled from the same transcripts, selecting
an unobeserved peptide with as close as possible the same predicted affinity.


    *** + repeat vector for each position


*** conv ***
*** conv ***
*** ... conv n ***
***                             repeat vector for each position
*** dense per-position
*** output [35-vector]
*** extract: position 10 and max of peptide positions [2-vector]
*** dense
*** output


IDEA 2:

- Two inputs: N-flank + peptide (left aligned), peptide (right alighted +  C-flank
- Bunch of convolutions
- Global max pooling
- Dense


"""

from __future__ import print_function

import time
import collections
import numpy

from .hyperparameters import HyperparameterDefaults
from .class1_neural_network import DEFAULT_PREDICT_BATCH_SIZE
from .encodable_sequences import EncodableSequences


class Class1CleavageNeuralNetwork(object):
    network_hyperparameter_defaults = HyperparameterDefaults(
        amino_acid_encoding="BLOSUM62",
        peptide_max_length=15,
        n_flank_length=10,
        c_flank_length=10,
        vector_encoding_name="BLOSUM62",
        flanking_averages=False,
        convolutional_filters=16,
        convolutional_kernel_size=8,
        convolutional_activation="relu",
        convoluational_kernel_l1_l2=(0.001, 0.001),
        dropout_rate=0.5,
    )
    """
    Hyperparameters (and their default values) that affect the neural network
    architecture.
    """

    fit_hyperparameter_defaults = HyperparameterDefaults(
        max_epochs=500,
        validation_split=0.1,
        early_stopping=True,
        minibatch_size=256,
    )
    """
    Hyperparameters for neural network training.
    """

    early_stopping_hyperparameter_defaults = HyperparameterDefaults(
        patience=30,
        min_delta=0.0,
    )
    """
    Hyperparameters for early stopping.
    """

    compile_hyperparameter_defaults = HyperparameterDefaults(
        optimizer="adam",
        learning_rate=None,
    )
    """
    Loss and optimizer hyperparameters. Any values supported by keras may be
    used.
    """

    auxiliary_input_hyperparameter_defaults = HyperparameterDefaults(
    )
    """
    Allele feature hyperparameters.
    """

    hyperparameter_defaults = network_hyperparameter_defaults.extend(
        fit_hyperparameter_defaults).extend(
        early_stopping_hyperparameter_defaults).extend(
        compile_hyperparameter_defaults).extend(
        auxiliary_input_hyperparameter_defaults)

    def __init__(self, **hyperparameters):
        self.hyperparameters = self.hyperparameter_defaults.with_defaults(
            hyperparameters)
        self.network = None
        self.fit_info = []

    def fit(
            self,
            peptides,
            n_flanks,
            c_flanks,
            targets,
            sample_weights=None,
            shuffle_permutation=None,
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0):
        """


        Parameters
        ----------
        peptides
        n_flanks
        c_flanks
        targets : array of {0, 1} indicating hits (1) or decoys (0)

        Returns
        -------

        """
        import keras.backend as K

        peptides = EncodableSequences.create(peptides)
        n_flanks = EncodableSequences.create(n_flanks)
        c_flanks = EncodableSequences.create(c_flanks)

        x_list = self.peptides_and_flanking_to_network_input(
            peptides, n_flanks, c_flanks)

        # Shuffle
        if shuffle_permutation is None:
            shuffle_permutation = numpy.random.permutation(len(targets))
        targets = targets[shuffle_permutation]
        assert numpy.isnan(targets).sum() == 0, targets
        if sample_weights is not None:
            sample_weights = numpy.array(sample_weights)[shuffle_permutation]
        x_list = [x[shuffle_permutation] for x in x_list]

        fit_info = collections.defaultdict(list)

        if self.network is None:
            self.network = self.make_network(
                **self.network_hyperparameter_defaults.subselect(
                    self.hyperparameters))
            if verbose > 0:
                self.network.summary()

        self.network.compile(
            loss="binary_crossentropy",
            optimizer=self.hyperparameters['optimizer'])

        last_progress_print = None
        min_val_loss_iteration = None
        min_val_loss = None
        start = time.time()
        for i in range(self.hyperparameters['max_epochs']):
            epoch_start = time.time()
            fit_history = self.network.fit(
                x_list,
                targets,
                validation_split=self.hyperparameters['validation_split'],
                batch_size=self.hyperparameters['minibatch_size'],
                epochs=i + 1,
                sample_weight=sample_weights,
                initial_epoch=i,
                verbose=verbose)
            epoch_time = time.time() - epoch_start

            for (key, value) in fit_history.history.items():
                fit_info[key].extend(value)

            # Print progress no more often than once every few seconds.
            if progress_print_interval is not None and (
                    not last_progress_print or (
                        time.time() - last_progress_print
                        > progress_print_interval)):
                print((progress_preamble + " " +
                       "Epoch %3d / %3d [%0.2f sec]: loss=%g. "
                       "Min val loss (%s) at epoch %s" % (
                           i,
                           self.hyperparameters['max_epochs'],
                           epoch_time,
                           fit_info['loss'][-1],
                           str(min_val_loss),
                           min_val_loss_iteration)).strip())
                last_progress_print = time.time()

            if self.hyperparameters['validation_split']:
                val_loss = fit_info['val_loss'][-1]

                if min_val_loss is None or (
                        val_loss < min_val_loss - self.hyperparameters['min_delta']):
                    min_val_loss = val_loss
                    min_val_loss_iteration = i

                if self.hyperparameters['early_stopping']:
                    threshold = (
                        min_val_loss_iteration +
                        self.hyperparameters['patience'])
                    if i > threshold:
                        if progress_print_interval is not None:
                            print((progress_preamble + " " +
                                "Stopping at epoch %3d / %3d: loss=%g. "
                                "Min val loss (%g) at epoch %s" % (
                                    i,
                                    self.hyperparameters['max_epochs'],
                                    fit_info['loss'][-1],
                                    (
                                        min_val_loss if min_val_loss is not None
                                        else numpy.nan),
                                    min_val_loss_iteration)).strip())
                        break

            if progress_callback:
                progress_callback()

        fit_info["time"] = time.time() - start
        fit_info["num_points"] = len(peptides)
        self.fit_info.append(dict(fit_info))

        if verbose:
            print(
                "Output weights",
                *numpy.array(
                    self.network.get_layer("output").get_weights()).flatten())

    def predict(
            self,
            peptides,
            n_flanks,
            c_flanks,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):
        """
        """
        x_list = self.peptides_and_flanking_to_network_input(
            peptides, n_flanks, c_flanks)
        raw_predictions = self.network.predict(
            x_list, batch_size=batch_size)
        predictions = numpy.array(raw_predictions, dtype="float64")[:,0]
        return predictions

    def peptides_and_flanking_to_network_input(self, peptides, n_flanks, c_flanks):
        """
        Encode peptides to the fixed-length encoding expected by the neural
        network (which depends on the architecture).

        Parameters
        ----------
        peptides : EncodableSequences or list of string

        Returns
        -------
        numpy.array
        """
        peptides = EncodableSequences.create(peptides)
        n_flanks = EncodableSequences.create(n_flanks)
        c_flanks = EncodableSequences.create(c_flanks)

        peptide_encoded1 = peptides.variable_length_to_fixed_length_vector_encoding(
            vector_encoding_name=self.hyperparameters['vector_encoding_name'],
            max_length=self.hyperparameters['peptide_max_length'],
            alignment_method='right_pad')
        peptide_encoded2 = peptides.variable_length_to_fixed_length_vector_encoding(
            vector_encoding_name=self.hyperparameters['vector_encoding_name'],
            max_length=self.hyperparameters['peptide_max_length'],
            alignment_method='left_pad')
        n_flanks_encoded = n_flanks.variable_length_to_fixed_length_vector_encoding(
            vector_encoding_name=self.hyperparameters['vector_encoding_name'],
            max_length=self.hyperparameters['n_flank_length'],
            alignment_method='right_pad')
        c_flanks_encoded = c_flanks.variable_length_to_fixed_length_vector_encoding(
            vector_encoding_name=self.hyperparameters['vector_encoding_name'],
            max_length=self.hyperparameters['c_flank_length'],
            alignment_method='left_pad')

        return [
            peptide_encoded1,
            peptide_encoded2,
            n_flanks_encoded,
            c_flanks_encoded
        ]


    def make_network(
            self,
            amino_acid_encoding,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            vector_encoding_name,
            flanking_averages,
            convolutional_filters,
            convolutional_kernel_size,
            convolutional_activation,
            convoluational_kernel_l1_l2,
            dropout_rate):
        """
        Helper function to make a keras network
        """

        # We import keras here to avoid tensorflow debug output, etc. unless we
        # are actually about to use Keras.

        from keras.layers import Input
        import keras.layers
        import keras.layers.pooling
        from keras.layers.core import Dense, Flatten, Dropout
        from keras.layers.embeddings import Embedding
        from keras.layers.normalization import BatchNormalization
        from keras.layers.merge import Concatenate
        import keras.backend as K

        (peptides_empty, _, n_flanks_empty, c_flanks_empty) = (
            self.peptides_and_flanking_to_network_input(
                peptides=[],
                n_flanks=[],
                c_flanks=[]))

        print((peptides_empty, _, n_flanks_empty, c_flanks_empty))

        peptide_input1 = Input(
            shape=peptides_empty.shape[1:],
            dtype='float32',
            name='peptide1')
        peptide_input2 = Input(
            shape=peptides_empty.shape[1:],
            dtype='float32',
            name='peptide2')
        n_flank_input = Input(
            shape=n_flanks_empty.shape[1:],
            dtype='float32',
            name='n_flank')
        c_flank_input = Input(
            shape=c_flanks_empty.shape[1:],
            dtype='float32',
            name='c_flank')

        inputs = [peptide_input1, peptide_input2, n_flank_input, c_flank_input]

        conv_outputs = []
        single_outputs = []
        for input_pair in [(n_flank_input, peptide_input1), (peptide_input2, c_flank_input)]:
            # need to stack them together
            current_layer = Concatenate(axis=1)(list(input_pair))
            current_layer = keras.layers.Conv1D(
                filters=convolutional_filters,
                kernel_size=convolutional_kernel_size,
                kernel_regularizer=keras.regularizers.l1_l2(
                    *convoluational_kernel_l1_l2),
                padding="same",
                activation=convolutional_activation)(current_layer)
            if dropout_rate > 0:
                current_layer = keras.layers.Dropout(
                    rate=dropout_rate,
                    noise_shape=(
                        None, 1, int(current_layer.get_shape()[-1])))(
                    current_layer)
            conv_outputs.append(current_layer)
            current_layer = keras.layers.Conv1D(
                filters=1,
                kernel_size=1,
                kernel_regularizer=keras.regularizers.l1_l2(
                    *convoluational_kernel_l1_l2),
                activation=convolutional_activation)(current_layer)
            single_outputs.append(current_layer)

        extracted_layers = []
        extracted_layers.append(
                keras.layers.Lambda(
                    lambda x: x[:, n_flank_length])(single_outputs[0]))
        if flanking_averages:
            n_flank = keras.layers.Lambda(
                lambda x: x[
                    :, : n_flank_length
                ])(conv_outputs[0])
            extracted_layers.append(
                keras.layers.pooling.GlobalAveragePooling1D()(n_flank))
        peptide_n_cleavage = keras.layers.Lambda(
            lambda x: x[
                :, (n_flank_length + 1) :
            ])(single_outputs[0])
        extracted_layers.append(
            keras.layers.Lambda(lambda x: -x)(
                keras.layers.pooling.GlobalMaxPooling1D()(
                    peptide_n_cleavage)))

        extracted_layers.append(
            keras.layers.Lambda(
                lambda x: x[:, peptide_max_length])(single_outputs[1]))
        if flanking_averages:
            c_flank = keras.layers.Lambda(
                lambda x: x[
                    :, peptide_max_length :
                ])(conv_outputs[1])
            extracted_layers.append(
                keras.layers.pooling.GlobalAveragePooling1D()(c_flank))
        peptide_c_cleavage = keras.layers.Lambda(
            lambda x: x[
                :, 0 : peptide_max_length
            ])(single_outputs[1])
        extracted_layers.append(
            keras.layers.Lambda(lambda x: -x)(
                keras.layers.pooling.GlobalMaxPooling1D()(peptide_c_cleavage)))

        current_layer = Concatenate()(extracted_layers)
        output = Dense(
            1,
            activation="sigmoid",
            name="output",
            kernel_initializer=keras.initializers.Ones(),
        )(current_layer)
        model = keras.models.Model(
            inputs=inputs,
            outputs=[output],
            name="predictor")

        return model

    def __getstate__(self):
        """
        serialize to a dict. Model weights are included. For pickle support.

        Returns
        -------
        dict

        """
        result = self.get_config()
        result['network_weights'] = self.get_weights()
        return result

    def __setstate__(self, state):
        """
        Deserialize. For pickle support.
        """
        network_json = state.pop("network_json")
        network_weights = state.pop("network_weights")
        self.__dict__.update(state)
        if network_json is not None:
            import keras.models
            self.network = keras.models.model_from_json(network_json)
            if network_weights is not None:
                self.network.set_weights(network_weights)

    def get_weights(self):
        """
        Get the network weights

        Returns
        -------
        list of numpy.array giving weights for each layer or None if there is no
        network
        """
        if self.network is None:
            return None
        return self.network.get_weights()

    def get_config(self):
        """
        serialize to a dict all attributes except model weights

        Returns
        -------
        dict
        """
        result = dict(self.__dict__)
        del result['network']
        result['network_json'] = None
        if self.network:
            result['network_json'] = self.network.to_json()
        return result

    @classmethod
    def from_config(cls, config, weights=None):
        """
        deserialize from a dict returned by get_config().

        Parameters
        ----------
        config : dict
        weights : list of array, optional
            Network weights to restore
        weights_loader : callable, optional
            Function to call (no arguments) to load weights when needed

        Returns
        -------
        Class1CleavageNeuralNetwork
        """
        config = dict(config)
        instance = cls(**config.pop('hyperparameters'))
        network_json = config.pop('network_json')
        instance.__dict__.update(config)
        assert instance.network is None
        if network_json is not None:
            import keras.models
            instance.network = keras.models.model_from_json(network_json)
            if weights is not None:
                instance.network.set_weights(weights)
        return instance