"""
Antigen processing neural network implementation
"""

from __future__ import print_function

import time
import collections
import numpy

from .hyperparameters import HyperparameterDefaults
from .class1_neural_network import DEFAULT_PREDICT_BATCH_SIZE
from .flanking_encoding import FlankingEncoding


class Class1ProcessingNeuralNetwork(object):
    """
    A neural network for antigen processing prediction
    """
    network_hyperparameter_defaults = HyperparameterDefaults(
        amino_acid_encoding="BLOSUM62",
        peptide_max_length=15,
        n_flank_length=10,
        c_flank_length=10,
        flanking_averages=False,
        convolutional_filters=16,
        convolutional_kernel_size=8,
        convolutional_activation="tanh",
        convolutional_kernel_l1_l2=[0.0001, 0.0001],
        dropout_rate=0.5,
        post_convolutional_dense_layer_sizes=[],
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
        self._network = None
        self.network_json = None
        self.network_weights = None
        self.fit_info = []

    @property
    def sequence_lengths(self):
        """
        Supported maximum sequence lengths

        Returns
        -------
        dict of string -> int

        Keys are "peptide", "n_flank", "c_flank". Values give the maximum
        supported sequence length.
        """
        return {
            "peptide": self.hyperparameters['peptide_max_length'],
            "n_flank": self.hyperparameters['n_flank_length'],
            "c_flank": self.hyperparameters['c_flank_length'],
        }

    def network(self):
        """
        Return the keras model associated with this network.
        """
        if self._network is None and self.network_json is not None:
            import keras.models
            self._network = keras.models.model_from_json(self.network_json)
            if self.network_weights is not None:
                self._network.set_weights(self.network_weights)
        return self._network

    def update_network_description(self):
        """
        Update self.network_json and self.network_weights properties based on
        this instances's neural network.
        """
        if self._network is not None:
            self.network_json = self._network.to_json()
            self.network_weights = self._network.get_weights()

    def fit(
            self,
            sequences,
            targets,
            sample_weights=None,
            shuffle_permutation=None,
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0):
        """
        Fit the neural network.

        Parameters
        ----------
        sequences : FlankingEncoding
            Peptides and upstream/downstream flanking sequences
        targets : list of float
            1 indicates hit, 0 indicates decoy
        sample_weights : list of float
            If not specified all samples have equal weight.
        shuffle_permutation : list of int
            Permutation (integer list) of same length as peptides and affinities
            If None, then a random permutation will be generated.
        verbose : int
            Keras verbosity level
        progress_callback : function
            No-argument function to call after each epoch.
        progress_preamble : string
            Optional string of information to include in each progress update
        progress_print_interval : float
            How often (in seconds) to print progress update. Set to None to
            disable.
        """
        x_dict = self.network_input(sequences)

        # Shuffle
        if shuffle_permutation is None:
            shuffle_permutation = numpy.random.permutation(len(targets))
        targets = targets[shuffle_permutation]
        assert numpy.isnan(targets).sum() == 0, targets
        if sample_weights is not None:
            sample_weights = numpy.array(sample_weights)[shuffle_permutation]
        for key in list(x_dict):
            x_dict[key] = x_dict[key][shuffle_permutation]

        fit_info = collections.defaultdict(list)

        if self._network is None:
            self._network = self.make_network(
                **self.network_hyperparameter_defaults.subselect(
                    self.hyperparameters))
            if verbose > -1:
                self._network.summary()

        self.network().compile(
            loss="binary_crossentropy",
            optimizer=self.hyperparameters['optimizer'])

        last_progress_print = None
        min_val_loss_iteration = None
        min_val_loss = None
        start = time.time()
        for i in range(self.hyperparameters['max_epochs']):
            epoch_start = time.time()
            fit_history = self.network().fit(
                x_dict,
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
        fit_info["num_points"] = len(sequences.dataframe)
        self.fit_info.append(dict(fit_info))

        if verbose > -1:
            print(
                "Output weights",
                *numpy.array(
                    self.network().get_layer(
                        "output").get_weights()).flatten())

    def predict(
            self,
            peptides,
            n_flanks=None,
            c_flanks=None,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):
        """
        Predict antigen processing.

        Parameters
        ----------
        peptides : list of string
            Peptide sequences
        n_flanks : list of string
            Upstream sequence before each peptide
        c_flanks : list of string
            Downstream sequence after each peptide
        batch_size : int
            Prediction keras batch size.

        Returns
        -------
        numpy.array

        Processing scores. Range is 0-1, higher indicates more favorable
        processing.
        """
        if n_flanks is None:
            n_flanks = [""] * len(peptides)
        if c_flanks is None:
            c_flanks = [""] * len(peptides)

        sequences = FlankingEncoding(
            peptides=peptides, n_flanks=n_flanks, c_flanks=c_flanks)
        return self.predict_encoded(sequences=sequences, batch_size=batch_size)

    def predict_encoded(
            self,
            sequences,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):
        """
        Predict antigen processing.

        Parameters
        ----------
        sequences : FlankingEncoding
            Peptides and flanking sequences
        batch_size : int
            Prediction keras batch size.

        Returns
        -------
        numpy.array
        """
        x_dict = self.network_input(sequences)
        raw_predictions = self.network().predict(
            x_dict, batch_size=batch_size)
        predictions = numpy.squeeze(raw_predictions).astype("float64")
        return predictions

    def network_input(self, sequences):
        """
        Encode peptides to the fixed-length encoding expected by the neural
        network (which depends on the architecture).

        Parameters
        ----------
        sequences : FlankingEncoding
            Peptides and flanking sequences

        Returns
        -------
        numpy.array
        """
        encoded = sequences.vector_encode(
            self.hyperparameters['amino_acid_encoding'],
            self.hyperparameters['peptide_max_length'],
            n_flank_length=self.hyperparameters['n_flank_length'],
            c_flank_length=self.hyperparameters['c_flank_length'],
            allow_unsupported_amino_acids=True)

        result = {
            "sequence": encoded.array,
            "peptide_length": encoded.peptide_lengths,
        }
        return result

    def make_network(
            self,
            amino_acid_encoding,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            flanking_averages,
            convolutional_filters,
            convolutional_kernel_size,
            convolutional_activation,
            convolutional_kernel_l1_l2,
            dropout_rate,
            post_convolutional_dense_layer_sizes):
        """
        Helper function to make a keras network given hyperparameters.
        """

        # We import keras here to avoid tensorflow debug output, etc. unless we
        # are actually about to use Keras.

        from keras.layers import Input
        import keras.initializers
        from keras.layers.core import Dense, Flatten, Dropout
        from keras.layers.merge import Concatenate

        model_inputs = {}

        empty_x_dict = self.network_input(FlankingEncoding([], [], []))
        sequence_dims = empty_x_dict['sequence'].shape[1:]

        numpy.testing.assert_equal(
            sequence_dims[0],
            peptide_max_length + n_flank_length + c_flank_length)

        model_inputs['sequence'] = Input(
            shape=sequence_dims,
            dtype='float32',
            name='sequence')
        model_inputs['peptide_length'] = Input(
            shape=(1,),
            dtype='int32',
            name='peptide_length')

        current_layer = model_inputs['sequence']
        current_layer = keras.layers.Conv1D(
            filters=convolutional_filters,
            kernel_size=convolutional_kernel_size,
            kernel_regularizer=keras.regularizers.l1_l2(
                *convolutional_kernel_l1_l2),
            padding="same",
            activation=convolutional_activation,
            name="conv1")(current_layer)
        if dropout_rate > 0:
            current_layer = keras.layers.Dropout(
                name="conv1_dropout",
                rate=dropout_rate,
                noise_shape=(
                    None, 1, int(current_layer.get_shape()[-1])))(
                current_layer)

        convolutional_result = current_layer

        outputs_for_final_dense = []

        for flank in ["n_flank", "c_flank"]:
            current_layer = convolutional_result
            for (i, size) in enumerate(
                    list(post_convolutional_dense_layer_sizes) + [1]):
                current_layer = keras.layers.Conv1D(
                    name="%s_post_%d" % (flank, i),
                    filters=size,
                    kernel_size=1,
                    kernel_regularizer=keras.regularizers.l1_l2(
                        *convolutional_kernel_l1_l2),
                    activation=(
                        "tanh" if size == 1 else convolutional_activation
                    ))(current_layer)
            single_output_result = current_layer

            dense_flank = None
            if flank == "n_flank":
                def cleavage_extractor(x):
                    return x[:, n_flank_length]

                single_output_at_cleavage_position = keras.layers.Lambda(
                    cleavage_extractor, name="%s_cleaved" % flank)(
                    single_output_result)

                def max_pool_over_peptide_extractor(lst):
                    import tensorflow as tf
                    (x, peptide_length) = lst

                    # We generate a per-sample mask that is 1 for all peptide
                    # positions except the first position, and 0 for all other
                    # positions (i.e. n flank, c flank, and the first peptide
                    # position).
                    starts = n_flank_length + 1
                    limits = n_flank_length + peptide_length
                    row = tf.expand_dims(tf.range(0, x.shape[1]), axis=0)
                    mask = tf.logical_and(
                        tf.greater_equal(row, starts),
                        tf.less(row, limits))

                    # We are assuming that x >= -1. The final activation in the
                    # previous layer should be a function that satisfies this
                    # (e.g. sigmoid, tanh, relu).
                    max_value = tf.reduce_max(
                        (x + 1) * tf.expand_dims(
                            tf.cast(mask, tf.float32), axis=-1),
                        axis=1) - 1

                    # We flip the sign so that initializing the final dense
                    # layer weights to 1s is reasonable.
                    return -1 * max_value

                max_over_peptide = keras.layers.Lambda(
                    max_pool_over_peptide_extractor,
                    name="%s_internal_cleaved" % flank)([
                        single_output_result,
                        model_inputs['peptide_length']
                    ])

                def flanking_extractor(lst):
                    import tensorflow as tf
                    (x, peptide_length) = lst

                    # mask is 1 for n_flank positions and 0 elsewhere.
                    starts = 0
                    limits = n_flank_length
                    row = tf.expand_dims(tf.range(0, x.shape[1]), axis=0)
                    mask = tf.logical_and(
                        tf.greater_equal(row, starts),
                        tf.less(row, limits))

                    # We are assuming that x >= -1. The final activation in the
                    # previous layer should be a function that satisfies this
                    # (e.g. sigmoid, tanh, relu).
                    average_value = tf.reduce_mean(
                        (x + 1) * tf.expand_dims(
                            tf.cast(mask, tf.float32), axis=-1),
                        axis=1) - 1
                    return average_value

                if flanking_averages and n_flank_length > 0:
                    # Also include average pooled of flanking sequences
                    pooled_flank = keras.layers.Lambda(
                        flanking_extractor, name="%s_extracted" % flank)([
                            convolutional_result,
                            model_inputs['peptide_length']
                    ])
                    dense_flank = Dense(
                        1, activation="tanh", name="%s_avg_dense" % flank)(
                        pooled_flank)
            else:
                assert flank == "c_flank"

                def cleavage_extractor(lst):
                    import tensorflow as tf
                    (x, peptide_length) = lst
                    indexer = peptide_length + n_flank_length - 1
                    result = tf.squeeze(
                        tf.gather(x, indexer, batch_dims=1, axis=1),
                        -1)
                    return result

                single_output_at_cleavage_position = keras.layers.Lambda(
                    cleavage_extractor, name="%s_cleaved" % flank)([
                        single_output_result,
                        model_inputs['peptide_length']
                    ])

                def max_pool_over_peptide_extractor(lst):
                    import tensorflow as tf
                    (x, peptide_length) = lst

                    # We generate a per-sample mask that is 1 for all peptide
                    # positions except the last position, and 0 for all other
                    # positions (i.e. n flank, c flank, and the last peptide
                    # position).
                    starts = n_flank_length
                    limits = n_flank_length + peptide_length - 1
                    row = tf.expand_dims(tf.range(0, x.shape[1]), axis=0)
                    mask = tf.logical_and(
                        tf.greater_equal(row, starts),
                        tf.less(row, limits))

                    # We are assuming that x >= -1. The final activation in the
                    # previous layer should be a function that satisfies this
                    # (e.g. sigmoid, tanh, relu).
                    max_value = tf.reduce_max(
                        (x + 1) * tf.expand_dims(
                            tf.cast(mask, tf.float32), axis=-1),
                        axis=1) - 1

                    # We flip the sign so that initializing the final dense
                    # layer weights to 1s is reasonable.
                    return -1 * max_value

                max_over_peptide = keras.layers.Lambda(
                    max_pool_over_peptide_extractor,
                    name="%s_internal_cleaved" % flank)([
                        single_output_result,
                        model_inputs['peptide_length']
                    ])

                def flanking_extractor(lst):
                    import tensorflow as tf
                    (x, peptide_length) = lst

                    # mask is 1 for c_flank positions and 0 elsewhere.
                    starts = n_flank_length + peptide_length
                    limits = n_flank_length + peptide_length + c_flank_length
                    row = tf.expand_dims(tf.range(0, x.shape[1]), axis=0)
                    mask = tf.logical_and(
                        tf.greater_equal(row, starts),
                        tf.less(row, limits))

                    # We are assuming that x >= -1. The final activation in the
                    # previous layer should be a function that satisfies this
                    # (e.g. sigmoid, tanh, relu).
                    average_value = tf.reduce_mean(
                        (x + 1) * tf.expand_dims(
                            tf.cast(mask, tf.float32), axis=-1),
                        axis=1) - 1
                    return average_value

                if flanking_averages and c_flank_length > 0:
                    # Also include average pooled of flanking sequences
                    pooled_flank = keras.layers.Lambda(
                        flanking_extractor, name="%s_extracted" % flank)([
                            convolutional_result,
                            model_inputs['peptide_length']
                    ])
                    dense_flank = Dense(
                        1, activation="tanh", name="%s_avg_dense" % flank)(
                        pooled_flank)

            outputs_for_final_dense.append(single_output_at_cleavage_position)
            outputs_for_final_dense.append(max_over_peptide)
            if dense_flank is not None:
                outputs_for_final_dense.append(dense_flank)

        if len(outputs_for_final_dense) == 1:
            (current_layer,) = outputs_for_final_dense
        else:
            current_layer = Concatenate(name="final")(outputs_for_final_dense)
        output = Dense(
            1,
            activation="sigmoid",
            name="output",
            kernel_initializer=keras.initializers.Ones(),
        )(current_layer)
        model = keras.models.Model(
            inputs=[model_inputs[name] for name in sorted(model_inputs)],
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
        self.update_network_description()
        result = dict(self.__dict__)
        result['_network'] = None
        return result

    def __setstate__(self, state):
        """
        Deserialize. For pickle support.
        """
        self.__dict__.update(state)

    def get_weights(self):
        """
        Get the network weights

        Returns
        -------
        list of numpy.array giving weights for each layer or None if there is no
        network
        """
        self.update_network_description()
        return self.network_weights

    def get_config(self):
        """
        serialize to a dict all attributes except model weights

        Returns
        -------
        dict
        """
        self.update_network_description()
        result = dict(self.__dict__)
        del result['_network']
        result['network_weights'] = None
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

        Returns
        -------
        Class1ProcessingNeuralNetwork
        """
        config = dict(config)
        instance = cls(**config.pop('hyperparameters'))
        instance.__dict__.update(config)
        instance.network_weights = weights
        assert instance._network is None
        return instance
