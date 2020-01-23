from __future__ import print_function

import time
import collections
from six import string_types

import numpy
import pandas
import mhcnames
import hashlib
from copy import copy

from .hyperparameters import HyperparameterDefaults
from .class1_neural_network import Class1NeuralNetwork, DEFAULT_PREDICT_BATCH_SIZE
from .class1_cleavage_neural_network import Class1CleavageNeuralNetwork
from .encodable_sequences import EncodableSequences
from .allele_encoding import MultipleAlleleEncoding, AlleleEncoding
from .auxiliary_input import AuxiliaryInputEncoder
from .flanking_encoding import FlankingEncoding


class Class1PresentationNeuralNetwork(object):
    network_hyperparameter_defaults = HyperparameterDefaults(
        max_alleles=6,
    )
    """
    Hyperparameters (and their default values) that affect the neural network
    architecture.
    """

    fit_hyperparameter_defaults = HyperparameterDefaults(
        trainable_cleavage_predictor=False,
        trainable_affinity_predictor=False,
        max_epochs=500,
        validation_split=0.1,
        early_stopping=True,
        minibatch_size=256,
    )
    """
    Hyperparameters for neural network training.
    """

    early_stopping_hyperparameter_defaults = HyperparameterDefaults(
        patience=20,
        min_delta=0.0,
    )
    """
    Hyperparameters for early stopping.
    """

    compile_hyperparameter_defaults = HyperparameterDefaults(
        loss="binary_crossentropy",
        optimizer="rmsprop",
        learning_rate=None,
    )
    """
    Loss and optimizer hyperparameters. Any values supported by keras may be
    used.
    """

    auxiliary_input_hyperparameter_defaults = HyperparameterDefaults(
        auxiliary_input_features=["gene"],
        auxiliary_input_feature_parameters={},
        include_cleavage=True,
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
        self.allele_representation_hash = None
        self.affinity_model = None
        self.cleavage_model = None

    def build(self, affinity_model, cleavage_model=None):
        import keras.backend as K
        import keras.models

        assert isinstance(affinity_model, Class1NeuralNetwork), affinity_model
        affinity_model = copy(affinity_model)

        self.affinity_model = affinity_model
        affinity_network = affinity_model.network()

        model_inputs = {}

        peptide_shape = tuple(
            int(x) for x in K.int_shape(affinity_network.inputs[0])[1:])

        model_inputs['allele_set'] = keras.layers.Input(
            shape=(self.hyperparameters['max_alleles'],), name="allele_set")
        model_inputs['peptide'] = keras.layers.Input(
            shape=peptide_shape,
            dtype='float32',
            name='peptide')

        peptides_flattened = keras.layers.Flatten()(model_inputs['peptide'])
        peptides_repeated = keras.layers.RepeatVector(
            self.hyperparameters['max_alleles'])(
            peptides_flattened)

        allele_representation = keras.layers.Embedding(
            name="allele_representation",
            input_dim=64,  # arbitrary, how many alleles to have room for
            output_dim=affinity_network.get_layer(
                "allele_representation").output_shape[-1],
            input_length=self.hyperparameters['max_alleles'],
            trainable=False,
            mask_zero=False)(model_inputs['allele_set'])

        allele_flat = allele_representation

        allele_peptide_merged = keras.layers.concatenate(
            [peptides_repeated, allele_flat], name="allele_peptide_merged")

        layer_names = [
            layer.name for layer in affinity_network.layers
        ]

        pan_allele_layer_initial_names = [
            'allele', 'peptide',
            'allele_representation', 'flattened_0', 'allele_flat',
            'allele_peptide_merged', 'dense_0', 'dropout_0',
        ]

        def startswith(lst, prefix):
            return lst[:len(prefix)] == prefix

        assert startswith(
            layer_names, pan_allele_layer_initial_names), layer_names

        layers = affinity_network.layers[
            pan_allele_layer_initial_names.index(
                "allele_peptide_merged") + 1:
        ]
        node = allele_peptide_merged
        affinity_predictor_layer_name_to_new_node = {
            "allele_peptide_merged": allele_peptide_merged,
        }

        for layer in layers:
            assert layer.name not in affinity_predictor_layer_name_to_new_node
            input_layer_names = []
            for inbound_node in layer._inbound_nodes:
                for inbound_layer in inbound_node.inbound_layers:
                    input_layer_names.append(inbound_layer.name)
            input_nodes = [
                affinity_predictor_layer_name_to_new_node[name]
                for name in input_layer_names
            ]

            if len(input_nodes) == 1:
                lifted = keras.layers.TimeDistributed(layer, name=layer.name)
                node = lifted(input_nodes[0])
            else:
                node = layer(input_nodes)
            affinity_predictor_layer_name_to_new_node[layer.name] = node

        def logit(x):
            import tensorflow as tf
            return -tf.log(1. / x - 1.)

        #node = keras.layers.Lambda(logit, name="logit")(node)
        affinity_prediction_and_other_signals = [node]
        if self.hyperparameters['include_cleavage']:
            assert isinstance(cleavage_model, Class1CleavageNeuralNetwork)
            cleavage_model = copy(cleavage_model)
            self.cleavage_model = cleavage_model
            cleavage_network = cleavage_model.network()

            model_inputs['sequence'] = keras.layers.Input(
                shape=cleavage_network.get_layer("sequence").output_shape[1:],
                dtype='float32',
                name='sequence')
            model_inputs['peptide_length'] = keras.layers.Input(
                shape=(1,),
                dtype='int32',
                name='peptide_length')
            cleavage_network.name = "cleavage_predictor"
            cleavage_prediction = cleavage_network([
                model_inputs['peptide_length'],
                model_inputs['sequence'],
            ])
            cleavage_prediction.trainable = False
            cleavage_prediction_repeated = keras.layers.RepeatVector(
                self.hyperparameters['max_alleles'])(cleavage_prediction)
            affinity_prediction_and_other_signals.append(
                cleavage_prediction_repeated)

        if self.hyperparameters['auxiliary_input_features']:
            model_inputs['auxiliary'] = keras.layers.Input(
                shape=(
                    self.hyperparameters['max_alleles'],
                    len(
                        AuxiliaryInputEncoder.get_columns(
                            self.hyperparameters['auxiliary_input_features'],
                            feature_parameters=self.hyperparameters[
                                'auxiliary_input_feature_parameters']))),
                dtype="float32",
                name="auxiliary")
            affinity_prediction_and_other_signals.append(
                model_inputs['auxiliary'])

        if len(affinity_prediction_and_other_signals) > 1:
            node = keras.layers.concatenate(
                affinity_prediction_and_other_signals,
                name="affinity_prediction_and_other_signals")
            layer = keras.layers.Dense(
                1,
                activation="sigmoid",
                kernel_initializer=keras.initializers.Ones(),
                name="combine")
            lifted = keras.layers.TimeDistributed(layer, name="per_allele_output")
            node = lifted(node)
        else:
            (node,) = affinity_prediction_and_other_signals

        # Apply allele mask: zero out all outputs corresponding to alleles
        # with the special index 0.
        #def alleles_to_mask(x):
        #    import keras.backend as K
        #    result = K.expand_dims(
        #        K.cast(K.not_equal(x, 0), "float32"), axis=-1)
        #    return result

        #allele_mask = keras.layers.Lambda(
        #    alleles_to_mask, name="allele_mask")(model_inputs['allele_set'])

        #node = keras.layers.Multiply(
        #    name="masked_per_allele_outputs")(
        #    [allele_mask, node])

        presentation_output = keras.layers.Reshape(
            target_shape=(self.hyperparameters['max_alleles'],))(
            node)

        self.network = keras.models.Model(
            inputs=list(model_inputs.values()),
            outputs=presentation_output,
            name="presentation",
        )

        if not self.hyperparameters['trainable_cleavage_predictor']:
            if self.hyperparameters['include_cleavage']:
                self.network.get_layer("cleavage_predictor").trainable = False

        self.affinity_predictor_layer_names = list(
            affinity_predictor_layer_name_to_new_node)

        self.set_trainable(
            trainable_affinity_predictor=(
                self.hyperparameters['trainable_affinity_predictor']))

    def set_trainable(self, trainable_affinity_predictor=None):
        if trainable_affinity_predictor is not None:
            for name in self.affinity_predictor_layer_names:
                self.network.get_layer(name).trainable = trainable_affinity_predictor


    @staticmethod
    def loss(y_true, y_pred):
        # Binary cross entropy
        from keras import backend as K
        import tensorflow as tf

        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)

        #y_pred = tf.Print(y_pred, [y_pred], message="y_pred", summarize=50)
        #y_true = tf.Print(y_true, [y_true], message="y_true", summarize=50)

        #logit_y_pred = -tf.log(1. / y_pred - 1.)
        #logit_y_pred = tf.Print(logit_y_pred, [logit_y_pred], message="logit_y_pred", summarize=50)

        #softmax = K.softmax(5 * logit_y_pred, axis=-1)
        #softmax = tf.Print(softmax, [softmax], message="softmax", summarize=50)

        #product = softmax * y_pred
        #product = tf.Print(product, [product], message="product", summarize=50)

        #result = tf.reduce_sum(product, axis=-1)
        #result = tf.Print(result, [result], message="result", summarize=50)

        #result = tf.reduce_max(y_pred, axis=-1)
        result = tf.reduce_sum(y_pred, axis=-1)

        return K.mean(
            K.binary_crossentropy(y_true, result),
            axis=-1)

    def network_input(
            self, peptides, allele_encoding, flanking_encoding=None):
        """

        Parameters
        ----------
        peptides : EncodableSequences or list of string

        allele_encoding : AlleleEncoding

        flanking_encoding: Flank


        Returns
        -------
        numpy.array
        """
        assert self.affinity_model is not None

        (allele_input, allele_representations) = (
            self.affinity_model.allele_encoding_to_network_input(
                allele_encoding))
        peptides = EncodableSequences.create(peptides)
        x_dict = {
            'peptide': self.affinity_model.peptides_to_network_input(peptides),
            'allele_set': allele_input,
        }
        if self.hyperparameters['include_cleavage']:
            assert self.cleavage_model  is not None
            numpy.testing.assert_array_equal(
                peptides.sequences,
                flanking_encoding.dataframe.peptide.values)
            if flanking_encoding is None:
                raise RuntimeError("flanking_encoding required")
            cleavage_x_dict = self.cleavage_model.network_input(
                flanking_encoding)
            x_dict.update(cleavage_x_dict)
        if self.hyperparameters['auxiliary_input_features']:
            auxiliary_encoder = AuxiliaryInputEncoder(
                alleles=allele_encoding.alleles,
                peptides=peptides.sequences)
            x_dict[
                'auxiliary'
            ] = auxiliary_encoder.get_array(
                features=self.hyperparameters['auxiliary_input_features'],
                feature_parameters=self.hyperparameters[
                    'auxiliary_input_feature_parameters']) * 0.01
        #import ipdb;ipdb.set_trace()
        return (x_dict, allele_representations)

    def fit(
            self,
            targets,
            peptides,
            allele_encoding,
            flanking_encoding=None,
            sample_weights=None,
            shuffle_permutation=None,
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0):

        import keras.backend as K

        assert isinstance(allele_encoding, MultipleAlleleEncoding)
        assert (
            allele_encoding.max_alleles_per_experiment ==
            self.hyperparameters['max_alleles'])

        (x_dict, allele_representations) = (
            self.network_input(
                peptides=peptides,
                allele_encoding=allele_encoding,
                flanking_encoding=flanking_encoding))

        # Shuffle
        if shuffle_permutation is None:
            shuffle_permutation = numpy.random.permutation(len(targets))
        targets = numpy.array(targets)[shuffle_permutation]
        assert numpy.isnan(targets).sum() == 0, targets
        if sample_weights is not None:
            sample_weights = numpy.array(sample_weights)[shuffle_permutation]
        for key in list(x_dict):
            x_dict[key] = x_dict[key][shuffle_permutation]
        del peptides
        del allele_encoding
        del flanking_encoding

        fit_info = collections.defaultdict(list)

        allele_representations_hash = self.set_allele_representations(
            allele_representations)

        self.network.compile(
            loss=self.loss,
            optimizer=self.hyperparameters['optimizer'])
        if self.hyperparameters['learning_rate'] is not None:
            K.set_value(
                self.network.optimizer.lr,
                self.hyperparameters['learning_rate'])
        fit_info["learning_rate"] = float(K.get_value(self.network.optimizer.lr))

        if verbose:
            self.network.summary()

        training_start = time.time()

        min_val_loss_iteration = None
        min_val_loss = None
        last_progress_print = 0
        for i in range(self.hyperparameters['max_epochs']):
            epoch_start = time.time()
            self.assert_allele_representations_hash(allele_representations_hash)
            fit_history = self.network.fit(
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

            if numpy.isnan(fit_info['loss'][-1]):
                raise ValueError("NaN loss")

            # Print progress no more often than once every few seconds.
            if progress_print_interval is not None and (
                    not last_progress_print or (
                        time.time() - last_progress_print
                        > progress_print_interval)):
                print((progress_preamble + " " +
                       "Epoch %3d / %3d [%0.2f sec]: loss=%g val_loss=%g. "
                       "Min val loss (%s) at epoch %s" % (
                           i,
                           self.hyperparameters['max_epochs'],
                           epoch_time,
                           fit_info['loss'][-1],
                           (
                               fit_info['val_loss'][-1]
                               if 'val_loss' in fit_info else numpy.nan
                           ),
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

        fit_info["time"] = time.time() - training_start
        fit_info["num_points"] = len(targets)
        self.fit_info.append(dict(fit_info))

    def predict(
            self,
            peptides,
            allele_encoding,
            flanking_encoding=None,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):

        peptides = EncodableSequences.create(peptides)
        assert isinstance(allele_encoding, MultipleAlleleEncoding)

        (x_dict, allele_representations) = self.network_input(
            peptides=peptides,
            allele_encoding=allele_encoding,
            flanking_encoding=flanking_encoding)

        self.set_allele_representations(allele_representations)
        raw_predictions = self.network.predict(x_dict, batch_size=batch_size)
        return raw_predictions

    def clear_allele_representations(self):
        """
        Set allele representations to an empty array. Useful before saving to
        save a smaller version of the model.
        """
        layer = self.network.get_layer("allele_representation")
        existing_weights_shape = (layer.input_dim, layer.output_dim)
        self.set_allele_representations(
            numpy.zeros(shape=(0,) + existing_weights_shape[1:]),
            force_surgery=True)

    def set_allele_representations(self, allele_representations, force_surgery=False):
        """
        """
        from keras.models import clone_model
        import keras.backend as K
        import tensorflow as tf

        reshaped = allele_representations.reshape((
            allele_representations.shape[0],
            numpy.product(allele_representations.shape[1:])))
        original_model = self.network

        layer = original_model.get_layer("allele_representation")
        existing_weights_shape = (layer.input_dim, layer.output_dim)

        # Only changes to the number of supported alleles (not the length of
        # the allele sequences) are allowed.
        assert existing_weights_shape[1:] == reshaped.shape[1:]

        if existing_weights_shape[0] > reshaped.shape[0] and not force_surgery:
            # Extend with NaNs so we can avoid having to reshape the weights
            # matrix, which is expensive.
            reshaped = numpy.append(
                reshaped,
                numpy.ones([
                    existing_weights_shape[0] - reshaped.shape[0],
                    reshaped.shape[1]
                ]) * numpy.nan,
                axis=0)

        if existing_weights_shape != reshaped.shape:
            print(
                "Performing network surgery", existing_weights_shape, reshaped.shape)
            # Network surgery required. Make a new network with this layer's
            # dimensions changed. Kind of a hack.
            layer.input_dim = reshaped.shape[0]
            new_model = clone_model(original_model)

            # copy weights for other layers over
            for layer in new_model.layers:
                if layer.name != "allele_representation":
                    layer.set_weights(
                        original_model.get_layer(name=layer.name).get_weights())

            self.network = new_model

            layer = new_model.get_layer("allele_representation")

            # Disable the old model to catch bugs.
            def throw(*args, **kwargs):
                raise RuntimeError("Using a disabled model!")
            original_model.predict = \
                original_model.fit = \
                original_model.fit_generator = throw

        layer.set_weights([reshaped])
        self.allele_representation_hash = hashlib.sha1(
            allele_representations.tobytes()).hexdigest()
        return self.allele_representation_hash

    def assert_allele_representations_hash(self, value):
        numpy.testing.assert_equal(self.allele_representation_hash, value)

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
        Class1NeuralNetwork
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
