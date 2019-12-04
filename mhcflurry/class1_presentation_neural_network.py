from __future__ import print_function

import time
import collections
from six import string_types

import numpy
import pandas
import mhcnames
import hashlib

from .hyperparameters import HyperparameterDefaults
from .class1_neural_network import Class1NeuralNetwork, DEFAULT_PREDICT_BATCH_SIZE
from .encodable_sequences import EncodableSequences
from .regression_target import from_ic50, to_ic50
from .random_negative_peptides import RandomNegativePeptides
from .allele_encoding import MultipleAlleleEncoding, AlleleEncoding
from .auxiliary_input import AuxiliaryInputEncoder
from .batch_generator import MultiallelicMassSpecBatchGenerator
from .custom_loss import (
    MSEWithInequalities,
    MultiallelicMassSpecLoss,
    ZeroLoss)


class Class1PresentationNeuralNetwork(object):
    network_hyperparameter_defaults = HyperparameterDefaults(
        allele_amino_acid_encoding="BLOSUM62",
        peptide_encoding={
            'vector_encoding_name': 'BLOSUM62',
            'alignment_method': 'left_pad_centered_right_pad',
            'max_length': 15,
        },
        max_alleles=6,
    )
    """
    Hyperparameters (and their default values) that affect the neural network
    architecture.
    """

    fit_hyperparameter_defaults = HyperparameterDefaults(
        max_epochs=500,
        early_stopping=True,
        random_negative_affinity_min=20000.0,).extend(
        RandomNegativePeptides.hyperparameter_defaults).extend(
        MultiallelicMassSpecBatchGenerator.hyperparameter_defaults
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
        loss_multiallelic_mass_spec_delta=0.2,
        loss_multiallelic_mass_spec_multiplier=1.0,
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

    def lift_from_class1_neural_network(self, class1_neural_network):
        import keras.backend as K
        from keras.layers import (
            Input,
            TimeDistributed,
            Dense,
            Flatten,
            RepeatVector,
            concatenate,
            Activation,
            Lambda,
            Add,
            Embedding)
        from keras.models import Model

        peptide_shape = tuple(
            int(x) for x in K.int_shape(class1_neural_network.inputs[0])[1:])

        input_alleles = Input(
            shape=(self.hyperparameters['max_alleles'],), name="allele")
        input_peptides = Input(
            shape=peptide_shape,
            dtype='float32',
            name='peptide')

        peptides_flattened = Flatten()(input_peptides)
        peptides_repeated = RepeatVector(self.hyperparameters['max_alleles'])(
            peptides_flattened)

        allele_representation = Embedding(
            name="allele_representation",
            input_dim=64,  # arbitrary, how many alleles to have room for
            output_dim=1029,
            input_length=self.hyperparameters['max_alleles'],
            trainable=False,
            mask_zero=True)(input_alleles)

        allele_flat = allele_representation

        allele_peptide_merged = concatenate(
            [peptides_repeated, allele_flat], name="allele_peptide_merged")

        layer_names = [
            layer.name for layer in class1_neural_network.layers
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

        layers = class1_neural_network.layers[
            pan_allele_layer_initial_names.index(
                "allele_peptide_merged") + 1:
        ]
        node = allele_peptide_merged
        layer_name_to_new_node = {
            "allele_peptide_merged": allele_peptide_merged,
        }
        for layer in layers:
            assert layer.name not in layer_name_to_new_node
            input_layer_names = []
            for inbound_node in layer._inbound_nodes:
                for inbound_layer in inbound_node.inbound_layers:
                    input_layer_names.append(inbound_layer.name)
            input_nodes = [
                layer_name_to_new_node[name]
                for name in input_layer_names
            ]

            if len(input_nodes) == 1:
                lifted = TimeDistributed(layer)
                node = lifted(input_nodes[0])
            else:
                node = layer(input_nodes)
            layer_name_to_new_node[layer.name] = node

        affinity_predictor_matrix_output = node

        affinity_predictor_output = Lambda(
            lambda x: x[:, 0], name="affinity_output")(
                affinity_predictor_matrix_output)

        auxiliary_input = None
        if self.hyperparameters['auxiliary_input_features']:
            auxiliary_input = Input(
                shape=(
                    self.hyperparameters['max_alleles'],
                    len(
                        AuxiliaryInputEncoder.get_columns(
                            self.hyperparameters['auxiliary_input_features'],
                            feature_parameters=self.hyperparameters[
                                'auxiliary_input_feature_parameters']))),
                dtype="float32",
                name="auxiliary")
            node = concatenate(
                [node, auxiliary_input], name="affinities_with_auxiliary")

        layer = Dense(8, activation="tanh")
        lifted = TimeDistributed(layer, name="presentation_hidden1")
        node = lifted(node)

        layer = Dense(1, activation="tanh")
        lifted = TimeDistributed(layer, name="presentation_hidden2")
        presentation_adjustment = lifted(node)

        def logit(x):
            import tensorflow as tf
            return - tf.log(1. / x - 1.)

        presentation_output_pre_sigmoid = Add()([
            Lambda(logit)(affinity_predictor_matrix_output),
            presentation_adjustment,
        ])
        presentation_output = Activation("sigmoid", name="presentation_output")(
            presentation_output_pre_sigmoid)

        self.network = Model(
            inputs=[
                input_peptides,
                input_alleles,
            ] + ([] if auxiliary_input is None else [auxiliary_input]),
            outputs=[
                affinity_predictor_output,
                presentation_output,
                affinity_predictor_matrix_output
            ],
            name="presentation",
        )
        self.network.summary()

    def peptides_to_network_input(self, peptides):
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
        encoder = EncodableSequences.create(peptides)
        encoded = encoder.variable_length_to_fixed_length_vector_encoding(
            **self.hyperparameters['peptide_encoding'])
        assert len(encoded) == len(peptides)
        return encoded

    def allele_encoding_to_network_input(self, allele_encoding):
        """
        Encode alleles to the fixed-length encoding expected by the neural
        network (which depends on the architecture).

        Parameters
        ----------
        allele_encoding : AlleleEncoding

        Returns
        -------
        (numpy.array, numpy.array)

        Indices and allele representations.

        """
        return (
            allele_encoding.indices,
            allele_encoding.allele_representations(
                self.hyperparameters['allele_amino_acid_encoding']))

    def fit(
            self,
            peptides,
            labels,
            allele_encoding,
            affinities_mask=None,  # True when a peptide/label is actually a peptide and an affinity
            inequalities=None,  # interpreted only for elements where affinities_mask is True, otherwise ignored
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0):

        import keras.backend as K

        assert isinstance(allele_encoding, MultipleAlleleEncoding)
        assert (
            allele_encoding.max_alleles_per_experiment ==
            self.hyperparameters['max_alleles'])

        encodable_peptides = EncodableSequences.create(peptides)

        if labels is not None:
            labels = numpy.array(labels, copy=False)
        if inequalities is not None:
            inequalities = numpy.array(inequalities, copy=True)
        else:
            inequalities = numpy.tile("=", len(labels))
        if affinities_mask is not None:
            affinities_mask = numpy.array(affinities_mask, copy=False)
        else:
            affinities_mask = numpy.tile(False, len(labels))
        inequalities[~affinities_mask] = "="

        random_negatives_planner = RandomNegativePeptides(
            **RandomNegativePeptides.hyperparameter_defaults.subselect(
                self.hyperparameters))
        random_negatives_planner.plan(
            peptides=encodable_peptides.sequences,
            affinities=numpy.where(affinities_mask, labels, to_ic50(labels)),
            alleles=[
                numpy.random.choice(row[row != numpy.array(None)])
                for row in allele_encoding.alleles
            ],
            inequalities=inequalities)

        peptide_input = self.peptides_to_network_input(encodable_peptides)

        # Optional optimization
        (allele_encoding_input, allele_representations) = (
            self.allele_encoding_to_network_input(allele_encoding))

        x_dict_without_random_negatives = {
            'peptide': peptide_input,
            'allele': allele_encoding_input,
        }
        if self.hyperparameters['auxiliary_input_features']:
            auxiliary_encoder = AuxiliaryInputEncoder(
                alleles=allele_encoding.alleles,
                peptides=peptides)
            x_dict_without_random_negatives[
                'auxiliary'
            ] = auxiliary_encoder.get_array(
                features=self.hyperparameters['auxiliary_input_features'],
                feature_parameters=self.hyperparameters[
                    'auxiliary_input_feature_parameters'])

        y1 = numpy.zeros(shape=len(labels))
        y1[affinities_mask] = from_ic50(labels[affinities_mask])

        random_negative_alleles = random_negatives_planner.get_alleles()
        random_negatives_allele_encoding = MultipleAlleleEncoding(
            experiment_names=random_negative_alleles,
            experiment_to_allele_list=dict(
                (a, [a]) for a in random_negative_alleles),
            max_alleles_per_experiment=(
                allele_encoding.max_alleles_per_experiment),
            borrow_from=allele_encoding.allele_encoding)
        num_random_negatives = random_negatives_planner.get_total_count()

        # Reverse inequalities because from_ic50() flips the direction
        # (i.e. lower affinity results in higher y values).
        adjusted_inequalities = pandas.Series(inequalities).map({
            "=": "=",
            ">": "<",
            "<": ">",
        }).values
        adjusted_inequalities[~affinities_mask] = ">"

        # Note: we are using "<" here not ">" because the inequalities are
        # now in target-space (0-1) not affinity-space.
        adjusted_inequalities_with_random_negative = numpy.concatenate([
            numpy.tile("<", num_random_negatives),
            adjusted_inequalities
        ])
        random_negative_ic50 = self.hyperparameters[
            'random_negative_affinity_min'
        ]
        y1_with_random_negatives = numpy.concatenate([
            numpy.tile(
                from_ic50(random_negative_ic50), num_random_negatives),
            y1,
        ])

        affinities_loss = MSEWithInequalities()
        encoded_y1 = affinities_loss.encode_y(
            y1_with_random_negatives,
            inequalities=adjusted_inequalities_with_random_negative)

        mms_loss = MultiallelicMassSpecLoss(
            delta=self.hyperparameters['loss_multiallelic_mass_spec_delta'],
            multiplier=self.hyperparameters[
                'loss_multiallelic_mass_spec_multiplier'])
        y2 = labels.copy()
        y2[affinities_mask] = -1
        y2_with_random_negatives = numpy.concatenate([
            numpy.tile(0.0, num_random_negatives),
            y2,
        ])
        encoded_y2 = mms_loss.encode_y(y2_with_random_negatives)

        fit_info = collections.defaultdict(list)

        allele_representations_hash = self.set_allele_representations(
            allele_representations)
        self.network.compile(
            loss=[affinities_loss.loss, mms_loss.loss, ZeroLoss.loss],
            optimizer=self.hyperparameters['optimizer'])
        if self.hyperparameters['learning_rate'] is not None:
            K.set_value(
                self.network.optimizer.lr,
                self.hyperparameters['learning_rate'])
        fit_info["learning_rate"] = float(
            K.get_value(self.network.optimizer.lr))

        if verbose:
            self.network.summary()

        batch_generator = MultiallelicMassSpecBatchGenerator(
            MultiallelicMassSpecBatchGenerator.hyperparameter_defaults.subselect(
                self.hyperparameters))
        start = time.time()
        batch_generator.plan(
            affinities_mask=numpy.concatenate([
                numpy.tile(True, num_random_negatives),
                affinities_mask
            ]),
            experiment_names=numpy.concatenate([
                numpy.tile(None, num_random_negatives),
                allele_encoding.experiment_names
            ]),
            alleles_matrix=numpy.concatenate([
                random_negatives_allele_encoding.alleles,
                allele_encoding.alleles,
            ]),
            is_binder=numpy.concatenate([
                numpy.tile(False, num_random_negatives),
                numpy.where(affinities_mask, labels, to_ic50(labels)) < 1000.0
            ]),
            potential_validation_mask=numpy.concatenate([
                numpy.tile(False, num_random_negatives),
                numpy.tile(True, len(labels))
            ]),
        )
        if verbose:
            print("Generated batch generation plan in %0.2f sec." % (
                time.time() - start))
            print(batch_generator.summary())

        min_val_loss_iteration = None
        min_val_loss = None
        last_progress_print = 0
        start = time.time()
        x_dict_with_random_negatives = {}
        for i in range(self.hyperparameters['max_epochs']):
            epoch_start = time.time()

            random_negative_peptides = EncodableSequences.create(
                random_negatives_planner.get_peptides())
            random_negative_peptides_encoding = (
                self.peptides_to_network_input(random_negative_peptides))

            if not x_dict_with_random_negatives:
                if len(random_negative_peptides) > 0:
                    x_dict_with_random_negatives[
                        "peptide"
                    ] = numpy.concatenate([
                        random_negative_peptides_encoding,
                        x_dict_without_random_negatives['peptide'],
                    ])
                    x_dict_with_random_negatives[
                        'allele'
                    ] = numpy.concatenate([
                        self.allele_encoding_to_network_input(
                            random_negatives_allele_encoding)[0],
                        x_dict_without_random_negatives['allele']
                    ])
                    if 'auxiliary' in x_dict_without_random_negatives:
                        random_negative_auxiliary_encoder = AuxiliaryInputEncoder(
                            alleles=random_negatives_allele_encoding.alleles,
                            #peptides=random_negative_peptides.sequences
                        )
                        x_dict_with_random_negatives['auxiliary'] = (
                            numpy.concatenate([
                                random_negative_auxiliary_encoder.get_array(
                                    features=self.hyperparameters[
                                        'auxiliary_input_features'],
                                    feature_parameters=self.hyperparameters[
                                        'auxiliary_input_feature_parameters']),
                                x_dict_without_random_negatives['auxiliary']
                            ]))
                else:
                    x_dict_with_random_negatives = (
                        x_dict_without_random_negatives)
            else:
                # Update x_dict_with_random_negatives in place.
                # This is more memory efficient than recreating it as above.
                if len(random_negative_peptides) > 0:
                    x_dict_with_random_negatives[
                        "peptide"
                    ][:num_random_negatives] = random_negative_peptides_encoding

            (train_generator, test_generator) = (
                batch_generator.get_train_and_test_generators(
                    x_dict=x_dict_with_random_negatives,
                    y_list=[encoded_y1, encoded_y2, encoded_y2],
                    epochs=1))
            self.assert_allele_representations_hash(allele_representations_hash)
            fit_history = self.network.fit_generator(
                train_generator,
                steps_per_epoch=batch_generator.num_train_batches,
                epochs=i + 1,
                initial_epoch=i,
                verbose=verbose,
                use_multiprocessing=False,
                workers=0,
                validation_data=test_generator,
                validation_steps=batch_generator.num_test_batches)

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

            if batch_generator.num_test_batches:
                #import ipdb ; ipdb.set_trace()
                val_loss = fit_info['val_loss'][-1]
                if min_val_loss is None or (
                        val_loss < min_val_loss -
                        self.hyperparameters['min_delta']):
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
        fit_info["num_points"] = len(labels)
        self.fit_info.append(dict(fit_info))

        return {
            'batch_generator': batch_generator,
            'last_x': x_dict_with_random_negatives,
            'last_y': [encoded_y1, encoded_y2, encoded_y2],
            'fit_info': fit_info,
        }

    Predictions = collections.namedtuple(
        "ligandone_neural_network_predictions",
        "score affinity")

    def predict(
            self,
            peptides,
            allele_encoding=None,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):

        peptides = EncodableSequences.create(peptides)
        assert isinstance(AlleleEncoding, MultipleAlleleEncoding)

        (allele_encoding_input, allele_representations) = (
                self.allele_encoding_to_network_input(allele_encoding.compact()))
        self.set_allele_representations(allele_representations)
        x_dict = {
            'peptide': self.peptides_to_network_input(peptides),
            'allele': allele_encoding_input,
        }
        if self.hyperparameters['auxiliary_input_features']:
            auxiliary_encoder = AuxiliaryInputEncoder(
                alleles=allele_encoding.alleles,
                peptides=peptides.sequences)
            x_dict[
                'auxiliary'
            ] = auxiliary_encoder.get_array(
                features=self.hyperparameters['auxiliary_input_features'],
                feature_parameters=self.hyperparameters[
                    'auxiliary_input_feature_parameters'])

        predictions = self.Predictions._make([
            numpy.squeeze(output)
            for output in self.network.predict(
                x_dict, batch_size=batch_size)[1:]
        ])
        return predictions

    def set_allele_representations(self, allele_representations):
        """
        """
        from keras.models import clone_model
        import keras.backend as K
        import tensorflow as tf

        reshaped = allele_representations.reshape(
            (allele_representations.shape[0], -1))
        original_model = self.network

        layer = original_model.get_layer("allele_representation")
        existing_weights_shape = (layer.input_dim, layer.output_dim)

        # Only changes to the number of supported alleles (not the length of
        # the allele sequences) are allowed.
        assert existing_weights_shape[1:] == reshaped.shape[1:]

        if existing_weights_shape[0] > reshaped.shape[0]:
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
        result = dict(self.__dict__)
        result['network'] = None
        result['network_json'] = None
        result['network_weights'] = None

        if self.network is not None:
            result['network_json'] = self.network.to_json()
            result['network_weights'] = self.network.get_weights()
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

    def get_config(self):
        """
        serialize to a dict all attributes except model weights

        Returns
        -------
        dict
        """
        result = dict(self.__dict__)
        result['network'] = None
        result['network_weights'] = None
        result['network_json'] = None
        if self.network:
            result['network_weights'] = self.network.get_weights()
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
        network_weights = config.pop('network_weights')
        instance.__dict__.update(config)
        assert instance.network is None
        if network_json is not None:
            import keras.models
            instance.network = keras.models.model_from_json(network_json)
            if network_weights is not None:
                instance.network.set_weights(network_weights)
        return instance