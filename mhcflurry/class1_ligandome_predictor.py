from __future__ import print_function

import time
import collections
from functools import partial

import numpy
import pandas

from .hyperparameters import HyperparameterDefaults
from .class1_neural_network import Class1NeuralNetwork, DEFAULT_PREDICT_BATCH_SIZE
from .encodable_sequences import EncodableSequences
from .regression_target import from_ic50, to_ic50
from .random_negative_peptides import RandomNegativePeptides
from .custom_loss import (
    MSEWithInequalities,
    MSEWithInequalitiesAndMultipleOutputs,
    MultiallelicMassSpecLoss)

class Class1LigandomePredictor(object):
    network_hyperparameter_defaults = HyperparameterDefaults(
        allele_amino_acid_encoding="BLOSUM62",
        peptide_encoding={
            'vector_encoding_name': 'BLOSUM62',
            'alignment_method': 'left_pad_centered_right_pad',
            'max_length': 15,
        },
        additional_dense_layers=[],
        additional_dense_activation="sigmoid",
    )
    """
    Hyperparameters (and their default values) that affect the neural network
    architecture.
    """

    fit_hyperparameter_defaults = HyperparameterDefaults(
        max_epochs=500,
        validation_split=0.1,
        early_stopping=True,
        minibatch_size=128,).extend(
        RandomNegativePeptides.hyperparameter_defaults
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
        loss_delta=0.2,
        optimizer="rmsprop",
        learning_rate=None,
    )
    """
    Loss and optimizer hyperparameters. Any values supported by keras may be
    used.
    """

    allele_features_hyperparameter_defaults = HyperparameterDefaults(
        allele_features_include_gene=True,
    )
    """
    Allele feature hyperparameters.
    """

    hyperparameter_defaults = network_hyperparameter_defaults.extend(
        fit_hyperparameter_defaults).extend(
        early_stopping_hyperparameter_defaults).extend(
        compile_hyperparameter_defaults).extend(
        allele_features_hyperparameter_defaults)

    def __init__(
            self,
            class1_affinity_predictor,
            max_ensemble_size=None,
            **hyperparameters):
        if not class1_affinity_predictor.class1_pan_allele_models:
            raise NotImplementedError("Pan allele models required")
        if class1_affinity_predictor.allele_to_allele_specific_models:
            raise NotImplementedError("Only pan allele models are supported")

        self.hyperparameters = self.hyperparameter_defaults.with_defaults(
            hyperparameters)

        models = class1_affinity_predictor.class1_pan_allele_models
        if max_ensemble_size is not None:
            models = models[:max_ensemble_size]

        self.network = self.make_network(
            models,
            self.hyperparameters)

        self.fit_info = []

    @staticmethod
    def make_network(pan_allele_class1_neural_networks, hyperparameters):
        import keras.backend as K
        from keras.layers import (
            Input,
            TimeDistributed,
            Dense,
            Flatten,
            RepeatVector,
            concatenate,
            Reshape,
            Embedding)
        from keras.models import Model

        networks = [model.network() for model in pan_allele_class1_neural_networks]
        merged_ensemble = Class1NeuralNetwork.merge(
            networks,
            merge_method="average")

        peptide_shape = tuple(
            int(x) for x in K.int_shape(merged_ensemble.inputs[0])[1:])

        input_alleles = Input(shape=(6,), name="allele")  # up to 6 alleles
        input_peptides = Input(
            shape=peptide_shape,
            dtype='float32',
            name='peptide')

        peptides_flattened = Flatten()(input_peptides)
        peptides_repeated = RepeatVector(6)(peptides_flattened)

        allele_representation = Embedding(
            name="allele_representation",
            input_dim=64,  # arbitrary, how many alleles to have room for
            output_dim=1029,
            input_length=6,
            trainable=False)(input_alleles)

        allele_flat = Reshape((6, -1), name="allele_flat")(allele_representation)

        allele_peptide_merged = concatenate(
            [peptides_repeated, allele_flat], name="allele_peptide_merged")

        layer_names = [
            layer.name for layer in merged_ensemble.layers
        ]

        pan_allele_layer_initial_names = [
            'allele', 'peptide',
            'allele_representation', 'flattened_0', 'allele_flat',
            'allele_peptide_merged', 'dense_0', 'dropout_0',
        ]

        def startswith(lst, prefix):
            return lst[:len(prefix)] == prefix

        assert startswith(layer_names, pan_allele_layer_initial_names), layer_names

        layers = merged_ensemble.layers[
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
            print(layer, layer.name, node, lifted)

            layer_name_to_new_node[layer.name] = node

        affinity_predictor_output = node

        for (i, layer_size) in enumerate(
                hyperparameters['additional_dense_layers']):
            layer = Dense(
                layer_size,
                activation=hyperparameters['additional_dense_activation'])
            lifted = TimeDistributed(layer)
            node = lifted(node)

        layer = Dense(1, activation="sigmoid")
        lifted = TimeDistributed(layer)
        ligandome_output = lifted(node)

        network = Model(
            inputs=[input_peptides, input_alleles],
            outputs=[affinity_predictor_output, ligandome_output],
            name="ligandome",
        )
        network.summary()
        return network

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
            shuffle_permutation=None,
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0):

        import keras.backend as K

        #for layer in self.network._layers[:8]:
        #    print("Setting non trainable", layer)
        #    layer.trainable = False
        #    import ipdb ; ipdb.set_trace()

        peptides = EncodableSequences.create(peptides)
        peptide_encoding = self.peptides_to_network_input(peptides)

        # Optional optimization
        allele_encoding = allele_encoding.compact()

        (allele_encoding_input, allele_representations) = (
            self.allele_encoding_to_network_input(allele_encoding))

        # Shuffle
        if shuffle_permutation is None:
            shuffle_permutation = numpy.random.permutation(len(labels))
        peptide_encoding = peptide_encoding[shuffle_permutation]
        allele_encoding_input = allele_encoding_input[shuffle_permutation]
        labels = labels[shuffle_permutation]

        x_dict = {
            'peptide': peptide_encoding,
            'allele': allele_encoding_input,
        }

        loss = MSEWithInequalitiesAndMultipleOutputs(losses=[
            MSEWithInequalities(),
            MultiallelicMassSpecLoss(
                delta=self.hyperparameters['loss_delta']),
        ])

        y_values_pre_encoding = labels.copy()
        if affinities_mask is not None:
            y_values_pre_encoding[affinities_mask] = from_ic50(labels)
        y_values = loss.encode_y(
            y_values_pre_encoding,
            inequalities=inequalities[affinities_mask] if inequalities is not None else None,
            output_indices=(~affinities_mask).astype(int) if affinities_mask is not None else numpy.ones(len(y_values_pre_encoding), dtype=int))

        fit_info = collections.defaultdict(list)

        self.set_allele_representations(allele_representations)
        self.network.compile(
            loss=loss.loss,
            optimizer=self.hyperparameters['optimizer'])
        if self.hyperparameters['learning_rate'] is not None:
            K.set_value(
                self.network.optimizer.lr,
                self.hyperparameters['learning_rate'])
        fit_info["learning_rate"] = float(
            K.get_value(self.network.optimizer.lr))

        if verbose:
            self.network.summary()

        min_val_loss_iteration = None
        min_val_loss = None
        last_progress_print = 0
        start = time.time()
        for i in range(self.hyperparameters['max_epochs']):
            epoch_start = time.time()

            # TODO: need to use fit_generator to keep each minibatch corresponding
            # to a single experiment
            fit_history = self.network.fit(
                x_dict,
                labels,
                shuffle=True,
                batch_size=self.hyperparameters['minibatch_size'],
                verbose=verbose,
                epochs=i + 1,
                initial_epoch=i,
                validation_split=self.hyperparameters['validation_split'],
            )
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
        fit_info["num_points"] = len(peptides)
        self.fit_info.append(dict(fit_info))

    def predict(
            self,
            peptides,
            allele_encoding,
            output="affinities",
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):
        (allele_encoding_input, allele_representations) = (
                self.allele_encoding_to_network_input(allele_encoding.compact()))
        self.set_allele_representations(allele_representations)
        x_dict = {
            'peptide': self.peptides_to_network_input(peptides),
            'allele': allele_encoding_input,
        }
        predictions = self.network.predict(x_dict, batch_size=batch_size)
        if output == "affinities":
            predictions = to_ic50(predictions[0])
        elif output == "ligandome_presentation":
            predictions = predictions[1]
        elif output == "both":
            pass
        else:
            raise NotImplementedError("Unknown output", output)
        return numpy.squeeze(predictions)

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
            print("Performing network surgery", existing_weights_shape, reshaped.shape)
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

    @staticmethod
    def allele_features(allele_names, hyperparameters):
        df = pandas.DataFrame({"allele_name": allele_names})
        if hyperparameters['allele_features_include_gene']:
            # TODO: support other organisms.
            for gene in ["A", "B", "C"]:
                df[gene] = df.allele_name.str.startswith(
                    "HLA-%s" % gene).astype(float)
        return gene
