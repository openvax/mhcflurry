import time
import collections
import logging

import numpy
import pandas

import keras.models
import keras.layers.pooling
import keras.regularizers
from keras.layers import Input
import keras.layers.merge
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization

from mhcflurry.hyperparameters import HyperparameterDefaults

from ..encodable_sequences import EncodableSequences
from ..regression_target import to_ic50, from_ic50
from ..common import random_peptides, amino_acid_distribution


class Class1NeuralNetwork(object):
    weights_filename_extension = "npz"

    network_hyperparameter_defaults = HyperparameterDefaults(
        kmer_size=15,
        use_embedding=True,
        embedding_input_dim=21,
        embedding_output_dim=8,
        pseudosequence_use_embedding=True,
        layer_sizes=[32],
        dense_layer_l1_regularization=0.0,
        dense_layer_l2_regularization=0.0,
        activation="tanh",
        init="glorot_uniform",
        output_activation="sigmoid",
        dropout_probability=0.0,
        batch_normalization=True,
        embedding_init_method="glorot_uniform",
        locally_connected_layers=[],
    )

    compile_hyperparameter_defaults = HyperparameterDefaults(
        loss="mse",
        optimizer="rmsprop",
    )

    input_encoding_hyperparameter_defaults = HyperparameterDefaults(
        left_edge=4,
        right_edge=4)

    fit_hyperparameter_defaults = HyperparameterDefaults(
        max_epochs=250,
        validation_split=None,
        early_stopping=False,
        take_best_epoch=False,
        random_negative_rate=0.0,
        random_negative_constant=0,
        random_negative_affinity_min=50000.0,
        random_negative_affinity_max=50000.0,
        random_negative_match_distribution=True,
        random_negative_distribution_smoothing=0.0)

    early_stopping_hyperparameter_defaults = HyperparameterDefaults(
        monitor='val_loss',
        min_delta=0,
        patience=0,
        verbose=1,
        mode='auto')

    hyperparameter_defaults = network_hyperparameter_defaults.extend(
        compile_hyperparameter_defaults).extend(
        input_encoding_hyperparameter_defaults).extend(
        fit_hyperparameter_defaults).extend(
        early_stopping_hyperparameter_defaults)

    def __init__(self, **hyperparameters):
        self.hyperparameters = self.hyperparameter_defaults.with_defaults(
            hyperparameters)
        self.network = None
        self.loss_history = None
        self.fit_seconds = None
        self.fit_num_points = None

    def get_config(self):
        result = dict(self.__dict__)
        del result['network']
        result['network_json'] = self.network.to_json()
        return result

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        instance = cls(**config.pop('hyperparameters'))
        instance.network = keras.models.model_from_json(
            config.pop('network_json'))
        instance.__dict__.update(config)
        return instance

    def __getstate__(self):
        result = self.get_config()
        result['network_weights'] = self.get_weights()
        return result

    def __setstate__(self, state):
        network_json = state.pop('network_json')
        network_weights = state.pop('network_weights')
        self.__dict__.update(state)
        self.network = keras.models.model_from_json(network_json)
        self.set_weights(network_weights)

    def save_weights(self, filename):
        weights_list = self.network.get_weights()
        numpy.savez(
            filename,
            **dict((("array_%d" % i), w) for (i, w) in enumerate(weights_list)))

    def restore_weights(self, filename):
        loaded = numpy.load(filename)
        weights = [
            loaded["array_%d" % i]
            for i in range(len(loaded.keys()))
        ]
        loaded.close()
        self.network.set_weights(weights)

    def peptides_to_network_input(self, peptides):
        encoder = EncodableSequences.create(peptides)
        if self.hyperparameters['use_embedding']:
            encoded = encoder.variable_length_to_fixed_length_categorical(
                max_length=self.hyperparameters['kmer_size'],
                **self.input_encoding_hyperparameter_defaults.subselect(
                    self.hyperparameters))
        else:
            encoded = encoder.variable_length_to_fixed_length_one_hot(
                max_length=self.hyperparameters['kmer_size'],
                **self.input_encoding_hyperparameter_defaults.subselect(
                    self.hyperparameters))
        assert len(encoded) == len(peptides)
        return encoded

    def pseudosequence_to_network_input(self, pseudosequences):
        encoder = EncodableSequences.create(pseudosequences)
        if self.hyperparameters['pseudosequence_use_embedding']:
            encoded = encoder.fixed_length_categorical()
        else:
            encoded = encoder.fixed_length_one_hot()
        assert len(encoded) == len(pseudosequences)
        return encoded

    def fit(
            self,
            peptides,
            affinities,
            allele_pseudosequences=None,
            sample_weights=None,
            verbose=1):

        self.fit_num_points = len(peptides)

        encodable_peptides = EncodableSequences.create(peptides)
        peptide_encoding = self.peptides_to_network_input(encodable_peptides)

        length_counts = (
            pandas.Series(encodable_peptides.sequences)
            .str.len().value_counts().to_dict())

        num_random_negative = {}
        for length in range(8, 16):
            num_random_negative[length] = int(
                length_counts.get(length, 0) *
                self.hyperparameters['random_negative_rate'] +
                self.hyperparameters['random_negative_constant'])
        num_random_negative = pandas.Series(num_random_negative)
        logging.info("Random negative counts per length:\n%s" % (
            str(num_random_negative)))

        aa_distribution = None
        if self.hyperparameters['random_negative_match_distribution']:
            aa_distribution = amino_acid_distribution(
                encodable_peptides.sequences,
                smoothing=self.hyperparameters[
                    'random_negative_distribution_smoothing'])
            logging.info(
                "Using amino acid distribution for random negative:\n%s" % (
                str(aa_distribution)))

        y_values = from_ic50(affinities)
        assert numpy.isnan(y_values).sum() == 0, numpy.isnan(y_values).sum()

        x_dict_without_random_negatives = {
            'peptide': peptide_encoding,
        }
        pseudosequence_length = None
        if allele_pseudosequences is not None:
            pseudosequences_input = self.pseudosequence_to_network_input(
                allele_pseudosequences)
            pseudosequence_length = len(pseudosequences_input[0])
            x_dict_without_random_negatives['pseudosequence'] = (
                pseudosequences_input)

        if self.network is None:
            self.network = self.make_network(
                pseudosequence_length=pseudosequence_length,
                **self.network_hyperparameter_defaults.subselect(
                    self.hyperparameters))
            self.compile()

        y_dict_with_random_negatives = {
            "output": numpy.concatenate([
                from_ic50(
                    numpy.random.uniform(
                        self.hyperparameters[
                            'random_negative_affinity_min'],
                        self.hyperparameters[
                            'random_negative_affinity_max'],
                        int(num_random_negative.sum()))),
                y_values,
            ]),
        }
        if sample_weights is not None:
            sample_weights_with_random_negatives = numpy.concatenate([
                numpy.ones(int(num_random_negative.sum())),
                sample_weights])

        val_losses = []
        min_val_loss_iteration = None
        min_val_loss = None

        self.loss_history = collections.defaultdict(list)
        start = time.time()
        for i in range(self.hyperparameters['max_epochs']):
            random_negative_peptides_list = []
            for (length, count) in num_random_negative.items():
                random_negative_peptides_list.extend(
                    random_peptides(
                        count,
                        length=length,
                        distribution=aa_distribution))
            random_negative_peptides_encodable = (
                EncodableSequences.create(
                    random_negative_peptides_list))
            random_negative_peptides_encoding = (
                self.peptides_to_network_input(
                    random_negative_peptides_encodable))
            x_dict_with_random_negatives = {
                "peptide": numpy.concatenate([
                    random_negative_peptides_encoding,
                    peptide_encoding,
                ]) if len(random_negative_peptides_encoding) > 0
                else peptide_encoding
            }
            if pseudosequence_length:
                # TODO: add random pseudosequences for random negative peptides
                raise NotImplemented(
                    "Allele pseudosequences unsupported with random negatives")

            fit_history = self.network.fit(
                x_dict_with_random_negatives,
                y_dict_with_random_negatives,
                shuffle=True,
                verbose=verbose,
                epochs=1,
                validation_split=self.hyperparameters[
                    'validation_split'],
                sample_weight=sample_weights)

            for (key, value) in fit_history.history.items():
                self.loss_history[key].extend(value)

            logging.info(
                "Epoch %3d / %3d: loss=%g. Min val loss at epoch %s" % (
                    i,
                    self.hyperparameters['max_epochs'],
                    self.loss_history['loss'][-1],
                    min_val_loss_iteration))

            if self.hyperparameters['validation_split']:
                val_loss = self.loss_history['val_loss'][-1]
                val_losses.append(val_loss)

                if min_val_loss is None or val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    min_val_loss_iteration = i

                if self.hyperparameters['early_stopping']:
                    threshold = (
                        min_val_loss_iteration +
                        self.hyperparameters['patience'])
                    if i > threshold:
                        logging.info("Early stopping")
                        break
        self.fit_seconds = time.time() - start

    def predict(self, peptides, allele_pseudosequences=None):
        x_dict = {
            'peptide': self.peptides_to_network_input(peptides)
        }
        if allele_pseudosequences is not None:
            pseudosequences_input = self.pseudosequence_to_network_input(
                allele_pseudosequences)
            x_dict['pseudosequence'] = pseudosequences_input
        (predictions,) = numpy.array(self.network.predict(x_dict)).T
        return to_ic50(predictions)

    def compile(self):
        self.network.compile(
            **self.compile_hyperparameter_defaults.subselect(
                self.hyperparameters))

    @staticmethod
    def make_network(
            pseudosequence_length,
            kmer_size,
            use_embedding,
            embedding_input_dim,
            embedding_output_dim,
            pseudosequence_use_embedding,
            layer_sizes,
            dense_layer_l1_regularization,
            dense_layer_l2_regularization,
            activation,
            init,
            output_activation,
            dropout_probability,
            batch_normalization,
            embedding_init_method,
            locally_connected_layers):

        if use_embedding:
            peptide_input = Input(
                shape=(kmer_size,), dtype='int32', name='peptide')
            current_layer = Embedding(
                input_dim=embedding_input_dim,
                output_dim=embedding_output_dim,
                input_length=kmer_size,
                embeddings_initializer=embedding_init_method)(peptide_input)
        else:
            peptide_input = Input(
                shape=(kmer_size, 21), dtype='float32', name='peptide')
            current_layer = peptide_input

        inputs = [peptide_input]

        for locally_connected_params in locally_connected_layers:
            current_layer = keras.layers.LocallyConnected1D(
                **locally_connected_params)(current_layer)

        current_layer = Flatten()(current_layer)

        if batch_normalization:
            current_layer = BatchNormalization()(current_layer)

        if dropout_probability:
            current_layer = Dropout(dropout_probability)(current_layer)

        if pseudosequence_length:
            if pseudosequence_use_embedding:
                pseudosequence_input = Input(
                    shape=(pseudosequence_length,),
                    dtype='int32',
                    name='pseudosequence')
                pseudo_embedding_layer = Embedding(
                    input_dim=embedding_input_dim,
                    output_dim=embedding_output_dim,
                    input_length=pseudosequence_length,
                    embeddings_initializer=embedding_init_method)(
                    pseudosequence_input)
            else:
                pseudosequence_input = Input(
                    shape=(pseudosequence_length, 21),
                    dtype='float32', name='peptide')
                pseudo_embedding_layer = pseudosequence_input
            inputs.append(pseudosequence_input)
            pseudo_embedding_layer = Flatten()(pseudo_embedding_layer)

            current_layer = keras.layers.concatenate([
                current_layer, pseudo_embedding_layer
            ])
            
        for layer_size in layer_sizes:
            kernel_regularizer = None
            l1 = dense_layer_l1_regularization
            l2 = dense_layer_l2_regularization
            if l1 > 0 or l2 > 0:
                kernel_regularizer = keras.regularizers.l1_l2(l1, l2)

            current_layer = Dense(
                layer_size,
                activation=activation,
                kernel_regularizer=kernel_regularizer)(current_layer)

            if batch_normalization:
                current_layer = BatchNormalization()(current_layer)

            if dropout_probability > 0:
                current_layer = Dropout(dropout_probability)(current_layer)

        output = Dense(
            1,
            kernel_initializer=init,
            activation=output_activation,
            name="output")(current_layer)
        model = keras.models.Model(inputs=inputs, outputs=[output])
        return model
