import time
import os
import tempfile
import logging

import numpy
import pandas

import keras.models
import keras.layers.pooling
import keras.regularizers
from keras.layers import Input
import keras.layers.merge
from keras.layers.core import Dense, Flatten, Dropout, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import keras.backend as K
import theano.tensor

from mhcflurry.hyperparameters import HyperparameterDefaults

from ..encodable_sequences import EncodableSequences
from ..regression_target import to_ic50, from_ic50
from ..common import random_peptides, amino_acid_distribution



class Class1BindingPredictor(object):
    network_hyperparameter_defaults = HyperparameterDefaults(
        kmer_size=15,
        use_embedding=True,
        embedding_input_dim=21,
        embedding_output_dim=8,
        pseudosequence_use_embedding=True,
        pseudosequence_generate_weights=False,
        extra_data_length=None,
        extra_data_layer_sizes=(),
        multiple_output_strategy=None,
        multiple_output_activity_regularizer=1.0,
        layer_sizes=[100, 32],
        dense_layer_l1_regularization=0.0,
        dense_layer_l2_regularization=0.0,
        activation="tanh",
        init="glorot_uniform",
        output_activation="sigmoid",
        dropout_probability=0.0,
        batch_normalization=True,
        embedding_init_method="glorot_uniform",
        locally_connected=None,
        concatenate_locally_connected_with_raw_embedding=False,
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
        input_encoding_hyperparameter_defaults).extend(
        fit_hyperparameter_defaults).extend(
        early_stopping_hyperparameter_defaults)

    def __init__(self, **hyperparameters):
        self.hyperparameters = self.hyperparameter_defaults.with_defaults(
            hyperparameters)
        self.network = None
        self.fit_history = None
        self.fit_seconds = None
        self.output_names = None

    def __getstate__(self):
        result = dict(self.__dict__)
        del result['network']
        result['fit_history'] = None
        result['network_json'] = self.network.to_json()
        result['network_weights'] = self.get_weights()
        return result

    def __setstate__(self, state):
        network_json = state.pop('network_json')
        network_weights = state.pop('network_weights')
        self.__dict__.update(state)
        self.network = keras.models.model_from_json(network_json)
        self.set_weights(network_weights)

    def get_weights(self):
        """
        Returns weights, which can be passed to set_weights later.
        """
        return [x.copy() for x in self.network.get_weights()]

    def set_weights(self, weights):
        """
        Reset the model weights.
        """
        self.network.set_weights(weights)

    def peptides_to_network_input(self, peptides):
        encoder = EncodableSequences.create(peptides)
        if self.hyperparameters['use_embedding']:
            encoded = encoder.fixed_length_categorical_encoding(
                max_length=self.hyperparameters['kmer_size'],
                **self.input_encoding_hyperparameter_defaults.subselect(
                    self.hyperparameters))
        else:
            encoded = encoder.fixed_length_one_hot_encoding(
                max_length=self.hyperparameters['kmer_size'],
                **self.input_encoding_hyperparameter_defaults.subselect(
                    self.hyperparameters))
        assert len(encoded) == len(peptides)
        return encoded

    def pseudosequence_to_network_input(self, pseudosequences):
        encoder = EncodableSequences.create(pseudosequences)
        if self.hyperparameters['pseudosequence_use_embedding']:
            encoded = encoder.categorical_encoding()
        else:
            encoded = encoder.one_hot_encoding()
        assert len(encoded) == len(pseudosequences)
        return encoded

    def fit(
            self,
            peptides,
            affinities,
            output_assignments,
            allele_pseudosequences=None,
            sample_weights=None,
            verbose=1):
        self.output_names = sorted(set(output_assignments))

        encodable_peptides = EncodableSequences.create(peptides)
        peptide_encoding = self.peptides_to_network_input(encodable_peptides)
        peptide_to_encoding = dict(
            zip(encodable_peptides.sequences, peptide_encoding))

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
        print("Random negative counts per length: %s" % (
            str(num_random_negative)))

        aa_distribution = None
        if self.hyperparameters['random_negative_match_distribution']:
            aa_distribution = amino_acid_distribution(
                encodable_peptides.sequences,
                smoothing=self.hyperparameters[
                    'random_negative_distribution_smoothing'])
            print("Using amino acid distribution for random negative: %s" % (
                str(aa_distribution)))

        y_values = from_ic50(affinities)
        assert numpy.isnan(y_values).sum() == 0, (
            numpy.isnan(y_values).sum())

        if self.hyperparameters['multiple_output_strategy'] is not None:
            network_output_names = self.output_names
            y_df = pandas.DataFrame({
                'y': y_values,
                'output_assignment': output_assignments,
            }).pivot(values="y", columns="output_assignment")
            y_df["peptide"] = encodable_peptides.sequences
            y_df.groupby("peptide").mean()
            network_output_names = self.output_names

            y_dict = dict((c, y_df[c].values) for c in y_df.columns)
            x = numpy.stack(
                y_df.peptide.map(peptide_to_encoding).values)
        else:
            network_output_names = ["output"]
            y_dict = {'output': y_values}
            x = peptide_encoding

        try:
            callbacks = []
            if self.hyperparameters['take_best_epoch']:
                weights_file_fd = tempfile.NamedTemporaryFile(
                    prefix="mhcflurry-model-checkpoint-",
                    suffix=".hdf5",
                    delete=False)
                weights_file = weights_file_fd.name
                print("Checkpointing to: %s" % weights_file)
                weights_file_fd.close()

                checkpointer = keras.callbacks.ModelCheckpoint(
                    weights_file,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False)
                callbacks.append(checkpointer)
            else:
                weights_file = None

            if self.hyperparameters['early_stopping']:
                assert self.hyperparameters['validation_split'] > 0
                callback = EarlyStopping(
                    **self.early_stopping_hyperparameter_defaults.subselect(
                        self.hyperparameters))
                callbacks.append(callback)

            x_dict = {
                'peptide': x,
            }
            pseudosequence_length = None
            if allele_pseudosequences is not None:
                pseudosequences_input = self.pseudosequence_to_network_input(
                    allele_pseudosequences)
                pseudosequence_length = len(pseudosequences_input[0])
                x_dict['pseudosequence'] = pseudosequences_input

            if self.network is None:
                self.network = self.make_network(
                    output_names=network_output_names,
                    pseudosequence_length=pseudosequence_length,
                    **self.network_hyperparameter_defaults.subselect(
                        self.hyperparameters))

            start = time.time()
            if num_random_negative.sum() == 0:
                self.fit_history = self.network.fit(
                    x_dict,
                    y_dict,
                    shuffle=True,
                    verbose=verbose,
                    epochs=self.hyperparameters['max_epochs'],
                    validation_split=self.hyperparameters['validation_split'],
                    sample_weight=sample_weights,
                    callbacks=callbacks)
            else:
                assert len(y_dict) == 1
                y_dict['output'] = numpy.concatenate([
                    from_ic50(
                        numpy.random.uniform(
                            self.hyperparameters[
                                'random_negative_affinity_min'],
                            self.hyperparameters[
                                'random_negative_affinity_max'],
                            int(num_random_negative.sum()))),
                    y_dict['output'],
                ])
                if sample_weights is not None:
                    sample_weights = numpy.concatenate([
                        numpy.ones(int(num_random_negative.sum())),
                        sample_weights])
                val_losses = []
                min_val_loss_iteration = None
                min_val_loss = None

                for i in range(self.hyperparameters['max_epochs']):
                    # TODO: handle pseudosequence here
                    assert len(x_dict) == 1
                    random_negative_peptides_list = []
                    for (length, count) in num_random_negative.items():
                        random_negative_peptides_list.extend(
                            random_peptides(
                                count,
                                length=length,
                                distribution=aa_distribution))
                        #peptide_lengths.extend([length] * count)
                    #peptide_lengths.extend([
                    #    len(s) for s in encodable_peptides.sequences
                    #])
                    random_negative_peptides_encodable = (
                        EncodableSequences.create(
                            random_negative_peptides_list))
                    random_negative_peptides_encoding = (
                        self.peptides_to_network_input(
                            random_negative_peptides_encodable))
                    x_dict["peptide"] = numpy.concatenate([
                        random_negative_peptides_encoding, x
                    ])
                    print("Epoch %3d / %3d. Min val loss at epoch %s" % (
                        i,
                        self.hyperparameters['max_epochs'],
                        min_val_loss_iteration))
                    self.fit_history = self.network.fit(
                        x_dict,
                        y_dict,
                        shuffle=True,
                        verbose=verbose,
                        epochs=1,
                        validation_split=self.hyperparameters[
                            'validation_split'],
                        sample_weight=sample_weights,
                        callbacks=callbacks)

                    if self.hyperparameters['validation_split']:
                        val_loss = self.fit_history.history['val_loss'][-1]
                        val_losses.append(val_loss)

                        if min_val_loss is None or val_loss <= min_val_loss:
                            min_val_loss = val_loss
                            min_val_loss_iteration = i

                        if self.hyperparameters['early_stopping']:
                            threshold = (
                                min_val_loss_iteration +
                                self.hyperparameters['patience'])
                            if i > threshold:
                                print("Early stopping")
                                break
            if weights_file is not None:
                self.network.load_weights(weights_file)
            self.fit_seconds = time.time() - start

        finally:
            if weights_file is not None:
                os.unlink(weights_file)

    def predict(self, peptides, allele_pseudosequences=None):
        x_dict = {
            'peptide': self.peptides_to_network_input(peptides)
        }
        if allele_pseudosequences is not None:
            pseudosequences_input = self.pseudosequence_to_network_input(
                allele_pseudosequences)
            x_dict['pseudosequence'] = pseudosequences_input
        predictions_raw = numpy.array(self.network.predict(x_dict))
        if predictions_raw.ndim == 3:
            predictions_raw = numpy.squeeze(predictions_raw, axis=2).T

        assert predictions_raw.shape == (
            len(peptides),
            len(self.network.output_layers)), predictions_raw.shape

        result = dict(
            (k.name, to_ic50(v))
            for (k, v)
            in zip(self.network.output_layers, predictions_raw.T))

        if set(result) != set(self.output_names):
            # Simulate multiple outputs
            assert set(result) == set(["output"]), set(result)
            result = dict((k, result["output"]) for k in self.output_names)
        return result

    @staticmethod
    def make_network(
            output_names,
            pseudosequence_length,
            kmer_size,
            use_embedding,
            embedding_input_dim,
            embedding_output_dim,
            pseudosequence_use_embedding,
            pseudosequence_generate_weights,
            extra_data_length,
            extra_data_layer_sizes,
            multiple_output_strategy,
            multiple_output_activity_regularizer,
            layer_sizes,
            dense_layer_l1_regularization,
            dense_layer_l2_regularization,
            activation,
            init,
            output_activation,
            dropout_probability,
            batch_normalization,
            embedding_init_method,
            locally_connected,
            concatenate_locally_connected_with_raw_embedding,
            optimizer):

        if multiple_output_strategy is None:
            assert len(output_names) == 1
        else:
            assert multiple_output_strategy in ("simple", "bottleneck")

        if use_embedding:
            peptide_input = Input(
                shape=(kmer_size,), dtype='int32', name='peptide')
            raw_embedding_layer = Embedding(
                input_dim=embedding_input_dim,
                output_dim=embedding_output_dim,
                input_length=kmer_size,
                embeddings_initializer=embedding_init_method)(peptide_input)
        else:
            peptide_input = Input(
                shape=(kmer_size, 21), dtype='float32', name='peptide')
            raw_embedding_layer = peptide_input

        inputs = [peptide_input]

        embedding_layer = raw_embedding_layer

        if locally_connected is not None:
            for locally_connected_params in locally_connected:
                embedding_layer = keras.layers.LocallyConnected1D(
                    **locally_connected_params)(embedding_layer)
            if concatenate_locally_connected_with_raw_embedding:
                embedding_layer = keras.layers.concatenate([
                    Flatten()(raw_embedding_layer),
                    Flatten()(embedding_layer),
                ])
            else:
                embedding_layer = Flatten()(embedding_layer)
        else:
            embedding_layer = Flatten()(embedding_layer)
        if batch_normalization:
            embedding_layer = BatchNormalization()(embedding_layer)
        if dropout_probability:
            embedding_layer = Dropout(dropout_probability)(embedding_layer)

        if extra_data_length:
            extra_info_input = Input(
                shape=(extra_data_length,), dtype='float32', name='extra')
            inputs.append(extra_info_input)

            for layer_size in extra_data_layer_sizes:
                extra_info_input = Dense(layer_size, activation=activation)(
                    extra_info_input)
                if batch_normalization:
                    extra_info_input = BatchNormalization()(
                        extra_info_input)
                if dropout_probability > 0:
                    extra_info_input = Dropout(dropout_probability)(
                        extra_info_input)
            x = keras.layers.concatenate([embedding_layer, extra_info_input])
        else:
            x = embedding_layer

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
            
            if pseudosequence_generate_weights:
                pseudo_dense = Dense(
                    32, activation="tanh")(pseudo_embedding_layer)
                
                num_filters = 7
                kernel_size = 3
                output_length = 11
                
                pseudo_lc1_kernel_shape = (output_length * kernel_size * embedding_output_dim, num_filters)
                pseudo_lc2_kernel_shape = (output_length * kernel_size * num_filters, num_filters)
                
                
                pseudo_lc1_kernel = Dense(numpy.prod(pseudo_lc1_kernel_shape), activation="tanh")(pseudo_dense)
                pseudo_lc1_bias = Dense(num_filters * 11, activation="tanh")(pseudo_dense)
                pseudo_lc2_kernel = Dense(numpy.prod(pseudo_lc2_kernel_shape), activation="tanh")(pseudo_dense)
                pseudo_lc2_bias = Dense(num_filters * 11, activation="tanh")(pseudo_dense)
                pseudo_hidden_kernel = Dense(num_filters * 11 * 32, activation="tanh")(pseudo_dense)
                pseudo_hidden_bias = Dense(32, activation="tanh")(pseudo_dense)
                
                pseudo_lc1_kernel = Reshape(pseudo_lc1_kernel_shape)(pseudo_lc1_kernel)
                #pseudo_lc1_bias = Reshape((11, num_filters))(pseudo_lc1_bias)
                pseudo_lc2_kernel = Reshape(pseudo_lc2_kernel_shape)(pseudo_lc2_kernel)
                #pseudo_lc2_bias = Reshape((11, num_filters))(pseudo_lc2_bias)
                pseudo_hidden_kernel = Reshape((num_filters * 11, 32))(pseudo_hidden_kernel)
                
                def make_peptide_tiles(input_tensor):
                    components = []
                    for start in range(11):
                        components.append(K.flatten(
                            input_tensor[:, start : start + kernel_size]))
                    return K.concatenate(components, axis=0)
                
                print("Raw embedding layer", raw_embedding_layer)
                peptide_tiles = keras.layers.Lambda(
                    make_peptide_tiles,
                    output_shape=(11 * kernel_size * embedding_output_dim,))(
                    raw_embedding_layer)
                
                def merger(inputs):
                    # TODO: A*b + c
                    print("inside merger", inputs)
                    kernel = inputs[0]
                    bias = inputs[1]
                    data = inputs[2]
                    print("kernel", kernel._keras_shape)
                    print("data", data._keras_shape)
                    _, kernel_size, filters = kernel._keras_shape
                    print(kernel_size, filters)
                    
                    #dots = []
                    #for f in range(filters):
                    #    dots.append(K.dot(data, kernel[:,:,f]))))
                    
                    #result = K.reshape(K.concatenate(dots), (-1, )
                    #print("after dot", result._keras_shape)
                    #assert result._keras_shape[1:] == (), result._keras_shape
                    #result += K.reshape(bias, (1, output_length, filters))
                    return result
                
                lc1_output = keras.layers.merge(
                    [pseudo_lc1_kernel, pseudo_lc1_bias, peptide_tiles],
                    mode=merger,
                    output_shape=(11 * num_filters,)
                )
                print("merged", lc1_output._keras_shape)
                lc1_output = keras.layers.Activation("relu")(lc1_output)
                
                #lc1_output_tiles = keras.layers.Lambda(
                #    make_peptide_tiles,
                #    output_shape=(11 * kernel_size * num_filters,))(lc1_output)
                
                #lc2_output = keras.layers.merge(
                #    [pseudo_lc1_kernel, pseudo_lc1_bias, lc1_output_tiles],
                #    mode=merger,
                #    output_shape=(11, num_filters)
                #)                
                #lc2_output = keras.layers.Activation("relu")(lc2_output)
                #lc2_output = Flatten()(lc2_output)
                lc2_output = lc1_output
                
                #def dense_merger(inputs):
                #    print("inside dense merger", inputs)
                #    print([i._keras_shape for i in inputs])
                #    return inputs[0]
                
                x = keras.layers.merge(
                    [pseudo_hidden_kernel, pseudo_hidden_bias, lc2_output],
                    mode=merger,
                    output_shape=(32, 1))
                x = keras.layers.Activation("relu")(x)
                print("x shape", type(x), x._keras_shape)
            else:
                x = keras.layers.concatenate([
                    x, pseudo_embedding_layer
                ])
            
        for layer_size in layer_sizes:
            kernel_regularizer = None
            l1 = dense_layer_l1_regularization
            l2 = dense_layer_l2_regularization
            if l1 > 0 or l2 > 0:
                kernel_regularizer = keras.regularizers.l1_l2(l1, l2)

            x = Dense(
                layer_size,
                activation=activation,
                kernel_regularizer=kernel_regularizer)(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if dropout_probability > 0:
                x = Dropout(dropout_probability)(x)

        outputs = []
        if multiple_output_strategy == "bottleneck":
            print("x shape2", type(x), x._keras_shape)
            bottleneck = Dense(
                1,
                kernel_initializer=init,
                activation="linear")(x)

            peptide_average = keras.layers.pooling.AveragePooling1D(
                pool_size=1)(raw_embedding_layer)
            peptide_average = Flatten()(peptide_average)

            peptide_and_bottleneck = keras.layers.concatenate([
                bottleneck, peptide_average
            ])

            for output_name in output_names:
                nudge = Dense(
                    8,
                    kernel_initializer=init,
                    activation=activation,
                )(peptide_and_bottleneck)

                nudge = Dense(
                    1,
                    kernel_initializer="zeros",
                    activity_regularizer=keras.regularizers.l2(
                        multiple_output_activity_regularizer))(nudge)

                output = keras.layers.add(
                    [bottleneck, nudge])
                output = keras.layers.Activation(
                    output_activation, name=output_name)(output)
                outputs.append(output)
        else:
            for output_name in output_names:
                output = Dense(
                    1,
                    kernel_initializer=init,
                    activation=output_activation,
                    name=output_name)(x)
                outputs.append(output)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss="mse" if len(output_names) == 1 else mse_loss_supporting_nans,
            optimizer=optimizer)
        return model


def mse_loss_supporting_nans(y_true, y_pred):
    squared = K.square(y_pred - y_true)
    loss = K.sum(
        K.switch(theano.tensor.isnan(y_true), 0.0, squared),
        axis=-1)
    return loss
