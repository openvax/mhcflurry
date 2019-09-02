import time
import collections
import json
import weakref
import itertools
import os
import logging

import numpy
import pandas

from .hyperparameters import HyperparameterDefaults

from .encodable_sequences import EncodableSequences, EncodingError
from .regression_target import to_ic50, from_ic50
from .common import random_peptides, amino_acid_distribution
from .custom_loss import get_loss
from .data_dependent_weights_initialization import lsuv_init


DEFAULT_PREDICT_BATCH_SIZE = 4096
if os.environ.get("MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE"):
    DEFAULT_PREDICT_BATCH_SIZE = os.environ[
        "MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE"
    ]
    logging.info(
        "Configured default predict batch size: %d" % DEFAULT_PREDICT_BATCH_SIZE)


class Class1NeuralNetwork(object):
    """
    Low level class I predictor consisting of a single neural network.
    
    Both single allele and pan-allele prediction are supported, but pan-allele
    is in development and not yet well performing.
    
    Users will generally use Class1AffinityPredictor, which gives a higher-level
    interface and supports ensembles.
    """

    network_hyperparameter_defaults = HyperparameterDefaults(
        allele_amino_acid_encoding="BLOSUM62",
        allele_dense_layer_sizes=[],
        peptide_encoding={
            'vector_encoding_name': 'BLOSUM62',
            'alignment_method': 'pad_middle',
            'left_edge': 4,
            'right_edge': 4,
            'max_length': 15,
        },
        peptide_dense_layer_sizes=[],
        peptide_allele_merge_method="multiply",
        peptide_allele_merge_activation="",
        layer_sizes=[32],
        dense_layer_l1_regularization=0.001,
        dense_layer_l2_regularization=0.0,
        activation="tanh",
        init="glorot_uniform",
        output_activation="sigmoid",
        dropout_probability=0.0,
        batch_normalization=False,
        locally_connected_layers=[
            {
                "filters": 8,
                "activation": "tanh",
                "kernel_size": 3
            }
        ],
        num_outputs=1,
    )
    """
    Hyperparameters (and their default values) that affect the neural network
    architecture.
    """

    compile_hyperparameter_defaults = HyperparameterDefaults(
        loss="custom:mse_with_inequalities",
        optimizer="rmsprop",
        learning_rate=None,
    )
    """
    Loss and optimizer hyperparameters. Any values supported by keras may be
    used.
    """

    fit_hyperparameter_defaults = HyperparameterDefaults(
        max_epochs=500,
        validation_split=0.1,
        early_stopping=True,
        minibatch_size=128,
        data_dependent_initialization_method=None,
        random_negative_rate=0.0,
        random_negative_constant=25,
        random_negative_affinity_min=20000.0,
        random_negative_affinity_max=50000.0,
        random_negative_match_distribution=True,
        random_negative_distribution_smoothing=0.0,
        random_negative_output_indices=None)
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

    miscelaneous_hyperparameter_defaults = HyperparameterDefaults(
        train_data={},
    )
    """
    Miscelaneous hyperaparameters. These parameters are not used by this class
    but may be interpreted by other code.
    """

    hyperparameter_defaults = network_hyperparameter_defaults.extend(
        compile_hyperparameter_defaults).extend(
        fit_hyperparameter_defaults).extend(
        early_stopping_hyperparameter_defaults).extend(
        miscelaneous_hyperparameter_defaults
    )
    """
    Combined set of all supported hyperparameters and their default values.
    """

    # Hyperparameter renames.
    # These are updated from time to time as new versions are developed. It
    # provides a primitive way to allow new code to work with models trained
    # using older code.
    # None indicates the hyperparameter has been dropped.
    hyperparameter_renames = {
        "use_embedding": None,
        "pseudosequence_use_embedding": None,
        "monitor": None,
        "min_delta": None,
        "verbose": None,
        "mode": None,
        "take_best_epoch": None,
        'kmer_size': None,
        'peptide_amino_acid_encoding': None,
        'embedding_input_dim': None,
        'embedding_output_dim': None,
        'embedding_init_method': None,
        'left_edge': None,
        'right_edge': None,
    }

    @classmethod
    def apply_hyperparameter_renames(cls, hyperparameters):
        """
        Handle hyperparameter renames.

        Parameters
        ----------
        hyperparameters : dict

        Returns
        -------
        dict : updated hyperparameters

        """
        for (from_name, to_name) in cls.hyperparameter_renames.items():
            if from_name in hyperparameters:
                value = hyperparameters.pop(from_name)
                if to_name:
                    hyperparameters[to_name] = value
        return hyperparameters

    def __init__(self, **hyperparameters):
        self.hyperparameters = self.hyperparameter_defaults.with_defaults(
            self.apply_hyperparameter_renames(hyperparameters))

        self._network = None
        self.network_json = None
        self.network_weights = None
        self.network_weights_loader = None

        self.fit_info = []
        self.prediction_cache = weakref.WeakKeyDictionary()

    KERAS_MODELS_CACHE = {}
    """
    Process-wide keras model cache, a map from: architecture JSON string to
    (Keras model, existing network weights)
    """

    @classmethod
    def clear_model_cache(klass):
        """
        Clear the Keras model cache.
        """
        klass.KERAS_MODELS_CACHE.clear()

    @classmethod
    def borrow_cached_network(klass, network_json, network_weights):
        """
        Return a keras Model with the specified architecture and weights.
        As an optimization, when possible this will reuse architectures from a
        process-wide cache.

        The returned object is "borrowed" in the sense that its weights can
        change later after subsequent calls to this method from other objects.

        If you're using this from a parallel implementation you'll need to
        hold a lock while using the returned object.

        Parameters
        ----------
        network_json : string of JSON
        network_weights : list of numpy.array

        Returns
        -------
        keras.models.Model
        """
        assert network_weights is not None
        key = klass.keras_network_cache_key(network_json)
        if key not in klass.KERAS_MODELS_CACHE:
            # Cache miss.
            import keras.models
            network = keras.models.model_from_json(network_json)
            existing_weights = None
        else:
            # Cache hit.
            (network, existing_weights) = klass.KERAS_MODELS_CACHE[key]
        if existing_weights is not network_weights:
            network.set_weights(network_weights)
            klass.KERAS_MODELS_CACHE[key] = (network, network_weights)

        # As an added safety check we overwrite the fit method on the returned
        # model to throw an error if it is called.
        def throw(*args, **kwargs):
            raise NotImplementedError("Do not call fit on cached model.")

        network.fit = throw
        return network

    def network(self, borrow=False):
        """
        Return the keras model associated with this predictor.

        Parameters
        ----------
        borrow : bool
            Whether to return a cached model if possible. See
            borrow_cached_network for details

        Returns
        -------
        keras.models.Model
        """
        if self._network is None and self.network_json is not None:
            self.load_weights()
            if borrow:
                return self.borrow_cached_network(
                    self.network_json,
                    self.network_weights)
            else:
                import keras.models
                self._network = keras.models.model_from_json(self.network_json)
                if self.network_weights is not None:
                    self._network.set_weights(self.network_weights)
                self.network_json = None
                self.network_weights = None
        return self._network

    def update_network_description(self):
        if self._network is not None:
            self.network_json = self._network.to_json()
            self.network_weights = self._network.get_weights()

    @staticmethod
    def keras_network_cache_key(network_json):
        # As an optimization, we remove anything about regularization as these
        # do not affect predictions.
        def drop_properties(d):
            if 'kernel_regularizer' in d:
                del d['kernel_regularizer']
            return d

        description = json.loads(
            network_json,
            object_hook=drop_properties)
        return json.dumps(description)

    def get_config(self):
        """
        serialize to a dict all attributes except model weights
        
        Returns
        -------
        dict
        """
        self.update_network_description()
        result = dict(self.__dict__)
        result['_network'] = None
        result['network_weights'] = None
        result['network_weights_loader'] = None
        result['prediction_cache'] = None
        return result

    @classmethod
    def from_config(cls, config, weights=None, weights_loader=None):
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
        instance.__dict__.update(config)
        instance.network_weights = weights
        instance.network_weights_loader = weights_loader
        instance.prediction_cache = weakref.WeakKeyDictionary()
        return instance

    def load_weights(self):
        """
        Load weights by evaluating self.network_weights_loader, if needed.

        After calling this, self.network_weights_loader will be None and
        self.network_weights will be the weights list, if available.
        """
        if self.network_weights_loader:
            self.network_weights = self.network_weights_loader()
            self.network_weights_loader = None

    def get_weights(self):
        """
        Get the network weights
        
        Returns
        -------
        list of numpy.array giving weights for each layer or None if there is no
        network
        """
        self.update_network_description()
        self.load_weights()
        return self.network_weights

    def __getstate__(self):
        """
        serialize to a dict. Model weights are included. For pickle support.
        
        Returns
        -------
        dict

        """
        self.update_network_description()
        self.load_weights()
        result = dict(self.__dict__)
        result['_network'] = None
        result['prediction_cache'] = None
        return result

    def __setstate__(self, state):
        """
        Deserialize. For pickle support.
        """
        self.__dict__.update(state)
        self.prediction_cache = weakref.WeakKeyDictionary()

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

    @property
    def supported_peptide_lengths(self):
        """
        (minimum, maximum) lengths of peptides supported, inclusive.
        
        Returns
        -------
        (int, int) tuple

        """
        # We currently have an arbitrary hard floor of 5, even if the underlying
        # peptide encoding supports smaller lengths.
        #
        # We empirically find the supported peptide lengths based on the
        # lengths for which peptides_to_network_input throws ValueError.
        try:
            self.peptides_to_network_input([""])
        except EncodingError as e:
            return e.supported_peptide_lengths
        raise RuntimeError("peptides_to_network_input did not raise")

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
        """
        return (
            allele_encoding.indices,
            allele_encoding.allele_representations(
                self.hyperparameters['allele_amino_acid_encoding']))

    @staticmethod
    def data_dependent_weights_initialization(
            network,
            x_dict=None,
            method="lsuv",
            verbose=1):
        """
        Data dependent weights initialization.

        Parameters
        ----------
        method

        Returns
        -------

        """
        if verbose:
            print("Performing data-dependent init: ", method)
        if method == "lsuv":
            assert x_dict is not None, "Data required for LSUV init"
            lsuv_init(network, x_dict, verbose=verbose > 0)
        else:
            raise RuntimeError("Unsupported init method: ", method)

    def fit_generator(
            self,
            generator,
            validation_peptide_encoding,
            validation_affinities,
            validation_allele_encoding=None,
            validation_inequalities=None,
            validation_output_indices=None,
            steps_per_epoch=10,
            epochs=1000,
            min_epochs=0,
            patience=10,
            min_delta=0.0,
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0):
        """
        Fit using a generator. Does not support many of the features of fit(),
        such as random negative peptides.

        Parameters
        ----------
        generator : generator yielding (alleles, peptides, affinities) tuples
            where alleles and peptides are lists of strings, and affinities
            is list of floats.

        validation_peptide_encoding
        validation_affinities
        validation_allele_encoding
        validation_inequalities
        validation_output_indices
        steps_per_epoch
        epochs
        patience
        verbose

        Returns
        -------

        """
        import keras
        from keras import backend as K

        fit_info = collections.defaultdict(list)

        loss = get_loss(self.hyperparameters['loss'])

        (validation_allele_input, allele_representations) = (
            self.allele_encoding_to_network_input(validation_allele_encoding))

        if self.network() is None:
            self._network = self.make_network(
                allele_representations=allele_representations,
                **self.network_hyperparameter_defaults.subselect(
                    self.hyperparameters))
            if verbose > 0:
                self.network().summary()
        network = self.network()

        network.compile(
            loss=loss.loss, optimizer=self.hyperparameters['optimizer'])
        network._make_predict_function()
        self.set_allele_representations(allele_representations)

        if self.hyperparameters['learning_rate'] is not None:
            K.set_value(
                self.network().optimizer.lr,
                self.hyperparameters['learning_rate'])
        fit_info["learning_rate"] = float(
            K.get_value(self.network().optimizer.lr))

        validation_x_dict = {
            'peptide': self.peptides_to_network_input(
                validation_peptide_encoding),
            'allele': validation_allele_input,
        }
        encode_y_kwargs = {}
        if validation_inequalities is not None:
            encode_y_kwargs["inequalities"] = validation_inequalities
        if validation_output_indices is not None:
            encode_y_kwargs["output_indices"] = validation_output_indices

        output = loss.encode_y(
            from_ic50(validation_affinities), **encode_y_kwargs)

        validation_y_dict = {
            'output': output,
        }

        mutable_generator_state = {
            'yielded_values': 0  # total number of data points yielded
        }

        def wrapped_generator():
            for (alleles, peptides, affinities) in generator:
                (allele_encoding_input, _) = (
                    self.allele_encoding_to_network_input(alleles))
                x_dict = {
                    'peptide': self.peptides_to_network_input(peptides),
                    'allele': allele_encoding_input,
                }
                y_dict = {
                    'output': from_ic50(affinities)
                }
                yield (x_dict, y_dict)
                mutable_generator_state['yielded_values'] += len(affinities)

        start = time.time()

        iterator = wrapped_generator()

        # Initialization required if a data_dependent_initialization_method
        # is set and this is our first time fitting (i.e. fit_info is empty).
        data_dependent_init = self.hyperparameters[
            'data_dependent_initialization_method'
        ]
        if data_dependent_init and not self.fit_info:
            first_chunk = next(iterator)
            self.data_dependent_weights_initialization(
                network,
                first_chunk[0],  # x_dict
                method=data_dependent_init,
                verbose=verbose)
            iterator = itertools.chain([first_chunk], iterator)

        min_val_loss_iteration = None
        min_val_loss = None
        last_progress_print = 0
        epoch = 1
        while True:
            epoch_start_time = time.time()
            fit_history = network.fit_generator(
                iterator,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=epoch - 1,
                epochs=epoch,
                use_multiprocessing=False,
                workers=0,
                validation_data=(validation_x_dict, validation_y_dict),
                verbose=verbose,
            )
            epoch_time = time.time() - epoch_start_time
            for (key, value) in fit_history.history.items():
                fit_info[key].extend(value)
            val_loss = fit_info['val_loss'][-1]

            if min_val_loss is None or val_loss < min_val_loss - min_delta:
                min_val_loss = val_loss
                min_val_loss_iteration = epoch

            patience_epoch_threshold = min(
                epochs, max(min_val_loss_iteration + patience, min_epochs))

            progress_message = (
                "epoch %3d/%3d [%0.2f sec.]: loss=%g val_loss=%g. Min val "
                "loss %g at epoch %s. Cum. points: %d. Stop at epoch %d." % (
                    epoch,
                    epochs,
                    epoch_time,
                    fit_info['loss'][-1],
                    val_loss,
                    min_val_loss,
                    min_val_loss_iteration,
                    mutable_generator_state['yielded_values'],
                    patience_epoch_threshold,
                )).strip()

            # Print progress no more often than once every few seconds.
            if progress_print_interval is not None and (
                    time.time() - last_progress_print > progress_print_interval):
                print(progress_preamble, progress_message)
                last_progress_print = time.time()

            if progress_callback:
                progress_callback()

            if epoch >= patience_epoch_threshold:
                if progress_print_interval is not None:
                    print(progress_preamble, "STOPPING", progress_message)
                    break
            epoch += 1

        fit_info["time"] = time.time() - start
        fit_info["num_points"] = mutable_generator_state["yielded_values"]
        self.fit_info.append(dict(fit_info))

    def fit(
            self,
            peptides,
            affinities,
            allele_encoding=None,
            inequalities=None,
            output_indices=None,
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
        peptides : EncodableSequences or list of string
        
        affinities : list of float
            nM affinities. Must be same length of as peptides.
        
        allele_encoding : AlleleEncoding, optional
            If not specified, the model will be a single-allele predictor.

        inequalities : list of string, each element one of ">", "<", or "=".
            Inequalities to use for fitting. Same length as affinities.
            Each element must be one of ">", "<", or "=". For example, a ">"
            will train on y_pred > y_true for that element in the training set.
            Requires using a custom losses that support inequalities (e.g.
            mse_with_ineqalities).
            If None all inequalities are taken to be "=".
            
        sample_weights : list of float, optional
            If not specified, all samples (including random negatives added
            during training) will have equal weight. If specified, the random
            negatives will be assigned weight=1.0.

        shuffle_permutation : list of int, optional
            Permutation (integer list) of same length as peptides and affinities
            If None, then a random permutation will be generated.

        verbose : int
            Keras verbosity level

        progress_preamble : string
            Optional string of information to include in each progress update

        progress_print_interval : float
            How often (in seconds) to print progress update. Set to None to
            disable.
        """
        from keras import backend as K
        encodable_peptides = EncodableSequences.create(peptides)
        peptide_encoding = self.peptides_to_network_input(encodable_peptides)
        fit_info = collections.defaultdict(list)

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
            str(num_random_negative.to_dict())))

        aa_distribution = None
        if self.hyperparameters['random_negative_match_distribution']:
            aa_distribution = amino_acid_distribution(
                encodable_peptides.sequences,
                smoothing=self.hyperparameters[
                    'random_negative_distribution_smoothing'])
            logging.info(
                "Using amino acid distribution for random negative:\n%s" % (
                str(aa_distribution.to_dict())))

        y_values = from_ic50(numpy.array(affinities, copy=False))
        assert numpy.isnan(y_values).sum() == 0, y_values
        if inequalities is not None:
            # Reverse inequalities because from_ic50() flips the direction
            # (i.e. lower affinity results in higher y values).
            adjusted_inequalities = pandas.Series(inequalities).map({
                "=": "=",
                ">": "<",
                "<": ">",
            }).values
        else:
            adjusted_inequalities = numpy.tile("=", len(y_values))
        if len(adjusted_inequalities) != len(y_values):
            raise ValueError("Inequalities and y_values must have same length")

        x_dict_without_random_negatives = {
            'peptide': peptide_encoding,
        }
        allele_representations = None
        if allele_encoding is not None:
            (allele_encoding_input, allele_representations) = (
                self.allele_encoding_to_network_input(allele_encoding))
            x_dict_without_random_negatives['allele'] = allele_encoding_input

        # Shuffle y_values and the contents of x_dict_without_random_negatives
        # This ensures different data is used for the test set for early stopping
        # when multiple models are trained.
        if shuffle_permutation is None:
            shuffle_permutation = numpy.random.permutation(len(y_values))
        y_values = y_values[shuffle_permutation]
        assert numpy.isnan(y_values).sum() == 0, y_values
        peptide_encoding = peptide_encoding[shuffle_permutation]
        adjusted_inequalities = adjusted_inequalities[shuffle_permutation]
        for key in x_dict_without_random_negatives:
            x_dict_without_random_negatives[key] = (
                x_dict_without_random_negatives[key][shuffle_permutation])
        if sample_weights is not None:
            sample_weights = numpy.array(sample_weights, copy=False)[
                shuffle_permutation
            ]
        if output_indices is not None:
            output_indices = numpy.array(output_indices, copy=False)[
                shuffle_permutation
            ]

        loss = get_loss(self.hyperparameters['loss'])

        if not loss.supports_inequalities and (
                any(inequality != "=" for inequality in adjusted_inequalities)):
            raise ValueError("Loss %s does not support inequalities" % loss)

        if (not loss.supports_multiple_outputs and output_indices is not None
                and (output_indices != 0).any()):
            raise ValueError("Loss %s does not support multiple outputs" % loss)

        if self.hyperparameters['num_outputs'] != 1:
            if output_indices is None:
                raise ValueError(
                    "Must supply output_indices for multi-output predictor")

        if self.network() is None:
            self._network = self.make_network(
                allele_representations=allele_representations,
                **self.network_hyperparameter_defaults.subselect(
                    self.hyperparameters))
            if verbose > 0:
                self.network().summary()

        if allele_representations is not None:
            self.set_allele_representations(allele_representations)

        self.network().compile(
            loss=loss.loss, optimizer=self.hyperparameters['optimizer'])

        if self.hyperparameters['learning_rate'] is not None:
            K.set_value(
                self.network().optimizer.lr,
                self.hyperparameters['learning_rate'])
        fit_info["learning_rate"] = float(
            K.get_value(self.network().optimizer.lr))

        if loss.supports_inequalities:
            # Do not sample negative affinities: just use an inequality.
            random_negative_ic50 = self.hyperparameters['random_negative_affinity_min']
            random_negative_target = from_ic50(random_negative_ic50)

            y_dict_with_random_negatives = {
                "output": numpy.concatenate([
                    numpy.tile(
                        random_negative_target, int(num_random_negative.sum())),
                    y_values,
                ]),
            }
            # Note: we are using "<" here not ">" because the inequalities are
            # now in target-space (0-1) not affinity-space.
            adjusted_inequalities_with_random_negatives = (
                ["<"] * int(num_random_negative.sum()) +
                list(adjusted_inequalities))
        else:
            # Randomly sample random negative affinities
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
            adjusted_inequalities_with_random_negatives = None
        assert numpy.isnan(y_dict_with_random_negatives['output']).sum() == 0, (
            y_dict_with_random_negatives)
        if sample_weights is not None:
            sample_weights_with_random_negatives = numpy.concatenate([
                numpy.ones(int(num_random_negative.sum())),
                sample_weights])
        else:
            sample_weights_with_random_negatives = None

        if output_indices is not None:
            random_negative_output_indices = (
                self.hyperparameters['random_negative_output_indices']
                if self.hyperparameters['random_negative_output_indices']
                else list(range(0, self.hyperparameters['num_outputs'])))
            output_indices_with_random_negatives = numpy.concatenate([
                pandas.Series(random_negative_output_indices, dtype=int).sample(
                    n=int(num_random_negative.sum()), replace=True).values,
                output_indices
            ])
        else:
            output_indices_with_random_negatives = None

        encode_y_kwargs = {}
        if adjusted_inequalities_with_random_negatives is not None:
            encode_y_kwargs["inequalities"] = (
                adjusted_inequalities_with_random_negatives)
        if output_indices_with_random_negatives is not None:
            encode_y_kwargs["output_indices"] = (
                output_indices_with_random_negatives)

        y_dict_with_random_negatives['output'] = loss.encode_y(
            y_dict_with_random_negatives['output'],
            **encode_y_kwargs)

        min_val_loss_iteration = None
        min_val_loss = None

        # Initialization required if a data_dependent_initialization_method
        # is set and this is our first time fitting (i.e. fit_info is empty).
        needs_initialization = self.hyperparameters[
            'data_dependent_initialization_method'
        ] is not None and not self.fit_info

        start = time.time()
        last_progress_print = None
        x_dict_with_random_negatives = {}
        for i in range(self.hyperparameters['max_epochs']):
            random_negative_peptides_list = []
            for (length, count) in num_random_negative.iteritems():
                random_negative_peptides_list.extend(
                    random_peptides(
                        count,
                        length=length,
                        distribution=aa_distribution))
            random_negative_peptides = EncodableSequences.create(
                random_negative_peptides_list)
            random_negative_peptides_encoding = (
                self.peptides_to_network_input(random_negative_peptides))

            if not x_dict_with_random_negatives:
                if len(random_negative_peptides) > 0:
                    x_dict_with_random_negatives["peptide"] = numpy.concatenate([
                        random_negative_peptides_encoding,
                        peptide_encoding,
                    ])
                    if 'allele' in x_dict_without_random_negatives:
                        x_dict_with_random_negatives['allele'] = numpy.concatenate([
                            x_dict_without_random_negatives['allele'][
                                numpy.random.choice(
                                    x_dict_without_random_negatives[
                                        'allele'].shape[0],
                                    size=len(random_negative_peptides_list))],
                            x_dict_without_random_negatives['allele']
                        ])
                else:
                    x_dict_with_random_negatives = (
                        x_dict_without_random_negatives)
            else:
                # Update x_dict_with_random_negatives in place.
                # This is more memory efficient than recreating it as above.
                if len(random_negative_peptides) > 0:
                    x_dict_with_random_negatives["peptide"][:len(random_negative_peptides)] = (
                        random_negative_peptides_encoding
                    )
                    if 'allele' in x_dict_with_random_negatives:
                        x_dict_with_random_negatives['allele'][:len(random_negative_peptides)] = (
                            x_dict_with_random_negatives['allele'][
                                len(random_negative_peptides) + numpy.random.choice(
                                    x_dict_with_random_negatives['allele'].shape[0] -
                                    len(random_negative_peptides),
                                    size=len(random_negative_peptides))
                            ]
                        )

            if needs_initialization:
                self.data_dependent_weights_initialization(
                    self.network(),
                    x_dict_with_random_negatives,
                    method=self.hyperparameters[
                        'data_dependent_initialization_method'],
                    verbose=verbose)
                needs_initialization = False

            epoch_start = time.time()
            fit_history = self.network().fit(
                x_dict_with_random_negatives,
                y_dict_with_random_negatives,
                shuffle=True,
                batch_size=self.hyperparameters['minibatch_size'],
                verbose=verbose,
                epochs=i + 1,
                initial_epoch=i,
                validation_split=self.hyperparameters['validation_split'],
                sample_weight=sample_weights_with_random_negatives)
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

    def predict(
            self,
            peptides,
            allele_encoding=None,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE,
            output_index=0):
        """
        Predict affinities.

        If peptides are specified as EncodableSequences, then the predictions
        will be cached for this predictor as long as the EncodableSequences object
        remains in memory. The cache is keyed in the object identity of the
        EncodableSequences, not the sequences themselves.

        Parameters
        ----------
        peptides : EncodableSequences or list of string
        
        allele_encoding : AlleleEncoding, optional
            Only required when this model is a pan-allele model

        batch_size : int
            batch_size passed to Keras

        Returns
        -------
        numpy.array of nM affinity predictions 
        """
        assert self.prediction_cache is not None
        use_cache = (
            allele_encoding is None and
            isinstance(peptides, EncodableSequences))
        if use_cache and peptides in self.prediction_cache:
            return self.prediction_cache[peptides].copy()

        x_dict = {
            'peptide': self.peptides_to_network_input(peptides)
        }

        if allele_encoding is not None:
            (allele_encoding_input, allele_representations) = (
                self.allele_encoding_to_network_input(allele_encoding))
            x_dict['allele'] = allele_encoding_input
            self.set_allele_representations(allele_representations)
            network = self.network()
        else:
            network = self.network(borrow=True)
        raw_predictions = network.predict(x_dict, batch_size=batch_size)
        predictions = numpy.array(raw_predictions, dtype = "float64")
        if output_index is not None:
            predictions = predictions[:,output_index]
        result = to_ic50(predictions)
        if use_cache:
            self.prediction_cache[peptides] = result
        return result

    def make_network(
            self,
            peptide_encoding,
            allele_amino_acid_encoding,
            allele_dense_layer_sizes,
            peptide_dense_layer_sizes,
            peptide_allele_merge_method,
            peptide_allele_merge_activation,
            layer_sizes,
            dense_layer_l1_regularization,
            dense_layer_l2_regularization,
            activation,
            init,
            output_activation,
            dropout_probability,
            batch_normalization,
            locally_connected_layers,
            num_outputs=1,
            allele_representations=None):
        """
        Helper function to make a keras network for class1 affinity prediction.
        """

        # We import keras here to avoid tensorflow debug output, etc. unless we
        # are actually about to use Keras.

        from keras.layers import Input
        import keras.layers
        from keras.layers.core import Dense, Flatten, Dropout
        from keras.layers.embeddings import Embedding
        from keras.layers.normalization import BatchNormalization

        peptide_encoding_shape = self.peptides_to_network_input([]).shape[1:]
        peptide_input = Input(
            shape=peptide_encoding_shape,
            dtype='float32',
            name='peptide')
        current_layer = peptide_input

        inputs = [peptide_input]

        kernel_regularizer = None
        l1 = dense_layer_l1_regularization
        l2 = dense_layer_l2_regularization
        if l1 > 0 or l2 > 0:
            kernel_regularizer = keras.regularizers.l1_l2(l1, l2)

        for (i, locally_connected_params) in enumerate(locally_connected_layers):
            current_layer = keras.layers.LocallyConnected1D(
                name="lc_%d" % i,
                **locally_connected_params)(current_layer)

        current_layer = Flatten(name="flattened_0")(current_layer)

        for (i, layer_size) in enumerate(peptide_dense_layer_sizes):
            current_layer = Dense(
                layer_size,
                name="peptide_dense_%d" % i,
                kernel_regularizer=kernel_regularizer,
                activation=activation)(current_layer)

        if batch_normalization:
            current_layer = BatchNormalization(name="batch_norm_early")(
                current_layer)

        if allele_representations is not None:
            allele_input = Input(
                shape=(1,),
                dtype='float32',
                name='allele')
            inputs.append(allele_input)

            allele_layer = Embedding(
                name="allele_representation",
                input_dim=allele_representations.shape[0],
                output_dim=numpy.product(allele_representations.shape[1:], dtype=int),
                input_length=1,
                trainable=False)(allele_input)

            for (i, layer_size) in enumerate(allele_dense_layer_sizes):
                allele_layer = Dense(
                    layer_size,
                    name="allele_dense_%d" % i,
                    kernel_regularizer=kernel_regularizer,
                    activation=activation)(allele_layer)

            allele_layer = Flatten(name="allele_flat")(allele_layer)

            if peptide_allele_merge_method == 'concatenate':
                current_layer = keras.layers.concatenate([
                    current_layer, allele_layer
                ], name="allele_peptide_merged")
            elif peptide_allele_merge_method == 'multiply':
                current_layer = keras.layers.multiply([
                    current_layer, allele_layer
                ], name="allele_peptide_merged")
            else:
                raise ValueError(
                    "Unsupported peptide_allele_encoding_merge_method: %s"
                    % peptide_allele_merge_method)

            if peptide_allele_merge_activation:
                current_layer = keras.layers.Activation(
                    peptide_allele_merge_activation,
                    name="alelle_peptide_merged_%s" %
                         peptide_allele_merge_activation)(current_layer)
            
        for (i, layer_size) in enumerate(layer_sizes):
            current_layer = Dense(
                layer_size,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                name="dense_%d" % i)(current_layer)

            if batch_normalization:
                current_layer = BatchNormalization(
                    name="batch_norm_%d" % i)(current_layer)

            if dropout_probability > 0:
                current_layer = Dropout(
                    rate=1 - dropout_probability,
                    name="dropout_%d" % i)(current_layer)

        output = Dense(
            num_outputs,
            kernel_initializer=init,
            activation=output_activation,
            name="output")(current_layer)
        model = keras.models.Model(
            inputs=inputs,
            outputs=[output],
            name="predictor")

        return model

    def set_allele_representations(self, allele_representations):
        """

        Parameters
        ----------
        model
        allele_representations

        """
        from keras.models import model_from_json
        reshaped = allele_representations.reshape(
            (allele_representations.shape[0], -1))
        original_model = self.network()
        layer = original_model.get_layer("allele_representation")
        (existing_weights,) = layer.get_weights()

        # Only changes to the number of supported alleles (not the length of
        # the allele sequences) are allowed.
        assert existing_weights.shape[1:] == reshaped.shape[1:]

        if existing_weights.shape != reshaped.shape:
            # Network surgery required. Make a new network with this layer's
            # dimensions changed. Kind of a hack.
            layer.input_dim = reshaped.shape[0]
            new_model = model_from_json(original_model.to_json())

            # copy weights for other layers over
            for layer in new_model.layers:
                if layer.name != "allele_representation":
                    layer.set_weights(
                        original_model.get_layer(name=layer.name).get_weights())

            self._network = new_model
            self.update_network_description()

            layer = new_model.get_layer("allele_representation")
            (existing_weights,) = layer.get_weights()

            # Disable the old model to catch bugs.
            def throw(*args, **kwargs):
                raise RuntimeError("Using a disabled model!")
            original_model.predict = \
                original_model.fit = \
                original_model.fit_generator = throw

        assert existing_weights.shape == reshaped.shape, (
            existing_weights.shape, reshaped.shape)
        layer.set_weights([reshaped])
