import collections
import hashlib
import json
import logging
import time
import warnings
from os.path import join, exists, abspath
from os import mkdir
from socket import gethostname
from getpass import getuser
from functools import partial

import mhcnames
import numpy
import pandas
from numpy.testing import assert_equal
from six import string_types

from .class1_neural_network import Class1NeuralNetwork
from .common import random_peptides
from .downloads import get_default_class1_models_dir
from .encodable_sequences import EncodableSequences
from .percent_rank_transform import PercentRankTransform
from .regression_target import to_ic50
from .version import __version__
from .ensemble_centrality import CENTRALITY_MEASURES
from .allele_encoding import AlleleEncoding


# Default function for combining predictions across models in an ensemble.
# See ensemble_centrality.py for other options.
DEFAULT_CENTRALITY_MEASURE = "mean"


class Class1AffinityPredictor(object):
    """
    High-level interface for peptide/MHC I binding affinity prediction.

    This class manages low-level `Class1NeuralNetwork` instances, each of which
    wraps a single Keras network. The purpose of `Class1AffinityPredictor` is to
    implement ensembles, handling of multiple alleles, and predictor loading and
    saving.
    """
    def __init__(
            self,
            allele_to_allele_specific_models=None,
            class1_pan_allele_models=None,
            allele_to_fixed_length_sequence=None,
            manifest_df=None,
            allele_to_percent_rank_transform=None,
            metadata_dataframes=None):
        """
        Parameters
        ----------
        allele_to_allele_specific_models : dict of string -> list of `Class1NeuralNetwork`
            Ensemble of single-allele models to use for each allele. 
        
        class1_pan_allele_models : list of `Class1NeuralNetwork`
            Ensemble of pan-allele models.
        
        allele_to_fixed_length_sequence : dict of string -> string
            Required only if class1_pan_allele_models is specified.
        
        manifest_df : `pandas.DataFrame`, optional
            Must have columns: model_name, allele, config_json, model.
            Only required if you want to update an existing serialization of a
            Class1AffinityPredictor. Otherwise this dataframe will be generated
            automatically based on the supplied models.

        allele_to_percent_rank_transform : dict of string -> `PercentRankTransform`, optional
            `PercentRankTransform` instances to use for each allele

        metadata_dataframes : dict of string -> pandas.DataFrame, optional
            Optional additional dataframes to write to the models dir when
            save() is called. Useful for tracking provenance.
        """

        if allele_to_allele_specific_models is None:
            allele_to_allele_specific_models = {}
        if class1_pan_allele_models is None:
            class1_pan_allele_models = []

        if class1_pan_allele_models:
            assert allele_to_fixed_length_sequence, "Allele sequences required"

        self.allele_to_allele_specific_models = allele_to_allele_specific_models
        self.class1_pan_allele_models = class1_pan_allele_models
        self.allele_to_fixed_length_sequence = allele_to_fixed_length_sequence
        self._manifest_df = manifest_df

        if not allele_to_percent_rank_transform:
            allele_to_percent_rank_transform = {}
        self.allele_to_percent_rank_transform = allele_to_percent_rank_transform
        self.metadata_dataframes = metadata_dataframes
        self._cache = {}

    @property
    def manifest_df(self):
        if self._manifest_df is None:
            # Make a manifest based on
            #  - self.class1_pan_allele_models
            #  - self.allele_to_allele_specific_models
            rows = []
            for (i, model) in enumerate(self.class1_pan_allele_models):
                rows.append((
                    self.model_name("pan-class1", i),
                    "pan-class1",
                    json.dumps(model.get_config()),
                    model
                ))
            for (allele,
                 models) in self.allele_to_allele_specific_models.items():
                for (i, model) in enumerate(models):
                    rows.append((
                        self.model_name(allele, i),
                        allele,
                        json.dumps(model.get_config()),
                        model
                    ))
            self._manifest_df = pandas.DataFrame(
                rows,
                columns=["model_name", "allele", "config_json", "model"])
        return self._manifest_df

    def clear_cache(self):
        """
        Clear values cached based on the neural networks in this predictor.

        Users should call this after mutating any of the following:
            - class1_pan_allele_models
            - allele_to_allele_specific_models
            - allele_to_fixed_length_sequence

        Methods that mutate these instance variables will call this method on
        their own if needed.
        """
        self._cache.clear()

    @property
    def neural_networks(self):
        """
        List of the neural networks in the ensemble.

        Returns
        -------
        list of `Class1NeuralNetwork`
        """
        result = []
        for models in self.allele_to_allele_specific_models.values():
            result.extend(models)
        result.extend(self.class1_pan_allele_models)
        return result

    @classmethod
    def merge(cls, predictors):
        """
        Merge the ensembles of two or more `Class1AffinityPredictor` instances.

        Note: the resulting merged predictor will NOT have calibrated percentile
        ranks. Call `calibrate_percentile_ranks` on it if these are needed.

        Parameters
        ----------
        predictors : sequence of `Class1AffinityPredictor`

        Returns
        -------
        `Class1AffinityPredictor` instance

        """
        assert len(predictors) > 0
        if len(predictors) == 1:
            return predictors[0]

        allele_to_allele_specific_models = collections.defaultdict(list)
        class1_pan_allele_models = []
        allele_to_fixed_length_sequence = predictors[0].allele_to_fixed_length_sequence

        for predictor in predictors:
            for (allele, networks) in (
                    predictor.allele_to_allele_specific_models.items()):
                allele_to_allele_specific_models[allele].extend(networks)
            class1_pan_allele_models.extend(
                predictor.class1_pan_allele_models)

        return Class1AffinityPredictor(
            allele_to_allele_specific_models=allele_to_allele_specific_models,
            class1_pan_allele_models=class1_pan_allele_models,
            allele_to_fixed_length_sequence=allele_to_fixed_length_sequence
        )

    def merge_in_place(self, others):
        """
        Add the models present other predictors into the current predictor.

        Parameters
        ----------
        others : list of Class1AffinityPredictor
            Other predictors to merge into the current predictor.

        Returns
        -------
        list of string : names of newly added models
        """

        new_model_names = []
        for predictor in others:
            for model in predictor.class1_pan_allele_models:
                model_name = self.model_name(
                    "pan-class1",
                    len(self.class1_pan_allele_models))
                self.class1_pan_allele_models.append(model)
                row = pandas.Series(collections.OrderedDict([
                    ("model_name", model_name),
                    ("allele", "pan-class1"),
                    ("config_json", json.dumps(model.get_config())),
                    ("model", model),
                ])).to_frame().T
                self._manifest_df = pandas.concat(
                    [self.manifest_df, row], ignore_index=True)
                new_model_names.append(model_name)

            for allele in predictor.allele_to_allele_specific_models:
                if allele not in self.allele_to_allele_specific_models:
                    self.allele_to_allele_specific_models[allele] = []
                current_models = self.allele_to_allele_specific_models[allele]
                for model in predictor.allele_to_allele_specific_models[allele]:
                    model_name = self.model_name(allele, len(current_models))
                    row = pandas.Series(collections.OrderedDict([
                        ("model_name", model_name),
                        ("allele", allele),
                        ("config_json", json.dumps(model.get_config())),
                        ("model", model),
                    ])).to_frame().T
                    self._manifest_df = pandas.concat(
                        [self.manifest_df, row], ignore_index=True)
                    current_models.append(model)
                    new_model_names.append(model_name)

        self.clear_cache()
        return new_model_names

    @property
    def supported_alleles(self):
        """
        Alleles for which predictions can be made.
        
        Returns
        -------
        list of string
        """
        if 'supported_alleles' not in self._cache:
            result = set(self.allele_to_allele_specific_models)
            if self.allele_to_fixed_length_sequence:
                result = result.union(self.allele_to_fixed_length_sequence)
            self._cache["supported_alleles"] = sorted(result)
        return self._cache["supported_alleles"]

    @property
    def supported_peptide_lengths(self):
        """
        (minimum, maximum) lengths of peptides supported by *all models*,
        inclusive.

        Returns
        -------
        (int, int) tuple

        """
        if 'supported_peptide_lengths' not in self._cache:
            length_ranges = set(
                network.supported_peptide_lengths
                for network in self.neural_networks)
            result = (
                max(lower for (lower, upper) in length_ranges),
                min(upper for (lower, upper) in length_ranges))
            self._cache["supported_peptide_lengths"] = result
        return self._cache["supported_peptide_lengths"]

    def save(self, models_dir, model_names_to_write=None, write_metadata=True):
        """
        Serialize the predictor to a directory on disk. If the directory does
        not exist it will be created.
        
        The serialization format consists of a file called "manifest.csv" with
        the configurations of each Class1NeuralNetwork, along with per-network
        files giving the model weights. If there are pan-allele predictors in
        the ensemble, the allele sequences are also stored in the
        directory. There is also a small file "index.txt" with basic metadata:
        when the models were trained, by whom, on what host.
        
        Parameters
        ----------
        models_dir : string
            Path to directory
            
        model_names_to_write : list of string, optional
            Only write the weights for the specified models. Useful for
            incremental updates during training.

        write_metadata : boolean, optional
            Whether to write optional metadata
        """
        num_models = len(self.class1_pan_allele_models) + sum(
            len(v) for v in self.allele_to_allele_specific_models.values())
        assert len(self.manifest_df) == num_models, (
            "Manifest seems out of sync with models: %d vs %d entries" % (
                len(self.manifest_df), num_models))

        if model_names_to_write is None:
            # Write all models
            model_names_to_write = self.manifest_df.model_name.values

        if not exists(models_dir):
            mkdir(models_dir)

        sub_manifest_df = self.manifest_df.ix[
            self.manifest_df.model_name.isin(model_names_to_write)
        ]

        for (_, row) in sub_manifest_df.iterrows():
            weights_path = self.weights_path(models_dir, row.model_name)
            Class1AffinityPredictor.save_weights(
                row.model.get_weights(), weights_path)
            logging.info("Wrote: %s" % weights_path)

        write_manifest_df = self.manifest_df[[
            c for c in self.manifest_df.columns if c != "model"
        ]]
        manifest_path = join(models_dir, "manifest.csv")
        write_manifest_df.to_csv(manifest_path, index=False)
        logging.info("Wrote: %s" % manifest_path)

        if write_metadata:
            # Write "info.txt"
            info_path = join(models_dir, "info.txt")
            rows = [
                ("trained on", time.asctime()),
                ("package   ", "mhcflurry %s" % __version__),
                ("hostname  ", gethostname()),
                ("user      ", getuser()),
            ]
            pandas.DataFrame(rows).to_csv(
                info_path, sep="\t", header=False, index=False)

            if self.metadata_dataframes:
                for (name, df) in self.metadata_dataframes.items():
                    metadata_df_path = join(models_dir, "%s.csv.bz2" % name)
                    df.to_csv(metadata_df_path, index=False, compression="bz2")

        if self.allele_to_fixed_length_sequence is not None:
            allele_to_sequence_df = pandas.DataFrame(
                list(self.allele_to_fixed_length_sequence.items()),
                columns=['allele', 'sequence']
            )
            allele_to_sequence_df.to_csv(
                join(models_dir, "allele_sequences.csv"), index=False)
            logging.info("Wrote: %s" % join(models_dir, "allele_sequences.csv"))

        if self.allele_to_percent_rank_transform:
            percent_ranks_df = None
            for (allele, transform) in self.allele_to_percent_rank_transform.items():
                series = transform.to_series()
                if percent_ranks_df is None:
                    percent_ranks_df = pandas.DataFrame(index=series.index)
                assert_equal(series.index.values, percent_ranks_df.index.values)
                percent_ranks_df[allele] = series
            percent_ranks_path = join(models_dir, "percent_ranks.csv")
            percent_ranks_df.to_csv(
                percent_ranks_path,
                index=True,
                index_label="bin")
            logging.info("Wrote: %s" % percent_ranks_path)

    @staticmethod
    def load(models_dir=None, max_models=None):
        """
        Deserialize a predictor from a directory on disk.
        
        Parameters
        ----------
        models_dir : string
            Path to directory
            
        max_models : int, optional
            Maximum number of `Class1NeuralNetwork` instances to load

        Returns
        -------
        `Class1AffinityPredictor` instance
        """
        if models_dir is None:
            models_dir = get_default_class1_models_dir()

        manifest_path = join(models_dir, "manifest.csv")
        manifest_df = pandas.read_csv(manifest_path, nrows=max_models)

        allele_to_allele_specific_models = collections.defaultdict(list)
        class1_pan_allele_models = []
        all_models = []
        for (_, row) in manifest_df.iterrows():
            weights_filename = Class1AffinityPredictor.weights_path(
                models_dir, row.model_name)
            config = json.loads(row.config_json)

            # We will lazy-load weights when the network is used.
            model = Class1NeuralNetwork.from_config(
                config,
                weights_loader=partial(
                    Class1AffinityPredictor.load_weights,
                    abspath(weights_filename)))
            if row.allele == "pan-class1":
                class1_pan_allele_models.append(model)
            else:
                allele_to_allele_specific_models[row.allele].append(model)
            all_models.append(model)

        manifest_df["model"] = all_models

        allele_to_fixed_length_sequence = None
        if exists(join(models_dir, "allele_sequences.csv")):
            allele_to_fixed_length_sequence = pandas.read_csv(
                join(models_dir, "allele_sequences.csv"),
                index_col="allele").to_dict()

        allele_to_percent_rank_transform = {}
        percent_ranks_path = join(models_dir, "percent_ranks.csv")
        if exists(percent_ranks_path):
            percent_ranks_df = pandas.read_csv(percent_ranks_path, index_col=0)
            for allele in percent_ranks_df.columns:
                allele_to_percent_rank_transform[allele] = (
                    PercentRankTransform.from_series(percent_ranks_df[allele]))

        logging.info(
            "Loaded %d class1 pan allele predictors, %d allele sequences, "
            "%d percent rank distributions, and %d allele specific models: %s" % (
                len(class1_pan_allele_models),
                len(allele_to_fixed_length_sequence) if allele_to_fixed_length_sequence else 0,
                len(allele_to_percent_rank_transform),
                sum(len(v) for v in allele_to_allele_specific_models.values()),
                ", ".join(
                    "%s (%d)" % (allele, len(v))
                    for (allele, v)
                    in sorted(allele_to_allele_specific_models.items()))))

        result = Class1AffinityPredictor(
            allele_to_allele_specific_models=allele_to_allele_specific_models,
            class1_pan_allele_models=class1_pan_allele_models,
            allele_to_fixed_length_sequence=allele_to_fixed_length_sequence,
            manifest_df=manifest_df,
            allele_to_percent_rank_transform=allele_to_percent_rank_transform,
        )
        return result

    @staticmethod
    def model_name(allele, num):
        """
        Generate a model name
        
        Parameters
        ----------
        allele : string
        num : int

        Returns
        -------
        string

        """
        random_string = hashlib.sha1(
            str(time.time()).encode()).hexdigest()[:16]
        return "%s-%d-%s" % (allele.upper(), num, random_string)

    @staticmethod
    def weights_path(models_dir, model_name):
        """
        Generate the path to the weights file for a model
        
        Parameters
        ----------
        models_dir : string
        model_name : string

        Returns
        -------
        string
        """
        return join(models_dir, "weights_%s.npz" % model_name)

    def fit_allele_specific_predictors(
            self,
            n_models,
            architecture_hyperparameters_list,
            allele,
            peptides,
            affinities,
            inequalities=None,
            train_rounds=None,
            models_dir_for_save=None,
            verbose=0,
            progress_preamble="",
            progress_print_interval=5.0):
        """
        Fit one or more allele specific predictors for a single allele using one
        or more neural network architectures.
        
        The new predictors are saved in the Class1AffinityPredictor instance
        and will be used on subsequent calls to `predict`.
        
        Parameters
        ----------
        n_models : int
            Number of neural networks to fit
        
        architecture_hyperparameters_list : list of dict
            List of hyperparameter sets.
               
        allele : string
        
        peptides : `EncodableSequences` or list of string
        
        affinities : list of float
            nM affinities

        inequalities : list of string, each element one of ">", "<", or "="
            See Class1NeuralNetwork.fit for details.

        train_rounds : sequence of int
            Each training point i will be used on training rounds r for which
            train_rounds[i] > r, r >= 0.
        
        models_dir_for_save : string, optional
            If specified, the Class1AffinityPredictor is (incrementally) written
            to the given models dir after each neural network is fit.
        
        verbose : int
            Keras verbosity

        progress_preamble : string
            Optional string of information to include in each progress update

        progress_print_interval : float
            How often (in seconds) to print progress. Set to None to disable.

        Returns
        -------
        list of `Class1NeuralNetwork`
        """

        allele = mhcnames.normalize_allele_name(allele)
        if allele not in self.allele_to_allele_specific_models:
            self.allele_to_allele_specific_models[allele] = []

        encodable_peptides = EncodableSequences.create(peptides)
        peptides_affinities_inequalities_per_round = [
            (encodable_peptides, affinities, inequalities)
        ]

        if train_rounds is not None:
            for round in sorted(set(train_rounds)):
                round_mask = train_rounds > round
                if round_mask.any():
                    sub_encodable_peptides = EncodableSequences.create(
                        encodable_peptides.sequences[round_mask])
                    peptides_affinities_inequalities_per_round.append((
                        sub_encodable_peptides,
                        affinities[round_mask],
                        None if inequalities is None else inequalities[round_mask]))
        n_rounds = len(peptides_affinities_inequalities_per_round)

        n_architectures = len(architecture_hyperparameters_list)

        # Adjust progress info to indicate number of models and
        # architectures.
        pieces = []
        if n_models > 1:
            pieces.append("Model {model_num:2d} / {n_models:2d}")
        if n_architectures > 1:
            pieces.append(
                "Architecture {architecture_num:2d} / {n_architectures:2d}")
        if len(peptides_affinities_inequalities_per_round) > 1:
            pieces.append("Round {round:2d} / {n_rounds:2d}")
        pieces.append("{n_peptides:4d} peptides")
        progress_preamble_template = "[ %s ] {user_progress_preamble}" % (
            ", ".join(pieces))

        models = []
        for model_num in range(n_models):
            for (architecture_num, architecture_hyperparameters) in enumerate(
                    architecture_hyperparameters_list):
                model = Class1NeuralNetwork(**architecture_hyperparameters)
                for round_num in range(n_rounds):
                    (round_peptides, round_affinities, round_inequalities) = (
                        peptides_affinities_inequalities_per_round[round_num]
                    )
                    model.fit(
                        round_peptides,
                        round_affinities,
                        inequalities=round_inequalities,
                        verbose=verbose,
                        progress_preamble=progress_preamble_template.format(
                            n_peptides=len(round_peptides),
                            round=round_num,
                            n_rounds=n_rounds,
                            user_progress_preamble=progress_preamble,
                            model_num=model_num + 1,
                            n_models=n_models,
                            architecture_num=architecture_num + 1,
                            n_architectures=n_architectures),
                        progress_print_interval=progress_print_interval)

                model_name = self.model_name(allele, model_num)
                row = pandas.Series(collections.OrderedDict([
                    ("model_name", model_name),
                    ("allele", allele),
                    ("config_json", json.dumps(model.get_config())),
                    ("model", model),
                ])).to_frame().T
                self._manifest_df = pandas.concat(
                    [self.manifest_df, row], ignore_index=True)
                self.allele_to_allele_specific_models[allele].append(model)
                if models_dir_for_save:
                    self.save(
                        models_dir_for_save, model_names_to_write=[model_name])
                models.append(model)

        self.clear_cache()
        return models

    def fit_class1_pan_allele_models(
            self,
            n_models,
            architecture_hyperparameters,
            alleles,
            peptides,
            affinities,
            inequalities,
            models_dir_for_save=None,
            verbose=1,
            progress_preamble="",
            progress_print_interval=5.0):
        """
        Fit one or more pan-allele predictors using a single neural network
        architecture.
        
        The new predictors are saved in the Class1AffinityPredictor instance
        and will be used on subsequent calls to `predict`.
        
        Parameters
        ----------
        n_models : int
            Number of neural networks to fit
            
        architecture_hyperparameters : dict
        
        alleles : list of string
            Allele names (not sequences) corresponding to each peptide
        
        peptides : `EncodableSequences` or list of string
        
        affinities : list of float
            nM affinities

        inequalities : list of string, each element one of ">", "<", or "="
            See Class1NeuralNetwork.fit for details.
        
        models_dir_for_save : string, optional
            If specified, the Class1AffinityPredictor is (incrementally) written
            to the given models dir after each neural network is fit.
        
        verbose : int
            Keras verbosity

        progress_preamble : string
            Optional string of information to include in each progress update

        progress_print_interval : float
            How often (in seconds) to print progress. Set to None to disable.

        Returns
        -------
        list of `Class1NeuralNetwork`
        """

        alleles = pandas.Series(alleles).map(mhcnames.normalize_allele_name)
        allele_encoding = AlleleEncoding(
            alleles,
            allele_to_fixed_length_sequence=self.allele_to_fixed_length_sequence)

        encodable_peptides = EncodableSequences.create(peptides)
        models = []
        for i in range(n_models):
            logging.info("Training model %d / %d" % (i + 1, n_models))
            model = Class1NeuralNetwork(**architecture_hyperparameters)
            model.fit(
                encodable_peptides,
                affinities,
                inequalities=inequalities,
                allele_encoding=allele_encoding,
                verbose=verbose,
                progress_preamble=progress_preamble,
                progress_print_interval=progress_print_interval)

            model_name = self.model_name("pan-class1", i)
            self.class1_pan_allele_models.append(model)
            row = pandas.Series(collections.OrderedDict([
                ("model_name", model_name),
                ("allele", "pan-class1"),
                ("config_json", json.dumps(model.get_config())),
                ("model", model),
            ])).to_frame().T
            self._manifest_df = pandas.concat(
                [self.manifest_df, row], ignore_index=True)
            if models_dir_for_save:
                self.save(
                    models_dir_for_save, model_names_to_write=[model_name])
            models.append(model)

        self.clear_cache()
        return models

    def percentile_ranks(self, affinities, allele=None, alleles=None, throw=True):
        """
        Return percentile ranks for the given ic50 affinities and alleles.

        The 'allele' and 'alleles' argument are as in the `predict` method.
        Specify one of these.

        Parameters
        ----------
        affinities : sequence of float
            nM affinities
        allele : string
        alleles : sequence of string
        throw : boolean
            If True, a ValueError will be raised in the case of unsupported
            alleles. If False, a warning will be logged and NaN will be returned
            for those percentile ranks.

        Returns
        -------
        numpy.array of float
        """
        if allele is not None:
            try:
                transform = self.allele_to_percent_rank_transform[allele]
                return transform.transform(affinities)
            except KeyError:
                msg = "Allele %s has no percentile rank information" % allele
                if throw:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)
                    # Return NaNs
                    return numpy.ones(len(affinities)) * numpy.nan

        if alleles is None:
            raise ValueError("Specify allele or alleles")

        df = pandas.DataFrame({"affinity": affinities})
        df["allele"] = alleles
        df["result"] = numpy.nan
        for (allele, sub_df) in df.groupby("allele"):
            df.loc[sub_df.index, "result"] = self.percentile_ranks(
                sub_df.affinity, allele=allele, throw=throw)
        return df.result.values

    def predict(
            self,
            peptides,
            alleles=None,
            allele=None,
            throw=True,
            centrality_measure=DEFAULT_CENTRALITY_MEASURE):
        """
        Predict nM binding affinities.
        
        If multiple predictors are available for an allele, the predictions are
        the geometric means of the individual model predictions.
        
        One of 'allele' or 'alleles' must be specified. If 'allele' is specified
        all predictions will be for the given allele. If 'alleles' is specified
        it must be the same length as 'peptides' and give the allele
        corresponding to each peptide.
        
        Parameters
        ----------
        peptides : `EncodableSequences` or list of string
        alleles : list of string
        allele : string
        throw : boolean
            If True, a ValueError will be raised in the case of unsupported
            alleles or peptide lengths. If False, a warning will be logged and
            the predictions for the unsupported alleles or peptides will be NaN.
        centrality_measure : string or callable
            Measure of central tendency to use to combine predictions in the
            ensemble. Options include: mean, median, robust_mean.

        Returns
        -------
        numpy.array of predictions
        """
        df = self.predict_to_dataframe(
            peptides=peptides,
            alleles=alleles,
            allele=allele,
            throw=throw,
            include_percentile_ranks=False,
            include_confidence_intervals=False,
            centrality_measure=centrality_measure,
        )
        return df.prediction.values

    def predict_to_dataframe(
            self,
            peptides,
            alleles=None,
            allele=None,
            throw=True,
            include_individual_model_predictions=False,
            include_percentile_ranks=True,
            include_confidence_intervals=True,
            centrality_measure=DEFAULT_CENTRALITY_MEASURE):
        """
        Predict nM binding affinities. Gives more detailed output than `predict`
        method, including 5-95% prediction intervals.
        
        If multiple predictors are available for an allele, the predictions are
        the geometric means of the individual model predictions.
        
        One of 'allele' or 'alleles' must be specified. If 'allele' is specified
        all predictions will be for the given allele. If 'alleles' is specified
        it must be the same length as 'peptides' and give the allele
        corresponding to each peptide. 
        
        Parameters
        ----------
        peptides : `EncodableSequences` or list of string
        alleles : list of string
        allele : string
        throw : boolean
            If True, a ValueError will be raised in the case of unsupported
            alleles or peptide lengths. If False, a warning will be logged and
            the predictions for the unsupported alleles or peptides will be NaN.
        include_individual_model_predictions : boolean
            If True, the predictions of each individual model are included as
            columns in the result dataframe.
        include_percentile_ranks : boolean, default True
            If True, a "prediction_percentile" column will be included giving the
            percentile ranks. If no percentile rank information is available,
            this will be ignored with a warning.
        centrality_measure : string or callable
            Measure of central tendency to use to combine predictions in the
            ensemble. Options include: mean, median, robust_mean.

        Returns
        -------
        `pandas.DataFrame` of predictions
        """
        if isinstance(peptides, string_types):
            raise TypeError("peptides must be a list or array, not a string")
        if isinstance(alleles, string_types):
            raise TypeError("alleles must be a list or array, not a string")
        if allele is None and alleles is None:
            raise ValueError("Must specify 'allele' or 'alleles'.")

        peptides = EncodableSequences.create(peptides)
        df = pandas.DataFrame({
            'peptide': peptides.sequences
        }, copy=False)

        if allele is not None:
            if alleles is not None:
                raise ValueError("Specify exactly one of allele or alleles")
            df["allele"] = allele
            normalized_allele = mhcnames.normalize_allele_name(allele)
            df["normalized_allele"] = normalized_allele
            unique_alleles = [normalized_allele]
        else:
            df["allele"] = numpy.array(alleles)
            df["normalized_allele"] = df.allele.map(
                mhcnames.normalize_allele_name)
            unique_alleles = df.normalized_allele.unique()

        if len(df) == 0:
            # No predictions.
            logging.warning("Predicting for 0 peptides.")
            empty_result = pandas.DataFrame(
                columns=[
                    'peptide',
                    'allele',
                    'prediction',
                    'prediction_low',
                    'prediction_high'
                ])
            return empty_result

        (min_peptide_length, max_peptide_length) = (
            self.supported_peptide_lengths)

        if (peptides.min_length < min_peptide_length or
                peptides.max_length > max_peptide_length):
            # Only compute this if needed
            all_peptide_lengths_supported = False
            sequence_length = df.peptide.str.len()
            df["supported_peptide_length"] = (
                (sequence_length >= min_peptide_length) &
                (sequence_length <= max_peptide_length))
            if (~df.supported_peptide_length).any():
                msg = (
                    "%d peptides have lengths outside of supported range [%d, %d]: "
                    "%s" % (
                        (~df.supported_peptide_length).sum(),
                        min_peptide_length,
                        max_peptide_length,
                        str(df.ix[~df.supported_peptide_length].peptide.unique())))
                logging.warning(msg)
                if throw:
                    raise ValueError(msg)
        else:
            # Handle common case efficiently.
            df["supported_peptide_length"] = True
            all_peptide_lengths_supported = True

        num_pan_models = len(self.class1_pan_allele_models)
        max_single_allele_models = max(
            len(self.allele_to_allele_specific_models.get(allele, []))
            for allele in unique_alleles
        )
        predictions_array = numpy.zeros(
            shape=(df.shape[0], num_pan_models + max_single_allele_models),
            dtype="float64")
        predictions_array[:] = numpy.nan

        if self.class1_pan_allele_models:
            unsupported_alleles = [
                allele for allele in
                df.normalized_allele.unique()
                if allele not in self.allele_to_fixed_length_sequence
            ]
            if unsupported_alleles:
                msg = (
                    "No sequences for allele(s): %s.\n"
                    "Supported alleles: %s" % (
                        " ".join(unsupported_alleles),
                        " ".join(sorted(self.allele_to_fixed_length_sequence))))
                logging.warning(msg)
                if throw:
                    raise ValueError(msg)
            mask = df.supported_peptide_length
            if mask.sum() > 0:
                masked_allele_encoding = AlleleEncoding(
                    df.loc[mask].normalized_allele,
                    allele_to_fixed_length_sequence=self.allele_to_fixed_length_sequence)
                masked_peptides = peptides.sequences[mask]
                for (i, model) in enumerate(self.class1_pan_allele_models):
                    predictions_array[mask, i] = model.predict(
                        masked_peptides,
                        allele_encoding=masked_allele_encoding)

        if self.allele_to_allele_specific_models:
            unsupported_alleles = [
                allele for allele in unique_alleles
                if not self.allele_to_allele_specific_models.get(allele)
            ]
            if unsupported_alleles:
                msg = (
                    "No single-allele models for allele(s): %s.\n"
                    "Supported alleles are: %s" % (
                        " ".join(unsupported_alleles),
                        " ".join(sorted(self.allele_to_allele_specific_models))))
                logging.warning(msg)
                if throw:
                    raise ValueError(msg)

            for allele in unique_alleles:
                models = self.allele_to_allele_specific_models.get(allele, [])
                if len(unique_alleles) == 1 and all_peptide_lengths_supported:
                    mask = None
                else:
                    mask = (
                        (df.normalized_allele == allele) &
                        df.supported_peptide_length).values
                if mask is None or mask.all():
                    # Common case optimization
                    for (i, model) in enumerate(models):
                        predictions_array[:, num_pan_models + i] = (
                            model.predict(peptides))
                elif mask.sum() > 0:
                    peptides_for_allele = EncodableSequences.create(
                        df.ix[mask].peptide.values)
                    for (i, model) in enumerate(models):
                        predictions_array[
                            mask,
                            num_pan_models + i,
                        ] = model.predict(peptides_for_allele)

        if callable(centrality_measure):
            centrality_function = centrality_measure
        else:
            centrality_function = CENTRALITY_MEASURES[centrality_measure]

        logs = numpy.log(predictions_array)
        log_centers = centrality_function(logs)
        df["prediction"] = numpy.exp(log_centers)

        if include_confidence_intervals:
            df["prediction_low"] = numpy.exp(numpy.nanpercentile(logs, 5.0, axis=1))
            df["prediction_high"] = numpy.exp(numpy.nanpercentile(logs, 95.0, axis=1))

        if include_individual_model_predictions:
            for i in range(num_pan_models):
                df["model_pan_%d" % i] = predictions_array[:, i]

            for i in range(max_single_allele_models):
                df["model_single_%d" % i] = predictions_array[
                    :, num_pan_models + i
                ]

        if include_percentile_ranks:
            if self.allele_to_percent_rank_transform:
                df["prediction_percentile"] = self.percentile_ranks(
                    df.prediction,
                    alleles=df.normalized_allele.values,
                    throw=throw)
            else:
                warnings.warn("No percentile rank information available.")

        del df["supported_peptide_length"]
        del df["normalized_allele"]
        return df

    @staticmethod
    def save_weights(weights_list, filename):
        """
        Save the model weights to the given filename using numpy's ".npz"
        format.
    
        Parameters
        ----------
        weights_list : list of array
        
        filename : string
            Should end in ".npz".
    
        """
        numpy.savez(
            filename,
            **dict((("array_%d" % i), w) for (i, w) in enumerate(weights_list)))

    @staticmethod
    def load_weights(filename):
        """
        Restore model weights from the given filename, which should have been
        created with `save_weights`.
    
        Parameters
        ----------
        filename : string
            Should end in ".npz".

        Returns
        ----------
        list of array
        """
        loaded = numpy.load(filename)
        weights = [
            loaded["array_%d" % i]
            for i in range(len(loaded.keys()))
        ]
        loaded.close()
        return weights

    def calibrate_percentile_ranks(
            self,
            peptides=None,
            num_peptides_per_length=int(1e5),
            alleles=None,
            bins=None):
        """
        Compute the cumulative distribution of ic50 values for a set of alleles
        over a large universe of random peptides, to enable computing quantiles in
        this distribution later.

        Parameters
        ----------
        peptides : sequence of string or EncodableSequences, optional
            Peptides to use
        num_peptides_per_length : int, optional
            If peptides argument is not specified, then num_peptides_per_length
            peptides are randomly sampled from a uniform distribution for each
            supported length
        alleles : sequence of string, optional
            Alleles to perform calibration for. If not specified all supported
            alleles will be calibrated.
        bins : object
            Anything that can be passed to numpy.histogram's "bins" argument
            can be used here, i.e. either an integer or a sequence giving bin
            edges. This is in ic50 space.

        Returns
        ----------
        EncodableSequences : peptides used for calibration
        """
        if bins is None:
            bins = to_ic50(numpy.linspace(1, 0, 1000))

        if alleles is None:
            alleles = self.supported_alleles

        if peptides is None:
            peptides = []
            lengths = range(
                self.supported_peptide_lengths[0],
                self.supported_peptide_lengths[1] + 1)
            for length in lengths:
                peptides.extend(
                    random_peptides(num_peptides_per_length, length))

        encoded_peptides = EncodableSequences.create(peptides)

        for (i, allele) in enumerate(alleles):
            predictions = self.predict(encoded_peptides, allele=allele)
            transform = PercentRankTransform()
            transform.fit(predictions, bins=bins)
            self.allele_to_percent_rank_transform[allele] = transform

        return encoded_peptides

    def filter_networks(self, predicate):
        """
        Return a new Class1AffinityPredictor containing a subset of this
        predictor's neural networks.

        Parameters
        ----------
        predicate : Class1NeuralNetwork -> boolean
            Function specifying which neural networks to include

        Returns
        -------
        Class1AffinityPredictor
        """
        allele_to_allele_specific_models = {}
        for (allele, models) in self.allele_to_allele_specific_models.items():
            allele_to_allele_specific_models[allele] = [
                m for m in models if predicate(m)
            ]
        class1_pan_allele_models = [
            m for m in self.class1_pan_allele_models if predicate(m)
        ]

        return Class1AffinityPredictor(
            allele_to_allele_specific_models=allele_to_allele_specific_models,
            class1_pan_allele_models=class1_pan_allele_models,
            allele_to_fixed_length_sequence=self.allele_to_fixed_length_sequence,
        )

    def model_select(
            self,
            score_function,
            alleles=None,
            min_models=1,
            max_models=10000):
        """
        Perform model selection using a user-specified scoring function.

        Model selection is done using a "step up" variable selection procedure,
        in which models are repeatedly added to an ensemble until the score
        stops improving.

        Parameters
        ----------
        score_function : Class1AffinityPredictor -> float function
            Scoring function

        alleles : list of string, optional
            If not specified, model selection is performed for all alleles.

        min_models : int, optional
            Min models to select per allele

        max_models : int, optional
            Max models to select per allele

        Returns
        -------
        Class1AffinityPredictor : predictor containing the selected models
        """

        if alleles is None:
            alleles = self.supported_alleles

        dfs = []
        allele_to_allele_specific_models = {}
        for allele in alleles:
            df = pandas.DataFrame({
                'model': self.allele_to_allele_specific_models[allele]
            })
            df["model_num"] = df.index
            df["allele"] = allele
            df["selected"] = False

            round_num = 1

            while not df.selected.all() and sum(df.selected) < max_models:
                score_col = "score_%2d" % round_num
                prev_score_col = "score_%2d" % (round_num - 1)

                existing_selected = list(df[df.selected].model)
                df[score_col] = [
                    numpy.nan if row.selected else
                    score_function(
                        Class1AffinityPredictor(
                            allele_to_allele_specific_models={
                                allele: [row.model] + existing_selected
                            }
                        )
                    )
                    for (_, row) in df.iterrows()
                ]

                if round_num > min_models and (
                        df[score_col].max() < df[prev_score_col].max()):
                    break

                # In case of a tie, pick a model at random.
                (best_model_index,) = df.loc[
                    (df[score_col] == df[score_col].max())
                ].sample(1).index
                df.loc[best_model_index, "selected"] = True
                round_num += 1

            dfs.append(df)
            allele_to_allele_specific_models[allele] = list(
                df.loc[df.selected].model)

        df = pandas.concat(dfs, ignore_index=True)

        new_predictor = Class1AffinityPredictor(
            allele_to_allele_specific_models,
            metadata_dataframes={
                "model_selection": df,
            })
        return new_predictor

