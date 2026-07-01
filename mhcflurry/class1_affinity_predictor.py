# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import hashlib
import json
import logging
import shlex
import time
import warnings
from os.path import join, abspath, dirname
from os import environ
import numpy
import pandas


from .class1_neural_network import (
    Class1NeuralNetwork,
    cartesian_network_output,
    cartesian_output_log_ic50_sum,
)
from .motif_summary import (
    prepare_motif_summary_state_gpu,
    motif_summary_chunk_gpu,
)
from .common import (
    derive_seed,
    random_peptides,
    positional_frequency_matrix,
    normalize_allele_name,
    AlleleKeyResolver,
)
from .encodable_sequences import EncodableSequences
from .percent_rank_transform import PercentRankTransform
from .regression_target import to_ic50
from .version import __version__
from .ensemble_centrality import CENTRALITY_MEASURES
from .allele_encoding import AlleleEncoding
from .affinity import calibration_sizing, model_selection, persistence


# Default function for combining predictions across models in an ensemble.
# See ensemble_centrality.py for other options.
DEFAULT_CENTRALITY_MEASURE = "mean"

# Any value > 0 will result in attempting to optimize models after loading.
OPTIMIZATION_LEVEL = int(environ.get("MHCFLURRY_OPTIMIZATION_LEVEL", 1))


class Class1AffinityPredictor(object):
    """
    High-level interface for peptide/MHC I binding affinity prediction.

    This class manages low-level `Class1NeuralNetwork` instances, each of which
    wraps a single PyTorch network. The purpose of `Class1AffinityPredictor` is to
    implement ensembles, handling of multiple alleles, and predictor loading and
    saving. It also provides a place to keep track of metadata like prediction
    histograms for percentile rank calibration.
    """
    def __init__(
            self,
            allele_to_allele_specific_models=None,
            class1_pan_allele_models=None,
            allele_to_sequence=None,
            manifest_df=None,
            allele_to_percent_rank_transform=None,
            metadata_dataframes=None,
            provenance_string=None,
            optimization_info=None,
            models_dir=None):
        """
        Parameters
        ----------
        allele_to_allele_specific_models : dict of string -> list of `Class1NeuralNetwork`
            Ensemble of single-allele models to use for each allele.

        class1_pan_allele_models : list of `Class1NeuralNetwork`
            Ensemble of pan-allele models.

        allele_to_sequence : dict of string -> string
            MHC allele name to fixed-length pseudosequence. Required only if
            class1_pan_allele_models is specified.

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

        provenance_string : string, optional
            Optional info string to use in __str__.

        optimization_info : dict, optional
            Dict describing any optimizations already performed on the model.
            The only currently supported optimization is to merge ensembles
            together into one PyTorch model.

        models_dir : string, optional
            Directory this predictor was loaded from. Used for diagnostics.
        """

        if allele_to_allele_specific_models is None:
            allele_to_allele_specific_models = {}
        if class1_pan_allele_models is None:
            class1_pan_allele_models = []

        self.allele_to_sequence = (
            dict(allele_to_sequence)
            if allele_to_sequence is not None else None)  # make a copy

        self._master_allele_encoding = None
        if class1_pan_allele_models:
            assert self.allele_to_sequence

        self.allele_to_allele_specific_models = allele_to_allele_specific_models
        self.class1_pan_allele_models = class1_pan_allele_models
        self._manifest_df = manifest_df

        if not allele_to_percent_rank_transform:
            allele_to_percent_rank_transform = {}
        self.allele_to_percent_rank_transform = allele_to_percent_rank_transform
        self.metadata_dataframes = (
            dict(metadata_dataframes) if metadata_dataframes else {})
        self._cache = {}
        self.optimization_info = optimization_info if optimization_info else {}

        assert isinstance(self.allele_to_allele_specific_models, dict)
        assert isinstance(self.class1_pan_allele_models, list)

        self.provenance_string = provenance_string
        self.models_dir = models_dir
        self.allele_to_canonical = {}  # populated by load()

    @property
    def manifest_df(self):
        """
        A pandas.DataFrame describing the models included in this predictor.

        Based on:
        - self.class1_pan_allele_models
        - self.allele_to_allele_specific_models

        Returns
        -------
        pandas.DataFrame
        """
        if self._manifest_df is None:
            rows = []
            for (i, model) in enumerate(self.class1_pan_allele_models):
                rows.append((
                    self.model_name("pan-class1", i),
                    "pan-class1",
                    json.dumps(model.get_config()),
                    model
                ))
            for (allele, models) in self.allele_to_allele_specific_models.items():
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
            - self.class1_pan_allele_models
            - self.allele_to_allele_specific_models
            - self.allele_to_sequence

        Methods that mutate these instance variables will call this method on
        their own if needed.
        """
        self._cache.clear()
        self.provenance_string = None
        self.models_dir = None

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
        allele_to_sequence = predictors[0].allele_to_sequence

        for predictor in predictors:
            for (allele, networks) in (
                    predictor.allele_to_allele_specific_models.items()):
                allele_to_allele_specific_models[allele].extend(networks)
            class1_pan_allele_models.extend(
                predictor.class1_pan_allele_models)

        return Class1AffinityPredictor(
            allele_to_allele_specific_models=allele_to_allele_specific_models,
            class1_pan_allele_models=class1_pan_allele_models,
            allele_to_sequence=allele_to_sequence
        )

    def merge_in_place(self, others):
        """
        Add the models present in other predictors into the current predictor.

        Parameters
        ----------
        others : list of Class1AffinityPredictor
            Other predictors to merge into the current predictor.

        Returns
        -------
        list of string : names of newly added models
        """
        new_model_names = []
        original_manifest = self.manifest_df
        new_manifest_rows = []
        for predictor in others:
            for model in predictor.class1_pan_allele_models:
                model_name = self.model_name(
                    "pan-class1",
                    len(self.class1_pan_allele_models))
                row = pandas.Series(collections.OrderedDict([
                    ("model_name", model_name),
                    ("allele", "pan-class1"),
                    ("config_json", json.dumps(model.get_config())),
                    ("model", model),
                ])).to_frame().T
                new_manifest_rows.append(row)
                self.class1_pan_allele_models.append(model)
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
                    new_manifest_rows.append(row)
                    current_models.append(model)
                    new_model_names.append(model_name)

        self._manifest_df = pandas.concat(
            [original_manifest] + new_manifest_rows,
            ignore_index=True)

        self.clear_cache()
        self.check_consistency()
        return new_model_names

    def canonicalize_allele_name(self, raw_name):
        """
        Normalize an allele name and map it to the canonical pseudosequence
        key if possible.

        Tries without IMGT aliases first so that alleles like HLA-C*01:01
        (which aliases map to C*01:02) resolve to their own pseudosequence
        when one exists.

        Raises on names that cannot be normalized (loud failure for explicit
        prediction/calibration inputs). The no-alias-first logic lives in
        ``AlleleKeyResolver``; training ingestion shares it via
        ``canonicalize_allele_series``.

        Parameters
        ----------
        raw_name : str

        Returns
        -------
        str
        """
        # Cache one resolver (built from the already-built load-time maps) so
        # the per-row calls in predict_to_dataframe don't reconstruct it each
        # time. Lives in ``self._cache`` so ``clear_cache`` drops it whenever
        # the allele maps change, exactly as ``supported_alleles`` is handled.
        if "allele_key_resolver" not in self._cache:
            self._cache["allele_key_resolver"] = AlleleKeyResolver(
                self.allele_to_sequence, self.allele_to_canonical)
        return self._cache["allele_key_resolver"].resolve(
            raw_name, raise_on_error=True)

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
            if self.allele_to_sequence:
                result = result.union(self.allele_to_sequence)
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

    def check_consistency(self):
        """
        Verify that self.manifest_df is consistent with:
        - self.class1_pan_allele_models
        - self.allele_to_allele_specific_models

        Currently only checks for agreement on the total number of models.

        Throws AssertionError if inconsistent.
        """
        num_models = len(self.class1_pan_allele_models) + sum(
            len(v) for v in self.allele_to_allele_specific_models.values())
        assert len(self.manifest_df) == num_models, (
            "Manifest seems out of sync with models: %d vs %d entries: "
            "\n%s\npan-allele: %s\nallele-specific: %s"% (
                len(self.manifest_df),
                num_models,
                str(self.manifest_df),
                str(self.class1_pan_allele_models),
                str(self.allele_to_allele_specific_models)))

    def save(self, models_dir, model_names_to_write=None, write_metadata=True):
        """
        Serialize the predictor to a directory on disk. If the directory does
        not exist it will be created.

        The serialization format consists of a file called "manifest.csv" with
        the configurations of each Class1NeuralNetwork, along with per-network
        files giving the model weights. If there are pan-allele predictors in
        the ensemble, the pseudosequences are also stored in the
        directory. There is also a small file "info.txt" with basic metadata:
        when the models were trained, by whom, on what host.

        Parameters
        ----------
        models_dir : string
            Path to directory. It will be created if it doesn't exist.

        model_names_to_write : list of string, optional
            Only write the weights for the specified models. Useful for
            incremental updates during training. Passing an explicit empty
            list writes no model artifacts; this is used by calibration-only
            updates that should replace ``percent_ranks.csv`` without touching
            the manifest, weights, model provenance, allele sequences, or
            optimization metadata. Explicit ``metadata_dataframes`` are still
            written when ``write_metadata`` is true.

        write_metadata : boolean, optional
            Whether to write optional metadata
        """
        return persistence.save_predictor(
            self,
            models_dir,
            model_names_to_write=model_names_to_write,
            write_metadata=write_metadata,
        )

    @staticmethod
    def load(models_dir=None, max_models=None, optimization_level=None):
        """
        Deserialize a predictor from a directory on disk.

        Parameters
        ----------
        models_dir : string
            Path to directory. If unspecified the default downloaded models are
            used.

        max_models : int, optional
            Maximum number of `Class1NeuralNetwork` instances to load

        optimization_level : int
            If >0, model optimization will be attempted. Defaults to value of
            environment variable MHCFLURRY_OPTIMIZATION_LEVEL.

        Returns
        -------
        `Class1AffinityPredictor` instance
        """
        return persistence.load_predictor(
            Class1AffinityPredictor,
            models_dir=models_dir,
            max_models=max_models,
            optimization_level=optimization_level,
            optimization_level_default=OPTIMIZATION_LEVEL,
        )

    def __repr__(self):
        pieces = ["at 0x%0x" % id(self), "[mhcflurry %s]" % __version__]

        pan_models = len(self.class1_pan_allele_models)
        total_models = len(self.neural_networks)
        if total_models == 0:
            pieces.append("[empty]")
        elif pan_models == total_models:
            pieces.append("[pan]")
        elif pan_models == 0:
            pieces.append("[allele-specific]")
        else:
            pieces.append("[pan+allele-specific]")

        if self.provenance_string:
            pieces.append(self.provenance_string)

        return "<Class1AffinityPredictor %s>" % " ".join(pieces)

    def optimize(self, warn=True):
        """
        EXPERIMENTAL: Optimize the predictor for faster predictions.

        Currently the only optimization implemented is to merge multiple pan-
        allele predictors at the PyTorch level.

        The optimization is performed in-place, mutating the instance.

        Returns
        ----------
        bool
            Whether optimization was performed

        """
        num_class1_pan_allele_models = len(self.class1_pan_allele_models)
        if num_class1_pan_allele_models > 1:
            provenance_string = self.provenance_string
            try:
                self.class1_pan_allele_models = [
                    Class1NeuralNetwork.merge(
                        self.class1_pan_allele_models,
                        merge_method="concatenate")
                ]
            except NotImplementedError as e:
                if warn:
                    logging.warning("Optimization failed: %s", str(e))
                return False
            self._manifest_df = None
            self.clear_cache()
            self.optimization_info["pan_models_merged"] = True
            self.optimization_info["num_pan_models_merged"] = (
                num_class1_pan_allele_models)
            self.provenance_string = provenance_string
        else:
            return False
        return True

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
        return "%s-%d-%s" % (
            allele.upper().replace("*", "_").replace(":", "_"),
            num,
            random_string)

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

    @property
    def master_allele_encoding(self):
        """
        An AlleleEncoding containing the universe of alleles specified by
        self.allele_to_sequence.

        Returns
        -------
        AlleleEncoding
        """
        if (self._master_allele_encoding is None or
                self._master_allele_encoding.allele_to_sequence !=
                self.allele_to_sequence):
            self._master_allele_encoding = AlleleEncoding(
                allele_to_sequence=self.allele_to_sequence)
        return self._master_allele_encoding

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
            progress_print_interval=5.0,
            seed=None):
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
            See `Class1NeuralNetwork.fit` for details.

        train_rounds : sequence of int
            Each training point i will be used on training rounds r for which
            train_rounds[i] > r, r >= 0.

        models_dir_for_save : string, optional
            If specified, the Class1AffinityPredictor is (incrementally) written
            to the given models dir after each neural network is fit.

        verbose : int
            Verbosity level for training output

        progress_preamble : string
            Optional string of information to include in each progress update

        progress_print_interval : float
            How often (in seconds) to print progress. Set to None to disable.

        seed : int, optional
            Base seed for this allele's fits. When given, each (model_num,
            architecture_num) pair gets a distinct sub-seed derived from it,
            so ensemble members are decorrelated but the whole call is
            reproducible. When None, each `Class1NeuralNetwork.fit` is left
            entropy-seeded as before.

        Returns
        -------
        list of `Class1NeuralNetwork`
        """

        allele = normalize_allele_name(allele)
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
                    # Distinct sub-seed per (ensemble member, architecture,
                    # round) so members don't initialize identically and each
                    # round sees a different shuffle / random-negative draw,
                    # while the whole call stays reproducible from the
                    # caller's base seed. fit() re-seeds the global RNG at its
                    # start, so the round_num must be part of the mix —
                    # otherwise every round would replay round 0's randomness.
                    fit_seed = (
                        None if seed is None
                        else derive_seed(
                            seed, model_num, architecture_num, round_num))
                    (round_peptides, round_affinities, round_inequalities) = (
                        peptides_affinities_inequalities_per_round[round_num]
                    )
                    model.fit(
                        round_peptides,
                        round_affinities,
                        inequalities=round_inequalities,
                        verbose=verbose,
                        seed=fit_seed,
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
            Verbosity level for training output

        progress_preamble : string
            Optional string of information to include in each progress update

        progress_print_interval : float
            How often (in seconds) to print progress. Set to None to disable.

        Returns
        -------
        list of `Class1NeuralNetwork`
        """

        alleles = pandas.Series(alleles).map(normalize_allele_name)
        allele_encoding = AlleleEncoding(
            alleles,
            borrow_from=self.master_allele_encoding)

        encodable_peptides = EncodableSequences.create(peptides)
        models = []
        for i in range(n_models):
            logging.info("Training model %d / %d", i + 1, n_models)
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
            row = pandas.Series(collections.OrderedDict([
                ("model_name", model_name),
                ("allele", "pan-class1"),
                ("config_json", json.dumps(model.get_config())),
                ("model", model),
            ])).to_frame().T
            self._manifest_df = pandas.concat(
                [self.manifest_df, row], ignore_index=True)
            self.class1_pan_allele_models.append(model)
            if models_dir_for_save:
                self.save(
                    models_dir_for_save, model_names_to_write=[model_name])
            models.append(model)

        self.clear_cache()
        return models

    def add_pan_allele_model(self, model, models_dir_for_save=None):
        """
        Add a pan-allele model to the ensemble and optionally do an incremental
        save.

        Parameters
        ----------
        model : Class1NeuralNetwork
        models_dir_for_save : string
            Directory to save resulting ensemble to
        """
        model_name = self.model_name("pan-class1", 1)
        row = pandas.Series(collections.OrderedDict([
            ("model_name", model_name),
            ("allele", "pan-class1"),
            ("config_json", json.dumps(model.get_config())),
            ("model", model),
        ])).to_frame().T
        self._manifest_df = pandas.concat(
            [self.manifest_df, row], ignore_index=True)
        self.class1_pan_allele_models.append(model)
        self.clear_cache()
        self.check_consistency()
        if models_dir_for_save:
            self.save(
                models_dir_for_save, model_names_to_write=[model_name])

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
            normalized_allele = self.canonicalize_allele_name(allele)
            calibrated_allele = self.percent_rank_calibrated_allele(
                normalized_allele
            )
            if calibrated_allele is not None:
                transform = self.allele_to_percent_rank_transform[calibrated_allele]
                return transform.transform(affinities)

            allele_repr = allele + (
                "" if allele == normalized_allele
                else " (normalized to %s)" % normalized_allele)
            affinity_known = (
                self.allele_to_sequence is not None
                and normalized_allele in self.allele_to_sequence
            )
            hint_lines = [
                "Missing percentile-rank calibration for %s." % allele_repr,
            ]
            if affinity_known:
                hint_lines.append(
                    "Affinity predictions are available; percentile ranks are not."
                )
            else:
                hint_lines.append(
                    "The predictor also lacks an allele sequence for %s; "
                    "affinity prediction is unavailable." % normalized_allele
                )
            calibrate_command = ["mhcflurry-calibrate-percentile-ranks"]
            models_dir = self.models_dir_for_diagnostics()
            if models_dir:
                calibrate_command.extend(["--models-dir", models_dir])
            calibrate_command.extend(["--allele", normalized_allele, "..."])
            calibrate_command = " ".join(
                shlex.quote(part) for part in calibrate_command
            )
            if models_dir:
                command_message = "Calibrate with: `%s`." % calibrate_command
            else:
                command_message = (
                    "Calibrate with: `%s` against this models directory."
                    % calibrate_command
                )
            hint_lines.append(
                command_message
            )
            msg = " ".join(hint_lines)
            if throw:
                raise ValueError(msg)
            warnings.warn(msg)
            return numpy.ones(len(affinities)) * numpy.nan  # Return NaNs

        if alleles is None:
            raise ValueError("Specify allele or alleles")

        df = pandas.DataFrame({"affinity": affinities})
        df["allele"] = alleles
        df["result"] = numpy.nan
        for (allele, sub_df) in df.groupby("allele"):
            df.loc[sub_df.index, "result"] = self.percentile_ranks(
                sub_df.affinity, allele=allele, throw=throw)
        return df.result.values

    def model_source_description(self):
        """Return a compact human-readable description of this predictor."""
        pieces = []
        models_dir = self.models_dir_for_diagnostics()
        if models_dir:
            pieces.append("models_dir=%s" % models_dir)
        if self.provenance_string:
            pieces.append(self.provenance_string)
        pieces.append("%d model(s)" % len(self.neural_networks))
        pieces.append(
            "%d percent-rank calibration(s)" % (
                len(self.allele_to_percent_rank_transform)))
        return "; ".join(pieces)

    def models_dir_for_diagnostics(self):
        """Return explicit or inferred models dir for user-facing messages."""
        if self.models_dir:
            return self.models_dir

        source_dirs = set()
        for model in self.neural_networks:
            for path in getattr(model, "network_weight_paths", ()):
                source_dir = dirname(path)
                if source_dir:
                    source_dirs.add(abspath(source_dir))
        if len(source_dirs) == 1:
            return next(iter(source_dirs))
        return None

    def percent_rank_calibrated_allele(self, allele):
        """Return the allele key whose percentile-rank transform applies.

        Percent-rank calibration is allele-specific, but pan predictors can
        reuse calibration from another allele with the same pseudosequence.
        This helper centralizes that equivalence check for prediction and CLI
        status/filtering code.
        """
        normalized_allele = self.canonicalize_allele_name(allele)
        if normalized_allele in self.allele_to_percent_rank_transform:
            return normalized_allele

        if (
                self.allele_to_sequence is None
                or normalized_allele not in self.allele_to_sequence):
            return None

        sequence = self.allele_to_sequence[normalized_allele]
        for other_allele in sorted(self.allele_to_sequence):
            if (
                    self.allele_to_sequence[other_allele] == sequence
                    and other_allele in self.allele_to_percent_rank_transform):
                return other_allele
        return None

    def predict(
            self,
            peptides,
            alleles=None,
            allele=None,
            throw=True,
            centrality_measure=DEFAULT_CENTRALITY_MEASURE,
            model_kwargs={}):
        """
        Predict nM binding affinities.

        If multiple predictors are available for an allele, the predictions are
        the geometric means of the individual model (nM) predictions.

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
        model_kwargs : dict
            Additional keyword arguments to pass to Class1NeuralNetwork.predict
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
            model_kwargs=model_kwargs,
        )
        return df.prediction.values

    def predict_cartesian_pan_allele(
            self,
            peptides,
            alleles,
            throw=True,
            centrality_measure=DEFAULT_CENTRALITY_MEASURE,
            model_kwargs=None):
        """
        Predict a peptide x allele matrix using the optimized pan-allele path.

        Returns ``None`` when the predictor is not a single optimized merged
        pan-allele model. Callers can then fall back to ``predict``. This path
        keeps peptide encodings unique, computes the peptide stage once per
        peptide batch, and combines it with allele embeddings on-device.
        """
        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = dict(model_kwargs)

        if (
                not self.optimization_info.get("pan_models_merged", False)
                or len(self.class1_pan_allele_models) != 1
                or self.allele_to_allele_specific_models):
            return None

        import torch
        from .pytorch_sizing import (
            DEFAULT_PREDICT_BATCH_SIZE,
            resolve_prediction_batch_size,
        )
        from .pytorch_training import (
            configure_matmul_precision,
            maybe_compile_network,
        )

        peptides = EncodableSequences.create(peptides)
        if len(peptides) == 0 or len(alleles) == 0:
            return numpy.empty((len(peptides), len(alleles)), dtype="float64")

        (min_peptide_length, max_peptide_length) = (
            self.supported_peptide_lengths)
        peptide_df = pandas.DataFrame({"peptide": peptides.sequences})
        sequence_length = peptide_df.peptide.str.len()
        supported_peptide = (
            (sequence_length >= min_peptide_length) &
            (sequence_length <= max_peptide_length)
        )
        if (~supported_peptide).any():
            msg = (
                "%d peptides have lengths outside of supported range [%d, %d]: "
                "%s" % (
                    (~supported_peptide).sum(),
                    min_peptide_length,
                    max_peptide_length,
                    str(peptide_df.loc[~supported_peptide].peptide.unique())))
            logging.warning(msg)
            if throw:
                raise ValueError(msg)

        peptide_has_valid_amino_acids = (
            (~supported_peptide) |
            peptide_df.peptide.str.upper().str.match("^[ACDEFGHIKLMNPQRSTVWY]+$")
        )
        supported_peptide = supported_peptide & peptide_has_valid_amino_acids
        if (~peptide_has_valid_amino_acids).any():
            msg = (
                "%d peptides have nonstandard amino acids: "
                "%s" % (
                    (~peptide_has_valid_amino_acids).sum(),
                    str(peptide_df.loc[
                        ~peptide_has_valid_amino_acids
                    ].peptide.unique())))
            logging.warning(msg)
            if throw:
                raise ValueError(msg)

        normalized_alleles = [
            self.canonicalize_allele_name(allele)
            for allele in alleles
        ]
        unsupported_alleles = [
            allele for allele in normalized_alleles
            if allele not in self.allele_to_sequence
        ]
        if unsupported_alleles:
            msg = "No sequences for allele(s): %s." % " ".join(
                unsupported_alleles)
            logging.warning(msg)
            if throw:
                raise ValueError(msg)
            return None

        model_obj = self.class1_pan_allele_models[0]
        master = self.master_allele_encoding
        allele_encoding = AlleleEncoding(
            normalized_alleles,
            borrow_from=master,
        ).compact()
        (
            allele_encoding_input,
            allele_representations,
        ) = model_obj.allele_encoding_to_network_input(allele_encoding)
        model_obj.set_allele_representations(allele_representations)

        device = model_obj.get_device()
        configure_matmul_precision(device)
        network = model_obj.network(borrow=True)
        network.to(device)
        network = maybe_compile_network(network, device)
        network.eval()

        n_peptides = len(peptides)
        n_alleles = len(normalized_alleles)
        result = numpy.full((n_peptides, n_alleles), numpy.nan, dtype="float64")
        supported_indices = numpy.flatnonzero(supported_peptide.to_numpy())
        if len(supported_indices) == 0:
            return result
        peptides_to_predict = EncodableSequences.create(
            peptides.sequences[supported_indices])
        peptide_input = model_obj.peptides_to_network_input(peptides_to_predict)
        batch_size = resolve_prediction_batch_size(
            model_kwargs.get("batch_size", DEFAULT_PREDICT_BATCH_SIZE),
            device,
            model=network,
            num_workers_per_gpu=model_kwargs.get("num_workers_per_gpu", 1),
        )
        batch_size = int(batch_size)

        allele_encoding_input = numpy.asarray(allele_encoding_input)
        if not allele_encoding_input.flags.writeable:
            allele_encoding_input = allele_encoding_input.copy()
        allele_idx = torch.from_numpy(allele_encoding_input).to(device)
        peptide_is_indices = model_obj.uses_peptide_torch_encoding()

        if callable(centrality_measure):
            centrality_function = centrality_measure
        else:
            centrality_function = CENTRALITY_MEASURES[centrality_measure]

        def peptide_tensor(batch_array):
            batch_array = numpy.asarray(batch_array)
            if not batch_array.flags.writeable:
                batch_array = batch_array.copy()
            keep_int = (
                peptide_is_indices
                and batch_array.ndim == 2
                and numpy.issubdtype(batch_array.dtype, numpy.integer)
            )
            if not keep_int:
                batch_array = numpy.asarray(batch_array, dtype=numpy.float32)
            return torch.from_numpy(batch_array).to(device)

        with torch.no_grad():
            for start in range(0, len(peptides_to_predict), batch_size):
                end = min(start + batch_size, len(peptides_to_predict))
                peptide_batch = peptide_tensor(peptide_input[start:end])
                peptide_stage = network.forward_peptide_stage(peptide_batch)
                output = network.forward_cartesian_from_peptide_stage(
                    peptide_stage,
                    allele_idx,
                )
                affinities = to_ic50(output.detach().cpu().numpy())
                if affinities.ndim == 3 and affinities.shape[2] > 1:
                    log_values = numpy.log(
                        affinities.reshape((-1, affinities.shape[2]))
                    )
                    centers = numpy.exp(centrality_function(log_values))
                    affinities = centers.reshape(n_alleles, end - start)
                else:
                    affinities = affinities.reshape(n_alleles, end - start)
                result[supported_indices[start:end]] = affinities.T
        return result

    def predict_to_dataframe(
            self,
            peptides,
            alleles=None,
            allele=None,
            throw=True,
            include_individual_model_predictions=False,
            include_percentile_ranks=True,
            include_confidence_intervals=True,
            centrality_measure=DEFAULT_CENTRALITY_MEASURE,
            model_kwargs={}):
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
            columns in the result DataFrame.
        include_percentile_ranks : boolean, default True
            If True, a "prediction_percentile" column will be included giving
            the percentile ranks. If no percentile rank info is available,
            this will be ignored with a warning.
        centrality_measure : string or callable
            Measure of central tendency to use to combine predictions in the
            ensemble. Options include: mean, median, robust_mean.
        model_kwargs : dict
            Additional keyword arguments to pass to Class1NeuralNetwork.predict

        Returns
        -------
        `pandas.DataFrame` of predictions
        """
        if isinstance(peptides, str):
            raise TypeError("peptides must be a list or array, not a string")
        if isinstance(alleles, str):
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
            normalized_allele = self.canonicalize_allele_name(allele)
            df["allele"] = normalized_allele
            df["normalized_allele"] = normalized_allele
            unique_alleles = [normalized_allele]
        else:
            df["allele"] = [
                self.canonicalize_allele_name(a) for a in alleles]
            df["normalized_allele"] = df["allele"]
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
            all_peptides_supported = False
            sequence_length = df.peptide.str.len()
            df["supported_peptide"] = (
                (sequence_length >= min_peptide_length) &
                (sequence_length <= max_peptide_length))
            if (~df.supported_peptide).any():
                msg = (
                    "%d peptides have lengths outside of supported range [%d, %d]: "
                    "%s" % (
                        (~df.supported_peptide).sum(),
                        min_peptide_length,
                        max_peptide_length,
                        str(df.loc[~df.supported_peptide].peptide.unique())))
                logging.warning(msg)
                if throw:
                    raise ValueError(msg)
        else:
            # Handle common case efficiently.
            df["supported_peptide"] = True
            all_peptides_supported = True

        peptide_has_valid_amino_acids = (
            (~df.supported_peptide) |
            df.peptide.str.upper().str.match("^[ACDEFGHIKLMNPQRSTVWY]+$"))
        df["supported_peptide"] = (
                df["supported_peptide"] & peptide_has_valid_amino_acids)

        if (~peptide_has_valid_amino_acids).any():
            all_peptides_supported = False
            msg = (
                "%d peptides have nonstandard amino acids: "
                "%s" % (
                    (~peptide_has_valid_amino_acids).sum(),
                    str(df.loc[~peptide_has_valid_amino_acids].peptide.unique())))
            logging.warning(msg)
            if throw:
                raise ValueError(msg)

        num_pan_models = (
            len(self.class1_pan_allele_models)
            if not self.optimization_info.get("pan_models_merged", False)
            else self.optimization_info["num_pan_models_merged"])
        max_single_allele_models = max(
            len(self.allele_to_allele_specific_models.get(allele, []))
            for allele in unique_alleles
        )
        predictions_array = numpy.zeros(
            shape=(df.shape[0], num_pan_models + max_single_allele_models),
            dtype="float64")
        predictions_array[:] = numpy.nan

        if self.class1_pan_allele_models:
            master_allele_encoding = self.master_allele_encoding
            unsupported_alleles = [
                allele for allele in
                df.normalized_allele.unique()
                if allele not in self.allele_to_sequence
            ]
            if unsupported_alleles:
                truncate_at = 100
                allele_string = " ".join(
                    sorted(self.allele_to_sequence)[:truncate_at])
                if len(self.allele_to_sequence) > truncate_at:
                    allele_string += " + %d more alleles" % (
                        len(self.allele_to_sequence) - truncate_at)
                msg = (
                    "No sequences for allele(s): %s.\n"
                    "Supported alleles: %s" % (
                        " ".join(unsupported_alleles), allele_string))
                logging.warning(msg)
                if throw:
                    raise ValueError(msg)
            mask = df.supported_peptide & (
                ~df.normalized_allele.isin(unsupported_alleles))

            row_slice = None
            if mask is None or mask.all():
                row_slice = slice(None, None, None)  # all rows
                masked_allele_encoding = AlleleEncoding(
                    df.normalized_allele,
                    borrow_from=master_allele_encoding)
                masked_peptides = EncodableSequences.create(peptides)
            elif mask.sum() > 0:
                row_slice = mask
                masked_allele_encoding = AlleleEncoding(
                    df.loc[mask].normalized_allele,
                    borrow_from=master_allele_encoding)
                masked_peptides = EncodableSequences.create(
                    peptides.sequences[mask])

            if row_slice is not None:
                # The following line is a performance optimization that may be
                # revisited. It causes the neural network to set to include
                # only the alleles actually being predicted for. This makes
                # the network much smaller. However, subsequent calls to
                # predict will need to reset these weights, so there is a
                # tradeoff.
                masked_allele_encoding = masked_allele_encoding.compact()

                if self.optimization_info.get("pan_models_merged"):
                    # Multiple pan-allele models have been merged into one
                    # at the PyTorch level.
                    assert len(self.class1_pan_allele_models) == 1
                    predictions = self.class1_pan_allele_models[0].predict(
                        masked_peptides,
                        allele_encoding=masked_allele_encoding,
                        output_index=None,
                        **model_kwargs)
                    predictions_array[row_slice, :num_pan_models] = predictions
                else:
                    for (i, model) in enumerate(self.class1_pan_allele_models):
                        predictions_array[row_slice, i] = model.predict(
                            masked_peptides,
                            allele_encoding=masked_allele_encoding,
                            **model_kwargs)

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
                if len(unique_alleles) == 1 and all_peptides_supported:
                    mask = None
                else:
                    mask = (
                        (df.normalized_allele == allele) &
                        df.supported_peptide).values

                row_slice = None
                if mask is None or mask.all():
                    peptides_for_allele = peptides
                    row_slice = slice(None, None, None)
                elif mask.sum() > 0:
                    peptides_for_allele = EncodableSequences.create(
                        df.loc[mask].peptide.values)
                    row_slice = mask

                if row_slice is not None:
                    for (i, model) in enumerate(models):
                        predictions_array[
                            row_slice,
                            num_pan_models + i,
                        ] = model.predict(peptides_for_allele, **model_kwargs)

        if callable(centrality_measure):
            centrality_function = centrality_measure
        else:
            centrality_function = CENTRALITY_MEASURES[centrality_measure]

        logs = numpy.log(predictions_array)
        row_has_predictions = (~numpy.isnan(logs)).any(axis=1)
        log_centers = numpy.full(df.shape[0], numpy.nan, dtype="float64")
        if row_has_predictions.any():
            log_centers[row_has_predictions] = centrality_function(
                logs[row_has_predictions]
            )
        df["prediction"] = numpy.exp(log_centers)

        if include_confidence_intervals:
            prediction_low = numpy.full(df.shape[0], numpy.nan, dtype="float64")
            prediction_high = numpy.full(df.shape[0], numpy.nan, dtype="float64")
            if row_has_predictions.any():
                prediction_low[row_has_predictions] = numpy.exp(
                    numpy.nanpercentile(logs[row_has_predictions], 5.0, axis=1)
                )
                prediction_high[row_has_predictions] = numpy.exp(
                    numpy.nanpercentile(logs[row_has_predictions], 95.0, axis=1)
                )
            df["prediction_low"] = prediction_low
            df["prediction_high"] = prediction_high

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

        del df["supported_peptide"]
        del df["normalized_allele"]
        return df

    def calibrate_percentile_ranks(
            self,
            peptides=None,
            num_peptides_per_length=int(1e5),
            alleles=None,
            bins=None,
            motif_summary=False,
            summary_top_peptide_fractions=[0.001],
            verbose=False,
            model_kwargs={}):
        """
        Compute the cumulative distribution of ic50 values for a set of alleles
        over a large universe of random peptides, to enable taking quantiles
        of this distribution later.

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
        motif_summary : bool
            If True, the length distribution and per-position amino acid
            frequencies are also calculated for the top x fraction of tightest-
            binding peptides, where each value of x is given in the
            summary_top_peptide_fractions list.
        summary_top_peptide_fractions : list of float
            Only used if motif_summary is True
        verbose : boolean
            Whether to print status updates to stdout
        model_kwargs : dict
            Additional low-level Class1NeuralNetwork.predict() kwargs.

        Returns
        ----------
        dict of string -> pandas.DataFrame

        If motif_summary is True, this will have keys  "frequency_matrices" and
        "length_distributions". Otherwise it will be empty.

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

        if motif_summary:
            frequency_matrices = []
            length_distributions = []
        else:
            frequency_matrices = None
            length_distributions = None
        for allele in alleles:
            start = time.time()
            predictions = self.predict(
                encoded_peptides, allele=allele, model_kwargs=model_kwargs)
            if verbose:
                elapsed = time.time() - start
                print(
                    "Generated %d predictions for allele %s in %0.2f sec: "
                    "%0.2f predictions / sec" % (
                        len(encoded_peptides.sequences),
                        allele,
                        elapsed,
                        len(encoded_peptides.sequences) / elapsed))
            transform = PercentRankTransform()
            transform.fit(predictions, bins=bins)
            self.allele_to_percent_rank_transform[allele] = transform

            if frequency_matrices is not None:
                predictions_df = pandas.DataFrame({
                    'peptide': encoded_peptides.sequences,
                    'prediction': predictions
                }).drop_duplicates('peptide').set_index("peptide")
                predictions_df["length"] = predictions_df.index.str.len()
                for (length, sub_df) in predictions_df.groupby("length"):
                    for cutoff_fraction in summary_top_peptide_fractions:
                        selected = sub_df.prediction.nsmallest(
                            max(
                                int(len(sub_df) * cutoff_fraction),
                                1)).index.values
                        matrix = positional_frequency_matrix(selected).reset_index()
                        original_columns = list(matrix.columns)
                        matrix["allele"] = allele
                        matrix["length"] = length
                        matrix["cutoff_fraction"] = cutoff_fraction
                        matrix["cutoff_count"] = len(selected)
                        matrix = matrix[
                            ["allele", "length", "cutoff_fraction", "cutoff_count"]
                            + original_columns
                        ]
                        frequency_matrices.append(matrix)

                # Length distribution
                for cutoff_fraction in summary_top_peptide_fractions:
                    cutoff_count = max(
                        int(len(predictions_df) * cutoff_fraction), 1)
                    length_distribution = predictions_df.prediction.nsmallest(
                        cutoff_count).index.str.len().value_counts()
                    length_distribution.index.name = "length"
                    length_distribution /= length_distribution.sum()
                    length_distribution = length_distribution.to_frame()
                    length_distribution.columns = ["fraction"]
                    length_distribution = length_distribution.reset_index()
                    length_distribution["allele"] = allele
                    length_distribution["cutoff_fraction"] = cutoff_fraction
                    length_distribution["cutoff_count"] = cutoff_count
                    length_distribution = length_distribution[[
                        "allele",
                        "cutoff_fraction",
                        "cutoff_count",
                        "length",
                        "fraction"
                    ]].sort_values(["cutoff_fraction", "length"])
                    length_distributions.append(length_distribution)

        if frequency_matrices is not None:
            frequency_matrices = pandas.concat(
                frequency_matrices, ignore_index=True)

        if length_distributions is not None:
            length_distributions = pandas.concat(
                length_distributions, ignore_index=True)

        if motif_summary:
            return {
                'frequency_matrices': frequency_matrices,
                'length_distributions': length_distributions,
            }
        return {}

    def clear_calibration_fast_cache(self):
        return calibration_sizing.clear_calibration_fast_cache(self)

    def calibrate_percentile_ranks_fast(
            self,
            peptides,
            alleles,
            bins=None,
            motif_summary=False,
            summary_top_peptide_fractions=(0.001,),
            allele_batch_size="auto",
            peptide_batch_size="auto",
            num_workers_per_gpu=1,
            device=None,
            verbose=False):
        """GPU-hoisted calibration for many alleles sharing a peptide set.

        Drop-in replacement for the bulk of ``calibrate_percentile_ranks``
        when calibration is dominated by per-allele Python dispatch (the
        pan-allele full-universe calibration workload, which sweeps
        ~20k alleles across the same peptide set). Two structural changes:

        1. **Precompute peptide-stage activations once per network.** The
           network's locally-connected + peptide-dense + early-batchnorm
           layers are identity across alleles; we forward the calibration
           peptides through that stage exactly once per network and
           cache the output tensor on device.
        2. **Batch ``allele_batch_size`` alleles per forward.** A single
           ``forward_from_peptide_stage`` call replaces
           ``allele_batch_size`` separate ``model.predict`` calls —
           amortizing all the small kernel-launch + Python-dispatch
           overhead that was dominating wall-clock time on the pan-allele
           full-universe calibration. Effective batch size per forward:
           ``allele_batch_size * peptide_batch_size``, sized so it fits
           comfortably in an A100's 80 GB.

        Semantics-preserving w.r.t. ``calibrate_percentile_ranks``: same
        peptides → same per-network IC50 predictions (numerically equivalent,
        ≤1e-12, when the network is deterministic — batched vs per-allele
        matmul scheduling can differ in the last ULPs) → same geometric-mean
        ensemble aggregation → same ``PercentRankTransform.fit`` per allele.
        Only the schedule of Python dispatch and GPU kernel launches
        changes.

        Caching note: this method memoizes the peptide-stage activations on
        the predictor instance across calls.
        That cache is keyed on network identity, not weight content, so do not
        mutate the predictor's network weights in place between calls on the
        same instance without first calling ``clear_calibration_fast_cache()``.

        Only handles pan-allele models. Mass-spec-only models and
        ``class1_presentation_predictor`` should keep using the slower
        per-allele path.

        Parameters
        ----------
        peptides : sequence of string or EncodableSequences
        alleles : sequence of string — already-normalized allele names
            (canonicalization is the caller's responsibility for speed).
        bins : sequence of bin edges in IC50 space (default: 999 log-spaced
            IC50 bins, matching ``calibrate_percentile_ranks``). Scalar
            integer ``numpy.histogram`` bin counts are rejected in this fast
            path because they imply data-dependent edges per allele, while
            the batched GPU histogram uses one explicit edge vector for the
            whole allele chunk.
        motif_summary : bool — populate frequency matrices + length
            distributions, same format as ``calibrate_percentile_ranks``.
        summary_top_peptide_fractions : iterable of float — only used
            when ``motif_summary=True``.
        allele_batch_size : int — how many alleles share a single forward
            through the merge + main dense. 64 is a reasonable default
            on an A100-80GB with the release pan-allele arch + the 800k
            peptide calibration set (peak VRAM ~ allele_batch_size *
            peptide_batch_size * 4 bytes per hidden unit).
        peptide_batch_size : int — peptide chunk size on device.
        device : str or torch.device — defaults to CUDA if available.
        verbose : bool — per-batch timing to stdout.

        Returns
        -------
        dict (same schema as ``calibrate_percentile_ranks``). Also
        populates ``self.allele_to_percent_rank_transform[allele]`` for
        every allele in ``alleles``.
        """
        import torch

        from .encodable_sequences import EncodableSequences
        from .allele_encoding import AlleleEncoding
        from .regression_target import to_ic50
        from .percent_rank_transform import PercentRankTransform

        if not self.class1_pan_allele_models:
            raise ValueError(
                "calibrate_percentile_ranks_fast is pan-allele-only; "
                "this predictor has no pan-allele models."
            )

        if bins is None:
            bins = to_ic50(numpy.linspace(1, 0, 1000))
        bin_edges_array = numpy.asarray(bins)
        if bin_edges_array.ndim == 0:
            raise ValueError(
                "calibrate_percentile_ranks_fast requires explicit IC50 bin "
                "edges. Scalar integer bins use numpy.histogram's "
                "data-dependent per-allele edges in the legacy path and "
                "cannot be represented by one batched GPU edge vector. Use "
                "bins=to_ic50(numpy.linspace(1, 0, n_bins + 1)) for fixed "
                "log-space IC50 edges."
            )
        if bin_edges_array.ndim != 1 or bin_edges_array.shape[0] < 2:
            raise ValueError(
                "calibrate_percentile_ranks_fast requires a one-dimensional "
                "sequence of at least two IC50 bin edges."
            )

        if device is None:
            from .common import get_pytorch_device
            device = get_pytorch_device()
        else:
            device = torch.device(device)

        encoded_peptides = EncodableSequences.create(peptides)
        n_peptides = len(encoded_peptides.sequences)
        if n_peptides == 0:
            raise ValueError("No peptides supplied for calibration")
        alleles = list(alleles)
        if not alleles:
            raise ValueError("No alleles supplied for calibration")

        master = self.master_allele_encoding
        unknown = [a for a in alleles if a not in master.allele_to_index]
        if unknown:
            raise KeyError(
                "calibrate_percentile_ranks_fast: %d allele(s) not in the "
                "predictor's master encoding (first 5: %s)" % (
                    len(unknown), unknown[:5],
                )
            )
        allele_idx_np = numpy.array(
            [master.allele_to_index[a] for a in alleles], dtype=numpy.int64,
        )

        # Per-network prep: ensure allele representations are wired up,
        # move the eager (uncompiled) model to device, cache the
        # peptide-stage output.
        networks = self.class1_pan_allele_models

        # Cross-task cached_stages reuse: when calibrate is sharded into
        # many tasks, each task is a separate ``calibrate_percentile_ranks_fast``
        # call inside the same worker process and re-builds
        # ``cached_stages`` from scratch even though the peptide universe
        # and the predictor are identical. Cache the built tensors on
        # ``calibration_sizing.calibration_fast_cache`` and skip the rebuild whenever
        # the signature matches. ``forward_peptide_stage`` runs once per
        # peptide-batch per network, so for the production workload
        # (~400k peptides × many allele chunks/worker) each saved rebuild
        # is a couple-second-per-task win that compounds.
        #
        # The signature uses a SHA-256 fingerprint of the full peptide
        # list rather than (count, first, last) — the latter would
        # silently reuse a stale cache for two distinct peptide sets that
        # share count/first/last (rare but real, and the failure mode is
        # wrong PercentRankTransforms with no error). It intentionally
        # omits ``peptide_batch_size``: that size only controls how the
        # cache tensor is filled, not its contents or shape.
        #
        # Cache validity also assumes the predictor's network weights are not
        # mutated in place between calls on the same instance: the key tracks
        # network *identity*, not weight content (see
        # ``calibration_sizing.calibration_stage_cache_signature`` for why a
        # content fingerprint is unreliable here), so a caller that re-fits a
        # network in place must
        # call ``clear_calibration_fast_cache()`` to avoid reusing stale
        # stages. Adding/removing/replacing whole models is already safe (new
        # network objects -> new ids -> cache miss).
        cache = calibration_sizing.calibration_fast_cache(self)
        cache_signature = calibration_sizing.calibration_stage_cache_signature(
            encoded_peptides, networks, device,
        )
        cache_hit = (
            cache.stage_signature == cache_signature
            and cache.cached_stages is not None
        )

        if allele_batch_size in (None, "auto") or peptide_batch_size in (None, "auto"):
            # Resolve once, using the first network as the architecture
            # probe — all ensembles we serve have homogeneous layer
            # sizing, so one probe is sufficient.
            probe_net_obj = networks[0]
            probe_net = probe_net_obj.network(borrow=True)
            probe_net.to(device)
            # Probe the *actual* peptide_stage output dim so the cache
            # estimate doesn't fall back to the encoding-shape floor.
            probed_stage_dim = calibration_sizing.probe_peptide_stage_dim(
                probe_net_obj, encoded_peptides, device,
            )
            sub_networks = getattr(probe_net, "networks", None)
            num_sub_networks_probed = (
                len(sub_networks) if sub_networks is not None else 1
            )
            # Mixed pin/auto: when the user pinned one axis, pass it through so
            # the auto axis is sized against the *actual* pinned value rather
            # than an assumed auto one (which would underestimate peak VRAM).
            pinned_peptide = (
                None if peptide_batch_size in (None, "auto")
                else int(peptide_batch_size))
            pinned_allele = (
                None if allele_batch_size in (None, "auto")
                else int(allele_batch_size))
            auto_peptide, auto_allele = calibration_sizing.auto_size_calibration_batches(
                probe_net, device, n_peptides, len(alleles),
                num_workers_per_gpu=num_workers_per_gpu,
                # If the peptide-stage cache is already resident in this
                # worker, current CUDA free memory has already been reduced
                # by that cache. Counting it again here makes later shards
                # collapse to tiny fallback batches.
                num_cached_networks=0 if cache_hit else len(networks),
                peptide_stage_dim=probed_stage_dim,
                num_sub_networks=num_sub_networks_probed,
                fixed_peptide_batch=pinned_peptide,
                fixed_allele_batch=pinned_allele,
            )
            if peptide_batch_size in (None, "auto"):
                peptide_batch_size = auto_peptide
            if allele_batch_size in (None, "auto"):
                allele_batch_size = auto_allele
            if verbose:
                print(
                    "calibrate_percentile_ranks_fast auto-sized: "
                    f"peptide_batch={peptide_batch_size}, "
                    f"allele_batch={allele_batch_size} "
                    f"(workers_per_gpu={num_workers_per_gpu}, "
                    f"cached_networks={len(networks)}, "
                    f"sub_networks={num_sub_networks_probed}, "
                    f"probed_stage_dim={probed_stage_dim})"
                )
        peptide_batch_size = int(peptide_batch_size)
        allele_batch_size = int(allele_batch_size)

        if cache_hit:
            cached_stages = cache.cached_stages
        else:
            # Drop stale cache before building the new one — releases
            # the previous peptide-stage tensor's VRAM before we
            # allocate the next, which matters when the per-call
            # peptide universe size changes (e.g. a smoke test
            # followed by the production calibrate in the same worker).
            cache.stage_signature = None
            cache.cached_stages = None
            cache.motif_signature = None
            cache.motif_state = None
            cached_stages = []
            for net_obj in networks:
                (_, allele_reps) = net_obj.allele_encoding_to_network_input(
                    AlleleEncoding(alleles=[], borrow_from=master),
                )
                net_obj.set_allele_representations(allele_reps)
                model = net_obj.network(borrow=True)
                model.to(device)
                model.eval()
                peptide_input = net_obj.peptides_to_network_input(encoded_peptides)
                peptide_is_indices = net_obj.uses_peptide_torch_encoding()
                # Pre-size the cache tensor by probing one row's stage_dim
                # then write each chunk in-place. The previous build path
                # used append+torch.cat, which transiently held *both* the
                # per-chunk parts and the new contiguous tensor at once —
                # a 2× VRAM peak (~13 GB extra on the production 400k
                # peptide × 8-subnet ensemble) that pushed --max-workers-
                # per-gpu auto into OOM territory. Pre-sized fill keeps
                # the peak at 1× the cache.
                with torch.no_grad():
                    probe_chunk = peptide_input[:1]
                    probe_keep_int = (
                        peptide_is_indices
                        and probe_chunk.ndim == 2
                        and numpy.issubdtype(probe_chunk.dtype, numpy.integer)
                    )
                    if probe_keep_int:
                        probe_tensor = torch.from_numpy(probe_chunk).to(device)
                    else:
                        probe_tensor = torch.from_numpy(
                            numpy.asarray(probe_chunk, dtype=numpy.float32),
                        ).to(device)
                    probe_stage = model.forward_peptide_stage(probe_tensor)
                    stage_dim_cached = int(probe_stage.shape[-1])
                    stage_dtype = probe_stage.dtype
                cached_tensor = torch.empty(
                    n_peptides, stage_dim_cached,
                    dtype=stage_dtype, device=device,
                )
                with torch.no_grad():
                    for start in range(0, n_peptides, peptide_batch_size):
                        end = min(start + peptide_batch_size, n_peptides)
                        chunk = peptide_input[start:end]
                        keep_int = (
                            peptide_is_indices
                            and chunk.ndim == 2
                            and numpy.issubdtype(chunk.dtype, numpy.integer)
                        )
                        if keep_int:
                            tensor = torch.from_numpy(chunk).to(device)
                        else:
                            tensor = torch.from_numpy(
                                numpy.asarray(chunk, dtype=numpy.float32),
                            ).to(device)
                        cached_tensor[start:end] = model.forward_peptide_stage(
                            tensor,
                        )
                cached_stages.append((net_obj, model, cached_tensor))
            cache.cached_stages = cached_stages
            cache.stage_signature = cache_signature
            del peptide_input

        log50000 = float(numpy.log(50000.0))
        n_alleles = len(alleles)
        frequency_matrices = [] if motif_summary else None
        length_distributions = [] if motif_summary else None
        # CPU parity tests compare against the legacy per-allele forward.
        # Preserve that matmul order on CPU; GPU keeps the lower-memory
        # factored cartesian path used by the production calibration run.
        exact_cartesian_forward = device.type == "cpu"
        # Bin edges for the on-device batched histogram fit. Materialize
        # once outside the per-chunk loop — same edges across alleles.
        calibration_float_dtype = (
            torch.float32 if device.type == "mps" else torch.float64
        )
        bins_tensor = torch.as_tensor(
            bin_edges_array, dtype=calibration_float_dtype, device=device,
        )

        # Hoist the per-allele dedup + length-bucket + AA-encoding work
        # out of the chunk loop. The peptide universe is identical across
        # alleles so this only needs to run once per calibrate call —
        # and across the many tasks per worker that share the same
        # peptide universe, we reuse the device-resident state via the
        # same signature key as ``cached_stages`` above.
        if motif_summary:
            motif_signature = (cache_signature, "motif")
            if cache.motif_signature == motif_signature:
                motif_state = cache.motif_state
            else:
                motif_state = prepare_motif_summary_state_gpu(
                    encoded_peptides, device,
                )
                cache.motif_state = motif_state
                cache.motif_signature = motif_signature
        else:
            motif_state = None

        for abatch_start in range(0, n_alleles, allele_batch_size):
            abatch_end = min(abatch_start + allele_batch_size, n_alleles)
            a_size = abatch_end - abatch_start
            batch_alleles = alleles[abatch_start:abatch_end]
            batch_idx = torch.from_numpy(
                allele_idx_np[abatch_start:abatch_end],
            ).to(device)

            # log-space accumulator across ensemble members:
            # (a_size, n_peptides).
            # Same aggregation scheme as the existing ensemble code:
            # arithmetic mean of per-network log(IC50) = geometric mean
            # of per-network IC50. Use fp64 where supported so the
            # aggregation matches the legacy path bit-for-bit; fall
            # back to fp32 on MPS (which has no fp64) — drift there is
            # ~1e-6 in log-IC50, well below histogram-bin resolution.
            accum_dtype = (
                torch.float32 if device.type == "mps" else torch.float64
            )
            log_ic50_sum = torch.zeros(
                a_size, n_peptides, dtype=accum_dtype, device=device,
            )
            ensemble_member_count = 0

            for (_, model, peptide_stage) in cached_stages:
                model_member_count = None
                with torch.no_grad():
                    for start in range(0, n_peptides, peptide_batch_size):
                        end = min(start + peptide_batch_size, n_peptides)
                        p_chunk = peptide_stage[start:end]  # (chunk_n, d)
                        network_output = cartesian_network_output(
                            model,
                            p_chunk,
                            batch_idx,
                            exact_forward=exact_cartesian_forward,
                        )
                        (
                            log_ic50_chunk,
                            chunk_member_count,
                        ) = cartesian_output_log_ic50_sum(
                            network_output, model, log50000, accum_dtype,
                        )
                        if model_member_count is None:
                            model_member_count = chunk_member_count
                        elif model_member_count != chunk_member_count:
                            raise ValueError(
                                "cartesian network output member count changed "
                                "within one calibration batch"
                            )
                        log_ic50_sum[:, start:end] += log_ic50_chunk

                if model_member_count is None:
                    raise ValueError("No peptide batches were evaluated")
                ensemble_member_count += model_member_count

            if ensemble_member_count < 1:
                raise ValueError("No ensemble members were evaluated")
            log_mean = log_ic50_sum / float(ensemble_member_count)
            ic50_device = torch.exp(log_mean)  # (a_size, n_peptides) on device
            # Batched torch fit replaces the per-allele numpy.histogram
            # loop that dominated calibrate wall time. One bucketize +
            # one scatter_add covers all 30 alleles in the chunk; the
            # final per-allele cdf is materialized as numpy only at
            # storage time so the persistent format is unchanged.
            transforms = PercentRankTransform.fit_batch_torch(
                ic50_device, bins_tensor,
            )
            for local_i, allele in enumerate(batch_alleles):
                self.allele_to_percent_rank_transform[allele] = transforms[local_i]
            if motif_summary:
                # All motif-summary work now runs on device — topk +
                # scatter_add for AA frequency matrices, topk + scatter_add
                # bincount for length distributions. Pandas only assembles
                # final per-row schema at chunk-end. This replaces the
                # per-allele DataFrame.drop_duplicates().groupby().nsmallest()
                # block that dominated calibrate wall time after the
                # PercentRankTransform.fit GPU port.
                chunk_freq, chunk_ld = motif_summary_chunk_gpu(
                    ic50_device,
                    motif_state,
                    summary_top_peptide_fractions,
                    batch_alleles,
                )
                frequency_matrices.extend(chunk_freq)
                length_distributions.extend(chunk_ld)
            if verbose:
                print(
                    "calibrate_percentile_ranks_fast: "
                    f"{abatch_end}/{n_alleles} alleles done"
                )

        if motif_summary:
            return {
                "frequency_matrices": pandas.concat(
                    frequency_matrices, ignore_index=True,
                ),
                "length_distributions": pandas.concat(
                    length_distributions, ignore_index=True,
                ),
            }
        return {}

    def model_select(
            self,
            score_function,
            alleles=None,
            min_models=1,
            max_models=10000):
        """
        Perform model selection using a user-specified scoring function.

        This works only with allele-specific models, not pan-allele models.

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
        return model_selection.model_select(
            self,
            Class1AffinityPredictor,
            score_function,
            alleles=alleles,
            min_models=min_models,
            max_models=max_models,
        )
