import collections
import time
import hashlib
import json
from os.path import join, exists
from six import string_types
import logging
import warnings
import sys

import numpy
import pandas
from numpy.testing import assert_equal

import mhcnames

from ..encodable_sequences import EncodableSequences
from ..downloads import get_path
from ..common import random_peptides
from ..percent_rank_transform import PercentRankTransform
from ..regression_target import to_ic50

from .class1_neural_network import Class1NeuralNetwork


class Class1AffinityPredictor(object):
    """
    High-level interface for peptide/MHC I binding affinity prediction.
    
    This is the class most users will want to use.
    
    This class delegates to one or more `Class1NeuralNetwork` instances.
    It supports prediction across multiple alleles using ensembles of single-
    or pan-allele predictors.
    """
    def __init__(
            self,
            allele_to_allele_specific_models=None,
            class1_pan_allele_models=None,
            allele_to_pseudosequence=None,
            manifest_df=None,
            allele_to_percent_rank_transform=None):
        """
        Parameters
        ----------
        allele_to_allele_specific_models : dict of string -> list of Class1NeuralNetwork
            Ensemble of single-allele models to use for each allele. 
        
        class1_pan_allele_models : list of Class1NeuralNetwork
            Ensemble of pan-allele models.
        
        allele_to_pseudosequence : dict of string -> string
            Required only if class1_pan_allele_models is specified.
        
        manifest_df : pandas.DataFrame, optional
            Must have columns: model_name, allele, config_json, model.
            Only required if you want to update an existing serialization of a
            Class1AffinityPredictor. Otherwise this dataframe will be generated
            automatically based on the supplied models.

        allele_to_percent_rank_transform : dict of string -> PercentRankTransform, optional
            PercentRankTransform instances to use for each allele
        """

        if allele_to_allele_specific_models is None:
            allele_to_allele_specific_models = {}
        if class1_pan_allele_models is None:
            class1_pan_allele_models = []

        if class1_pan_allele_models:
            assert allele_to_pseudosequence, "Pseudosequences required"

        self.allele_to_allele_specific_models = allele_to_allele_specific_models
        self.class1_pan_allele_models = class1_pan_allele_models
        self.allele_to_pseudosequence = allele_to_pseudosequence

        if manifest_df is None:
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
            manifest_df = pandas.DataFrame(
                rows,
                columns=["model_name", "allele", "config_json", "model"])
        self.manifest_df = manifest_df

        if not allele_to_percent_rank_transform:
            allele_to_percent_rank_transform = {}
        self.allele_to_percent_rank_transform = allele_to_percent_rank_transform

    @property
    def neural_networks(self):
        """
        List of the neural networks in the ensemble.

        Returns
        -------
        list of Class1NeuralNetwork
        """
        result = []
        for models in self.allele_to_allele_specific_models.values():
            result.extend(models)
        result.extend(self.class1_pan_allele_models)
        return result

    @classmethod
    def merge(cls, predictors):
        """
        Merge the ensembles of two or more Class1AffinityPredictor instances.

        Note: the resulting merged predictor will NOT have calibrated percentile
        ranks. Call calibrate_percentile_ranks() on it if these are needed.

        Parameters
        ----------
        predictors : sequence of Class1AffinityPredictor

        Returns
        -------
        Class1AffinityPredictor

        """
        assert len(predictors) > 0
        if len(predictors) == 1:
            return predictors[0]

        allele_to_allele_specific_models = collections.defaultdict(list)
        class1_pan_allele_models = []
        allele_to_pseudosequence = predictors[0].allele_to_pseudosequence

        for predictor in predictors:
            for (allele, networks) in (
                    predictor.allele_to_allele_specific_models.items()):
                allele_to_allele_specific_models[allele].extend(networks)
            class1_pan_allele_models.extend(
                predictor.class1_pan_allele_models)

        return Class1AffinityPredictor(
            allele_to_allele_specific_models=allele_to_allele_specific_models,
            class1_pan_allele_models=class1_pan_allele_models,
            allele_to_pseudosequence=allele_to_pseudosequence
        )

    @property
    def num_networks(self):
        return self.manifest_df.shape[0]


    @property
    def supported_alleles(self):
        """
        Alleles for which predictions can be made.
        
        Returns
        -------
        list of string
        """
        result = set(self.allele_to_allele_specific_models)
        if self.allele_to_pseudosequence:
            result = result.union(self.allele_to_pseudosequence)
        return sorted(result)

    @property
    def supported_peptide_lengths(self):
        """
        (minimum, maximum) lengths of peptides supported by *all models*,
        inclusive.

        Returns
        -------
        (int, int) tuple

        """
        models = list(self.class1_pan_allele_models)
        for allele_models in self.allele_to_allele_specific_models.values():
            models.extend(allele_models)
        length_ranges = [model.supported_peptide_lengths for model in models]
        return (
            max(lower for (lower, upper) in length_ranges),
            min(upper for (lower, upper) in length_ranges))

    def save(self, models_dir, model_names_to_write=None):
        """
        Serialize the predictor to a directory on disk.
        
        The serialization format consists of a file called "manifest.csv" with
        the configurations of each Class1NeuralNetwork, along with per-network
        files giving the model weights. If there are pan-allele predictors in
        the ensemble, the allele pseudosequences are also stored in the
        directory.
        
        Parameters
        ----------
        models_dir : string
            Path to directory
            
        model_names_to_write : list of string, optional
            Only write the weights for the specified models. Useful for
            incremental updates during training.
        """
        num_models = len(self.class1_pan_allele_models) + sum(
            len(v) for v in self.allele_to_allele_specific_models.values())
        assert len(self.manifest_df) == num_models, (
            "Manifest seems out of sync with models: %d vs %d entries" % (
                len(self.manifest_df), num_models))

        if model_names_to_write is None:
            # Write all models
            model_names_to_write = self.manifest_df.model_name.values

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
            Maximum number of Class1NeuralNetwork instances to load

        Returns
        -------
        Class1AffinityPredictor
        """
        if models_dir is None:
            models_dir = get_path("models_class1", "models")

        manifest_path = join(models_dir, "manifest.csv")
        manifest_df = pandas.read_csv(manifest_path, nrows=max_models)

        allele_to_allele_specific_models = collections.defaultdict(list)
        class1_pan_allele_models = []
        all_models = []
        for (_, row) in manifest_df.iterrows():
            weights_filename = Class1AffinityPredictor.weights_path(
                models_dir, row.model_name)
            weights = Class1AffinityPredictor.load_weights(weights_filename)
            config = json.loads(row.config_json)
            model = Class1NeuralNetwork.from_config(config, weights=weights)
            if row.allele == "pan-class1":
                class1_pan_allele_models.append(model)
            else:
                allele_to_allele_specific_models[row.allele].append(model)
            all_models.append(model)

        manifest_df["model"] = all_models

        pseudosequences = None
        if exists(join(models_dir, "pseudosequences.csv")):
            pseudosequences = pandas.read_csv(
                join(models_dir, "pseudosequences.csv"),
                index_col="allele").to_dict()

        allele_to_percent_rank_transform = {}
        percent_ranks_path = join(models_dir, "percent_ranks.csv")
        if exists(percent_ranks_path):
            percent_ranks_df = pandas.read_csv(percent_ranks_path, index_col=0)
            for allele in percent_ranks_df.columns:
                allele_to_percent_rank_transform[allele] = (
                    PercentRankTransform.from_series(percent_ranks_df[allele]))

        logging.info(
            "Loaded %d class1 pan allele predictors, %d pseudosequences, "
            "%d percent rank distributions, and %d allele specific models: %s" % (
                len(class1_pan_allele_models),
                len(pseudosequences) if pseudosequences else 0,
                len(allele_to_percent_rank_transform),
                sum(len(v) for v in allele_to_allele_specific_models.values()),
                ", ".join(
                    "%s (%d)" % (allele, len(v))
                    for (allele, v)
                    in sorted(allele_to_allele_specific_models.items()))))

        result = Class1AffinityPredictor(
            allele_to_allele_specific_models=allele_to_allele_specific_models,
            class1_pan_allele_models=class1_pan_allele_models,
            allele_to_pseudosequence=pseudosequences,
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
            architecture_hyperparameters,
            allele,
            peptides,
            affinities,
            models_dir_for_save=None,
            verbose=1,
            progress_preamble=""):
        """
        Fit one or more allele specific predictors for a single allele using a
        single neural network architecture.
        
        The new predictors are saved in the Class1AffinityPredictor instance
        and will be used on subsequent calls to `predict`.
        
        Parameters
        ----------
        n_models : int
            Number of neural networks to fit
        
        architecture_hyperparameters : dict 
               
        allele : string
        
        peptides : EncodableSequences or list of string
        
        affinities : list of float
            nM affinities
        
        models_dir_for_save : string, optional
            If specified, the Class1AffinityPredictor is (incrementally) written
            to the given models dir after each neural network is fit.
        
        verbose : int
            Keras verbosity

        progress_preamble : string
            Optional string of information to include in each progress update

        Returns
        -------
        list of Class1NeuralNetwork
        """

        allele = mhcnames.normalize_allele_name(allele)
        models = self._fit_predictors(
            n_models=n_models,
            architecture_hyperparameters=architecture_hyperparameters,
            peptides=peptides,
            affinities=affinities,
            allele_pseudosequences=None,
            verbose=verbose,
            progress_preamble=progress_preamble)

        if allele not in self.allele_to_allele_specific_models:
            self.allele_to_allele_specific_models[allele] = []

        models_list = []
        for (i, model) in enumerate(models):
            model_name = self.model_name(allele, i)
            models_list.append(model)  # models is a generator
            row = pandas.Series(collections.OrderedDict([
                ("model_name", model_name),
                ("allele", allele),
                ("config_json", json.dumps(model.get_config())),
                ("model", model),
            ])).to_frame().T
            self.manifest_df = pandas.concat(
                [self.manifest_df, row], ignore_index=True)
            self.allele_to_allele_specific_models[allele].append(model)
            if models_dir_for_save:
                self.save(
                    models_dir_for_save, model_names_to_write=[model_name])
        return models_list

    def fit_class1_pan_allele_models(
            self,
            n_models,
            architecture_hyperparameters,
            alleles,
            peptides,
            affinities,
            models_dir_for_save=None,
            verbose=1,
            progress_preamble=""):
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
            Allele names (not pseudosequences) corresponding to each peptide 
        
        peptides : EncodableSequences or list of string
        
        affinities : list of float
            nM affinities
        
        models_dir_for_save : string, optional
            If specified, the Class1AffinityPredictor is (incrementally) written
            to the given models dir after each neural network is fit.
        
        verbose : int
            Keras verbosity

        progress_preamble : string
            Optional string of information to include in each progress update

        Returns
        -------
        list of Class1NeuralNetwork
        """

        alleles = pandas.Series(alleles).map(mhcnames.normalize_allele_name)
        allele_pseudosequences = alleles.map(self.allele_to_pseudosequence)

        models = self._fit_predictors(
            n_models=n_models,
            architecture_hyperparameters=architecture_hyperparameters,
            peptides=peptides,
            affinities=affinities,
            allele_pseudosequences=allele_pseudosequences,
            verbose=verbose,
            progress_preamble=progress_preamble)

        models_list = []
        for (i, model) in enumerate(models):
            model_name = self.model_name("pan-class1", i)
            models_list.append(model)  # models is a generator
            self.class1_pan_allele_models.append(model)
            row = pandas.Series(collections.OrderedDict([
                ("model_name", model_name),
                ("allele", "pan-class1"),
                ("config_json", json.dumps(model.get_config())),
                ("model", model),
            ])).to_frame().T
            self.manifest_df = pandas.concat(
                [self.manifest_df, row], ignore_index=True)
            if models_dir_for_save:
                self.save(
                    models_dir_for_save, model_names_to_write=[model_name])
        return models_list

    def _fit_predictors(
            self,
            n_models,
            architecture_hyperparameters,
            peptides,
            affinities,
            allele_pseudosequences,
            verbose=1,
            progress_preamble = ""):
        """
        Private helper method
        
        Parameters
        ----------
        n_models : int
        architecture_hyperparameters : dict
        peptides : EncodableSequences or list of string
        affinities : list of float
        allele_pseudosequences : EncodableSequences or list of string
        verbose : int
        progress_preamble : string
            Optional string of information to include in each progress update

        Returns
        -------
        generator of Class1NeuralNetwork
        """
        encodable_peptides = EncodableSequences.create(peptides)
        for i in range(n_models):
            logging.info("Training model %d / %d" % (i + 1, n_models))
            model = Class1NeuralNetwork(**architecture_hyperparameters)
            model.fit(
                encodable_peptides,
                affinities,
                allele_pseudosequences=allele_pseudosequences,
                verbose=verbose,
                progress_preamble=progress_preamble)
            yield model

    def calibrate_percentile_ranks(
            self,
            peptides=None,
            num_peptides_per_length=int(1e5),
            alleles=None,
            bins=None,
            quiet=False):
        """
        Compute the cumulative distribution of ic50 values for a set of alleles
        over a large universe of random peptides, to enable computing quantiles in
        this distribution later.

        Parameters
        ----------
        peptides : sequence of string, optional
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
        quiet : boolean
            If False (default), status updates will be printed to stdout.
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

        if quiet:
            def msg(s):
                pass
        else:
            def msg(s):
                print(s)
                sys.stdout.flush()

        encoded_peptides = EncodableSequences.create(peptides)
        for (i, allele) in enumerate(alleles):
            msg("Calibrating percentile ranks for allele %03d/%03d: %s" % (
                i + 1, len(alleles), allele))
            start = time.time()
            predictions = self.predict(encoded_peptides, allele=allele)
            msg("Generated %d predictions in %0.2f sec." % (
                len(predictions), time.time() - start))
            transform = PercentRankTransform()
            transform.fit(predictions, bins=bins)
            self.allele_to_percent_rank_transform[allele] = transform
            msg("Done calibrating allele %s in %0.2f sec." % (
                allele, time.time() - start))

    def percentile_ranks(self, affinities, allele=None, alleles=None, throw=True):
        """
        Return percentile ranks for the given ic50 affinities and alleles.

        The 'allele' and 'alleles' argument are as in the predict() method.
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

    def predict(self, peptides, alleles=None, allele=None, throw=True):
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
        peptides : EncodableSequences or list of string
        alleles : list of string
        allele : string
        throw : boolean
            If True, a ValueError will be raised in the case of unsupported
            alleles or peptide lengths. If False, a warning will be logged and
            the predictions for the unsupported alleles or peptides will be NaN.

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
        )
        return df.prediction.values

    def predict_to_dataframe(
            self,
            peptides,
            alleles=None,
            allele=None,
            throw=True,
            include_individual_model_predictions=False,
            include_percentile_ranks=True):
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
        peptides : EncodableSequences or list of string
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

        Returns
        -------
        pandas.DataFrame of predictions
        """
        if isinstance(peptides, string_types):
            raise TypeError("peptides must be a list or array, not a string")
        if isinstance(alleles, string_types):
            raise TypeError("alleles must be a list or array, not a string")
        if allele is not None:
            if alleles is not None:
                raise ValueError("Specify exactly one of allele or alleles")
            alleles = [allele] * len(peptides)

        alleles = numpy.array(alleles)
        peptides = EncodableSequences.create(peptides)

        df = pandas.DataFrame({
            'peptide': peptides.sequences,
            'allele': alleles,
        })
        df["normalized_allele"] = df.allele.map(
            mhcnames.normalize_allele_name)

        (min_peptide_length, max_peptide_length) = (
            self.supported_peptide_lengths)
        df["supported_peptide_length"] = (
            (df.peptide.str.len() >= min_peptide_length) &
            (df.peptide.str.len() <= max_peptide_length))
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

        if self.class1_pan_allele_models:
            unsupported_alleles = [
                allele for allele in
                df.normalized_allele.unique()
                if allele not in self.allele_to_pseudosequence
            ]
            if unsupported_alleles:
                msg = (
                    "No pseudosequences for allele(s): %s.\n"
                    "Supported alleles: %s" % (
                        " ".join(unsupported_alleles),
                        " ".join(sorted(self.allele_to_pseudosequence))))
                logging.warning(msg)
                if throw:
                    raise ValueError(msg)
            mask = df.supported_peptide_length
            if mask.sum() > 0:
                masked_allele_pseudosequences = (
                    df.ix[mask].normalized_allele.map(
                        self.allele_to_pseudosequence))
                masked_peptides = peptides.sequences[mask]
                for (i, model) in enumerate(self.class1_pan_allele_models):
                    df.loc[mask, "model_pan_%d" % i] = model.predict(
                        masked_peptides,
                        allele_pseudosequences=masked_allele_pseudosequences)

        if self.allele_to_allele_specific_models:
            query_alleles = df.normalized_allele.unique()
            unsupported_alleles = [
                allele for allele in query_alleles
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
            for allele in query_alleles:
                models = self.allele_to_allele_specific_models.get(allele, [])
                mask = (
                    (df.normalized_allele == allele) &
                    df.supported_peptide_length).values
                if mask.all():
                    # Common case optimization
                    for (i, model) in enumerate(models):
                        df["model_single_%d" % i] = model.predict(peptides)
                elif mask.sum() > 0:
                    allele_peptides = EncodableSequences.create(
                        df.ix[mask].peptide.values)
                    for (i, model) in enumerate(models):
                        df.loc[
                            mask, "model_single_%d" % i
                        ] = model.predict(allele_peptides)

        # Geometric mean
        df_predictions = df[
            [c for c in df.columns if c.startswith("model_")]
        ]
        logs = numpy.log(df_predictions)
        log_means = logs.mean(1)
        df["prediction"] = numpy.exp(log_means)
        df["prediction_low"] = numpy.exp(logs.quantile(0.05, axis=1))
        df["prediction_high"] = numpy.exp(logs.quantile(0.95, axis=1))

        if include_individual_model_predictions:
            columns = sorted(df.columns, key=lambda c: c.startswith('model_'))
        else:
            columns = [
                c for c in df.columns if c not in df_predictions.columns
            ]
        columns.remove("normalized_allele")
        columns.remove("supported_peptide_length")

        if include_percentile_ranks:
            if self.allele_to_percent_rank_transform:
                df["prediction_percentile"] = self.percentile_ranks(
                    df.prediction,
                    alleles=df.normalized_allele.values,
                    throw=throw)
                columns.append("prediction_percentile")
            else:
                warnings.warn("No percentile rank information available.")
        return df[columns].copy()

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
