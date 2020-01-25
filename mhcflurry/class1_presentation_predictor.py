from __future__ import print_function

from os.path import join, exists, abspath
from os import mkdir, environ
from socket import gethostname
from getpass import getuser

import time
import collections
import json
import hashlib
import logging
from six import string_types

import numpy
import pandas
import sklearn
import sklearn.linear_model

import mhcnames

try:
    import tqdm
except ImportError:
    tdqm = None

from .version import __version__
from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_cleavage_predictor import Class1CleavagePredictor
from .class1_neural_network import DEFAULT_PREDICT_BATCH_SIZE
from .encodable_sequences import EncodableSequences
from .regression_target import from_ic50, to_ic50
from .multiple_allele_encoding import MultipleAlleleEncoding
from .downloads import get_default_class1_presentation_models_dir
from .common import load_weights


MAX_ALLELES_PER_SAMPLE = 6
PREDICT_BATCH_SIZE = DEFAULT_PREDICT_BATCH_SIZE


class Class1PresentationPredictor(object):
    model_inputs = ["affinity_score", "cleavage_prediction"]

    def __init__(
            self,
            affinity_predictor=None,
            cleavage_predictor_with_flanks=None,
            cleavage_predictor_without_flanks=None,
            weights_dataframe=None,
            metadata_dataframes=None):

        self.affinity_predictor = affinity_predictor
        self.cleavage_predictor_with_flanks = cleavage_predictor_with_flanks
        self.cleavage_predictor_without_flanks = cleavage_predictor_without_flanks
        self.weights_dataframe = weights_dataframe
        self.metadata_dataframes = (
            dict(metadata_dataframes) if metadata_dataframes else {})
        self._models_cache = {}

    @property
    def supported_alleles(self):
        return self.affinity_predictor.supported_alleles

    @property
    def supported_peptide_lengths(self):
        return self.affinity_predictor.supported_peptide_lengths

    def predict_affinity(
            self,
            peptides,
            experiment_names,
            alleles,
            include_affinity_percentile=False,
            verbose=1,
            throw=True):
        df = pandas.DataFrame({
            "peptide": numpy.array(peptides, copy=False),
            "experiment_name": numpy.array(experiment_names, copy=False),
        })

        iterator = df.groupby("experiment_name")
        if verbose > 0:
            print("Predicting affinities.")
            if tqdm is not None:
                iterator = tqdm.tqdm(
                    iterator, total=df.experiment_name.nunique())

        for (experiment, sub_df) in iterator:
            predictions_df = pandas.DataFrame(index=sub_df.index)
            experiment_peptides = EncodableSequences.create(sub_df.peptide.values)
            for allele in alleles[experiment]:
                predictions_df[allele] = self.affinity_predictor.predict(
                    peptides=experiment_peptides,
                    allele=allele,
                    model_kwargs={'batch_size': PREDICT_BATCH_SIZE},
                    throw=throw)
            df.loc[
                sub_df.index, "affinity"
            ] = predictions_df.min(1).values
            df.loc[
                sub_df.index, "best_allele"
            ] = predictions_df.idxmin(1).values

            if include_affinity_percentile:
                df.loc[sub_df.index, "affinity_percentile"] = (
                    self.affinity_predictor.percentile_ranks(
                        df.loc[sub_df.index, "affinity"].values,
                        alleles=df.loc[sub_df.index, "best_allele"].values,
                        throw=False))

        return df

    def predict_cleavage(
            self, peptides, n_flanks=None, c_flanks=None, verbose=1):

        if verbose > 0:
            print("Predicting cleavage.")

        if (n_flanks is None) != (c_flanks is None):
            raise ValueError("Specify both or neither of n_flanks, c_flanks")

        if n_flanks is None:
            if self.cleavage_predictor_without_flanks is None:
                raise ValueError("No cleavage predictor without flanks")
            predictor = self.cleavage_predictor_without_flanks
            n_flanks = [""] * len(peptides)
            c_flanks = n_flanks
        else:
            if self.cleavage_predictor_with_flanks is None:
                raise ValueError("No cleavage predictor with flanks")
            predictor = self.cleavage_predictor_with_flanks

        result = predictor.predict(
            peptides=peptides,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            batch_size=PREDICT_BATCH_SIZE)

        return result

    def fit(
            self,
            targets,
            peptides,
            experiment_names,
            alleles,
            n_flanks=None,
            c_flanks=None,
            verbose=1):

        df = self.predict_affinity(
            peptides=peptides,
            experiment_names=experiment_names,
            alleles=alleles,
            verbose=verbose)
        df["affinity_score"] = from_ic50(df.affinity)
        df["target"] = numpy.array(targets, copy=False)

        if (n_flanks is None) != (c_flanks is None):
            raise ValueError("Specify both or neither of n_flanks, c_flanks")

        with_flanks_list = []
        if self.cleavage_predictor_without_flanks is not None:
            with_flanks_list.append(False)

        if n_flanks is not None and self.cleavage_predictor_with_flanks is not None:
            with_flanks_list.append(True)

        if not with_flanks_list:
            raise RuntimeError("Can't fit any models")

        if self.weights_dataframe is None:
            self.weights_dataframe = pandas.DataFrame()

        for with_flanks in with_flanks_list:
            model_name = 'with_flanks' if with_flanks else "without_flanks"
            if verbose > 0:
                print("Training variant", model_name)

            df["cleavage_prediction"] = self.predict_cleavage(
                peptides=df.peptide.values,
                n_flanks=n_flanks if with_flanks else None,
                c_flanks=c_flanks if with_flanks else None,
                verbose=verbose)

            model = self.get_model()
            if verbose > 0:
                print("Fitting LR model.")
                print(df)

            model.fit(
                X=df[self.model_inputs].values,
                y=df.target.astype(float))

            self.weights_dataframe.loc[model_name, "intercept"] = model.intercept_
            for (name, value) in zip(self.model_inputs, numpy.squeeze(model.coef_)):
                self.weights_dataframe.loc[model_name, name] = value
            self._models_cache[model_name] = model

    def get_model(self, name=None):
        if name is None or name not in self._models_cache:
            model = sklearn.linear_model.LogisticRegression(solver="lbfgs")
            if name is not None:
                row = self.weights_dataframe.loc[name]
                model.intercept_ = row.intercept
                model.coef_ = numpy.expand_dims(
                    row[self.model_inputs].values, axis=0)
        else:
            model = self._models_cache[name]
        return model

    def predict_sequences(self, alleles, sequences):
        raise NotImplementedError

    def predict(
            self,
            peptides,
            alleles,
            experiment_names=None,
            n_flanks=None,
            c_flanks=None,
            verbose=1):
        return self.predict_to_dataframe(
            peptides=peptides,
            alleles=alleles,
            experiment_names=experiment_names,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            verbose=verbose).presentation_score.values

    def predict_to_dataframe(
            self,
            peptides,
            alleles,
            experiment_names=None,
            n_flanks=None,
            c_flanks=None,
            include_affinity_percentile=False,
            verbose=1,
            throw=True):

        if isinstance(peptides, string_types):
            raise TypeError("peptides must be a list not a string")
        if isinstance(alleles, string_types):
            raise TypeError("alleles must be a list or dict")

        if isinstance(alleles, dict):
            if experiment_names is None:
                raise ValueError(
                    "experiment_names must be supplied when alleles is a dict")
        else:
            if experiment_names is not None:
                raise ValueError(
                    "alleles must be a dict when experiment_names is specified")
            alleles = numpy.array(alleles, copy=False)
            if len(alleles) > MAX_ALLELES_PER_SAMPLE:
                raise ValueError(
                    "When alleles is a list, it must have at most %d elements. "
                    "These alleles are taken to be a genotype for an "
                    "individual, and the strongest prediction across alleles "
                    "will be taken for each peptide. Note that this differs "
                    "from Class1AffinityPredictor.predict(), where alleles "
                    "is expected to be the same length as peptides."
                    % MAX_ALLELES_PER_SAMPLE)

            experiment_names = ["experiment1"] * len(peptides)
            alleles = {
                "experiment1": alleles,
            }

        df = self.predict_affinity(
            peptides=peptides,
            experiment_names=experiment_names,
            alleles=alleles,
            include_affinity_percentile=include_affinity_percentile,
            verbose=verbose,
            throw=throw)
        df["affinity_score"] = from_ic50(df.affinity)

        if (n_flanks is None) != (c_flanks is None):
            raise ValueError("Specify both or neither of n_flanks, c_flanks")

        df["cleavage_prediction"] = self.predict_cleavage(
            peptides=df.peptide.values,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            verbose=verbose)

        model_name = 'with_flanks' if n_flanks is not None else "without_flanks"
        model = self.get_model(model_name)
        df["presentation_score"] = model.predict_proba(
            df[self.model_inputs].values)[:,1]
        del df["affinity_score"]
        return df

    def save(self, models_dir):
        """
        Serialize the predictor to a directory on disk. If the directory does
        not exist it will be created.

        Parameters
        ----------
        models_dir : string
            Path to directory. It will be created if it doesn't exist.
        """

        if self.weights_dataframe is None:
            raise RuntimeError("Can't save before fitting")

        if not exists(models_dir):
            mkdir(models_dir)

        # Save underlying predictors
        self.affinity_predictor.save(join(models_dir, "affinity_predictor"))
        if self.cleavage_predictor_with_flanks is not None:
            self.cleavage_predictor_with_flanks.save(
                join(models_dir, "cleavage_predictor_with_flanks"))
        if self.cleavage_predictor_without_flanks is not None:
            self.cleavage_predictor_without_flanks.save(
                join(models_dir, "cleavage_predictor_without_flanks"))

        # Save model coefficients.
        self.weights_dataframe.to_csv(join(models_dir, "weights.csv"))

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


    @classmethod
    def load(cls, models_dir=None, max_models=None):
        """
        Deserialize a predictor from a directory on disk.

        Parameters
        ----------
        models_dir : string
            Path to directory. If unspecified the default downloaded models are
            used.

        max_models : int, optional
            Maximum number of affinity and cleavage (counted separately)
            models to load

        Returns
        -------
        `Class1PresentationPredictor` instance
        """
        if models_dir is None:
            models_dir = get_default_class1_presentation_models_dir()

        affinity_predictor = Class1AffinityPredictor.load(
            join(models_dir, "affinity_predictor"), max_models=max_models)

        cleavage_predictor_with_flanks = None
        if exists(join(models_dir, "cleavage_predictor_with_flanks")):
            cleavage_predictor_with_flanks = Class1CleavagePredictor.load(
                join(models_dir, "cleavage_predictor_with_flanks"),
                max_models=max_models)

        cleavage_predictor_without_flanks = None
        if exists(join(models_dir, "cleavage_predictor_without_flanks")):
            cleavage_predictor_without_flanks = Class1CleavagePredictor.load(
                join(models_dir, "cleavage_predictor_without_flanks"),
                max_models=max_models)

        weights_dataframe = pandas.read_csv(
            join(models_dir, "weights.csv"),
            index_col=0)

        result = cls(
            affinity_predictor=affinity_predictor,
            cleavage_predictor_with_flanks=cleavage_predictor_with_flanks,
            cleavage_predictor_without_flanks=cleavage_predictor_without_flanks,
            weights_dataframe=weights_dataframe)
        return result
