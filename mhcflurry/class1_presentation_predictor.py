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
from .class1_processing_predictor import Class1ProcessingPredictor
from .class1_neural_network import DEFAULT_PREDICT_BATCH_SIZE
from .encodable_sequences import EncodableSequences
from .regression_target import from_ic50, to_ic50
from .multiple_allele_encoding import MultipleAlleleEncoding
from .downloads import get_default_class1_presentation_models_dir
from .common import load_weights


MAX_ALLELES_PER_SAMPLE = 6
PREDICT_BATCH_SIZE = DEFAULT_PREDICT_BATCH_SIZE
PREDICT_CHUNK_SIZE = 100000  # currently used only for cleavage prediction


class Class1PresentationPredictor(object):
    model_inputs = ["affinity_score", "processing_score"]

    def __init__(
            self,
            affinity_predictor=None,
            processing_predictor_with_flanks=None,
            processing_predictor_without_flanks=None,
            weights_dataframe=None,
            metadata_dataframes=None):

        self.affinity_predictor = affinity_predictor
        self.processing_predictor_with_flanks = processing_predictor_with_flanks
        self.processing_predictor_without_flanks = processing_predictor_without_flanks
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

    def predict_processing(
            self, peptides, n_flanks=None, c_flanks=None, verbose=1):

        if (n_flanks is None) != (c_flanks is None):
            raise ValueError("Specify both or neither of n_flanks, c_flanks")

        if n_flanks is None:
            if self.processing_predictor_without_flanks is None:
                raise ValueError("No processing predictor without flanks")
            predictor = self.processing_predictor_without_flanks
            n_flanks = [""] * len(peptides)
            c_flanks = n_flanks
        else:
            if self.processing_predictor_with_flanks is None:
                raise ValueError("No processing predictor with flanks")
            predictor = self.processing_predictor_with_flanks

        num_chunks = int(numpy.ceil(float(len(peptides)) / PREDICT_CHUNK_SIZE))
        peptide_chunks = numpy.array_split(peptides, num_chunks)
        n_flank_chunks = numpy.array_split(n_flanks, num_chunks)
        c_flank_chunks = numpy.array_split(c_flanks, num_chunks)

        iterator = zip(peptide_chunks, n_flank_chunks, c_flank_chunks)
        if verbose > 0:
            print("Predicting processing.")
            if tqdm is not None:
                iterator = tqdm.tqdm(iterator, total=len(peptide_chunks))

        result_chunks = []
        for (peptide_chunk, n_flank_chunk, c_flank_chunk) in iterator:
            result_chunk = predictor.predict(
                peptides=peptide_chunk,
                n_flanks=n_flank_chunk,
                c_flanks=c_flank_chunk,
                batch_size=PREDICT_BATCH_SIZE)
            result_chunks.append(result_chunk)
        return numpy.concatenate(result_chunks)

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
        if self.processing_predictor_without_flanks is not None:
            with_flanks_list.append(False)

        if n_flanks is not None and self.processing_predictor_with_flanks is not None:
            with_flanks_list.append(True)

        if not with_flanks_list:
            raise RuntimeError("Can't fit any models")

        if self.weights_dataframe is None:
            self.weights_dataframe = pandas.DataFrame()

        for with_flanks in with_flanks_list:
            model_name = 'with_flanks' if with_flanks else "without_flanks"
            if verbose > 0:
                print("Training variant", model_name)

            df["processing_score"] = self.predict_processing(
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
                model.classes_ = numpy.array([0, 1])
        else:
            model = self._models_cache[name]
        return model

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

    def predict_sequences(
            self,
            sequences,
            alleles,
            result="best",  # or "all" or "filtered"
            comparison_quantity="presentation_score",
            comparison_value=None,
            peptide_lengths=[8, 9, 10, 11],
            use_flanks=True,
            include_affinity_percentile=False,
            verbose=1,
            throw=True):

        processing_predictor = self.processing_predictor_with_flanks
        if not use_flanks or processing_predictor is None:
            processing_predictor = self.processing_predictor_without_flanks

        supported_sequence_lengths = processing_predictor.sequence_lengths
        n_flank_length = supported_sequence_lengths["n_flank"]
        c_flank_length = supported_sequence_lengths["c_flank"]

        sequence_names = []
        n_flanks = [] if use_flanks else None
        c_flanks = [] if use_flanks else None
        peptides = []

        if isinstance(sequences, string_types):
            sequences = [sequences]

        if not isinstance(sequences, dict):
            sequences = collections.OrderedDict(
                ("sequence_%04d" % (i + 1), sequence)
                for (i, sequence) in enumerate(sequences))

        if isinstance(alleles, string_types):
            alleles = [alleles]

        if not isinstance(alleles, dict):
            if all([isinstance(item, string_types) for item in alleles]):
                alleles = dict((name, alleles) for name in sequences.keys())
            elif len(alleles) != len(sequences):
                raise ValueError(
                    "alleles must be (1) a string (a single allele), (2) a list of "
                    "strings (a single genotype), (3) a list of list of strings ("
                    "(multiple genotypes, where the total number of genotypes "
                    "must equal the number of sequences), or (4) a dict (in which "
                    "case the keys must match the sequences dict keys). Here "
                    "it seemed like option (3) was being used, but the length "
                    "of alleles (%d) did not match the length of sequences (%d)."
                    % (len(alleles), len(sequences)))
            else:
                alleles = dict(zip(sequences.keys(), alleles))

        missing = [key for key in sequences if key not in alleles]
        if missing:
            raise ValueError(
                "Sequence names missing from alleles dict: ", missing)

        for (name, sequence) in sequences.items():
            if not isinstance(sequence, string_types):
                raise ValueError("Expected string, not %s (%s)" % (
                    sequence, type(sequence)))
            for peptide_start in range(len(sequence) - min(peptide_lengths)):
                n_flank_start = max(0, peptide_start - n_flank_length)
                for peptide_length in peptide_lengths:
                    c_flank_end = (
                        peptide_start + peptide_length + c_flank_length)
                    sequence_names.append(name)
                    peptides.append(
                        sequence[peptide_start: peptide_start + peptide_length])
                    if use_flanks:
                        n_flanks.append(
                            sequence[n_flank_start : peptide_start])
                        c_flanks.append(
                            sequence[peptide_start + peptide_length : c_flank_end])

        result_df = self.predict_to_dataframe(
            peptides=peptides,
            alleles=alleles,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            experiment_names=sequence_names,
            include_affinity_percentile=include_affinity_percentile,
            verbose=verbose,
            throw=throw)

        result_df = result_df.rename(
            columns={"experiment_name": "sequence_name"})

        comparison_is_score = comparison_quantity.endswith("score")

        result_df = result_df.sort_values(
            comparison_quantity,
            ascending=not comparison_is_score)

        if result == "best":
            result_df = result_df.drop_duplicates(
                "sequence_name", keep="first").sort_values("sequence_name")
        elif result == "filtered":
            if comparison_is_score:
                result_df = result_df.loc[
                    result_df[comparison_quantity] >= comparison_value
                ]
            else:
                result_df = result_df.loc[
                    result_df[comparison_quantity] <= comparison_value
                ]
        elif result == "all":
            pass
        else:
            raise ValueError(
                "Unknown result: %s. Valid choices are: best, filtered, all"
                % result)

        result_df = result_df.reset_index(drop=True)
        result_df = result_df.copy()

        return result_df



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

        if (n_flanks is None) != (c_flanks is None):
            raise ValueError("Specify both or neither of n_flanks, c_flanks")

        processing_scores = self.predict_processing(
            peptides=peptides,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            verbose=verbose)

        df = self.predict_affinity(
            peptides=peptides,
            experiment_names=experiment_names,
            alleles=alleles,
            include_affinity_percentile=include_affinity_percentile,
            verbose=verbose,
            throw=throw)
        df["affinity_score"] = from_ic50(df.affinity)
        df["processing_score"] = processing_scores
        if c_flanks is not None:
            df.insert(1, "c_flank", c_flanks)
        if n_flanks is not None:
            df.insert(1, "n_flank", n_flanks)

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
        if self.processing_predictor_with_flanks is not None:
            self.processing_predictor_with_flanks.save(
                join(models_dir, "processing_predictor_with_flanks"))
        if self.processing_predictor_without_flanks is not None:
            self.processing_predictor_without_flanks.save(
                join(models_dir, "processing_predictor_without_flanks"))

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
            Maximum number of affinity and processing (counted separately)
            models to load

        Returns
        -------
        `Class1PresentationPredictor` instance
        """
        if models_dir is None:
            models_dir = get_default_class1_presentation_models_dir()

        affinity_predictor = Class1AffinityPredictor.load(
            join(models_dir, "affinity_predictor"), max_models=max_models)

        processing_predictor_with_flanks = None
        if exists(join(models_dir, "processing_predictor_with_flanks")):
            processing_predictor_with_flanks = Class1ProcessingPredictor.load(
                join(models_dir, "processing_predictor_with_flanks"),
                max_models=max_models)
        else:
            logging.warning(
                "Presentation predictor is missing processing predictor: %s",
                join(models_dir, "processing_predictor_with_flanks"))

        processing_predictor_without_flanks = None
        if exists(join(models_dir, "processing_predictor_without_flanks")):
            processing_predictor_without_flanks = Class1ProcessingPredictor.load(
                join(models_dir, "processing_predictor_without_flanks"),
                max_models=max_models)
        else:
            logging.warning(
                "Presentation predictor is missing processing predictor: %s",
                join(models_dir, "processing_predictor_without_flanks"))

        weights_dataframe = pandas.read_csv(
            join(models_dir, "weights.csv"),
            index_col=0)

        result = cls(
            affinity_predictor=affinity_predictor,
            processing_predictor_with_flanks=processing_predictor_with_flanks,
            processing_predictor_without_flanks=processing_predictor_without_flanks,
            weights_dataframe=weights_dataframe)
        return result
