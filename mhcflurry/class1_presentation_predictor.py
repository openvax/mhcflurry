from __future__ import print_function

from os.path import join, exists
from os import mkdir
from socket import gethostname
from getpass import getuser

import time
import collections
import logging
import warnings
from six import string_types

import numpy
import pandas
import sklearn
import sklearn.linear_model


try:
    import tqdm
except ImportError:
    tdqm = None

from .version import __version__
from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_processing_predictor import Class1ProcessingPredictor
from .class1_neural_network import DEFAULT_PREDICT_BATCH_SIZE
from .encodable_sequences import EncodableSequences
from .regression_target import from_ic50
from .downloads import get_default_class1_presentation_models_dir
from .percent_rank_transform import PercentRankTransform


MAX_ALLELES_PER_SAMPLE = 6
PREDICT_BATCH_SIZE = DEFAULT_PREDICT_BATCH_SIZE
PREDICT_CHUNK_SIZE = 100000  # currently used only for cleavage prediction


class Class1PresentationPredictor(object):
    """
    A logistic regression model over predicted binding affinity (BA) and antigen
    processing (AP) score.

    Instances of this class delegate to Class1AffinityPredictor and
    Class1ProcessingPredictor instances to generate BA and AP predictions.
    These predictions are combined using a logistic regression model to give
    a "presentation score" prediction.

    Most users will call the `load` static method to get an instance of this
    class, then call the `predict` method to generate predictions.
    """
    model_inputs = ["affinity_score", "processing_score"]

    def __init__(
            self,
            affinity_predictor=None,
            processing_predictor_with_flanks=None,
            processing_predictor_without_flanks=None,
            weights_dataframe=None,
            metadata_dataframes=None,
            percent_rank_transform=None,
            provenance_string=None):

        self.affinity_predictor = affinity_predictor
        self.processing_predictor_with_flanks = processing_predictor_with_flanks
        self.processing_predictor_without_flanks = processing_predictor_without_flanks
        self.weights_dataframe = weights_dataframe
        self.metadata_dataframes = (
            dict(metadata_dataframes) if metadata_dataframes else {})
        self._models_cache = {}
        self.percent_rank_transform = percent_rank_transform
        self.provenance_string = provenance_string

    @property
    def supported_alleles(self):
        """
        List of alleles supported by the underlying Class1AffinityPredictor
        """
        return self.affinity_predictor.supported_alleles

    @property
    def supported_peptide_lengths(self):
        """
        (min, max) of supported peptide lengths, inclusive.
        """
        return self.affinity_predictor.supported_peptide_lengths

    @property
    def supports_affinity_prediction(self):
        """Is there an affinity predictor associated with this instance?"""
        return self.affinity_predictor is not None

    @property
    def supports_processing_prediction(self):
        """Is there a processing predictor associated with this instance?"""
        return (
            self.processing_predictor_with_flanks is not None or
            self.processing_predictor_without_flanks is not None)

    @property
    def supports_presentation_prediction(self):
        """Can this instance predict presentation?"""
        return (
            self.supports_affinity_prediction and
            self.supports_processing_prediction and
            self.weights_dataframe is not None)

    def predict_affinity(
            self,
            peptides,
            alleles,
            sample_names=None,
            include_affinity_percentile=True,
            verbose=1,
            throw=True):
        """
        Predict binding affinities across samples (each corresponding to up to
        six MHC  I alleles).

        Two modes are supported: each peptide can be evaluated for binding to
        any of the alleles in any sample (this is what happens when sample_names
        is None), or the i'th peptide can be evaluated for binding the alleles
        of the sample given by the i'th entry in sample_names.

        For example, if we don't specify sample_names, then predictions
        are taken for all combinations of samples and peptides, for a result
        size of num peptides *  num samples:

        >>> predictor = Class1PresentationPredictor.load()
        >>> predictor.predict_affinity(
        ...    peptides=["SIINFEKL", "PEPTIDE"],
        ...    alleles={
        ...        "sample1": ["A0201", "A0301", "B0702"],
        ...        "sample2": ["A0101", "C0202"],
        ...    },
        ...    verbose=0)
            peptide  peptide_num sample_name   affinity best_allele  affinity_percentile
        0  SIINFEKL            0     sample1  11927.161       A0201                6.296
        1   PEPTIDE            1     sample1  32507.082       A0201               71.249
        2  SIINFEKL            0     sample2   2725.593       C0202                6.662
        3   PEPTIDE            1     sample2  28304.336       C0202               54.652

        In contrast, here we specify sample_names, so peptide is evaluated for
        binding the alleles in the corresponding sample, for a result size equal
        to the number of peptides:

        >>> predictor.predict_affinity(
        ...    peptides=["SIINFEKL", "PEPTIDE"],
        ...    alleles={
        ...        "sample1": ["A0201", "A0301", "B0702"],
        ...        "sample2": ["A0101", "C0202"],
        ...    },
        ...    sample_names=["sample2", "sample1"],
        ...    verbose=0)
            peptide  peptide_num sample_name   affinity best_allele  affinity_percentile
        0  SIINFEKL            0     sample2   2725.592       C0202                6.662
        1   PEPTIDE            1     sample1  32507.078       A0201               71.249

        Parameters
        ----------
        peptides : list of string
            Peptide sequences
        alleles : dict of string -> list of string
            Keys are sample names, values are the alleles (genotype) for
            that sample
        sample_names : list of string [same length as peptides]
            Sample names corresponding to each peptide. If None, then
            predictions are generated for all sample genotypes  across all
            peptides.
        include_affinity_percentile : bool
            Whether to include affinity percentile ranks
        verbose : int
            Set to 0 for quiet.
        throw : verbose
            Whether to throw exception (vs. just log a warning) on invalid
            peptides, etc.

        Returns
        -------
        pandas.DataFrame : predictions
        """
        df = pandas.DataFrame({
            "peptide": numpy.array(peptides, copy=False),
        })
        df["peptide_num"] = df.index
        if sample_names is None:
            peptides = EncodableSequences.create(peptides)
            all_alleles = set()
            for lst in alleles.values():
                all_alleles.update(lst)

            iterator = sorted(all_alleles)

            if verbose > 0:
                print("Predicting affinities.")
                if tqdm is not None:
                    iterator = tqdm.tqdm(iterator, total=len(all_alleles))

            predictions_df = pandas.DataFrame(index=df.index)
            for allele in iterator:
                predictions_df[allele] = self.affinity_predictor.predict(
                    peptides=peptides,
                    allele=allele,
                    model_kwargs={'batch_size': PREDICT_BATCH_SIZE},
                    throw=throw)

            dfs = []
            for (sample_name, sample_alleles) in alleles.items():
                new_df = df.copy()
                new_df["sample_name"] = sample_name
                new_df["affinity"] = predictions_df[
                    sample_alleles
                ].min(1).values
                if len(df) == 0:
                    new_df["best_allele"] = []
                else:
                    new_df["best_allele"] = predictions_df[
                        sample_alleles
                    ].idxmin(1).values
                dfs.append(new_df)

            result_df = pandas.concat(dfs, ignore_index=True)
        else:
            df["sample_name"] = numpy.array(sample_names, copy=False)

            iterator = df.groupby("sample_name")
            if verbose > 0:
                print("Predicting affinities.")
                if tqdm is not None:
                    iterator = tqdm.tqdm(
                        iterator, total=df.sample_name.nunique())

            for (sample, sub_df) in iterator:
                predictions_df = pandas.DataFrame(index=sub_df.index)
                sample_peptides = EncodableSequences.create(sub_df.peptide.values)
                for allele in alleles[sample]:
                    predictions_df[allele] = self.affinity_predictor.predict(
                        peptides=sample_peptides,
                        allele=allele,
                        model_kwargs={'batch_size': PREDICT_BATCH_SIZE},
                        throw=throw)
                df.loc[
                    sub_df.index, "affinity"
                ] = predictions_df.min(1).values
                df.loc[
                    sub_df.index, "best_allele"
                ] = predictions_df.idxmin(1).values

            result_df = df

        if include_affinity_percentile:
            result_df["affinity_percentile"] = (
                self.affinity_predictor.percentile_ranks(
                    result_df.affinity.values,
                    alleles=result_df.best_allele.values,
                    throw=False))

        return result_df

    def predict_processing(
            self, peptides, n_flanks=None, c_flanks=None, throw=True, verbose=1):
        """
        Predict antigen processing scores for individual peptides, optionally
        including flanking sequences for better cleavage prediction.

        Parameters
        ----------
        peptides : list of string
        n_flanks : list of string [same length as peptides]
        c_flanks : list of string [same length as peptides]
        throw : boolean
            Whether to raise exception on unsupported peptides
        verbose  : int

        Returns
        -------
        numpy.array : Antigen processing scores for each peptide
        """

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

        if len(peptides) == 0:
            return numpy.array([], dtype=float)

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
                throw=throw,
                batch_size=PREDICT_BATCH_SIZE)
            result_chunks.append(result_chunk)
        return numpy.concatenate(result_chunks)

    def fit(
            self,
            targets,
            peptides,
            sample_names,
            alleles,
            n_flanks=None,
            c_flanks=None,
            verbose=1):
        """
        Fit the presentation score logistic regression model.

        Parameters
        ----------
        targets : list of int/float
            1 indicates hit, 0 indicates decoy
        peptides : list of string [same length as targets]
        sample_names : list of string [same length as targets]
        alleles : dict of string -> list of string
            Keys are sample names, values are the alleles for that sample
        n_flanks : list of string [same length as targets]
        c_flanks : list of string [same length as targets]
        verbose : int
        """

        df = self.predict_affinity(
            peptides=peptides,
            alleles=alleles,
            sample_names=sample_names,
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

            (intercept,) = model.intercept_.flatten()
            self.weights_dataframe.loc[model_name, "intercept"] = intercept
            for (name, value) in zip(self.model_inputs, numpy.squeeze(model.coef_)):
                self.weights_dataframe.loc[model_name, name] = value
            self._models_cache[model_name] = model

    def get_model(self, name=None):
        """
        Load or instantiate a new logistic regression model. Private helper
        method.

        Parameters
        ----------
        name : string
            If None (the default), an un-fit LR model is returned. Otherwise the
            weights are loaded for the specified model.

        Returns
        -------
        sklearn.linear_model.LogisticRegression
        """
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
            sample_names=None,
            n_flanks=None,
            c_flanks=None,
            include_affinity_percentile=False,
            verbose=1,
            throw=True):
        """
        Predict presentation scores across a set of peptides.

        Presentation scores combine predictions for MHC I binding affinity
        and antigen processing.

        This method returns a pandas.DataFrame giving presentation scores plus
        the binding affinity and processing predictions and other intermediate
        results.

        Example:

        >>> predictor = Class1PresentationPredictor.load()
        >>> predictor.predict(
        ...    peptides=["SIINFEKL", "PEPTIDE"],
        ...    n_flanks=["NNN", "SNS"],
        ...    c_flanks=["CCC", "CNC"],
        ...    alleles={
        ...        "sample1": ["A0201", "A0301", "B0702"],
        ...        "sample2": ["A0101", "C0202"],
        ...    },
        ...    verbose=0)
            peptide n_flank c_flank  peptide_num sample_name   affinity best_allele  processing_score  presentation_score  presentation_percentile
        0  SIINFEKL     NNN     CCC            0     sample1  11927.161       A0201             0.838               0.145                    2.282
        1   PEPTIDE     SNS     CNC            1     sample1  32507.082       A0201             0.025               0.003                  100.000
        2  SIINFEKL     NNN     CCC            0     sample2   2725.593       C0202             0.838               0.416                    1.017
        3   PEPTIDE     SNS     CNC            1     sample2  28304.338       C0202             0.025               0.003                   99.287

        You can also specify sample_names, in which case peptide is evaluated
        for binding the alleles in the corresponding sample only. See
        `predict_affinity` for an examples.

        Parameters
        ----------
        peptides : list of string
            Peptide sequences
        alleles : list of string or dict of string -> list of string
            If you are predicting for a single sample, pass a list of strings
            (up to 6) indicating the genotype. If you are predicting across
            multiple samples, pass a dict where the keys are (arbitrary)
            sample names and the values are the alleles to predict for that
            sample. Set to an empty list or dict to perform processing
            prediction only.
        sample_names : list of string [same length as peptides]
            If you are passing a dict for 'alleles', you can use this
            argument to specify which peptides go with which samples. If it is
            None, then predictions will be performed for each peptide across all
            samples.
        n_flanks : list of string [same length as peptides]
            Upstream sequences before the peptide. Sequences of any length can
            be given and a suffix of the size supported by the model will be
            used.
        c_flanks : list of string [same length as peptides]
            Downstream sequences after the peptide. Sequences of any length can
            be given and a prefix of the size supported by the model will be
            used.
        include_affinity_percentile : bool
            Whether to include affinity percentile ranks
        verbose : int
            Set to 0 for quiet.
        throw : verbose
            Whether to throw exception (vs. just log a warning) on invalid
            peptides, etc.

        Returns
        -------
        pandas.DataFrame

        Presentation scores and intermediate results.
        """

        if isinstance(peptides, string_types):
            raise TypeError("peptides must be a list not a string")
        if isinstance(alleles, string_types):
            raise TypeError("alleles must be a list or dict")

        if not isinstance(alleles, dict):
            # Make alleles into a dict.
            if sample_names is not None:
                raise ValueError(
                    "alleles must be a dict when sample_names is specified")

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

            alleles = {
                "sample1": alleles,
            }

        if (n_flanks is None) != (c_flanks is None):
            raise ValueError("Specify both or neither of n_flanks, c_flanks")

        if self.supports_processing_prediction:
            processing_scores = self.predict_processing(
                peptides=peptides,
                n_flanks=n_flanks,
                c_flanks=c_flanks,
                throw=throw,
                verbose=verbose)
        else:
            processing_scores = None

        if alleles:
            df = self.predict_affinity(
                peptides=peptides,
                alleles=alleles,
                sample_names=sample_names,  # might be None
                include_affinity_percentile=include_affinity_percentile,
                verbose=verbose,
                throw=throw)

            df["affinity_score"] = from_ic50(df.affinity)
        else:
            # Processing prediction only.
            df = pandas.DataFrame({
                "peptide_num": numpy.arange(len(peptides)),
                "peptide": peptides,
            })
            df["sample_name"] = "sample1"

        if processing_scores is not None:
            df["processing_score"] = df.peptide_num.map(
                pandas.Series(processing_scores))
            if c_flanks is not None:
                df.insert(1, "c_flank", df.peptide_num.map(pandas.Series(c_flanks)))
            if n_flanks is not None:
                df.insert(1, "n_flank", df.peptide_num.map(pandas.Series(n_flanks)))

        predict_presentation = (
                "affinity_score" in df.columns and
                "processing_score" in df.columns and
                self.supports_presentation_prediction)
        if predict_presentation:
            if len(df) > 0:
                model_name = 'with_flanks' if n_flanks is not None else \
                    "without_flanks"
                model = self.get_model(model_name)
                input_matrix = df[self.model_inputs]
                null_mask = None
                if not throw:
                    # Invalid peptides will be null.
                    null_mask = input_matrix.isnull().any(1)
                    input_matrix = input_matrix.fillna(0.0)
                df["presentation_score"] = model.predict_proba(
                    input_matrix.values)[:,1]
                if null_mask is not None:
                    df.loc[null_mask, "presentation_score"] = numpy.nan
                df["presentation_percentile"] = self.percentile_ranks(
                    df["presentation_score"], throw=False)
            else:
                df["presentation_score"] = []
                df["presentation_percentile"] = []
            del df["affinity_score"]
        return df

    def predict_sequences(
            self,
            sequences,
            alleles,
            result="best",
            comparison_quantity=None,
            filter_value=None,
            peptide_lengths=(8, 9, 10, 11),
            use_flanks=True,
            include_affinity_percentile=True,
            verbose=1,
            throw=True):
        """
        Predict presentation across protein sequences.

        Example:

        >>> predictor = Class1PresentationPredictor.load()
        >>> predictor.predict_sequences(
        ...    sequences={
        ...        'protein1': "MDSKGSSQKGSRLLLLLVVSNLL",
        ...        'protein2': "SSLPTPEDKEQAQQTHH",
        ...    },
        ...    alleles={
        ...        "sample1": ["A0201", "A0301", "B0702"],
        ...        "sample2": ["A0101", "C0202"],
        ...    },
        ...    result="filtered",
        ...    comparison_quantity="affinity",
        ...    filter_value=500,
        ...    verbose=0)
          sequence_name  pos     peptide n_flank c_flank sample_name  affinity best_allele  affinity_percentile  processing_score  presentation_score  presentation_percentile
        0      protein1   14   LLLVVSNLL   GSRLL             sample1    57.180       A0201                0.398             0.233               0.754                    0.351
        1      protein1   13   LLLLVVSNL   KGSRL       L     sample1    57.339       A0201                0.398             0.031               0.586                    0.643
        2      protein1    5   SSQKGSRLL   MDSKG   LLLVV     sample2   110.779       C0202                0.782             0.061               0.456                    0.920
        3      protein1    6   SQKGSRLLL   DSKGS   LLVVS     sample2   254.480       C0202                1.735             0.102               0.303                    1.356
        4      protein1   13  LLLLVVSNLL   KGSRL             sample1   260.390       A0201                1.012             0.158               0.345                    1.215
        5      protein1   12  LLLLLVVSNL   QKGSR       L     sample1   308.150       A0201                1.094             0.015               0.206                    1.802
        6      protein2    0   SSLPTPEDK           EQAQQ     sample2   410.354       C0202                2.398             0.003               0.158                    2.155
        7      protein1    5    SSQKGSRL   MDSKG   LLLLV     sample2   444.321       C0202                2.512             0.026               0.159                    2.138
        8      protein2    0   SSLPTPEDK           EQAQQ     sample1   459.296       A0301                0.971             0.003               0.144                    2.292
        9      protein1    4   GSSQKGSRL    MDSK   LLLLV     sample2   469.052       C0202                2.595             0.014               0.146                    2.261

        Parameters
        ----------
        sequences : str, list of string, or string -> string dict
            Protein sequences. If a dict is given, the keys are arbitrary (
            e.g. protein names), and the values are the amino acid sequences.
        alleles : list of string, list of list of string, or dict of string -> list of string
            MHC I alleles. Can be: (1) a string (a single allele), (2) a list of
            strings (a single genotype), (3) a list of list of strings
            (multiple genotypes, where the total number of genotypes must equal
            the number of sequences), or (4) a dict giving multiple genotypes,
            which will each be run over the sequences.
        result : string
            Specify 'best' to return the strongest peptide for each sequence,
            'all' to return predictions for all peptides, or 'filtered' to
            return predictions where the comparison_quantity is stronger
            (i.e (<) for affinity, (>) for scores) than filter_value.
        comparison_quantity : string
            One of "presentation_score", "processing_score", "affinity", or
            "affinity_percentile". Prediction to use to rank (if result is
            "best") or filter (if result is "filtered") results. Default is
            "presentation_score".
        filter_value : float
            Threshold value to use, only relevant when result is "filtered".
            If comparison_quantity is "affinity", then all results less than
            (i.e. tighter than) the specified nM affinity are retained. If it's
            "presentation_score" or "processing_score" then results greater than
            the indicated filter_value are retained.
        peptide_lengths : list of int
            Peptide lengths to predict for.
        use_flanks : bool
            Whether to include flanking sequences when running the AP predictor
            (for better cleavage prediction).
        include_affinity_percentile : bool
            Whether to include affinity percentile ranks in output.
        verbose : int
            Set to 0 for quiet mode.
        throw : boolean
            Whether to throw exceptions (vs. log warnings) on invalid inputs.

        Returns
        -------
        pandas.DataFrame with columns:
            peptide, n_flank, c_flank, sequence_name, affinity, best_allele,
            processing_score, presentation_score
        """
        if len(alleles) == 0:
            alleles = {}

        if len(alleles) > 0 and not self.supports_affinity_prediction:
            raise ValueError(
                "Affinity prediction not supported by this predictor")

        if comparison_quantity is None:
            if len(alleles) > 0:
                if self.supports_presentation_prediction:
                    comparison_quantity = "presentation_score"
                else:
                    comparison_quantity = "affinity"
            else:
                comparison_quantity = "processing_score"

        if comparison_quantity == "presentation_score":
            if not self.supports_presentation_prediction:
                raise ValueError(
                    "Presentation prediction not supported by this predictor")
        elif comparison_quantity == "processing_score":
            if not self.supports_processing_prediction:
                raise ValueError(
                    "Processing prediction not supported by this predictor")
        elif comparison_quantity in ("affinity", "affinity_percentile"):
            if not self.supports_affinity_prediction:
                raise ValueError(
                    "Affinity prediction not supported by this predictor")
        else:
            raise ValueError(
                "Unknown comparison quantity: %s" % comparison_quantity)

        processing_predictor = self.processing_predictor_with_flanks
        if not use_flanks or processing_predictor is None:
            processing_predictor = self.processing_predictor_without_flanks

        if processing_predictor is not None:
            supported_sequence_lengths = processing_predictor.sequence_lengths
            n_flank_length = supported_sequence_lengths["n_flank"]
            c_flank_length = supported_sequence_lengths["c_flank"]
        else:
            n_flank_length = 0
            c_flank_length = 0

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

        cross_product = True
        if isinstance(alleles, string_types):
            # Case (1) - alleles is a string
            alleles = [alleles]

        if isinstance(alleles, dict):
            if any([isinstance(v, string_types) for v in alleles.values()]):
                raise ValueError(
                    "The values in the alleles dict must be lists, not strings")
        else:
            if all(isinstance(a, string_types) for a in alleles):
                # Case (2) - a simple list of alleles
                alleles = {
                    'sample1': alleles
                }
            else:
                # Case (3) - a list of lists
                alleles = collections.OrderedDict(
                    ("genotype_%04d" % (i + 1), genotype)
                    for (i, genotype) in enumerate(alleles))
                cross_product = False

                if len(alleles) != len(sequences):
                    raise ValueError(
                        "When passing a list of lists for the alleles argument "
                        "the length of the list (%d) must match the length of "
                        "the sequences being predicted (%d)" % (
                            len(alleles), len(sequences)))

        if not isinstance(alleles, dict):
            raise ValueError("Invalid type for alleles: ", type(alleles))

        sample_names = None if cross_product else []
        genotype_names = list(alleles)
        position_in_sequence = []
        for (i, (name, sequence)) in enumerate(sequences.items()):
            genotype_name = None if cross_product else genotype_names[i]

            if not isinstance(sequence, string_types):
                raise ValueError("Expected string, not %s (%s)" % (
                    sequence, type(sequence)))
            for peptide_start in range(len(sequence) - min(peptide_lengths) + 1):
                n_flank_start = max(0, peptide_start - n_flank_length)
                for peptide_length in peptide_lengths:
                    peptide = sequence[
                        peptide_start: peptide_start + peptide_length
                    ]
                    if len(peptide) != peptide_length:
                        continue
                    c_flank_end = (
                        peptide_start + peptide_length + c_flank_length)
                    sequence_names.append(name)
                    position_in_sequence.append(peptide_start)
                    if not cross_product:
                        sample_names.append(genotype_name)
                    peptides.append(peptide)
                    if use_flanks:
                        n_flanks.append(
                            sequence[n_flank_start : peptide_start])
                        c_flanks.append(
                            sequence[peptide_start + peptide_length : c_flank_end])

        result_df = self.predict(
            peptides=peptides,
            alleles=alleles,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            sample_names=sample_names,
            include_affinity_percentile=include_affinity_percentile,
            verbose=verbose,
            throw=throw)

        result_df.insert(
            0,
            "sequence_name",
            result_df.peptide_num.map(pandas.Series(sequence_names)))
        result_df.insert(
            1,
            "pos",
            result_df.peptide_num.map(pandas.Series(position_in_sequence)))
        del result_df["peptide_num"]

        comparison_is_score = comparison_quantity.endswith("score")

        result_df = result_df.sort_values(
            comparison_quantity,
            ascending=not comparison_is_score)

        if result == "best":
            result_df = result_df.drop_duplicates(
                ["sequence_name", "sample_name"], keep="first"
            ).sort_values("sequence_name")
        elif result == "filtered":
            if comparison_is_score:
                result_df = result_df.loc[
                    result_df[comparison_quantity] >= filter_value
                ]
            else:
                result_df = result_df.loc[
                    result_df[comparison_quantity] <= filter_value
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

    def save(
            self,
            models_dir,
            write_affinity_predictor=True,
            write_processing_predictor=True,
            write_weights=True,
            write_percent_ranks=True,
            write_info=True,
            write_metdata=True):
        """
        Save the predictor to a directory on disk. If the directory does
        not exist it will be created.

        The wrapped Class1AffinityPredictor and Class1ProcessingPredictor
        instances are included in the saved data.

        Parameters
        ----------
        models_dir : string
            Path to directory. It will be created if it doesn't exist.
        """

        if write_weights and self.weights_dataframe is None:
            raise RuntimeError("Can't save before fitting")

        if not exists(models_dir):
            mkdir(models_dir)

        # Save underlying predictors
        if write_affinity_predictor:
            self.affinity_predictor.save(join(models_dir, "affinity_predictor"))
        if write_processing_predictor:
            if self.processing_predictor_with_flanks is not None:
                self.processing_predictor_with_flanks.save(
                    join(models_dir, "processing_predictor_with_flanks"))
            if self.processing_predictor_without_flanks is not None:
                self.processing_predictor_without_flanks.save(
                    join(models_dir, "processing_predictor_without_flanks"))

        if write_weights:
            # Save model coefficients.
            self.weights_dataframe.to_csv(join(models_dir, "weights.csv"))

        if write_percent_ranks:
            # Percent ranks
            if self.percent_rank_transform:
                series = self.percent_rank_transform.to_series()
                percent_ranks_df = pandas.DataFrame(index=series.index)
                numpy.testing.assert_array_almost_equal(
                    series.index.values,
                    percent_ranks_df.index.values)
                percent_ranks_df["presentation_score"] = series.values
                percent_ranks_path = join(models_dir, "percent_ranks.csv")
                percent_ranks_df.to_csv(
                    percent_ranks_path,
                    index=True,
                    index_label="bin")
                logging.info("Wrote: %s", percent_ranks_path)

        if write_info:
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

        if write_metdata:
            if self.metadata_dataframes:
                for (name, df) in self.metadata_dataframes.items():
                    metadata_df_path = join(models_dir, "%s.csv.bz2" % name)
                    df.to_csv(metadata_df_path, index=False, compression="bz2")


    @classmethod
    def load(cls, models_dir=None, max_models=None):
        """
        Deserialize a predictor from a directory on disk.

        This will also load the wrapped Class1AffinityPredictor and
        Class1ProcessingPredictor instances.

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

        # Load percent ranks if available
        percent_rank_transform = None
        percent_ranks_path = join(models_dir, "percent_ranks.csv")
        if exists(percent_ranks_path):
            percent_ranks_df = pandas.read_csv(percent_ranks_path, index_col=0)
            percent_rank_transform = PercentRankTransform.from_series(
                percent_ranks_df["presentation_score"])

        provenance_string = None
        try:
            info_path = join(models_dir, "info.txt")
            info = pandas.read_csv(
                info_path, sep="\t", header=None, index_col=0).iloc[
                :, 0
            ].to_dict()
            provenance_string = "generated on %s" % info["trained on"]
        except OSError:
            pass

        result = cls(
            affinity_predictor=affinity_predictor,
            processing_predictor_with_flanks=processing_predictor_with_flanks,
            processing_predictor_without_flanks=processing_predictor_without_flanks,
            weights_dataframe=weights_dataframe,
            percent_rank_transform=percent_rank_transform,
            provenance_string=provenance_string)
        return result

    def __repr__(self):
        pieces = ["at 0x%0x" % id(self), "[mhcflurry %s]" % __version__]
        if self.provenance_string:
            pieces.append(self.provenance_string)
        return "<Class1PresentationPredictor %s>" % " ".join(pieces)

    def percentile_ranks(self, presentation_scores, throw=True):
        """
        Return percentile ranks for the given presentation scores.

        Parameters
        ----------
        presentation_scores : sequence of float

        Returns
        -------
        numpy.array of float
        """

        if self.percent_rank_transform is None:
            msg = "No presentation predictor percentile rank information"
            if throw:
                raise ValueError(msg)
            warnings.warn(msg)
            return numpy.ones(len(presentation_scores)) * numpy.nan

        # We subtract from 100 so that strong binders have low percentile ranks,
        # making them comparable to affinity percentile ranks.
        return 100 - self.percent_rank_transform.transform(presentation_scores)

    def calibrate_percentile_ranks(self, scores, bins=None):
        """
        Compute the cumulative distribution of scores, to enable taking
        quantiles of this distribution later.

        Parameters
        ----------
        scores : sequence of float
            Presentation prediction scores
        bins : object
            Anything that can be passed to numpy.histogram's "bins" argument
            can be used here, i.e. either an integer or a sequence giving bin
            edges.
        """
        if bins is None:
            bins = numpy.linspace(0, 1, 1000)

        self.percent_rank_transform = PercentRankTransform()
        self.percent_rank_transform.fit(scores, bins=bins)
