from __future__ import print_function

from os.path import join, exists
from os import mkdir
from socket import gethostname
from getpass import getuser

import time
import collections
import logging
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
from .regression_target import from_ic50, to_ic50
from .downloads import get_default_class1_presentation_models_dir


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
            peptide  peptide_num sample_name      affinity best_allele
        0  SIINFEKL            0     sample1  12906.787792       A0201
        1   PEPTIDE            1     sample1  36827.681130       B0702
        2  SIINFEKL            0     sample2   3588.413748       C0202
        3   PEPTIDE            1     sample2  34362.109211       C0202

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
            peptide  peptide_num sample_name      affinity best_allele
        0  SIINFEKL            0     sample2   3588.412141       C0202
        1   PEPTIDE            1     sample1  36827.682779       B0702


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
            self, peptides, n_flanks=None, c_flanks=None, verbose=1):
        """
        Predict antigen processing scores for individual peptides, optionally
        including flanking sequences for better cleavage prediction.

        Parameters
        ----------
        peptides : list of string
        n_flanks : list of string [same length as peptides]
        c_flanks : list of string [same length as peptides]
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

            self.weights_dataframe.loc[model_name, "intercept"] = model.intercept_
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
            peptide n_flank c_flank  peptide_num sample_name      affinity best_allele  processing_score  presentation_score
        0  SIINFEKL     NNN     CCC            0     sample1  12906.787792       A0201          0.802466            0.140365
        1   PEPTIDE     SNS     CNC            1     sample1  36827.681130       B0702          0.105260            0.004059
        2  SIINFEKL     NNN     CCC            0     sample2   3588.413748       C0202          0.802466            0.338647
        3   PEPTIDE     SNS     CNC            1     sample2  34362.109211       C0202          0.105260            0.004317

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
            sample.
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

        processing_scores = self.predict_processing(
            peptides=peptides,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            verbose=verbose)

        df = self.predict_affinity(
            peptides=peptides,
            alleles=alleles,
            sample_names=sample_names,  # might be None
            include_affinity_percentile=include_affinity_percentile,
            verbose=verbose,
            throw=throw)

        df["affinity_score"] = from_ic50(df.affinity)
        df["processing_score"] = df.peptide_num.map(
            pandas.Series(processing_scores))
        if c_flanks is not None:
            df.insert(1, "c_flank", df.peptide_num.map(pandas.Series(c_flanks)))
        if n_flanks is not None:
            df.insert(1, "n_flank", df.peptide_num.map(pandas.Series(n_flanks)))

        model_name = 'with_flanks' if n_flanks is not None else "without_flanks"
        model = self.get_model(model_name)
        if len(df) > 0:
            df["presentation_score"] = model.predict_proba(
                df[self.model_inputs].values)[:,1]
        else:
            df["presentation_score"] = []
        del df["affinity_score"]
        return df

    def predict_sequences(
            self,
            sequences,
            alleles,
            result="best",
            comparison_quantity="presentation_score",
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
          sequence_name  pos     peptide         n_flank     c_flank sample_name    affinity best_allele  affinity_percentile  processing_score  presentation_score
        0      protein1   13   LLLLVVSNL   MDSKGSSQKGSRL           L     sample1   38.206225       A0201             0.380125          0.017644            0.571060
        1      protein1   14   LLLVVSNLL  MDSKGSSQKGSRLL                 sample1   42.243472       A0201             0.420250          0.090984            0.619213
        2      protein1    5   SSQKGSRLL           MDSKG   LLLVVSNLL     sample2   66.749223       C0202             0.803375          0.383608            0.774468
        3      protein1    6   SQKGSRLLL          MDSKGS    LLVVSNLL     sample2  178.033474       C0202             1.820000          0.275019            0.482206
        4      protein1   13  LLLLVVSNLL   MDSKGSSQKGSRL                 sample1  202.208167       A0201             1.112500          0.058782            0.261320
        5      protein1   12  LLLLLVVSNL    MDSKGSSQKGSR           L     sample1  202.506582       A0201             1.112500          0.010025            0.225648
        6      protein2    0   SSLPTPEDK                    EQAQQTHH     sample1  335.529377       A0301             1.011750          0.010443            0.156798
        7      protein2    0   SSLPTPEDK                    EQAQQTHH     sample2  353.451759       C0202             2.674250          0.010443            0.150753
        8      protein1    8   KGSRLLLLL        MDSKGSSQ      VVSNLL     sample2  410.327286       C0202             2.887000          0.121374            0.194081
        9      protein1    5    SSQKGSRL           MDSKG  LLLLVVSNLL     sample2  477.285954       C0202             3.107375          0.111982            0.168572

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
            "best") or filter (if result is "filtered") results.
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
        if comparison_quantity is None:
            comparison_quantity = "presentation_score"

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

    def save(self, models_dir):
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

        result = cls(
            affinity_predictor=affinity_predictor,
            processing_predictor_with_flanks=processing_predictor_with_flanks,
            processing_predictor_without_flanks=processing_predictor_without_flanks,
            weights_dataframe=weights_dataframe)
        return result
