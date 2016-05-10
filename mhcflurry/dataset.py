# Copyright (c) 2016. Mount Sinai School of Medicine
#
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

from __future__ import (
    print_function,
    division,
    absolute_import,
)
from collections import defaultdict, OrderedDict
import logging


from six import string_types
import pandas as pd
import numpy as np

from .common import geometric_mean
from .dataset_helpers import (
    prepare_pMHC_affinity_arrays,
    load_dataframe
)
from .peptide_encoding import fixed_length_index_encoding
from .imputation_helpers import (
    check_dense_pMHC_array,
    prune_dense_matrix_and_labels,
    dense_pMHC_matrix_to_nested_dict,
)


class Dataset(object):

    """
    Peptide-MHC binding dataset with helper methods for constructing
    different representations (arrays, DataFrames, dictionaries, &c).

    Design considerations:
        - want to allow multiple measurements for each pMHC pair (which can
          be dynamically combined)
        - optional sample weights associated with each pMHC measurement
    """
    def __init__(self, df):
        """
        Constructs a DataSet from a pandas DataFrame with the following
        columns:
            - allele
            - peptide
            - affinity

        Also, there is an optional column:
            - sample_weight

        If `sample_weight` is missing then it is filled with a default value
        of 1.0

        Parameters
        ----------
        df : pandas.DataFrame
        """
        columns = set(df.columns)

        for expected_column_name in {"allele", "peptide", "affinity"}:
            if expected_column_name not in columns:
                raise ValueError("Missing column '%s' from DataFrame")
        # make allele and peptide columns the index, and copy it
        # so we can add a column without any observable side-effect in
        # the calling code
        df = df.set_index(["allele", "peptide"], drop=False)

        if "sample_weight" not in columns:
            df["sample_weight"] = np.ones(len(df), dtype=float)

        self._df = df
        self._alleles = np.asarray(df["allele"])
        self._peptides = np.asarray(df["peptide"])
        self._affinities = np.asarray(df["affinity"])
        self._sample_weights = np.asarray(df["sample_weight"])

    def to_dataframe(self):
        """
        Returns DataFrame representation of data contained in Dataset
        """
        return self._df

    @property
    def peptides(self):
        """
        Array of peptides from pMHC measurements.
        """
        return self._peptides

    @property
    def alleles(self):
        """
        Array of MHC allele names from pMHC measurements.
        """
        return self._alleles

    @property
    def affinities(self):
        """
        Array of affinities from pMHC measurements.
        """
        return self._affinities

    @property
    def sample_weights(self):
        """
        Array of sample weights for each pMHC measurement.
        """
        return self._sample_weights

    def __len__(self):
        return len(self.to_dataframe())

    def __str__(self):
        return "Dataset(n=%d, alleles=%s)" % (
            len(self), self.unique_alleles())

    def __eq__(self, other):
        """
        Two datasets are equal if they contain the same number of samples
        with the same properties and values.
        """
        if type(other) is not Dataset:
            return False
        elif len(self) != len(other):
            return False
        elif len(self.columns) != len(other.columns):
            return False

        for ci, cj in zip(self.columns, other.columns):
            if ci != cj:
                return False

        self_df = self.to_dataframe()
        other_df = other.to_dataframe()

        for column_name in self.columns:
            if not (self_df[column_name] == other_df[column_name]).all():
                return False
        return True

    def iterrows(self):
        """
        Iterate over tuples containing: (allele, peptide), other_fields
        for each pMHC measurement.
        """
        return self.to_dataframe().iterrows()

    @property
    def columns(self):
        return self.to_dataframe().columns

    def unique_alleles(self):
        return set(self.alleles)

    def unique_peptides(self):
        return set(self.peptides)

    def groupby_allele(self):
        """
        Yields a sequence of tuples of allele names with Datasets containing
        entries just for that allele.
        """
        for (allele_name, group_df) in self.to_dataframe().groupby("allele"):
            yield (allele_name, Dataset(group_df))

    def groupby_allele_dictionary(self):
        """
        Returns dictionary mapping each allele name to a Dataset containing
        only entries from that allele.
        """
        return dict(self.groupby_allele())

    def to_nested_dictionary(self, combine_fn=geometric_mean):
        """
        Returns a dictionary mapping from allele name to a dictionary which
        maps from peptide to measured value. Caution, this eliminates sample
        weights!

        Parameters
        ----------
        combine_fn : function
            How to combine multiple measurements for the same pMHC complex.
            Takes affinities and optional `weights` argument.
        """
        allele_to_peptide_to_affinity_dict = {}
        for allele, allele_dataset in self.groupby_allele():
            # dictionary mapping each peptide to a list of affinities
            peptide_to_affinity_dict = defaultdict(list)
            peptide_to_weight_dict = defaultdict(list)
            for (allele, peptide), row in allele_dataset.iterrows():
                affinity = row["affinity"]
                sample_weight = row["sample_weight"]
                peptide_to_affinity_dict[peptide].append(affinity)
                peptide_to_weight_dict[peptide].append(sample_weight)
            allele_to_peptide_to_affinity_dict[allele] = {
                peptide: combine_fn(
                    peptide_to_affinity_dict[peptide],
                    peptide_to_weight_dict[peptide])
                for peptide in peptide_to_affinity_dict.keys()
            }
        return allele_to_peptide_to_affinity_dict

    @classmethod
    def from_sequences(
            cls,
            alleles,
            peptides,
            affinities,
            sample_weights=None,
            extra_columns={}):
        """
        Parameters
        ----------
        alleles : numpy.ndarray, pandas.Series, or list
            Name of allele for that pMHC measurement

        peptides : numpy.ndarray, pandas.Series, or list
            Sequence of peptide in that pMHC measurement.

        affinities : numpy.ndarray, pandas.Series, or list
            Affinity value (typically IC50 concentration) for that pMHC

        sample_weights : numpy.ndarray of float, optional

        extra_columns : dict
            Dictionary of any extra properties associated with a
            pMHC measurement
        """
        alleles, peptides, affinities, sample_weights = \
            prepare_pMHC_affinity_arrays(
                alleles=alleles,
                peptides=peptides,
                affinities=affinities,
                sample_weights=sample_weights)
        df = pd.DataFrame()
        df["allele"] = alleles
        df["peptide"] = peptides
        df["affinity"] = affinities
        df["sample_weight"] = sample_weights
        for column_name, column in extra_columns.items():
            if len(column) != len(alleles):
                raise ValueError(
                    "Wrong length for column '%s', expected %d but got %d" % (
                        column_name,
                        len(alleles),
                        len(column)))
            df[column_name] = np.asarray(column)
        return cls(df)

    @classmethod
    def from_single_allele_dataframe(cls, allele_name, single_allele_df):
        """
        Construct a Dataset from a single MHC allele's DataFrame
        """
        df = single_allele_df.copy()
        df["allele"] = allele_name
        return cls(df)

    @classmethod
    def from_nested_dictionary(
            cls,
            allele_to_peptide_to_affinity_dict):
        """
        Given nested dictionaries mapping allele -> peptide -> affinity,
        construct a Dataset with uniform sample weights.
        """
        alleles = []
        peptides = []
        affinities = []
        for allele, allele_dict in allele_to_peptide_to_affinity_dict.items():
            for peptide, affinity in allele_dict.items():
                alleles.append(allele)
                peptides.append(peptide)
                affinities.append(affinity)
        return cls.from_sequences(
            alleles=alleles,
            peptides=peptides,
            affinities=affinities)

    @classmethod
    def create_empty(cls):
        """
        Returns an empty Dataset containing no pMHC entries.
        """
        return cls.from_nested_dictionary({})

    @classmethod
    def from_single_allele_dictionary(
            cls,
            allele_name,
            peptide_to_affinity_dict):
        """
        Given a peptide->affinity dictionary for a single allele,
        create a Dataset.
        """
        return cls.from_nested_dictionary({allele_name: peptide_to_affinity_dict})

    @classmethod
    def from_csv(
            cls,
            filename,
            sep=None,
            allele_column_name=None,
            peptide_column_name=None,
            affinity_column_name=None):
        df, allele_column_name, peptide_column_name, affinity_column_name = \
            load_dataframe(
                filename=filename,
                sep=sep,
                allele_column_name=allele_column_name,
                peptide_column_name=peptide_column_name,
                affinity_column_name=affinity_column_name)
        df = df.rename(columns={
            allele_column_name: "allele",
            peptide_column_name: "peptide",
            affinity_column_name: "affinity"})
        return cls(df)

    def get_allele(self, allele_name):
        """
        Get Dataset for a single allele
        """
        if allele_name not in self.unique_alleles():
            raise KeyError("Allele '%s' not found" % (allele_name,))
        df = self.to_dataframe()
        df_allele = df[df.allele == allele_name]
        return self.__class__(df_allele)

    def get_alleles(self, allele_names):
        """
        Restrict Dataset to several allele names.
        """
        datasets = []
        for allele_name in allele_names:
            datasets.append(self.get_allele(allele_name))
        return self.concat(datasets)

    @classmethod
    def concat(cls, datasets):
        """
        Concatenate several datasets into a single object.
        """
        dataframes = [dataset.to_dataframe() for dataset in datasets]
        return cls(pd.concat(dataframes))

    def replace_allele(self, allele_name, new_dataset):
        """
        Replace data for given allele with new entries.
        """
        if allele_name not in self.unique_alleles():
            raise ValueError("Allele '%s' not found" % (allele_name,))
        df = self.to_dataframe()
        df_without = df[df.allele != allele_name]
        new_df = new_dataset.to_dataframe()
        combined_df = pd.concat([df_without, new_df])
        return self.__class__(combined_df)

    def flatmap_peptides(self, peptide_fn):
        """
        Create zero or more peptides from each pMHC entry. The affinity of all
        new peptides is identical to the original, but sample weights are
        divided across the number of new peptides.
        """
        columns = self.to_dataframe().columns
        new_data_dict = OrderedDict(
            (column_name, [])
            for column_name in columns
        )
        if "original_peptide" not in new_data_dict:
            create_original_peptide_column = True
            new_data_dict["original_peptide"] = []

        for (allele, peptide), row in self.iterrows():
            new_peptides = peptide_fn(peptide)
            n = len(new_peptides)
            weight = row["sample_weight"]
            # we're either going to create a fresh original peptide column
            # or extend the existing original peptide tuple that tracks
            # the provenance of entries in the new Dataset
            original_peptide = row.get("original_peptide")
            if original_peptide is None:
                original_peptide = ()
            elif isinstance(original_peptide, string_types):
                original_peptide = (original_peptide,)
            else:
                original_peptide = tuple(original_peptide)

            for new_peptide in new_peptides:
                for column_name in columns:
                    if column_name == "peptide":
                        new_data_dict["peptide"].append(new_peptide)
                    elif column_name == "sample_weight":
                        new_data_dict["sample_weight"].append(weight / n)
                    elif column_name == "original_peptide":
                        new_data_dict["original_peptide"] = original_peptide + (peptide,)
                    else:
                        new_data_dict[column_name].append(row[column_name])
                if create_original_peptide_column:
                    new_data_dict["original_peptide"].append((peptide,))
        df = pd.DataFrame(new_data_dict)
        return self.__class__(df)

    def kmer_index_encoding(
            self,
            kmer_size=9,
            allow_unknown_amino_acids=True):
        """
        Encode peptides in this dataset using a fixed-length vector
        representation.

        Parameters
        ----------
        kmer_size : int
            Length of encoding for each peptide

        allow_unknown_amino_acids : bool
            If True, then extend shorter amino acids using "X" character,
            otherwise fill in all possible combinations of real amino acids.

        Returns:
            - 2d array of encoded kmers
            - 1d array of affinity value corresponding to the source
              peptide for each kmer
            - sample_weights (1 / kmer count per peptide)
            - indices of original peptides from which kmers were extracted
        """
        if len(self.peptides) == 0:
            return (
                np.empty((0, kmer_size), dtype=int),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=int)
            )

        X_index, _, original_peptide_indices, counts = \
            fixed_length_index_encoding(
                peptides=self.peptides,
                desired_length=kmer_size,
                start_offset_shorten=0,
                end_offset_shorten=0,
                start_offset_extend=0,
                end_offset_extend=0,
                allow_unknown_amino_acids=allow_unknown_amino_acids)
        original_peptide_indices = np.asarray(original_peptide_indices)

        counts = np.asarray(counts)
        kmer_affinities = self.affinities[original_peptide_indices]
        kmer_sample_weights = self.sample_weights[original_peptide_indices]

        assert len(original_peptide_indices) == len(kmer_affinities)
        assert len(counts) == len(kmer_affinities)
        assert len(kmer_sample_weights) == len(kmer_affinities)

        # combine the original sample weights of varying length peptides
        # with a 1/n_kmers factor for the number of kmers pulled out of each
        # original peptide
        combined_sample_weights = kmer_sample_weights * (1.0 / counts)
        return X_index, kmer_affinities, combined_sample_weights, original_peptide_indices

    def to_dense_pMHC_affinity_matrix(self):
        """
        Returns a tuple with a dense matrix of affinities, a dense matrix of
        sample weights, a list of peptide labels for each row and a list of
        allele labels for each column.
        """
        allele_to_peptide_to_affinity_dict = self.to_nested_dictionary()
        peptides_list = list(sorted(self.unique_peptides()))
        peptide_order = {p: i for (i, p) in enumerate(peptides_list)}
        n_peptides = len(peptides_list)
        alleles_list = list(sorted(self.unique_alleles()))
        allele_order = {a: i for (i, a) in enumerate(alleles_list)}
        n_alleles = len(alleles_list)
        shape = (n_peptides, n_alleles)
        X = np.ones(shape, dtype=float) * np.nan
        for (allele, allele_dict) in allele_to_peptide_to_affinity_dict.items():
            column_index = allele_order[allele]
            for (peptide, affinity) in allele_dict.items():
                row_index = peptide_order[peptide]
                X[row_index, column_index] = affinity
        return X, peptides_list, alleles_list

    def impute_missing_values(
            self,
            imputation_method,
            log_transform=True,
            min_observations_per_peptide=1,
            min_observations_per_allele=1):
        """
        Synthesize new measurements for missing pMHC pairs using the given
        imputation_method.

        Parameters
        ----------
        imputation_method : object
            Expected to have a method called `complete` which takes a 2d array
            of floats and replaces some or all NaN values with synthetic
            affinities.

        log_transform : function
            Transform affinities with to log10 values before imputation
            (and then transform back afterward).

        min_observations_per_peptide : int
            Drop peptide rows with fewer than this number of observed values.

        min_observations_per_allele : int
            Drop allele columns with fewer than this number of observed values.

        Returns Dataset with original pMHC affinities and additional
        synthetic samples.
        """
        X_incomplete, peptide_list, allele_list = self.to_dense_pMHC_affinity_matrix()

        check_dense_pMHC_array(X_incomplete, peptide_list, allele_list)

        # drop alleles and peptides with small amounts of data
        X_incomplete, peptide_list, allele_list = prune_dense_matrix_and_labels(
            X_incomplete, peptide_list, allele_list,
            min_observations_per_peptide=min_observations_per_peptide,
            min_observations_per_allele=min_observations_per_allele)

        if log_transform:
            X_incomplete = np.log(X_incomplete)

        if np.isnan(X_incomplete).sum() == 0:
            # if all entries in the matrix are already filled in then don't
            # try using an imputation algorithm since it might raise an
            # exception.
            logging.warn("No missing values, using original data instead of imputation")
            X_complete = X_incomplete
        else:
            X_complete = imputation_method.complete(X_incomplete)

        if log_transform:
            X_complete = np.exp(X_complete)

        allele_to_peptide_to_affinity_dict = dense_pMHC_matrix_to_nested_dict(
            X=X_complete,
            peptide_list=peptide_list,
            allele_list=allele_list)
        return self.from_nested_dictionary(allele_to_peptide_to_affinity_dict)
