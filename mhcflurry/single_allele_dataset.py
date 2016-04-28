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
from collections import OrderedDict
from itertools import groupby

import numpy as np
from six import string_types, integer_types
from typechecks import require_iterable_of, require_instance

from .dataset_helpers import (
    prepare_peptides_array,
    prepare_alleles_array,
    prepare_ic50_values_array,
    transform_ic50_values_into_regression_outputs,
    infer_csv_separator
    combine_multiple_peptide_ic50_measurements,
    load_dataframe,
)
from .class1_allele_specific_hyperparameters import MAX_IC50
from .common import normalize_allele_name

class SingleAlleleDataset(object):
    """
    Meant to expand on the functionality of this namedtuple:
    AlleleData = namedtuple(
        "AlleleData",
        [
            "X_index",    # index-based featue encoding of fixed length peptides
            "X_binary",  # binary encoding of fixed length peptides
            "Y",     # regression encoding of IC50 (log scaled between 0..1)
            "peptides",  # list of fixed length peptide string
            "ic50",      # IC50 value associated with each entry
            "original_peptides",  # original peptides may be of different lengths
            "original_lengths",   # len(original_peptide)
            "substring_counts",   # how many substrings were extracted from
                                  # each original peptide string
            "weights",    # 1.0 / count
        ])
    """

    def __init__(
            self,
            allele_name,
            peptides,
            ic50_values,
            weights=None,
            max_ic50=50000,
            kmer_size=9,
            combine_redundant_measurements=False):
        """
        Parameters
        ----------
        allele_name : str
            Name of the allele for which these pMHC affinity measurements
            were made.

        peptides : sequence of str
            Sequences of peptides evaluated.

        ic50_values : sequence of float
            Affinities between peptides and this allele.

        max_ic50 : float
            Used for converting to rescaled values between 0 and 1
            (everything greater than the max IC50 value gets mapped to 0)

        kmer_size : int
            When building fixed length representations of the peptides,
            which length encoding to use.

        combine_redundant_measurements : bool
            If the same peptide has two measured values, should they
            both be included in the dataset or should they be combined
            into the geometric mean of their IC50 values?
        """
        self.allele_name = normalize_allele_name(allele_name)
        self.max_ic50 = max_ic50
        self.kmer_size = kmer_size
        self.original_peptides = prepare_peptides_array(peptides=peptides)
        self.original_ic50_values = prepare_ic50_values_array(
            ic50_values=ic50_values,
            required_count=len(self.peptides))
        # since data may contain multiple measurements for the same pMHC,
        # combine them using the geometric mean of the measured IC50 values
        self.peptides, self.ic50_values = \
            combine_multiple_peptide_ic50_measurements(
                peptides=self.original_peptides,
                ic50_values=self.original_ic50_values)

        self.regression_targets = transform_ic50_values_into_regression_outputs(
            ic50_values=self.ic50_values,
            max_ic50=self.max_ic50)

    def peptide_count(self):
        """
        Returns number of distinct peptides in this dataset
        (not k-mers extracted from peptides)
        """
        return len(self.peptides)

    def kmer_count(self):
        return len(self.kmers)

    @property
    def peptide_to_ic50_dict(self):
        """
        Returns dictionary mapping each peptide to an IC50 value.
        """
        return OrderedDict([
            (p, ic50) for (p, ic50) in zip(self.peptides, self.ic50_values)])

    @property
    def peptide_to_rescaled_affinity_dict(self):
        """
        Returns dictionary mapping each peptide to a value between [0, 1]
        where 0 means the IC50 was self.max_ic50 or greater and 1 means a strong
        binder.
        """
        affinities =
        return OrderedDict([
            p, tran])

    @property
    def kmers(self):