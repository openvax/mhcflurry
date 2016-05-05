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

from typechecks import require_instance
import numpy as np
from math import exp, log

def geometric_mean(xs, weights=None):
    """
    Geometric mean of a collection of affinity measurements, with optional
    sample weights.
    """
    if len(xs) == 1:
        return xs[0]
    elif weights is None:
        return exp(sum(log(xi) for xi in xs))
    sum_weighted_log = sum(log(xi) * wi for (xi, wi) in zip(xs, weights))
    denom = sum(weights)
    return exp(sum_weighted_log / denom)

def check_pMHC_affinity_arrays(alleles, peptides, affinities, sample_weights):
    """
    Make sure that we have the same number of peptides, affinity values,
    and weights.
    """
    require_instance(alleles, np.ndarray)
    require_instance(peptides, np.ndarray)
    require_instance(affinities, np.ndarray)
    require_instance(sample_weights, np.ndarray)

    if len(alleles.shape) != 1:
        raise ValueError("Expected 1d array of alleles but got shape %s" % (
            alleles.shape,))
    if len(peptides.shape) != 1:
        raise ValueError("Expected 1d array of peptides but got shape %s" % (
            peptides.shape,))
    if len(affinities.shape) != 1:
        raise ValueError("Expected 1d array of affinity values but got shape %s" % (alleles.shape,))
    if len(sample_weights.shape) != 1:
        raise ValueError("Expected 1d array of sample weights but got shape %s" % (
            sample_weights.shape,))

    n = len(alleles)
    if len(peptides) != n:
        raise ValueError("Expected %d peptides but got %d" % (n, len(peptides)))
    if len(affinities) != n:
        raise ValueError("Expected %d affinity values but got %d" % (n, len(affinities)))
    if len(sample_weights) != n:
        raise ValueError("Expected %d sample weights but got %d" % (n, len(sample_weights)))


def prepare_pMHC_affinity_arrays(alleles, peptides, affinities, sample_weights=None):
    """
    Converts every sequence to an array and if sample_weights is missing then
    create an array of ones.
    """
    alleles = np.asarray(alleles)
    peptides = np.asarray(peptides)
    affinities = np.asarray(affinities)
    if sample_weights is None:
        sample_weights = np.ones(len(alleles), dtype=float)
    check_pMHC_affinity_arrays(
        alleles=alleles,
        peptides=peptides,
        affinities=affinities,
        sample_weights=sample_weights)
    return alleles, peptides, affinities, sample_weights
