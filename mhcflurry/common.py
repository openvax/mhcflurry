from __future__ import print_function, division, absolute_import
import collections
import logging
import sys

import numpy
import pandas

from . import amino_acid


def configure_logging(verbose=False):
    """
    Configure logging module using defaults.

    Parameters
    ----------
    verbose : boolean
        If true, output will be at level DEBUG, otherwise, INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(funcName)s:"
        " %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        level=level)


def amino_acid_distribution(peptides, smoothing=0.0):
    """
    Compute the fraction of each amino acid across a collection of peptides.
    
    Parameters
    ----------
    peptides : list of string
    smoothing : float, optional
        Small number (e.g. 0.01) to add to all amino acid fractions. The higher
        the number the more uniform the distribution.

    Returns
    -------
    pandas.Series indexed by amino acids
    """
    peptides = pandas.Series(peptides)
    aa_counts = pandas.Series(peptides.map(collections.Counter).sum())
    normalized = aa_counts / aa_counts.sum()
    if smoothing:
        normalized += smoothing
        normalized /= normalized.sum()
    return normalized


def random_peptides(num, length=9, distribution=None):
    """
    Generate random peptides (kmers).

    Parameters
    ----------
    num : int
        Number of peptides to return

    length : int
        Length of each peptide

    distribution : pandas.Series
        Maps 1-letter amino acid abbreviations to
        probabilities. If not specified a uniform
        distribution is used.

    Returns
    ----------
    list of string

    """
    if num == 0:
        return []
    if distribution is None:
        distribution = pandas.Series(
            1, index=sorted(amino_acid.COMMON_AMINO_ACIDS))
        distribution /= distribution.sum()

    return [
        ''.join(peptide_sequence)
        for peptide_sequence in
        numpy.random.choice(
            distribution.index,
            p=distribution.values,
            size=(int(num), int(length)))
    ]
