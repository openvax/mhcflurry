from __future__ import print_function, division, absolute_import
import collections
import logging
import sys
import os
from struct import unpack
from hashlib import sha256

import numpy
import pandas

from . import amino_acid


def set_keras_backend(backend=None, gpu_device_nums=None):
    """
    Configure Keras backend to use GPU or CPU. Only tensorflow is supported.

    Parameters
    ----------
    backend : string, optional
        one of 'tensorflow-default', 'tensorflow-cpu', 'tensorflow-gpu'

    gpu_device_nums : list of int, optional
        GPU devices to potentially use

    """
    os.environ["KERAS_BACKEND"] = "tensorflow"

    if not backend:
        backend = "tensorflow-default"

    if gpu_device_nums is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in gpu_device_nums])

    if backend == "tensorflow-cpu" or gpu_device_nums == []:
        print("Forcing tensorflow/CPU backend.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device_count = {'CPU': 1, 'GPU': 0}
    elif backend == "tensorflow-gpu":
        print("Forcing tensorflow/GPU backend.")
        device_count = {'CPU': 0, 'GPU': 1}
    elif backend == "tensorflow-default":
        print("Forcing tensorflow backend.")
        device_count = None
    else:
        raise ValueError("Unsupported backend: %s" % backend)

    import tensorflow
    from keras import backend as K
    config = tensorflow.ConfigProto(
        device_count=device_count)
    config.gpu_options.allow_growth=True 
    session = tensorflow.Session(config=config)
    K.set_session(session)


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
