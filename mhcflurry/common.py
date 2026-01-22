from __future__ import print_function, division, absolute_import
import collections
import logging
import sys
import os
import json

import numpy
import pandas
from mhcgnomes import parse, Allele, AlleleWithoutGene, Gene


from . import amino_acid


def normalize_allele_name(
        raw_name,
        forbidden_substrings=("MIC", "HFE"),
        raise_on_error=True,
        default_value=None):
    """
    Parses a string into a normalized allele representation.

    Parameters
    ----------
    raw_name : str
        Input string to normalize

    forbidden_substrings : tuple of str
        Fail on inputs which contain any of these strings

    raise_on_error : bool
        If an allele fails to parse raise an exception if this argument is True

    default_value : str or None
        If raise_on_error is False and allele fails to parse, return this value

    Returns
    -------
    str or None
    """
    for forbidden_substring in forbidden_substrings:
        if forbidden_substring in raw_name:
            if raise_on_error:
                raise ValueError("Unsupported gene in MHC allele name: %s" % raw_name)
            else:
                return default_value
    result = parse(
        raw_name,
        only_class1=True,
        required_result_types=[Allele, AlleleWithoutGene, Gene],
        preferred_result_types=[Allele],
        use_allele_aliases=True,
        infer_class2_pairing=False,
        collapse_singleton_haplotypes=True,
        collapse_singleton_serotypes=True,
        raise_on_error=False,
    )
    if result is None:
        if raise_on_error:
            raise ValueError("Invalid MHC allele name: %s" % raw_name)
        else:
            return default_value
    if (
        result.annotation_pseudogene
        or result.annotation_null
        or result.annotation_questionable
    ):
        if raise_on_error:
            raise ValueError("Unsupported annotation on MHC allele: %s" % raw_name)
        else:
            return default_value
    return result.restrict_allele_fields(2).to_string()


PYTORCH_CONFIGURED = False


def configure_pytorch(gpu_device_nums=None, num_threads=None):
    """
    Configure PyTorch to use GPU or CPU.

    Parameters
    ----------
    gpu_device_nums : list of int, optional
        GPU devices to potentially use

    num_threads : int, optional
        Number of threads for PyTorch operations

    """
    import torch

    global PYTORCH_CONFIGURED

    if PYTORCH_CONFIGURED:
        return

    PYTORCH_CONFIGURED = True

    # Configure GPU devices
    if gpu_device_nums is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_device_nums))

    # Configure number of threads
    if num_threads:
        torch.set_num_threads(num_threads)


def get_pytorch_device():
    """
    Get the appropriate PyTorch device (GPU if available, else CPU).

    By default, MPS (Apple Metal) is disabled due to known issues with
    subprocess-based training and placeholder tensor allocation in embedding
    layers. Set MHCFLURRY_ENABLE_MPS=1 to enable MPS.

    Returns
    -------
    torch.device
    """
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif (os.environ.get('MHCFLURRY_ENABLE_MPS', '0') == '1' and
          hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return torch.device('mps')
    else:
        return torch.device('cpu')


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
        level=level,
    )



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
        distribution = pandas.Series(1, index=sorted(amino_acid.COMMON_AMINO_ACIDS))
        distribution /= distribution.sum()

    return [
        "".join(peptide_sequence)
        for peptide_sequence in numpy.random.choice(
            distribution.index, p=distribution.values, size=(int(num), int(length))
        )
    ]


def positional_frequency_matrix(peptides):
    """
    Given a set of peptides, calculate a length x amino acids frequency matrix.

    Parameters
    ----------
    peptides : list of string
        All of same length

    Returns
    -------
    pandas.DataFrame
        Index is position, columns are amino acids
    """
    length = len(peptides[0])
    assert all(len(peptide) == length for peptide in peptides)
    counts = pandas.DataFrame(
        index=[a for a in amino_acid.BLOSUM62_MATRIX.index if a != "X"],
        columns=numpy.arange(1, length + 1),
    )
    for i in range(length):
        counts[i + 1] = pandas.Series([p[i] for p in peptides]).value_counts()
    result = (counts / len(peptides)).fillna(0.0).T
    result.index.name = "position"
    return result


def save_weights(weights_list, filename):
    """
    Save model weights to the given filename using numpy's ".npz" format.

    Parameters
    ----------
    weights_list : list of numpy array

    filename : string
    """
    numpy.savez(
        filename, **dict((("array_%d" % i), w) for (i, w) in enumerate(weights_list))
    )


def load_weights(filename):
    """
    Restore model weights from the given filename, which should have been
    created with `save_weights`.

    Parameters
    ----------
    filename : string

    Returns
    ----------
    list of array
    """
    with numpy.load(filename) as loaded:
        weights = [loaded["array_%d" % i] for i in range(len(loaded.keys()))]
    return weights


class NumpyJSONEncoder(json.JSONEncoder):
    """
    JSON encoder (used with json module) that can handle numpy arrays.
    """

    def default(self, obj):
        if isinstance(
            obj,
            (
                numpy.int_,
                numpy.intc,
                numpy.intp,
                numpy.int8,
                numpy.int16,
                numpy.int32,
                numpy.int64,
                numpy.uint8,
                numpy.uint16,
                numpy.uint32,
                numpy.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(
            obj, (numpy.float_, numpy.float16, numpy.float32, numpy.float64)
        ):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
