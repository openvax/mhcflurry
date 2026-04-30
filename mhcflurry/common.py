import collections
import datetime
import logging
import shlex
import sys
import os
import json
import warnings

import numpy
import pandas
from mhcgnomes import parse, Allele, AlleleWithoutGene, Gene


from . import amino_acid


def normalize_allele_name(
        raw_name,
        forbidden_substrings=("MIC", "HFE"),
        raise_on_error=True,
        default_value=None,
        use_allele_aliases=True):
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

    use_allele_aliases : bool
        If True, use mhcgnomes allele alias table (IMGT historical name
        reassignments). Some old allele names (e.g. B*44:01, Cw*0201) were
        retired by IMGT when the original sequences were found to contain
        errors. Defaults to False to preserve current IMGT nomenclature;
        the pseudosequence loading code explicitly handles aliases with
        fallback logic.

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
        use_allele_aliases=use_allele_aliases,
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


def filter_canonicalizable_alleles(alleles, log_label="alleles"):
    """Drop alleles that ``normalize_allele_name`` refuses to canonicalize.

    ``predictor.supported_alleles`` (and any user-supplied
    ``--alleles-file``) can contain pseudogenes / null alleles /
    questionable annotations — those are real entries in the public
    ``allele_sequences.csv`` (which aims to be exhaustive), but
    ``predict_to_dataframe`` raises on them mid-iteration. Filter
    once up front so iteration doesn't crash partway through, and
    log the dropped sample so the missing rows in the resulting
    output table are explainable.

    A local memo keeps the per-allele ``normalize_allele_name`` cost
    bounded — the predictor's allele set is ~20K entries and
    mhcgnomes' parse is millisecond-scale, so without caching this
    pre-pass alone would add ~20 sec of single-threaded startup.
    """
    seen = {}

    def _ok(allele):
        if allele not in seen:
            try:
                normalize_allele_name(allele)
                seen[allele] = True
            except (ValueError, TypeError):
                seen[allele] = False
        return seen[allele]

    filtered = []
    dropped = []
    for a in alleles:
        (filtered if _ok(a) else dropped).append(a)
    if dropped:
        sample = ", ".join(dropped[:5]) + (
            ", ..." if len(dropped) > 5 else ""
        )
        print(
            "Skipping %d %s that fail canonicalization "
            "(pseudogene/null/questionable): %s"
            % (len(dropped), log_label, sample)
        )
    return filtered


_pytorch_backend = "auto"
_PYTORCH_BACKEND_ALIASES = {
    "default": "auto",
}
_TENSORFLOW_BACKEND_ALIASES = {
    "tensorflow": "auto",
    "tensorflow-default": "auto",
    "tensorflow-gpu": "gpu",
    "tensorflow-cpu": "cpu",
}
_VALID_PYTORCH_BACKENDS = ("auto", "gpu", "mps", "cpu")


def normalize_pytorch_backend(backend):
    """
    Normalize a requested backend name and validate it.

    Parameters
    ----------
    backend : str or None

    Returns
    -------
    str or None
    """
    if backend is None:
        return None
    backend = _PYTORCH_BACKEND_ALIASES.get(backend, backend)
    if backend not in _VALID_PYTORCH_BACKENDS:
        raise ValueError(
            "Invalid backend %r. Expected one of: %s" % (
                backend,
                ", ".join(_VALID_PYTORCH_BACKENDS),
            )
        )
    return backend


def configure_pytorch(backend=None, gpu_device_nums=None, num_threads=None):
    """
    Configure PyTorch device backend and threading.

    Can be called multiple times. Each call updates the settings provided.

    Parameters
    ----------
    backend : str, optional
        Device backend: "auto", "gpu", "mps", or "cpu".
        "auto" selects the best available device (GPU > MPS > CPU).
    gpu_device_nums : list of int, optional
        CUDA devices to expose via CUDA_VISIBLE_DEVICES. An empty list hides
        CUDA entirely for the current process.
    num_threads : int, optional
        Number of threads for PyTorch operations
    """
    global _pytorch_backend

    if backend is not None:
        _pytorch_backend = normalize_pytorch_backend(backend)

    if gpu_device_nums is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_device_nums))

    if num_threads:
        import torch
        torch.set_num_threads(num_threads)


def configure_tensorflow(backend=None, gpu_device_nums=None, num_threads=None):
    """
    Backward-compatible configuration entry point from the TF backend era.

    Parameters
    ----------
    backend : str, optional
        Legacy backend value retained for API compatibility. TensorFlow-era
        names such as "tensorflow-cpu" are translated to the equivalent
        PyTorch backend and emit a deprecation warning.
    gpu_device_nums : list of int, optional
        GPU devices to potentially use.
    num_threads : int, optional
        Number of threads for backend operations.
    """
    translated_backend = None
    if backend is not None:
        translated_backend = _TENSORFLOW_BACKEND_ALIASES.get(backend)
        if translated_backend is not None:
            warnings.warn(
                (
                    "configure_tensorflow(backend=%r) is deprecated; "
                    "using PyTorch backend=%r. Use configure_pytorch() instead."
                ) % (backend, translated_backend),
                FutureWarning,
                stacklevel=2,
            )
        else:
            translated_backend = normalize_pytorch_backend(backend)
    configure_pytorch(
        backend=translated_backend,
        gpu_device_nums=gpu_device_nums,
        num_threads=num_threads,
    )


def get_pytorch_device():
    """
    Get the PyTorch device based on the backend set by ``configure_pytorch``.

    Returns
    -------
    torch.device
    """
    import torch

    backend = _pytorch_backend

    if backend == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Backend 'gpu' requested but CUDA is not available")
        return torch.device('cuda')
    elif backend == "mps":
        if not (hasattr(torch.backends, 'mps') and
                torch.backends.mps.is_available()):
            raise RuntimeError(
                "Backend 'mps' requested but MPS is not available")
        return torch.device('mps')
    elif backend == "cpu":
        return torch.device('cpu')
    else:
        # auto: GPU > MPS > CPU
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif (hasattr(torch.backends, 'mps') and
              torch.backends.mps.is_available()):
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


def random_peptides(num, length=9, distribution=None, rng=None):
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

    rng : numpy.random.Generator, optional
        If provided, peptides are sampled from this generator instead
        of the global ``numpy.random`` state. Lets callers make peptide
        generation deterministic (e.g. for the pre-generated-negative
        pool in Phase 1 of #268). When None the global state is used,
        preserving pre-existing call-site semantics.

    Returns
    ----------
    list of string

    """
    if num == 0:
        return []
    if distribution is None:
        distribution = pandas.Series(1, index=sorted(amino_acid.COMMON_AMINO_ACIDS))
        distribution /= distribution.sum()

    if rng is None:
        sampled = numpy.random.choice(
            distribution.index, p=distribution.values, size=(int(num), int(length))
        )
    else:
        sampled = rng.choice(
            distribution.index.to_numpy(),
            p=distribution.values,
            size=(int(num), int(length)),
        )
    return ["".join(peptide_sequence) for peptide_sequence in sampled]


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


_GENERATE_SH_TEMPLATE = """\
#!/usr/bin/env bash
# Auto-generated by mhcflurry {version} on {timestamp}.
#
# Run this script with no arguments to produce a tar.gz snapshot of
# the artifact directory it lives in:
#
#     bash GENERATE.sh
#
# Pass `regenerate` as the first argument to re-run the originating
# mhcflurry command (which recreates the directory) and then snapshot:
#
#     bash GENERATE.sh regenerate
#
# The originating command is recorded in ORIGINAL_COMMAND below; paths
# inside it may need adjustment if you run from a different machine.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
NAME="$(basename "$HERE")"
PARENT="$(dirname "$HERE")"

ORIGINAL_CWD={cwd}
ORIGINAL_COMMAND=(
    {cmd}
)

if [ "${{1:-}}" = "regenerate" ]; then
    # Safety checks before destroying the artifact dir:
    #   * $HERE must resolve to a directory we own,
    #   * not be the filesystem root or $HOME,
    #   * and contain the GENERATE.sh we are running -- the canonical
    #     marker that this dir was produced by mhcflurry. Without these
    #     checks, a user re-running GENERATE.sh from inside an already-
    #     replaced mountpoint, or with $HERE corrupted by a path edit,
    #     could rm -rf an unrelated tree.
    if [ -z "$HERE" ] || [ "$HERE" = "/" ] || [ "$HERE" = "$HOME" ]; then
        echo "regenerate: refusing to rm -rf suspicious path: $HERE" >&2
        exit 1
    fi
    if [ ! -d "$HERE" ] || [ ! -O "$HERE" ]; then
        echo "regenerate: $HERE is not a directory we own" >&2
        exit 1
    fi
    if [ ! -f "$HERE/GENERATE.sh" ]; then
        echo "regenerate: GENERATE.sh missing in $HERE -- refusing" >&2
        exit 1
    fi
    cd "$ORIGINAL_CWD"
    rm -rf "$HERE"
    "${{ORIGINAL_COMMAND[@]}}"
fi

TARBALL="$PARENT/$NAME.tar.gz"
echo "Snapshotting $HERE -> $TARBALL ..."
tar czf "$TARBALL" -C "$PARENT" "$NAME"
echo "Snapshot complete: $TARBALL"
"""


def write_generate_sh(out_dir, argv=None, mhcflurry_version=None):
    """
    Write a GENERATE.sh in ``out_dir`` recording how to reproduce the
    artifact directory and snapshot it as a tar.gz.

    Called by every artifact-producing CLI (train/select/calibrate)
    so each output directory ships a one-liner snapshot script.

    Parameters
    ----------
    out_dir : str
        Directory to write GENERATE.sh into. Created by the caller.
    argv : list of str, optional
        Command-line invocation to record. Defaults to ``sys.argv``.
    mhcflurry_version : str, optional
        Version string for the audit comment. Defaults to current.

    Returns
    -------
    str
        Absolute path to the GENERATE.sh that was written.
    """
    if argv is None:
        argv = list(sys.argv)
    if mhcflurry_version is None:
        try:
            from mhcflurry.version import __version__ as mhcflurry_version
        except Exception:
            mhcflurry_version = "unknown"

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    cmd_lines = " \\\n    ".join(shlex.quote(a) for a in argv)
    cwd = shlex.quote(os.getcwd())

    contents = _GENERATE_SH_TEMPLATE.format(
        version=mhcflurry_version,
        timestamp=timestamp,
        cwd=cwd,
        cmd=cmd_lines,
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "GENERATE.sh")
    with open(path, "w") as f:
        f.write(contents)
    os.chmod(path, 0o755)
    return path
