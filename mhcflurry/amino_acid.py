"""
Functions for encoding fixed length sequences of amino acids into various
vector representations, such as one-hot and BLOSUM62.
"""

import collections
import warnings
from copy import copy
from io import StringIO

import pandas


COMMON_AMINO_ACIDS = collections.OrderedDict(sorted({
    "A": "Alanine",
    "R": "Arginine",
    "N": "Asparagine",
    "D": "Aspartic Acid",
    "C": "Cysteine",
    "E": "Glutamic Acid",
    "Q": "Glutamine",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "L": "Leucine",
    "K": "Lysine",
    "M": "Methionine",
    "F": "Phenylalanine",
    "P": "Proline",
    "S": "Serine",
    "T": "Threonine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
    "V": "Valine",
}.items()))
COMMON_AMINO_ACIDS_WITH_UNKNOWN = copy(COMMON_AMINO_ACIDS)
COMMON_AMINO_ACIDS_WITH_UNKNOWN["X"] = "Unknown"

AMINO_ACID_INDEX = dict(
    (letter, i) for (i, letter) in enumerate(COMMON_AMINO_ACIDS_WITH_UNKNOWN))

for (letter, i) in list(AMINO_ACID_INDEX.items()):
    AMINO_ACID_INDEX[letter.lower()] = i  # Support lower-case as well.

AMINO_ACIDS = list(COMMON_AMINO_ACIDS_WITH_UNKNOWN.keys())

BLOSUM62_MATRIX = pandas.read_csv(StringIO("""
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  X
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3  0
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  0
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  0
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1  0
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  0
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3  0
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3  0
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1  0
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1  0
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1  0
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2  0
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3  0
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1  0
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4  0
X  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
"""), sep=r'\s+').loc[AMINO_ACIDS, AMINO_ACIDS].astype("int8")
assert (BLOSUM62_MATRIX == BLOSUM62_MATRIX.T).all().all()

# Five Atchley physicochemical factors from Atchley et al. 2005
# (PNAS 102:6395-6400). Columns are, respectively: polarity /
# accessibility / hydrophobicity, secondary-structure propensity,
# molecular size, codon composition, and electrostatic charge. X is
# represented as zeros, matching the "unknown/no signal" convention
# used by the BLOSUM62 X row.
ATCHLEY_FACTORS = pandas.DataFrame.from_dict({
    "A": [-0.591, -1.302, -0.733, 1.570, -0.146],
    "R": [1.538, -0.055, 1.502, 0.440, 2.897],
    "N": [0.945, 0.828, 1.299, -0.169, 0.933],
    "D": [1.050, 0.302, -3.656, -0.259, -3.242],
    "C": [-1.343, 0.465, -0.862, -1.020, -0.255],
    "E": [1.357, -1.453, 1.477, 0.113, -0.837],
    "Q": [0.931, -0.179, -3.005, -0.503, -1.853],
    "G": [-0.384, 1.652, 1.330, 1.045, 2.064],
    "H": [0.336, -0.417, -1.673, -1.474, -0.078],
    "I": [-1.239, -0.547, 2.131, 0.393, 0.816],
    "L": [-1.019, -0.987, -1.505, 1.266, -0.912],
    "K": [1.831, -0.561, 0.533, -0.277, 1.648],
    "M": [-0.663, -1.524, 2.219, -1.005, 1.212],
    "F": [-1.006, -0.590, 1.891, -0.397, 0.412],
    "P": [0.189, 2.081, -1.628, 0.421, -1.392],
    "S": [-0.228, 1.399, -4.760, 0.670, -2.647],
    "T": [-0.032, 0.326, 2.213, 0.908, 1.313],
    "W": [-0.595, 0.009, 0.672, -2.128, -0.184],
    "Y": [0.260, 0.830, 3.097, -0.838, 1.512],
    "V": [-1.337, -0.279, -0.544, 1.242, -1.262],
    "X": [0.0, 0.0, 0.0, 0.0, 0.0],
}, orient="index", columns=[
    "atchley_polarity",
    "atchley_secondary_structure",
    "atchley_molecular_size",
    "atchley_codon_composition",
    "atchley_electrostatic_charge",
]).loc[AMINO_ACIDS].astype("float32")

ENCODING_DATA_FRAMES = {
    "BLOSUM62": BLOSUM62_MATRIX,
    "one-hot": pandas.DataFrame([
        [1 if i == j else 0 for i in range(len(AMINO_ACIDS))]
        for j in range(len(AMINO_ACIDS))
    ], index=AMINO_ACIDS, columns=AMINO_ACIDS),
    "physchem": ATCHLEY_FACTORS,
    "atchley": ATCHLEY_FACTORS,
}
_COMPOSITE_ENCODING_DATA_FRAMES = {}


def get_vector_encoding_df(name):
    """
    Return the amino-acid vector encoding table for ``name``.

    ``name`` may be a base encoding such as ``"BLOSUM62"`` or
    ``"physchem"``, or a ``+``-joined composite such as
    ``"BLOSUM62+physchem"``. Composite encodings concatenate the component
    columns in order and keep the same amino-acid row order.
    """
    if name in ENCODING_DATA_FRAMES:
        return ENCODING_DATA_FRAMES[name]
    if not isinstance(name, str) or "+" not in name:
        raise KeyError(name)
    if name not in _COMPOSITE_ENCODING_DATA_FRAMES:
        parts = [part.strip() for part in name.split("+") if part.strip()]
        if len(parts) < 2:
            raise KeyError(name)
        frames = [
            get_vector_encoding_df(part).loc[AMINO_ACIDS].add_prefix(
                "%s:" % part
            )
            for part in parts
        ]
        composite = pandas.concat(
            frames,
            axis=1,
        )
        if any(frame.values.dtype.kind == "f" for frame in frames):
            composite = composite.astype("float32")
        _COMPOSITE_ENCODING_DATA_FRAMES[name] = composite
    return _COMPOSITE_ENCODING_DATA_FRAMES[name]


def available_vector_encodings():
    """
    Return list of registered amino acid vector encodings.

    ``get_vector_encoding_df`` also accepts ``+``-joined composites of
    these names, for example ``"BLOSUM62+physchem"``.

    Returns
    -------
    list of string

    """
    return list(ENCODING_DATA_FRAMES)


def vector_encoding_length(name):
    """
    Return the length of the given vector encoding.

    Parameters
    ----------
    name : string

    Returns
    -------
    int
    """
    return get_vector_encoding_df(name).shape[1]


def index_encoding(sequences, letter_to_index_dict):
    """
    Encode a sequence of same-length strings to a matrix of integers of the
    same shape. The map from characters to integers is given by
    `letter_to_index_dict`.

    Given a sequence of `n` strings all of length `k`, return a `k * n` array where
    the (`i`, `j`)th element is `letter_to_index_dict[sequence[i][j]]`.

    Parameters
    ----------
    sequences : list of length n of strings of length k
    letter_to_index_dict : dict : string -> int

    Returns
    -------
    numpy.array of integers with shape (`k`, `n`)
    """
    df = pandas.DataFrame(iter(s) for s in sequences)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning,
                                message=".*Downcasting.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message=".*no_silent_downcasting.*")
        result = df.replace(letter_to_index_dict).infer_objects()
    return result.values


def fixed_vectors_encoding(index_encoded_sequences, letter_to_vector_df):
    """
    Given a `n` x `k` matrix of integers such as that returned by `index_encoding()` and
    a dataframe mapping each index to an arbitrary vector, return a `n * k * m`
    array where the (`i`, `j`)'th element is `letter_to_vector_df.iloc[sequence[i][j]]`.

    The dataframe index and columns names are ignored here; the indexing is done
    entirely by integer position in the dataframe.

    Parameters
    ----------
    index_encoded_sequences : `n` x `k` array of integers

    letter_to_vector_df : pandas.DataFrame of shape (`alphabet size`, `m`)

    Returns
    -------
    numpy.array of integers with shape (`n`, `k`, `m`)
    """
    (num_sequences, sequence_length) = index_encoded_sequences.shape
    target_shape = (
        num_sequences, sequence_length, letter_to_vector_df.shape[1])
    result = letter_to_vector_df.iloc[
        index_encoded_sequences.reshape((-1,))  # reshape() avoids copy
    ].values.reshape(target_shape)
    return result
