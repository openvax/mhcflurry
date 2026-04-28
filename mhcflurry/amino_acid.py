"""
Functions for encoding fixed length sequences of amino acids into various
vector representations, such as one-hot and BLOSUM62.
"""

import collections
import warnings
from copy import copy
from io import StringIO

import numpy
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

# Single canonical letter<->index mapping for the whole codebase. Position
# of a letter in ``AMINO_ACIDS`` equals ``AMINO_ACID_INDEX[letter]``; X is
# at index 20. Anything that round-trips between peptide strings and
# integer index arrays must go through these helpers — that is the
# contract device-resident encoding paths rely on for re-materializing
# peptides from index tensors.

X_INDEX = AMINO_ACID_INDEX["X"]
"""Canonical index of the X (unknown) amino acid; equal to
``len(COMMON_AMINO_ACIDS)``."""

NUM_COMMON_AMINO_ACIDS = len(COMMON_AMINO_ACIDS)
"""Count of the 20 common amino acids (excludes X)."""


def peptide_to_indices(peptide, dtype="int8"):
    """
    Convert a peptide string to a 1-D integer index array.

    Uppercases the input and uses :data:`AMINO_ACID_INDEX` for the
    letter→index map so the result matches what
    :func:`index_encoding` produces for one row.

    Parameters
    ----------
    peptide : str
    dtype : numpy dtype, default ``"int8"``
        Index payloads fit comfortably in int8 (alphabet size 21 ≪ 127).

    Returns
    -------
    numpy.ndarray of shape ``(len(peptide),)``
    """
    return numpy.asarray(
        [AMINO_ACID_INDEX[char] for char in peptide.upper()],
        dtype=dtype,
    )


def indices_to_peptide(indices):
    """
    Convert a 1-D integer index array (or sequence) back to its peptide
    string.

    Inverse of :func:`peptide_to_indices` for canonical alphabet
    members. Out-of-range indices raise :class:`IndexError`.

    Parameters
    ----------
    indices : iterable of int

    Returns
    -------
    str
    """
    return "".join(AMINO_ACIDS[int(i)] for i in indices)

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


def _matrix_with_unknown(df, dtype="float32"):
    """
    Extend a 20 x 20 amino-acid matrix with an all-zero X row and column.
    """
    common = list(COMMON_AMINO_ACIDS)
    if set(df.index) != set(common) or set(df.columns) != set(common):
        raise ValueError(
            "Expected matrix over common amino acids %s; got rows=%s cols=%s"
            % (common, list(df.index), list(df.columns))
        )
    result = df.loc[common, common].copy()
    result.loc["X"] = 0.0
    result["X"] = 0.0
    return result.loc[AMINO_ACIDS, AMINO_ACIDS].astype(dtype)


PMBEC_MATRIX = _matrix_with_unknown(pandas.read_csv(StringIO("""
   A C D E F G H I K L M N P Q R S T V W Y
A 0.322860152036 0.0113750373506 -0.0156239175966 -0.00259952715456 -0.0508792185716 0.0382679273874 -0.0832539299638 -0.00196691041626 -0.0103729638696 -0.042393907322 -0.0651042403697 -0.0853704925231 0.0757409633086 -0.0483151514798 -0.0136431408498 0.038455041596 0.0520376087986 0.081101427454 -0.125564718844 -0.0747500389698
C 0.0113750373506 0.100680270274 0.0102951033136 0.0147570340938 0.0345785831581 0.00933463557214 -0.00750101609651 0.00476007239717 -0.0459237939975 -0.0182998264075 -0.0155971113182 0.0021128481374 -0.00860770840682 -0.0309903425175 -0.0482562439545 -0.0217965163697 -0.0227322740574 -0.0154276574266 0.0412325888637 0.00600631739163
D -0.0156239175966 0.0102951033136 0.157208255034 0.0724279735923 -0.0189545540921 -0.00870389879389 -0.0180188107498 -0.0283467966687 -0.0634240071162 -0.0279979457557 -0.0241192288182 0.0194310374127 0.042784078891 0.000437307476866 -0.0591268568576 -0.0104660502173 0.00656101264316 -0.0193560886308 0.00415097887978 -0.0191575919464
E -0.00259952715456 0.0147570340938 0.0724279735923 0.131775168933 -0.00519060032543 -0.00547805492393 -0.0335600821273 -0.0135417817213 -0.069471604426 0.00353800457557 -0.017166710134 0.00534055417468 0.022589833552 0.0281404974641 -0.0697402405064 -0.0172364513778 -0.0054830504799 -0.00806269508269 -0.00791955104235 -0.0231187170833
F -0.0508792185716 0.0345785831581 -0.0189545540921 -0.00519060032543 0.259179996995 -0.00445131805782 -0.00639743486807 0.0628717025094 -0.049227253611 0.0488666377736 0.0315353570161 -0.0223593028205 -0.0919732521492 -0.0930189756622 -0.0626297946351 -0.0868415233743 -0.0777292391855 -0.015794520965 0.0625957490761 0.0858189617896
G 0.0382679273874 0.00933463557214 -0.00870389879389 -0.00547805492393 -0.00445131805782 0.122499934434 -0.025558278086 -0.0207027221208 -0.0137316756786 -0.0326424665142 -0.0264215095016 -0.00403752148352 0.0094352664965 0.00425299048772 -0.0232280105465 0.0304312733191 0.00861853592388 -0.0127217072682 -0.0246539147339 -0.0205094859119
H -0.0832539299638 -0.00750101609651 -0.0180188107498 -0.0335600821273 -0.00639743486807 -0.025558278086 0.207657765989 -0.0888505073496 0.0761447053198 -0.0351727012494 -0.000760393877348 0.0353903619255 -0.0682087048807 -0.00886454093107 0.109052662874 0.00938179429131 -0.0234122309305 -0.0870188771708 0.0123622841944 0.0365879336865
I -0.00196691041626 0.00476007239717 -0.0283467966687 -0.0135417817213 0.0628717025094 -0.0207027221208 -0.0888505073496 0.27773187827 -0.0381642025534 0.0886112938313 0.0551293441776 -0.0593694184462 -0.039207153398 -0.0626883806129 -0.110160438997 -0.0618078497671 -0.0339233811197 0.091300054417 0.00138488610169 -0.0230596885329
K -0.0103729638696 -0.0459237939975 -0.0634240071162 -0.069471604426 -0.049227253611 -0.0137316756786 0.0761447053198 -0.0381642025534 0.273355694189 -0.0177282663533 -0.00817300753785 -0.0339854863013 -0.0484016323395 -0.0331603641198 0.21516548555 0.00476731287861 -0.0331318604828 -0.0400367780545 -0.0598522551401 -0.00464804635586
L -0.042393907322 -0.0182998264075 -0.0279979457557 0.00353800457557 0.0488666377736 -0.0326424665142 -0.0351727012494 0.0886112938313 -0.0177282663533 0.162738321535 0.0750528874999 -0.0111666419731 -0.051023845781 -0.00134001844501 -0.074934598492 -0.0584956357369 -0.031311528799 0.0449678806271 -0.00754567267671 -0.0137219703366
M -0.0651042403697 -0.0155971113182 -0.0241192288182 -0.017166710134 0.0315353570161 -0.0264215095016 -0.000760393877348 0.0551293441776 -0.00817300753785 0.0750528874999 0.156957428383 0.00753829785887 -0.091647674076 0.00190198496329 -0.0257018542091 -0.0295349216339 -0.0454820084051 -0.0120310888206 0.0210041287765 0.0126203200265
N -0.0853704925231 0.0021128481374 0.0194310374127 0.00534055417468 -0.0223593028205 -0.00403752148352 0.0353903619255 -0.0593694184462 -0.0339854863013 -0.0111666419731 0.00753829785887 0.151487423988 -0.0106077901881 0.0413965183445 -0.0338327997913 0.0170820288313 -0.00295174153884 -0.0436807942705 0.0296409813073 -0.00205806264345
P 0.0757409633086 -0.00860770840682 0.042784078891 0.022589833552 -0.0919732521492 0.0094352664965 -0.0682087048807 -0.039207153398 -0.0484016323395 -0.051023845781 -0.091647674076 -0.0106077901881 0.354629507834 0.0481497903134 -0.0377142358446 -0.00687173098621 0.0199181111388 0.0225294243984 -0.0525069717881 -0.0890062760945
Q -0.0483151514798 -0.0309903425175 0.000437307476866 0.0281404974641 -0.0930189756622 0.00425299048772 -0.00886454093107 -0.0626883806129 -0.0331603641198 -0.00134001844501 0.00190198496329 0.0413965183445 0.0481497903134 0.177175171536 0.00715630304762 0.0357241930907 0.027467611659 -0.032780800211 -0.0118972341632 -0.0487465602402
R -0.0136431408498 -0.0482562439545 -0.0591268568576 -0.0697402405064 -0.0626297946351 -0.0232280105465 0.109052662874 -0.110160438997 0.21516548555 -0.074934598492 -0.0257018542091 -0.0338327997913 -0.0377142358446 0.00715630304762 0.389022190137 0.0204288367942 -0.0408668326839 -0.0934989556047 -0.0557627605155 0.00827128508526
S 0.038455041596 -0.0217965163697 -0.0104660502173 -0.0172364513778 -0.0868415233743 0.0304312733191 0.00938179429131 -0.0618078497671 0.00476731287861 -0.0584956357369 -0.0295349216339 0.0170820288313 -0.00687173098621 0.0357241930907 0.0204288367942 0.161573840097 0.0839261885951 -0.00816241136786 -0.0444334801409 -0.0561239385213
T 0.0520376087986 -0.0227322740574 0.00656101264316 -0.0054830504799 -0.0777292391855 0.00861853592388 -0.0234122309305 -0.0339233811197 -0.0331318604828 -0.031311528799 -0.0454820084051 -0.00295174153884 0.0199181111388 0.027467611659 -0.0408668326839 0.0839261885951 0.142525860495 0.0493244941272 -0.0264928932645 -0.0468623824337
V 0.081101427454 -0.0154276574266 -0.0193560886308 -0.00806269508269 -0.015794520965 -0.0127217072682 -0.0870188771708 0.091300054417 -0.0400367780545 0.0449678806271 -0.0120310888206 -0.0436807942705 0.0225294243984 -0.032780800211 -0.0934989556047 -0.00816241136786 0.0493244941272 0.172778293246 -0.0289445753682 -0.0444846240282
W -0.125564718844 0.0412325888637 0.00415097887978 -0.00791955104235 0.0625957490761 -0.0246539147339 0.0123622841944 0.00138488610169 -0.0598522551401 -0.00754567267671 0.0210041287765 0.0296409813073 -0.0525069717881 -0.0118972341632 -0.0557627605155 -0.0444334801409 -0.0264928932645 -0.0289445753682 0.194048086876 0.0791543436022
Y -0.0747500389698 0.00600631739163 -0.0191575919464 -0.0231187170833 0.0858189617896 -0.0205094859119 0.0365879336865 -0.0230596885329 -0.00464804635586 -0.0137219703366 0.0126203200265 -0.00205806264345 -0.0890062760945 -0.0487465602402 0.00827128508526 -0.0561239385213 -0.0468623824337 -0.0444846240282 0.0791543436022 0.237788221516
"""), sep=r"\s+", index_col=0))
assert numpy.allclose(PMBEC_MATRIX, PMBEC_MATRIX.T)


def _lower_triangular_matrix(values, amino_acids):
    """
    Parse an AAindex-style lower-triangular symmetric matrix.
    """
    values = [float(value) for value in values.split()]
    size = len(amino_acids)
    expected = size * (size + 1) // 2
    if len(values) != expected:
        raise ValueError("Expected %d values, got %d" % (expected, len(values)))
    matrix = numpy.zeros((size, size), dtype=numpy.float32)
    offset = 0
    for i in range(size):
        for j in range(i + 1):
            matrix[i, j] = values[offset]
            matrix[j, i] = values[offset]
            offset += 1
    return pandas.DataFrame(matrix, index=amino_acids, columns=amino_acids)


CONTACT_MATRIX = _matrix_with_unknown(_lower_triangular_matrix("""
-0.06711
0.06154 -0.08474
0.09263 -0.15773 -0.17967
0.08686 -0.30946 -0.15017 -0.17210
0.04917 0.10341 0.02032 0.08274 -0.17110
0.07189 -0.13486 -0.11794 -0.12196 0.02698 -0.09693
0.10110 -0.28982 -0.09284 -0.14442 0.12837 -0.10631 -0.17005
0.02605 -0.05025 -0.02792 -0.01798 -0.06868 -0.02051 0.04571 -0.08089
0.06456 -0.02235 -0.07118 -0.12234 -0.05122 -0.07235 -0.08473 -0.00788 -0.36006
-0.10028 0.19111 0.16954 0.17048 0.00854 0.13563 0.14349 0.02672 0.16715 -0.12746
-0.08988 0.18513 0.12818 0.14879 -0.00733 0.12921 0.14917 0.01380 0.12272 -0.08644 -0.06027
0.06167 -0.03348 -0.17589 -0.28130 0.10403 -0.16154 -0.32208 0.00219 0.05396 0.16985 0.10764 -0.19507
-0.03220 0.07068 0.11448 0.06630 -0.10678 0.01964 0.07144 -0.02177 0.03768 -0.00634 0.01406 0.15021 -0.00959
0.02583 0.15212 0.09736 0.14568 -0.05523 0.06410 0.11831 0.07023 0.03701 -0.08883 -0.11808 0.14062 -0.11397 -0.10140
0.04566 -0.16615 -0.09111 -0.05572 0.01966 -0.08858 -0.10634 -0.04698 -0.05851 0.12930 0.07467 -0.12627 0.09356 0.12370 -0.13648
0.05356 -0.09178 -0.07509 0.00009 -0.00836 -0.05424 0.00702 0.00867 -0.06447 0.04214 0.01289 -0.03768 -0.02411 -0.01294 -0.02111 0.01170
0.04740 -0.04650 -0.02649 0.00758 -0.04399 -0.01817 -0.02133 -0.00796 -0.03512 0.00418 0.01068 -0.01572 -0.00949 0.03108 0.02446 -0.01390 -0.01606
0.08732 0.07735 0.00987 0.02263 -0.02572 -0.06867 -0.04921 0.10405 -0.11836 0.03430 -0.06007 0.14702 -0.06644 -0.11108 0.01048 0.01889 0.01650 -0.07522
0.03904 0.02232 -0.01661 -0.05851 -0.01389 -0.04713 -0.08186 0.04000 -0.03306 0.02135 0.00677 0.06447 -0.00096 -0.02173 -0.05590 0.00139 0.00633 -0.09071 -0.03925
-0.07279 0.17288 0.10802 0.14779 0.00772 0.11813 0.11895 0.02340 0.07075 -0.13605 -0.06701 0.12223 -0.01508 -0.04855 0.11169 0.03754 0.01024 0.01594 0.03319 -0.10756
""", list("ARNDCQEGHILKMFPSTWYV")))
assert numpy.allclose(CONTACT_MATRIX, CONTACT_MATRIX.T)

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


def _standardize_over_common_amino_acids(df):
    """
    Z-score columns over the 20 common amino acids, leaving X as zeros.
    """
    result = df.copy()
    common = list(COMMON_AMINO_ACIDS)
    means = result.loc[common].mean(axis=0)
    stds = result.loc[common].std(axis=0, ddof=0).replace(0.0, 1.0)
    result.loc[common] = (result.loc[common] - means) / stds
    result.loc["X"] = 0.0
    return result.astype("float32")


# Explicit physicochemical descriptors. Continuous scales are z-scored over
# the 20 common amino acids so no single unit system dominates the dense layer:
# Kyte-Doolittle hydropathy plus Grantham composition, polarity, and volume.
# The remaining columns are direct side-chain chemistry indicators. X is the
# neutral/no-signal row, matching the convention used by BLOSUM62.
_PHYSCHEM_CONTINUOUS_RAW = pandas.DataFrame.from_dict({
    "A": [1.8, 0.00, 8.1, 31.0],
    "R": [-4.5, 0.65, 10.5, 124.0],
    "N": [-3.5, 1.33, 11.6, 56.0],
    "D": [-3.5, 1.38, 13.0, 54.0],
    "C": [2.5, 2.75, 5.5, 55.0],
    "Q": [-3.5, 0.89, 10.5, 85.0],
    "E": [-3.5, 0.92, 12.3, 83.0],
    "G": [-0.4, 0.74, 9.0, 3.0],
    "H": [-3.2, 0.58, 10.4, 96.0],
    "I": [4.5, 0.00, 5.2, 111.0],
    "L": [3.8, 0.00, 4.9, 111.0],
    "K": [-3.9, 0.33, 11.3, 119.0],
    "M": [1.9, 0.00, 5.7, 105.0],
    "F": [2.8, 0.00, 5.2, 132.0],
    "P": [-1.6, 0.39, 8.0, 32.5],
    "S": [-0.8, 1.42, 9.2, 32.0],
    "T": [-0.7, 0.71, 8.6, 61.0],
    "W": [-0.9, 0.13, 5.4, 170.0],
    "Y": [-1.3, 0.20, 6.2, 136.0],
    "V": [4.2, 0.00, 5.9, 84.0],
    "X": [0.0, 0.0, 0.0, 0.0],
}, orient="index", columns=[
    "z_kd_hydropathy",
    "z_grantham_composition",
    "z_grantham_polarity",
    "z_grantham_volume",
]).loc[AMINO_ACIDS]

_PHYSCHEM_BINARY = pandas.DataFrame(0.0, index=AMINO_ACIDS, columns=[
    "side_chain_charge",
    "aromatic",
    "sulfur",
    "hydroxyl",
    "amide",
    "acidic",
    "basic",
    "aliphatic",
    "glycine",
    "proline",
])
_PHYSCHEM_BINARY.loc[["D", "E"], "side_chain_charge"] = -1.0
_PHYSCHEM_BINARY.loc[["K", "R"], "side_chain_charge"] = 1.0
_PHYSCHEM_BINARY.loc["H", "side_chain_charge"] = 0.1
_PHYSCHEM_BINARY.loc[["F", "W", "Y", "H"], "aromatic"] = 1.0
_PHYSCHEM_BINARY.loc[["C", "M"], "sulfur"] = 1.0
_PHYSCHEM_BINARY.loc[["S", "T", "Y"], "hydroxyl"] = 1.0
_PHYSCHEM_BINARY.loc[["N", "Q"], "amide"] = 1.0
_PHYSCHEM_BINARY.loc[["D", "E"], "acidic"] = 1.0
_PHYSCHEM_BINARY.loc[["K", "R", "H"], "basic"] = 1.0
_PHYSCHEM_BINARY.loc[["A", "I", "L", "M", "V"], "aliphatic"] = 1.0
_PHYSCHEM_BINARY.loc["G", "glycine"] = 1.0
_PHYSCHEM_BINARY.loc["P", "proline"] = 1.0

PHYSICOCHEMICAL_PROPERTIES = pandas.concat([
    _standardize_over_common_amino_acids(_PHYSCHEM_CONTINUOUS_RAW),
    _PHYSCHEM_BINARY,
], axis=1).astype("float32")
assert numpy.isfinite(PHYSICOCHEMICAL_PROPERTIES.values).all()
assert (PHYSICOCHEMICAL_PROPERTIES.loc["X"].values == 0.0).all()

ENCODING_DATA_FRAMES = {
    "BLOSUM62": BLOSUM62_MATRIX,
    "one-hot": pandas.DataFrame([
        [1 if i == j else 0 for i in range(len(AMINO_ACIDS))]
        for j in range(len(AMINO_ACIDS))
    ], index=AMINO_ACIDS, columns=AMINO_ACIDS),
    "PMBEC": PMBEC_MATRIX,
    "pmbec": PMBEC_MATRIX,
    "contact": CONTACT_MATRIX,
    "simons1999-contact": CONTACT_MATRIX,
    "SIMK990103": CONTACT_MATRIX,
    "physchem": PHYSICOCHEMICAL_PROPERTIES,
    "atchley": ATCHLEY_FACTORS,
}
_COMPOSITE_ENCODING_DATA_FRAMES = {}
_NORMALIZED_ENCODING_DATA_FRAMES = {}


def _minmax_scaled_encoding(df):
    """
    Scale non-X encoding values to [-1, 1], preserving X as all zeros.
    """
    result = df.astype("float32").copy()
    common_rows = list(COMMON_AMINO_ACIDS)
    value_columns = [column for column in result.columns if column != "X"]
    values = result.loc[common_rows, value_columns].values
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value == min_value:
        result.loc[common_rows, value_columns] = 0.0
    else:
        result.loc[common_rows, value_columns] = (
            2.0 * (values - min_value) / (max_value - min_value) - 1.0
        )
    if "X" in result.index:
        result.loc["X"] = 0.0
    if "X" in result.columns:
        result["X"] = 0.0
    return result.astype("float32")


def get_vector_encoding_df(name):
    """
    Return the amino-acid vector encoding table for ``name``.

    ``name`` may be a base encoding such as ``"BLOSUM62"``, ``"PMBEC"``,
    ``"contact"``, ``"physchem"``, or ``"atchley"``. The aliases
    ``"simons1999-contact"`` and the AAindex id ``"SIMK990103"`` both refer
    to ``"contact"``. A component may use the ``":minmax"`` suffix to scale
    its non-X values to [-1, 1] while keeping X neutral. ``+``-joined
    composites such as ``"BLOSUM62+physchem"`` concatenate the component
    columns in order and keep the same amino-acid row order.
    """
    if name in ENCODING_DATA_FRAMES:
        return ENCODING_DATA_FRAMES[name]
    if not isinstance(name, str):
        raise KeyError(name)
    if "+" in name:
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

    if ":" in name:
        base_name, modifier = name.rsplit(":", 1)
        if modifier == "minmax" and base_name:
            if name not in _NORMALIZED_ENCODING_DATA_FRAMES:
                _NORMALIZED_ENCODING_DATA_FRAMES[name] = _minmax_scaled_encoding(
                    get_vector_encoding_df(base_name)
                )
            return _NORMALIZED_ENCODING_DATA_FRAMES[name]

    raise KeyError(name)


def available_vector_encodings():
    """
    Return list of registered amino acid vector encodings.

    ``get_vector_encoding_df`` also accepts ``+``-joined composites of
    these names and the ``:minmax`` suffix, for example
    ``"BLOSUM62+PMBEC:minmax"``.

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
