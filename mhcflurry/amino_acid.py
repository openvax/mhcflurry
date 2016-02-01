# Copyright (c) 2015. Mount Sinai School of Medicine
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

import numpy as np


class Alphabet(object):
    """
    Used to track the order of amino acids used for peptide encodings
    """

    def __init__(self, **kwargs):
        self.letters_to_names = {}
        for (k, v) in kwargs.items():
            self.add(k, v)

    def add(self, letter, name):
        assert letter not in self.letters_to_names
        assert len(letter) == 1
        self.letters_to_names[letter] = name

    def letters(self):
        return list(sorted(self.letters_to_names.keys()))

    def names(self):
        return [self.letters_to_names[k] for k in self.letters()]

    def index_dict(self):
        return {c: i for (i, c) in enumerate(self.letters())}

    def copy(self):
        return Alphabet(**self.letters_to_names)

    def __getitem__(self, k):
        return self.letters_to_names[k]

    def __setitem__(self, k, v):
        self.add(k, v)

    def index_encoding_list(self, peptides):
        index_dict = self.index_dict()
        return [
            [index_dict[amino_acid] for amino_acid in peptide]
            for peptide in peptides
        ]

    def index_encoding(self, peptides, peptide_length):
        """
        Encode a set of equal length peptides as a matrix of their
        amino acid indices.
        """
        X = np.zeros((len(peptides), peptide_length), dtype=int)
        index_dict = self.index_dict()
        for i, peptide in enumerate(peptides):
            for j, amino_acid in enumerate(peptide):
                X[i, j] = index_dict[amino_acid]
        return X

    def hotshot_encoding(
            self,
            peptides,
            peptide_length):
        """
        Encode a set of equal length peptides as a binary matrix,
        where each letter is transformed into a length 20 vector with a single
        element that is 1 (and the others are 0).
        """
        shape = (len(peptides), peptide_length, 20)
        index_dict = self.index_dict()
        X = np.zeros(shape, dtype=bool)
        for i, peptide in enumerate(peptides):
            for j, amino_acid in enumerate(peptide):
                k = index_dict[amino_acid]
                X[i, j, k] = 1
        return X


common_amino_acids = Alphabet(**{
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
})

amino_acids_with_unknown = common_amino_acids.copy()
amino_acids_with_unknown.add("X", "Unknown")
