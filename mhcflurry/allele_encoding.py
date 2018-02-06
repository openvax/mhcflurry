import numpy
import pandas

from .encodable_sequences import EncodableSequences
from . import amino_acid

class AlleleEncoding(object):
    def __init__(
            self,
            alleles,
            allele_to_fixed_length_sequence=None):
        """
        A place to cache encodings for a (potentially large) sequence of alleles.

        Parameters
        ----------
        alleles : list of string
            Allele names

        allele_to_fixed_length_sequence : dict of str -> str
            Allele name to fixed lengths sequence ("pseudosequence")
        """

        alleles = pandas.Series(alleles)

        all_alleles = list(sorted(alleles.unique()))

        self.allele_to_index = dict(
            (allele, i)
            for (i, allele) in enumerate(all_alleles))

        self.indices = alleles.map(self.allele_to_index)

        self.fixed_length_sequences = pandas.Series(
            [allele_to_fixed_length_sequence[a] for a in all_alleles],
            index=all_alleles)

        self.encoding_cache = {}

    def fixed_length_vector_encoded_sequences(self, vector_encoding_name):
        """
        Encode alleles.

        Parameters
        ----------
        vector_encoding_name : string
            How to represent amino acids.
            One of "BLOSUM62", "one-hot", etc. Full list of supported vector
            encodings is given by available_vector_encodings() in amino_acid.

        Returns
        -------
        numpy.array with shape (num sequences, sequence length, m) where m is
        vector_encoding_length(vector_encoding_name)
        """
        cache_key = (
            "fixed_length_vector_encoding",
            vector_encoding_name)
        if cache_key not in self.encoding_cache:
            index_encoded_matrix = amino_acid.index_encoding(
                self.fixed_length_sequences.values,
                amino_acid.AMINO_ACID_INDEX)
            vector_encoded = amino_acid.fixed_vectors_encoding(
                index_encoded_matrix,
                amino_acid.ENCODING_DATA_FRAMES[vector_encoding_name])
            result = vector_encoded[self.indices]
            self.encoding_cache[cache_key] = result
        return self.encoding_cache[cache_key]


