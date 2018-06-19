import pandas

from . import amino_acid


class AlleleEncoding(object):
    def __init__(self, alleles=None, allele_to_sequence=None, borrow_from=None):
        """
        A place to cache encodings for a (potentially large) sequence of alleles.

        Parameters
        ----------
        alleles : list of string
            Allele names

        allele_to_sequence : dict of str -> str
            Allele name to amino acid sequence
        """

        if alleles is not None:
            alleles = pandas.Series(alleles)
        self.borrow_from = borrow_from
        self.allele_to_sequence = allele_to_sequence

        if self.borrow_from is None:
            assert allele_to_sequence is not None
            all_alleles = (
                sorted(allele_to_sequence)
                if alleles is None
                else list(sorted(alleles.unique())))
            self.allele_to_index = dict(
                (allele, i)
                for (i, allele) in enumerate(all_alleles))
            unpadded = pandas.Series(
                [allele_to_sequence[a] for a in all_alleles],
                index=all_alleles)
            self.sequences = unpadded.str.pad(
                unpadded.str.len().max(), fillchar="X")
        else:
            assert allele_to_sequence is None
            self.allele_to_index = borrow_from.allele_to_index
            self.sequences = borrow_from.sequences
            self.allele_to_sequence = borrow_from.allele_to_sequence

        if alleles is not None:
            assert all(
                allele in self.allele_to_index for allele in alleles)
            self.indices = alleles.map(self.allele_to_index)
            assert not self.indices.isnull().any()
        else:
            self.indices = None

        self.encoding_cache = {}

    def allele_representations(self, vector_encoding_name):
        if self.borrow_from is not None:
            return self.borrow_from.allele_representations(vector_encoding_name)

        cache_key = (
            "allele_representations",
            vector_encoding_name)
        if cache_key not in self.encoding_cache:
            index_encoded_matrix = amino_acid.index_encoding(
                self.sequences.values,
                amino_acid.AMINO_ACID_INDEX)
            vector_encoded = amino_acid.fixed_vectors_encoding(
                index_encoded_matrix,
                amino_acid.ENCODING_DATA_FRAMES[vector_encoding_name])
            self.encoding_cache[cache_key] = vector_encoded
        return self.encoding_cache[cache_key]

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
            vector_encoded = self.allele_representations(vector_encoding_name)
            result = vector_encoded[self.indices]
            self.encoding_cache[cache_key] = result
        return self.encoding_cache[cache_key]

