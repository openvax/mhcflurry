import numpy
import pandas

from .encodable_sequences import EncodableSequences
from . import amino_acid

class AlleleEncoding(object):
    def __init__(
            self,
            alleles,
            allele_to_fixed_length_sequence):
        """
        A place to cache encodings for a (potentially large) sequence of alleles.

        Parameters
        ----------
        alleles : list of string
            Allele names

        allele_to_fixed_length_sequence : dict of str -> str
            Allele name to fixed lengths sequence ("pseudosequence"), or a
            pandas dataframe with allele names as the index and arbitrary values
            to use for the encoding of those alleles
        """

        self.alleles = pandas.Series(alleles)

        if isinstance(allele_to_fixed_length_sequence, dict):
            self.allele_to_fixed_length_sequence = pandas.DataFrame(
                index=allele_to_fixed_length_sequence)
            self.allele_to_fixed_length_sequence["value"] = (
                self.allele_to_fixed_length_sequence.index.map(
                    allele_to_fixed_length_sequence.get))
        else:
            assert isinstance(allele_to_fixed_length_sequence, pandas.DataFrame)
            self.allele_to_fixed_length_sequence = allele_to_fixed_length_sequence

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

            If a DataFrame was provided as `allele_to_fixed_length_sequence`
            in the constructor, then those values will be used and this argument
            will be ignored.

        Returns
        -------
        list of numpy arrays. Pass it to numpy.array to get an array with shape
        (num sequences, sequence length, m) where m is
        vector_encoding_length(vector_encoding_name)

        The reason to return a list instead of an array is that the list can
        use much less memory in the common case where many of the rows are
        the same.
        """
        cache_key = (
            "fixed_length_vector_encoding",
            vector_encoding_name)
        if cache_key not in self.encoding_cache:
            all_alleles = list(sorted(self.alleles.unique()))
            allele_to_index = dict(
                (allele, i)
                for (i, allele) in enumerate(all_alleles))
            indices = self.alleles.map(allele_to_index)

            allele_to_fixed_length_sequence = self.allele_to_fixed_length_sequence.loc[
                all_alleles
            ].copy()

            if list(allele_to_fixed_length_sequence) == ["value"]:
                # Pseudosequence
                index_encoded_matrix = amino_acid.index_encoding(
                    allele_to_fixed_length_sequence["value"].values,
                    amino_acid.AMINO_ACID_INDEX)
                vector_encoded = amino_acid.fixed_vectors_encoding(
                    index_encoded_matrix,
                    amino_acid.ENCODING_DATA_FRAMES[vector_encoding_name])
            else:
                # Raw values
                vector_encoded = allele_to_fixed_length_sequence.values
            result = [vector_encoded[i] for i in indices]
            self.encoding_cache[cache_key] = result
        return self.encoding_cache[cache_key]


