import pandas

from . import amino_acid


class AlleleEncoding(object):
    def __init__(self, alleles=None, allele_to_sequence=None, borrow_from=None):
        """
        A place to cache encodings for a sequence of alleles.

        We frequently work with alleles by integer indices, for example as
        inputs to neural networks. This class is used to map allele names to
        integer indices in a consistent way by keeping track of the universe
        of alleles under use, i.e. a distinction is made between the universe
        of supported alleles (what's in `allele_to_sequence`) and the actual
        set of alleles used for some task (what's in `alleles`).

        Parameters
        ----------
        alleles : list of string
            Allele names. If any allele is None instead of string, it will be
            mapped to the special index value -1.

        allele_to_sequence : dict of str -> str
            Allele name to amino acid sequence

        borrow_from : AlleleEncoding, optional
            If specified, do not specify allele_to_sequence. The sequences from
            the provided instance are used. This guarantees that the mappings
            from allele to index and from allele to sequence are the same
            between the instances.
        """

        if alleles is not None:
            alleles = pandas.Series(alleles)
        self.borrow_from = borrow_from
        self.allele_to_sequence = allele_to_sequence

        if self.borrow_from is None:
            assert allele_to_sequence is not None
            all_alleles = (
                sorted(allele_to_sequence))
            self.allele_to_index = dict(
                (allele, i)
                for (i, allele) in enumerate([None] + all_alleles))
            unpadded = pandas.Series([
                    allele_to_sequence[a] if a is not None else ""
                    for a in [None] + all_alleles
                ],
                index=[None] + all_alleles)
            self.sequences = unpadded.str.pad(
                unpadded.str.len().max(), fillchar="X")
        else:
            assert allele_to_sequence is None
            self.allele_to_index = borrow_from.allele_to_index
            self.sequences = borrow_from.sequences
            self.allele_to_sequence = borrow_from.allele_to_sequence

        if alleles is not None:
            assert all(
                allele in self.allele_to_index for allele in alleles),\
                "Missing alleles: " + " ".join(set(
                    a for a in alleles if a not in self.allele_to_index))
            self.indices = alleles.map(self.allele_to_index)
            assert not self.indices.isnull().any()
            self.alleles = alleles
        else:
            self.indices = None
            self.alleles = None

        self.encoding_cache = {}

    def compact(self):
        """
        Return a new AlleleEncoding in which the universe of supported alleles
        is only the alleles actually used.

        Returns
        -------
        AlleleEncoding
        """
        return AlleleEncoding(
            alleles=self.alleles,
            allele_to_sequence=dict(
                (allele, self.allele_to_sequence[allele])
                for allele in self.alleles.unique()
                if allele is not None))

    def allele_representations(self, encoding_name):
        """
        Encode the universe of supported allele sequences to a matrix.

        Parameters
        ----------
        encoding_name : string
            How to represent amino acids. Valid names are "BLOSUM62" or
            "one-hot". See `amino_acid.ENCODING_DATA_FRAMES`.

        Returns
        -------
        numpy.array of shape
            (num alleles in universe, sequence length, vector size)
        where vector size is usually 21 (20 amino acids + X character)
        """
        if self.borrow_from is not None:
            return self.borrow_from.allele_representations(encoding_name)

        cache_key = (
            "allele_representations",
            encoding_name)
        if cache_key not in self.encoding_cache:
            index_encoded_matrix = amino_acid.index_encoding(
                self.sequences.values,
                amino_acid.AMINO_ACID_INDEX)
            vector_encoded = amino_acid.fixed_vectors_encoding(
                index_encoded_matrix,
                amino_acid.ENCODING_DATA_FRAMES[encoding_name])
            self.encoding_cache[cache_key] = vector_encoded
        return self.encoding_cache[cache_key]

    def fixed_length_vector_encoded_sequences(self, encoding_name):
        """
        Encode allele sequences (not the universe of alleles) to a matrix.

        Parameters
        ----------
        encoding_name : string
            How to represent amino acids. Valid names are "BLOSUM62" or
            "one-hot". See `amino_acid.ENCODING_DATA_FRAMES`.

        Returns
        -------
        numpy.array with shape:
            (num alleles, sequence length, vector size)
        where vector size is usually 21 (20 amino acids + X character)
        """
        cache_key = (
            "fixed_length_vector_encoding",
            encoding_name)
        if cache_key not in self.encoding_cache:
            vector_encoded = self.allele_representations(encoding_name)
            result = vector_encoded[self.indices]
            self.encoding_cache[cache_key] = result
        return self.encoding_cache[cache_key]
