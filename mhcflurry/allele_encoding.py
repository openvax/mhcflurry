from six import callable

import pandas

from . import amino_acid
from .allele_encoding_transforms import TRANSFORMS


class AlleleEncoding(object):
    def __init__(
            self,
            alleles=None,
            allele_to_sequence=None,
            transforms=None,
            borrow_from=None):
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

        if transforms is None:
            transforms = dict(
                (name, klass()) for (name, klass) in TRANSFORMS.items())
        self.transforms = transforms

        if self.borrow_from is None:
            assert allele_to_sequence is not None
            all_alleles = (
                sorted(allele_to_sequence))
                #if alleles is None
                #else list(sorted(alleles.unique())))
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
            self.transforms = borrow_from.transforms

        if alleles is not None:
            assert all(
                allele in self.allele_to_index for allele in alleles),\
                "Missing alleles: " + " ".join([
                    a for a in alleles if a not in self.allele_to_index])
            self.indices = alleles.map(self.allele_to_index)
            assert not self.indices.isnull().any()
        else:
            self.indices = None

        self.encoding_cache = {}

    def allele_representations(self, encoding_name):
        if self.borrow_from is not None:
            return self.borrow_from.allele_representations(encoding_name)

        cache_key = (
            "allele_representations",
            encoding_name)
        if cache_key not in self.encoding_cache:
            if ":" in encoding_name:
                # Transform
                pieces = encoding_name.split(":", 3)
                if pieces[0] != "transform":
                    raise RuntimeError(
                        "Expected 'transform' but saw: %s" % pieces[0])
                if len(pieces) == 1:
                    raise RuntimeError("Expected: 'transform:<name>[:argument]")
                transform_name = pieces[1]
                argument = None if len(pieces) == 2 else pieces[2]
                try:
                    transform = self.transforms[transform_name]
                except KeyError:
                    raise KeyError(
                        "Unsupported transform: %s. Supported transforms: %s" % (
                            transform_name,
                            " ".join(self.transforms) if self.transforms else "(none)"))
                vector_encoded = (
                    transform.transform(self) if argument is None
                    else transform.transform(self, argument))
            else:
                # No transform.
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
        Encode alleles.
        Parameters
        ----------
        encoding_name : string
            How to represent amino acids.
            One of "BLOSUM62", "one-hot", etc. Full list of supported vector
            encodings is given by available_vector_encodings() in amino_acid.

            Also supported are names like pca:BLOSUM62, which would run the
            "pca" transform on BLOSUM62-encoded sequences.
        Returns
        -------
        numpy.array with shape (num sequences, sequence length, m) where m is
        vector_encoding_length(vector_encoding_name)
        """
        cache_key = (
            "fixed_length_vector_encoding",
            encoding_name)
        if cache_key not in self.encoding_cache:
            vector_encoded = self.allele_representations(encoding_name)
            result = vector_encoded[self.indices]
            self.encoding_cache[cache_key] = result
        return self.encoding_cache[cache_key]

