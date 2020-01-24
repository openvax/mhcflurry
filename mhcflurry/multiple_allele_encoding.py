import numpy

from copy import copy

from .allele_encoding import AlleleEncoding


class MultipleAlleleEncoding(object):
    def __init__(
            self,
            experiment_names=[],
            experiment_to_allele_list={},
            max_alleles_per_experiment=6,
            allele_to_sequence=None,
            borrow_from=None):

        padded_experiment_to_allele_list = {}
        for (name, alleles) in experiment_to_allele_list.items():
            assert len(alleles) > 0
            assert len(alleles) <= max_alleles_per_experiment
            alleles_with_mask = alleles + [None] * (
                    max_alleles_per_experiment - len(alleles))
            padded_experiment_to_allele_list[name] = alleles_with_mask

        flattened_allele_list = []
        for name in experiment_names:
            flattened_allele_list.extend(padded_experiment_to_allele_list[name])

        self.allele_encoding = AlleleEncoding(
            alleles=flattened_allele_list,
            allele_to_sequence=allele_to_sequence,
            borrow_from=borrow_from
        )
        self.max_alleles_per_experiment = max_alleles_per_experiment
        self.experiment_names = numpy.array(experiment_names)

    def append_alleles(self, alleles):
        extended_alleles = list(self.allele_encoding.alleles)
        for allele in alleles:
            extended_alleles.append(allele)
            extended_alleles.extend(
                [None] * (self.max_alleles_per_experiment - 1))

        assert len(extended_alleles) % self.max_alleles_per_experiment == 0, (
            len(extended_alleles))

        self.allele_encoding = AlleleEncoding(
            alleles=extended_alleles,
            borrow_from=self.allele_encoding)

        self.experiment_names = numpy.concatenate([
            self.experiment_names,
            numpy.tile(None, len(alleles))
        ])

    @property
    def indices(self):
        return self.allele_encoding.indices.values.reshape(
            (-1, self.max_alleles_per_experiment))

    @property
    def alleles(self):
        return numpy.reshape(
            self.allele_encoding.alleles.values,
            (-1, self.max_alleles_per_experiment))

    def compact(self):
        result = copy(self)
        result.allele_encoding = self.allele_encoding.compact()
        return result

    def allele_representations(self, encoding_name):
        return self.allele_encoding.allele_representations(encoding_name)

    @property
    def allele_to_sequence(self):
        return self.allele_encoding.allele_to_sequence

    def fixed_length_vector_encoded_sequences(self, encoding_name):
        raise NotImplementedError()

    def shuffle_in_place(self, shuffle_permutation=None):
        alleles_matrix = self.alleles
        if shuffle_permutation is None:
            shuffle_permutation = numpy.random.permutation(len(alleles_matrix))
        self.allele_encoding = AlleleEncoding(
            alleles=alleles_matrix[shuffle_permutation].flatten(),
            borrow_from=self.allele_encoding
        )
        self.experiment_names = self.experiment_names[shuffle_permutation]