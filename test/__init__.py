'''
Utility functions for tests.
'''

import os

import numpy

from mhcflurry import amino_acid

def make_random_peptides(num, length=9):
    return [
        ''.join(peptide_sequence)
        for peptide_sequence in
        numpy.random.choice(
            amino_acid.common_amino_acid_letters, size=(num, length))
    ]

def data_path(name):
    '''
    Return the absolute path to a file in the test/data directory.
    The name specified should be relative to test/data.
    '''
    return os.path.join(os.path.dirname(__file__), "data", name)
