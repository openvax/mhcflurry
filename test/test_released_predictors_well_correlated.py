"""
Test that pan-allele and allele-specific predictors are highly correlated.
"""
from __future__ import print_function
import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True

import pytest
import sys
import argparse
import pandas
import numpy

from mhcflurry import Class1AffinityPredictor
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.downloads import get_path
from mhcflurry.common import random_peptides

from mhcflurry.testing_utils import cleanup, startup


# Define a fixture to initialize and clean up predictors
@pytest.fixture(scope="module")
def predictors():
    startup()
    predictors_dict = {
        'allele-specific': Class1AffinityPredictor.load(get_path("models_class1", "models")),
        'pan-allele': Class1AffinityPredictor.load(get_path("models_class1_pan", "models.combined")),
    }
    yield predictors_dict
    cleanup()


def test_correlation(
        predictors,
        alleles=None,
        num_peptides_per_length=1000,
        lengths=[8, 9, 10],
        debug=False,
        return_result=False):
    peptides = []
    for length in lengths:
        peptides.extend(random_peptides(num_peptides_per_length, length))

    # Cache encodings
    peptides = EncodableSequences.create(list(set(peptides)))

    if alleles is None:
        alleles = set.intersection(*[
            set(predictor.supported_alleles) for predictor in predictors.values()
        ])
    alleles = sorted(set(alleles))
    df = pandas.DataFrame(index=peptides.sequences)

    results_df = []
    for allele in alleles:
        for (name, predictor) in predictors.items():
            df[name] = predictor.predict(peptides, allele=allele)
        correlation = numpy.corrcoef(
            numpy.log10(df["allele-specific"]),
            numpy.log10(df["pan-allele"]))[0, 1]
        results_df.append((allele, correlation))
        print(len(results_df), len(alleles), *results_df[-1])

        if correlation < 0.6:
            print("Warning: low correlation", allele)
            df["tightest"] = df.min(1)
            print(df.sort_values("tightest").iloc[:, :-1])
            if debug:
                import ipdb ; ipdb.set_trace()
            del df["tightest"]

    results_df = pandas.DataFrame(results_df, columns=["allele", "correlation"])
    print(results_df)

    print("Mean correlation", results_df.correlation.mean())
    assert results_df.correlation.mean() > 0.65

    if return_result:
        return results_df


parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument(
    "--alleles",
    nargs="+",
    default=None,
    help="Which alleles to test")

if __name__ == '__main__':
    # If run directly from python, leave the user in a shell to explore results.
    startup()
    args = parser.parse_args(sys.argv[1:])
    result = test_correlation(alleles=args.alleles, debug=True, return_result=True)

    # Leave in ipython
    import ipdb  # pylint: disable=import-error
    ipdb.set_trace()
