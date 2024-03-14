"""
Profile prediction speed

"""

import numpy
import time
import cProfile
import pstats
import collections
import argparse
import sys
import pytest

import pandas

from mhcflurry import Class1AffinityPredictor
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.common import random_peptides
from mhcflurry.downloads import get_path

from mhcflurry.testing_utils import cleanup, startup

DEFAULT_NUM_PREDICTIONS = 10000


def load_predictors():
    return {
        'allele_specific': Class1AffinityPredictor.load(get_path("models_class1", "models")),
        'pan_allele': Class1AffinityPredictor.load(get_path("models_class1_pan", "models.combined")),
    }


# Define a fixture to initialize and clean up predictors
@pytest.fixture(scope="module")
def predictors():
    startup()
    predictors_dict = load_predictors()
    yield predictors_dict
    cleanup()

@pytest.fixture(autouse=True)
def init():
    from . import initialize
    initialize()

def test_speed_allele_specific(predictors, profile=False, num=DEFAULT_NUM_PREDICTIONS):
    starts = collections.OrderedDict()
    timings = collections.OrderedDict()
    profilers = collections.OrderedDict()

    predictor = predictors['allele_specific']

    def start(name):
        starts[name] = time.time()
        if profile:
            profilers[name] = cProfile.Profile()
            profilers[name].enable()

    def end(name):
        timings[name] = time.time() - starts[name]
        if profile:
            profilers[name].disable()

    start("first")
    predictor.predict(["SIINFEKL"], allele="HLA-A*02:01")
    end("first")

    peptides = random_peptides(num)
    start("pred_%d" % num)
    predictor.predict(peptides, allele="HLA-A*02:01")
    end("pred_%d" % num)

    NUM2 = 10000
    peptides = EncodableSequences.create(random_peptides(NUM2, length=13))
    start("encode_blosum_%d" % NUM2)
    peptides.variable_length_to_fixed_length_vector_encoding("BLOSUM62")
    end("encode_blosum_%d" % NUM2)

    start("pred_already_encoded_%d" % NUM2)
    predictor.predict(peptides, allele="HLA-A*02:01")
    end("pred_already_encoded_%d" % NUM2)

    NUM_REPEATS = 100
    start("pred_already_encoded_%d_%d_times" % (NUM2, NUM_REPEATS))
    for _ in range(NUM_REPEATS):
        predictor.predict(peptides, allele="HLA-A*02:01")
    end("pred_already_encoded_%d_%d_times" % (NUM2, NUM_REPEATS))

    print("SPEED BENCHMARK")
    print("Results:\n%s" % str(pandas.Series(timings)))

    if __name__ == '__main__':
        # Only return values if run directly from Python, since pytest complains
        # about return values from tests.
        return dict(
            (key, pstats.Stats(value)) for (key, value) in profilers.items())


def test_speed_pan_allele(predictors, profile=False, num=DEFAULT_NUM_PREDICTIONS):
    starts = collections.OrderedDict()
    timings = collections.OrderedDict()
    profilers = collections.OrderedDict()

    predictor = predictors['pan_allele']

    def start(name):
        starts[name] = time.time()
        if profile:
            profilers[name] = cProfile.Profile()
            profilers[name].enable()

    def end(name):
        timings[name] = time.time() - starts[name]
        if profile:
            profilers[name].disable()

    start("first")
    predictor.predict(["SIINFEKL"], allele="HLA-A*02:01")
    end("first")

    peptides = random_peptides(num)
    start("pred_%d" % num)
    predictor.predict(peptides, allele="HLA-A*02:01")
    end("pred_%d" % num)

    print("SPEED BENCHMARK")
    print("Results:\n%s" % str(pandas.Series(timings)))

    if __name__ == '__main__':
        # Only return values if run directly from Python, since pytest complains
        # about return values from tests.
        return dict(
            (key, pstats.Stats(value)) for (key, value) in profilers.items())


parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument(
    "--predictor",
    nargs="+",
    choices=["allele-specific", "pan-allele"],
    default=["allele-specific", "pan-allele"],
    help="Which predictors to run")

parser.add_argument(
    "--num-predictions",
    type=int,
    default=DEFAULT_NUM_PREDICTIONS,
    help="Number of predictions to run")

if __name__ == '__main__':
    # If run directly from python, do profiling and leave the user in a shell
    # to explore results.

    args = parser.parse_args(sys.argv[1:])
    predictors_dict = load_predictors()

    if "allele-specific" in args.predictor:
        print("Running allele-specific test")
        result = test_speed_allele_specific(
            predictors=predictors_dict,
            profile=True,
            num=args.num_predictions)
        result[
            "pred_%d" % args.num_predictions
        ].sort_stats("cumtime").reverse_order().print_stats()

    if "pan-allele" in args.predictor:
        print("Running pan-allele test")
        result = test_speed_pan_allele(
            predictors=predictors_dict,
            profile=True,
            num=args.num_predictions)
        result[
            "pred_%d" % args.num_predictions
        ].sort_stats("cumtime").reverse_order().print_stats()

    # Leave in ipython
    locals().update(result)
    import ipdb  # pylint: disable=import-error
    ipdb.set_trace()
