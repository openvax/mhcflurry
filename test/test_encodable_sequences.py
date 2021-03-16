"""
Profile prediction speed

"""
import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True

import numpy
numpy.random.seed(0)
import time
import cProfile
import pstats
import collections
import argparse
import sys

import pandas

from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.common import random_peptides


DEFAULT_NUM = 1000000


# Just a speed test, no asserts
def test_encoding_speed(profile=False, num=DEFAULT_NUM):
    starts = collections.OrderedDict()
    timings = collections.OrderedDict()
    profilers = collections.OrderedDict()

    def start(name):
        starts[name] = time.time()
        if profile:
            profilers[name] = cProfile.Profile()
            profilers[name].enable()

    def end(name):
        timings[name] = time.time() - starts[name]
        if profile:
            profilers[name].disable()

    peptides = list(random_peptides(num)) + ["SYYNFE{PTYR}KL"]
    start("encode_blosum_%d" % num)
    encodable_peptides = EncodableSequences.create(peptides)
    encodable_peptides.variable_length_to_fixed_length_vector_encoding("BLOSUM62")
    end("encode_blosum_%d" % num)

    print("SPEED BENCHMARK")
    print("Results:\n%s" % str(pandas.Series(timings)))

    return dict(
        (key, pstats.Stats(value)) for (key, value) in profilers.items())


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--num-predictions",
    type=int,
    default=DEFAULT_NUM,
    help="Number of predictions to run")

if __name__ == '__main__':
    # If run directly from python, do profiling and leave the user in a shell
    # to explore results.

    args = parser.parse_args(sys.argv[1:])

    result = test_encoding_speed(
        profile=True, num=args.num_predictions)
    for key, value in result.items():
        print("***", key, "***")
        value.sort_stats("cumtime").reverse_order().print_stats()


    # Leave in ipython
    locals().update(result)
    import ipdb  # pylint: disable=import-error
    ipdb.set_trace()
