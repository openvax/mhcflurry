import numpy
numpy.random.seed(0)
import time
import cProfile
import pstats
import collections

import pandas

from mhcflurry import Class1AffinityPredictor
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.common import random_peptides

DOWNLOADED_PREDICTOR = Class1AffinityPredictor.load()

NUM = 10000

def test_speed(profile=False):
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


    start("first")
    DOWNLOADED_PREDICTOR.predict(["SIINFEKL"], allele="HLA-A*02:01")
    end("first")

    peptides = random_peptides(NUM)
    start("pred_%d" % NUM)
    DOWNLOADED_PREDICTOR.predict(peptides, allele="HLA-A*02:01")
    end("pred_%d" % NUM)

    NUM2 = 10000
    peptides = EncodableSequences.create(random_peptides(NUM2, length=13))
    start("encode_blosum_%d" % NUM2)
    peptides.variable_length_to_fixed_length_vector_encoding("BLOSUM62")
    end("encode_blosum_%d" % NUM2)

    start("pred_already_encoded_%d" % NUM2)
    DOWNLOADED_PREDICTOR.predict(peptides, allele="HLA-A*02:01")
    end("pred_already_encoded_%d" % NUM2)

    NUM_REPEATS = 100
    start("pred_already_encoded_%d_%d_times" % (NUM2, NUM_REPEATS))
    for _ in range(NUM_REPEATS):
        DOWNLOADED_PREDICTOR.predict(peptides, allele="HLA-A*02:01")
    end("pred_already_encoded_%d_%d_times" % (NUM2, NUM_REPEATS))

    print("SPEED BENCHMARK")
    print("Results:\n%s" % str(pandas.Series(timings)))

    return dict(
        (key, pstats.Stats(value)) for (key, value) in profilers.items())


if __name__ == '__main__':
    # If run directly from python, do profiling and leave the user in a shell
    # to explore results.

    result = test_speed(profile=True)
    result["pred_%d" % NUM].sort_stats("cumtime").reverse_order().print_stats()

    # Leave in ipython
    locals().update(result)
    import ipdb  # pylint: disable=import-error
    ipdb.set_trace()
