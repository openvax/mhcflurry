import numpy
numpy.random.seed(0)
import time
import cProfile
import pstats

import pandas

from mhcflurry import Class1AffinityPredictor
from mhcflurry.common import random_peptides

NUM = 100000

DOWNLOADED_PREDICTOR = Class1AffinityPredictor.load()


def test_speed(profile=False):
    starts = {}
    timings = {}
    profilers = {}

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
    import ipdb ; ipdb.set_trace()
