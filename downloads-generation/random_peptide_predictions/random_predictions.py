"""
Generate predictions for random peptides.
"""
from __future__ import print_function

import sys
import argparse
import time
import math

import pandas

import mhcflurry
from mhcflurry.common import random_peptides


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument("--models", required=True)
parser.add_argument("--num-peptides", type=int)
parser.add_argument("--out", required=True)
parser.add_argument("--chunksize", type=int, default=10000)


def run():
    args = parser.parse_args(sys.argv[1:])
    print(args)

    predictor = mhcflurry.Class1AffinityPredictor.load(args.models)

    alleles = pandas.Series(predictor.supported_alleles)

    # Clear the file
    pandas.DataFrame(columns=alleles).to_csv(args.out, index=True)

    (min_length, max_length) = predictor.supported_peptide_lengths

    peptides_per_length = int(
        math.ceil(args.chunksize / (max_length - min_length)))

    peptides_written = 0
    i = 0
    while peptides_written < args.num_peptides:
        print("Chunk %d / %d" % (
            i + 1, math.ceil(args.num_peptides / args.chunksize)))
        start = time.time()
        peptides = []
        for l in range(8, 16):
            peptides.extend(random_peptides(peptides_per_length, length=l))

        peptides = pandas.Series(peptides).sample(
            n=min(args.chunksize, args.num_peptides - peptides_written)).values
        encodable_peptides = mhcflurry.encodable_sequences.EncodableSequences.create(
            peptides)
        df = pandas.DataFrame(index=peptides)
        for allele in alleles:
            df[allele] = predictor.predict(encodable_peptides, allele=allele)
        df.to_csv(
            args.out, index=True, mode='a', header=False, float_format='%.1f')
        print("Wrote: %s  [%0.2f sec]" % (args.out, time.time() - start))
        i += 1
        peptides_written += len(peptides)

    print("Done.")

if __name__ == '__main__':
    run()
