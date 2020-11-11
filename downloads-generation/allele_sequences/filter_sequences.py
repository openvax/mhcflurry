"""
Filter and combine class I sequence fastas.
"""
from __future__ import print_function

import sys
import argparse


import Bio.SeqIO  # pylint: disable=import-error

from normalize_allele_name import normalize_allele_name

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "fastas",
    nargs="+",
    help="Unaligned fastas")

parser.add_argument(
    "--out",
    required=True,
    help="Fasta output")

def run():
    args = parser.parse_args(sys.argv[1:])
    print(args)

    records = []
    total = 0
    seen = set()
    for fasta in args.fastas:
        reader = Bio.SeqIO.parse(fasta, "fasta")
        for record in reader:
            total += 1
            parts = record.description.split()
            candidate_strings = [
                record.description,
                parts[1],
                " ".join(parts[1:])
            ]
            name = None
            for candidate_string in candidate_strings:
                name = normalize_allele_name(candidate_string)
                if name is not None:
                    break
            if name is None:
                continue
            if name in seen:
                continue
            seen.add(name)
            record.description = name + " " + record.description
            records.append(record)

    with open(args.out, "w") as fd:
        Bio.SeqIO.write(records, fd, "fasta")

    print("Wrote %d / %d sequences: %s" % (len(records), total, args.out))


if __name__ == '__main__':
    run()
