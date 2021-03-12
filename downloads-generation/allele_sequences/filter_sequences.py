"""
Filter and combine class I sequence fastas.
"""
from __future__ import print_function

import sys
import argparse


import Bio.SeqIO  # pylint: disable=import-error

from mhcflurry.common import normalize_allele_name

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

    total = 0
    order = []
    name_to_record = {}
    for fasta in args.fastas:
        reader = Bio.SeqIO.parse(fasta, "fasta")
        for record in reader:
            total += 1
            if len(record.seq) < 50:
                print("-- Skipping '%s', sequence too short" % (
                    record.description,))
                continue

            parts = record.description.split()
            candidate_strings = [
                record.description,
                parts[1],
                " ".join(parts[1:])
            ]
            name = None
            for candidate_string in candidate_strings:
                name = normalize_allele_name(
                    candidate_string, raise_on_error=False)
                if name is not None:
                    break
            if name is None:
                print("Skipping '%s'" % (record.description,))
                continue

            print("Parsed '%s' as %s" % (record.description, name))
            record.description = name + " " + record.description

            if name in name_to_record:
                old_record = name_to_record[name]
                old_sequence = old_record.seq
                if len(old_sequence) < len(record.seq):
                    print("-- Replacing old record (%d aa) with new (%d aa)" % (
                        len(old_record.seq),
                        len(record.seq)))
                    name_to_record[name] = record
                else:
                    print("-- Skipping, already seen")
            else:
                order.append(name)
                name_to_record[name] = record

    records = [name_to_record[name] for name in order]

    with open(args.out, "w") as fd:
        Bio.SeqIO.write(records, fd, "fasta")

    print("Wrote %d / %d sequences: %s" % (len(records), total, args.out))


if __name__ == '__main__':
    run()
