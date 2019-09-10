"""
Filter and combine class I sequence fastas.
"""
from __future__ import print_function

import sys
import argparse

import mhcnames

import Bio.SeqIO  # pylint: disable=import-error


def normalize(s, disallowed=["MIC", "HFE"]):
    if any(item in s for item in disallowed):
        return None
    try:
        return mhcnames.normalize_allele_name(s)
    except:
        while s:
            s = ":".join(s.split(":")[:-1])
            try:
                return mhcnames.normalize_allele_name(s)
            except:
                pass
        return None


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "fastas",
    nargs="+",
    help="Unaligned fastas")

parser.add_argument(
    "--out",
    required=True,
    help="Fasta output")

class_ii_names = {
    "DRA",
    "DRB",
    "DPA",
    "DPB",
    "DQA",
    "DQB",
    "DMA",
    "DMB",
    "DOA",
    "DOB",
}


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
            name = record.description.split()[1]
            normalized = normalize(name)
            if not normalized and "," in record.description:
                # Try parsing uniprot-style sequence description
                if "MOUSE MHC class I L-q alpha-chain" in record.description:
                    # Special case.
                    name = "H2-Lq"
                else: 
                    name = (
                        record.description.split()[1].replace("-", "") +
                        "-" +
                        record.description.split(",")[-1].split()[0].replace("-",""))
                normalized = normalize(name)
            if not normalized:
                print("Couldn't parse: ", name)
                continue
            if normalized in seen:
                continue
            if any(n in name for n in class_ii_names):
                print("Dropping", name)
                continue
            seen.add(normalized)
            record.description = normalized + " " + record.description
            records.append(record)

    with open(args.out, "w") as fd:
        Bio.SeqIO.write(records, fd, "fasta")

    print("Wrote %d / %d sequences: %s" % (len(records), total, args.out))


if __name__ == '__main__':
    run()
