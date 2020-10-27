"""
Filter and combine class I sequence fastas.
"""
from __future__ import print_function

import sys
import argparse

from mhcgnomes import parse, Allele, AlleleWithoutGene, Gene

import Bio.SeqIO  # pylint: disable=import-error


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
            result = parse(name, raise_on_error=False)
            if not result:
                # TODO: are there other entries that require this?

                # Try parsing uniprot-style sequence description
                if "MOUSE MHC class I L-q alpha-chain" in record.description:
                    # Special case.
                    name = "H2-Lq"
                else:
                    print("Unable to parse: %s" % name)
                    continue
            else:
                if type(result) not in (Allele, AlleleWithoutGene, Gene):
                    print("Skpping %s, unexpected parsed type: %s" % (
                        name,
                        result))
                    continue
                if result.mhc_class not in {"I", "Ia", "Ib"}:
                    print(
                        "Skpping %s, wrong MHC class: %s" % (
                        name,
                        result.mhc_class))
                    continue
                if any(item in name.upper() for item in {"MIC", "HFE"}):
                    print("Skipping %s, gene too different from Class Ia" % (
                        name,))
                    continue
                if type(result) is Allele and (
                        result.annotation_pseudogene or
                        result.annotation_null or
                        result.annotation_questionable):
                    print("Skipping %s, due to annotation(s): %s" % (
                        name,
                        result.annotations))
                name = result.to_string()

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
