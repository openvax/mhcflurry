"""
Parse various publications' supplementary tables
"""
import sys
import argparse

import pandas

import mhcnames


def normalize_allele_name(s):
    try:
        return mhcnames.normalize_allele_name(s)
    except Exception:
        return "UNKNOWN"


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--format",
    metavar="PMID",
    required=True,
    help="pubmed ID of paper to parse")
parser.add_argument(
    "--input",
    required=True,
    help="Input data")
parser.add_argument(
    "--out-csv",
    required=True,
    help="Output file to write")

parser.add_argument(
    "--out-csv",
    required=True,
    help="Result file")


PARSERS = {}

# Di Marco et al 2017
def parse_28904123(input):
    import ipdb ; ipdb.set_trace()

PARSERS["28904123"] = parse_28904123

# Illing et al 2018
def parse_30410026(input):
    import ipdb ; ipdb.set_trace()

PARSERS["30410026"] = parse_30410026

# Mobbs et al 2017
def parse_28855257(input):
    import ipdb ; ipdb.set_trace()

PARSERS["28855257"] = parse_28855257

# Ramarathinam et al 2018
def parse_29437277(input):
    import ipdb ; ipdb.set_trace()

PARSERS["29437277"] = parse_29437277

# Pymm et al 2017
def parse_28218747(input):
    import ipdb ; ipdb.set_trace()

PARSERS["28218747"] = parse_28218747


def run():
    args = parser.parse_args(sys.argv[1:])

    if args.input.endswith(".xlsx"):
        handle = pandas.read_excel(args.input, sheet_name=None)
    else:
        raise ValueError("Unsupported input: %s" % args.input)

    parse_function = PARSERS.get(args.format)
    if not parse_function:
        raise ValueError("Unsupported format: %s" % args.format)

    result = parse_function(handle)

    result.to_csv(args.out_csv, index=False)
    print("Wrote dataframe of shape %s: %s" % (result.shape, args.out_csv))

if __name__ == '__main__':
    run()
