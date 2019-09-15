"""
Filter and combine various peptide/MHC datasets to derive a composite training set,
optionally including eluted peptides identified by mass-spec.
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
    "pmid",
    metavar="PMID",
    help="PMID of dataset to curate")
parser.add_argument(
    "files",
    nargs="+",
    metavar="FILE",
    help="File paths of data to curate")
parser.add_argument(
    "--out",
    metavar="OUT.csv",
    help="Out file path")
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Leave user in pdb if PMID is unsupported")

HANDLERS = {}


def load(filenames, **kwargs):
    result = {}
    for filename in filenames:
        if filename.endswith(".csv"):
            result[filename] = pandas.read_csv(filename, **kwargs)
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            result[filename] = pandas.read_excel(filename, **kwargs)
        else:
            result[filename] = filename

    return result


def debug(*filenames):
    loaded = load(filenames)
    import ipdb
    ipdb.set_trace()


def pmid_27600516(filename):
    df = pandas.read_csv(filename)

    sample_to_peptides = {}
    current_sample = None
    for peptide in df.peptide:
        if peptide.startswith("#"):
            current_sample = peptide[1:]
            sample_to_peptides[current_sample] = []
        else:
            assert current_sample is not None
            sample_to_peptides[current_sample].append(peptide.strip().upper())

    rows = []
    for (sample, peptides) in sample_to_peptides.items():
        for peptide in sorted(set(peptides)):
            rows.append([sample, peptide])

    result = pandas.DataFrame(rows, columns=["sample_id", "peptide"])
    return result


HANDLERS["27600516"] = pmid_27600516


def run():
    args = parser.parse_args(sys.argv[1:])

    if args.pmid in HANDLERS:
        df = HANDLERS[args.pmid](*args.files)
    elif args.debug:
        debug(*args.files)
    else:
        raise NotImplementedError(args.pmid)

    df.to_csv(args.out, index=False)
    print("Wrote: %s" % args.out)

if __name__ == '__main__':
    run()
