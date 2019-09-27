"""
"""
import sys
import argparse
import os
import collections
from six.moves import StringIO

import pandas

import mhcnames


def normalize_allele_name(s):
    try:
        return mhcnames.normalize_allele_name(s)
    except Exception:
        return "UNKNOWN"


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "input_path",
    help="Item to curate: PMID and list of files")
parser.add_argument(
    "--out",
    metavar="OUT.csv",
    help="Out file path")


# Build index
PREBUILT_INDEX = "datasets/uniprot-proteome_UP000005640.fasta.gz.fm"
USE_PREBUILT_INDEX = os.path.exists(PREBUILT_INDEX)
print("Using prebuilt index", USE_PREBUILT_INDEX)

fm = shellinford.FMIndex()
if USE_PREBUILT_INDEX:
    fm.read(PREBUILT_INDEX)

fm_keys = []
protein_database = "datasets/uniprot-proteome_UP000005640.fasta.gz"
start = time.time()
proteome_df = []
with gzip.open(protein_database, "rt") as fd:
    records = SeqIO.parse(fd, format='fasta')
    for (i, record) in enumerate(records):
        if i % 10000 == 0:
            print(i, time.time() - start)
        fm_keys.append(record.name)
        proteome_df.append((record.name, record.description, str(record.seq)))
        if not USE_PREBUILT_INDEX:
            fm.push_back("$" + str(record.seq) + "$")  # include sentinels

if not USE_PREBUILT_INDEX:
    print("Building")
    start = time.time()
    fm.build()
    print("Done building", time.time() - start)
    fm.write(PREBUILT_INDEX)

proteome_df = pandas.DataFrame(proteome_df, columns=["name", "description", "seq"]).set_index("name")
proteome_df

SEARCH_CACHE = {}
def search(peptide, fm=fm):
    if peptide in SEARCH_CACHE:
        return SEARCH_CACHE[peptide]
    hits = fm.search(peptide)
    result = proteome_df.iloc[
        [hit.doc_id for hit in hits]
    ]
    assert result.seq.str.contains(peptide).all(), (peptide, result)
    names = result.index.tolist()
    SEARCH_CACHE[peptide] = names
    return names
print(search("SIINFEKL"))
print(search("AAAAAKVPA"))
print(search("AAAAALQAK"))
print(search("DEGPLDVSM"))




def run():
    args = parser.parse_args(sys.argv[1:])

    df = pandas.read_csv(args.input_path)
    print("Read input", df.shape)

    import ipdb ; ipdb.set_trace()

    df.to_csv(args.out, index=False)
    print("Wrote: %s" % os.path.abspath(args.out))


if __name__ == '__main__':
    run()
