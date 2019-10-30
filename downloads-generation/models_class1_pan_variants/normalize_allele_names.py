"""
Normalize MHC allele names
"""

from sys import argv
import os
import pandas
import mhcnames
import argparse


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
parser.add_argument("input_csv")
parser.add_argument("--out", help="CSV output")

args = parser.parse_args(argv[1:])

df = pandas.read_csv(args.input_csv)
print("Read df with shape", df.shape)
df["allele"] = df["allele"].map(normalize)
df = df.loc[~df.allele.isnull()]
print("Done normalizing. After removing unparseable names, shape is", df.shape)
df = df.drop_duplicates("allele")
print("After dropping duplicates", df.shape)
df.to_csv(args.out, index=False)
print("Wrote", os.path.abspath(args.out))
