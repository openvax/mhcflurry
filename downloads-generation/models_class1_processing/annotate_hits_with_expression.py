# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Annotate hits with expression (tpm), and roll up to just the highest-expressed
gene for each peptide.
"""
import sys
import argparse
import os
import bz2
import hashlib
import json
import platform
from importlib.metadata import version

import pandas


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--hits",
    metavar="CSV",
    required=True,
    help="Multiallelic mass spec")
parser.add_argument(
    "--expression",
    metavar="CSV",
    required=True,
    help="Expression data")
parser.add_argument(
    "--out",
    metavar="CSV",
    required=True,
    help="File to write")
parser.add_argument(
    "--random-seed", type=int, default=42,
    help="Seed for maximum-expression annotation ties (default: 42).")
parser.add_argument(
    "--provenance", help="JSON provenance path (default: OUT.provenance.json).")
parser.add_argument(
    "--validate-existing", action="store_true",
    help="Validate a previously written CSV and its provenance without changing them.")

ANNOTATION_POLICY = "max-expression-canonical-seeded-ties-v1"


def sha256(path, decompress=False):
    """Hash input bytes, optionally decoding a bzip2 output table."""
    opener = bz2.open if decompress and str(path).endswith(".bz2") else open
    digest = hashlib.sha256()
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_annotations(hit_df, random_seed=42):
    """Select one maximum-TPM annotation per hit, invariant to input row order.

    Canonical row order precedes a local seeded shuffle, so neither process
    RNG state nor input ordering determines which tied source supplies flanks.
    Duplicate annotation rows have no extra weight. Output is ordered by hit ID.
    """
    if not 0 <= random_seed < 2**32:
        raise ValueError("random_seed must be in [0, 2**32)")
    if hit_df.hit_id.isna().any():
        raise ValueError("Annotation rows must have a hit_id")
    candidates = hit_df.loc[
        hit_df.tpm == hit_df.groupby("hit_id").tpm.transform("max")
    ].drop_duplicates().reset_index(drop=True)
    order = candidates.astype(str).sort_values(sorted(candidates.columns)).index
    return candidates.loc[order].sample(
        frac=1.0, random_state=random_seed,
    ).drop_duplicates("hit_id").sort_values(
        "hit_id", key=lambda values: values.astype(str),
    ).reset_index(drop=True)


def annotate_tpm(hit_df, expression_df):
    """Return per-hit TPM sums for each row's expression dataset."""
    tpm = pandas.Series(0.0, index=hit_df.index)
    for expression_dataset, sub_df in hit_df.groupby(
            "expression_dataset", sort=False):
        expression_by_gene = expression_df[expression_dataset]
        genes = sub_df.protein_ensembl.str.split().explode()
        gene_tpm = genes.map(expression_by_gene).fillna(0.0)
        tpm.loc[sub_df.index] = gene_tpm.groupby(level=0).sum()
    return tpm


def run():
    args = parser.parse_args(sys.argv[1:])
    args.out = os.path.abspath(args.out)
    provenance_path = args.provenance or args.out + ".provenance.json"
    provenance = {
        "schema_version": 1,
        "policy": ANNOTATION_POLICY,
        "random_seed": args.random_seed,
        "hits_sha256": sha256(args.hits),
        "expression_sha256": sha256(args.expression),
        "generator_sha256": sha256(__file__),
    }
    if args.validate_existing:
        try:
            with open(provenance_path) as stream:
                recorded = json.load(stream)
        except (OSError, ValueError) as error:
            raise ValueError(
                "Cannot reuse annotations without valid provenance; use a new "
                "run directory rather than relabeling historical annotations."
            ) from error
        mismatches = [key for key, value in provenance.items()
                      if recorded.get(key) != value]
        if recorded.get("output_csv_sha256") != sha256(args.out, decompress=True):
            mismatches.append("output_csv_sha256")
        if mismatches:
            raise ValueError("Annotation provenance mismatch: " + ", ".join(mismatches))
        print("Validated annotated hits:", args.out)
        return

    hit_df = pandas.read_csv(args.hits, dtype={"hit_id": str, "sample_id": str})
    hit_df = hit_df.loc[
        (~hit_df.protein_ensembl.isnull())
    ]
    print("Loaded hits from %d samples" % hit_df.sample_id.nunique())
    expression_df = pandas.read_csv(args.expression, index_col=0).fillna(0)

    # Add a column to hit_df giving expression value for that sample and that gene
    print("Annotating expression.")
    hit_df["tpm"] = annotate_tpm(hit_df, expression_df)

    # Discard hits except those that have max expression for each hit_id
    print("Selecting max-expression transcripts for each hit.")
    max_gene_hit_df = select_annotations(hit_df, args.random_seed)

    max_gene_hit_df.to_csv(args.out, index=False)
    provenance.update(
        output_csv_sha256=sha256(args.out, decompress=True),
        selected_rows=len(max_gene_hit_df),
        versions={"python": platform.python_version(), "pandas": pandas.__version__,
                  "numpy": version("numpy")},
    )
    with open(provenance_path, "w") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("Wrote", args.out)

if __name__ == '__main__':
    run()
