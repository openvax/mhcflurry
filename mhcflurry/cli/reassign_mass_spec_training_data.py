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

"""Reassign affinity values for mass-spec training rows."""
import argparse
import os
import sys

import pandas

from mhcflurry.release_holdout import exclude_affinity_pmhcs


def make_parser(prog=None):
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("data", metavar="CSV", help="Training data.")
    parser.add_argument("--ms-only", action="store_true", default=False)
    parser.add_argument("--drop-negative-ms", action="store_true", default=False)
    parser.add_argument("--set-measurement-value", type=float)
    parser.add_argument("--out-csv")
    parser.add_argument(
        "--exclude-pmhcs",
        help="CSV of frozen evaluation allele,peptide pairs to exclude.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print input and changed-row previews.")
    return parser


def reassign_mass_spec_training_data(
        data,
        ms_only=False,
        drop_negative_ms=False,
        set_measurement_value=None,
        exclude_pmhcs=None,
        out_csv=None,
        verbose=False):
    """Return a curated training dataframe with requested MS-row edits."""
    df = pandas.read_csv(data)
    print("Read %d rows from %s" % (len(df), data), file=sys.stderr)
    if verbose:
        print(df, file=sys.stderr)

    if drop_negative_ms:
        bad = df.loc[
            (df.measurement_kind == "mass_spec") &
            (df.measurement_inequality != "<")
        ]
        print("Dropping %d negative MS rows" % len(bad), file=sys.stderr)
        df = df.loc[~df.index.isin(bad.index)].copy()

    if ms_only:
        print("Filtering to MS only", file=sys.stderr)
        df = df.loc[df.measurement_kind == "mass_spec"].copy()

    if exclude_pmhcs:
        df = exclude_affinity_pmhcs(df, exclude_pmhcs)

    if set_measurement_value is not None:
        indexer = df.measurement_kind == "mass_spec"
        df.loc[indexer, "measurement_value"] = set_measurement_value
        print(
            "Reassigned measurement_value for %d MS rows to %s" % (
                indexer.sum(), set_measurement_value),
            file=sys.stderr)
        if verbose:
            print(df.loc[indexer], file=sys.stderr)

    if out_csv:
        out_csv = os.path.abspath(out_csv)
        df.to_csv(out_csv, index=False)
        print("Wrote %s" % out_csv, file=sys.stderr)

    return df


def run_argv(argv=None, prog=None):
    """Run the command."""
    args = make_parser(prog=prog).parse_args(argv)
    reassign_mass_spec_training_data(
        args.data,
        ms_only=args.ms_only,
        drop_negative_ms=args.drop_negative_ms,
        set_measurement_value=args.set_measurement_value,
        exclude_pmhcs=args.exclude_pmhcs,
        out_csv=args.out_csv,
        verbose=args.verbose)


def main(argv=None):
    """Entry point for script compatibility."""
    if argv is None:
        argv = sys.argv[1:]
    return run_argv(argv)


if __name__ == "__main__":
    main()
