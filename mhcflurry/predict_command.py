# Copyright (c) 2016. Mount Sinai School of Medicine
#
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
'''
Run MHCflurry predictor on specified peptide/allele pairs.

Examples:

Write a CSV file containing the contents of INPUT.csv plus an
additional column giving MHCflurry binding affinity predictions:

    mhcflurry-predict INPUT.csv --out RESULT.csv

The input CSV file is expected to contain columns 'allele' and 'peptide'.
The predictions are written to a column called 'mhcflurry_prediction'.
These default column names may be changed with the --allele-column,
--peptide-column, and --prediction-column options.

If --out is not specified, results are writtent to standard out.

You can also run on alleles and peptides specified on the commandline, in
which case predictions are written for all combinations of alleles and
peptides:

    mhcflurry-predict --alleles HLA-A0201 H-2Kb --peptides SIINFEKL DENDREKLLL
'''
from __future__ import (
    print_function,
    division,
    absolute_import,
)
import sys
import argparse
import logging
import pandas
import itertools

from .downloads import get_path
from . import class1_allele_specific

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument(
    "input",
    metavar="FILE.csv",
    nargs="?",
    help="Input CSV")

parser.add_argument(
    "--out",
    metavar="FILE.csv",
    help="Output CSV")

parser.add_argument(
    "--alleles",
    metavar="ALLELE",
    nargs="+",
    help="Alleles to predict (exclusive with --input)")

parser.add_argument(
    "--peptides",
    metavar="PEPTIDE",
    nargs="+",
    help="Peptides to predict (exclusive with --input)")

parser.add_argument(
    "--allele-column",
    metavar="NAME",
    default="allele",
    help="Input column name for alleles. Default: '%(default)s'")

parser.add_argument(
    "--peptide-column",
    metavar="NAME",
    default="peptide",
    help="Input column name for peptides. Default: '%(default)s'")

parser.add_argument(
    "--prediction-column",
    metavar="NAME",
    default="mhcflurry_prediction",
    help="Output column name for predictions. Default: '%(default)s'")

parser.add_argument(
    "--models-class1-allele-specific-single",
    metavar="DIR",
    default=get_path("models_class1_allele_specific_single"),
    help="Directory containing class1 allele specific single models. "
    "Default: '%(default)s'")


def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)

    if args.input:
        if args.alleles or args.peptides:
            parser.error(
                "If an input file is specified, do not specify --alleles "
                "or --peptides")
        df = pandas.read_csv(args.input)
        print("Read input CSV with %d rows, columns are: %s" % (
            len(df), ", ".join(df.columns)))
        for col in [args.allele_column, args.peptide_column]:
            if col not in df.columns:
                raise ValueError(
                    "No such column '%s' in CSV. Columns are: %s" % (
                        col, ", ".join(["'%s'" % c for c in df.columns])))
    else:
        if not args.alleles or not args.peptides:
            parser.error(
                "Specify either an input CSV file or both the "
                "--alleles and --peptides arguments")

        pairs = list(itertools.product(args.alleles, args.peptides))
        df = pandas.DataFrame({
            "allele": [p[0] for p in pairs],
            "peptide": [p[1] for p in pairs],
        })
        print("Predicting for %d alleles and %d peptides = %d predictions" % (
            len(args.alleles), len(args.peptides), len(df)))

    class1_allele_specific_loader = (
        class1_allele_specific.load.Class1AlleleSpecificPredictorLoader(
            args.models_class1_allele_specific_single))

    predictions = {}  # allele -> peptide -> value
    for (allele, sub_df) in df.groupby(args.allele_column):
        logging.info("Running %d predictions for allele %s" % (
            len(sub_df), allele))
        model = class1_allele_specific_loader.from_allele_name(allele)
        peptides = sub_df[args.peptide_column].values
        predictions[allele] = dict(
            (peptide, prediction)
            for (peptide, prediction)
            in zip(peptides, model.predict(peptides)))

    logging.info("Collecting result")
    df[args.prediction_column] = [
        predictions[row[args.allele_column]][row[args.peptide_column]]
        for (_, row) in df.iterrows()
    ]

    if args.out:
        df.to_csv(args.out, index=False)
        print("Wrote: %s" % args.out)
    else:
        df.to_csv(sys.stdout, index=False)
