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
import itertools
import logging

import pandas

from .downloads import get_path
from .class1_affinity_prediction import Class1AffinityPredictor


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
    "--prediction-column-prefix",
    metavar="NAME",
    default="mhcflurry_",
    help="Prefix for output column names. Default: '%(default)s'")

parser.add_argument(
    "--models",
    metavar="DIR",
    default=None,
    help="Directory containing models. "
    "Default: %s" % get_path("models_class1", "models", test_exists=False))

parser.add_argument(
    "--include-individual-model-predictions",
    action="store_true",
    default=False,
    help="Include predictions from each model in the ensemble"
)


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
        # split user specified allele and peptide strings in case they
        # contain multiple entries separated by commas
        alleles = []
        for allele_string in args.alleles:
            alleles.extend([s.strip() for s in allele_string.split(",")])
        peptides = []
        for peptide in args.peptides:
            peptides.extend(peptide.strip() for p in peptide.split(","))
        for peptide in peptides:
            if not peptide.isalpha():
                raise ValueError(
                    "Unexpected character(s) in peptide '%s'" % peptide)
        pairs = list(itertools.product(alleles, peptides))
        df = pandas.DataFrame({
            "allele": [p[0] for p in pairs],
            "peptide": [p[1] for p in pairs],
        })
        logging.info(
            "Predicting for %d alleles and %d peptides = %d predictions" % (
            len(args.alleles), len(args.peptides), len(df)))

    models_dir = args.models
    if models_dir is None:
        # The reason we set the default here instead of in the argument parser is that
        # we want to test_exists at this point, so the user gets a message instructing
        # them to download the models if needed.
        models_dir = get_path("models_class1", "models")
    predictor = Class1AffinityPredictor.load(models_dir)

    predictions = predictor.predict_to_dataframe(
        peptides=df[args.peptide_column].values,
        alleles=df[args.allele_column].values,
        include_individual_model_predictions=args.include_individual_model_predictions)

    for col in predictions.columns:
        if col not in ("allele", "peptide"):
            df[args.prediction_column_prefix + col] = predictions[col]

    if args.out:
        df.to_csv(args.out, index=False)
        print("Wrote: %s" % args.out)
    else:
        df.to_csv(sys.stdout, index=False)
