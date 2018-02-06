'''
Run MHCflurry predictor on specified peptide/allele pairs.

Examples:

Write a CSV file containing the contents of INPUT.csv plus an
additional column giving MHCflurry binding affinity predictions:

    $ mhcflurry-predict INPUT.csv --out RESULT.csv

The input CSV file is expected to contain columns ``allele`` and ``peptide``.
The predictions are written to a column called ``mhcflurry_prediction``.
These default column names may be changed with the `--allele-column`,
`--peptide-column`, and `--prediction-column` options.

If `--out` is not specified, results are written to standard out.

You can also run on alleles and peptides specified on the commandline, in
which case predictions are written for all combinations of alleles and
peptides:

    $ mhcflurry-predict --alleles HLA-A0201 H-2Kb --peptides SIINFEKL DENDREKLLL
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

from .downloads import get_default_class1_models_dir
from .class1_affinity_predictor import Class1AffinityPredictor
from .version import __version__


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False)


helper_args = parser.add_argument_group(title="Help")
helper_args.add_argument(
    "-h", "--help",
    action="help",
    help="Show this help message and exit"
)
helper_args.add_argument(
    "--list-supported-alleles",
    action="store_true",
    default=False,
    help="Prints the list of supported alleles and exits"
)
helper_args.add_argument(
    "--list-supported-peptide-lengths",
    action="store_true",
    default=False,
    help="Prints the list of supported peptide lengths and exits"
)
helper_args.add_argument(
    "--version",
    action="version",
    version="mhcflurry %s" % __version__,
)

input_args = parser.add_argument_group(title="Required input arguments")
input_args.add_argument(
    "input",
    metavar="INPUT.csv",
    nargs="?",
    help="Input CSV")
input_args.add_argument(
    "--alleles",
    metavar="ALLELE",
    nargs="+",
    help="Alleles to predict (exclusive with --input)")
input_args.add_argument(
    "--peptides",
    metavar="PEPTIDE",
    nargs="+",
    help="Peptides to predict (exclusive with --input)")


input_mod_args = parser.add_argument_group(title="Optional input modifiers")
input_mod_args.add_argument(
    "--allele-column",
    metavar="NAME",
    default="allele",
    help="Input column name for alleles. Default: '%(default)s'")
input_mod_args.add_argument(
    "--peptide-column",
    metavar="NAME",
    default="peptide",
    help="Input column name for peptides. Default: '%(default)s'")
input_mod_args.add_argument(
    "--no-throw",
    action="store_true",
    default=False,
    help="Return NaNs for unsupported alleles or peptides instead of raising")


output_args = parser.add_argument_group(title="Optional output modifiers")
output_args.add_argument(
    "--out",
    metavar="OUTPUT.csv",
    help="Output CSV")
output_args.add_argument(
    "--prediction-column-prefix",
    metavar="NAME",
    default="mhcflurry_",
    help="Prefix for output column names. Default: '%(default)s'")
output_args.add_argument(
    "--output-delimiter",
    metavar="CHAR",
    default=",",
    help="Delimiter character for results. Default: '%(default)s'")


model_args = parser.add_argument_group(title="Optional model settings")
model_args.add_argument(
    "--models",
    metavar="DIR",
    default=None,
    help="Directory containing models. "
    "Default: %s" % get_default_class1_models_dir(test_exists=False))
model_args.add_argument(
    "--include-individual-model-predictions",
    action="store_true",
    default=False,
    help="Include predictions from each model in the ensemble"
)


def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)

    # It's hard to pass a tab in a shell, so we correct a common error:
    if args.output_delimiter == "\\t":
        args.output_delimiter = "\t"

    models_dir = args.models
    if models_dir is None:
        # The reason we set the default here instead of in the argument parser is that
        # we want to test_exists at this point, so the user gets a message instructing
        # them to download the models if needed.
        models_dir = get_default_class1_models_dir(test_exists=True)
    predictor = Class1AffinityPredictor.load(models_dir)

    # The following two are informative commands that can come 
    # if a wrapper would like to incorporate input validation 
    # to not delibaretly make mhcflurry fail
    if args.list_supported_alleles:
        print("\n".join(predictor.supported_alleles))
        return

    if args.list_supported_peptide_lengths:
        min_len, max_len = predictor.supported_peptide_lengths
        print("\n".join([str(l) for l in range(min_len, max_len+1)]))
        return
    # End of early terminating routines

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

    predictions = predictor.predict_to_dataframe(
        peptides=df[args.peptide_column].values,
        alleles=df[args.allele_column].values,
        include_individual_model_predictions=args.include_individual_model_predictions,
        throw=not args.no_throw)

    for col in predictions.columns:
        if col not in ("allele", "peptide"):
            df[args.prediction_column_prefix + col] = predictions[col]

    if args.out:
        df.to_csv(args.out, index=False, sep=args.output_delimiter)
        print("Wrote: %s" % args.out)
    else:
        df.to_csv(sys.stdout, index=False, sep=args.output_delimiter)
