'''
Run MHCflurry predictor on specified peptides.

By default, the presentation predictor is used, and predictions for
MHC I binding affinity, antigen processing, and the composite presentation score
are returned. If you just want binding affinity predictions, pass
--affinity-only.

Examples:

Write a CSV file containing the contents of INPUT.csv plus additional columns
giving MHCflurry predictions:

$ mhcflurry-predict INPUT.csv --out RESULT.csv

The input CSV file is expected to contain columns "allele", "peptide", and,
optionally, "n_flank", and "c_flank".

If `--out` is not specified, results are written to stdout.

You can also run on alleles and peptides specified on the commandline, in
which case predictions are written for *all combinations* of alleles and
peptides:

$ mhcflurry-predict --alleles HLA-A0201 H-2Kb --peptides SIINFEKL DENDREKLLL

Instead of individual alleles (in a CSV or on the command line), you can also
give a comma separated list of alleles giving a sample genotype. In this case,
the tightest binding affinity across the alleles for the sample will be
returned. For example:

$ mhcflurry-predict --peptides SIINFEKL DENDREKLLL \
    --alleles \
        HLA-A*02:01,HLA-A*03:01,HLA-B*57:01,HLA-B*45:01,HLA-C*02:01,HLA-C*07:02 \
        HLA-A*01:01,HLA-A*02:06,HLA-B*44:02,HLA-B*07:02,HLA-C*01:01,HLA-C*03:01

will give the tightest predicted affinities across alleles for each of the two
genotypes specified for each peptide.
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
import os

import pandas

from .downloads import get_default_class1_presentation_models_dir
from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_presentation_predictor import Class1PresentationPredictor
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

input_args = parser.add_argument_group(title="Input (required)")
input_args.add_argument(
    "input",
    metavar="INPUT.csv",
    nargs="?",
    help="Input CSV")
input_args.add_argument(
    "--alleles",
    metavar="ALLELE",
    nargs="+",
    help="Alleles to predict (exclusive with passing an input CSV)")
input_args.add_argument(
    "--peptides",
    metavar="PEPTIDE",
    nargs="+",
    help="Peptides to predict (exclusive with passing an input CSV)")

input_mod_args = parser.add_argument_group(title="Input options")
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
    "--n-flank-column",
    metavar="NAME",
    default="n_flank",
    help="Column giving N-terminal flanking sequence. Default: '%(default)s'")
input_mod_args.add_argument(
    "--c-flank-column",
    metavar="NAME",
    default="c_flank",
    help="Column giving C-terminal flanking sequence. Default: '%(default)s'")
input_mod_args.add_argument(
    "--no-throw",
    action="store_true",
    default=False,
    help="Return NaNs for unsupported alleles or peptides instead of raising")

output_args = parser.add_argument_group(title="Output options")
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
output_args.add_argument(
    "--no-affinity-percentile",
    default=False,
    action="store_true",
    help="Do not include affinity percentile rank")
output_args.add_argument(
    "--always-include-best-allele",
    default=False,
    action="store_true",
    help="Always include the best_allele column even when it is identical "
    "to the allele column (i.e. all queries are monoallelic).")

model_args = parser.add_argument_group(title="Model options")
model_args.add_argument(
    "--models",
    metavar="DIR",
    default=None,
    help="Directory containing models. Either a binding affinity predictor or "
    "a presentation predictor can be used. "
    "Default: %s" % get_default_class1_presentation_models_dir(
        test_exists=False))
model_args.add_argument(
    "--affinity-only",
    action="store_true",
    default=False,
    help="Affinity prediction only (no antigen processing or presentation)")
model_args.add_argument(
    "--no-flanking",
    action="store_true",
    default=False,
    help="Do not use flanking sequence information even when available")


def run(argv=sys.argv[1:]):
    logging.getLogger('tensorflow').disabled = True

    if not argv:
        parser.print_help()
        parser.exit(1)

    args = parser.parse_args(argv)

    # It's hard to pass a tab in a shell, so we correct a common error:
    if args.output_delimiter == "\\t":
        args.output_delimiter = "\t"

    models_dir = args.models
    if models_dir is None:
        # The reason we set the default here instead of in the argument parser
        # is that we want to test_exists at this point, so the user gets a
        # message instructing them to download the models if needed.
        models_dir = get_default_class1_presentation_models_dir(test_exists=True)

    if os.path.exists(os.path.join(models_dir, "weights.csv")):
        # Using a presentation predictor.
        predictor = Class1PresentationPredictor.load(models_dir)
    else:
        # Using just an affinity predictor.
        affinity_predictor = Class1AffinityPredictor.load(models_dir)
        predictor = Class1PresentationPredictor(
            affinity_predictor=affinity_predictor)
        if not args.affinity_only:
            logging.warning(
                "Specified models are an affinity predictor, which implies "
                "--affinity-only. Specify this argument to silence this warning.")
            args.affinity_only = True

    if args.list_supported_alleles:
        print("\n".join(predictor.supported_alleles))
        return

    if args.list_supported_peptide_lengths:
        min_len, max_len = predictor.supported_peptide_lengths
        print("\n".join([str(l) for l in range(min_len, max_len+1)]))
        return

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
        logging.info(
            "Predicting for %d alleles and %d peptides = %d predictions" % (
                len(args.alleles), len(args.peptides), len(df)))

    allele_string_to_alleles = (
        df.drop_duplicates(args.allele_column).set_index(
            args.allele_column, drop=False)[
                args.allele_column
        ].str.split(r"[,\s]+")).to_dict()

    if args.affinity_only:
        predictions = predictor.predict_affinity(
            peptides=df[args.peptide_column].values,
            alleles=allele_string_to_alleles,
            sample_names=df[args.allele_column],
            throw=not args.no_throw,
            include_affinity_percentile=not args.no_affinity_percentile)
    else:
        n_flanks = None
        c_flanks = None
        if not args.no_flanking:
            if args.n_flank_column in df.columns and args.c_flank_column in df.columns:
                n_flanks = df[args.n_flank_column]
                c_flanks = df[args.c_flank_column]
            else:
                logging.warning(
                    "No flanking information provided. Specify --no-flanking "
                    "to silence this warning")

        predictions = predictor.predict(
            peptides=df[args.peptide_column].values,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            alleles=allele_string_to_alleles,
            sample_names=df[args.allele_column],
            throw=not args.no_throw,
            include_affinity_percentile=not args.no_affinity_percentile)

    # If each query is just for a single allele, the "best_allele" column
    # is redundant so we remove it.
    if not args.always_include_best_allele:
        if all(len(a) == 1 for a in allele_string_to_alleles.values()):
            del predictions["best_allele"]

    for col in predictions.columns:
        if col not in ("allele", "peptide", "sample_name", "peptide_num"):
            df[args.prediction_column_prefix + col] = predictions[col]

    if args.out:
        df.to_csv(args.out, index=False, sep=args.output_delimiter)
        print("Wrote: %s" % args.out)
    else:
        df.to_csv(sys.stdout, index=False, sep=args.output_delimiter)
