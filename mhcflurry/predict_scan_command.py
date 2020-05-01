'''
Scan protein sequences using the MHCflurry presentation predictor.

By default, sub-sequences (peptides) with affinity percentile ranks less than
2.0 are returned. You can also specify --results-all to return predictions for
all peptides, or --results-best to return the top peptide for each sequence.

Examples:

Scan a set of sequences in a FASTA file for binders to any alleles in a MHC I
genotype:

$ mhcflurry-predict-scan \
    test/data/example.fasta \
    --alleles HLA-A*02:01,HLA-A*03:01,HLA-B*57:01,HLA-B*45:01,HLA-C*02:01,HLA-C*07:02

Instead of a FASTA, you can also pass a CSV that has "sequence_id" and "sequence"
columns.

You can also specify multiple MHC I genotypes to scan as space-separated
arguments to the --alleles option:

$ mhcflurry-predict-scan \
    test/data/example.fasta \
    --alleles \
        HLA-A*02:01,HLA-A*03:01,HLA-B*57:01,HLA-B*45:01,HLA-C*02:02,HLA-C*07:02 \
        HLA-A*01:01,HLA-A*02:06,HLA-B*44:02,HLA-B*07:02,HLA-C*01:02,HLA-C*03:01

If `--out` is not specified, results are written to standard out.

You can also specify sequences on the commandline:

mhcflurry-predict-scan \
    --sequences MGYINVFAFPFTIYSLLLCRMNSRNYIAQVDVVNFNLT \
    --alleles HLA-A*02:01,HLA-A*03:01,HLA-B*57:01,HLA-B*45:01,HLA-C*02:02,HLA-C*07:02

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

from .downloads import get_default_class1_presentation_models_dir
from .class1_presentation_predictor import Class1PresentationPredictor
from .fasta import read_fasta_to_dataframe
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
    help="Print the list of supported alleles and exits"
)
helper_args.add_argument(
    "--list-supported-peptide-lengths",
    action="store_true",
    default=False,
    help="Print the list of supported peptide lengths and exits"
)
helper_args.add_argument(
    "--version",
    action="version",
    version="mhcflurry %s" % __version__,
)

input_args = parser.add_argument_group(title="Input options")
input_args.add_argument(
    "input",
    metavar="INPUT",
    nargs="?",
    help="Input CSV or FASTA")
input_args.add_argument(
    "--input-format",
    choices=("guess", "csv", "fasta"),
    default="guess",
    help="Format of input file. By default, it is guessed from the file "
         "extension.")
input_args.add_argument(
    "--alleles",
    metavar="ALLELE",
    nargs="+",
    help="Alleles to predict")
input_args.add_argument(
    "--sequences",
    metavar="SEQ",
    nargs="+",
    help="Sequences to predict (exclusive with passing an input file)")
input_args.add_argument(
    "--sequence-id-column",
    metavar="NAME",
    default="sequence_id",
    help="Input CSV column name for sequence IDs. Default: '%(default)s'")
input_args.add_argument(
    "--sequence-column",
    metavar="NAME",
    default="sequence",
    help="Input CSV column name for sequences. Default: '%(default)s'")
input_args.add_argument(
    "--no-throw",
    action="store_true",
    default=False,
    help="Return NaNs for unsupported alleles or peptides instead of raising")

results_args = parser.add_argument_group(title="Result options")
results_args.add_argument(
    "--peptide-lengths",
    default="8-11",
    metavar="L",
    help="Peptide lengths to consider. Pass as START-END (e.g. 8-11) or a "
    "comma-separated list (8,9,10,11). When using START-END, the range is "
    "INCLUSIVE on both ends. Default: %(default)s.")
comparison_quantities = [
    "presentation_score",
    "processing_score",
    "affinity",
    "affinity_percentile",
]
results_args.add_argument(
    "--results-all",
    action="store_true",
    default=False,
    help="Return results for all peptides regardless of affinity, etc.")
results_args.add_argument(
    "--results-best",
    choices=comparison_quantities,
    help="Take the top result for each sequence according to the specified "
    "predicted quantity")
results_args.add_argument(
    "--results-filtered",
    choices=comparison_quantities,
    help="Filter results by the specified quantity.")
results_args.add_argument(
    "--threshold-presentation-score",
    type=float,
    default=0.7,
    help="Threshold if filtering by presentation score. Default: %(default)s")
results_args.add_argument(
    "--threshold-processing-score",
    type=float,
    default=0.5,
    help="Threshold if filtering by processing score. Default: %(default)s")
results_args.add_argument(
    "--threshold-affinity",
    type=float,
    default=500,
    help="Threshold if filtering by affinity. Default: %(default)s")
results_args.add_argument(
    "--threshold-affinity-percentile",
    type=float,
    default=2.0,
    help="Threshold if filtering by affinity percentile. Default: %(default)s")


output_args = parser.add_argument_group(title="Output options")
output_args.add_argument(
    "--out",
    metavar="OUTPUT.csv",
    help="Output CSV")
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

model_args = parser.add_argument_group(title="Model options")
model_args.add_argument(
    "--models",
    metavar="DIR",
    default=None,
    help="Directory containing presentation models."
    "Default: %s" % get_default_class1_presentation_models_dir(
        test_exists=False))
model_args.add_argument(
    "--no-flanking",
    action="store_true",
    default=False,
    help="Do not use flanking sequence information in predictions")


def parse_peptide_lengths(value):
    try:
        if "-" in value:
            (start, end) = value.split("-", 2)
            start = int(start.strip())
            end = int(end.strip())
            peptide_lengths = list(range(start, end + 1))
        else:
            peptide_lengths = [
                int(length.strip())
                for length in value.split(",")
            ]
    except ValueError:
        raise ValueError("Couldn't parse peptide lengths: ", value)
    return peptide_lengths


def run(argv=sys.argv[1:]):
    logging.getLogger('tensorflow').disabled = True

    if not argv:
        parser.print_help()
        parser.exit(1)

    args = parser.parse_args(argv)

    # It's hard to pass a tab in a shell, so we correct a common error:
    if args.output_delimiter == "\\t":
        args.output_delimiter = "\t"

    peptide_lengths = parse_peptide_lengths(args.peptide_lengths)

    result_args = {
        "all": args.results_all,
        "best": args.results_best,
        "filtered": args.results_filtered,
    }
    if all([not bool(arg) for arg in result_args.values()]):
        result_args["filtered"] = "affinity_percentile"

    if sum([bool(arg) for arg in result_args.values()]) > 1:
        parser.error(
            "Specify at most one of --results-all, --results-best, "
            "--results-filtered")

    (result,) = [key for (key, value) in result_args.items() if value]
    result_comparison_quantity = (
        None if result == "all" else result_args[result])
    result_filter_value = None if result != "filtered" else {
        "presentation_score": args.threshold_presentation_score,
        "processing_score": args.threshold_processing_score,
        "affinity": args.threshold_affinity,
        "affinity_percentile": args.threshold_affinity_percentile,
    }[result_comparison_quantity]

    models_dir = args.models
    if models_dir is None:
        # The reason we set the default here instead of in the argument parser
        # is that we want to test_exists at this point, so the user gets a
        # message instructing them to download the models if needed.
        models_dir = get_default_class1_presentation_models_dir(test_exists=True)

    predictor = Class1PresentationPredictor.load(models_dir)

    if args.list_supported_alleles:
        print("\n".join(predictor.supported_alleles))
        return

    if args.list_supported_peptide_lengths:
        min_len, max_len = predictor.supported_peptide_lengths
        print("\n".join([str(l) for l in range(min_len, max_len+1)]))
        return

    if args.input:
        if args.sequences:
            parser.error(
                "If an input file is specified, do not specify --sequences")

        input_format = args.input_format
        if input_format == "guess":
            extension = args.input.lower().split(".")[-1]
            if extension in ["gz", "bzip2"]:
                extension = args.input.lower().split(".")[-2]

            if extension == "csv":
                input_format = "csv"
            elif extension in ["fasta", "fa"]:
                input_format = "fasta"
            else:
                parser.error(
                    "Couldn't guess input format from file extension: %s\n"
                    "Pass the --input-format argument to specify if it is a "
                    "CSV or fasta file" % args.input)
            print("Guessed input file format:", input_format)

        if input_format == "csv":
            df = pandas.read_csv(args.input)
            print("Read input CSV with %d rows, columns are: %s" % (
                len(df), ", ".join(df.columns)))
            for col in [args.sequence_column,]:
                if col not in df.columns:
                    raise ValueError(
                        "No such column '%s' in CSV. Columns are: %s" % (
                            col, ", ".join(["'%s'" % c for c in df.columns])))

        elif input_format == "fasta":
            df = read_fasta_to_dataframe(args.input)
            print("Read input fasta with %d sequences" % len(df))
            print(df)
        else:
            raise ValueError("Unsupported input format", input_format)
    else:
        if not args.sequences:
            parser.error(
                "Specify either an input file or the --sequences argument")

        df = pandas.DataFrame({
            args.sequence_column: args.sequences,
        })

    if args.sequence_id_column not in df:
        df[args.sequence_id_column] = "sequence_" + df.index.astype(str)

    df = df.set_index(args.sequence_id_column)

    if args.alleles:
        genotypes = pandas.Series(args.alleles).str.split(r"[,\s]+")
        genotypes.index = genotypes.index.map(lambda i: "genotype_%02d" % i)
        alleles = genotypes.to_dict()
    else:
        print("No alleles specified. Will perform processing prediction only.")
        alleles = {}

    result_df = predictor.predict_sequences(
        sequences=df[args.sequence_column].to_dict(),
        alleles=alleles,
        result=result,
        comparison_quantity=result_comparison_quantity,
        filter_value=result_filter_value,
        peptide_lengths=peptide_lengths,
        use_flanks=not args.no_flanking,
        include_affinity_percentile=not args.no_affinity_percentile,
        throw=not args.no_throw)

    if args.out:
        result_df.to_csv(args.out, index=False, sep=args.output_delimiter)
        print("Wrote: %s" % args.out)
    else:
        result_df.to_csv(sys.stdout, index=False, sep=args.output_delimiter)
