# PYTHON_ARGCOMPLETE_OK
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
Scan proteins for peptide presentation using a CSV, FASTA or --sequences.

With --alleles, return peptides with affinity percentile ranks at most 2.0 by
default. Use --results-all for every peptide or --threshold-* to change the
filters. Without --alleles, predict processing only. Results go to stdout
unless --out is given.

Examples:
  mhcflurry predict-scan proteins.fasta --alleles HLA-A0201 --out hits.csv
  mhcflurry predict-scan proteins.csv --alleles 'HLA-A*02:01;HLA-A*03:01'
  mhcflurry predict-scan --sequences SIINFEKLGGGNLVPMVATV --alleles HLA-A0201

CSV columns: sequence_id, sequence. Each --alleles argument is one sample;
delimit alleles within a quoted argument with commas or semicolons to give
a sample MHC allele set.

Output columns:
  sequence_name             Input sequence ID (CSV sequence_id or FASTA header)
  pos                       Zero-based peptide start position in the sequence
  peptide, n_flank, c_flank Peptide and optional flanking sequences
  affinity                  Binding affinity (IC50 nM; lower is tighter)
  affinity_percentile       Affinity percentile rank (0-100%; lower is tighter)
  processing_score          Antigen processing score (0-1; higher is stronger)
  presentation_score        Presentation score (0-1; higher is stronger)
  presentation_percentile   Presentation percentile (0-100%; lower is tighter)
"""
import collections
import itertools
import os
import re
import sys

import argcomplete

from .completion import is_completing
from .help import HelpArgumentParser
from ..cli_args import add_alleles_arg, add_out_arg, positive_int_arg
from ..downloads import get_default_class1_presentation_models_dir, get_model_release_dir
from ..parallelism import add_prediction_parallelism_args
from ..version import __version__


parser = HelpArgumentParser(
    usage="%(prog)s [OPTIONS] [INPUT]",
    description=__doc__,
    add_help=False)


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
add_alleles_arg(
    input_args,
    metavar="ALLELE",
    help="Sample MHC allele or allele-set queries. Each argument is one sample; "
    "delimit alleles within a sample with ',' or ';'.")
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
input_args.add_argument(
    "--chunk-size",
    type=positive_int_arg,
    metavar="N",
    default=None,
    help="Process input sequences in streaming chunks of at most N sequences "
    "to bound peak memory on large inputs. When set, predictions are sorted "
    "within each chunk rather than globally across the entire input.")

results_args = parser.add_argument_group(title="Result options")
results_args.add_argument(
    "--peptide-lengths",
    default="8-11",
    metavar="L",
    help="Peptide lengths to consider. Pass as START-END (e.g. 8-11) or a "
    "comma-separated list (8,9,10,11). When using START-END, the range is "
    "INCLUSIVE on both ends. Default: %(default)s.")
default_thresholds = {
    "presentation_score": 0.7,
    "processing_score": 0.5,
    "affinity": 500,
    "affinity_percentile": 2.0,
}
results_args.add_argument(
    "--results-all",
    action="store_true",
    default=False,
    help="Disable the default affinity-percentile filter. Explicit thresholds still apply.")
results_args.add_argument(
    "--threshold-presentation-score",
    type=float,
    help="Keep presentation scores >= this value. No default filter on this score.")
results_args.add_argument(
    "--threshold-processing-score",
    type=float,
    help="Keep processing scores >= this value. No default filter on this score.")
results_args.add_argument(
    "--threshold-affinity",
    type=float,
    help="Keep predicted affinities <= this value in nM. No default filter on nM affinity.")
results_args.add_argument(
    "--threshold-affinity-percentile",
    type=float,
    help=f"Keep affinity percentiles <= this value. Defaults to {default_thresholds['affinity_percentile']} "
         "when no thresholds or --results-all are supplied.")


output_args = parser.add_argument_group(title="Output options")
add_out_arg(output_args, metavar="OUTPUT.csv", help="Output CSV")
output_args.add_argument(
    "--output-delimiter",
    metavar="CHAR",
    default=",",
    help="Delimiter character for results. Default: '%(default)s'")
output_args.add_argument(
    "--float-format",
    metavar="FMT",
    default=None,
    help="Optional printf-style float format for numeric CSV columns "
    "(e.g. '%%.4f' or '%%.6g'). Default: full float precision.")
output_args.add_argument(
    "--no-affinity-percentile",
    default=False,
    action="store_true",
    help="Do not include affinity percentile rank")

model_args = parser.add_argument_group(title="Model options")
model_selection = model_args.add_mutually_exclusive_group()
model_selection.add_argument(
    "--models",
    metavar="DIR",
    default=None,
    help="Directory containing presentation models. "
    "Default: %s" % (
        # Resolving the path reads the download catalogue; a Tab press
        # does not need it.
        "the models_class1_presentation download" if is_completing()
        else get_default_class1_presentation_models_dir(test_exists=False)))
model_selection.add_argument(
    "--model-release",
    metavar="RELEASE",
    help="Use the presentation weights from a download release, overriding model "
    "path defaults. List choices: mhcflurry downloads releases models_class1_presentation.")

model_args.add_argument(
    "--no-flanking",
    action="store_true",
    default=False,
    help="Do not use flanking sequence information in predictions")

add_prediction_parallelism_args(parser)

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
    help="Print the list of supported alleles and exit"
)
helper_args.add_argument(
    "--list-supported-peptide-lengths",
    action="store_true",
    default=False,
    help="Print the list of supported peptide lengths and exit"
)
helper_args.add_argument(
    "--version",
    action="version",
    version="mhcflurry %s" % __version__,
)


_PREDICTOR_CACHE = {}


def _supported_alleles_for_command(models_dir):
    from ..affinity.persistence import read_supported_alleles

    return read_supported_alleles(models_dir)


def _supported_peptide_lengths_for_command(models_dir):
    from ..affinity.persistence import read_supported_peptide_lengths

    return read_supported_peptide_lengths(models_dir)


def _load_predictor_for_command(models_dir):
    from ..class1_presentation_predictor import Class1PresentationPredictor

    if models_dir not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[models_dir] = Class1PresentationPredictor.load(
            models_dir)
    return _PREDICTOR_CACHE[models_dir]


# One parallel chunk's predictions plus the comparison quantity the predictor
# resolved for it. ``comparison_quantity`` is identical across chunks (it
# depends only on the predictor's capabilities and the alleles, not on the
# chunk's sequences); it is carried per-chunk so the parent can apply the same
# global sort the serial path would without re-loading the predictor.
ChunkResult = collections.namedtuple(
    "ChunkResult", ["chunk_num", "predictions", "comparison_quantity"])


def _predict_sequences_chunk_worker(work_item):
    predictor = _load_predictor_for_command(work_item["models_dir"])
    result_df = predictor.predict_sequences(
        sequences=work_item["sequences"],
        alleles=work_item["alleles"],
        result="all",
        peptide_lengths=work_item["peptide_lengths"],
        use_flanks=work_item["use_flanks"],
        include_affinity_percentile=work_item[
            "include_affinity_percentile"],
        throw=work_item["throw"],
        affinity_model_kwargs=work_item["affinity_model_kwargs"],
        processing_batch_size=work_item["processing_batch_size"])
    return ChunkResult(
        chunk_num=work_item["chunk_num"],
        predictions=result_df,
        comparison_quantity=predictor.resolve_comparison_quantity(
            work_item["alleles"]))


def parse_peptide_lengths(value):
    try:
        if "-" in value:
            (start, end) = value.split("-", 2)
            start = int(start.strip())
            end = int(end.strip())
            if start <= 0 or end < start:
                raise ValueError(
                    "Invalid peptide length range: %r" % (value,)
                )
            peptide_lengths = list(range(start, end + 1))
        else:
            peptide_lengths = [
                int(length.strip())
                for length in value.split(",")
            ]
            if not peptide_lengths or any(length <= 0 for length in peptide_lengths):
                raise ValueError(
                    "Invalid peptide lengths: %r" % (value,)
                )
    except ValueError:
        raise ValueError("Couldn't parse peptide lengths: %r" % (value,)) from None
    return peptide_lengths


def _resolve_input_format(args, arg_parser):
    """Determine input file format ('csv' or 'fasta') from args."""
    input_format = args.input_format
    if input_format == "guess":
        parts = args.input.lower().split(".")
        extension = parts[-1]
        compression = None
        if extension in ("gz", "gzip", "bz2", "bzip2") and len(parts) >= 2:
            compression = extension
            extension = parts[-2]

        if extension == "csv":
            input_format = "csv"
        elif extension in ("fasta", "fa") and compression not in ("bz2", "bzip2"):
            input_format = "fasta"
        else:
            arg_parser.error(
                "Couldn't guess input format from file extension: %s\n"
                "Pass the --input-format argument to specify if it is a "
                "CSV or fasta file" % args.input)
        print(
            "Guessed input file format:", input_format,
            file=sys.stderr)
    elif input_format not in ("csv", "fasta"):
        arg_parser.error("Unsupported input format: %r" % (input_format,))
    return input_format


def _iter_sequence_chunks(args, input_format, arg_parser, pandas, FastaParser):
    """Yield ``dict[sequence_id, sequence]`` chunks of at most ``args.chunk_size``."""
    chunk_size = args.chunk_size
    if args.input:
        if input_format == "csv":
            try:
                header_df = pandas.read_csv(
                    args.input, keep_default_na=False, dtype=str, nrows=0)
            except (FileNotFoundError, OSError, ValueError) as error:
                arg_parser.error(str(error))
            if args.sequence_column not in header_df.columns:
                arg_parser.error(
                    "No such column '%s' in CSV. Columns are: %s" % (
                        args.sequence_column,
                        ", ".join(["'%s'" % c for c in header_df.columns])))
            print(
                "Streaming input CSV in chunks of %d sequences, columns are: %s" % (
                    chunk_size, ", ".join(header_df.columns)),
                file=sys.stderr)
            try:
                reader = pandas.read_csv(
                    args.input,
                    keep_default_na=False,
                    dtype=str,
                    chunksize=chunk_size)
                for chunk_df in reader:
                    if len(chunk_df) == 0:
                        continue
                    if args.sequence_id_column not in chunk_df:
                        chunk_df[args.sequence_id_column] = (
                            "sequence_" + chunk_df.index.astype(str))
                    chunk_df = chunk_df.set_index(args.sequence_id_column)
                    yield chunk_df[args.sequence_column].to_dict()
            except (FileNotFoundError, OSError, ValueError) as error:
                arg_parser.error(str(error))
        elif input_format == "fasta":
            print(
                "Streaming %s in chunks of %d sequences" % (
                    args.input, chunk_size),
                file=sys.stderr)
            use_fasta_id = args.sequence_id_column == "sequence_id"
            batch = {}
            row_index = 0
            try:
                for seq_id, sequence in FastaParser().iterate_over_file(args.input):
                    key = seq_id if use_fasta_id else ("sequence_%d" % row_index)
                    batch[key] = sequence
                    row_index += 1
                    if len(batch) >= chunk_size:
                        yield batch
                        batch = {}
                if batch:
                    yield batch
            except (FileNotFoundError, OSError, ValueError) as error:
                arg_parser.error(str(error))
        else:
            arg_parser.error("Unsupported input format: %r" % (input_format,))
    else:
        for start in range(0, len(args.sequences), chunk_size):
            end = min(start + chunk_size, len(args.sequences))
            yield {
                "sequence_%d" % idx: args.sequences[idx]
                for idx in range(start, end)
            }


def _filter_scan_results(result_df, alleles, args, explicit_thresholds, arg_parser):
    """Drop synthetic sample_name in processing-only mode and apply thresholds."""
    # In processing-only mode (no --alleles), omit the synthetic sample_name
    # placeholder ("sample1") since no sample was requested (#424).
    if not alleles and "sample_name" in result_df.columns:
        result_df = result_df.drop(columns=["sample_name"])

    # Apply thresholds, skipping ones whose column is missing (e.g.
    # processing-only run with no alleles, or --no-affinity-percentile
    # leaves affinity_percentile out).
    threshold_filters = [
        ("presentation_score", args.threshold_presentation_score, "ge"),
        ("processing_score", args.threshold_processing_score, "ge"),
        ("affinity", args.threshold_affinity, "le"),
        ("affinity_percentile", args.threshold_affinity_percentile, "le"),
    ]
    for col, thresh, op in threshold_filters:
        if thresh is None:
            continue
        if col not in result_df.columns:
            if col in explicit_thresholds:
                arg_parser.error(
                    "Cannot apply --threshold-%s: column '%s' is not present "
                    "in prediction output" % (col.replace("_", "-"), col)
                )
            continue
        if op == "ge":
            result_df = result_df.loc[result_df[col] >= thresh]
        else:
            result_df = result_df.loc[result_df[col] <= thresh]
    return result_df


def _predict_sequence_batch(
        sequences,
        alleles,
        models_dir,
        peptide_lengths,
        args,
        worker_pool,
        affinity_model_kwargs,
        chunk_ranges_for_local_parallelism,
        Class1PresentationPredictor,
        pandas,
        predictor=None):
    """Run presentation/processing prediction for a dict of sequences."""
    if worker_pool is None:
        if predictor is None:
            predictor = _load_predictor_for_command(models_dir)
        return predictor.predict_sequences(
            sequences=sequences,
            alleles=alleles,
            result="all",
            peptide_lengths=peptide_lengths,
            use_flanks=not args.no_flanking,
            include_affinity_percentile=not args.no_affinity_percentile,
            throw=not args.no_throw,
            affinity_model_kwargs=affinity_model_kwargs,
            processing_batch_size="auto")

    sequence_items = list(sequences.items())
    ranges = chunk_ranges_for_local_parallelism(
        len(sequence_items), args.num_jobs)
    work_items = []
    for (chunk_num, start, end) in ranges:
        work_items.append({
            "chunk_num": chunk_num,
            "models_dir": models_dir,
            "sequences": dict(sequence_items[start:end]),
            "alleles": alleles,
            "peptide_lengths": peptide_lengths,
            "use_flanks": not args.no_flanking,
            "include_affinity_percentile": (
                not args.no_affinity_percentile),
            "throw": not args.no_throw,
            "affinity_model_kwargs": affinity_model_kwargs,
            "processing_batch_size": "auto",
        })
    results = worker_pool.imap_unordered(
        _predict_sequences_chunk_worker, work_items, chunksize=1)
    chunks = [result for result in results]
    # Restore input sequence order by chunk_num before the global re-sort.
    chunks.sort(key=lambda chunk: chunk.chunk_num)
    result_df = pandas.concat(
        [chunk.predictions for chunk in chunks],
        ignore_index=True)
    # Apply the same global sort serial predict_sequences(result="all")
    # uses (Class1PresentationPredictor.sort_predictions). Each chunk is
    # sorted only within its sequence subset, so after concat we re-sort
    # globally. All chunks resolved the same comparison quantity, so we
    # take it from any of them — keeping parallel output ranking
    # identical to serial, including the deterministic peptide tie-break.
    comparison_quantity = chunks[0].comparison_quantity
    return Class1PresentationPredictor.sort_predictions(
        result_df, comparison_quantity).reset_index(drop=True)


def run(argv=sys.argv[1:]):
    argcomplete.autocomplete(parser)
    if not argv:
        parser.print_help()
        parser.exit(1)

    args = parser.parse_args(argv)

    try:
        selected_models_dir = (
            get_model_release_dir(args.model_release) if args.model_release else None)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))

    models_dir = args.models or selected_models_dir
    if models_dir is None:
        # The reason we set the default here instead of in the argument parser
        # is that we want to test_exists at this point, so the user gets a
        # message instructing them to download the models if needed.
        try:
            models_dir = get_default_class1_presentation_models_dir(test_exists=True)
        except RuntimeError as error:
            parser.error(str(error))
    elif not os.path.isdir(models_dir):
        parser.error("--models: no such directory: %s" % models_dir)

    if args.list_supported_alleles:
        try:
            print("\n".join(_supported_alleles_for_command(models_dir)))
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        return

    if args.list_supported_peptide_lengths:
        try:
            min_len, max_len = _supported_peptide_lengths_for_command(models_dir)
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        print("\n".join([str(length) for length in range(min_len, max_len + 1)]))
        return

    import pandas
    from ..encodable_sequences import EncodingError
    from ..parallelism import (
        chunk_ranges_for_local_parallelism,
        num_workers_per_gpu_from_args,
        worker_pool_with_gpu_assignments_from_args,
    )
    from ..pytorch_sizing import default_prediction_batch_is_auto
    from ..workload_planning import (
        WORKLOAD_PRESENTATION_INFERENCE,
        WORKLOAD_PROCESSING_INFERENCE,
        model_artifact_size_bytes,
        path_size_bytes,
    )
    from ..class1_presentation_predictor import Class1PresentationPredictor
    from ..fasta import FastaParser, read_fasta_to_dataframe

    # It's hard to pass a tab in a shell, so we correct a common error:
    if args.output_delimiter == "\\t":
        args.output_delimiter = "\t"
    if len(args.output_delimiter) != 1:
        parser.error(
            "--output-delimiter must be a single character, got %r"
            % (args.output_delimiter,)
        )
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir and not os.path.isdir(out_dir):
            parser.error(
                "--out: parent directory does not exist: %s" % out_dir
            )

    try:
        peptide_lengths = parse_peptide_lengths(args.peptide_lengths)
    except ValueError as error:
        parser.error(str(error))

    explicit_thresholds = {
        col
        for col, val in (
            ("presentation_score", args.threshold_presentation_score),
            ("processing_score", args.threshold_processing_score),
            ("affinity", args.threshold_affinity),
            ("affinity_percentile", args.threshold_affinity_percentile),
        )
        if val is not None
    }
    if "affinity_percentile" in explicit_thresholds and args.no_affinity_percentile:
        parser.error(
            "--threshold-affinity-percentile cannot be combined with "
            "--no-affinity-percentile"
        )
    if not args.alleles:
        for col in ("affinity", "affinity_percentile", "presentation_score"):
            if col in explicit_thresholds:
                parser.error(
                    "--threshold-%s requires --alleles" % col.replace("_", "-")
                )
    if (
            not args.results_all
            and not explicit_thresholds
            and args.alleles
            and not args.no_affinity_percentile):
        print(
            "Filtering by affinity-percentile <= %s "
            "(pass --results-all to show all predictions)."
            % default_thresholds["affinity_percentile"],
            file=sys.stderr)
        args.threshold_affinity_percentile = default_thresholds["affinity_percentile"]

    if args.input:
        if args.sequences:
            parser.error(
                "If an input file is specified, do not specify --sequences")
        input_format = _resolve_input_format(args, parser)
    else:
        if not args.sequences:
            parser.error(
                "Specify either an input file or the --sequences argument")
        input_format = None

    if args.alleles:
        alleles = {}
        for i, allele_string in enumerate(args.alleles):
            parsed = [
                token for token in re.split(r"[;,\s]+", allele_string.strip())
                if token
            ]
            if not parsed:
                parser.error(
                    "Empty allele specification in --alleles: %r" % (
                        allele_string,))
            alleles["genotype_%02d" % i] = parsed
    else:
        print(
            "No alleles specified. Will perform processing prediction only.",
            file=sys.stderr)
        alleles = {}

    csv_kwargs = {
        "index": False,
        "sep": args.output_delimiter,
    }
    if args.float_format is not None:
        csv_kwargs["float_format"] = args.float_format

    if args.chunk_size is not None:
        chunk_iter = _iter_sequence_chunks(
            args, input_format, parser, pandas, FastaParser)
        first_chunk = next(chunk_iter, None)
        if first_chunk is None:
            predictor = _load_predictor_for_command(models_dir)
            result_df = pandas.DataFrame(
                columns=predictor.predict_sequences_columns(
                    alleles=alleles,
                    use_flanks=not args.no_flanking,
                    include_affinity_percentile=not args.no_affinity_percentile,
                ))
            result_df = _filter_scan_results(
                result_df, alleles, args, explicit_thresholds, parser)
            if args.out:
                try:
                    result_df.to_csv(args.out, **csv_kwargs)
                except OSError as error:
                    parser.error(str(error))
                print("Wrote: %s" % args.out, file=sys.stderr)
            else:
                result_df.to_csv(sys.stdout, **csv_kwargs)
            return

        worker_pool = worker_pool_with_gpu_assignments_from_args(
            args,
            workload_name=(
                WORKLOAD_PRESENTATION_INFERENCE
                if alleles
                else WORKLOAD_PROCESSING_INFERENCE
            ),
            workload_hints={
                "data_bytes": path_size_bytes(args.input),
                "elastic_batch": (
                    not alleles or default_prediction_batch_is_auto()),
                "model_bytes": model_artifact_size_bytes(models_dir),
                "num_rows": args.chunk_size,
            },
            start_method="spawn",
        )
        affinity_model_kwargs = {}
        if hasattr(args, "max_workers_per_gpu"):
            affinity_model_kwargs["num_workers_per_gpu"] = (
                num_workers_per_gpu_from_args(args)
            )
        predictor = None
        out_fd = None
        wrote_header = False
        last_columns = None
        try:
            if worker_pool is None:
                predictor = _load_predictor_for_command(models_dir)
            for chunk_sequences in itertools.chain([first_chunk], chunk_iter):
                result_df = _predict_sequence_batch(
                    sequences=chunk_sequences,
                    alleles=alleles,
                    models_dir=models_dir,
                    peptide_lengths=peptide_lengths,
                    args=args,
                    worker_pool=worker_pool,
                    affinity_model_kwargs=affinity_model_kwargs,
                    chunk_ranges_for_local_parallelism=chunk_ranges_for_local_parallelism,
                    Class1PresentationPredictor=Class1PresentationPredictor,
                    pandas=pandas,
                    predictor=predictor,
                )
                result_df = _filter_scan_results(
                    result_df, alleles, args, explicit_thresholds, parser)
                last_columns = list(result_df.columns)
                if len(result_df) > 0:
                    if args.out and out_fd is None:
                        out_fd = open(args.out, "w", newline="")
                    target = out_fd if args.out else sys.stdout
                    result_df.to_csv(
                        target, header=not wrote_header, **csv_kwargs)
                    wrote_header = True
            if not wrote_header:
                empty_df = pandas.DataFrame(columns=last_columns or [])
                if args.out and out_fd is None:
                    out_fd = open(args.out, "w", newline="")
                target = out_fd if args.out else sys.stdout
                empty_df.to_csv(target, header=True, **csv_kwargs)
            if worker_pool is not None:
                worker_pool.close()
                worker_pool.join()
                worker_pool = None
        except (FileNotFoundError, OSError, ValueError, RuntimeError, EncodingError) as error:
            parser.error(str(error))
        finally:
            if out_fd is not None:
                out_fd.close()
            if worker_pool is not None:
                worker_pool.terminate()
                worker_pool.join()
        if args.out:
            print("Wrote: %s" % args.out, file=sys.stderr)
        return

    if args.input:
        if input_format == "csv":
            try:
                df = pandas.read_csv(
                    args.input, keep_default_na=False, dtype=str)
            except (FileNotFoundError, OSError, ValueError) as error:
                parser.error(str(error))
            print(
                "Read input CSV with %d rows, columns are: %s" % (
                    len(df), ", ".join(df.columns)),
                file=sys.stderr)
            for col in [args.sequence_column,]:
                if col not in df.columns:
                    parser.error(
                        "No such column '%s' in CSV. Columns are: %s" % (
                            col, ", ".join(["'%s'" % c for c in df.columns])))

        elif input_format == "fasta":
            try:
                df = read_fasta_to_dataframe(args.input)
            except (FileNotFoundError, OSError, ValueError) as error:
                parser.error(str(error))
            print(
                "Read %d sequences from %s" % (len(df), args.input),
                file=sys.stderr)
        else:
            parser.error("Unsupported input format: %r" % (input_format,))
    else:
        df = pandas.DataFrame({
            args.sequence_column: args.sequences,
        })

    if args.sequence_id_column not in df:
        df[args.sequence_id_column] = "sequence_" + df.index.astype(str)

    df = df.set_index(args.sequence_id_column)

    sequences = df[args.sequence_column].to_dict()
    if not sequences:
        # Empty input: return a DataFrame with the same schema the
        # predictor would have produced for non-empty input, so
        # threshold filters and downstream readers don't see a
        # schema-less object. The column list is derived from the
        # predictor (single source of truth) — see test_empty_and_populated_schemas_match.
        predictor = _load_predictor_for_command(models_dir)
        result_df = pandas.DataFrame(
            columns=predictor.predict_sequences_columns(
                alleles=alleles,
                use_flanks=not args.no_flanking,
                include_affinity_percentile=not args.no_affinity_percentile,
            ))
    else:
        worker_pool = worker_pool_with_gpu_assignments_from_args(
            args,
            workload_name=(
                WORKLOAD_PRESENTATION_INFERENCE
                if alleles
                else WORKLOAD_PROCESSING_INFERENCE
            ),
            workload_hints={
                "data_bytes": path_size_bytes(args.input),
                "elastic_batch": (
                    not alleles or default_prediction_batch_is_auto()),
                "model_bytes": model_artifact_size_bytes(models_dir),
                "num_rows": len(df),
            },
            # Predict-scan workers run PyTorch inference. On Linux, the default
            # fork context can inherit a parent process where PyTorch has
            # already been initialized by earlier command invocations/tests,
            # which makes later model .to(...) calls unsafe in workers.
            start_method="spawn",
        )
        affinity_model_kwargs = {}
        if hasattr(args, "max_workers_per_gpu"):
            affinity_model_kwargs["num_workers_per_gpu"] = (
                num_workers_per_gpu_from_args(args)
            )
        try:
            result_df = _predict_sequence_batch(
                sequences=sequences,
                alleles=alleles,
                models_dir=models_dir,
                peptide_lengths=peptide_lengths,
                args=args,
                worker_pool=worker_pool,
                affinity_model_kwargs=affinity_model_kwargs,
                chunk_ranges_for_local_parallelism=chunk_ranges_for_local_parallelism,
                Class1PresentationPredictor=Class1PresentationPredictor,
                pandas=pandas,
            )
            if worker_pool is not None:
                worker_pool.close()
                worker_pool.join()
                worker_pool = None
        except (FileNotFoundError, OSError, ValueError, RuntimeError, EncodingError) as error:
            parser.error(str(error))
        finally:
            # On failure mid-iteration, terminate() rather than
            # close()/join() (which can hang on a wedged worker) and
            # leave non-daemon workers behind.
            if worker_pool is not None:
                worker_pool.terminate()
                worker_pool.join()

    result_df = _filter_scan_results(
        result_df, alleles, args, explicit_thresholds, parser)

    # Write results
    if args.out:
        try:
            result_df.to_csv(args.out, **csv_kwargs)
        except OSError as error:
            parser.error(str(error))
        print("Wrote: %s" % args.out, file=sys.stderr)
    else:
        result_df.to_csv(sys.stdout, **csv_kwargs)
