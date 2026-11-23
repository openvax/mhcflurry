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

"""Import-light argument parsers for training, selection, and calibration CLIs.

Building these parsers here avoids importing torch, numpy, pandas, sklearn,
scipy, mhcgnomes, tqdm, or yaml during shell tab completion. Each command
module re-exports its parser as module-level ``parser`` so docs generation
(sphinx-autoprogram) and existing callers remain unchanged.
"""

import argparse

from ..cli_args import (
    add_alleles_arg,
    add_cluster_parallelism_args,
    add_data_arg,
    add_debug_arg,
    add_hyperparameters_arg,
    add_max_epochs_arg,
    add_models_dir_arg,
    add_out_models_dir_arg,
    add_random_seed_arg,
    add_verbosity_arg,
    fraction_arg,
    positive_float_arg,
    positive_int_arg,
)
from ..parallelism.cli_args import add_local_parallelism_args
from .help import HelpArgumentParser


def held_out_fraction_reciprocal_arg(value):
    result = positive_int_arg(value)
    if result < 2:
        raise argparse.ArgumentTypeError(
            "held-out fraction reciprocal must be at least 2"
        )
    return result


def batch_size_arg(value):
    """Accept either an int or the literal string 'auto' for --prediction-batch-size.

    ``auto`` (the default) delegates sizing to
    ``mhcflurry.pytorch_sizing.compute_prediction_batch_size``,
    which picks a per-GPU-memory batch, reserving the VRAM partition
    across co-resident workers.
    """
    if isinstance(value, str) and value.strip().lower() in ("auto", ""):
        return "auto"
    return positive_int_arg(value)


def make_train_allele_specific_models_parser(usage=None):
    parser = HelpArgumentParser(usage=usage)
    add_data_arg(
        parser,
        required=True,
        help=(
            "Training data CSV. Expected columns: "
            "allele, peptide, measurement_value"))
    add_out_models_dir_arg(
        parser, help="Directory to write models and manifest")
    add_hyperparameters_arg(
        parser,
        required=True,
        help="JSON or YAML of hyperparameters")
    add_alleles_arg(
        parser,
        dest="allele",
        help="Alleles to train models for. If not specified, all alleles with "
        "enough measurements will be used.")
    parser.add_argument(
        "--min-measurements-per-allele",
        type=positive_int_arg,
        metavar="N",
        default=50,
        help="Train models for alleles with >=N measurements.")
    parser.add_argument(
        "--held-out-fraction-reciprocal",
        type=held_out_fraction_reciprocal_arg,
        metavar="N",
        default=None,
        help="Hold out 1/N fraction of data (for e.g. subsequent model selection. "
        "For example, specify 5 to hold out 20 percent of the data.")
    parser.add_argument(
        "--held-out-fraction-seed",
        type=int,
        metavar="N",
        default=None,
        help="Seed for randomizing which measurements are held out. Only matters "
        "when --held-out-fraction-reciprocal is specified. When omitted, the "
        "held-out split "
        "is derived from --random-seed (so the whole run reproduces from one "
        "value). Pass this explicitly to control the split directly — e.g. to "
        "reproduce a pre-2.3.0 split — overriding --random-seed for the split.")
    add_random_seed_arg(parser)
    parser.add_argument(
        "--ignore-inequalities",
        action="store_true",
        default=False,
        help="Do not use affinity value inequalities even when present in data")
    parser.add_argument(
        "--n-models",
        type=positive_int_arg,
        metavar="N",
        help="Ensemble size, i.e. how many models to train for each architecture. "
        "If specified here it overrides any 'n_models' specified in the "
        "hyperparameters.")
    add_max_epochs_arg(parser)
    parser.add_argument(
        "--allele-sequences",
        metavar="FILE.csv",
        help="Allele sequences file. Used for computing allele similarity matrix.")
    parser.add_argument(
        "--save-interval",
        type=positive_float_arg,
        metavar="N",
        default=60,
        help="Write models to disk every N seconds. Only affects parallel runs; "
        "serial runs write each model to disk as it is trained.")
    add_verbosity_arg(parser)
    add_local_parallelism_args(parser)
    return parser


def make_train_pan_allele_models_parser(usage=None):
    parser = HelpArgumentParser(usage=usage)
    parser.add_argument(
        "--reuse-folds", action="store_true",
        help="Use existing fold_0..N columns instead of generating new folds.")
    add_data_arg(
        parser,
        help=(
            "Training data CSV. Expected columns: "
            "allele, peptide, measurement_value"))
    parser.add_argument(
        "--pretrain-data",
        metavar="FILE.csv",
        help=(
            "Pre-training data CSV. Expected columns: "
            "allele, peptide, measurement_value"))
    add_out_models_dir_arg(
        parser, help="Directory to write models and manifest")
    add_hyperparameters_arg(
        parser,
        help="JSON or YAML of hyperparameters")
    parser.add_argument(
        "--held-out-measurements-per-allele-fraction-and-max",
        type=float,
        metavar="X",
        nargs=2,
        default=[0.25, 100],
        help="Fraction of measurements per allele to hold out, and maximum number")
    parser.add_argument(
        "--ignore-inequalities",
        action="store_true",
        default=False,
        help="Do not use affinity value inequalities even when present in data")
    parser.add_argument(
        "--num-folds",
        type=positive_int_arg,
        default=4,
        metavar="N",
        help="Number of training folds.")
    parser.add_argument(
        "--num-replicates",
        type=positive_int_arg,
        metavar="N",
        default=1,
        help="Number of replicates per (architecture, fold) pair to train.")
    add_random_seed_arg(parser)
    add_max_epochs_arg(parser)
    parser.add_argument(
        "--allele-sequences",
        metavar="FILE.csv",
        help="Allele sequences file.")
    add_verbosity_arg(parser)
    add_debug_arg(parser)
    parser.add_argument(
        "--continue-incomplete",
        action="store_true",
        default=False,
        help="Continue training models from an incomplete training run. If this is "
        "specified then the only required argument is --out-models-dir")
    parser.add_argument(
        "--only-initialize",
        action="store_true",
        default=False,
        help="Do not actually train models. The initialized run can be continued "
        "later with --continue-incomplete.")
    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        default=False,
        help=(
            "Retain both terminal and minimum-validation weights for every fit. "
            "The restore_best_weights hyperparameter still selects the primary "
            "predictor weights."
        ))
    add_local_parallelism_args(parser)
    add_cluster_parallelism_args(parser)
    return parser


def make_train_processing_models_parser(usage=None):
    parser = HelpArgumentParser(usage=usage)
    parser.add_argument(
        "--processing-data-policy",
        choices=("matched", "legacy"),
        default="matched",
        help="Require matched processing data; legacy is explicit historical replay only.")
    parser.add_argument(
        "--reuse-folds", action="store_true",
        help="Use existing fold_0..N columns instead of generating new folds.")
    add_data_arg(
        parser,
        help="Training data CSV. Expected columns: peptide, n_flank, c_flank, hit")
    add_out_models_dir_arg(
        parser, help="Directory to write models and manifest")
    add_hyperparameters_arg(
        parser,
        help="JSON or YAML of hyperparameters")
    parser.add_argument(
        "--held-out-samples",
        type=positive_int_arg,
        metavar="N",
        default=10,
        help="Number of experiments to hold out per fold")
    parser.add_argument(
        "--num-folds",
        type=positive_int_arg,
        default=4,
        metavar="N",
        help="Number of training folds.")
    parser.add_argument(
        "--num-replicates",
        type=positive_int_arg,
        metavar="N",
        default=1,
        help="Number of replicates per (architecture, fold) pair to train.")
    add_random_seed_arg(parser)
    add_max_epochs_arg(parser)
    add_verbosity_arg(parser)
    add_debug_arg(parser)
    parser.add_argument(
        "--continue-incomplete",
        action="store_true",
        default=False,
        help="Continue training models from an incomplete training run. If this is "
        "specified then the only required argument is --out-models-dir")
    parser.add_argument(
        "--only-initialize",
        action="store_true",
        default=False,
        help="Do not actually train models. The initialized run can be continued "
        "later with --continue-incomplete.")
    add_local_parallelism_args(parser)
    add_cluster_parallelism_args(parser)
    return parser


def make_train_presentation_models_parser(usage=None):
    parser = HelpArgumentParser(usage=usage)
    add_data_arg(
        parser,
        help="Training data CSV. Expected columns: peptide, n_flank, c_flank, hit")
    add_out_models_dir_arg(
        parser, help="Directory to write models and manifest")
    parser.add_argument(
        "--affinity-predictor",
        metavar="DIR",
        required=True,
        help="Affinity predictor models dir")
    parser.add_argument(
        "--processing-predictor-with-flanks",
        metavar="DIR",
        required=True,
        help="Processing predictor with flanks")
    parser.add_argument(
        "--processing-predictor-without-flanks",
        metavar="DIR",
        required=True,
        help="Processing predictor without flanks")
    add_verbosity_arg(parser, default=1)
    add_debug_arg(parser)
    parser.add_argument(
        "--hla-column",
        default="hla",
        help="Column in data giving space-separated MHC I alleles")
    parser.add_argument(
        "--target-column",
        default="hit",
        help="Column in data giving hit (1) vs decoy (0)")
    parser.add_argument(
        "--feature-chunk-size",
        type=positive_int_arg,
        default=250000,
        metavar="N",
        help=(
            "Rows per parallel presentation feature-prediction task. Larger "
            "chunks reduce scheduling overhead but increase worker memory. "
            "Default: %(default)s"))
    add_local_parallelism_args(parser)
    add_random_seed_arg(parser)
    return parser


def make_select_allele_specific_models_parser(usage=None):
    parser = HelpArgumentParser(usage=usage)
    add_data_arg(
        parser,
        required=False,
        help=(
            "Model selection data CSV. Expected columns: "
            "allele, peptide, measurement_value"))
    parser.add_argument(
        "--exclude-data",
        metavar="FILE.csv",
        required=False,
        help=(
            "Data to EXCLUDE from model selection. Useful to specify the original "
            "training data used"))
    add_models_dir_arg(
        parser,
        required=True,
        help="Directory to read models")
    add_out_models_dir_arg(
        parser, help="Directory to write selected models")
    parser.add_argument(
        "--out-unselected-predictions",
        metavar="FILE.csv",
        help="Write predictions for validation data using unselected predictor to "
        "FILE.csv")
    parser.add_argument(
        "--unselected-accuracy-scorer",
        metavar="SCORER",
        default="combined:mass-spec,mse")
    parser.add_argument(
        "--unselected-accuracy-scorer-num-samples",
        type=positive_int_arg,
        default=1000)
    parser.add_argument(
        "--unselected-accuracy-percentile-threshold",
        type=float,
        metavar="X",
        default=95)
    add_alleles_arg(
        parser,
        dest="allele",
        help="Alleles to select models for. If not specified, all alleles with "
        "enough measurements will be used.")
    parser.add_argument(
        "--combined-min-models",
        type=positive_int_arg,
        default=8,
        metavar="N",
        help="Min number of models to select per allele when using combined selector")
    parser.add_argument(
        "--combined-max-models",
        type=positive_int_arg,
        default=1000,
        metavar="N",
        help="Max number of models to select per allele when using combined selector")
    parser.add_argument(
        "--combined-min-contribution-percent",
        type=float,
        default=1.0,
        metavar="X",
        help="Use only model selectors that can contribute at least X %% to the "
        "total score. Default: %(default)s")
    parser.add_argument(
        "--mass-spec-min-measurements",
        type=positive_int_arg,
        metavar="N",
        default=1,
        help="Min number of measurements required for an allele to use mass-spec model "
        "selection")
    parser.add_argument(
        "--mass-spec-min-models",
        type=positive_int_arg,
        default=8,
        metavar="N",
        help="Min number of models to select per allele when using mass-spec selector")
    parser.add_argument(
        "--mass-spec-max-models",
        type=positive_int_arg,
        default=1000,
        metavar="N",
        help="Max number of models to select per allele when using mass-spec selector")
    parser.add_argument(
        "--mse-min-measurements",
        type=positive_int_arg,
        metavar="N",
        default=1,
        help="Min number of measurements required for an allele to use MSE model "
        "selection")
    parser.add_argument(
        "--mse-min-models",
        type=positive_int_arg,
        default=8,
        metavar="N",
        help="Min number of models to select per allele when using MSE selector")
    parser.add_argument(
        "--mse-max-models",
        type=positive_int_arg,
        default=1000,
        metavar="N",
        help="Max number of models to select per allele when using MSE selector")
    parser.add_argument(
        "--scoring",
        nargs="+",
        default=["mse", "consensus"],
        help="Scoring procedures to use in order")
    parser.add_argument(
        "--consensus-min-models",
        type=positive_int_arg,
        default=8,
        metavar="N",
        help="Min number of models to select per allele when using consensus selector")
    parser.add_argument(
        "--consensus-max-models",
        type=positive_int_arg,
        default=1000,
        metavar="N",
        help="Max number of models to select per allele when using consensus selector")
    parser.add_argument(
        "--consensus-num-peptides-per-length",
        type=positive_int_arg,
        default=10000,
        help="Num peptides per length to use for consensus scoring")
    parser.add_argument(
        "--mass-spec-regex",
        metavar="REGEX",
        default="mass[- ]spec",
        help="Regular expression for mass-spec data. Runs on measurement_source col."
        "Default: %(default)s.")
    add_verbosity_arg(parser)
    add_local_parallelism_args(parser)
    add_random_seed_arg(parser)
    return parser


def make_select_pan_allele_models_parser(usage=None):
    parser = HelpArgumentParser(usage=usage)
    add_data_arg(
        parser,
        required=False,
        help=(
            "Model selection data CSV. Expected columns: "
            "allele, peptide, measurement_value"))
    add_models_dir_arg(
        parser,
        required=True,
        help="Directory to read models")
    add_out_models_dir_arg(
        parser, help="Directory to write selected models")
    parser.add_argument(
        "--min-models-per-fold",
        type=positive_int_arg,
        default=None,
        metavar="N",
        help=(
            "Min number of models to select per fold. Default: 2, or the "
            "requested maximum when it is smaller."))
    parser.add_argument(
        "--max-models-per-fold",
        type=positive_int_arg,
        default=1000,
        metavar="N",
        help="Max number of models to select per fold")
    parser.add_argument(
        "--mass-spec-regex",
        metavar="REGEX",
        default="mass[- ]spec",
        help="Regular expression for mass-spec data. Runs on measurement_source col."
        "Default: %(default)s.")
    add_verbosity_arg(parser)
    parser.add_argument(
        "--save-validation-predictions",
        action="store_true",
        default=False,
        help=(
            "Save row-level out-of-fold predictions from the selected ensemble "
            "as model_selection_predictions.csv.bz2."))
    add_local_parallelism_args(parser)
    add_cluster_parallelism_args(parser)
    return parser


def make_select_processing_models_parser(usage=None):
    parser = HelpArgumentParser(usage=usage)
    parser.add_argument(
        "--processing-data-policy",
        choices=("matched", "legacy"),
        default="matched")
    add_data_arg(
        parser,
        required=False,
        help=(
            "Model selection data CSV. Expected columns: "
            "peptide, hit, fold_0, ..., fold_N"))
    add_models_dir_arg(
        parser,
        required=True,
        help="Directory to read models")
    add_out_models_dir_arg(
        parser, help="Directory to write selected models")
    parser.add_argument(
        "--min-models-per-fold",
        type=positive_int_arg,
        default=None,
        metavar="N",
        help=(
            "Min number of models to select per fold. Default: 2, or the "
            "requested maximum when it is smaller."))
    parser.add_argument(
        "--max-models-per-fold",
        type=positive_int_arg,
        default=1000,
        metavar="N",
        help="Max number of models to select per fold")
    add_verbosity_arg(parser)
    parser.add_argument(
        "--save-validation-predictions",
        action="store_true",
        default=False,
        help=(
            "Save row-level out-of-fold predictions from the selected ensemble "
            "as model_selection_predictions.csv.bz2."))
    add_local_parallelism_args(parser)
    add_cluster_parallelism_args(parser)
    return parser


def make_calibrate_percentile_ranks_parser(usage=None):
    parser = HelpArgumentParser(usage=usage)
    parser.add_argument(
        "--predictor-kind",
        choices=("class1_affinity", "class1_processing", "class1_presentation"),
        default="class1_affinity",
        help="Type of predictor to calibrate")
    parser.add_argument(
        "--percentile-method", choices=("compact", "histogram"), default="compact",
        help="Method for new calibration (default: %(default)s). Loading alone "
        "never changes saved ranks. Histogram reproduction also requires the "
        "original reference and bin policy.")
    parser.add_argument(
        "--max-percentile-knots", type=int, choices=(64, 128), default=128,
        help="Compact starts at 64; allow 128 only when background validation "
        "justifies it. This is a ceiling (default: %(default)s), not a fixed size. "
        "Ignored for histogram calibration.")
    parser.add_argument(
        "--processing-reference-data",
        help="Required for processing: independent background CSV with peptide and, "
        "for a flanked model, n_flank and c_flank columns. No evaluation labels are used. "
        "Scores and input sequences are saved in percent_rank_reference.npz. "
        "Requires --num-jobs 0; reference-generation options do not apply.")
    add_models_dir_arg(
        parser,
        required=True,
        help="Directory to read and write models")
    add_alleles_arg(
        parser,
        dest="allele",
        help="Alleles to calibrate percentile ranks for. If not specified all "
        "alleles are used")
    parser.add_argument(
        "--match-amino-acid-distribution-data",
        help="Sample random peptides from the amino acid distribution of the "
        "peptides listed in the supplied CSV file, which must have a 'peptide' "
        "column. If not specified a uniform distribution is used.")
    parser.add_argument(
        "--alleles-file",
        default=None,
        help="Use alleles in supplied CSV file, which must have an 'allele' column.")
    parser.add_argument(
        "--list-percent-rank-status",
        default=False,
        action="store_true",
        help="For class1 affinity predictors, print a CSV indicating which requested "
        "alleles already have percentile-rank calibration and exit without "
        "generating calibration peptides.")
    parser.add_argument(
        "--only-missing",
        dest="only_missing_percent_ranks",
        default=False,
        action="store_true",
        help="For class1 affinity predictors, calibrate only requested alleles that "
        "do not already have direct or sequence-equivalent percentile-rank "
        "calibration.")
    parser.add_argument(
        "--num-peptides-per-length",
        type=positive_int_arg,
        metavar="N",
        default=int(1e5),
        help="For affinity/presentation, number of generated peptides per length "
        "to use to calibrate percent ranks. Processing uses --processing-reference-data. "
        "Default: %(default)s.")
    parser.add_argument(
        "--num-genotypes",
        type=positive_int_arg,
        metavar="N",
        default=25,
        help="Used when calibrating a presentation predictor. Number of MHC allele sets"
        "to sample")
    parser.add_argument(
        "--alleles-per-genotype",
        type=positive_int_arg,
        metavar="N",
        default=6,
        help="Used when calibrating a presentation predictor. Number of alleles "
        "per MHC allele set. Use 1 to calibrate for single alleles. Default: %(default)s")
    parser.add_argument(
        "--motif-summary",
        default=False,
        action="store_true",
        help="Calculate motifs and length preferences for each allele")
    parser.add_argument(
        "--summary-top-peptide-fraction",
        default=[0.0001, 0.001, 0.01, 0.1, 1.0],
        nargs="+",
        type=fraction_arg,
        metavar="X",
        help="The top X fraction of predictions (i.e. tightest binders) to use to "
        "generate motifs and length preferences. Default: %(default)s")
    parser.add_argument(
        "--length-range",
        default=(8, 15),
        type=positive_int_arg,
        nargs=2,
        help="Min and max peptide length to calibrate, inclusive. "
        "Default: %(default)s")
    parser.add_argument(
        "--prediction-batch-size",
        type=batch_size_arg,
        default="auto",
        help="Batch size for predictions. Pass an int to pin, or 'auto' "
             "(default) to size per GPU free memory / workers-per-GPU — see "
             "mhcflurry.pytorch_sizing.compute_prediction_batch_size.")
    parser.add_argument(
        "--alleles-per-work-chunk",
        type=positive_int_arg,
        metavar="N",
        default=1,
        help="Number of alleles per work chunk. Default: %(default)s.")
    add_verbosity_arg(parser)
    parser.add_argument(
        "--gpu-batched",
        default=False,
        action="store_true",
        help="[class1 affinity predictors only] Use the GPU-hoisted "
             "calibration fast path: precompute peptide-side activations "
             "per network and batch "
             "--gpu-allele-batch-size alleles into a single forward through "
             "the merge + main dense path. Uses the same ensemble aggregation "
             "and percentile method as the ordinary path; floating-point batching "
             "can change predictions and selected knots slightly, so saved curves "
             "are not guaranteed bit-identical. Applies to affinity in serial, "
             "local-pool, and cluster modes; ignored for processing/presentation.")
    parser.add_argument(
        "--gpu-allele-batch-size",
        type=batch_size_arg,
        default="auto",
        help="Alleles per GPU forward when --gpu-batched. Pass an int to "
             "pin; 'auto' (default) partitions the VRAM budget with "
             "--max-workers-per-gpu. Larger values trade off more VRAM for "
             "fewer kernel launches.")
    parser.add_argument(
        "--gpu-peptide-batch-size",
        type=batch_size_arg,
        default="auto",
        help="Peptide chunk size on device when --gpu-batched. Pass an int "
             "to pin; 'auto' (default) picks the peptide axis of the "
             "auto-sized budget. Reducing keeps peak VRAM down on smaller "
             "GPUs but adds kernel-launch overhead.")
    # Calibrate deliberately uses the broader training-flavored parallelism set
    # (add_local_parallelism_args) rather than the prediction subset
    # (add_prediction_parallelism_args used by predict / predict-scan): it needs
    # the training-only knobs (e.g. --max-tasks-per-worker, --worker-log-dir,
    # cluster parallelism). The shared flags (--num-jobs/--gpus/--max-workers-per-
    # gpu) are identical across all three, so do not "simplify" this to the
    # prediction helper.
    add_local_parallelism_args(parser)
    add_cluster_parallelism_args(parser)
    add_random_seed_arg(parser)
    return parser


SUBCOMMAND_PARSER_BUILDERS = {
    "class1-train-allele-specific-models": make_train_allele_specific_models_parser,
    "class1-train-pan-allele-models": make_train_pan_allele_models_parser,
    "class1-train-processing-models": make_train_processing_models_parser,
    "class1-train-presentation-models": make_train_presentation_models_parser,
    "class1-select-allele-specific-models": make_select_allele_specific_models_parser,
    "class1-select-pan-allele-models": make_select_pan_allele_models_parser,
    "class1-select-processing-models": make_select_processing_models_parser,
    "calibrate-percentile-ranks": make_calibrate_percentile_ranks_parser,
}
