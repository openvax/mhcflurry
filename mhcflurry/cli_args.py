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

"""Import-light CLI argument types and shared parser helpers.

Kept free of numerical and ML dependencies (numpy, pandas, torch, mhcgnomes,
yaml) so shell tab completion and ``--help`` stay fast.
"""

import argparse
import math
import numbers


# Default master seed for CLI commands that accept --random-seed. A fixed
# seed supports reproducible sampling; numerical results can still depend on
# hardware, library versions and nondeterministic kernels.
DEFAULT_RANDOM_SEED = 42


def positive_int_arg(value):
    """Parse an argparse value as an integer greater than zero."""
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError(
            "expected a positive integer, got %r" % (value,)
        )
    if (
        isinstance(value, numbers.Real)
        and not isinstance(value, numbers.Integral)
        and not float(value).is_integer()
    ):
        raise argparse.ArgumentTypeError(
            "expected a positive integer, got %r" % (value,)
        )
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "expected a positive integer, got %r" % (value,)
        ) from None
    if result < 1:
        raise argparse.ArgumentTypeError(
            "expected a positive integer, got %r" % (value,)
        )
    return result


def positive_float_arg(value):
    """Parse an argparse value as a finite float greater than zero."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "expected a positive number, got %r" % (value,)
        ) from None
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError(
            "expected a positive number, got %r" % (value,)
        )
    return result


def fraction_arg(value):
    """Parse an argparse value as a finite fraction in ``(0, 1]``."""
    result = positive_float_arg(value)
    if result > 1:
        raise argparse.ArgumentTypeError(
            "expected a fraction in (0, 1], got %r" % (value,)
        )
    return result


def add_random_seed_arg(parser):
    """Add the standard ``--random-seed`` argument to a CLI parser.

    Training commands use this flag to record a shared source of random seeds.
    It does not guarantee identical results across devices or library versions.
    Pair with :func:`mhcflurry.common.configure_random_seed`
    at the start of the command's ``run``/``main``.
    """
    parser.add_argument(
        "--random-seed",
        type=int,
        metavar="N",
        default=DEFAULT_RANDOM_SEED,
        help="Master random seed for this command "
        "(numpy, Python `random`, and torch): data shuffles, fold/held-out "
        "sampling, weight initialization, and random peptide/negative "
        "sampling. Default: %(default)s. Exact numerical results can still "
        "vary with hardware, library versions and nondeterministic kernels.")


def add_verbosity_arg(parser, default=0):
    """Add the standard integer ``--verbosity`` argument to a CLI parser.

    The training and selection commands call
    ``configure_logging(verbose=args.verbosity > 1)``; most also forward the
    value to model ``fit`` as ``verbose``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the argument to.
    default : int
        Verbosity used when the flag is not given.
    """
    parser.add_argument(
        "--verbosity",
        type=int,
        help="Verbosity. Default: %(default)s",
        default=default)


def add_debug_arg(parser):
    """Add the standard ``--debug`` flag, which starts pdb on an error.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the flag to.
    """
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Launch python debugger on error")


def add_out_models_dir_arg(parser, help):
    """Add the required ``--out-models-dir`` argument to a CLI parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the argument to.
    help : str
        Command-specific help text, e.g. what is written to the directory.
    """
    parser.add_argument(
        "--out-models-dir",
        metavar="DIR",
        required=True,
        help=help)


def add_data_arg(parser, help=None, *, required=False, metavar="FILE.csv"):
    """Add the standard ``--data`` argument to a CLI parser."""
    kwargs = {"metavar": metavar, "required": required}
    if help is not None:
        kwargs["help"] = help
    parser.add_argument("--data", **kwargs)


def add_models_dir_arg(parser, help=None, *, required=True, metavar="DIR"):
    """Add the standard ``--models-dir`` argument to a CLI parser."""
    kwargs = {"metavar": metavar, "required": required}
    if help is not None:
        kwargs["help"] = help
    parser.add_argument("--models-dir", **kwargs)


def add_hyperparameters_arg(
        parser, help=None, *, required=False, metavar="FILE.json"):
    """Add the standard ``--hyperparameters`` argument to a CLI parser."""
    kwargs = {"metavar": metavar, "required": required}
    if help is not None:
        kwargs["help"] = help
    parser.add_argument("--hyperparameters", **kwargs)


def add_out_arg(parser, help=None, *, required=False, metavar=None):
    """Add the standard ``--out`` argument to a CLI parser."""
    kwargs = {"required": required}
    if metavar is not None:
        kwargs["metavar"] = metavar
    if help is not None:
        kwargs["help"] = help
    parser.add_argument("--out", **kwargs)


def add_alleles_arg(
        parser,
        help=None,
        *,
        dest="alleles",
        required=False,
        default=None,
        metavar=None):
    """Add ``--alleles`` / ``--allele`` (accepting both spellings) to a parser."""
    kwargs = {
        "dest": dest,
        "nargs": "+",
        "default": default,
        "required": required,
    }
    if metavar is not None:
        kwargs["metavar"] = metavar
    if help is not None:
        kwargs["help"] = help
    parser.add_argument("--alleles", "--allele", **kwargs)


def add_max_epochs_arg(parser):
    """Add ``--max-epochs``, which overrides the hyperparameters' ``max_epochs``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the argument to. The parsed value is a positive int, or
        ``None`` when the flag is not given.
    """
    parser.add_argument(
        "--max-epochs",
        type=positive_int_arg,
        metavar="N",
        help="Max training epochs. If specified here it overrides any "
        "'max_epochs' specified in the hyperparameters.")


def add_cluster_parallelism_args(parser):
    """
    Add commandline arguments controlling cluster parallelism to an argparse
    ArgumentParser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    group = parser.add_argument_group("Cluster parallelism")
    group.add_argument(
        "--cluster-parallelism",
        default=False,
        action="store_true")
    group.add_argument(
        "--cluster-submit-command",
        default="sh",
        help="Default: %(default)s")
    group.add_argument(
        "--cluster-results-workdir",
        default="./cluster-workdir",
        help="Default: %(default)s")
    group.add_argument(
        "--additional-complete-file",
        default="STDERR",
        help="Additional file to monitor for job completion. Default: %(default)s")
    group.add_argument(
        "--cluster-script-prefix-path",
        help="",
    )
    group.add_argument(
        "--cluster-max-retries",
        type=int,
        help="How many times to rerun failing jobs. Default: %(default)s",
        default=3)
