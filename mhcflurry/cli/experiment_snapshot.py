# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI for timestamped, plot-reconstructable experiment snapshots."""

import argparse

from mhcflurry.experiment_archive import (
    DEFAULT_MAX_COPY_BYTES,
    snapshot_experiment,
)


def make_parser(prog="mhcflurry train snapshot-experiment"):
    """Build the experiment-snapshot argument parser."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--collector-commit")
    parser.add_argument("--source-archive")
    parser.add_argument("--command-file", action="append", default=[])
    parser.add_argument("--input-file", action="append", default=[])
    parser.add_argument(
        "--max-copy-mb",
        type=float,
        default=DEFAULT_MAX_COPY_BYTES / (1024 * 1024),
        help=(
            "Maximum size of each copied artifact. Larger artifacts remain "
            "in the SHA256 inventory. Default: %(default)s"
        ),
    )
    return parser


def run_argv(argv=None, prog="mhcflurry train snapshot-experiment"):
    """Parse arguments, write a snapshot, and print its path."""
    args = make_parser(prog).parse_args(argv)
    if args.max_copy_mb < 0:
        raise ValueError("--max-copy-mb must be nonnegative")
    destination = snapshot_experiment(
        args.source_dir,
        args.experiments_dir,
        args.name,
        source_commit=args.source_commit,
        collector_commit=args.collector_commit,
        source_archive=args.source_archive,
        command_files=args.command_file,
        input_files=args.input_file,
        max_copy_bytes=int(args.max_copy_mb * 1024 * 1024),
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_argv())
