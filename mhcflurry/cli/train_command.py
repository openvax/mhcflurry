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

"""Release-training workflow commands.

``mhcflurry train`` is the semantic home for maintainer-level training
workflows that compose lower-level training CLIs. The first command is a
source-checkout wrapper around ``scripts/release/retrain_evaluate_deploy.sh``:
it keeps Brev/runplz orchestration, artifact sync, evaluation, plotting, and
optional deployment in one maintained implementation while giving users a
stable ``mhcflurry`` entry point.
"""

import argparse
from pathlib import Path
import subprocess
import sys


WORKFLOW_SCRIPT = Path("scripts/release/retrain_evaluate_deploy.sh")


def make_parser(prog="mhcflurry train"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Training workflows that compose lower-level MHCflurry training "
            "commands."
        ),
    )
    sub = parser.add_subparsers(dest="train_subcommand")
    pan = sub.add_parser(
        "pan-allele-release",
        add_help=False,
        help=(
            "Train pan-allele release weights, optionally evaluate, plot, "
            "sync remote artifacts, and deploy model archives."
        ),
        description=(
            "Delegate to scripts/release/retrain_evaluate_deploy.sh. "
            "Run 'mhcflurry train pan-allele-release --help' for the full "
            "workflow flags."
        ),
    )
    pan.add_argument(
        "workflow_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to scripts/release/retrain_evaluate_deploy.sh.",
    )
    holdout = sub.add_parser(
        "release-holdout",
        add_help=False,
        help="Build or validate the frozen release evaluation holdout.",
    )
    holdout.add_argument("holdout_args", nargs=argparse.REMAINDER)
    loss_curves = sub.add_parser(
        "plot-loss-curves",
        add_help=False,
        help="Plot training loss curves and highlight selected models.",
    )
    loss_curves.add_argument("plot_args", nargs=argparse.REMAINDER)
    snapshot = sub.add_parser(
        "snapshot-experiment",
        add_help=False,
        help="Archive reproducible experiment metadata and plotting tables.",
    )
    snapshot.add_argument("snapshot_args", nargs=argparse.REMAINDER)
    return parser


def _format_help(prog):
    return "\n".join([
        (
            "usage: %s {pan-allele-release,release-holdout,"
            "plot-loss-curves,snapshot-experiment} ..." % prog
        ),
        "",
        "Training workflows.",
        "",
        "Subcommands:",
        "  pan-allele-release  Train/evaluate/plot release weights from one entry point.",
        "  release-holdout     Build/validate frozen evaluation exclusions.",
        "  plot-loss-curves    Plot candidate losses and selected models.",
        "  snapshot-experiment Archive hashes, provenance, metrics, and epoch tables.",
        "",
        "Examples:",
        "  %s pan-allele-release --run-dir runs/2.3.0 --release 2.3.0 --backend local" % prog,
        "  %s pan-allele-release --run-dir runs/2.3.0 --release 2.3.0 --backend brev-provision" % prog,
        "  %s plot-loss-curves --selected-dir models.combined --out plots" % prog,
        "  %s snapshot-experiment --source-dir results/run --name batch-sweep" % prog,
        "",
        "Deployment is opt-in. Pass --deploy-mode dry-run, draft, or publish "
        "to run the model-artifact deployment step.",
    ])


def _workflow_script_path():
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / WORKFLOW_SCRIPT,
        Path.cwd() / WORKFLOW_SCRIPT,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Could not find %s. This command must be run from a source checkout "
        "or an editable install that includes the scripts/ directory." %
        WORKFLOW_SCRIPT
    )


def _print_pan_allele_release_help(script, prog):
    result = subprocess.run(
        ["bash", str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    help_text = result.stdout or result.stderr
    help_text = help_text.replace(str(WORKFLOW_SCRIPT), prog)
    print(help_text, end="" if help_text.endswith("\n") else "\n")
    return result.returncode


def _run_pan_allele_release(argv, prog):
    script = _workflow_script_path()
    if argv and argv[0] in ("-h", "--help"):
        return _print_pan_allele_release_help(script, prog)
    return subprocess.call(["bash", str(script)] + list(argv))


def run_argv(argv, prog="mhcflurry train"):
    if not argv or argv[0] in ("-h", "--help"):
        print(_format_help(prog))
        return 0
    if argv[0] == "pan-allele-release":
        return _run_pan_allele_release(
            argv[1:], "%s pan-allele-release" % prog)
    if argv[0] == "release-holdout":
        from mhcflurry import release_holdout
        return release_holdout.run_argv(
            argv[1:], prog="%s release-holdout" % prog)
    if argv[0] == "plot-loss-curves":
        from . import plot_loss_curves
        return plot_loss_curves.run_argv(
            argv[1:], prog="%s plot-loss-curves" % prog)
    if argv[0] == "snapshot-experiment":
        from . import experiment_snapshot
        return experiment_snapshot.run_argv(
            argv[1:], prog="%s snapshot-experiment" % prog)
    make_parser(prog).parse_args(argv)
    return 2


parser = make_parser()


if __name__ == "__main__":
    sys.exit(run_argv(sys.argv[1:]) or 0)
