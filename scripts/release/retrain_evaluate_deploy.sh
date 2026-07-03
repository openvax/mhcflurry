#!/usr/bin/env bash
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

set -euo pipefail

usage() {
    cat <<'EOF'
Run the release retraining workflow from one maintained entry point.

Usage:
  scripts/release/retrain_evaluate_deploy.sh \
      --run-dir /path/to/release-run \
      --release 2.3.0 \
      [--backend local|brev-existing|ssh] \
      [--minibatch-size 1024] \
      [--affinity-minibatch-size 1024] \
      [--processing-minibatch-size 1024] \
      [--processing-variants "with_flanks no_flank short_flanks"] \
      [--skip-train] [--skip-eval] [--skip-plots] [--skip-deploy] \
      [--deploy-mode dry-run|draft|publish]

Backends:
  local          Run scripts/training/pan_allele_release_full.sh here.
  brev-existing  Run on existing Brev/runplz capacity. This does not provision
                 a new machine; runplz/Brev handles the remote container,
                 package sync, and credentials.
  ssh            Run on a specific remote host, then rsync the run directory
                 back. Requires --remote, --remote-repo, and --remote-run-dir.
                 Authentication is whatever your local ssh/rsync configuration
                 uses, typically SSH keys or an SSH config Host entry.

Evaluation:
  After training, the script runs:
      mhcflurry compare-models --a RUN_DIR --b public
      mhcflurry plot-model-comparison --input RUN_DIR/eval_comparison
  compare-models writes release_summary.csv and release_summary.md with
  affinity, processing, and presentation release-gate tables.

Deployment:
  The final step calls deploy_trained_models.sh. The default deploy mode is
  dry-run, so the script validates and prints release assets without uploading.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

note() {
    echo "$*" >&2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

run_cmd() {
    printf '+'
    local arg
    for arg in "$@"; do
        printf ' %q' "$arg"
    done
    printf '\n'
    if [ "$DRY_RUN" != "1" ]; then
        "$@"
    fi
}

RUN_DIR=
RELEASE=
GITHUB_RELEASE=
BACKEND=local
REMOTE=
REMOTE_REPO=
REMOTE_RUN_DIR=
SYNC_REMOTE_OUTPUT=1
SKIP_TRAIN=0
SKIP_EVAL=0
SKIP_PLOTS=0
SKIP_DEPLOY=0
DEPLOY_MODE=dry-run
DATA_DIR=
COMPARE_INCLUDE=affinity,processing,presentation
PROCESSING_MODES=with_flanks,no_flank,short_flanks
PRESENTATION_MODES=with_flanks,without_flanks
RUN_LABEL=new
DRY_RUN=0
TRAINING_MINIBATCH_SIZE=1024
AFFINITY_MINIBATCH_SIZE=
PROCESSING_MINIBATCH_SIZE=
PROCESSING_VARIANTS="with_flanks no_flank short_flanks"
PRESENTATION_PROCESSING_WITH_FLANKS_KIND=with_flanks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

while [ $# -gt 0 ]; do
    case "$1" in
        --run-dir)
            RUN_DIR=$2
            shift 2
            ;;
        --release)
            RELEASE=$2
            shift 2
            ;;
        --github-release)
            GITHUB_RELEASE=$2
            shift 2
            ;;
        --backend)
            BACKEND=$2
            shift 2
            ;;
        --remote)
            REMOTE=$2
            shift 2
            ;;
        --remote-repo)
            REMOTE_REPO=$2
            shift 2
            ;;
        --remote-run-dir)
            REMOTE_RUN_DIR=$2
            shift 2
            ;;
        --no-sync-remote-output)
            SYNC_REMOTE_OUTPUT=0
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=1
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=1
            shift
            ;;
        --skip-plots)
            SKIP_PLOTS=1
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=1
            shift
            ;;
        --deploy-mode)
            DEPLOY_MODE=$2
            shift 2
            ;;
        --data-dir)
            DATA_DIR=$2
            shift 2
            ;;
        --compare-include)
            COMPARE_INCLUDE=$2
            shift 2
            ;;
        --presentation-modes)
            PRESENTATION_MODES=$2
            shift 2
            ;;
        --run-label)
            RUN_LABEL=$2
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --minibatch-size)
            TRAINING_MINIBATCH_SIZE=$2
            shift 2
            ;;
        --affinity-minibatch-size)
            AFFINITY_MINIBATCH_SIZE=$2
            shift 2
            ;;
        --processing-minibatch-size)
            PROCESSING_MINIBATCH_SIZE=$2
            shift 2
            ;;
        --processing-variants)
            PROCESSING_VARIANTS=$2
            shift 2
            ;;
        --presentation-processing-with-flanks-kind)
            PRESENTATION_PROCESSING_WITH_FLANKS_KIND=$2
            shift 2
            ;;
        --processing-modes)
            PROCESSING_MODES=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

[ -n "$RUN_DIR" ] || die "--run-dir is required"
[ -n "$RELEASE" ] || die "--release is required"
if [ -z "$GITHUB_RELEASE" ]; then
    GITHUB_RELEASE=$RELEASE
fi
if [ -z "$AFFINITY_MINIBATCH_SIZE" ]; then
    AFFINITY_MINIBATCH_SIZE=$TRAINING_MINIBATCH_SIZE
fi
if [ -z "$PROCESSING_MINIBATCH_SIZE" ]; then
    PROCESSING_MINIBATCH_SIZE=$TRAINING_MINIBATCH_SIZE
fi
case "$BACKEND" in
    local|brev-existing|ssh) ;;
    *) die "--backend must be one of: local, brev-existing, ssh" ;;
esac
case "$DEPLOY_MODE" in
    dry-run|draft|publish) ;;
    *) die "--deploy-mode must be one of: dry-run, draft, publish" ;;
esac

case "$RUN_DIR" in
    /*) ;;
    *) RUN_DIR="$(pwd)/$RUN_DIR" ;;
esac
RUN_DIR=${RUN_DIR%/}

note "Run directory: $RUN_DIR"
note "Release:       $RELEASE"
note "Backend:       $BACKEND"
note "Batch sizes:   affinity=$AFFINITY_MINIBATCH_SIZE processing=$PROCESSING_MINIBATCH_SIZE"
note "Processing:    variants=$PROCESSING_VARIANTS; eval_modes=$PROCESSING_MODES"

if [ "$SKIP_TRAIN" != "1" ]; then
    case "$BACKEND" in
        local)
            run_cmd mkdir -p "$RUN_DIR"
            run_cmd env \
                "MHCFLURRY_OUT=$RUN_DIR" \
                "REPO=$REPO" \
                "TRAINING_MINIBATCH_SIZE=$TRAINING_MINIBATCH_SIZE" \
                "AFFINITY_MINIBATCH_SIZE=$AFFINITY_MINIBATCH_SIZE" \
                "PROCESSING_MINIBATCH_SIZE=$PROCESSING_MINIBATCH_SIZE" \
                "PROCESSING_VARIANTS=$PROCESSING_VARIANTS" \
                "PRESENTATION_PROCESSING_WITH_FLANKS_KIND=$PRESENTATION_PROCESSING_WITH_FLANKS_KIND" \
                bash "$REPO/scripts/training/pan_allele_release_full.sh"
            ;;
        brev-existing)
            require_command runplz
            run_cmd mkdir -p "$RUN_DIR"
            run_cmd env \
                "MHCFLURRY_OUT=$RUN_DIR" \
                "REPO=$REPO" \
                "TRAINING_MINIBATCH_SIZE=$TRAINING_MINIBATCH_SIZE" \
                "AFFINITY_MINIBATCH_SIZE=$AFFINITY_MINIBATCH_SIZE" \
                "PROCESSING_MINIBATCH_SIZE=$PROCESSING_MINIBATCH_SIZE" \
                "PROCESSING_VARIANTS=$PROCESSING_VARIANTS" \
                "PRESENTATION_PROCESSING_WITH_FLANKS_KIND=$PRESENTATION_PROCESSING_WITH_FLANKS_KIND" \
                runplz --outputs-dir "$RUN_DIR" \
                "$REPO/scripts/training/launch_pan_allele_training_remote.py"
            ;;
        ssh)
            require_command ssh
            [ -n "$REMOTE" ] || die "--remote is required for --backend ssh"
            [ -n "$REMOTE_REPO" ] || die "--remote-repo is required for --backend ssh"
            [ -n "$REMOTE_RUN_DIR" ] || \
                die "--remote-run-dir is required for --backend ssh"
            REMOTE_COMMAND="cd '$REMOTE_REPO'"
            REMOTE_COMMAND="$REMOTE_COMMAND && MHCFLURRY_OUT='$REMOTE_RUN_DIR'"
            REMOTE_COMMAND="$REMOTE_COMMAND REPO='$REMOTE_REPO'"
            REMOTE_COMMAND="$REMOTE_COMMAND TRAINING_MINIBATCH_SIZE='$TRAINING_MINIBATCH_SIZE'"
            REMOTE_COMMAND="$REMOTE_COMMAND AFFINITY_MINIBATCH_SIZE='$AFFINITY_MINIBATCH_SIZE'"
            REMOTE_COMMAND="$REMOTE_COMMAND PROCESSING_MINIBATCH_SIZE='$PROCESSING_MINIBATCH_SIZE'"
            REMOTE_COMMAND="$REMOTE_COMMAND PROCESSING_VARIANTS='$PROCESSING_VARIANTS'"
            REMOTE_COMMAND="$REMOTE_COMMAND PRESENTATION_PROCESSING_WITH_FLANKS_KIND='$PRESENTATION_PROCESSING_WITH_FLANKS_KIND'"
            REMOTE_COMMAND="$REMOTE_COMMAND bash"
            REMOTE_COMMAND="$REMOTE_COMMAND scripts/training/pan_allele_release_full.sh"
            run_cmd ssh "$REMOTE" \
                "$REMOTE_COMMAND"
            if [ "$SYNC_REMOTE_OUTPUT" = "1" ]; then
                require_command rsync
                run_cmd mkdir -p "$RUN_DIR"
                run_cmd rsync -a "$REMOTE:$REMOTE_RUN_DIR/" "$RUN_DIR/"
            fi
            ;;
    esac
else
    note "Skipping training."
fi

if [ "$SKIP_EVAL" != "1" ]; then
    EVAL_OUT="$RUN_DIR/eval_comparison"
    run_cmd mkdir -p "$EVAL_OUT"
    if [ -z "$DATA_DIR" ]; then
        require_command mhcflurry-downloads
        run_cmd mhcflurry-downloads fetch data_evaluation models_class1_pan \
            models_class1_processing models_class1_presentation
        if [ "$DRY_RUN" = "1" ]; then
            DATA_DIR='<mhcflurry-downloads path data_evaluation>'
        else
            DATA_DIR="$(mhcflurry-downloads path data_evaluation)"
        fi
    fi
    run_cmd mhcflurry compare-models \
        --a "$RUN_DIR" \
        --a-label "$RUN_LABEL" \
        --b public \
        --data-dir "$DATA_DIR" \
        --include "$COMPARE_INCLUDE" \
        --processing-modes "$PROCESSING_MODES" \
        --presentation-modes "$PRESENTATION_MODES" \
        --out "$EVAL_OUT"
else
    note "Skipping evaluation."
fi

if [ "$SKIP_PLOTS" != "1" ]; then
    run_cmd mhcflurry plot-model-comparison --input "$RUN_DIR/eval_comparison"
else
    note "Skipping plots."
fi

if [ "$SKIP_DEPLOY" != "1" ]; then
    run_cmd "$SCRIPT_DIR/deploy_trained_models.sh" \
        --run-dir "$RUN_DIR" \
        --release "$RELEASE" \
        --github-release "$GITHUB_RELEASE" \
        --mode "$DEPLOY_MODE"
else
    note "Skipping deploy step."
fi
