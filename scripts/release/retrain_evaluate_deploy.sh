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
      [--backend local|brev-existing|brev-provision|ssh] \
      [--minibatch-size 1024] \
      [--affinity-minibatch-size 1024] \
      [--affinity-max-workers-per-gpu auto] \
      [--processing-minibatch-size 1024] \
      [--processing-variants "with_flanks no_flank short_flanks"] \
      [--brev-instance NAME] [--brev-on-finish leave|stop|delete] \
      [--brev-stop-failure-action warn|delete] \
      [--brev-sync-mode release|full] \
      [--brev-instance-type TYPE] \
      [--skip-train] [--skip-eval] [--skip-plots] [--skip-deploy] \
      [--deploy-mode dry-run|draft|publish]

Backends:
  local          Run scripts/training/pan_allele_release_full.sh here.
  brev-existing  Run on a named existing Brev instance. Requires
                 --brev-instance. A missing instance is an error.
  brev-provision Provision a named Brev instance if it does not exist, then run
                 the same remote training job. If --brev-instance is omitted,
                 this script generates a run-specific name. Defaults to the
                 4xA100 instance type used for the release pipeline; override
                 with --brev-instance-type. The release wrapper owns artifact
                 sync and cleanup; runplz is asked to leave the instance up
                 until those steps finish.
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

Logs:
  The wrapper writes per-step logs and a status table under:
      RUN_DIR/workflow_logs/
  Brev sync defaults to release mode: final selected model directories plus
  runplz events, training/eval logs, GPU telemetry, and generated configs.
  Use --brev-sync-mode full only for full post-mortem copies of all candidate
  pools and intermediate CSVs.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

note() {
    echo "$*" >&2
}

warn() {
    echo "WARNING: $*" >&2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

format_command() {
    local arg
    for arg in "$@"; do
        printf ' %q' "$arg"
    done
}

workflow_timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

record_workflow_event() {
    local step="$1"
    local status="$2"
    shift 2 || true
    if [ "${DRY_RUN:-0}" = "1" ]; then
        return 0
    fi
    if [ -z "${WORKFLOW_STATUS_LOG:-}" ]; then
        return 0
    fi
    printf '%s\t%s\t%s\t%s\n' \
        "$(workflow_timestamp)" \
        "$step" \
        "$status" \
        "$*" >> "$WORKFLOW_STATUS_LOG"
}

run_cmd() {
    printf '+'
    format_command "$@"
    printf '\n'
    if [ "$DRY_RUN" != "1" ]; then
        "$@"
    fi
}

run_logged_step() {
    local step="$1"
    local log_file="$WORKFLOW_LOG_DIR/$step.log"
    shift

    run_cmd mkdir -p "$WORKFLOW_LOG_DIR"
    record_workflow_event "$step" start \
        "log=$log_file command=$(format_command "$@")"
    if [ "$DRY_RUN" = "1" ]; then
        run_cmd "$@"
        record_workflow_event "$step" 0 "dry-run log=$log_file"
        return 0
    fi

    {
        printf '[%s] start step=%s command=' "$(workflow_timestamp)" "$step"
        format_command "$@"
        printf '\n'
    } | tee -a "$log_file" >&2

    set +e
    (
        set -o pipefail
        "$@" 2>&1 | tee -a "$log_file"
    )
    local status=$?
    set -e

    {
        printf '[%s] end step=%s status=%s\n' \
            "$(workflow_timestamp)" "$step" "$status"
    } | tee -a "$log_file" >&2
    record_workflow_event "$step" "$status" "log=$log_file"
    return "$status"
}

run_dir_has_model_artifacts() {
    [ -d "$RUN_DIR/affinity/models.combined" ] && \
        [ -d "$RUN_DIR/processing/models.selected.no_flank" ] && \
        [ -d "$RUN_DIR/presentation/models" ]
}

run_dir_has_synced_brev_outputs() {
    run_dir_has_model_artifacts || return 1
    if [ "${BREV_EXPECT_REMOTE_EVAL:-0}" = "1" ] && \
            [ ! -d "$RUN_DIR/eval_comparison" ]; then
        return 1
    fi
    if [ "${BREV_EXPECT_REMOTE_PLOTS:-0}" = "1" ] && \
            [ ! -d "$RUN_DIR/eval_comparison/plots" ]; then
        return 1
    fi
    return 0
}

brev_latest_remote_exit_code() {
    require_command brev
    local output
    output="$(
        brev exec "$BREV_INSTANCE" \
            "bash -lc \"grep 'remote_command_exit' ~/runplz-latest/out/.runplz/events.ndjson 2>/dev/null | tail -1\"" \
            2>/dev/null || true
    )"
    printf '%s\n' "$output" | sed -n \
        's/.*"exit_code"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
        | tail -1
}

brev_instance_status() {
    require_command brev
    BREV_INSTANCE_NAME="$BREV_INSTANCE" brev ls --json | python -c '
import json
import os
import sys

name = os.environ["BREV_INSTANCE_NAME"]
for item in json.load(sys.stdin):
    if item.get("name") == name:
        print(item.get("status", ""))
        break
'
}

sync_brev_output() {
    if [ "$SYNC_REMOTE_OUTPUT" != "1" ]; then
        note "Skipping Brev output sync because --no-sync-remote-output was set."
        return 0
    fi
    if run_dir_has_synced_brev_outputs; then
        note "Brev output already present locally; skipping explicit copy."
        return 0
    fi

    require_command brev
    require_command rsync
    require_command tar
    run_cmd mkdir -p "$RUN_DIR"
    local sync_parent="$RUN_DIR/.brev-sync"
    local copied_out="$sync_parent/out"
    run_cmd rm -rf "$sync_parent"
    run_cmd mkdir -p "$sync_parent"
    if [ "$BREV_SYNC_MODE" = "release" ]; then
        sync_brev_release_output "$sync_parent"
    else
        sync_brev_full_output "$sync_parent"
    fi
    run_dir_has_model_artifacts || \
        die "Brev sync finished but expected model artifacts are missing in $RUN_DIR"
}

sync_brev_full_output() {
    local sync_parent="$1"
    local copied_out="$sync_parent/out"
    set +e
    run_logged_step brev_sync_copy \
        brev copy "$BREV_INSTANCE:/root/runplz-latest/out" "$sync_parent/"
    local copy_status=$?
    set -e
    if [ "$copy_status" -ne 0 ]; then
        return "$copy_status"
    fi
    set +e
    if [ -d "$copied_out" ]; then
        run_logged_step brev_sync_merge rsync -a "$copied_out/" "$RUN_DIR/"
    else
        run_logged_step brev_sync_merge rsync -a "$sync_parent/" "$RUN_DIR/"
    fi
    local merge_status=$?
    set -e
    if [ "$merge_status" -ne 0 ]; then
        return "$merge_status"
    fi
    run_cmd rm -rf "$sync_parent"
}

sync_brev_release_output() {
    local sync_parent="$1"
    local remote_archive=/root/runplz-latest/out/.runplz/release_sync.tar.bz2
    local local_archive="$sync_parent/release_sync.tar.bz2"
    local sync_script="$sync_parent/build_release_sync_archive.sh"

    cat > "$sync_script" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd ~/runplz-latest/out
mkdir -p .runplz
manifest=.runplz/release_sync_paths.txt
archive=.runplz/release_sync.tar.bz2
rm -f "$manifest" "$archive"

add_path() {
    if [ -e "$1" ]; then
        printf '%s\n' "$1" >> "$manifest"
    else
        printf 'missing optional sync path: %s\n' "$1" >&2
    fi
}

add_glob() {
    local path
    for path in "$@"; do
        [ -e "$path" ] && printf '%s\n' "$path" >> "$manifest"
    done
}

add_path .runplz/events.ndjson
add_path .runplz/heartbeat.ndjson
add_path .runplz/last.log
add_path .runplz/run.json
add_path .runplz/run.sh
add_path .runplz/run_driver.log

add_path eval_comparison

add_path affinity/models.combined
add_path affinity/eval_comparison
add_path affinity/loss_plots
add_path affinity/calibrate.log
add_path affinity/eval.log
add_path affinity/fetch_eval_data.log
add_path affinity/gpu_occupancy.csv
add_path affinity/hyperparameters.yaml
add_path affinity/plot_loss_curves.log
add_path affinity/release_driver.log
add_path affinity/release_heartbeat.log
add_path affinity/select.log
add_path affinity/train.log
add_glob affinity/LOG-worker.*.txt

add_path processing/models.selected.with_flanks
add_path processing/models.selected.no_flank
add_path processing/models.selected.short_flanks
add_path processing/hits_with_tpm.csv.bz2
add_path processing/hyperparameters.base.yaml
add_path processing/hyperparameters.with_flanks.yaml
add_path processing/hyperparameters.no_flank.yaml
add_path processing/hyperparameters.short_flanks.yaml
add_path processing/train_data.csv.bz2
add_glob processing/LOG-worker.*.txt

add_path presentation/models
add_path presentation/make_train_data.presentation.py

sort -u "$manifest" -o "$manifest"
tar -cjf "$archive" -T "$manifest"
printf 'release sync manifest:\n'
cat "$manifest"
du -sh "$archive"
EOF
    chmod +x "$sync_script"

    set +e
    run_logged_step brev_sync_prepare_release_archive \
        brev exec "$BREV_INSTANCE" "@$sync_script"
    local prepare_status=$?
    set -e
    if [ "$prepare_status" -ne 0 ]; then
        return "$prepare_status"
    fi

    set +e
    run_logged_step brev_sync_copy_release_archive \
        brev copy "$BREV_INSTANCE:$remote_archive" "$sync_parent/"
    local copy_status=$?
    set -e
    if [ "$copy_status" -ne 0 ]; then
        return "$copy_status"
    fi

    if [ ! -f "$local_archive" ]; then
        die "Brev release sync archive was not copied to $local_archive"
    fi
    run_logged_step brev_sync_extract_release_archive \
        tar -C "$RUN_DIR" -xjf "$local_archive"
    run_cmd rm -rf "$sync_parent"
}

apply_brev_cleanup() {
    case "$BACKEND" in
        brev-existing|brev-provision) ;;
        *) return 0 ;;
    esac
    case "$BREV_ON_FINISH" in
        leave)
            note "Leaving Brev instance running: $BREV_INSTANCE"
            return 0
            ;;
        stop)
            set +e
            run_logged_step brev_stop brev stop "$BREV_INSTANCE"
            local stop_status=$?
            set -e
            if [ "$stop_status" -ne 0 ]; then
                warn "brev stop failed for $BREV_INSTANCE with status $stop_status"
            fi
            sleep 15
            local status
            status="$(brev_instance_status || true)"
            if [ "$status" = "RUNNING" ]; then
                warn "Brev instance is still RUNNING after stop: $BREV_INSTANCE"
                if [ "$BREV_STOP_FAILURE_ACTION" = "delete" ]; then
                    warn "Deleting provisioned instance because stop did not take effect."
                    run_logged_step brev_delete_after_failed_stop \
                        brev delete "$BREV_INSTANCE"
                else
                    warn "Leaving instance running; rerun 'brev stop $BREV_INSTANCE' or delete it manually."
                fi
            else
                note "Brev instance status after stop: ${status:-unknown}"
            fi
            ;;
        delete)
            run_logged_step brev_delete brev delete "$BREV_INSTANCE"
            ;;
    esac
}

run_brev_training() {
    local auto_create=$1
    require_command runplz
    run_cmd mkdir -p "$RUN_DIR"
    local runplz_on_finish=leave
    local run_release_eval=0
    local run_release_plots=0
    if [ "$SKIP_EVAL" != "1" ]; then
        run_release_eval=1
    fi
    if [ "$SKIP_PLOTS" != "1" ]; then
        run_release_plots=1
    fi
    BREV_EXPECT_REMOTE_EVAL=$run_release_eval
    BREV_EXPECT_REMOTE_PLOTS=$run_release_plots
    local runplz_env=(
        "MHCFLURRY_OUT=$RUN_DIR"
        "REPO=$REPO"
        "TRAINING_MINIBATCH_SIZE=$TRAINING_MINIBATCH_SIZE"
        "AFFINITY_MINIBATCH_SIZE=$AFFINITY_MINIBATCH_SIZE"
        "AFFINITY_MAX_WORKERS_PER_GPU=$AFFINITY_MAX_WORKERS_PER_GPU"
        "PROCESSING_MINIBATCH_SIZE=$PROCESSING_MINIBATCH_SIZE"
        "PROCESSING_VARIANTS=$PROCESSING_VARIANTS"
        "PRESENTATION_PROCESSING_WITH_FLANKS_KIND=$PRESENTATION_PROCESSING_WITH_FLANKS_KIND"
        "RUN_RELEASE_EVAL=$run_release_eval"
        "RUN_RELEASE_PLOTS=$run_release_plots"
        "COMPARE_INCLUDE=$COMPARE_INCLUDE"
        "PROCESSING_MODES=$PROCESSING_MODES"
        "PRESENTATION_MODES=$PRESENTATION_MODES"
        "RUN_LABEL=$RUN_LABEL"
        "RUNPLZ_BREV_AUTO_CREATE=$auto_create"
        "RUNPLZ_BREV_ON_FINISH=$runplz_on_finish"
        "RUNPLZ_BREV_INSTANCE_TYPE_FALLBACK_COUNT=$BREV_INSTANCE_TYPE_FALLBACK_COUNT"
        "RUNPLZ_BREV_EXCLUDE_PROVIDERS=$BREV_EXCLUDE_PROVIDERS"
    )
    if [ -n "$BREV_INSTANCE_TYPE" ]; then
        runplz_env+=("RUNPLZ_BREV_INSTANCE_TYPE=$BREV_INSTANCE_TYPE")
    fi
    if [ -n "$BREV_MAX_RUNTIME_SECONDS" ]; then
        runplz_env+=("RUNPLZ_BREV_MAX_RUNTIME_SECONDS=$BREV_MAX_RUNTIME_SECONDS")
    fi
    set +e
    run_logged_step train_brev env \
        "${runplz_env[@]}" \
        runplz brev --outputs-dir "$RUN_DIR" \
        --log-file "$RUN_DIR/runplz-driver.log" \
        --instance "$BREV_INSTANCE" \
        "$REPO/scripts/training/launch_pan_allele_training_remote.py"
    local runplz_status=$?
    set -e

    local remote_exit
    remote_exit="$(brev_latest_remote_exit_code || true)"
    if [ "$runplz_status" -ne 0 ]; then
        if [ "$remote_exit" = "0" ]; then
            warn "runplz exited with $runplz_status, but remote command exit_code=0; continuing after explicit sync."
        else
            warn "runplz exited with $runplz_status; remote exit_code=${remote_exit:-unknown}."
            sync_brev_output || true
            apply_brev_cleanup
            return "$runplz_status"
        fi
    fi

    sync_brev_output || {
        warn "Brev output sync failed; leaving $BREV_INSTANCE available to preserve artifacts."
        return 1
    }
    if [ "$run_release_eval" = "1" ] && [ -d "$RUN_DIR/eval_comparison" ]; then
        BREV_REMOTE_EVAL_DONE=1
    fi
    if [ "$run_release_plots" = "1" ] && \
            [ -d "$RUN_DIR/eval_comparison/plots" ]; then
        BREV_REMOTE_PLOTS_DONE=1
    fi
    apply_brev_cleanup
}

RUN_DIR=
RELEASE=
GITHUB_RELEASE=
BACKEND=local
REMOTE=
REMOTE_REPO=
REMOTE_RUN_DIR=
SYNC_REMOTE_OUTPUT=1
BREV_INSTANCE="${RUNPLZ_BREV_INSTANCE:-${BREV_INSTANCE:-}}"
BREV_ON_FINISH="${RUNPLZ_BREV_ON_FINISH:-${BREV_ON_FINISH:-}}"
BREV_INSTANCE_TYPE="${RUNPLZ_BREV_INSTANCE_TYPE:-${BREV_INSTANCE_TYPE:-}}"
DEFAULT_BREV_PROVISION_INSTANCE_TYPE="${DEFAULT_BREV_PROVISION_INSTANCE_TYPE:-a2-highgpu-4g:nvidia-tesla-a100:4}"
BREV_MAX_RUNTIME_SECONDS="${RUNPLZ_BREV_MAX_RUNTIME_SECONDS:-${BREV_MAX_RUNTIME_SECONDS:-}}"
BREV_INSTANCE_TYPE_FALLBACK_COUNT="${RUNPLZ_BREV_INSTANCE_TYPE_FALLBACK_COUNT:-3}"
BREV_EXCLUDE_PROVIDERS="${RUNPLZ_BREV_EXCLUDE_PROVIDERS:-oci}"
BREV_STOP_FAILURE_ACTION="${BREV_STOP_FAILURE_ACTION:-}"
BREV_SYNC_MODE="${BREV_SYNC_MODE:-release}"
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
AFFINITY_MAX_WORKERS_PER_GPU="${AFFINITY_MAX_WORKERS_PER_GPU:-auto}"
PROCESSING_MINIBATCH_SIZE=
PROCESSING_VARIANTS="with_flanks no_flank short_flanks"
PRESENTATION_PROCESSING_WITH_FLANKS_KIND=with_flanks
WORKFLOW_LOG_DIR=
WORKFLOW_STATUS_LOG=
BREV_EXPECT_REMOTE_EVAL=0
BREV_EXPECT_REMOTE_PLOTS=0
BREV_REMOTE_EVAL_DONE=0
BREV_REMOTE_PLOTS_DONE=0

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
        --brev-instance)
            BREV_INSTANCE=$2
            shift 2
            ;;
        --brev-on-finish)
            BREV_ON_FINISH=$2
            shift 2
            ;;
        --brev-instance-type)
            BREV_INSTANCE_TYPE=$2
            shift 2
            ;;
        --brev-stop-failure-action)
            BREV_STOP_FAILURE_ACTION=$2
            shift 2
            ;;
        --brev-sync-mode)
            BREV_SYNC_MODE=$2
            shift 2
            ;;
        --brev-max-runtime-seconds)
            BREV_MAX_RUNTIME_SECONDS=$2
            shift 2
            ;;
        --brev-instance-type-fallback-count)
            BREV_INSTANCE_TYPE_FALLBACK_COUNT=$2
            shift 2
            ;;
        --brev-exclude-providers)
            BREV_EXCLUDE_PROVIDERS=$2
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
        --affinity-max-workers-per-gpu)
            AFFINITY_MAX_WORKERS_PER_GPU=$2
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
    local|brev-existing|brev-provision|ssh) ;;
    *) die "--backend must be one of: local, brev-existing, brev-provision, ssh" ;;
esac
case "$DEPLOY_MODE" in
    dry-run|draft|publish) ;;
    *) die "--deploy-mode must be one of: dry-run, draft, publish" ;;
esac
if [ -z "$BREV_ON_FINISH" ]; then
    case "$BACKEND" in
        brev-provision)
            BREV_ON_FINISH=stop
            ;;
        *)
            BREV_ON_FINISH=leave
            ;;
    esac
fi
case "$BREV_ON_FINISH" in
    leave|stop|delete) ;;
    *) die "--brev-on-finish must be one of: leave, stop, delete" ;;
esac
if [ -z "$BREV_STOP_FAILURE_ACTION" ]; then
    case "$BACKEND" in
        brev-provision)
            BREV_STOP_FAILURE_ACTION=delete
            ;;
        *)
            BREV_STOP_FAILURE_ACTION=warn
            ;;
    esac
fi
case "$BREV_STOP_FAILURE_ACTION" in
    warn|delete) ;;
    *) die "--brev-stop-failure-action must be one of: warn, delete" ;;
esac
case "$BREV_SYNC_MODE" in
    release|full) ;;
    *) die "--brev-sync-mode must be one of: release, full" ;;
esac
if [ "$BACKEND" = "brev-provision" ] && [ -z "$BREV_INSTANCE" ]; then
    RELEASE_SLUG=$(
        printf '%s' "$RELEASE" | tr -cs 'A-Za-z0-9' '-' | sed 's/^-//; s/-$//'
    )
    BREV_INSTANCE="mhcflurry-${RELEASE_SLUG}-$(date +%Y%m%d-%H%M%S)"
fi
if [ "$BACKEND" = "brev-provision" ] && [ -z "$BREV_INSTANCE_TYPE" ]; then
    BREV_INSTANCE_TYPE=$DEFAULT_BREV_PROVISION_INSTANCE_TYPE
fi
case "$BACKEND" in
    brev-existing)
        [ -n "$BREV_INSTANCE" ] || \
            die "--brev-instance is required for --backend brev-existing"
        ;;
    brev-provision)
        [ -n "$BREV_INSTANCE" ] || \
            die "could not derive a Brev instance name for --backend brev-provision"
        ;;
esac

case "$RUN_DIR" in
    /*) ;;
    *) RUN_DIR="$(pwd)/$RUN_DIR" ;;
esac
RUN_DIR=${RUN_DIR%/}
WORKFLOW_LOG_DIR="$RUN_DIR/workflow_logs"
WORKFLOW_STATUS_LOG="$WORKFLOW_LOG_DIR/status.tsv"
run_cmd mkdir -p "$WORKFLOW_LOG_DIR"
if [ "$DRY_RUN" != "1" ]; then
    printf 'timestamp\tstep\tstatus\tdetails\n' > "$WORKFLOW_STATUS_LOG"
fi

note "Run directory: $RUN_DIR"
note "Release:       $RELEASE"
note "Backend:       $BACKEND"
note "Batch sizes:   affinity=$AFFINITY_MINIBATCH_SIZE processing=$PROCESSING_MINIBATCH_SIZE"
note "Affinity MWPG: $AFFINITY_MAX_WORKERS_PER_GPU"
note "Processing:    variants=$PROCESSING_VARIANTS; eval_modes=$PROCESSING_MODES"
case "$BACKEND" in
    brev-existing|brev-provision)
        note "Brev instance: $BREV_INSTANCE"
        note "Brev cleanup:  $BREV_ON_FINISH"
        note "Stop fallback:  $BREV_STOP_FAILURE_ACTION"
        note "Brev sync:     $BREV_SYNC_MODE"
        note "Brev type:     ${BREV_INSTANCE_TYPE:-runplz auto-select}"
        ;;
esac

if [ "$SKIP_TRAIN" != "1" ]; then
    case "$BACKEND" in
        local)
            run_cmd mkdir -p "$RUN_DIR"
            run_logged_step train_local env \
                "MHCFLURRY_OUT=$RUN_DIR" \
                "REPO=$REPO" \
                "TRAINING_MINIBATCH_SIZE=$TRAINING_MINIBATCH_SIZE" \
                "AFFINITY_MINIBATCH_SIZE=$AFFINITY_MINIBATCH_SIZE" \
                "AFFINITY_MAX_WORKERS_PER_GPU=$AFFINITY_MAX_WORKERS_PER_GPU" \
                "PROCESSING_MINIBATCH_SIZE=$PROCESSING_MINIBATCH_SIZE" \
                "PROCESSING_VARIANTS=$PROCESSING_VARIANTS" \
                "PRESENTATION_PROCESSING_WITH_FLANKS_KIND=$PRESENTATION_PROCESSING_WITH_FLANKS_KIND" \
                bash "$REPO/scripts/training/pan_allele_release_full.sh"
            ;;
        brev-existing)
            run_brev_training 0
            ;;
        brev-provision)
            run_brev_training 1
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
            REMOTE_COMMAND="$REMOTE_COMMAND AFFINITY_MAX_WORKERS_PER_GPU='$AFFINITY_MAX_WORKERS_PER_GPU'"
            REMOTE_COMMAND="$REMOTE_COMMAND PROCESSING_MINIBATCH_SIZE='$PROCESSING_MINIBATCH_SIZE'"
            REMOTE_COMMAND="$REMOTE_COMMAND PROCESSING_VARIANTS='$PROCESSING_VARIANTS'"
            REMOTE_COMMAND="$REMOTE_COMMAND PRESENTATION_PROCESSING_WITH_FLANKS_KIND='$PRESENTATION_PROCESSING_WITH_FLANKS_KIND'"
            REMOTE_COMMAND="$REMOTE_COMMAND bash"
            REMOTE_COMMAND="$REMOTE_COMMAND scripts/training/pan_allele_release_full.sh"
            run_logged_step train_ssh ssh "$REMOTE" \
                "$REMOTE_COMMAND"
            if [ "$SYNC_REMOTE_OUTPUT" = "1" ]; then
                require_command rsync
                run_cmd mkdir -p "$RUN_DIR"
                run_logged_step ssh_sync_output \
                    rsync -a "$REMOTE:$REMOTE_RUN_DIR/" "$RUN_DIR/"
            fi
            ;;
    esac
else
    note "Skipping training."
fi

if [ "$SKIP_EVAL" != "1" ]; then
    if [ "$BREV_REMOTE_EVAL_DONE" = "1" ]; then
        note "Using evaluation produced on the Brev instance."
    else
        EVAL_OUT="$RUN_DIR/eval_comparison"
        run_cmd mkdir -p "$EVAL_OUT"
        if [ -z "$DATA_DIR" ]; then
            require_command mhcflurry-downloads
            run_logged_step fetch_eval_downloads \
                mhcflurry-downloads fetch data_evaluation models_class1_pan \
                models_class1_processing models_class1_presentation
            if [ "$DRY_RUN" = "1" ]; then
                DATA_DIR='<mhcflurry-downloads path data_evaluation>'
            else
                DATA_DIR="$(mhcflurry-downloads path data_evaluation)"
            fi
        fi
        run_logged_step compare_models mhcflurry compare-models \
            --a "$RUN_DIR" \
            --a-label "$RUN_LABEL" \
            --b public \
            --data-dir "$DATA_DIR" \
            --include "$COMPARE_INCLUDE" \
            --processing-modes "$PROCESSING_MODES" \
            --presentation-modes "$PRESENTATION_MODES" \
            --out "$EVAL_OUT"
    fi
else
    note "Skipping evaluation."
fi

if [ "$SKIP_PLOTS" != "1" ]; then
    if [ "$BREV_REMOTE_PLOTS_DONE" = "1" ]; then
        note "Using plots produced on the Brev instance."
    else
        run_logged_step plot_model_comparison \
            mhcflurry plot-model-comparison --input "$RUN_DIR/eval_comparison"
    fi
else
    note "Skipping plots."
fi

if [ "$SKIP_DEPLOY" != "1" ]; then
    run_logged_step deploy_trained_models \
        "$SCRIPT_DIR/deploy_trained_models.sh" \
        --run-dir "$RUN_DIR" \
        --release "$RELEASE" \
        --github-release "$GITHUB_RELEASE" \
        --mode "$DEPLOY_MODE"
else
    note "Skipping deploy step."
fi
