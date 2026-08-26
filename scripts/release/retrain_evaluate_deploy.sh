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
Implementation engine for mhcflurry train pan-allele-release.

Usage:
  mhcflurry train pan-allele-release \
      --run-dir /path/to/release-run \
      --release 2.3.0 \
      [--backend local|brev-existing|brev-provision|ssh] \
      [--release-profile full|fast-8xa100|minimal-processing|fast-minimal] \
      [--minibatch-size 1024] \
      [--affinity-minibatch-size 1024] \
      [--affinity-max-workers-per-gpu auto] \
      [--processing-minibatch-size 512] \
      [--processing-num-jobs auto] \
      [--processing-max-workers-per-gpu auto] \
      [--processing-held-out-samples 10] \
      [--processing-variants "with_flanks no_flank short_flanks"] \
      [--presentation-processing-with-flanks-kind short_flanks] \
      [--presentation-decoys-per-hit 2] \
      [--presentation-sample-fraction 0.1] \
      [--presentation-feature-chunk-size 250000] \
      [--presentation-num-jobs auto] \
      [--presentation-max-workers-per-gpu 1] \
      [--presentation-calibration-num-jobs auto] \
      [--presentation-calibration-max-workers-per-gpu auto] \
      [--presentation-calibration-prediction-batch-size auto] \
      [--compare-presentation-num-jobs 1] \
      [--compare-presentation-max-workers-per-gpu 1] \
      [--compare-presentation-torch-compile 0] \
      [--eval-max-benchmark-files N] \
      [--compare-baseline public:2.0.0] \
      [--compare-baseline-label "MHCflurry 2.0"] \
      [--compare-gpus auto|N] \
      [--brev-instance NAME] [--brev-on-finish leave|stop|delete] \
      [--brev-provider auto|gcp|denvr|denvr-80gb] \
      [--brev-stop-failure-action warn|delete] \
      [--brev-cleanup-timeout-seconds 60] \
      [--brev-create-timeout-seconds 2400] \
      [--brev-container-image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime] \
      [--brev-sync-mode release|full] \
      [--paper-figures-scores-dir DIR] \
      [--paper-figures-multiallelic-predictions FILE] \
      [--paper-figures-monoallelic-predictions FILE] \
      [--paper-figures-prepare-command COMMAND] \
      [--brev-instance-type TYPE] \
      [--no-sync-remote-output] \
      [--allow-dirty-repo] \
      [--skip-train] [--skip-eval] [--skip-plots] \
      [--deploy-mode none|dry-run|draft|publish]

Backends:
  local          Run scripts/training/pan_allele_release_full.sh here.
  brev-existing  Run on a named existing Brev instance. Requires
                 --brev-instance. A missing instance is an error.
  brev-provision Provision a named Brev instance if it does not exist, then run
                 the same remote training job. If --brev-instance is omitted,
                 this script generates a run-specific name. By default runplz
                 chooses an A100 machine matching the launcher resource
                 requirements; pass --brev-instance-type to pin a specific
                 Brev shape for reproducibility or price control. The release
                 wrapper owns artifact sync and cleanup; runplz is asked to
                 leave the instance up until those steps finish.
                 --brev-provider is a convenience alias for common release
                 shapes: auto delegates to runplz price/availability selection,
                 gcp pins the old 4xA100 GCP shape, and denvr / denvr-80gb pin
                 the cheaper 8xA100 Denvr shapes when available.
  ssh            Run on a specific remote host, then rsync the run directory
                 back. Requires --remote, --remote-repo, and --remote-run-dir.
                 Authentication is whatever your local ssh/rsync configuration
                 uses, typically SSH keys or an SSH config Host entry.

Remote output:
  --no-sync-remote-output intentionally leaves newly trained artifacts on the
  remote machine. It is a training-only mode and therefore requires
  --skip-eval, --skip-plots, and --deploy-mode none. SSH and Brev artifacts are
  still provenance-validated remotely before this command succeeds.

Release profiles:
  full                Default. Train all release processing artifacts on the
                      configured backend/provider.
  fast-8xa100         For throughput runs on 8xA100 / 80 GB machines. When
                      provisioning Brev and no provider/type was explicitly
                      set, request the Denvr 8xA100 80 GB shape. Worker
                      packing still uses the normal auto resolver unless
                      explicitly overridden.
  minimal-processing  Train only with_flanks and no_flank processing artifacts
                      and evaluate only those processing modes. Use this only
                      when the short_flanks processing artifact is intentionally
                      out of scope for the run.
  fast-minimal        Apply fast-8xa100 and minimal-processing together.

Evaluation:
  After training, the script runs:
      mhcflurry eval compare-models --a RUN_DIR --b COMPARE_BASELINE
      mhcflurry eval plot-comparison --input RUN_DIR/eval_comparison
  compare-models writes release_summary.csv and release_summary.md with
  affinity, processing, and presentation release-gate tables. Presentation
  inference is memory-heavier than affinity/processing, so the release wrapper
  defaults it to one GPU worker unless overridden. The default baseline is the
  closest older public release available in downloads.yml, public:2.0.0; pass
  --compare-baseline public to compare against the currently configured public
  release, or pass a model-run directory / public:<release_name>.
  When affinity is included, the workflow also writes
  RUN_DIR/eval_comparison_train_excluded_affinity using the official
  models.no_additional_ms predictor and its matching train-excluded benchmark.
  This is the generalization gate; a production-release comparison may overlap
  historical training data and is retained only for descriptive continuity.
  --eval-max-benchmark-files limits each evaluation benchmark family to the
  first N benchmark input CSV files. It is intended only for smoke proofs that
  the end-to-end command wiring works.
  The plotting step also renders publication-style SVG/PDF/PNG panels under
  RUN_DIR/eval_comparison/plots/paper_figures/.
  By default those panels use the current compare-models output as their score
  directory. Pass --paper-figures-scores-dir or a saved prediction table to add
  external predictor score tables or other paper-figure inputs.
  External predictors such as NetMHCpan and MixMHCpred are not run by
  MHCflurry. To keep a one-line remote-training launch, pass
  --paper-figures-prepare-command COMMAND. The command runs locally on the
  control machine while remote training is active and should write canonical
  saved prediction tables or score caches to the path supplied through
  --paper-figures-scores-dir / --paper-figures-*-predictions. When local paper
  inputs are requested during a full Brev run, the wrapper waits for those
  inputs after training sync, stages them to the Brev machine, renders
  paper-ready figures remotely, syncs the PDFs/plots back, then cleans up. For
  NetMHCpan/MixMHCpred via mhctools, use the optional subprocess adapter:
      mhcflurry eval paper-figures external-predictors

Deployment:
  Deployment is opt-in. By default --deploy-mode is none and the final
  deploy_trained_models.sh step is skipped. Pass --deploy-mode dry-run to
  validate and print release assets without uploading, or pass draft / publish
  for upload modes. The legacy --skip-deploy flag is accepted as a no-op for
  old automation; new workflows should omit it.

Logs:
  The wrapper writes per-step logs and a status table under:
      RUN_DIR/workflow_logs/
  Brev sync defaults to release mode: final selected model directories plus
  runplz events, training/eval logs, component comparison predictions, plots,
  GPU telemetry, and generated configs.
  Use --brev-sync-mode full only for full post-mortem copies of all candidate
  pools and intermediate CSVs.

Postprocess-only Brev runs:
  With --backend brev-provision --skip-train, the wrapper can provision a
  temporary Brev machine, copy the final selected model artifacts from RUN_DIR,
  run compare-models / plot-model-comparison remotely, sync summary tables and
  figures back, and then apply --brev-on-finish. It does not upload the full
  training run or row-level prediction CSVs.
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

require_clean_runplz_3160() {
    local runplz_executable
    local runplz_shebang
    local -a runplz_python
    runplz_executable="$(command -v runplz)" || \
        die "Required command not found: runplz"
    IFS= read -r runplz_shebang < "$runplz_executable" || \
        die "Could not read runplz executable: $runplz_executable"
    case "$runplz_shebang" in
        '#!'*) ;;
        *) die "runplz executable has no interpreter shebang: $runplz_executable" ;;
    esac
    read -r -a runplz_python <<< "${runplz_shebang#\#!}"
    [ "${#runplz_python[@]}" -gt 0 ] || \
        die "Could not resolve the runplz interpreter: $runplz_executable"
    "${runplz_python[@]}" \
        "$SCRIPT_DIR/validate_runplz_provenance.py" \
        --executable "$runplz_executable" \
        --required-version "$RUNPLZ_REQUIRED_VERSION" || \
        die "runplz $RUNPLZ_REQUIRED_VERSION from PyPI or a clean checkout is required"
}

validate_release_provenance() {
    local step="$1"
    local require_artifacts="$2"
    local args=(
        python3 "$SCRIPT_DIR/validate_release_provenance.py"
        --repo "$REPO"
        --run-dir "$RUN_DIR"
        --release "$RELEASE"
        --workflow-id "$WORKFLOW_RUN_ID"
        --processing-variants "$PROCESSING_VARIANTS"
        --out "$RUN_DIR/release_provenance.json"
    )
    if [ "$require_artifacts" = "1" ]; then
        args+=(--require-artifacts)
        if [ "$SKIP_TRAIN" != "1" ]; then
            args+=(--expected-artifact-workflow-id "$WORKFLOW_RUN_ID")
        fi
    fi
    if [ "$ALLOW_DIRTY_REPO" = "1" ]; then
        args+=(--allow-dirty-repo)
    fi
    if [ "$require_artifacts" = "1" ] && [ "$SKIP_TRAIN" = "1" ]; then
        args+=(--allow-artifact-source-mismatch)
    fi
    run_logged_step "$step" "${args[@]}"
}

validate_ssh_remote_release_provenance() {
    local args=(
        python3 "$REMOTE_REPO/scripts/release/validate_release_provenance.py"
        --repo "$REMOTE_REPO"
        --run-dir "$REMOTE_RUN_DIR"
        --release "$RELEASE"
        --workflow-id "$WORKFLOW_RUN_ID"
        --processing-variants "$PROCESSING_VARIANTS"
        --out "$REMOTE_RUN_DIR/release_provenance.json"
        --require-artifacts
        --expected-artifact-workflow-id "$WORKFLOW_RUN_ID"
    )
    if [ "$ALLOW_DIRTY_REPO" = "1" ]; then
        args+=(--allow-dirty-repo)
    fi
    local arg
    local remote_command=
    for arg in "${args[@]}"; do
        if [ -n "$remote_command" ]; then
            remote_command="$remote_command "
        fi
        remote_command="$remote_command$(shell_quote "$arg")"
    done
    run_logged_step ssh_model_provenance ssh "$REMOTE" "$remote_command"
}

lowercase() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

display_release_version() {
    printf '%s' "$1" | sed 's/[.-]\{0,1\}rc[0-9][0-9]*$//'
}

public_release_from_spec() {
    case "$1" in
        public:*)
            printf '%s\n' "${1#public:}"
            ;;
    esac
}

fetch_pinned_public_baseline_downloads() {
    local release
    release="$(public_release_from_spec "$COMPARE_BASELINE")"
    if [ -z "$release" ]; then
        return 0
    fi
    MHCFLURRY_DOWNLOADS_CURRENT_RELEASE="$release" \
        mhcflurry-downloads fetch \
        models_class1_pan models_class1_processing models_class1_presentation
}

normalize_compare_torch_compile() {
    local value="${1:-auto}"
    case "$(lowercase "$value")" in
        auto) printf 'auto\n' ;;
        1|true|yes|on) printf '1\n' ;;
        0|false|no|off) printf '0\n' ;;
        *)
            die "COMPARE_TORCH_COMPILE must be auto, true/false, or 1/0; got '$value'"
            ;;
    esac
}

normalize_compare_matmul_precision() {
    local value="${1:-high}"
    local normalized
    normalized="$(lowercase "$value")"
    case "$normalized" in
        none|highest|high|medium) printf '%s\n' "$normalized" ;;
        *)
            die "COMPARE_MATMUL_PRECISION must be one of none, highest, high, medium; got '$value'"
            ;;
    esac
}

validate_compare_gpus() {
    local value="${1:-auto}"
    if [ "$(lowercase "$value")" = "auto" ]; then
        return 0
    fi
    case "$value" in
        ''|*[!0-9]*)
            die "COMPARE_GPUS must be auto or a non-negative integer; got '$value'"
            ;;
    esac
}

validate_positive_integer() {
    local option=$1
    local value=$2
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        die "$option must be a positive integer; got '$value'"
    fi
}

validate_nonnegative_integer() {
    local option=$1
    local value=$2
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        die "$option must be a non-negative integer; got '$value'"
    fi
}

validate_auto_or_nonnegative_integer() {
    local option=$1
    local value=$2
    if [ "$(lowercase "$value")" != "auto" ]; then
        validate_nonnegative_integer "$option" "$value"
    fi
}

validate_auto_or_positive_integer() {
    local option=$1
    local value=$2
    if [ "$(lowercase "$value")" != "auto" ]; then
        validate_positive_integer "$option" "$value"
    fi
}

validate_positive_number() {
    local option=$1
    local value=$2
    if ! [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || \
            ! awk -v value="$value" 'BEGIN { exit !(value > 0) }'; then
        die "$option must be a finite positive number; got '$value'"
    fi
}

validate_fraction() {
    local option=$1
    local value=$2
    validate_positive_number "$option" "$value"
    if ! awk -v value="$value" 'BEGIN { exit !(value <= 1) }'; then
        die "$option must be no greater than 1; got '$value'"
    fi
}

brev_provider_instance_type() {
    case "$(lowercase "$1")" in
        auto|'')
            printf '\n'
            ;;
        gcp)
            printf 'a2-highgpu-4g:nvidia-tesla-a100:4\n'
            ;;
        denvr)
            printf 'denvr_A100_sxm4x8\n'
            ;;
        denvr-80gb)
            printf 'denvr_A100_sxm4_80Gx8\n'
            ;;
        *)
            die "--brev-provider must be one of: auto, gcp, denvr, denvr-80gb"
            ;;
    esac
}

apply_fast_gpu_profile() {
    if [ "$BACKEND" = "brev-provision" ] && \
            [ "$BREV_PROVIDER_EXPLICIT" = "0" ] && \
            [ "$BREV_INSTANCE_TYPE_EXPLICIT" = "0" ]; then
        BREV_PROVIDER=denvr-80gb
    fi
}

apply_minimal_processing_profile() {
    if [ "$PROCESSING_VARIANTS_EXPLICIT" = "0" ]; then
        PROCESSING_VARIANTS="with_flanks no_flank"
    fi
    if [ "$PROCESSING_MODES_EXPLICIT" = "0" ]; then
        PROCESSING_MODES="with_flanks,no_flank"
    fi
    if [ "$PRESENTATION_PROCESSING_WITH_FLANKS_KIND_EXPLICIT" = "0" ]; then
        PRESENTATION_PROCESSING_WITH_FLANKS_KIND=with_flanks
    fi
}

apply_release_profile() {
    case "$RELEASE_PROFILE" in
        full)
            ;;
        fast-8xa100)
            apply_fast_gpu_profile
            ;;
        minimal-processing)
            apply_minimal_processing_profile
            ;;
        fast-minimal)
            apply_fast_gpu_profile
            apply_minimal_processing_profile
            ;;
        *)
            die "--release-profile must be one of: full, fast-8xa100, minimal-processing, fast-minimal"
            ;;
    esac
}

processing_variant_requested() {
    case " $PROCESSING_VARIANTS " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

validate_processing_configuration() {
    local derived_modes=
    local mode
    local modes_list
    local modes_seen=0
    local seen_modes=" "
    local seen_variants=" "
    local variant

    for variant in $PROCESSING_VARIANTS; do
        case "$variant" in
            with_flanks|no_flank|short_flanks) ;;
            *) die "Unknown --processing-variants entry: $variant" ;;
        esac
        case "$seen_variants" in
            *" $variant "*) die "Duplicate --processing-variants entry: $variant" ;;
        esac
        seen_variants="$seen_variants$variant "
        if [ -n "$derived_modes" ]; then
            derived_modes="$derived_modes,$variant"
        else
            derived_modes=$variant
        fi
    done
    [ -n "$derived_modes" ] || \
        die "--processing-variants must contain at least one variant"

    if [ "$PROCESSING_MODES_EXPLICIT" = "0" ]; then
        PROCESSING_MODES=$derived_modes
    fi
    case "$PROCESSING_MODES" in
        ,*|*,|*,,*) die "--processing-modes contains an empty entry" ;;
    esac
    modes_list=$(printf '%s' "$PROCESSING_MODES" | tr ',' ' ')
    for mode in $modes_list; do
        modes_seen=1
        case "$mode" in
            with_flanks|no_flank|short_flanks) ;;
            *) die "Unknown --processing-modes entry: $mode" ;;
        esac
        case "$seen_modes" in
            *" $mode "*) die "Duplicate --processing-modes entry: $mode" ;;
        esac
        seen_modes="$seen_modes$mode "
        processing_variant_requested "$mode" || die \
            "--processing-modes requests '$mode', but --processing-variants does not train it"
    done
    [ "$modes_seen" = "1" ] || \
        die "--processing-modes must contain at least one mode"

    if [ "$SKIP_TRAIN" != "1" ]; then
        processing_variant_requested no_flank || die \
            "--processing-variants must include no_flank for presentation training"
        case "$PRESENTATION_PROCESSING_WITH_FLANKS_KIND" in
            with_flanks|short_flanks) ;;
            *) die "--presentation-processing-with-flanks-kind must be with_flanks or short_flanks" ;;
        esac
        processing_variant_requested \
            "$PRESENTATION_PROCESSING_WITH_FLANKS_KIND" || die \
            "--processing-variants must include the presentation with-flanks variant '$PRESENTATION_PROCESSING_WITH_FLANKS_KIND'"
    fi
    if [ "$DEPLOY_MODE" != "none" ]; then
        processing_variant_requested with_flanks || die \
            "--deploy-mode $DEPLOY_MODE requires the canonical with_flanks processing artifact"
    fi
}

paper_figure_inputs_requested() {
    [ -n "$PAPER_FIGURES_SCORES_DIR" ] || \
        [ -n "$PAPER_FIGURES_ARTIFACTS_DIR" ] || \
        [ -n "$PAPER_FIGURES_MULTIALLELIC_PREDICTIONS" ] || \
        [ -n "$PAPER_FIGURES_MONOALLELIC_PREDICTIONS" ]
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
    local errexit_was_set=0
    case "$-" in
        *e*) errexit_was_set=1 ;;
    esac
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
    if [ "$errexit_was_set" = "1" ]; then
        set -e
    else
        set +e
    fi

    {
        printf '[%s] end step=%s status=%s\n' \
            "$(workflow_timestamp)" "$step" "$status"
    } | tee -a "$log_file" >&2
    record_workflow_event "$step" "$status" "log=$log_file"
    return "$status"
}

shell_quote() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

verify_ssh_remote_checkout() {
    local check_script
    local remote_command

    check_script='set -eu
remote_repo=$1
expected_commit=$2
allow_dirty=$3
actual_commit=$(git -C "$remote_repo" rev-parse HEAD)
if [ "$actual_commit" != "$expected_commit" ]; then
    printf "ERROR: remote checkout commit %s does not match local release commit %s: %s\n" "$actual_commit" "$expected_commit" "$remote_repo" >&2
    exit 2
fi
if [ "$allow_dirty" != "1" ]; then
    dirty=$(git -C "$remote_repo" status --porcelain --untracked-files=no)
    if [ -n "$dirty" ]; then
        printf "ERROR: remote checkout has tracked changes: %s\n" "$remote_repo" >&2
        printf "%s\n" "$dirty" >&2
        exit 2
    fi
fi
printf "Verified remote source provenance: commit=%s repo=%s\n" "$actual_commit" "$remote_repo"'
    remote_command="bash -c $(shell_quote "$check_script") _"
    remote_command="$remote_command $(shell_quote "$REMOTE_REPO")"
    remote_command="$remote_command $(shell_quote "$RELEASE_GIT_COMMIT")"
    remote_command="$remote_command $(shell_quote "$ALLOW_DIRTY_REPO")"
    run_logged_step ssh_source_provenance ssh "$REMOTE" "$remote_command"
}

run_with_timeout() {
    local timeout_seconds="$1"
    shift
    python3 - "$timeout_seconds" "$@" <<'PY'
import subprocess
import sys

timeout_seconds = float(sys.argv[1])
command = sys.argv[2:]
try:
    result = subprocess.run(command, timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    print(
        "Command timed out after %.0f seconds: %s" % (
            timeout_seconds, " ".join(command),
        ),
        file=sys.stderr,
    )
    raise SystemExit(124)
raise SystemExit(result.returncode)
PY
}

run_logged_step_with_timeout() {
    local step="$1"
    local timeout_seconds="$2"
    shift 2
    run_logged_step "$step" run_with_timeout "$timeout_seconds" "$@"
}

start_paper_figures_prepare() {
    if [ -z "${PAPER_FIGURES_PREPARE_COMMAND:-}" ]; then
        return 0
    fi
    if [ "${PAPER_FIGURES_PREPARE_DONE:-0}" = "1" ] || \
            [ -n "${PAPER_FIGURES_PREPARE_PID:-}" ]; then
        return 0
    fi

    local step=paper_figures_prepare
    local log_file="$WORKFLOW_LOG_DIR/$step.log"
    run_cmd mkdir -p "$WORKFLOW_LOG_DIR"
    record_workflow_event "$step" start \
        "background=1 log=$log_file command=$PAPER_FIGURES_PREPARE_COMMAND"

    if [ "$DRY_RUN" = "1" ]; then
        run_cmd bash -lc "$PAPER_FIGURES_PREPARE_COMMAND"
        record_workflow_event "$step" 0 "dry-run log=$log_file"
        PAPER_FIGURES_PREPARE_DONE=1
        return 0
    fi

    {
        printf '[%s] start step=%s background=1 command=%s\n' \
            "$(workflow_timestamp)" "$step" "$PAPER_FIGURES_PREPARE_COMMAND"
    } | tee -a "$log_file" >&2

    (
        set -o pipefail
        bash -lc "$PAPER_FIGURES_PREPARE_COMMAND" 2>&1 | tee -a "$log_file"
        exit "${PIPESTATUS[0]}"
    ) &
    PAPER_FIGURES_PREPARE_PID=$!
    note "Started local paper-figure input preparation as PID $PAPER_FIGURES_PREPARE_PID."
}

wait_paper_figures_prepare() {
    if [ -z "${PAPER_FIGURES_PREPARE_COMMAND:-}" ]; then
        return 0
    fi
    if [ "${PAPER_FIGURES_PREPARE_DONE:-0}" = "1" ]; then
        return 0
    fi
    if [ -z "${PAPER_FIGURES_PREPARE_PID:-}" ]; then
        return 0
    fi

    local step=paper_figures_prepare
    local log_file="$WORKFLOW_LOG_DIR/$step.log"
    local errexit_was_set=0
    local status
    case "$-" in
        *e*) errexit_was_set=1 ;;
    esac

    note "Waiting for local paper-figure input preparation (PID $PAPER_FIGURES_PREPARE_PID)."
    set +e
    wait "$PAPER_FIGURES_PREPARE_PID"
    status=$?
    if [ "$errexit_was_set" = "1" ]; then
        set -e
    else
        set +e
    fi
    PAPER_FIGURES_PREPARE_DONE=1
    PAPER_FIGURES_PREPARE_PID=

    {
        printf '[%s] end step=%s status=%s\n' \
            "$(workflow_timestamp)" "$step" "$status"
    } | tee -a "$log_file" >&2
    record_workflow_event "$step" "$status" "log=$log_file"
    return "$status"
}

cleanup_background_jobs() {
    if [ -n "${PAPER_FIGURES_PREPARE_PID:-}" ] && \
            [ "${PAPER_FIGURES_PREPARE_DONE:-0}" != "1" ]; then
        if kill -0 "$PAPER_FIGURES_PREPARE_PID" 2>/dev/null; then
            warn "Stopping unfinished paper-figure input preparation (PID $PAPER_FIGURES_PREPARE_PID)."
            kill "$PAPER_FIGURES_PREPARE_PID" 2>/dev/null || true
            wait "$PAPER_FIGURES_PREPARE_PID" 2>/dev/null || true
        fi
    fi
}

write_git_repo_archive() {
    local output="$1"
    local tar_output="${output%.bz2}"
    rm -f "$tar_output" "$output" || return $?
    git -C "$REPO" archive --format=tar HEAD -o "$tar_output" || return $?
    bzip2 -f "$tar_output" || return $?
    [ -s "$output" ]
}

run_dir_has_model_artifacts() {
    [ -d "$RUN_DIR/affinity/models.combined" ] || return 1
    [ -d "$RUN_DIR/presentation/models" ] || return 1
    local kind
    for kind in $PROCESSING_VARIANTS; do
        [ -d "$RUN_DIR/processing/models.selected.$kind" ] || return 1
    done
    return 0
}

run_dir_matches_workflow() {
    local marker="$RUN_DIR/.runplz/mhcflurry_release_workflow_id"
    [ -f "$marker" ] || return 1
    [ "$(cat "$marker")" = "$WORKFLOW_RUN_ID" ]
}

run_dir_has_synced_brev_outputs() {
    run_dir_matches_workflow || return 1
    run_dir_has_model_artifacts || return 1
    [ -f "$RUN_DIR/release_holdout/policy.json" ] || return 1
    [ -f "$RUN_DIR/release_holdout/validation.json" ] || return 1
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

brev_latest_remote_workflow_id() {
    require_command brev
    local output
    output="$(
        run_with_timeout "$BREV_CLEANUP_TIMEOUT_SECONDS" brev exec "$BREV_INSTANCE" \
            "bash -lc 'marker=\$(cat ~/runplz-latest/out/.runplz/mhcflurry_release_workflow_id 2>/dev/null || true); printf \"MHCFLURRY_WORKFLOW_ID=%s\\n\" \"\$marker\"'" \
            2>/dev/null || true
    )"
    printf '%s\n' "$output" | sed -n \
        's/^MHCFLURRY_WORKFLOW_ID=//p' \
        | tail -1
}

brev_latest_remote_exit_code() {
    require_command brev
    local output
    local expected
    expected="$(printf '%q' "$WORKFLOW_RUN_ID")"
    output="$(
        run_with_timeout "$BREV_CLEANUP_TIMEOUT_SECONDS" brev exec "$BREV_INSTANCE" \
            "MHCFLURRY_RELEASE_WORKFLOW_ID=$expected bash -lc 'marker=\$(cat ~/runplz-latest/out/.runplz/mhcflurry_release_workflow_id 2>/dev/null || true); [ \"\$marker\" = \"\$MHCFLURRY_RELEASE_WORKFLOW_ID\" ] || exit 0; cat ~/runplz-latest/out/.runplz/mhcflurry_release_workflow_exit_code 2>/dev/null || true'" \
            2>/dev/null || true
    )"
    printf '%s\n' "$output" | sed -n \
        's/[^0-9]*\([0-9][0-9]*\).*/\1/p' \
        | tail -1
}

brev_instance_status() {
    require_command brev
    BREV_INSTANCE_NAME="$BREV_INSTANCE" \
    BREV_CLEANUP_TIMEOUT_SECONDS="$BREV_CLEANUP_TIMEOUT_SECONDS" \
        python3 - <<'PY'
import json
import os
import subprocess
import sys

name = os.environ["BREV_INSTANCE_NAME"]
timeout_seconds = float(os.environ["BREV_CLEANUP_TIMEOUT_SECONDS"])
try:
    result = subprocess.run(
        ["brev", "ls", "--json"],
        capture_output=True,
        check=True,
        text=True,
        timeout=timeout_seconds,
    )
except subprocess.TimeoutExpired:
    print(
        "brev ls --json timed out after %.0f seconds" % timeout_seconds,
        file=sys.stderr,
    )
    raise SystemExit(124)
except subprocess.CalledProcessError as exc:
    if exc.stderr:
        print(exc.stderr, file=sys.stderr, end="")
    raise SystemExit(exc.returncode)

for item in json.loads(result.stdout or "[]"):
    if item.get("name") == name:
        print(item.get("status", ""))
        break
PY
}

brev_instance_field() {
    require_command brev
    local field="$1"
    BREV_INSTANCE_NAME="$BREV_INSTANCE" \
    BREV_INSTANCE_FIELD="$field" \
    BREV_CLEANUP_TIMEOUT_SECONDS="$BREV_CLEANUP_TIMEOUT_SECONDS" \
        python3 - <<'PY'
import json
import os
import subprocess
import sys

name = os.environ["BREV_INSTANCE_NAME"]
field = os.environ["BREV_INSTANCE_FIELD"]
timeout_seconds = float(os.environ["BREV_CLEANUP_TIMEOUT_SECONDS"])
try:
    result = subprocess.run(
        ["brev", "ls", "--json"],
        capture_output=True,
        check=True,
        text=True,
        timeout=timeout_seconds,
    )
except subprocess.TimeoutExpired:
    print(
        "brev ls --json timed out after %.0f seconds" % timeout_seconds,
        file=sys.stderr,
    )
    raise SystemExit(124)
except subprocess.CalledProcessError as exc:
    if exc.stderr:
        print(exc.stderr, file=sys.stderr, end="")
    raise SystemExit(exc.returncode)

for item in json.loads(result.stdout or "[]"):
    if item.get("name") == name:
        print(item.get(field, ""))
        break
PY
}

wait_for_brev_shell_ready() {
    local attempt
    local status
    local shell_status
    local health_status
    for attempt in $(seq 1 "$BREV_SHELL_READY_ATTEMPTS"); do
        status="$(brev_instance_status || true)"
        shell_status="$(brev_instance_field shell_status || true)"
        health_status="$(brev_instance_field health_status || true)"
        note "Brev readiness check $attempt/$BREV_SHELL_READY_ATTEMPTS: status=${status:-unknown} shell=${shell_status:-unknown} health=${health_status:-unknown}"
        if [ "$status" = "RUNNING" ] && [ "$shell_status" = "READY" ]; then
            return 0
        fi
        if [ "$status" = "FAILURE" ]; then
            return 1
        fi
        sleep "$BREV_SHELL_READY_DELAY_SECONDS"
    done
    return 1
}

build_brev_postprocess_archives() {
    local staging="$1"
    local remote_paper_inputs_root="$2"
    local remote_data_dir="$3"
    local repo_archive="$staging/repo.tar.bz2"
    local models_archive="$staging/model_artifacts.tar.bz2"
    local artifact_paths=(
        affinity/models.combined
        presentation/models
        release_holdout
    )
    if [ "$SKIP_PLOTS" != "1" ]; then
        artifact_paths+=(affinity/models.unselected.combined)
    fi
    local kind
    for kind in $PROCESSING_VARIANTS; do
        artifact_paths+=("processing/models.selected.$kind")
    done

    run_cmd mkdir -p "$staging"
    run_logged_step postprocess_package_repo \
        write_git_repo_archive "$repo_archive"
    run_logged_step postprocess_package_models \
        tar -C "$RUN_DIR" -cjf "$models_archive" "${artifact_paths[@]}"
    build_brev_paper_input_archive "$staging" "$remote_paper_inputs_root"
    build_brev_data_dir_archive "$staging" "$remote_data_dir"
}

canonical_existing_path() {
    local path="$1"
    if [ -d "$path" ]; then
        (cd "$path" && pwd -P)
    else
        local dir
        local base
        dir="$(dirname "$path")"
        base="$(basename "$path")"
        printf '%s/%s\n' "$(cd "$dir" && pwd -P)" "$base"
    fi
}

same_existing_path() {
    local left="$1"
    local right="$2"
    if [ -z "$left" ] || [ -z "$right" ]; then
        return 1
    fi
    if [ ! -e "$left" ] || [ ! -e "$right" ]; then
        return 1
    fi
    [ "$(canonical_existing_path "$left")" = "$(canonical_existing_path "$right")" ]
}

stage_brev_paper_input() {
    local source="$1"
    local target_relative="$2"
    local input_dir="$3"
    local remote_root="$4"
    [ -e "$source" ] || die "Paper-figure input not found: $source"
    local target="$input_dir/$target_relative"
    rm -rf "$target"
    mkdir -p "$(dirname "$target")"
    if [ -d "$source" ]; then
        mkdir -p "$target"
        (
            cd "$source"
            tar -cf - .
        ) | (
            cd "$target"
            tar -xf -
        )
    else
        cp "$source" "$target"
    fi
    printf '%s/%s\n' "$remote_root" "$target_relative"
}

build_brev_paper_input_archive() {
    local staging="$1"
    local remote_root="$2"
    local paper_input_dir="$staging/paper_inputs"
    local paper_archive="$staging/paper_inputs.tar.bz2"
    local multiallelic_basename
    local monoallelic_basename

    BREV_REMOTE_PAPER_FIGURES_SCORES_DIR=
    BREV_REMOTE_PAPER_FIGURES_ARTIFACTS_DIR=
    BREV_REMOTE_PAPER_FIGURES_MULTIALLELIC_PREDICTIONS=
    BREV_REMOTE_PAPER_FIGURES_MONOALLELIC_PREDICTIONS=

    rm -rf "$paper_input_dir" "$paper_archive"
    if [ -z "$PAPER_FIGURES_SCORES_DIR" ] && \
            [ -z "$PAPER_FIGURES_ARTIFACTS_DIR" ] && \
            [ -z "$PAPER_FIGURES_MULTIALLELIC_PREDICTIONS" ] && \
            [ -z "$PAPER_FIGURES_MONOALLELIC_PREDICTIONS" ]; then
        return 0
    fi

    mkdir -p "$paper_input_dir"
    if [ -n "$PAPER_FIGURES_SCORES_DIR" ]; then
        BREV_REMOTE_PAPER_FIGURES_SCORES_DIR="$(
            stage_brev_paper_input \
                "$PAPER_FIGURES_SCORES_DIR" \
                scores_dir \
                "$paper_input_dir" \
                "$remote_root"
        )"
    fi
    if [ -n "$PAPER_FIGURES_ARTIFACTS_DIR" ]; then
        if same_existing_path \
                "$PAPER_FIGURES_ARTIFACTS_DIR" "$PAPER_FIGURES_SCORES_DIR"; then
            BREV_REMOTE_PAPER_FIGURES_ARTIFACTS_DIR="$BREV_REMOTE_PAPER_FIGURES_SCORES_DIR"
        else
            BREV_REMOTE_PAPER_FIGURES_ARTIFACTS_DIR="$(
                stage_brev_paper_input \
                    "$PAPER_FIGURES_ARTIFACTS_DIR" \
                    artifacts_dir \
                    "$paper_input_dir" \
                    "$remote_root"
            )"
        fi
    fi
    if [ -n "$PAPER_FIGURES_MULTIALLELIC_PREDICTIONS" ]; then
        multiallelic_basename="$(basename "$PAPER_FIGURES_MULTIALLELIC_PREDICTIONS")"
        BREV_REMOTE_PAPER_FIGURES_MULTIALLELIC_PREDICTIONS="$(
            stage_brev_paper_input \
                "$PAPER_FIGURES_MULTIALLELIC_PREDICTIONS" \
                "multiallelic_predictions/$multiallelic_basename" \
                "$paper_input_dir" \
                "$remote_root"
        )"
    fi
    if [ -n "$PAPER_FIGURES_MONOALLELIC_PREDICTIONS" ]; then
        monoallelic_basename="$(basename "$PAPER_FIGURES_MONOALLELIC_PREDICTIONS")"
        BREV_REMOTE_PAPER_FIGURES_MONOALLELIC_PREDICTIONS="$(
            stage_brev_paper_input \
                "$PAPER_FIGURES_MONOALLELIC_PREDICTIONS" \
                "monoallelic_predictions/$monoallelic_basename" \
                "$paper_input_dir" \
                "$remote_root"
        )"
    fi

    run_logged_step postprocess_package_paper_inputs \
        tar -C "$paper_input_dir" -cjf "$paper_archive" .
}

build_brev_data_dir_archive() {
    local staging="$1"
    local remote_data_dir="$2"
    local data_input_dir="$staging/data_dir"
    local data_archive="$staging/data_dir.tar.bz2"

    BREV_REMOTE_DATA_DIR=
    rm -rf "$data_input_dir" "$data_archive"
    if [ -z "$DATA_DIR" ]; then
        return 0
    fi
    [ -d "$DATA_DIR" ] || die "Evaluation data directory not found: $DATA_DIR"
    mkdir -p "$data_input_dir"
    (
        cd "$DATA_DIR"
        tar -cf - .
    ) | (
        cd "$data_input_dir"
        tar -xf -
    )
    BREV_REMOTE_DATA_DIR="$remote_data_dir"
    run_logged_step postprocess_package_data_dir \
        tar -C "$data_input_dir" -cjf "$data_archive" .
}

ensure_brev_postprocess_instance() {
    local auto_create="$1"
    require_command brev

    local status
    status="$(brev_instance_status || true)"
    if [ -z "$status" ]; then
        if [ "$auto_create" != "1" ]; then
            die "Brev instance not found: $BREV_INSTANCE"
        fi
        local create_args=(
            brev create "$BREV_INSTANCE"
            --mode container
            --container-image "$BREV_CONTAINER_IMAGE"
            --timeout "$BREV_CREATE_TIMEOUT_SECONDS"
        )
        if [ -n "$BREV_INSTANCE_TYPE" ]; then
            create_args+=(--type "$BREV_INSTANCE_TYPE")
        fi
        run_logged_step_with_timeout \
            brev_create_postprocess "$BREV_CREATE_TIMEOUT_SECONDS" \
            "${create_args[@]}"
        return 0
    fi
    if [ "$status" != "RUNNING" ]; then
        run_logged_step_with_timeout \
            brev_start_postprocess "$BREV_CREATE_TIMEOUT_SECONDS" \
            brev start "$BREV_INSTANCE"
    fi
}

run_brev_postprocess() {
    local auto_create="$1"
    set +e
    (
        set -e
        run_brev_postprocess_impl "$auto_create"
    )
    local status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        warn "Brev postprocess failed with status $status; leaving $BREV_INSTANCE available to preserve remote artifacts."
        return "$status"
    fi
    if [ "$SKIP_EVAL" != "1" ]; then
        BREV_REMOTE_EVAL_DONE=1
    fi
    if [ "$SKIP_PLOTS" != "1" ]; then
        if [ "$DRY_RUN" = "1" ] || [ -d "$RUN_DIR/eval_comparison/plots" ]; then
            BREV_REMOTE_PLOTS_DONE=1
        fi
    fi
}

run_brev_postprocess_impl() {
    local auto_create="$1"
    if [ "$SKIP_EVAL" = "1" ]; then
        return 0
    fi
    if [ "$DRY_RUN" = "1" ]; then
        note "Would run Brev postprocess-only eval/plot on $BREV_INSTANCE."
        BREV_REMOTE_EVAL_DONE=1
        if [ "$SKIP_PLOTS" != "1" ]; then
            BREV_REMOTE_PLOTS_DONE=1
        fi
        return 0
    fi
    run_dir_has_model_artifacts || \
        die "Postprocess-only Brev run requires final model artifacts in $RUN_DIR"

    require_command brev
    require_command tar
    run_cmd mkdir -p "$RUN_DIR"
    local staging="$RUN_DIR/.brev-postprocess"
    local remote_root=/root/mhcflurry-postprocess
    local repo_archive="$staging/repo.tar.bz2"
    local models_archive="$staging/model_artifacts.tar.bz2"
    local paper_archive="$staging/paper_inputs.tar.bz2"
    local data_archive="$staging/data_dir.tar.bz2"
    local remote_script="$staging/run_remote_postprocess.sh"
    local remote_sync_script="$staging/build_postprocess_sync_archive.sh"
    local remote_archive="$remote_root/postprocess_sync.tar.bz2"
    local local_archive="$staging/postprocess_sync.tar.bz2"
    local remote_paper_inputs_root="$remote_root/paper_inputs"
    local remote_data_dir="$remote_root/data_dir"

    BREV_EXPECT_REMOTE_EVAL=1
    BREV_EXPECT_REMOTE_PLOTS=0
    if [ "$SKIP_PLOTS" != "1" ]; then
        BREV_EXPECT_REMOTE_PLOTS=1
    fi

    run_cmd rm -rf "$staging"
    build_brev_postprocess_archives \
        "$staging" "$remote_paper_inputs_root" "$remote_data_dir"
    ensure_brev_postprocess_instance "$auto_create"

    run_logged_step postprocess_wait_for_shell \
        wait_for_brev_shell_ready
    run_logged_step_with_timeout \
        postprocess_prepare_remote_dir "$BREV_CREATE_TIMEOUT_SECONDS" \
        brev exec "$BREV_INSTANCE" "rm -rf '$remote_root' && mkdir -p '$remote_root'"
    run_logged_step_with_timeout \
        postprocess_copy_repo "$BREV_CREATE_TIMEOUT_SECONDS" \
        brev copy "$repo_archive" "$BREV_INSTANCE:$remote_root/repo.tar.bz2"
    run_logged_step_with_timeout \
        postprocess_copy_models "$BREV_CREATE_TIMEOUT_SECONDS" \
        brev copy "$models_archive" "$BREV_INSTANCE:$remote_root/model_artifacts.tar.bz2"
    if [ -f "$paper_archive" ]; then
        run_logged_step_with_timeout \
            postprocess_copy_paper_inputs "$BREV_CREATE_TIMEOUT_SECONDS" \
            brev copy "$paper_archive" "$BREV_INSTANCE:$remote_root/paper_inputs.tar.bz2"
    fi
    if [ -f "$data_archive" ]; then
        run_logged_step_with_timeout \
            postprocess_copy_data_dir "$BREV_CREATE_TIMEOUT_SECONDS" \
            brev copy "$data_archive" "$BREV_INSTANCE:$remote_root/data_dir.tar.bz2"
    fi

    {
        cat <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF
        printf 'export RUN_LABEL=%q\n' "$RUN_LABEL"
        printf 'export RUN_RELEASE_PLOTS=%q\n' "$BREV_EXPECT_REMOTE_PLOTS"
        printf 'export COMPARE_BASELINE=%q\n' "$COMPARE_BASELINE"
        printf 'export COMPARE_BASELINE_LABEL=%q\n' "$COMPARE_BASELINE_LABEL"
        printf 'export COMPARE_INCLUDE=%q\n' "$COMPARE_INCLUDE"
        printf 'export EVAL_MAX_BENCHMARK_FILES=%q\n' "$EVAL_MAX_BENCHMARK_FILES"
        printf 'export PROCESSING_MODES=%q\n' "$PROCESSING_MODES"
        printf 'export PRESENTATION_MODES=%q\n' "$PRESENTATION_MODES"
        printf 'export COMPARE_BACKEND=%q\n' "$COMPARE_BACKEND"
        printf 'export COMPARE_NUM_JOBS=%q\n' "$COMPARE_NUM_JOBS"
        printf 'export COMPARE_MAX_WORKERS_PER_GPU=%q\n' "$COMPARE_MAX_WORKERS_PER_GPU"
        printf 'export COMPARE_MAX_TASKS_PER_WORKER=%q\n' "$COMPARE_MAX_TASKS_PER_WORKER"
        printf 'export COMPARE_TORCH_COMPILE=%q\n' "$COMPARE_TORCH_COMPILE"
        printf 'export COMPARE_MATMUL_PRECISION=%q\n' "$COMPARE_MATMUL_PRECISION"
        printf 'export COMPARE_GPUS=%q\n' "$COMPARE_GPUS"
        printf 'export DATA_DIR=%q\n' "$BREV_REMOTE_DATA_DIR"
        printf 'export COMPARE_PRESENTATION_NUM_JOBS=%q\n' "$COMPARE_PRESENTATION_NUM_JOBS"
        printf 'export COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU=%q\n' "$COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU"
        printf 'export COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER=%q\n' "$COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER"
        printf 'export COMPARE_PRESENTATION_TORCH_COMPILE=%q\n' "$COMPARE_PRESENTATION_TORCH_COMPILE"
        printf 'export PAPER_FIGURES_SCORES_DIR=%q\n' "$BREV_REMOTE_PAPER_FIGURES_SCORES_DIR"
        printf 'export PAPER_FIGURES_ARTIFACTS_DIR=%q\n' "$BREV_REMOTE_PAPER_FIGURES_ARTIFACTS_DIR"
        printf 'export PAPER_FIGURES_MULTIALLELIC_PREDICTIONS=%q\n' "$BREV_REMOTE_PAPER_FIGURES_MULTIALLELIC_PREDICTIONS"
        printf 'export PAPER_FIGURES_MONOALLELIC_PREDICTIONS=%q\n' "$BREV_REMOTE_PAPER_FIGURES_MONOALLELIC_PREDICTIONS"
        printf 'export PAPER_FIGURES_FORMATS=%q\n' "$PAPER_FIGURES_FORMATS"
        printf 'export PAPER_FIGURES_CANDIDATE_PREDICTOR=%q\n' "$PAPER_FIGURES_CANDIDATE_PREDICTOR"
        printf 'export PAPER_FIGURES_EXTERNAL_BASELINES=%q\n' "$PAPER_FIGURES_EXTERNAL_BASELINES"
        printf 'export PAPER_FIGURES_PREFERRED_PREDICTORS=%q\n' "$PAPER_FIGURES_PREFERRED_PREDICTORS"
        printf 'export PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS=%q\n' "$PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS"
        printf 'export PAPER_FIGURES_PRESENTATION_PANEL_BASELINES=%q\n' "$PAPER_FIGURES_PRESENTATION_PANEL_BASELINES"
        cat <<'EOF'

remote_root=/root/mhcflurry-postprocess
repo_dir="$remote_root/repo"
run_dir="$remote_root/run"

# BEGIN BREV POSTPROCESS REPLAY GUARD
# Some Brev CLI versions replay the complete remote command after an SSH
# transport failure. Coalesce that retry with the original invocation so setup
# and evaluation are never executed twice. The outer workflow recreates
# remote_root for every intentional run, so this state cannot block a later
# user-requested rerun.
postprocess_state_dir="$remote_root/postprocess_state"
postprocess_status_file="$postprocess_state_dir/exit_status"
postprocess_owner_file="$postprocess_state_dir/owner_pid"
if ! mkdir "$postprocess_state_dir" 2>/dev/null; then
    echo "Detected replay of Brev postprocess command; waiting for original invocation." >&2
    while [ ! -f "$postprocess_status_file" ]; do
        postprocess_owner_pid="$(cat "$postprocess_owner_file" 2>/dev/null || true)"
        postprocess_owner_state=""
        if [ -n "$postprocess_owner_pid" ]; then
            postprocess_owner_state="$(
                awk '{print $3}' "/proc/$postprocess_owner_pid/stat" \
                    2>/dev/null || true
            )"
        fi
        if [ -n "$postprocess_owner_pid" ] && {
                [ -z "$postprocess_owner_state" ] || \
                [ "$postprocess_owner_state" = "Z" ];
        }; then
            # The owner may have exited between its final command and the EXIT
            # trap's atomic marker rename. Give that narrow race one interval
            # to publish the real status before declaring the run abandoned.
            sleep 1
            [ -f "$postprocess_status_file" ] && continue
            printf '70\n' > "$postprocess_status_file.tmp.$$"
            mv "$postprocess_status_file.tmp.$$" "$postprocess_status_file"
            echo "Original Brev postprocess owner exited without a status marker." >&2
            break
        fi
        sleep 1
    done
    postprocess_original_status="$(cat "$postprocess_status_file")"
    echo "Brev postprocess replay returning original status $postprocess_original_status." >&2
    exit "$postprocess_original_status"
fi
printf '%s\n' "$$" > "$postprocess_owner_file"

record_postprocess_status() {
    local status="$?"
    trap - EXIT
    printf '%s\n' "$status" > "$postprocess_status_file.tmp.$$"
    mv "$postprocess_status_file.tmp.$$" "$postprocess_status_file"
    exit "$status"
}
trap record_postprocess_status EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
# END BREV POSTPROCESS REPLAY GUARD

# Reuse the PyTorch environment that produced the models when the Brev
# training image exposes it. Falling back to the host interpreter here can
# silently install a newer Torch/CUDA stack for evaluation than for training.
if [ -x /opt/conda/bin/python ]; then
    export PATH="/opt/conda/bin:$PATH"
fi

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
# Compiling every model in every short-lived comparison worker multiplies
# Inductor compiler pools by the worker count. Eager inference is the safe
# default; COMPARE_TORCH_COMPILE still carries any explicit caller override.
export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-0}"
export MHCFLURRY_MATMUL_PRECISION="${MHCFLURRY_MATMUL_PRECISION:-highest}"
export MHCFLURRY_ENABLE_TIMING="${MHCFLURRY_ENABLE_TIMING:-1}"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python-is-python3 python3-pip bzip2 wget rsync build-essential git \
    libhdf5-dev libxml2-dev libxslt1-dev procps

python -m pip install --upgrade pip
python -m pip install pypdf

rm -rf "$repo_dir" "$run_dir"
mkdir -p "$repo_dir" "$run_dir"
tar -C "$repo_dir" -xjf "$remote_root/repo.tar.bz2"
tar -C "$run_dir" -xjf "$remote_root/model_artifacts.tar.bz2"
if [ -f "$remote_root/paper_inputs.tar.bz2" ]; then
    rm -rf "$remote_root/paper_inputs"
    mkdir -p "$remote_root/paper_inputs"
    tar -C "$remote_root/paper_inputs" -xjf "$remote_root/paper_inputs.tar.bz2"
fi
if [ -f "$remote_root/data_dir.tar.bz2" ]; then
    rm -rf "$remote_root/data_dir"
    mkdir -p "$remote_root/data_dir"
    tar -C "$remote_root/data_dir" -xjf "$remote_root/data_dir.tar.bz2"
fi

cd "$repo_dir"
python -m pip install -e .

if [ -n "${DATA_DIR:-}" ]; then
    data_dir="$DATA_DIR"
    mhcflurry downloads fetch \
        models_class1_pan models_class1_processing models_class1_presentation
else
    mhcflurry downloads fetch \
        data_evaluation models_class1_pan \
        models_class1_processing models_class1_presentation
    data_dir="$(mhcflurry downloads path data_evaluation)"
fi
case ",${COMPARE_INCLUDE:-affinity,processing,presentation}," in
    *,affinity,*) mhcflurry downloads fetch models_class1_pan_variants ;;
esac
baseline_release="${COMPARE_BASELINE#public:}"
if [ "$baseline_release" != "$COMPARE_BASELINE" ]; then
    MHCFLURRY_DOWNLOADS_CURRENT_RELEASE="$baseline_release" \
        mhcflurry downloads fetch \
        models_class1_pan models_class1_processing models_class1_presentation
fi

compare_args=(
    mhcflurry eval compare-models
    --a "$run_dir" \
    --a-label "${RUN_LABEL:-new}" \
    --b "${COMPARE_BASELINE:-public:2.0.0}" \
    --b-label "${COMPARE_BASELINE_LABEL:-MHCflurry 2.0}" \
    --data-dir "$data_dir" \
    --release-holdout-dir "$run_dir/release_holdout" \
    --affinity-training-overlap-policy audit \
    --include "${COMPARE_INCLUDE:-affinity,processing,presentation}" \
    --processing-modes "${PROCESSING_MODES:-with_flanks,no_flank,short_flanks}" \
    --presentation-modes "${PRESENTATION_MODES:-with_flanks,without_flanks}" \
    --out "$run_dir/eval_comparison" \
    --backend "$COMPARE_BACKEND" \
    --num-jobs "$COMPARE_NUM_JOBS" \
    --max-workers-per-gpu "$COMPARE_MAX_WORKERS_PER_GPU" \
    --max-tasks-per-worker "$COMPARE_MAX_TASKS_PER_WORKER" \
    --presentation-num-jobs "$COMPARE_PRESENTATION_NUM_JOBS" \
    --presentation-max-workers-per-gpu "$COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU" \
    --presentation-max-tasks-per-worker "$COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER" \
    --presentation-torch-compile "$COMPARE_PRESENTATION_TORCH_COMPILE" \
    --worker-log-dir "$run_dir/eval_comparison/worker_logs" \
    --torch-compile "$COMPARE_TORCH_COMPILE" \
    --matmul-precision "$COMPARE_MATMUL_PRECISION"
)
if [ -n "${EVAL_MAX_BENCHMARK_FILES:-}" ]; then
    compare_args+=(--limit-files "$EVAL_MAX_BENCHMARK_FILES")
fi
case "$(printf '%s' "$COMPARE_GPUS" | tr '[:upper:]' '[:lower:]')" in
    auto) ;;
    *)
    compare_args+=(--gpus "$COMPARE_GPUS")
        ;;
esac
"${compare_args[@]}"

case ",${COMPARE_INCLUDE:-affinity,processing,presentation}," in
    *,affinity,*)
        train_excluded_affinity_dir="$(
            mhcflurry downloads path models_class1_pan_variants
        )/models.no_additional_ms"
        fair_affinity_args=(
            mhcflurry eval compare-models
            --a "$run_dir"
            --a-label "${RUN_LABEL:-new}"
            --b "$train_excluded_affinity_dir"
            --b-affinity-dir "$train_excluded_affinity_dir"
            --b-label "MHCflurry no-additional-MS (train-excluded)"
            --data-dir "$data_dir"
            --release-holdout-dir "$run_dir/release_holdout"
            --affinity-training-overlap-policy exclude
            --include affinity
            --affinity-source no_additional_ms
            --out "$run_dir/eval_comparison_train_excluded_affinity"
            --backend "$COMPARE_BACKEND"
            --num-jobs "$COMPARE_NUM_JOBS"
            --max-workers-per-gpu "$COMPARE_MAX_WORKERS_PER_GPU"
            --max-tasks-per-worker "$COMPARE_MAX_TASKS_PER_WORKER"
            --worker-log-dir \
                "$run_dir/eval_comparison_train_excluded_affinity/worker_logs"
            --torch-compile "$COMPARE_TORCH_COMPILE"
            --matmul-precision "$COMPARE_MATMUL_PRECISION"
        )
        if [ -n "${EVAL_MAX_BENCHMARK_FILES:-}" ]; then
            fair_affinity_args+=(--limit-files "$EVAL_MAX_BENCHMARK_FILES")
        fi
        case "$(printf '%s' "$COMPARE_GPUS" | tr '[:upper:]' '[:lower:]')" in
            auto) ;;
            *) fair_affinity_args+=(--gpus "$COMPARE_GPUS") ;;
        esac
        "${fair_affinity_args[@]}"
        ;;
esac

if [ "${RUN_RELEASE_PLOTS:-1}" = "1" ]; then
    plot_args=(
        mhcflurry eval plot-comparison
        --input "$run_dir/eval_comparison"
        --a-label "${RUN_LABEL:-new}"
        --b-label "${COMPARE_BASELINE_LABEL:-MHCflurry 2.0}"
        --summary-pdf "$run_dir/eval_comparison/plots/model_comparison_figures.pdf"
        --paper-figures-out "$run_dir/eval_comparison/plots/paper_figures"
        --paper-figures-formats "${PAPER_FIGURES_FORMATS:-svg,pdf,png}"
        --paper-figures-scores-dir "${PAPER_FIGURES_SCORES_DIR:-$run_dir/eval_comparison}"
    )
    if [ -n "${PAPER_FIGURES_MULTIALLELIC_PREDICTIONS:-}" ]; then
        plot_args+=(--paper-figures-multiallelic-predictions "$PAPER_FIGURES_MULTIALLELIC_PREDICTIONS")
    fi
    if [ -n "${PAPER_FIGURES_MONOALLELIC_PREDICTIONS:-}" ]; then
        plot_args+=(--paper-figures-monoallelic-predictions "$PAPER_FIGURES_MONOALLELIC_PREDICTIONS")
    fi
    if [ -n "${PAPER_FIGURES_CANDIDATE_PREDICTOR:-}" ]; then
        plot_args+=(--paper-figures-candidate-predictor "$PAPER_FIGURES_CANDIDATE_PREDICTOR")
    fi
    if [ -n "${PAPER_FIGURES_EXTERNAL_BASELINES:-}" ]; then
        plot_args+=(--paper-figures-external-baselines "$PAPER_FIGURES_EXTERNAL_BASELINES")
    fi
    if [ -n "${PAPER_FIGURES_PREFERRED_PREDICTORS:-}" ]; then
        plot_args+=(--paper-figures-preferred-predictors "$PAPER_FIGURES_PREFERRED_PREDICTORS")
    fi
    if [ -n "${PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS:-}" ]; then
        plot_args+=(--paper-figures-presentation-panel-predictors "$PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS")
    fi
    if [ -n "${PAPER_FIGURES_PRESENTATION_PANEL_BASELINES:-}" ]; then
        plot_args+=(--paper-figures-presentation-panel-baselines "$PAPER_FIGURES_PRESENTATION_PANEL_BASELINES")
    fi
    "${plot_args[@]}"
    if [ -d "$run_dir/eval_comparison_train_excluded_affinity" ]; then
        mhcflurry eval plot-comparison \
            --input "$run_dir/eval_comparison_train_excluded_affinity" \
            --a-label "${RUN_LABEL:-new}" \
            --b-label "MHCflurry no-additional-MS (train-excluded)" \
            --summary-pdf \
                "$run_dir/eval_comparison_train_excluded_affinity/plots/model_comparison_figures.pdf"
    fi
    mhcflurry train plot-loss-curves \
        --selected-dir "$run_dir/affinity/models.combined" \
        --unselected-dir "$run_dir/affinity/models.unselected.combined" \
        --out "$run_dir/affinity/loss_plots"
fi
EOF
    } > "$remote_script"
    chmod +x "$remote_script"

    run_logged_step_with_timeout \
        postprocess_run_remote "$BREV_POSTPROCESS_TIMEOUT_SECONDS" \
        brev exec "$BREV_INSTANCE" "@$remote_script"

    cat > "$remote_sync_script" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
remote_root=/root/mhcflurry-postprocess
run_dir="$remote_root/run"
archive="$remote_root/postprocess_sync.tar.bz2"
manifest="$remote_root/postprocess_sync_paths.txt"
rm -f "$archive" "$manifest"
cd "$run_dir"

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
    return 0
}

add_path eval_comparison/release_summary.csv
add_path eval_comparison/release_summary.md
add_path eval_comparison/summary.md
add_path eval_comparison/side_a.json
add_path eval_comparison/side_b.json
add_path eval_comparison/plots
add_path affinity/loss_plots
add_glob eval_comparison/*/summary.json
add_glob eval_comparison/*/training_overlap.json
add_glob eval_comparison/*/summary_table.csv
add_glob eval_comparison/*/per_*.csv
add_glob eval_comparison/*/predictions*.csv.bz2

add_path eval_comparison_train_excluded_affinity/release_summary.csv
add_path eval_comparison_train_excluded_affinity/release_summary.md
add_path eval_comparison_train_excluded_affinity/summary.md
add_path eval_comparison_train_excluded_affinity/side_a.json
add_path eval_comparison_train_excluded_affinity/side_b.json
add_path eval_comparison_train_excluded_affinity/plots
add_glob eval_comparison_train_excluded_affinity/*/summary.json
add_glob eval_comparison_train_excluded_affinity/*/training_overlap.json
add_glob eval_comparison_train_excluded_affinity/*/per_*.csv
add_glob eval_comparison_train_excluded_affinity/*/predictions*.csv.bz2

sort -u "$manifest" -o "$manifest"
tar -cjf "$archive" -T "$manifest"
printf 'postprocess sync manifest:\n'
cat "$manifest"
du -sh "$archive"
EOF
    chmod +x "$remote_sync_script"

    run_logged_step_with_timeout \
        postprocess_prepare_sync "$BREV_CREATE_TIMEOUT_SECONDS" \
        brev exec "$BREV_INSTANCE" "@$remote_sync_script"
    run_logged_step_with_timeout \
        postprocess_copy_outputs "$BREV_CREATE_TIMEOUT_SECONDS" \
        brev copy "$BREV_INSTANCE:$remote_archive" "$staging/"
    run_logged_step postprocess_extract_outputs \
        tar -C "$RUN_DIR" -xjf "$local_archive"
    run_cmd rm -rf "$staging"

    BREV_REMOTE_EVAL_DONE=1
    if [ "$SKIP_PLOTS" != "1" ] && [ -d "$RUN_DIR/eval_comparison/plots" ]; then
        BREV_REMOTE_PLOTS_DONE=1
    fi
    apply_brev_cleanup
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
    local remote_workflow_id
    remote_workflow_id="$(brev_latest_remote_workflow_id)"
    if [ "$remote_workflow_id" != "$WORKFLOW_RUN_ID" ]; then
        die "Refusing to sync Brev output for workflow '${remote_workflow_id:-unknown}'; expected '$WORKFLOW_RUN_ID'"
    fi
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
    run_dir_matches_workflow || \
        die "Brev sync finished but its local workflow marker does not match $WORKFLOW_RUN_ID"
}

sync_brev_full_output() {
    local sync_parent="$1"
    local copied_out="$sync_parent/out"
    set +e
    run_logged_step_with_timeout \
        brev_sync_copy "$BREV_CREATE_TIMEOUT_SECONDS" \
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
    return 0
}

add_path .runplz/events.ndjson
add_path .runplz/heartbeat.ndjson
add_path .runplz/last.log
add_path .runplz/run.json
add_path .runplz/run.sh
add_path .runplz/run_driver.log
add_path .runplz/mhcflurry_release_workflow_id
add_path .runplz/mhcflurry_release_workflow_exit_code

add_path eval_comparison/release_summary.csv
add_path eval_comparison/release_summary.md
add_path eval_comparison/summary.md
add_path eval_comparison/side_a.json
add_path eval_comparison/side_b.json
add_path eval_comparison/plots
add_glob eval_comparison/*/summary.json
add_glob eval_comparison/*/training_overlap.json
add_glob eval_comparison/*/summary_table.csv
add_glob eval_comparison/*/per_*.csv
add_glob eval_comparison/*/predictions*.csv.bz2

add_path eval_comparison_train_excluded_affinity/release_summary.csv
add_path eval_comparison_train_excluded_affinity/release_summary.md
add_path eval_comparison_train_excluded_affinity/summary.md
add_path eval_comparison_train_excluded_affinity/side_a.json
add_path eval_comparison_train_excluded_affinity/side_b.json
add_path eval_comparison_train_excluded_affinity/plots
add_glob eval_comparison_train_excluded_affinity/*/summary.json
add_glob eval_comparison_train_excluded_affinity/*/training_overlap.json
add_glob eval_comparison_train_excluded_affinity/*/per_*.csv
add_glob eval_comparison_train_excluded_affinity/*/predictions*.csv.bz2

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

add_path release_holdout

add_glob processing/models.selected.*
add_path processing/hits_with_tpm.csv.bz2
add_path processing/gpu_occupancy.csv
add_path processing/hyperparameters.base.yaml
add_glob processing/hyperparameters.*.yaml
add_path processing/train_data.csv.bz2
add_glob processing/LOG-worker.*.txt

add_path presentation/models
add_path presentation/gpu_occupancy.csv
add_path presentation/make_train_data.presentation.py
add_path presentation/train_data.csv.bz2

sort -u "$manifest" -o "$manifest"
tar -cjf "$archive" -T "$manifest"
printf 'release sync manifest:\n'
cat "$manifest"
du -sh "$archive"
EOF
    chmod +x "$sync_script"

    set +e
    run_logged_step_with_timeout \
        brev_sync_prepare_release_archive "$BREV_CREATE_TIMEOUT_SECONDS" \
        brev exec "$BREV_INSTANCE" "@$sync_script"
    local prepare_status=$?
    set -e
    if [ "$prepare_status" -ne 0 ]; then
        return "$prepare_status"
    fi

    set +e
    run_logged_step_with_timeout \
        brev_sync_copy_release_archive "$BREV_CREATE_TIMEOUT_SECONDS" \
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
            run_logged_step_with_timeout \
                brev_stop "$BREV_CLEANUP_TIMEOUT_SECONDS" \
                brev stop "$BREV_INSTANCE"
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
                    set +e
                    run_logged_step_with_timeout \
                        brev_delete_after_failed_stop \
                        "$BREV_CLEANUP_TIMEOUT_SECONDS" \
                        brev delete "$BREV_INSTANCE"
                    local delete_status=$?
                    set -e
                    if [ "$delete_status" -ne 0 ]; then
                        warn "brev delete failed for $BREV_INSTANCE with status $delete_status"
                    fi
                else
                    warn "Leaving instance running; rerun 'brev stop $BREV_INSTANCE' or delete it manually."
                fi
            else
                note "Brev instance status after stop: ${status:-unknown}"
            fi
            ;;
        delete)
            set +e
            run_logged_step_with_timeout \
                brev_delete "$BREV_CLEANUP_TIMEOUT_SECONDS" \
                brev delete "$BREV_INSTANCE"
            local delete_status=$?
            set -e
            if [ "$delete_status" -ne 0 ]; then
                warn "brev delete failed for $BREV_INSTANCE with status $delete_status"
            fi
            ;;
    esac
}

run_brev_training() {
    local auto_create=$1
    if [ "$DRY_RUN" != "1" ]; then
        require_command runplz
        require_clean_runplz_3160
    fi
    run_cmd mkdir -p "$RUN_DIR"
    local runplz_on_finish=leave
    local run_release_eval=0
    local run_release_plots=0
    local run_remote_paper_postprocess=0
    if [ "$SKIP_EVAL" != "1" ]; then
        if [ -n "$DATA_DIR" ]; then
            note "Custom DATA_DIR is local to this wrapper; evaluation will run after Brev sync."
        else
            run_release_eval=1
        fi
    fi
    if [ "$SKIP_PLOTS" != "1" ]; then
        if paper_figure_inputs_requested; then
            note "Paper-figure inputs are local to this wrapper; plotting will run in a Brev postprocess after sync."
            run_remote_paper_postprocess=1
        elif [ "$run_release_eval" = "1" ]; then
            run_release_plots=1
        fi
    fi
    BREV_EXPECT_REMOTE_EVAL=$run_release_eval
    BREV_EXPECT_REMOTE_PLOTS=$run_release_plots
    local remote_paper_scores_dir=
    local remote_paper_artifacts_dir=
    local remote_paper_multiallelic_predictions=
    local remote_paper_monoallelic_predictions=
    if [ "$run_release_plots" = "1" ]; then
        remote_paper_scores_dir=$PAPER_FIGURES_SCORES_DIR
        remote_paper_artifacts_dir=$PAPER_FIGURES_ARTIFACTS_DIR
        remote_paper_multiallelic_predictions=$PAPER_FIGURES_MULTIALLELIC_PREDICTIONS
        remote_paper_monoallelic_predictions=$PAPER_FIGURES_MONOALLELIC_PREDICTIONS
    fi
    local runplz_env=(
        "MHCFLURRY_OUT=$RUN_DIR"
        "REPO=$REPO"
        "TRAINING_MINIBATCH_SIZE=$TRAINING_MINIBATCH_SIZE"
        "AFFINITY_MINIBATCH_SIZE=$AFFINITY_MINIBATCH_SIZE"
        "AFFINITY_MAX_WORKERS_PER_GPU=$AFFINITY_MAX_WORKERS_PER_GPU"
        "PROCESSING_MINIBATCH_SIZE=$PROCESSING_MINIBATCH_SIZE"
        "PROCESSING_NUM_JOBS=$PROCESSING_NUM_JOBS"
        "PROCESSING_MAX_WORKERS_PER_GPU=$PROCESSING_MAX_WORKERS_PER_GPU"
        "PROCESSING_HELD_OUT_SAMPLES=$PROCESSING_HELD_OUT_SAMPLES"
        "PROCESSING_VARIANTS=$PROCESSING_VARIANTS"
        "PRESENTATION_PROCESSING_WITH_FLANKS_KIND=$PRESENTATION_PROCESSING_WITH_FLANKS_KIND"
        "PRESENTATION_DECOYS_PER_HIT=$PRESENTATION_DECOYS_PER_HIT"
        "PRESENTATION_SAMPLE_FRACTION=$PRESENTATION_SAMPLE_FRACTION"
        "PRESENTATION_FEATURE_CHUNK_SIZE=$PRESENTATION_FEATURE_CHUNK_SIZE"
        "PRESENTATION_NUM_JOBS=$PRESENTATION_NUM_JOBS"
        "PRESENTATION_MAX_WORKERS_PER_GPU=$PRESENTATION_MAX_WORKERS_PER_GPU"
        "PRESENTATION_CALIBRATION_NUM_JOBS=$PRESENTATION_CALIBRATION_NUM_JOBS"
        "PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU=$PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU"
        "PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE=$PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE"
        "RUN_RELEASE_EVAL=$run_release_eval"
        "RUN_RELEASE_PLOTS=$run_release_plots"
        "COMPARE_INCLUDE=$COMPARE_INCLUDE"
        "EVAL_MAX_BENCHMARK_FILES=$EVAL_MAX_BENCHMARK_FILES"
        "COMPARE_BASELINE=$COMPARE_BASELINE"
        "COMPARE_BASELINE_LABEL=$COMPARE_BASELINE_LABEL"
        "COMPARE_BACKEND=$COMPARE_BACKEND"
        "COMPARE_NUM_JOBS=$COMPARE_NUM_JOBS"
        "COMPARE_MAX_WORKERS_PER_GPU=$COMPARE_MAX_WORKERS_PER_GPU"
        "COMPARE_MAX_TASKS_PER_WORKER=$COMPARE_MAX_TASKS_PER_WORKER"
        "COMPARE_TORCH_COMPILE=$COMPARE_TORCH_COMPILE"
        "COMPARE_MATMUL_PRECISION=$COMPARE_MATMUL_PRECISION"
        "COMPARE_GPUS=$COMPARE_GPUS"
        "DATA_DIR="
        "COMPARE_PRESENTATION_NUM_JOBS=$COMPARE_PRESENTATION_NUM_JOBS"
        "COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU=$COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU"
        "COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER=$COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER"
        "COMPARE_PRESENTATION_TORCH_COMPILE=$COMPARE_PRESENTATION_TORCH_COMPILE"
        "MHCFLURRY_TORCH_COMPILE=$MHCFLURRY_TORCH_COMPILE"
        "MHCFLURRY_TORCH_COMPILE_LOSS=$MHCFLURRY_TORCH_COMPILE_LOSS"
        "MHCFLURRY_MATMUL_PRECISION=$MHCFLURRY_MATMUL_PRECISION"
        "MATMUL_PRECISION=$MATMUL_PRECISION"
        "MATMUL_PRECISION_CLI=$MATMUL_PRECISION_CLI"
        "PROCESSING_MODES=$PROCESSING_MODES"
        "PRESENTATION_MODES=$PRESENTATION_MODES"
        "PAPER_FIGURES_SCORES_DIR=$remote_paper_scores_dir"
        "PAPER_FIGURES_ARTIFACTS_DIR=$remote_paper_artifacts_dir"
        "PAPER_FIGURES_MULTIALLELIC_PREDICTIONS=$remote_paper_multiallelic_predictions"
        "PAPER_FIGURES_MONOALLELIC_PREDICTIONS=$remote_paper_monoallelic_predictions"
        "PAPER_FIGURES_FORMATS=$PAPER_FIGURES_FORMATS"
        "PAPER_FIGURES_CANDIDATE_PREDICTOR=$PAPER_FIGURES_CANDIDATE_PREDICTOR"
        "PAPER_FIGURES_EXTERNAL_BASELINES=$PAPER_FIGURES_EXTERNAL_BASELINES"
        "PAPER_FIGURES_PREFERRED_PREDICTORS=$PAPER_FIGURES_PREFERRED_PREDICTORS"
        "PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS=$PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS"
        "PAPER_FIGURES_PRESENTATION_PANEL_BASELINES=$PAPER_FIGURES_PRESENTATION_PANEL_BASELINES"
        "RUN_LABEL=$RUN_LABEL"
        "MHCFLURRY_RELEASE_WORKFLOW_ID=$WORKFLOW_RUN_ID"
        "MHCFLURRY_RELEASE_GIT_COMMIT=$RELEASE_GIT_COMMIT"
        "MHCFLURRY_RELEASE_VERSION=$RELEASE"
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

    if [ "$DRY_RUN" = "1" ]; then
        if [ "$run_remote_paper_postprocess" = "1" ]; then
            wait_paper_figures_prepare
            run_brev_postprocess 0
        else
            if [ "$run_release_eval" = "1" ]; then
                BREV_REMOTE_EVAL_DONE=1
            fi
            if [ "$run_release_plots" = "1" ]; then
                BREV_REMOTE_PLOTS_DONE=1
            fi
        fi
        return 0
    fi

    local remote_exit
    remote_exit="$(brev_latest_remote_exit_code || true)"
    if [ "$runplz_status" -ne 0 ]; then
        if [ "$remote_exit" = "0" ]; then
            warn "runplz exited with $runplz_status, but remote command exit_code=0; continuing after explicit sync."
        else
            warn "runplz exited with $runplz_status; remote exit_code=${remote_exit:-unknown}."
            local remote_status
            remote_status="$(brev_instance_status || true)"
            if [ -z "$remote_status" ]; then
                warn "Brev instance $BREV_INSTANCE does not exist; skipping sync and cleanup."
                return "$runplz_status"
            fi
            sync_brev_output || {
                warn "Brev output sync failed; leaving $BREV_INSTANCE available to preserve artifacts."
                return "$runplz_status"
            }
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
    if [ "$run_remote_paper_postprocess" = "1" ]; then
        set +e
        wait_paper_figures_prepare
        local prepare_status=$?
        set -e
        if [ "$prepare_status" -ne 0 ]; then
            warn "Paper-figure input preparation failed; applying Brev cleanup without remote paper plotting."
            apply_brev_cleanup
            return "$prepare_status"
        fi
        run_brev_postprocess 0
        return
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
BREV_PROVIDER_EXPLICIT=0
if [ -n "${RUNPLZ_BREV_PROVIDER:-}" ] || [ -n "${BREV_PROVIDER:-}" ]; then
    BREV_PROVIDER_EXPLICIT=1
fi
BREV_PROVIDER="${RUNPLZ_BREV_PROVIDER:-${BREV_PROVIDER:-auto}}"
BREV_INSTANCE_TYPE_EXPLICIT=0
if [ -n "${RUNPLZ_BREV_INSTANCE_TYPE:-}" ] || \
        [ -n "${BREV_INSTANCE_TYPE:-}" ]; then
    BREV_INSTANCE_TYPE_EXPLICIT=1
fi
BREV_INSTANCE_TYPE="${RUNPLZ_BREV_INSTANCE_TYPE:-${BREV_INSTANCE_TYPE:-}}"
DEFAULT_BREV_PROVISION_INSTANCE_TYPE="${DEFAULT_BREV_PROVISION_INSTANCE_TYPE:-}"
BREV_CONTAINER_IMAGE="${BREV_CONTAINER_IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime}"
BREV_MAX_RUNTIME_SECONDS="${RUNPLZ_BREV_MAX_RUNTIME_SECONDS:-${BREV_MAX_RUNTIME_SECONDS:-}}"
RUNPLZ_REQUIRED_VERSION="3.16.0"
BREV_INSTANCE_TYPE_FALLBACK_COUNT="${RUNPLZ_BREV_INSTANCE_TYPE_FALLBACK_COUNT:-3}"
BREV_EXCLUDE_PROVIDERS="${RUNPLZ_BREV_EXCLUDE_PROVIDERS:-oci}"
BREV_STOP_FAILURE_ACTION_EXPLICIT=0
if [ -n "${BREV_STOP_FAILURE_ACTION:-}" ]; then
    BREV_STOP_FAILURE_ACTION_EXPLICIT=1
fi
BREV_STOP_FAILURE_ACTION="${BREV_STOP_FAILURE_ACTION:-}"
BREV_CLEANUP_TIMEOUT_SECONDS="${BREV_CLEANUP_TIMEOUT_SECONDS:-60}"
BREV_CREATE_TIMEOUT_SECONDS="${BREV_CREATE_TIMEOUT_SECONDS:-2400}"
BREV_POSTPROCESS_TIMEOUT_SECONDS="${BREV_POSTPROCESS_TIMEOUT_SECONDS:-86400}"
BREV_SHELL_READY_ATTEMPTS="${BREV_SHELL_READY_ATTEMPTS:-40}"
BREV_SHELL_READY_DELAY_SECONDS="${BREV_SHELL_READY_DELAY_SECONDS:-15}"
BREV_SYNC_MODE="${BREV_SYNC_MODE:-release}"
SKIP_TRAIN=0
SKIP_EVAL=0
SKIP_PLOTS=0
SKIP_DEPLOY=0
DEPLOY_MODE=none
RELEASE_PROFILE="${RELEASE_PROFILE:-full}"
DATA_DIR=
COMPARE_INCLUDE=affinity,processing,presentation
EVAL_MAX_BENCHMARK_FILES="${EVAL_MAX_BENCHMARK_FILES:-}"
PROCESSING_MODES_EXPLICIT=0
if [ -n "${PROCESSING_MODES:-}" ]; then
    PROCESSING_MODES_EXPLICIT=1
fi
PROCESSING_MODES="${PROCESSING_MODES:-with_flanks,no_flank,short_flanks}"
PRESENTATION_MODES=with_flanks,without_flanks
COMPARE_BACKEND="${COMPARE_BACKEND:-auto}"
COMPARE_NUM_JOBS="${COMPARE_NUM_JOBS:-auto}"
COMPARE_MAX_WORKERS_PER_GPU="${COMPARE_MAX_WORKERS_PER_GPU:-auto}"
COMPARE_MAX_TASKS_PER_WORKER="${COMPARE_MAX_TASKS_PER_WORKER:-12}"
MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-0}"
MHCFLURRY_TORCH_COMPILE_LOSS="${MHCFLURRY_TORCH_COMPILE_LOSS:-0}"
MHCFLURRY_MATMUL_PRECISION="${MHCFLURRY_MATMUL_PRECISION:-highest}"
MATMUL_PRECISION="${MATMUL_PRECISION:-highest}"
MATMUL_PRECISION_CLI="${MATMUL_PRECISION_CLI:-highest}"
COMPARE_TORCH_COMPILE="${COMPARE_TORCH_COMPILE:-$MHCFLURRY_TORCH_COMPILE}"
COMPARE_MATMUL_PRECISION="${COMPARE_MATMUL_PRECISION:-$MHCFLURRY_MATMUL_PRECISION}"
COMPARE_GPUS="${COMPARE_GPUS:-auto}"
COMPARE_PRESENTATION_NUM_JOBS="${COMPARE_PRESENTATION_NUM_JOBS:-auto}"
COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU="${COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU:-auto}"
COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER="${COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER:-1}"
COMPARE_PRESENTATION_TORCH_COMPILE="${COMPARE_PRESENTATION_TORCH_COMPILE:-0}"
COMPARE_BASELINE="${COMPARE_BASELINE:-public:2.0.0}"
COMPARE_BASELINE_LABEL="${COMPARE_BASELINE_LABEL:-}"
PAPER_FIGURES_ARTIFACTS_DIR="${PAPER_FIGURES_ARTIFACTS_DIR:-}"
PAPER_FIGURES_SCORES_DIR="${PAPER_FIGURES_SCORES_DIR:-$PAPER_FIGURES_ARTIFACTS_DIR}"
PAPER_FIGURES_MULTIALLELIC_PREDICTIONS="${PAPER_FIGURES_MULTIALLELIC_PREDICTIONS:-}"
PAPER_FIGURES_MONOALLELIC_PREDICTIONS="${PAPER_FIGURES_MONOALLELIC_PREDICTIONS:-}"
PAPER_FIGURES_PREPARE_COMMAND="${PAPER_FIGURES_PREPARE_COMMAND:-}"
PAPER_FIGURES_PREPARE_PID=
PAPER_FIGURES_PREPARE_DONE=0
PAPER_FIGURES_FORMATS="${PAPER_FIGURES_FORMATS:-svg,pdf,png}"
PAPER_FIGURES_CANDIDATE_PREDICTOR="${PAPER_FIGURES_CANDIDATE_PREDICTOR:-}"
PAPER_FIGURES_EXTERNAL_BASELINES="${PAPER_FIGURES_EXTERNAL_BASELINES:-}"
PAPER_FIGURES_PREFERRED_PREDICTORS="${PAPER_FIGURES_PREFERRED_PREDICTORS:-}"
PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS="${PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS:-}"
PAPER_FIGURES_PRESENTATION_PANEL_BASELINES="${PAPER_FIGURES_PRESENTATION_PANEL_BASELINES:-}"
RUN_LABEL="${RUN_LABEL:-}"
DRY_RUN=0
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
TRAINING_MINIBATCH_SIZE="${TRAINING_MINIBATCH_SIZE:-1024}"
AFFINITY_MINIBATCH_SIZE=
AFFINITY_MAX_WORKERS_PER_GPU_EXPLICIT=0
if [ -n "${AFFINITY_MAX_WORKERS_PER_GPU:-}" ]; then
    AFFINITY_MAX_WORKERS_PER_GPU_EXPLICIT=1
fi
AFFINITY_MAX_WORKERS_PER_GPU="${AFFINITY_MAX_WORKERS_PER_GPU:-auto}"
PROCESSING_MINIBATCH_SIZE=
PROCESSING_NUM_JOBS="${PROCESSING_NUM_JOBS:-auto}"
PROCESSING_MAX_WORKERS_PER_GPU="${PROCESSING_MAX_WORKERS_PER_GPU:-auto}"
PROCESSING_HELD_OUT_SAMPLES="${PROCESSING_HELD_OUT_SAMPLES:-10}"
PROCESSING_VARIANTS_EXPLICIT=0
if [ -n "${PROCESSING_VARIANTS:-}" ]; then
    PROCESSING_VARIANTS_EXPLICIT=1
fi
PROCESSING_VARIANTS="${PROCESSING_VARIANTS:-with_flanks no_flank short_flanks}"
PRESENTATION_PROCESSING_WITH_FLANKS_KIND_EXPLICIT=0
if [ -n "${PRESENTATION_PROCESSING_WITH_FLANKS_KIND:-}" ]; then
    PRESENTATION_PROCESSING_WITH_FLANKS_KIND_EXPLICIT=1
fi
PRESENTATION_PROCESSING_WITH_FLANKS_KIND="${PRESENTATION_PROCESSING_WITH_FLANKS_KIND:-short_flanks}"
PRESENTATION_DECOYS_PER_HIT="${PRESENTATION_DECOYS_PER_HIT:-2}"
PRESENTATION_SAMPLE_FRACTION="${PRESENTATION_SAMPLE_FRACTION:-0.1}"
PRESENTATION_FEATURE_CHUNK_SIZE="${PRESENTATION_FEATURE_CHUNK_SIZE:-250000}"
PRESENTATION_NUM_JOBS="${PRESENTATION_NUM_JOBS:-auto}"
PRESENTATION_MAX_WORKERS_PER_GPU="${PRESENTATION_MAX_WORKERS_PER_GPU:-auto}"
PRESENTATION_CALIBRATION_NUM_JOBS="${PRESENTATION_CALIBRATION_NUM_JOBS:-auto}"
PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU="${PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU:-auto}"
PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE="${PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE:-auto}"
WORKFLOW_LOG_DIR=
WORKFLOW_STATUS_LOG=
WORKFLOW_RUN_ID="${WORKFLOW_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RELEASE_GIT_COMMIT=
BREV_EXPECT_REMOTE_EVAL=0
BREV_EXPECT_REMOTE_PLOTS=0
BREV_REMOTE_EVAL_DONE=0
BREV_REMOTE_PLOTS_DONE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

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
        --release-profile)
            RELEASE_PROFILE=$2
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
        --brev-provider)
            BREV_PROVIDER=$2
            BREV_PROVIDER_EXPLICIT=1
            shift 2
            ;;
        --brev-instance-type)
            BREV_INSTANCE_TYPE=$2
            BREV_INSTANCE_TYPE_EXPLICIT=1
            shift 2
            ;;
        --brev-stop-failure-action)
            BREV_STOP_FAILURE_ACTION=$2
            BREV_STOP_FAILURE_ACTION_EXPLICIT=1
            shift 2
            ;;
        --brev-cleanup-timeout-seconds)
            BREV_CLEANUP_TIMEOUT_SECONDS=$2
            shift 2
            ;;
        --brev-create-timeout-seconds)
            BREV_CREATE_TIMEOUT_SECONDS=$2
            shift 2
            ;;
        --brev-container-image)
            BREV_CONTAINER_IMAGE=$2
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
        --eval-max-benchmark-files)
            EVAL_MAX_BENCHMARK_FILES=$2
            shift 2
            ;;
        --presentation-modes)
            PRESENTATION_MODES=$2
            shift 2
            ;;
        --compare-presentation-num-jobs)
            COMPARE_PRESENTATION_NUM_JOBS=$2
            shift 2
            ;;
        --compare-presentation-max-workers-per-gpu)
            COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU=$2
            shift 2
            ;;
        --compare-presentation-max-tasks-per-worker)
            COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER=$2
            shift 2
            ;;
        --compare-presentation-torch-compile)
            COMPARE_PRESENTATION_TORCH_COMPILE=$2
            shift 2
            ;;
        --compare-baseline)
            COMPARE_BASELINE=$2
            shift 2
            ;;
        --compare-baseline-label)
            COMPARE_BASELINE_LABEL=$2
            shift 2
            ;;
        --compare-gpus)
            COMPARE_GPUS=$2
            shift 2
            ;;
        --paper-figures-scores-dir)
            PAPER_FIGURES_SCORES_DIR=$2
            shift 2
            ;;
        --paper-figures-artifacts-dir)
            PAPER_FIGURES_ARTIFACTS_DIR=$2
            if [ -z "$PAPER_FIGURES_SCORES_DIR" ]; then
                PAPER_FIGURES_SCORES_DIR=$2
            fi
            shift 2
            ;;
        --paper-figures-multiallelic-predictions)
            PAPER_FIGURES_MULTIALLELIC_PREDICTIONS=$2
            shift 2
            ;;
        --paper-figures-monoallelic-predictions)
            PAPER_FIGURES_MONOALLELIC_PREDICTIONS=$2
            shift 2
            ;;
        --paper-figures-prepare-command)
            PAPER_FIGURES_PREPARE_COMMAND=$2
            shift 2
            ;;
        --paper-figures-candidate-predictor)
            PAPER_FIGURES_CANDIDATE_PREDICTOR=$2
            shift 2
            ;;
        --paper-figures-external-baselines)
            PAPER_FIGURES_EXTERNAL_BASELINES=$2
            shift 2
            ;;
        --paper-figures-preferred-predictors)
            PAPER_FIGURES_PREFERRED_PREDICTORS=$2
            shift 2
            ;;
        --paper-figures-presentation-panel-predictors)
            PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS=$2
            shift 2
            ;;
        --paper-figures-presentation-panel-baselines)
            PAPER_FIGURES_PRESENTATION_PANEL_BASELINES=$2
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
        --allow-dirty-repo)
            ALLOW_DIRTY_REPO=1
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
            AFFINITY_MAX_WORKERS_PER_GPU_EXPLICIT=1
            shift 2
            ;;
        --processing-minibatch-size)
            PROCESSING_MINIBATCH_SIZE=$2
            shift 2
            ;;
        --processing-num-jobs)
            PROCESSING_NUM_JOBS=$2
            shift 2
            ;;
        --processing-max-workers-per-gpu)
            PROCESSING_MAX_WORKERS_PER_GPU=$2
            shift 2
            ;;
        --processing-held-out-samples)
            PROCESSING_HELD_OUT_SAMPLES=$2
            shift 2
            ;;
        --processing-variants)
            PROCESSING_VARIANTS=$2
            PROCESSING_VARIANTS_EXPLICIT=1
            shift 2
            ;;
        --presentation-processing-with-flanks-kind)
            PRESENTATION_PROCESSING_WITH_FLANKS_KIND=$2
            PRESENTATION_PROCESSING_WITH_FLANKS_KIND_EXPLICIT=1
            shift 2
            ;;
        --presentation-decoys-per-hit)
            PRESENTATION_DECOYS_PER_HIT=$2
            shift 2
            ;;
        --presentation-sample-fraction)
            PRESENTATION_SAMPLE_FRACTION=$2
            shift 2
            ;;
        --presentation-feature-chunk-size)
            PRESENTATION_FEATURE_CHUNK_SIZE=$2
            shift 2
            ;;
        --presentation-num-jobs)
            PRESENTATION_NUM_JOBS=$2
            shift 2
            ;;
        --presentation-max-workers-per-gpu)
            PRESENTATION_MAX_WORKERS_PER_GPU=$2
            shift 2
            ;;
        --presentation-calibration-num-jobs)
            PRESENTATION_CALIBRATION_NUM_JOBS=$2
            shift 2
            ;;
        --presentation-calibration-max-workers-per-gpu)
            PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU=$2
            shift 2
            ;;
        --presentation-calibration-prediction-batch-size)
            PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE=$2
            shift 2
            ;;
        --processing-modes)
            PROCESSING_MODES=$2
            PROCESSING_MODES_EXPLICIT=1
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
if [ -z "$RUN_LABEL" ]; then
    RUN_LABEL="MHCflurry $(display_release_version "$RELEASE")"
fi
if [ -z "$COMPARE_BASELINE_LABEL" ]; then
    baseline_release="$(public_release_from_spec "$COMPARE_BASELINE")"
    if [ -n "$baseline_release" ]; then
        COMPARE_BASELINE_LABEL="MHCflurry $(display_release_version "$baseline_release")"
    elif [ "$COMPARE_BASELINE" = "public" ]; then
        COMPARE_BASELINE_LABEL="MHCflurry public"
    else
        COMPARE_BASELINE_LABEL="$(basename "$COMPARE_BASELINE")"
    fi
fi
if [ -z "$AFFINITY_MINIBATCH_SIZE" ]; then
    AFFINITY_MINIBATCH_SIZE=$TRAINING_MINIBATCH_SIZE
fi
if [ -z "$PROCESSING_MINIBATCH_SIZE" ]; then
    PROCESSING_MINIBATCH_SIZE=512
fi
apply_release_profile
validate_processing_configuration
validate_positive_integer "--affinity-minibatch-size" "$AFFINITY_MINIBATCH_SIZE"
validate_positive_integer "--processing-minibatch-size" "$PROCESSING_MINIBATCH_SIZE"
validate_positive_integer "--processing-held-out-samples" "$PROCESSING_HELD_OUT_SAMPLES"
validate_positive_number "--presentation-decoys-per-hit" "$PRESENTATION_DECOYS_PER_HIT"
validate_fraction "--presentation-sample-fraction" "$PRESENTATION_SAMPLE_FRACTION"
validate_positive_integer "--presentation-feature-chunk-size" "$PRESENTATION_FEATURE_CHUNK_SIZE"
validate_auto_or_positive_integer "--affinity-max-workers-per-gpu" "$AFFINITY_MAX_WORKERS_PER_GPU"
validate_auto_or_nonnegative_integer "--processing-num-jobs" "$PROCESSING_NUM_JOBS"
validate_auto_or_positive_integer "--processing-max-workers-per-gpu" "$PROCESSING_MAX_WORKERS_PER_GPU"
validate_auto_or_nonnegative_integer "--presentation-num-jobs" "$PRESENTATION_NUM_JOBS"
validate_auto_or_positive_integer "--presentation-max-workers-per-gpu" "$PRESENTATION_MAX_WORKERS_PER_GPU"
validate_auto_or_nonnegative_integer "--presentation-calibration-num-jobs" "$PRESENTATION_CALIBRATION_NUM_JOBS"
validate_auto_or_positive_integer "--presentation-calibration-max-workers-per-gpu" "$PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU"
validate_auto_or_positive_integer "--presentation-calibration-prediction-batch-size" "$PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE"
validate_auto_or_nonnegative_integer "COMPARE_NUM_JOBS" "$COMPARE_NUM_JOBS"
validate_auto_or_positive_integer "COMPARE_MAX_WORKERS_PER_GPU" "$COMPARE_MAX_WORKERS_PER_GPU"
validate_positive_integer "COMPARE_MAX_TASKS_PER_WORKER" "$COMPARE_MAX_TASKS_PER_WORKER"
validate_auto_or_nonnegative_integer "COMPARE_PRESENTATION_NUM_JOBS" "$COMPARE_PRESENTATION_NUM_JOBS"
validate_auto_or_positive_integer "COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU" "$COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU"
validate_positive_integer "COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER" "$COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER"
validate_positive_number "--brev-cleanup-timeout-seconds" "$BREV_CLEANUP_TIMEOUT_SECONDS"
validate_positive_number "--brev-create-timeout-seconds" "$BREV_CREATE_TIMEOUT_SECONDS"
validate_positive_number "BREV_POSTPROCESS_TIMEOUT_SECONDS" "$BREV_POSTPROCESS_TIMEOUT_SECONDS"
validate_positive_integer "BREV_SHELL_READY_ATTEMPTS" "$BREV_SHELL_READY_ATTEMPTS"
validate_positive_number "BREV_SHELL_READY_DELAY_SECONDS" "$BREV_SHELL_READY_DELAY_SECONDS"
validate_nonnegative_integer "BREV_INSTANCE_TYPE_FALLBACK_COUNT" "$BREV_INSTANCE_TYPE_FALLBACK_COUNT"
if [ -n "$BREV_MAX_RUNTIME_SECONDS" ]; then
    validate_positive_number "--brev-max-runtime-seconds" "$BREV_MAX_RUNTIME_SECONDS"
fi
COMPARE_TORCH_COMPILE="$(normalize_compare_torch_compile "$COMPARE_TORCH_COMPILE")"
COMPARE_PRESENTATION_TORCH_COMPILE="$(normalize_compare_torch_compile "$COMPARE_PRESENTATION_TORCH_COMPILE")"
COMPARE_MATMUL_PRECISION="$(normalize_compare_matmul_precision "$COMPARE_MATMUL_PRECISION")"
validate_compare_gpus "$COMPARE_GPUS"
if [ -n "$EVAL_MAX_BENCHMARK_FILES" ]; then
    case "$EVAL_MAX_BENCHMARK_FILES" in
        ''|*[!0-9]*)
            die "--eval-max-benchmark-files must be a positive integer; got '$EVAL_MAX_BENCHMARK_FILES'"
            ;;
        0)
            die "--eval-max-benchmark-files must be a positive integer; got '$EVAL_MAX_BENCHMARK_FILES'"
            ;;
    esac
fi
case "$BACKEND" in
    local|brev-existing|brev-provision|ssh) ;;
    *) die "--backend must be one of: local, brev-existing, brev-provision, ssh" ;;
esac
case "$DEPLOY_MODE" in
    none|dry-run|draft|publish) ;;
    *) die "--deploy-mode must be one of: none, dry-run, draft, publish" ;;
esac
if [ "$SYNC_REMOTE_OUTPUT" != "1" ] && [ "$SKIP_TRAIN" != "1" ]; then
    case "$BACKEND" in
        local)
            die "--no-sync-remote-output requires a remote backend"
            ;;
        ssh|brev-existing|brev-provision)
            [ "$SKIP_EVAL" = "1" ] || die \
                "--no-sync-remote-output requires --skip-eval because the trained models will not exist locally"
            [ "$SKIP_PLOTS" = "1" ] || die \
                "--no-sync-remote-output requires --skip-plots because the evaluation artifacts will not exist locally"
            [ "$DEPLOY_MODE" = "none" ] || die \
                "--no-sync-remote-output requires --deploy-mode none because deployment reads local artifacts"
            ;;
    esac
fi
if [ "$ALLOW_DIRTY_REPO" = "1" ]; then
    case "$DEPLOY_MODE" in
        draft|publish)
            die "--allow-dirty-repo cannot be combined with --deploy-mode $DEPLOY_MODE"
            ;;
    esac
fi
if [ "$SKIP_DEPLOY" = "1" ]; then
    if [ "$DEPLOY_MODE" != "none" ]; then
        die "--skip-deploy cannot be combined with --deploy-mode $DEPLOY_MODE; deployment is opt-in, so omit --skip-deploy or use --deploy-mode none"
    fi
    warn "--skip-deploy is deprecated and no longer needed; deployment is opt-in."
fi
if [ -n "$PAPER_FIGURES_PREPARE_COMMAND" ]; then
    if [ "$SKIP_PLOTS" = "1" ]; then
        die "--paper-figures-prepare-command cannot be combined with --skip-plots"
    fi
    if ! paper_figure_inputs_requested; then
        die "--paper-figures-prepare-command requires --paper-figures-scores-dir or --paper-figures-*-predictions so its outputs are used"
    fi
fi
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
    if [ "$SYNC_REMOTE_OUTPUT" != "1" ]; then
        # A failed stop must never fall through to deletion while the only
        # artifact copy still lives on the remote disk.
        BREV_STOP_FAILURE_ACTION=warn
    else
        case "$BACKEND" in
            brev-provision)
                BREV_STOP_FAILURE_ACTION=delete
                ;;
            *)
                BREV_STOP_FAILURE_ACTION=warn
                ;;
        esac
    fi
fi
case "$BREV_STOP_FAILURE_ACTION" in
    warn|delete) ;;
    *) die "--brev-stop-failure-action must be one of: warn, delete" ;;
esac
if [ "$SYNC_REMOTE_OUTPUT" != "1" ]; then
    case "$BACKEND" in
        brev-existing|brev-provision)
            [ "$BREV_ON_FINISH" != "delete" ] || die \
                "--no-sync-remote-output cannot be combined with --brev-on-finish delete because that would destroy the only artifact copy"
            if [ "$BREV_STOP_FAILURE_ACTION" = "delete" ]; then
                if [ "$BREV_STOP_FAILURE_ACTION_EXPLICIT" = "1" ]; then
                    die "--no-sync-remote-output cannot be combined with --brev-stop-failure-action delete because that could destroy the only artifact copy"
                fi
                BREV_STOP_FAILURE_ACTION=warn
            fi
            ;;
    esac
fi
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
if [ "$BACKEND" = "brev-provision" ] && \
        [ -n "$BREV_INSTANCE_TYPE" ]; then
    if [ "$BREV_PROVIDER_EXPLICIT" = "1" ] && \
            [ "$(lowercase "$BREV_PROVIDER")" != "auto" ]; then
        die "a non-auto --brev-provider cannot be combined with --brev-instance-type"
    fi
    # An exact type is a complete Brev selection. Do not let an implicit
    # provider default or a release profile add a second selection mechanism.
    BREV_PROVIDER=auto
elif [ "$BACKEND" = "brev-provision" ] && \
        [ "$(lowercase "$BREV_PROVIDER")" != "auto" ]; then
    BREV_INSTANCE_TYPE="$(brev_provider_instance_type "$BREV_PROVIDER")"
fi
if [ "$BACKEND" = "brev-provision" ] && \
        [ -z "$BREV_INSTANCE_TYPE" ] && \
        [ -n "$DEFAULT_BREV_PROVISION_INSTANCE_TYPE" ]; then
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
note "Profile:       $RELEASE_PROFILE"
note "Batch sizes:   affinity=$AFFINITY_MINIBATCH_SIZE processing=$PROCESSING_MINIBATCH_SIZE"
note "Compare:       $RUN_LABEL vs $COMPARE_BASELINE_LABEL ($COMPARE_BASELINE)"
note "Affinity MWPG: $AFFINITY_MAX_WORKERS_PER_GPU"
note "Processing:    variants=$PROCESSING_VARIANTS; eval_modes=$PROCESSING_MODES; jobs=$PROCESSING_NUM_JOBS; workers/gpu=$PROCESSING_MAX_WORKERS_PER_GPU"
note "Presentation:  decoys/hit=$PRESENTATION_DECOYS_PER_HIT; sample_fraction=$PRESENTATION_SAMPLE_FRACTION; jobs=$PRESENTATION_NUM_JOBS; workers/gpu=$PRESENTATION_MAX_WORKERS_PER_GPU"
if [ -n "$PAPER_FIGURES_PREPARE_COMMAND" ]; then
    note "Paper inputs:  local prepare command configured"
fi
case "$BACKEND" in
    brev-existing|brev-provision)
        note "Brev instance: $BREV_INSTANCE"
        note "Brev cleanup:  $BREV_ON_FINISH"
        note "Brev provider: $BREV_PROVIDER"
        note "Stop fallback:  $BREV_STOP_FAILURE_ACTION"
        note "Brev sync:     $BREV_SYNC_MODE"
        note "Brev type:     ${BREV_INSTANCE_TYPE:-runplz auto-select}"
        note "Brev image:    $BREV_CONTAINER_IMAGE"
        ;;
esac

validate_release_provenance source_provenance 0
if [ "$DRY_RUN" = "1" ]; then
    RELEASE_GIT_COMMIT='<git rev-parse HEAD>'
else
    RELEASE_GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
fi

trap cleanup_background_jobs EXIT
start_paper_figures_prepare

if [ "$SKIP_TRAIN" = "1" ]; then
    validate_release_provenance model_provenance 1
fi

if [ "$SKIP_TRAIN" != "1" ]; then
    case "$BACKEND" in
        local)
            run_cmd mkdir -p "$RUN_DIR"
            run_logged_step train_local env \
                "MHCFLURRY_OUT=$RUN_DIR" \
                "REPO=$REPO" \
                "MHCFLURRY_RELEASE_WORKFLOW_ID=$WORKFLOW_RUN_ID" \
                "MHCFLURRY_RELEASE_GIT_COMMIT=$RELEASE_GIT_COMMIT" \
                "TRAINING_MINIBATCH_SIZE=$TRAINING_MINIBATCH_SIZE" \
                "AFFINITY_MINIBATCH_SIZE=$AFFINITY_MINIBATCH_SIZE" \
                "AFFINITY_MAX_WORKERS_PER_GPU=$AFFINITY_MAX_WORKERS_PER_GPU" \
                "PROCESSING_MINIBATCH_SIZE=$PROCESSING_MINIBATCH_SIZE" \
                "PROCESSING_NUM_JOBS=$PROCESSING_NUM_JOBS" \
                "PROCESSING_MAX_WORKERS_PER_GPU=$PROCESSING_MAX_WORKERS_PER_GPU" \
                "PROCESSING_HELD_OUT_SAMPLES=$PROCESSING_HELD_OUT_SAMPLES" \
                "PROCESSING_VARIANTS=$PROCESSING_VARIANTS" \
                "PRESENTATION_PROCESSING_WITH_FLANKS_KIND=$PRESENTATION_PROCESSING_WITH_FLANKS_KIND" \
                "PRESENTATION_DECOYS_PER_HIT=$PRESENTATION_DECOYS_PER_HIT" \
                "PRESENTATION_SAMPLE_FRACTION=$PRESENTATION_SAMPLE_FRACTION" \
                "PRESENTATION_FEATURE_CHUNK_SIZE=$PRESENTATION_FEATURE_CHUNK_SIZE" \
                "PRESENTATION_NUM_JOBS=$PRESENTATION_NUM_JOBS" \
                "PRESENTATION_MAX_WORKERS_PER_GPU=$PRESENTATION_MAX_WORKERS_PER_GPU" \
                "PRESENTATION_CALIBRATION_NUM_JOBS=$PRESENTATION_CALIBRATION_NUM_JOBS" \
                "PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU=$PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU" \
                "PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE=$PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE" \
                "MHCFLURRY_TORCH_COMPILE=$MHCFLURRY_TORCH_COMPILE" \
                "MHCFLURRY_TORCH_COMPILE_LOSS=$MHCFLURRY_TORCH_COMPILE_LOSS" \
                "MHCFLURRY_MATMUL_PRECISION=$MHCFLURRY_MATMUL_PRECISION" \
                "MATMUL_PRECISION=$MATMUL_PRECISION" \
                "MATMUL_PRECISION_CLI=$MATMUL_PRECISION_CLI" \
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
            verify_ssh_remote_checkout
            REMOTE_REPO_QUOTED="$(shell_quote "$REMOTE_REPO")"
            REMOTE_COMMAND="cd $REMOTE_REPO_QUOTED"
            REMOTE_COMMAND="$REMOTE_COMMAND && MHCFLURRY_OUT=$(shell_quote "$REMOTE_RUN_DIR")"
            REMOTE_COMMAND="$REMOTE_COMMAND REPO=$(shell_quote "$REMOTE_REPO")"
            REMOTE_COMMAND="$REMOTE_COMMAND MHCFLURRY_RELEASE_WORKFLOW_ID=$(shell_quote "$WORKFLOW_RUN_ID")"
            REMOTE_COMMAND="$REMOTE_COMMAND MHCFLURRY_RELEASE_GIT_COMMIT=\$(git -C $REMOTE_REPO_QUOTED rev-parse HEAD)"
            REMOTE_COMMAND="$REMOTE_COMMAND TRAINING_MINIBATCH_SIZE=$(shell_quote "$TRAINING_MINIBATCH_SIZE")"
            REMOTE_COMMAND="$REMOTE_COMMAND AFFINITY_MINIBATCH_SIZE=$(shell_quote "$AFFINITY_MINIBATCH_SIZE")"
            REMOTE_COMMAND="$REMOTE_COMMAND AFFINITY_MAX_WORKERS_PER_GPU=$(shell_quote "$AFFINITY_MAX_WORKERS_PER_GPU")"
            REMOTE_COMMAND="$REMOTE_COMMAND PROCESSING_MINIBATCH_SIZE=$(shell_quote "$PROCESSING_MINIBATCH_SIZE")"
            REMOTE_COMMAND="$REMOTE_COMMAND PROCESSING_NUM_JOBS=$(shell_quote "$PROCESSING_NUM_JOBS")"
            REMOTE_COMMAND="$REMOTE_COMMAND PROCESSING_MAX_WORKERS_PER_GPU=$(shell_quote "$PROCESSING_MAX_WORKERS_PER_GPU")"
            REMOTE_COMMAND="$REMOTE_COMMAND PROCESSING_HELD_OUT_SAMPLES=$(shell_quote "$PROCESSING_HELD_OUT_SAMPLES")"
            REMOTE_COMMAND="$REMOTE_COMMAND PROCESSING_VARIANTS=$(shell_quote "$PROCESSING_VARIANTS")"
            REMOTE_COMMAND="$REMOTE_COMMAND PRESENTATION_PROCESSING_WITH_FLANKS_KIND=$(shell_quote "$PRESENTATION_PROCESSING_WITH_FLANKS_KIND")"
            REMOTE_COMMAND="$REMOTE_COMMAND PRESENTATION_DECOYS_PER_HIT=$(shell_quote "$PRESENTATION_DECOYS_PER_HIT")"
            REMOTE_COMMAND="$REMOTE_COMMAND PRESENTATION_SAMPLE_FRACTION=$(shell_quote "$PRESENTATION_SAMPLE_FRACTION")"
            REMOTE_COMMAND="$REMOTE_COMMAND PRESENTATION_FEATURE_CHUNK_SIZE=$(shell_quote "$PRESENTATION_FEATURE_CHUNK_SIZE")"
            REMOTE_COMMAND="$REMOTE_COMMAND PRESENTATION_NUM_JOBS=$(shell_quote "$PRESENTATION_NUM_JOBS")"
            REMOTE_COMMAND="$REMOTE_COMMAND PRESENTATION_MAX_WORKERS_PER_GPU=$(shell_quote "$PRESENTATION_MAX_WORKERS_PER_GPU")"
            REMOTE_COMMAND="$REMOTE_COMMAND PRESENTATION_CALIBRATION_NUM_JOBS=$(shell_quote "$PRESENTATION_CALIBRATION_NUM_JOBS")"
            REMOTE_COMMAND="$REMOTE_COMMAND PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU=$(shell_quote "$PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU")"
            REMOTE_COMMAND="$REMOTE_COMMAND PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE=$(shell_quote "$PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE")"
            REMOTE_COMMAND="$REMOTE_COMMAND MHCFLURRY_TORCH_COMPILE=$(shell_quote "$MHCFLURRY_TORCH_COMPILE")"
            REMOTE_COMMAND="$REMOTE_COMMAND MHCFLURRY_TORCH_COMPILE_LOSS=$(shell_quote "$MHCFLURRY_TORCH_COMPILE_LOSS")"
            REMOTE_COMMAND="$REMOTE_COMMAND MHCFLURRY_MATMUL_PRECISION=$(shell_quote "$MHCFLURRY_MATMUL_PRECISION")"
            REMOTE_COMMAND="$REMOTE_COMMAND MATMUL_PRECISION=$(shell_quote "$MATMUL_PRECISION")"
            REMOTE_COMMAND="$REMOTE_COMMAND MATMUL_PRECISION_CLI=$(shell_quote "$MATMUL_PRECISION_CLI")"
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
    case "$BACKEND" in
        brev-existing)
            wait_paper_figures_prepare
            run_brev_postprocess 0
            ;;
        brev-provision)
            wait_paper_figures_prepare
            run_brev_postprocess 1
            ;;
    esac
fi

if [ "$SKIP_TRAIN" != "1" ]; then
    if [ "$BACKEND" = "ssh" ] && [ "$SYNC_REMOTE_OUTPUT" != "1" ]; then
        validate_ssh_remote_release_provenance
    elif [ "$BACKEND" != "local" ] && [ "$SYNC_REMOTE_OUTPUT" != "1" ]; then
        note "Using model provenance validation completed by the remote Brev launcher."
    else
        validate_release_provenance model_provenance 1
    fi
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
        case ",$COMPARE_INCLUDE," in
            *,affinity,*)
                run_logged_step fetch_train_excluded_affinity_baseline \
                    mhcflurry-downloads fetch models_class1_pan_variants
                ;;
        esac
        run_logged_step fetch_compare_baseline_downloads \
            fetch_pinned_public_baseline_downloads
        compare_args=(
            mhcflurry eval compare-models
            --a "$RUN_DIR"
            --a-label "$RUN_LABEL"
            --b "$COMPARE_BASELINE"
            --b-label "$COMPARE_BASELINE_LABEL"
            --data-dir "$DATA_DIR"
            --release-holdout-dir "$RUN_DIR/release_holdout"
            --affinity-training-overlap-policy audit
            --include "$COMPARE_INCLUDE"
            --processing-modes "$PROCESSING_MODES"
            --presentation-modes "$PRESENTATION_MODES"
            --backend "$COMPARE_BACKEND"
            --num-jobs "$COMPARE_NUM_JOBS"
            --max-workers-per-gpu "$COMPARE_MAX_WORKERS_PER_GPU"
            --max-tasks-per-worker "$COMPARE_MAX_TASKS_PER_WORKER"
            --presentation-num-jobs "$COMPARE_PRESENTATION_NUM_JOBS"
            --presentation-max-workers-per-gpu "$COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU"
            --presentation-max-tasks-per-worker "$COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER"
            --presentation-torch-compile "$COMPARE_PRESENTATION_TORCH_COMPILE"
            --torch-compile "$COMPARE_TORCH_COMPILE"
            --matmul-precision "$COMPARE_MATMUL_PRECISION"
            --out "$EVAL_OUT"
        )
        if [ -n "$EVAL_MAX_BENCHMARK_FILES" ]; then
            compare_args+=(--limit-files "$EVAL_MAX_BENCHMARK_FILES")
        fi
        case "$(lowercase "$COMPARE_GPUS")" in
            auto) ;;
            *) compare_args+=(--gpus "$COMPARE_GPUS") ;;
        esac
        run_logged_step compare_models "${compare_args[@]}"
        case ",$COMPARE_INCLUDE," in
            *,affinity,*)
                if [ "$DRY_RUN" = "1" ]; then
                    TRAIN_EXCLUDED_AFFINITY_DIR='<mhcflurry-downloads path models_class1_pan_variants>/models.no_additional_ms'
                else
                    TRAIN_EXCLUDED_AFFINITY_DIR="$(
                        mhcflurry-downloads path models_class1_pan_variants
                    )/models.no_additional_ms"
                fi
                fair_affinity_args=(
                    mhcflurry eval compare-models
                    --a "$RUN_DIR"
                    --a-label "$RUN_LABEL"
                    --b "$TRAIN_EXCLUDED_AFFINITY_DIR"
                    --b-affinity-dir "$TRAIN_EXCLUDED_AFFINITY_DIR"
                    --b-label "MHCflurry no-additional-MS (train-excluded)"
                    --data-dir "$DATA_DIR"
                    --release-holdout-dir "$RUN_DIR/release_holdout"
                    --affinity-training-overlap-policy exclude
                    --include affinity
                    --affinity-source no_additional_ms
                    --backend "$COMPARE_BACKEND"
                    --num-jobs "$COMPARE_NUM_JOBS"
                    --max-workers-per-gpu "$COMPARE_MAX_WORKERS_PER_GPU"
                    --max-tasks-per-worker "$COMPARE_MAX_TASKS_PER_WORKER"
                    --torch-compile "$COMPARE_TORCH_COMPILE"
                    --matmul-precision "$COMPARE_MATMUL_PRECISION"
                    --out "$RUN_DIR/eval_comparison_train_excluded_affinity"
                )
                if [ -n "$EVAL_MAX_BENCHMARK_FILES" ]; then
                    fair_affinity_args+=(
                        --limit-files "$EVAL_MAX_BENCHMARK_FILES")
                fi
                case "$(lowercase "$COMPARE_GPUS")" in
                    auto) ;;
                    *) fair_affinity_args+=(--gpus "$COMPARE_GPUS") ;;
                esac
                run_logged_step compare_models_train_excluded_affinity \
                    "${fair_affinity_args[@]}"
                ;;
        esac
    fi
else
    note "Skipping evaluation."
fi

if [ "$SKIP_PLOTS" != "1" ]; then
    wait_paper_figures_prepare
    if [ "$BREV_REMOTE_PLOTS_DONE" = "1" ]; then
        note "Using plots produced on the Brev instance."
    else
        plot_args=(
            mhcflurry eval plot-comparison
            --input "$RUN_DIR/eval_comparison"
            --a-label "$RUN_LABEL"
            --b-label "$COMPARE_BASELINE_LABEL"
            --summary-pdf "$RUN_DIR/eval_comparison/plots/model_comparison_figures.pdf"
            --paper-figures-out "$RUN_DIR/eval_comparison/plots/paper_figures"
            --paper-figures-formats "$PAPER_FIGURES_FORMATS"
            --paper-figures-scores-dir "${PAPER_FIGURES_SCORES_DIR:-$RUN_DIR/eval_comparison}"
        )
        if [ -n "$PAPER_FIGURES_MULTIALLELIC_PREDICTIONS" ]; then
            plot_args+=(--paper-figures-multiallelic-predictions "$PAPER_FIGURES_MULTIALLELIC_PREDICTIONS")
        fi
        if [ -n "$PAPER_FIGURES_MONOALLELIC_PREDICTIONS" ]; then
            plot_args+=(--paper-figures-monoallelic-predictions "$PAPER_FIGURES_MONOALLELIC_PREDICTIONS")
        fi
        if [ -n "$PAPER_FIGURES_CANDIDATE_PREDICTOR" ]; then
            plot_args+=(--paper-figures-candidate-predictor "$PAPER_FIGURES_CANDIDATE_PREDICTOR")
        fi
        if [ -n "$PAPER_FIGURES_EXTERNAL_BASELINES" ]; then
            plot_args+=(--paper-figures-external-baselines "$PAPER_FIGURES_EXTERNAL_BASELINES")
        fi
        if [ -n "$PAPER_FIGURES_PREFERRED_PREDICTORS" ]; then
            plot_args+=(--paper-figures-preferred-predictors "$PAPER_FIGURES_PREFERRED_PREDICTORS")
        fi
        if [ -n "$PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS" ]; then
            plot_args+=(--paper-figures-presentation-panel-predictors "$PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS")
        fi
        if [ -n "$PAPER_FIGURES_PRESENTATION_PANEL_BASELINES" ]; then
            plot_args+=(--paper-figures-presentation-panel-baselines "$PAPER_FIGURES_PRESENTATION_PANEL_BASELINES")
        fi
        run_logged_step plot_model_comparison \
            "${plot_args[@]}"
        if [ -d "$RUN_DIR/eval_comparison_train_excluded_affinity" ]; then
            run_logged_step plot_train_excluded_affinity_comparison \
                mhcflurry eval plot-comparison \
                --input "$RUN_DIR/eval_comparison_train_excluded_affinity" \
                --a-label "$RUN_LABEL" \
                --b-label "MHCflurry no-additional-MS (train-excluded)" \
                --summary-pdf \
                    "$RUN_DIR/eval_comparison_train_excluded_affinity/plots/model_comparison_figures.pdf"
        fi
    fi
else
    note "Skipping plots."
fi

if [ "$DEPLOY_MODE" != "none" ]; then
    deploy_args=(
        "$SCRIPT_DIR/deploy_trained_models.sh"
        --run-dir "$RUN_DIR"
        --release "$RELEASE"
        --github-release "$GITHUB_RELEASE"
        --processing-variants "$PROCESSING_VARIANTS"
        --repo "$REPO"
        --mode "$DEPLOY_MODE"
    )
    if [ "$ALLOW_DIRTY_REPO" = "1" ]; then
        deploy_args+=(--allow-dirty-repo)
    fi
    if [ "$SKIP_TRAIN" = "1" ]; then
        deploy_args+=(--allow-artifact-source-mismatch)
    fi
    run_logged_step deploy_trained_models \
        "${deploy_args[@]}"
else
    note "Skipping deploy step. Pass --deploy-mode dry-run, draft, or publish to opt in."
fi
