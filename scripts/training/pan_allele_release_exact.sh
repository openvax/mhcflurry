#!/usr/bin/env bash
#
# Exact replication of the public mhcflurry pan-allele release training
# recipe (models_class1_pan, 2.2.0's GENERATE.sh — local mode, not
# cluster). Uses the same `generate_hyperparameters.py`,
# `reassign_mass_spec_training_data.py`, `additional_alleles.txt`, the
# same curated_training_data and pretrain data (random_peptide_predictions),
# same 4 folds, same architecture sweep, same --min-models/--max-models
# selection, and runs percentile-rank calibration on the final ensemble.
#
# Env:
#   MHCFLURRY_OUT              required — where artifacts are written.
#                              Final selected ensemble lives at
#                              $MHCFLURRY_OUT/models.combined/.
#   NUM_JOBS=N                 parallelism (default: $GPUS). Controls
#                              how many networks train concurrently;
#                              mhcflurry pairs jobs 1:1 with GPUs.
set -euo pipefail
set -x

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"

export PYTHONUNBUFFERED=1
# torch.compile on by default after the Phase-4 sweep landed it.
# Compile cost (~30-60s codegen) is paid once per worker. We deliberately
# recycle workers after a moderate number of tasks to avoid the mysterious
# long-lived-worker death mode seen on multi-day runs, while still
# amortizing compile / CUDA init across several networks.
export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-1}"
# Inductor defaults to a large compile helper pool per training process.
# With 8-16 concurrent mhcflurry workers that multiplies into thousands of
# short-lived subprocesses and can stall the box. Keep the default bounded;
# callers can override upward when running fewer workers.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
# OMP/MKL/OPENBLAS set via set_cpu_threads helper below — auto-computes
# from nproc + GPU/worker layout. Manually override any of them before
# calling this script to pin an explicit value.

SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_ABSOLUTE_PATH")"
RECIPE_DIR="$SCRIPT_DIR/release_exact"

mkdir -p "$MHCFLURRY_OUT"
cd "$MHCFLURRY_OUT"

RELEASE_LOG="$MHCFLURRY_OUT/release_driver.log"
HEARTBEAT_LOG="$MHCFLURRY_OUT/release_heartbeat.log"
TRAIN_LOG="$MHCFLURRY_OUT/train.log"
SELECT_LOG="$MHCFLURRY_OUT/select.log"
CALIBRATE_LOG="$MHCFLURRY_OUT/calibrate.log"
FETCH_LOG="$MHCFLURRY_OUT/fetch_eval_data.log"
EVAL_LOG="$MHCFLURRY_OUT/eval.log"
PLOT_LOG="$MHCFLURRY_OUT/plot_loss_curves.log"

: > "$RELEASE_LOG"
: > "$HEARTBEAT_LOG"
: > "$TRAIN_LOG"
: > "$SELECT_LOG"
: > "$CALIBRATE_LOG"
: > "$FETCH_LOG"
: > "$EVAL_LOG"
: > "$PLOT_LOG"

CURRENT_PHASE="bootstrap"
GPU_SAMPLER_PID=""
HEARTBEAT_PID=""

timestamp_utc() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

format_command() {
    printf '%q ' "$@"
}

log_release_event() {
    local event="$1"
    shift || true
    printf '%s event=%s phase=%s %s\n' \
        "$(timestamp_utc)" \
        "$event" \
        "$CURRENT_PHASE" \
        "$*" | tee -a "$RELEASE_LOG" >&2
}

write_snapshot() {
    local reason="$1"
    {
        printf '=== snapshot timestamp=%s reason=%s phase=%s pid=%s ppid=%s ===\n' \
            "$(timestamp_utc)" \
            "$reason" \
            "$CURRENT_PHASE" \
            "$$" \
            "$PPID"
        printf 'host=%s cwd=%s script=%s\n' \
            "$(hostname)" \
            "$(pwd)" \
            "$SCRIPT_ABSOLUTE_PATH"
        df -h . 2>/dev/null || true
        ps -Ao pid,ppid,etime,%cpu,%mem,command --sort=-%cpu 2>/dev/null | head -n 25 || true
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi \
                --query-gpu=index,utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits 2>/dev/null || true
        fi
        printf '\n'
    } >> "$HEARTBEAT_LOG" 2>&1 || true
}

start_heartbeat() {
    local interval="${RELEASE_HEARTBEAT_SECONDS:-60}"
    (
        while :; do
            write_snapshot heartbeat
            sleep "$interval"
        done
    ) &
    HEARTBEAT_PID=$!
    log_release_event heartbeat_started "pid=$HEARTBEAT_PID interval_seconds=$interval"
}

stop_background_loggers() {
    local pid
    for pid in "${HEARTBEAT_PID:-}" "${GPU_SAMPLER_PID:-}"; do
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}

on_error() {
    local status="$1"
    local line="$2"
    local command="$3"
    log_release_event error "status=$status line=$line command=$(printf '%q' "$command")"
    write_snapshot error
}

on_signal() {
    local signal_name="$1"
    log_release_event signal "name=$signal_name"
    write_snapshot "signal_$signal_name"
    case "$signal_name" in
        INT) exit 130 ;;
        TERM) exit 143 ;;
        *) exit 1 ;;
    esac
}

cleanup_release_logging() {
    local status="$?"
    log_release_event script_exit "status=$status"
    write_snapshot exit
    stop_background_loggers
}

run_logged_step() {
    local phase="$1"
    local log_file="$2"
    shift 2

    CURRENT_PHASE="$phase"
    log_release_event phase_start "log=$log_file command=$(format_command "$@")"
    write_snapshot "phase_start_$phase"

    set +e
    (
        set -o pipefail
        "$@" 2>&1 | tee -a "$log_file"
    )
    local status="$?"
    set -e

    log_release_event phase_end "status=$status log=$log_file"
    write_snapshot "phase_end_$phase"
    return "$status"
}

trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap cleanup_release_logging EXIT

log_release_event script_start \
    "pid=$$ out_dir=$MHCFLURRY_OUT pythonunbuffered=$PYTHONUNBUFFERED torch_compile=$MHCFLURRY_TORCH_COMPILE inductor_threads=$TORCHINDUCTOR_COMPILE_THREADS"
write_snapshot startup
start_heartbeat

# ---- GPU occupancy sampler (background, for later analysis) ----------
# Sample nvidia-smi every 30 s into a CSV that rsync-down can pull back.
# Captures per-GPU util, memory, and timestamp so we can answer "what
# was the steady-state occupancy?" without having to SSH live during
# the run. Auto-exits when the training process ends (watches its pid).
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LOG="$MHCFLURRY_OUT/gpu_occupancy.csv"
    {
        echo "timestamp,gpu_index,util_percent,mem_used_mib,mem_total_mib"
        while :; do
            ts=$(date +%s)
            nvidia-smi \
                --query-gpu=index,utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits 2>/dev/null \
              | while IFS=',' read -r idx util mem_used mem_total; do
                  echo "$ts,${idx// /},${util// /},${mem_used// /},${mem_total// /}"
              done
            sleep 30
        done
    } > "$GPU_LOG" 2>/dev/null &
    GPU_SAMPLER_PID=$!
    log_release_event gpu_sampler_started "pid=$GPU_SAMPLER_PID log=$GPU_LOG"
fi

# ---- parallelism -----------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    GPUS=0
fi
echo "Detected GPUS: $GPUS"

PROCESSORS=$(getconf _NPROCESSORS_ONLN)
echo "Detected processors: $PROCESSORS"

# Default to "auto" so the orchestrator's auto_max_workers_per_gpu
# picks based on detected free VRAM. Per-worker peak is 16-22 GB on
# mhcflurry pan-allele training: ``auto`` lands at 2 on 80GB cards
# and 1 on 40GB cards. Override with MAX_WORKERS_PER_GPU=N (or
# explicitly "auto") to pin.
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-auto}"
# NUM_JOBS arithmetic: bash can't multiply "auto", so fall back to a
# conservative 2 for the Pool size when the user picked auto. The Python
# orchestrator still applies the actual auto_max_workers_per_gpu cap to
# the GPU-assignment side, so an over-sized Pool just leaves some workers
# idle — never oversubscribed VRAM.
if [ "$MAX_WORKERS_PER_GPU" = "auto" ]; then
    _MWPG_FOR_NUM_JOBS=2
else
    _MWPG_FOR_NUM_JOBS="$MAX_WORKERS_PER_GPU"
fi
if [ "$GPUS" -eq "0" ]; then
    NUM_JOBS="${NUM_JOBS-1}"
else
    NUM_JOBS="${NUM_JOBS-$(( GPUS * _MWPG_FOR_NUM_JOBS ))}"
fi
echo "Num jobs: $NUM_JOBS (max-workers-per-gpu=$MAX_WORKERS_PER_GPU)"
# Recycle after a bounded number of tasks so compile is still amortized
# but leaks / descriptor creep / orphaned-runtime state cannot accumulate
# indefinitely inside one worker. Override with MAX_TASKS_PER_WORKER.
MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-12}"
PARALLELISM_ARGS=(
    --num-jobs "$NUM_JOBS"
    --max-tasks-per-worker "$MAX_TASKS_PER_WORKER"
    --gpus "$GPUS"
    --max-workers-per-gpu "$MAX_WORKERS_PER_GPU"
)

# Phase 1 (#268): enable the BLOSUM62 encoding cache + DataLoader prefetch
# by default. USE_ENCODING_CACHE=0 or DATALOADER_NUM_WORKERS=0 restores
# the pre-#268 legacy path (bit-identical; verified by Phase 1 tests).
USE_ENCODING_CACHE="${USE_ENCODING_CACHE:-1}"
# DataLoader prefetch workers per training worker. v11 sweep winner
# (batch=512, dl=1, wpg=2 on L40S) showed dl=1 marginally beat dl=2;
# dl=2 added overhead without parallelism benefit at single-item
# granularity. Default to 1 now.
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
CACHE_ARGS=()
if [ "$USE_ENCODING_CACHE" = "1" ]; then
    # Place the encoding cache OUTSIDE $MHCFLURRY_OUT so it
    #   (a) doesn't ride back on the post-run rsync (~7 GB of mmap
    #       BLOSUM62 we can rebuild locally in seconds), and
    #   (b) persists across runs on the same box — the second run on a
    #       reused instance hits a warm cache.
    # Override with MHCFLURRY_ENCODING_CACHE_DIR=/path/to/dir.
    ENCODING_CACHE_DIR="${MHCFLURRY_ENCODING_CACHE_DIR:-$HOME/runplz-cache/encoding_cache}"
    mkdir -p "$ENCODING_CACHE_DIR"
    CACHE_ARGS=(--use-encoding-cache --encoding-cache-dir "$ENCODING_CACHE_DIR")
    log_release_event phase_info "encoding cache dir: $ENCODING_CACHE_DIR"
fi

# Auto-configure OMP / MKL / OPENBLAS thread budget uniformly based on
# nproc, GPU count, worker count, dataloader worker count. User can
# override any of {OMP,MKL,OPENBLAS}_NUM_THREADS before invoking this
# script; the helper respects manual settings.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/set_cpu_threads.sh"
GPUS="$GPUS" MAX_WORKERS_PER_GPU="$MAX_WORKERS_PER_GPU" \
    DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" set_cpu_threads

# ---- data ------------------------------------------------------------
CURRENT_PHASE="data_setup"
log_release_event phase_info "starting data download and preprocessing"
mhcflurry-downloads fetch data_curated allele_sequences random_peptide_predictions

# Reassign mass-spec training rows per release recipe.
cp "$RECIPE_DIR/reassign_mass_spec_training_data.py" .
cp "$RECIPE_DIR/additional_alleles.txt" .

python reassign_mass_spec_training_data.py \
    "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" \
    --set-measurement-value 100 \
    --out-csv "$(pwd)/train_data.csv"
bzip2 -f "$(pwd)/train_data.csv"
TRAINING_DATA="$(pwd)/train_data.csv.bz2"

# ---- hyperparameters -------------------------------------------------
CURRENT_PHASE="hyperparameters"
cp "$RECIPE_DIR/generate_hyperparameters.py" .
python generate_hyperparameters.py > hyperparameters.full.yaml
# Inject dataloader_num_workers into every arch entry (Phase 1 #268).
python - <<PY
import yaml
hp = yaml.safe_load(open("hyperparameters.full.yaml"))
for d in hp:
    d["dataloader_num_workers"] = $DATALOADER_NUM_WORKERS
with open("hyperparameters.yaml", "w") as f:
    yaml.safe_dump(hp, f)
print(f"release_exact: {len(hp)} archs with dataloader_num_workers=$DATALOADER_NUM_WORKERS")
PY
ARCH_COUNT=$(python -c "import yaml; print(len(yaml.safe_load(open('hyperparameters.yaml'))))")
echo "Architectures in sweep: $ARCH_COUNT"

ALLELE_SEQUENCES="$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv"
PRETRAIN_DATA="$(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2"

# ---- train all candidates (4 folds × ARCH_COUNT architectures) ------
# `combined` is the only kind the public release trains (see GENERATE.sh).
for kind in combined
do
    UNSELECTED_DIR="models.unselected.${kind}"

    CONTINUE_ARGS=()
    if [ -d "$UNSELECTED_DIR" ]; then
        log_release_event continue_incomplete "models_dir=$UNSELECTED_DIR"
        CONTINUE_ARGS=(--continue-incomplete)
    fi

    run_logged_step "train_${kind}" "$TRAIN_LOG" \
        mhcflurry-class1-train-pan-allele-models \
        --data "$TRAINING_DATA" \
        --allele-sequences "$ALLELE_SEQUENCES" \
        --pretrain-data "$PRETRAIN_DATA" \
        --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
        --num-folds 4 \
        --hyperparameters hyperparameters.yaml \
        --out-models-dir "$(pwd)/$UNSELECTED_DIR" \
        --worker-log-dir "$MHCFLURRY_OUT" \
        "${CACHE_ARGS[@]}" \
        "${PARALLELISM_ARGS[@]}" \
        "${CONTINUE_ARGS[@]}"
done
log_release_event phase_info "training_complete beginning_model_selection"

# ---- select ---------------------------------------------------------
for kind in combined
do
    UNSELECTED_DIR="models.unselected.${kind}"
    SELECTED_DIR="models.${kind}"

    run_logged_step "select_${kind}" "$SELECT_LOG" \
        mhcflurry-class1-select-pan-allele-models \
        --data "$UNSELECTED_DIR/train_data.csv.bz2" \
        --models-dir "$UNSELECTED_DIR" \
        --out-models-dir "$SELECTED_DIR" \
        --min-models 2 \
        --max-models 8 \
        "${PARALLELISM_ARGS[@]}"
    cp "$UNSELECTED_DIR/train_data.csv.bz2" "$SELECTED_DIR/train_data.csv.bz2"

    # ---- percentile rank calibration (matches release) ---------------
    # Three speedups vs the legacy invocation, individually safe and
    # stacking to ~10-20x on CUDA on the pan-allele universe:
    #
    #   --gpu-batched: batches many alleles into one forward pass via
    #     the issue-#272 GPU-hoisted fast path. Bit-identical on CUDA;
    #     ~5-30x faster than the per-allele predict() loop.
    #   --alleles-per-work-chunk 30 (was 10): better amortization of
    #     per-chunk fixed costs (pool dispatch, model load, aggregate).
    #     ~3x fewer chunks. Override with CALIBRATE_ALLELES_PER_CHUNK.
    #   --num-peptides-per-length 50000 (was 100000): linear in
    #     calibration wall time. Quality trade-off matters only for
    #     the deep tail of the percent-rank distribution; for
    #     ranks > 0.5% the noise from halving is negligible.
    #     Override with CALIBRATE_PEPTIDES_PER_LENGTH.
    CALIBRATE_PEPTIDES_PER_LENGTH="${CALIBRATE_PEPTIDES_PER_LENGTH:-50000}"
    CALIBRATE_ALLELES_PER_CHUNK="${CALIBRATE_ALLELES_PER_CHUNK:-30}"
    run_logged_step "calibrate_${kind}" "$CALIBRATE_LOG" \
        mhcflurry-calibrate-percentile-ranks \
        --models-dir "$SELECTED_DIR" \
        --match-amino-acid-distribution-data "$UNSELECTED_DIR/train_data.csv.bz2" \
        --motif-summary \
        --num-peptides-per-length "$CALIBRATE_PEPTIDES_PER_LENGTH" \
        --alleles-per-work-chunk "$CALIBRATE_ALLELES_PER_CHUNK" \
        --gpu-batched \
        --verbosity 1 \
        "${PARALLELISM_ARGS[@]}"
done

# ---- eval against public release ------------------------------------
# Compares the freshly-trained ensemble vs the published 2.2.0 ensemble
# on the data_evaluation hit/decoy benchmark. Per-allele ROC-AUC,
# PR-AUC, PPV@N. Skip cleanly with SKIP_EVAL=1.
#
# CURRENT_PHASE is set unconditionally so the heartbeat / error trap
# logs the right phase even when the stage is skipped (otherwise the
# preceding ``calibrate_combined`` would leak through).
CURRENT_PHASE="eval"
SKIP_EVAL="${SKIP_EVAL:-0}"
if [ "$SKIP_EVAL" = "1" ]; then
    log_release_event phase_skipped "SKIP_EVAL=1"
else
    log_release_event phase_info "starting eval against public release"
    # ``mhcflurry-downloads fetch`` accepts multiple targets via nargs="*"
    # (see mhcflurry/downloads_command.py:74). Both go to FETCH_LOG so
    # the per-allele eval output stays unpolluted by extraction progress.
    run_logged_step "fetch_eval_data" "$FETCH_LOG" \
        mhcflurry-downloads fetch data_evaluation models_class1_pan
    PUBLIC_MODELS_DIR="$(mhcflurry-downloads path models_class1_pan)/models.combined"
    DATA_EVAL_DIR="$(mhcflurry-downloads path data_evaluation)"
    EVAL_OUT="$MHCFLURRY_OUT/eval_comparison"
    mkdir -p "$EVAL_OUT"
    COMPARE_SCRIPT="$SCRIPT_DIR/compare_new_vs_public.py"
    if [ ! -f "$COMPARE_SCRIPT" ]; then
        log_release_event eval_skipped "missing compare_new_vs_public.py"
    else
        run_logged_step "eval_compare_new_vs_public" "$EVAL_LOG" \
            python3 "$COMPARE_SCRIPT" \
            --new-models-dir "$MHCFLURRY_OUT/models.combined" \
            --public-models-dir "$PUBLIC_MODELS_DIR" \
            --data-dir "$DATA_EVAL_DIR" \
            --out "$EVAL_OUT"
    fi
fi

# ---- loss-curve plots -----------------------------------------------
# Reads fit_info from the candidate-pool + selected manifests; emits
# loss_curves_by_model.png, loss_curves_by_arch.png, per_fold_summary.png,
# summary.csv. Skip cleanly with SKIP_PLOTS=1.
CURRENT_PHASE="plot"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
if [ "$SKIP_PLOTS" = "1" ]; then
    log_release_event phase_skipped "SKIP_PLOTS=1"
else
    PLOT_SCRIPT="$SCRIPT_DIR/plot_loss_curves.py"
    PLOT_OUT="$MHCFLURRY_OUT/loss_plots"
    if [ ! -f "$PLOT_SCRIPT" ]; then
        log_release_event plot_skipped "missing plot_loss_curves.py"
    else
        run_logged_step "plot_loss_curves" "$PLOT_LOG" \
            python3 "$PLOT_SCRIPT" \
            --selected-dir "$MHCFLURRY_OUT/models.combined" \
            --unselected-dir "$MHCFLURRY_OUT/models.unselected.combined" \
            --out "$PLOT_OUT"
    fi
fi

CURRENT_PHASE="complete"
log_release_event complete "final_dir=$MHCFLURRY_OUT/models.combined"
{
    echo "Full release-exact pipeline completed."
    echo "Final ensemble:   $MHCFLURRY_OUT/models.combined/"
    echo "Eval comparison:  $MHCFLURRY_OUT/eval_comparison/  (if SKIP_EVAL!=1)"
    echo "Loss-curve plots: $MHCFLURRY_OUT/loss_plots/       (if SKIP_PLOTS!=1)"
    ls -la "$MHCFLURRY_OUT/models.combined" | head -30
} | tee -a "$RELEASE_LOG"
