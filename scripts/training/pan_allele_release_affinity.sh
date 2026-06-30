#!/usr/bin/env bash
#
# Exact replication of the public mhcflurry pan-allele release training
# recipe (models_class1_pan, 2.2.0's GENERATE.sh — local mode, not
# cluster). Uses the same `generate_hyperparameters.py`,
# `reassign_mass_spec_training_data.py`, `additional_alleles.txt`, the
# same curated_training_data and pretrain data (random_peptide_predictions),
# same 4 folds, same architecture sweep, same min/max-models-per-fold
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

pseudosequence_lookup() {
    python -c 'from mhcflurry.pseudosequences import main; main()' "$@"
}

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"

export PYTHONUNBUFFERED=1
# torch.compile is on by default. Compile cost (~30-60s codegen) is paid
# once per worker; we recycle workers after a moderate number of tasks to
# avoid long-lived-worker death modes on multi-day runs while still
# amortizing compile / CUDA init across several networks.
export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-1}"
# Inductor defaults to a large compile helper pool per training process.
# With 8-16 concurrent mhcflurry workers that multiplies into thousands of
# short-lived subprocesses and can stall the box. Leave the env unset by
# default: mhcflurry.local_parallelism resolves it from the final NUM_JOBS
# in the same place that resolves Pool/GPU concurrency. Callers can still
# export TORCHINDUCTOR_COMPILE_THREADS before invoking this script to pin.
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
    "pid=$$ out_dir=$MHCFLURRY_OUT pythonunbuffered=$PYTHONUNBUFFERED torch_compile=$MHCFLURRY_TORCH_COMPILE inductor_threads=${TORCHINDUCTOR_COMPILE_THREADS:-auto}"
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

# Default to "auto" so the shared Python resolver picks from the actual
# hardware budget. Resolve it here before shell arithmetic so every downstream
# consumer (Pool size, BLAS thread budget, CLI args) sees one numeric plan.
MAX_WORKERS_PER_GPU_REQUESTED="${MAX_WORKERS_PER_GPU:-auto}"
if [ "$GPUS" -eq "0" ]; then
    NUM_JOBS="${NUM_JOBS-1}"
    MAX_WORKERS_PER_GPU=1
elif [ "$MAX_WORKERS_PER_GPU_REQUESTED" = "auto" ]; then
    # When NUM_JOBS is explicit, honor it: by_jobs clamps MWPG to
    # NUM_JOBS // GPUS. When NUM_JOBS is unset (the typical "auto"
    # case), pass 0 so the orchestrator skips the by_jobs clamp and
    # picks MWPG on VRAM + hard_cap alone — otherwise the historical
    # _NUM_JOBS_FOR_AUTO=GPUS*2 default pinned MWPG to 2 even when
    # VRAM allowed 4.
    _NUM_JOBS_FOR_AUTO="${NUM_JOBS:-0}"
    MAX_WORKERS_PER_GPU="$(
        NUM_JOBS="$_NUM_JOBS_FOR_AUTO" GPUS="$GPUS" python - <<'PY'
import os
from mhcflurry.local_parallelism import auto_max_workers_per_gpu
print(auto_max_workers_per_gpu(
    num_jobs=int(os.environ["NUM_JOBS"]),
    num_gpus=int(os.environ["GPUS"]),
    backend="auto",
))
PY
    )"
    _GPU_WORKER_CAP=$(( GPUS * MAX_WORKERS_PER_GPU ))
    if [ -z "${NUM_JOBS+x}" ] || [ "$NUM_JOBS" -gt "$_GPU_WORKER_CAP" ]; then
        NUM_JOBS="$_GPU_WORKER_CAP"
    fi
    echo "Resolved MAX_WORKERS_PER_GPU=auto to $MAX_WORKERS_PER_GPU; NUM_JOBS=$NUM_JOBS"
else
    MAX_WORKERS_PER_GPU="$MAX_WORKERS_PER_GPU_REQUESTED"
    NUM_JOBS="${NUM_JOBS-$(( GPUS * MAX_WORKERS_PER_GPU ))}"
fi
echo "Num jobs: $NUM_JOBS (max-workers-per-gpu=$MAX_WORKERS_PER_GPU; requested=$MAX_WORKERS_PER_GPU_REQUESTED)"
# Recycle after a bounded number of tasks so compile is still amortized
# but leaks / descriptor creep / orphaned-runtime state cannot accumulate
# indefinitely inside one worker. Override with MAX_TASKS_PER_WORKER.
MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-12}"

# DataLoader prefetch workers per training worker. Default ``auto`` is
# resolved via the same helper the orchestrator uses so the shell-side OMP
# budget and the training hyperparameters stay in lockstep.
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-auto}"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/set_cpu_threads.sh"
DATALOADER_NUM_WORKERS_REQUESTED="$DATALOADER_NUM_WORKERS"
DATALOADER_NUM_WORKERS="$(resolve_dataloader_num_workers "$NUM_JOBS")"
printf >&2 \
    "[pan_allele_release_affinity.sh] DATALOADER_NUM_WORKERS=%s resolved to %s\n" \
    "$DATALOADER_NUM_WORKERS_REQUESTED" "$DATALOADER_NUM_WORKERS"

# Now that DATALOADER_NUM_WORKERS is a resolved int, build PARALLELISM_ARGS.
PARALLELISM_ARGS=(
    --num-jobs "$NUM_JOBS"
    --max-tasks-per-worker "$MAX_TASKS_PER_WORKER"
    --gpus "$GPUS"
    --max-workers-per-gpu "$MAX_WORKERS_PER_GPU"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    # Orchestrator-owned tuning knobs. CLI-resolved values are propagated
    # to env (MHCFLURRY_TORCH_COMPILE / MHCFLURRY_MATMUL_PRECISION /
    # MHCFLURRY_ENABLE_TIMING) inside resolve_local_parallelism_args, so
    # the existing maybe_compile_network / configure_matmul_precision /
    # _timing_enabled call sites continue to read env unchanged. Defaults
    # to 'auto' for --torch-compile so an env preset (e.g.
    # MHCFLURRY_TORCH_COMPILE=1 set by the runplz container) still wins
    # when the shell is invoked outside the runplz path.
    --torch-compile "${TORCH_COMPILE_CLI:-auto}"
    --matmul-precision "${MATMUL_PRECISION_CLI:-none}"
)
if [ "${MHCFLURRY_ENABLE_TIMING:-0}" = "1" ]; then
    PARALLELISM_ARGS+=(--enable-timing)
fi

# Auto-configure OMP / MKL / OPENBLAS thread budget uniformly based on
# nproc, GPU count, worker count, dataloader worker count. User can
# override any of {OMP,MKL,OPENBLAS}_NUM_THREADS before invoking this
# script; the helper respects manual settings.
NUM_JOBS="$NUM_JOBS" GPUS="$GPUS" MAX_WORKERS_PER_GPU="$MAX_WORKERS_PER_GPU" \
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
python generate_hyperparameters.py > hyperparameters.yaml
# ``dataloader_num_workers`` is no longer injected here. The orchestrator
# overrides each work item's hyperparameters via the
# ``--dataloader-num-workers`` CLI flag at planning time (see
# ``apply_dataloader_num_workers_to_work_items``). One source of truth →
# saved component-model configs reflect the value the orchestrator
# actually chose for the run, not whatever the YAML happened to say.
ARCH_COUNT=$(python -c "import yaml; print(len(yaml.safe_load(open('hyperparameters.yaml'))))")
echo "Architectures in sweep: $ARCH_COUNT"

ALLELE_SEQUENCES_DIR="$(mhcflurry-downloads path allele_sequences)"
ALLELE_SEQUENCES="$(pseudosequence_lookup path \
    --directory "$ALLELE_SEQUENCES_DIR" \
    --length 39 \
    --fallback-legacy)"
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
        --min-models-per-fold 2 \
        --max-models-per-fold 8 \
        "${PARALLELISM_ARGS[@]}"
    cp "$UNSELECTED_DIR/train_data.csv.bz2" "$SELECTED_DIR/train_data.csv.bz2"

    # ---- percentile rank calibration (matches release) ---------------
    # Three speedups vs the legacy invocation, individually safe and
    # stacking to ~10-20x on CUDA on the pan-allele universe:
    #
    #   --gpu-batched: batches many alleles into one forward pass via
    #     the GPU-hoisted fast path. Bit-identical on CUDA; ~5-30x
    #     faster than the per-allele predict() loop.
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
    # Calibrate's peak VRAM per worker is dominated by the merged
    # ensemble's peptide-stage cache (~8x stage_dim × n_peptides × 4
    # bytes ≈ 6-13 GB depending on architecture). After the GPU
    # PercentRankTransform.fit move this is GPU-bound (cartesian
    # forward + bucketize + scatter_add), not Python-bound, so
    # 1 worker/GPU saturates the SMs on its own — but we let the
    # auto-resolver pick: it'll choose 1 on small-VRAM cards or
    # workloads that hit the cache hard, and >1 only when there's
    # genuine slack. Override with CALIBRATE_MAX_WORKERS_PER_GPU.
    # Keep --num-jobs on auto by default too: if we manufacture a numeric
    # ceiling here, the planner must honor it as an explicit CLI override
    # and overflow workers spill to CPU when MWPG resolves lower.
    # Override with CALIBRATE_NUM_JOBS only when deliberately pinning.
    # CALIBRATE_PER_WORKER_GB tells the resolver how much VRAM to
    # budget per worker (default 12 — the cache + ~2 GB working
    # set on 8-network ensembles). Bumped from the training
    # default of 4 because the calibrate cache is much larger.
    CALIBRATE_NUM_JOBS="${CALIBRATE_NUM_JOBS:-auto}"
    CALIBRATE_MAX_WORKERS_PER_GPU="${CALIBRATE_MAX_WORKERS_PER_GPU:-auto}"
    CALIBRATE_PER_WORKER_GB="${CALIBRATE_PER_WORKER_GB:-12}"
    CALIBRATE_PARALLELISM_ARGS=(
        --num-jobs "$CALIBRATE_NUM_JOBS"
        --max-tasks-per-worker "$MAX_TASKS_PER_WORKER"
        --gpus "$GPUS"
        --max-workers-per-gpu "$CALIBRATE_MAX_WORKERS_PER_GPU"
        --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
        --torch-compile "${TORCH_COMPILE_CLI:-auto}"
        --matmul-precision "${MATMUL_PRECISION_CLI:-none}"
    )
    if [ "${MHCFLURRY_ENABLE_TIMING:-0}" = "1" ]; then
        CALIBRATE_PARALLELISM_ARGS+=(--enable-timing)
    fi
    run_logged_step "calibrate_${kind}" "$CALIBRATE_LOG" \
        env "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB=$CALIBRATE_PER_WORKER_GB" \
        mhcflurry-calibrate-percentile-ranks \
        --models-dir "$SELECTED_DIR" \
        --match-amino-acid-distribution-data "$UNSELECTED_DIR/train_data.csv.bz2" \
        --motif-summary \
        --num-peptides-per-length "$CALIBRATE_PEPTIDES_PER_LENGTH" \
        --alleles-per-work-chunk "$CALIBRATE_ALLELES_PER_CHUNK" \
        --gpu-batched \
        --verbosity 1 \
        "${CALIBRATE_PARALLELISM_ARGS[@]}"
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
    # (see mhcflurry/cli/downloads_command.py:74). Both go to FETCH_LOG so
    # the per-allele eval output stays unpolluted by extraction progress.
    run_logged_step "fetch_eval_data" "$FETCH_LOG" \
        mhcflurry-downloads fetch data_evaluation models_class1_pan
    DATA_EVAL_DIR="$(mhcflurry-downloads path data_evaluation)"
    EVAL_OUT="$MHCFLURRY_OUT/eval_comparison"
    mkdir -p "$EVAL_OUT"
    run_logged_step "eval_compare_new_vs_public" "$EVAL_LOG" \
        mhcflurry compare-models \
            --a "$MHCFLURRY_OUT/models.combined" \
            --a-label new \
            --b public \
            --data-dir "$DATA_EVAL_DIR" \
            --include affinity \
            --out "$EVAL_OUT"
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
