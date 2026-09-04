#!/usr/bin/env bash
# Train the matched 5-aa processing regularization/activation screen.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_processing_regularization_activation.sh [OPTIONS]

Required:
  --out PATH                         Experiment output directory
  --train-data PATH                  Frozen processing training table
  --data-eval-dir PATH               Evaluation data directory
  --release-holdout-dir PATH         Frozen release-holdout manifests
  --source-commit COMMIT             Exact source commit being evaluated

Experiment controls:
  --random-seed INTEGER              Shared folds/fits seed (default: 42)
  --evaluation MODE                  none, baselines, or all (default: all)

Experiment archiving:
  --experiments-dir PATH             Write a timestamped snapshot after success
  --experiment-name NAME             Snapshot label
  --source-archive PATH              Preserve exact source archive in snapshot

Execution controls:
  --gpus INTEGER|auto                GPU count (default: auto)
  --num-jobs INTEGER|auto            Concurrent jobs (default: auto)
  --max-workers-per-gpu INTEGER|auto Worker density (default: auto)
  --dataloader-num-workers N|auto    Workers per dataloader (default: 1)
  --max-tasks-per-worker INTEGER     Worker recycling interval (default: 12)
  -h, --help                         Show this help
EOF
}

require_value() {
    if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        printf 'Missing value for %s\n' "$1" >&2
        usage >&2
        exit 2
    fi
}

OUT=""
TRAIN_DATA=""
DATA_EVAL_DIR=""
RELEASE_HOLDOUT_DIR=""
SOURCE_COMMIT=""
RANDOM_SEED=42
EVALUATION=all
EXPERIMENTS_DIR=""
EXPERIMENT_NAME="processing-5aa-regularization-activation"
SOURCE_ARCHIVE=""
GPUS=auto
NUM_JOBS=auto
MAX_WORKERS_PER_GPU=auto
DATALOADER_NUM_WORKERS=1
MAX_TASKS_PER_WORKER=12
ORIGINAL_ARGS=("$@")

while [ "$#" -gt 0 ]; do
    case "$1" in
        --out) require_value "$@"; OUT="$2"; shift 2 ;;
        --train-data) require_value "$@"; TRAIN_DATA="$2"; shift 2 ;;
        --data-eval-dir) require_value "$@"; DATA_EVAL_DIR="$2"; shift 2 ;;
        --release-holdout-dir)
            require_value "$@"; RELEASE_HOLDOUT_DIR="$2"; shift 2 ;;
        --source-commit) require_value "$@"; SOURCE_COMMIT="$2"; shift 2 ;;
        --random-seed) require_value "$@"; RANDOM_SEED="$2"; shift 2 ;;
        --evaluation) require_value "$@"; EVALUATION="$2"; shift 2 ;;
        --experiments-dir)
            require_value "$@"; EXPERIMENTS_DIR="$2"; shift 2 ;;
        --experiment-name)
            require_value "$@"; EXPERIMENT_NAME="$2"; shift 2 ;;
        --source-archive) require_value "$@"; SOURCE_ARCHIVE="$2"; shift 2 ;;
        --gpus) require_value "$@"; GPUS="$2"; shift 2 ;;
        --num-jobs) require_value "$@"; NUM_JOBS="$2"; shift 2 ;;
        --max-workers-per-gpu)
            require_value "$@"; MAX_WORKERS_PER_GPU="$2"; shift 2 ;;
        --dataloader-num-workers)
            require_value "$@"; DATALOADER_NUM_WORKERS="$2"; shift 2 ;;
        --max-tasks-per-worker)
            require_value "$@"; MAX_TASKS_PER_WORKER="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for required in OUT TRAIN_DATA DATA_EVAL_DIR RELEASE_HOLDOUT_DIR SOURCE_COMMIT; do
    if [ -z "${!required}" ]; then
        printf 'Missing required argument for %s\n' "$required" >&2
        usage >&2
        exit 2
    fi
done
case "$EVALUATION" in
    none|baselines|all) ;;
    *) printf 'Invalid --evaluation: %s\n' "$EVALUATION" >&2; exit 2 ;;
esac
if [ ! -f "$TRAIN_DATA" ]; then
    printf 'Training data is not a file: %s\n' "$TRAIN_DATA" >&2
    exit 2
fi
if [ ! -d "$DATA_EVAL_DIR" ]; then
    printf 'Evaluation data is not a directory: %s\n' "$DATA_EVAL_DIR" >&2
    exit 2
fi
for holdout_file in policy.json processing_samples.csv; do
    if [ ! -f "$RELEASE_HOLDOUT_DIR/$holdout_file" ]; then
        printf 'Missing release holdout file: %s\n' "$holdout_file" >&2
        exit 2
    fi
done
if [ -n "$SOURCE_ARCHIVE" ] && [ ! -f "$SOURCE_ARCHIVE" ]; then
    printf 'Source archive is not a file: %s\n' "$SOURCE_ARCHIVE" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$GPUS" = auto ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    else
        GPUS=0
    fi
fi

mkdir -p "$OUT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/started_at_utc.txt"
{
    printf 'bash %q' "$0"
    printf ' %q' "${ORIGINAL_ARGS[@]}"
    printf '\n'
} > "$OUT/command.sh"
python "$SCRIPT_DIR/generate_processing_regularization_activation.py" \
    "$OUT" > "$OUT/manifest.stdout.json"

{
    printf '%s\n' \
        "schema_version=1" \
        "source_commit=$SOURCE_COMMIT" \
        "design=processing-5aa-regularization-activation" \
        "random_seed=$RANDOM_SEED" \
        "evaluation=$EVALUATION" \
        "gpus=$GPUS" \
        "num_jobs=$NUM_JOBS" \
        "max_workers_per_gpu=$MAX_WORKERS_PER_GPU" \
        "dataloader_num_workers=$DATALOADER_NUM_WORKERS" \
        "max_tasks_per_worker=$MAX_TASKS_PER_WORKER"
    sha256sum \
        "$TRAIN_DATA" \
        "$RELEASE_HOLDOUT_DIR/policy.json" \
        "$RELEASE_HOLDOUT_DIR/processing_samples.csv" \
        "$OUT/manifest.json"
} > "$OUT/provenance.txt"

COMMON_PARALLELISM_ARGS=(
    --num-jobs "$NUM_JOBS"
    --max-tasks-per-worker "$MAX_TASKS_PER_WORKER"
    --gpus "$GPUS"
    --max-workers-per-gpu "$MAX_WORKERS_PER_GPU"
    --torch-compile 0
    --matmul-precision highest
)
TRAINING_PARALLELISM_ARGS=(
    "${COMMON_PARALLELISM_ARGS[@]}"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/gpu_telemetry.sh"
GPU_TELEMETRY_PID=""
trap stop_gpu_telemetry EXIT
start_gpu_telemetry "$OUT/gpu_occupancy.csv"

tail -n +2 "$OUT/manifest.csv" | cut -d, -f1 | \
while IFS= read -r condition; do
    condition_out="$OUT/$condition"
    processing_out="$condition_out/processing"
    unselected="$processing_out/models.unselected.short_flanks"
    selected="$processing_out/models.selected.short_flanks"
    hyperparameters="$OUT/conditions/$condition.yaml"
    mkdir -p "$processing_out"
    if [ -f "$unselected/manifest.csv" ]; then
        mhcflurry-class1-train-processing-models \
            --out-models-dir "$unselected" \
            --continue-incomplete \
            "${TRAINING_PARALLELISM_ARGS[@]}"
    else
        mhcflurry-class1-train-processing-models \
            --data "$TRAIN_DATA" \
            --held-out-samples 10 \
            --num-folds 4 \
            --random-seed "$RANDOM_SEED" \
            --hyperparameters "$hyperparameters" \
            --out-models-dir "$unselected" \
            --worker-log-dir "$processing_out" \
            "${TRAINING_PARALLELISM_ARGS[@]}"
    fi
    if [ ! -f "$selected/train_data.csv.bz2" ]; then
        mhcflurry-class1-select-processing-models \
            --data "$unselected/train_data.csv.bz2" \
            --models-dir "$unselected" \
            --out-models-dir "$selected" \
            --min-models-per-fold 1 \
            --max-models-per-fold 1 \
            "${TRAINING_PARALLELISM_ARGS[@]}"
        cp "$unselected/train_data.csv.bz2" "$selected/train_data.csv.bz2"
    fi
    printf '%s\n' "$condition_out" > "$condition_out/predictor_path.txt"
    if [ ! -f "$condition_out/.loss-plots.done" ]; then
        mhcflurry train plot-loss-curves \
            --selected-dir "$selected" \
            --unselected-dir "$unselected" \
            --out "$condition_out/loss_plots"
        date -u +%Y-%m-%dT%H:%M:%SZ > "$condition_out/.loss-plots.done"
    fi
done

stop_gpu_telemetry
GPU_TELEMETRY_PID=""

if [ "$EVALUATION" != none ]; then
    tail -n +2 "$OUT/manifest.csv" | \
    while IFS=, read -r condition architecture baseline _rest; do
        if [ "$condition" != "$baseline" ] && [ "$EVALUATION" = baselines ]; then
            continue
        fi
        comparison="$OUT/$condition/comparison-vs-public"
        if [ ! -f "$comparison/.done" ]; then
            mkdir -p "$comparison"
            mhcflurry eval compare-models \
                --a "$OUT/$condition" \
                --a-label "$condition" \
                --b public \
                --b-label public \
                --data-dir "$DATA_EVAL_DIR" \
                --release-holdout-dir "$RELEASE_HOLDOUT_DIR" \
                --include processing \
                --processing-modes short_flanks \
                --out "$comparison" \
                "${COMMON_PARALLELISM_ARGS[@]}"
            mhcflurry plot-model-comparison \
                --input "$comparison" \
                --components processing \
                --summary-pdf "$comparison/plots/model_comparison_figures.pdf"
            date -u +%Y-%m-%dT%H:%M:%SZ > "$comparison/.done"
        fi
        printf 'Evaluated %s (%s)\n' "$condition" "$architecture"
    done
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/completed_at_utc.txt"

if [ -n "$EXPERIMENTS_DIR" ]; then
    snapshot_args=(
        --source-dir "$OUT"
        --experiments-dir "$EXPERIMENTS_DIR"
        --name "$EXPERIMENT_NAME"
        --source-commit "$SOURCE_COMMIT"
        --command-file "$OUT/command.sh"
        --input-file "$TRAIN_DATA"
        --input-file "$RELEASE_HOLDOUT_DIR/policy.json"
        --input-file "$RELEASE_HOLDOUT_DIR/processing_samples.csv"
    )
    if [ -n "$SOURCE_ARCHIVE" ]; then
        snapshot_args+=(--source-archive "$SOURCE_ARCHIVE")
    fi
    snapshot_path="$(mhcflurry train snapshot-experiment "${snapshot_args[@]}")"
    printf '%s\n' "$snapshot_path" | tee "$OUT/snapshot_path.txt"
fi
