#!/usr/bin/env bash
# Run the focused processing minibatch sweep without repeating prior ablations.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_processing_batch_sweep.sh \
    --out DIR \
    --train-data CSV.BZ2 \
    --data-eval-dir DIR \
    --release-holdout-dir DIR \
    --source-commit COMMIT \
    [--gpus N] [--num-jobs N|auto] \
    [--max-workers-per-gpu N|auto] \
    [--dataloader-num-workers N|auto] \
    [--max-tasks-per-worker N] [--random-seed N]

Trains the controlled 128/256/512/1024 processing batch panels. The primary
mode is 5-aa flanks and the secondary mode is no flanks. A broad 15-aa sweep
is intentionally excluded.
EOF
}

OUT=""
TRAIN_DATA=""
DATA_EVAL_DIR=""
RELEASE_HOLDOUT_DIR=""
SOURCE_COMMIT=""
GPUS="auto"
NUM_JOBS="auto"
MAX_WORKERS_PER_GPU="auto"
DATALOADER_NUM_WORKERS="1"
MAX_TASKS_PER_WORKER="12"
RANDOM_SEED="42"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --train-data) TRAIN_DATA="$2"; shift 2 ;;
        --data-eval-dir) DATA_EVAL_DIR="$2"; shift 2 ;;
        --release-holdout-dir) RELEASE_HOLDOUT_DIR="$2"; shift 2 ;;
        --source-commit) SOURCE_COMMIT="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --num-jobs) NUM_JOBS="$2"; shift 2 ;;
        --max-workers-per-gpu) MAX_WORKERS_PER_GPU="$2"; shift 2 ;;
        --dataloader-num-workers) DATALOADER_NUM_WORKERS="$2"; shift 2 ;;
        --max-tasks-per-worker) MAX_TASKS_PER_WORKER="$2"; shift 2 ;;
        --random-seed) RANDOM_SEED="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for required in OUT TRAIN_DATA DATA_EVAL_DIR RELEASE_HOLDOUT_DIR SOURCE_COMMIT; do
    if [ -z "${!required}" ]; then
        echo "Missing required argument for $required" >&2
        usage >&2
        exit 2
    fi
done
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Training data not found: $TRAIN_DATA" >&2
    exit 2
fi
if [ ! -f "$RELEASE_HOLDOUT_DIR/processing_samples.csv" ]; then
    echo "Processing holdout manifest not found in: $RELEASE_HOLDOUT_DIR" >&2
    exit 2
fi
if [ ! -d "$DATA_EVAL_DIR" ]; then
    echo "Evaluation data directory not found: $DATA_EVAL_DIR" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/gpu_telemetry.sh"
GPU_TELEMETRY_PID=""
trap stop_gpu_telemetry EXIT

if [ "$GPUS" = auto ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    else
        GPUS=0
    fi
fi

PANEL_DIR="$OUT/hyperparameter_panels"
PAIRWISE_DIR="$OUT/paired_comparisons"
mkdir -p "$PANEL_DIR" "$PAIRWISE_DIR"

python "$SCRIPT_DIR/generate_release_hyperparameter_ablations.py" "$PANEL_DIR" \
    > "$OUT/panel_manifest.stdout.json"

{
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'train_data=%s\n' "$TRAIN_DATA"
    printf 'train_data_sha256='
    sha256sum "$TRAIN_DATA" | cut -d' ' -f1
    printf 'release_holdout_dir=%s\n' "$RELEASE_HOLDOUT_DIR"
    printf 'processing_holdout_sha256='
    sha256sum "$RELEASE_HOLDOUT_DIR/processing_samples.csv" | cut -d' ' -f1
    printf 'batch_sizes=128,256,512,1024\n'
    printf 'mode_priority=short_flanks,no_flank\n'
    printf 'gpus=%s\n' "$GPUS"
    printf 'num_jobs=%s\n' "$NUM_JOBS"
    printf 'max_workers_per_gpu=%s\n' "$MAX_WORKERS_PER_GPU"
    printf 'dataloader_num_workers=%s\n' "$DATALOADER_NUM_WORKERS"
    printf 'max_tasks_per_worker=%s\n' "$MAX_TASKS_PER_WORKER"
    printf 'random_seed=%s\n' "$RANDOM_SEED"
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

train_panel() {
    local minibatch_size="$1"
    local variant="$2"
    local condition_root="$OUT/processing_batch_sweep.batch$minibatch_size"
    local processing_root="$condition_root/processing"
    local unselected="$processing_root/models.unselected.$variant"
    local selected="$processing_root/models.selected.$variant"
    local hyperparameters="$PANEL_DIR/processing_batch_sweep.batch$minibatch_size.$variant.yaml"

    mkdir -p "$processing_root"
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
            --worker-log-dir "$processing_root" \
            "${TRAINING_PARALLELISM_ARGS[@]}"
    fi

    if [ ! -f "$selected/train_data.csv.bz2" ]; then
        mhcflurry-class1-select-processing-models \
            --data "$unselected/train_data.csv.bz2" \
            --models-dir "$unselected" \
            --out-models-dir "$selected" \
            --min-models-per-fold 1 \
            --max-models-per-fold 2 \
            "${TRAINING_PARALLELISM_ARGS[@]}"
        cp "$unselected/train_data.csv.bz2" \
            "$selected/train_data.csv.bz2"
    fi
}

for minibatch_size in 128 256 512 1024; do
    processing_root="$OUT/processing_batch_sweep.batch$minibatch_size/processing"
    mkdir -p "$processing_root"
    start_gpu_telemetry "$processing_root/gpu_occupancy.csv"
    train_panel "$minibatch_size" short_flanks
    train_panel "$minibatch_size" no_flank
    stop_gpu_telemetry
done

batch512="$OUT/processing_batch_sweep.batch512"
for minibatch_size in 128 256 1024; do
    mhcflurry eval compare-models \
        --a "$OUT/processing_batch_sweep.batch$minibatch_size" \
        --a-label "processing_batch_$minibatch_size" \
        --b "$batch512" \
        --b-label processing_batch_512 \
        --data-dir "$DATA_EVAL_DIR" \
        --release-holdout-dir "$RELEASE_HOLDOUT_DIR" \
        --include processing \
        --processing-modes short_flanks,no_flank \
        --out "$PAIRWISE_DIR/processing_batch_$minibatch_size-vs-512" \
        "${COMMON_PARALLELISM_ARGS[@]}"
done

mhcflurry eval compare-models \
    --a "$batch512" \
    --a-label processing_batch_512 \
    --b public \
    --b-label public \
    --data-dir "$DATA_EVAL_DIR" \
    --release-holdout-dir "$RELEASE_HOLDOUT_DIR" \
    --include processing \
    --processing-modes short_flanks,no_flank \
    --out "$PAIRWISE_DIR/processing_batch_512-vs-public" \
    "${COMMON_PARALLELISM_ARGS[@]}"
