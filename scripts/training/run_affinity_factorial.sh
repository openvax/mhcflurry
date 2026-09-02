#!/usr/bin/env bash
# Train and evaluate the controlled affinity recipe sweep.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_affinity_factorial.sh [OPTIONS]

Required:
  --out PATH                         Experiment output directory
  --train-data PATH                  Affinity training measurements
  --allele-sequences PATH            Allele pseudosequences CSV
  --pretrain-data PATH               Affinity pretraining predictions
  --data-eval-dir PATH               Evaluation data directory
  --release-holdout-dir PATH         Frozen release-holdout manifests
  --source-commit COMMIT             Exact source commit being evaluated

Experiment controls:
  --mode MODE                        representative (default) or full
  --condition NAME                   Run one generated condition; repeatable
  --random-seed INTEGER              Release random seed (default: 42)

Experiment archiving:
  --experiments-dir PATH             Write a timestamped snapshot after success
  --experiment-name NAME             Snapshot label (default: affinity-factorial)
  --source-archive PATH              Preserve the exact source archive in snapshot

Execution controls:
  --gpus INTEGER|auto                GPU count (default: auto)
  --max-workers-per-gpu INTEGER|auto Worker density (default: auto)
  --dataloader-num-workers N|auto    Workers per training dataloader (default: 1)
  --max-tasks-per-worker INTEGER     Worker recycling interval (default: 12)
  --torch-compile VALUE              auto, 0, or 1 (default: 0)
  --torch-compile-loss VALUE         auto, 0, or 1 (default: 0)
  --matmul-precision VALUE           none, highest, high, or medium
                                      (default: highest)
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

MHCFLURRY_OUT=""
TRAIN_DATA=""
ALLELE_SEQUENCES=""
PRETRAIN_DATA=""
DATA_EVAL_DIR=""
RELEASE_HOLDOUT_DIR=""
SOURCE_COMMIT=""
FACTORIAL_MODE="representative"
RELEASE_RANDOM_SEED=42
MAX_TASKS_PER_WORKER=12
MAX_WORKERS_PER_GPU="auto"
DATALOADER_NUM_WORKERS=1
GPUS="auto"
TORCH_COMPILE=0
TORCH_COMPILE_LOSS=0
MATMUL_PRECISION="highest"
FACTORIAL_CONDITIONS=()
EXPERIMENTS_DIR=""
EXPERIMENT_NAME="affinity-factorial"
SOURCE_ARCHIVE=""
ORIGINAL_ARGS=("$@")

while [ "$#" -gt 0 ]; do
    case "$1" in
        --out)
            require_value "$@"; MHCFLURRY_OUT="$2"; shift 2 ;;
        --train-data)
            require_value "$@"; TRAIN_DATA="$2"; shift 2 ;;
        --allele-sequences)
            require_value "$@"; ALLELE_SEQUENCES="$2"; shift 2 ;;
        --pretrain-data)
            require_value "$@"; PRETRAIN_DATA="$2"; shift 2 ;;
        --data-eval-dir)
            require_value "$@"; DATA_EVAL_DIR="$2"; shift 2 ;;
        --release-holdout-dir)
            require_value "$@"; RELEASE_HOLDOUT_DIR="$2"; shift 2 ;;
        --source-commit)
            require_value "$@"; SOURCE_COMMIT="$2"; shift 2 ;;
        --mode)
            require_value "$@"; FACTORIAL_MODE="$2"; shift 2 ;;
        --condition)
            require_value "$@"; FACTORIAL_CONDITIONS+=("$2"); shift 2 ;;
        --random-seed)
            require_value "$@"; RELEASE_RANDOM_SEED="$2"; shift 2 ;;
        --experiments-dir)
            require_value "$@"; EXPERIMENTS_DIR="$2"; shift 2 ;;
        --experiment-name)
            require_value "$@"; EXPERIMENT_NAME="$2"; shift 2 ;;
        --source-archive)
            require_value "$@"; SOURCE_ARCHIVE="$2"; shift 2 ;;
        --gpus)
            require_value "$@"; GPUS="$2"; shift 2 ;;
        --max-workers-per-gpu)
            require_value "$@"; MAX_WORKERS_PER_GPU="$2"; shift 2 ;;
        --dataloader-num-workers)
            require_value "$@"; DATALOADER_NUM_WORKERS="$2"; shift 2 ;;
        --max-tasks-per-worker)
            require_value "$@"; MAX_TASKS_PER_WORKER="$2"; shift 2 ;;
        --torch-compile)
            require_value "$@"; TORCH_COMPILE="$2"; shift 2 ;;
        --torch-compile-loss)
            require_value "$@"; TORCH_COMPILE_LOSS="$2"; shift 2 ;;
        --matmul-precision)
            require_value "$@"; MATMUL_PRECISION="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2 ;;
    esac
done
if [ -n "$SOURCE_ARCHIVE" ] && [ ! -f "$SOURCE_ARCHIVE" ]; then
    printf '%s is not a file: %s\n' '--source-archive' "$SOURCE_ARCHIVE" >&2
    exit 2
fi

for required in \
        MHCFLURRY_OUT TRAIN_DATA ALLELE_SEQUENCES PRETRAIN_DATA \
        DATA_EVAL_DIR RELEASE_HOLDOUT_DIR SOURCE_COMMIT; do
    if [ -z "${!required}" ]; then
        printf 'Missing required argument for %s\n' "$required" >&2
        usage >&2
        exit 2
    fi
done

case "$FACTORIAL_MODE" in
    representative|full) ;;
    *) printf 'Invalid --mode: %s\n' "$FACTORIAL_MODE" >&2; exit 2 ;;
esac
case "$MATMUL_PRECISION" in
    none|highest|high|medium) ;;
    *) printf 'Invalid --matmul-precision: %s\n' "$MATMUL_PRECISION" >&2; exit 2 ;;
esac
case "$TORCH_COMPILE" in
    auto|0|1) ;;
    *) printf 'Invalid --torch-compile: %s\n' "$TORCH_COMPILE" >&2; exit 2 ;;
esac
case "$TORCH_COMPILE_LOSS" in
    auto|0|1) ;;
    *) printf 'Invalid --torch-compile-loss: %s\n' "$TORCH_COMPILE_LOSS" >&2; exit 2 ;;
esac
for value_and_name in \
        "$RELEASE_RANDOM_SEED:--random-seed" \
        "$MAX_TASKS_PER_WORKER:--max-tasks-per-worker"; do
    value="${value_and_name%%:*}"
    name="${value_and_name#*:}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        printf '%s must be a nonnegative integer: %s\n' "$name" "$value" >&2
        exit 2
    fi
done
if [ "$DATALOADER_NUM_WORKERS" != "auto" ] && \
        ! [[ "$DATALOADER_NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    printf '%s must be auto or a nonnegative integer: %s\n' \
        '--dataloader-num-workers' "$DATALOADER_NUM_WORKERS" >&2
    exit 2
fi
if [ "$MAX_TASKS_PER_WORKER" -eq 0 ]; then
    printf '%s must be positive\n' '--max-tasks-per-worker' >&2
    exit 2
fi
if [ "$MAX_WORKERS_PER_GPU" != "auto" ] && \
        ! [[ "$MAX_WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be auto or a positive integer: %s\n' \
        '--max-workers-per-gpu' "$MAX_WORKERS_PER_GPU" >&2
    exit 2
fi
if [ "$GPUS" != "auto" ] && ! [[ "$GPUS" =~ ^[0-9]+$ ]]; then
    printf '%s must be auto or a nonnegative integer: %s\n' \
        '--gpus' "$GPUS" >&2
    exit 2
fi

for path_and_name in \
        "$TRAIN_DATA:--train-data" \
        "$ALLELE_SEQUENCES:--allele-sequences" \
        "$PRETRAIN_DATA:--pretrain-data"; do
    path="${path_and_name%%:*}"
    name="${path_and_name#*:}"
    if [ ! -f "$path" ]; then
        printf '%s is not a file: %s\n' "$name" "$path" >&2
        exit 2
    fi
done
for path_and_name in \
        "$DATA_EVAL_DIR:--data-eval-dir" \
        "$RELEASE_HOLDOUT_DIR:--release-holdout-dir"; do
    path="${path_and_name%%:*}"
    name="${path_and_name#*:}"
    if [ ! -d "$path" ]; then
        printf '%s is not a directory: %s\n' "$name" "$path" >&2
        exit 2
    fi
done
for holdout_file in \
        policy.json affinity_samples.csv affinity_pmhcs.csv; do
    if [ ! -f "$RELEASE_HOLDOUT_DIR/$holdout_file" ]; then
        printf '%s is missing required file: %s\n' \
            '--release-holdout-dir' "$holdout_file" >&2
        exit 2
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONUNBUFFERED=1
# Training-batch shrinkage would confound this experiment. The library guard
# currently has no CLI equivalent, so keep this one invariant in the runtime
# environment and verify the effective minibatch in every saved model.
export MHCFLURRY_FAIL_ON_TRAINING_BATCH_SHRINK=1

if [ "$GPUS" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
    else
        GPUS=0
    fi
fi

mkdir -p "$MHCFLURRY_OUT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$MHCFLURRY_OUT/started_at_utc.txt"
{
    printf 'bash %q' "$0"
    printf ' %q' "${ORIGINAL_ARGS[@]}"
    printf '\n'
} > "$MHCFLURRY_OUT/command.sh"
python "$SCRIPT_DIR/generate_affinity_factorial.py" \
    "$MHCFLURRY_OUT" \
    --mode "$FACTORIAL_MODE" \
    > "$MHCFLURRY_OUT/manifest.stdout.json"

BASELINE_CONDITION="$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["baseline_condition"])' \
    "$MHCFLURRY_OUT/manifest.json")"

for selected in "${FACTORIAL_CONDITIONS[@]}"; do
    if ! tail -n +2 "$MHCFLURRY_OUT/manifest.csv" | cut -d, -f1 | \
            grep -Fqx -- "$selected"; then
        printf 'Unknown --condition: %s\n' "$selected" >&2
        exit 2
    fi
done

condition_selected() {
    local condition="$1"
    if [ "${#FACTORIAL_CONDITIONS[@]}" -eq 0 ] || \
            [ "$condition" = "$BASELINE_CONDITION" ]; then
        return 0
    fi
    local selected
    for selected in "${FACTORIAL_CONDITIONS[@]}"; do
        if [ "$selected" = "$condition" ]; then
            return 0
        fi
    done
    return 1
}

if [ "${#FACTORIAL_CONDITIONS[@]}" -eq 0 ]; then
    FACTORIAL_CONDITIONS_PROVENANCE=all
else
    FACTORIAL_CONDITIONS_PROVENANCE="$(
        IFS=,; printf '%s' "${FACTORIAL_CONDITIONS[*]}"
    )"
fi

AFFINITY_PREDICTION_ARTIFACT_ARGS=()
if [ "$FACTORIAL_MODE" = "representative" ]; then
    AFFINITY_PREDICTION_ARTIFACT_ARGS=(--skip-affinity-predictions)
fi

{
    printf '%s\n' \
        "schema_version=1" \
        "source_commit=$SOURCE_COMMIT" \
        "factorial_mode=$FACTORIAL_MODE" \
        "factorial_conditions=$FACTORIAL_CONDITIONS_PROVENANCE" \
        "random_seed=$RELEASE_RANDOM_SEED" \
        "gpus=$GPUS" \
        "max_workers_per_gpu=$MAX_WORKERS_PER_GPU" \
        "dataloader_num_workers=$DATALOADER_NUM_WORKERS" \
        "max_tasks_per_worker=$MAX_TASKS_PER_WORKER" \
        "torch_compile=$TORCH_COMPILE" \
        "torch_compile_loss=$TORCH_COMPILE_LOSS" \
        "matmul_precision=$MATMUL_PRECISION" \
        "experiments_dir=$EXPERIMENTS_DIR" \
        "experiment_name=$EXPERIMENT_NAME" \
        "source_archive=$SOURCE_ARCHIVE" \
        "baseline_condition=$BASELINE_CONDITION"
    sha256sum \
        "$TRAIN_DATA" \
        "$ALLELE_SEQUENCES" \
        "$PRETRAIN_DATA" \
        "$RELEASE_HOLDOUT_DIR/policy.json" \
        "$RELEASE_HOLDOUT_DIR/affinity_samples.csv" \
        "$RELEASE_HOLDOUT_DIR/affinity_pmhcs.csv" \
        "$MHCFLURRY_OUT/manifest.json"
} > "$MHCFLURRY_OUT/provenance.txt"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/gpu_telemetry.sh"
GPU_TELEMETRY_PID=""
trap stop_gpu_telemetry EXIT
start_gpu_telemetry "$MHCFLURRY_OUT/gpu_occupancy.csv"

tail -n +2 "$MHCFLURRY_OUT/manifest.csv" | cut -d, -f1 | \
while IFS= read -r condition; do
    if ! condition_selected "$condition"; then
        continue
    fi
    condition_out="$MHCFLURRY_OUT/$condition"
    unselected="$condition_out/models.unselected.combined"
    train_done="$condition_out/.train.done"
    mkdir -p "$condition_out"
    if [ -f "$train_done" ]; then
        python "$SCRIPT_DIR/verify_affinity_factorial_models.py" \
            "$unselected" \
            "$MHCFLURRY_OUT/conditions/$condition.yaml" \
            > "$condition_out/verification.json"
        printf 'Training already complete: %s\n' "$condition"
        continue
    fi
    continue_args=()
    if [ -f "$unselected/manifest.csv" ]; then
        continue_args=(--continue-incomplete)
    fi
    mhcflurry-class1-train-pan-allele-models \
        --data "$TRAIN_DATA" \
        --allele-sequences "$ALLELE_SEQUENCES" \
        --pretrain-data "$PRETRAIN_DATA" \
        --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
        --num-folds 4 \
        --random-seed "$RELEASE_RANDOM_SEED" \
        --hyperparameters "$MHCFLURRY_OUT/conditions/$condition.yaml" \
        --out-models-dir "$unselected" \
        --worker-log-dir "$condition_out" \
        --num-jobs auto \
        --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
        --gpus "$GPUS" \
        --max-workers-per-gpu "$MAX_WORKERS_PER_GPU" \
        --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
        --random-negative-pool-epochs 1 \
        --torch-compile "$TORCH_COMPILE" \
        --torch-compile-loss "$TORCH_COMPILE_LOSS" \
        --matmul-precision "$MATMUL_PRECISION" \
        --enable-timing \
        "${continue_args[@]}" \
        2>&1 | tee "$condition_out/train.log"
    python "$SCRIPT_DIR/verify_affinity_factorial_models.py" \
        "$unselected" \
        "$MHCFLURRY_OUT/conditions/$condition.yaml" \
        > "$condition_out/verification.json"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$train_done"
done

tail -n +2 "$MHCFLURRY_OUT/manifest.csv" | cut -d, -f1 | \
while IFS= read -r condition; do
    if ! condition_selected "$condition"; then
        continue
    fi
    condition_out="$MHCFLURRY_OUT/$condition"
    unselected="$condition_out/models.unselected.combined"
    if [ "$FACTORIAL_MODE" = "full" ]; then
        predictor="$condition_out/models.combined"
        select_done="$condition_out/.select.done"
        if [ ! -f "$select_done" ]; then
            mhcflurry-class1-select-pan-allele-models \
                --data "$unselected/train_data.csv.bz2" \
                --models-dir "$unselected" \
                --out-models-dir "$predictor" \
                --min-models-per-fold 2 \
                --max-models-per-fold 8 \
                --num-jobs auto \
                --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
                --gpus "$GPUS" \
                --max-workers-per-gpu "$MAX_WORKERS_PER_GPU" \
                --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
                --torch-compile "$TORCH_COMPILE" \
                --torch-compile-loss "$TORCH_COMPILE_LOSS" \
                --matmul-precision "$MATMUL_PRECISION" \
                --enable-timing \
                2>&1 | tee "$condition_out/select.log"
            cp "$unselected/train_data.csv.bz2" "$predictor/train_data.csv.bz2"
            date -u +%Y-%m-%dT%H:%M:%SZ > "$select_done"
        fi
    else
        predictor="$unselected"
    fi
    printf '%s\n' "$predictor" > "$condition_out/predictor_path.txt"
done

baseline_predictor="$(cat \
    "$MHCFLURRY_OUT/$BASELINE_CONDITION/predictor_path.txt")"
baseline_eval="$MHCFLURRY_OUT/baseline-vs-public"
if [ ! -f "$baseline_eval/.done" ]; then
    mkdir -p "$baseline_eval"
    mhcflurry eval compare-models \
        --a "$baseline_predictor" \
        --a-label "$BASELINE_CONDITION" \
        --b public \
        --data-dir "$DATA_EVAL_DIR" \
        --release-holdout-dir "$RELEASE_HOLDOUT_DIR" \
        --affinity-training-overlap-policy audit \
        "${AFFINITY_PREDICTION_ARTIFACT_ARGS[@]}" \
        --include affinity \
        --out "$baseline_eval" \
        --num-jobs auto \
        --gpus "$GPUS" \
        --max-workers-per-gpu "$MAX_WORKERS_PER_GPU" \
        --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
        --torch-compile "$TORCH_COMPILE" \
        --matmul-precision "$MATMUL_PRECISION" \
        2>&1 | tee "$baseline_eval/eval.log"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$baseline_eval/.done"
fi

tail -n +2 "$MHCFLURRY_OUT/manifest.csv" | cut -d, -f1 | \
while IFS= read -r condition; do
    if ! condition_selected "$condition"; then
        continue
    fi
    if [ "$condition" = "$BASELINE_CONDITION" ]; then
        continue
    fi
    condition_out="$MHCFLURRY_OUT/$condition"
    comparison="$condition_out/comparison-vs-baseline"
    predictor="$(cat "$condition_out/predictor_path.txt")"
    if [ ! -f "$comparison/.done" ]; then
        mkdir -p "$comparison"
        mhcflurry eval compare-models \
            --a "$predictor" \
            --a-label "$condition" \
            --b "$baseline_predictor" \
            --b-label "$BASELINE_CONDITION" \
            --data-dir "$DATA_EVAL_DIR" \
            --release-holdout-dir "$RELEASE_HOLDOUT_DIR" \
            --affinity-training-overlap-policy audit \
            "${AFFINITY_PREDICTION_ARTIFACT_ARGS[@]}" \
            --include affinity \
            --out "$comparison" \
            --num-jobs auto \
            --gpus "$GPUS" \
            --max-workers-per-gpu "$MAX_WORKERS_PER_GPU" \
            --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
            --torch-compile "$TORCH_COMPILE" \
            --matmul-precision "$MATMUL_PRECISION" \
            2>&1 | tee "$comparison/eval.log"
        date -u +%Y-%m-%dT%H:%M:%SZ > "$comparison/.done"
    fi
    if [ "$FACTORIAL_MODE" = "full" ]; then
        public_comparison="$condition_out/comparison-vs-public"
        if [ ! -f "$public_comparison/.done" ]; then
            mkdir -p "$public_comparison"
            mhcflurry eval compare-models \
                --a "$predictor" \
                --a-label "$condition" \
                --b public \
                --data-dir "$DATA_EVAL_DIR" \
                --release-holdout-dir "$RELEASE_HOLDOUT_DIR" \
                --affinity-training-overlap-policy audit \
                --include affinity \
                --out "$public_comparison" \
                --num-jobs auto \
                --gpus "$GPUS" \
                --max-workers-per-gpu "$MAX_WORKERS_PER_GPU" \
                --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
                --torch-compile "$TORCH_COMPILE" \
                --matmul-precision "$MATMUL_PRECISION" \
                2>&1 | tee "$public_comparison/eval.log"
            date -u +%Y-%m-%dT%H:%M:%SZ > "$public_comparison/.done"
        fi
    fi
    python "$SCRIPT_DIR/summarize_affinity_factorial.py" \
        "$MHCFLURRY_OUT" > "$MHCFLURRY_OUT/summary.stdout.json"
done

python "$SCRIPT_DIR/summarize_affinity_factorial.py" \
    "$MHCFLURRY_OUT" > "$MHCFLURRY_OUT/summary.stdout.json"

date -u +%Y-%m-%dT%H:%M:%SZ > "$MHCFLURRY_OUT/completed_at_utc.txt"
stop_gpu_telemetry
GPU_TELEMETRY_PID=""

if [ -n "$EXPERIMENTS_DIR" ]; then
    snapshot_args=(
        --source-dir "$MHCFLURRY_OUT"
        --experiments-dir "$EXPERIMENTS_DIR"
        --name "$EXPERIMENT_NAME"
        --source-commit "$SOURCE_COMMIT"
        --command-file "$MHCFLURRY_OUT/command.sh"
        --input-file "$TRAIN_DATA"
        --input-file "$ALLELE_SEQUENCES"
        --input-file "$PRETRAIN_DATA"
        --input-file "$RELEASE_HOLDOUT_DIR/policy.json"
        --input-file "$RELEASE_HOLDOUT_DIR/affinity_samples.csv"
        --input-file "$RELEASE_HOLDOUT_DIR/affinity_pmhcs.csv"
    )
    if [ -n "$SOURCE_ARCHIVE" ]; then
        snapshot_args+=(--source-archive "$SOURCE_ARCHIVE")
    fi
    snapshot_path="$(mhcflurry train snapshot-experiment "${snapshot_args[@]}")"
    printf '%s\n' "$snapshot_path" | tee "$MHCFLURRY_OUT/snapshot_path.txt"
fi
