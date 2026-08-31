#!/usr/bin/env bash
# Train and evaluate the controlled affinity recipe sweep.
set -euo pipefail

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
: "${TRAIN_DATA:?TRAIN_DATA must be set}"
: "${ALLELE_SEQUENCES:?ALLELE_SEQUENCES must be set}"
: "${PRETRAIN_DATA:?PRETRAIN_DATA must be set}"
: "${DATA_EVAL_DIR:?DATA_EVAL_DIR must be set}"
: "${RELEASE_HOLDOUT_DIR:?RELEASE_HOLDOUT_DIR must be set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACTORIAL_MODE="${FACTORIAL_MODE:-representative}"
RELEASE_RANDOM_SEED="${RELEASE_RANDOM_SEED:-42}"
MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-12}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-auto}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
FACTORIAL_CONDITIONS="${FACTORIAL_CONDITIONS:-}"

export PYTHONUNBUFFERED=1
export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-0}"
export MHCFLURRY_TORCH_COMPILE_LOSS="${MHCFLURRY_TORCH_COMPILE_LOSS:-0}"
export MHCFLURRY_MATMUL_PRECISION="${MHCFLURRY_MATMUL_PRECISION:-highest}"
export MHCFLURRY_FAIL_ON_TRAINING_BATCH_SHRINK=1

if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="${GPUS:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
else
    GPUS="${GPUS:-0}"
fi

mkdir -p "$MHCFLURRY_OUT"
python "$SCRIPT_DIR/generate_affinity_factorial.py" \
    "$MHCFLURRY_OUT" \
    --mode "$FACTORIAL_MODE" \
    > "$MHCFLURRY_OUT/manifest.stdout.json"

BASELINE_CONDITION="$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["baseline_condition"])' \
    "$MHCFLURRY_OUT/manifest.json")"

condition_selected() {
    local condition="$1"
    if [ -z "$FACTORIAL_CONDITIONS" ] || \
            [ "$condition" = "$BASELINE_CONDITION" ]; then
        return 0
    fi
    case " $FACTORIAL_CONDITIONS " in
        *" $condition "*) return 0 ;;
        *) return 1 ;;
    esac
}

{
    printf '%s\n' \
        "schema_version=1" \
        "source_commit=${SOURCE_COMMIT:-unknown}" \
        "factorial_mode=$FACTORIAL_MODE" \
        "factorial_conditions=${FACTORIAL_CONDITIONS:-all}" \
        "random_seed=$RELEASE_RANDOM_SEED" \
        "gpus=$GPUS" \
        "max_workers_per_gpu=$MAX_WORKERS_PER_GPU" \
        "torch_compile=$MHCFLURRY_TORCH_COMPILE" \
        "matmul_precision=$MHCFLURRY_MATMUL_PRECISION" \
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
        --torch-compile 0 \
        --matmul-precision highest \
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
                --torch-compile 0 \
                --matmul-precision highest \
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
        --include affinity \
        --out "$baseline_eval" \
        --num-jobs auto \
        --gpus "$GPUS" \
        --max-workers-per-gpu "$MAX_WORKERS_PER_GPU" \
        --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
        --torch-compile 0 \
        --matmul-precision highest \
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
            --include affinity \
            --out "$comparison" \
            --num-jobs auto \
            --gpus "$GPUS" \
            --max-workers-per-gpu "$MAX_WORKERS_PER_GPU" \
            --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
            --torch-compile 0 \
            --matmul-precision highest \
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
                --torch-compile 0 \
                --matmul-precision highest \
                2>&1 | tee "$public_comparison/eval.log"
            date -u +%Y-%m-%dT%H:%M:%SZ > "$public_comparison/.done"
        fi
    fi
    python "$SCRIPT_DIR/summarize_affinity_factorial.py" \
        "$MHCFLURRY_OUT" > "$MHCFLURRY_OUT/summary.stdout.json"
done

python "$SCRIPT_DIR/summarize_affinity_factorial.py" \
    "$MHCFLURRY_OUT" > "$MHCFLURRY_OUT/summary.stdout.json"
