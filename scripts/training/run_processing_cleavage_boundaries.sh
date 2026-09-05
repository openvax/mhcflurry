#!/usr/bin/env bash
# Train and evaluate paired cleavage-boundary processing models and controls.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_processing_cleavage_boundaries.sh [OPTIONS]

Required:
  --out PATH                         Experiment output directory
  --train-data PATH                  Frozen processing training table
  --data-eval-dir PATH               Evaluation data directory
  --release-holdout-dir PATH         Frozen release-holdout manifests
  --source-commit COMMIT             Exact source commit being evaluated

Experiment controls:
  --random-seed INTEGER              Shared folds/fits seed (default: 42)
  --evaluation MODE                  none or all (default: all)

Execution controls:
  --gpus INTEGER|auto                GPU count (default: auto)
  --num-jobs INTEGER|auto            Concurrent jobs (default: auto)
  --max-workers-per-gpu INTEGER|auto Worker density (default: auto)
  --dataloader-num-workers N|auto    Workers per dataloader (default: 1)
  --max-tasks-per-worker INTEGER     Worker recycling interval (default: 12)
  -h, --help                         Show this help

The fixed screen contains 32 networks: compact 5+2 and extended 5+5 boundary
models plus retrained legacy 5-aa and no-flank controls, for two representative
architectures and four shared folds. Every condition gets loss curves and one
saved prediction column on the same frozen held-out cohort.
EOF
}

require_value() {
    if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        printf 'Missing value for %s\n' "$1" >&2
        usage >&2
        exit 2
    fi
}

sha256_path() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

OUT=""
TRAIN_DATA=""
DATA_EVAL_DIR=""
RELEASE_HOLDOUT_DIR=""
SOURCE_COMMIT=""
RANDOM_SEED=42
EVALUATION=all
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
    none|all) ;;
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
python "$SCRIPT_DIR/generate_processing_cleavage_boundaries.py" \
    "$OUT" > "$OUT/manifest.stdout.json"

{
    printf '%s\n' \
        "schema_version=1" \
        "source_commit=$SOURCE_COMMIT" \
        "design=processing-cleavage-boundaries" \
        "random_seed=$RANDOM_SEED" \
        "evaluation=$EVALUATION" \
        "gpus=$GPUS" \
        "num_jobs=$NUM_JOBS" \
        "max_workers_per_gpu=$MAX_WORKERS_PER_GPU" \
        "dataloader_num_workers=$DATALOADER_NUM_WORKERS" \
        "max_tasks_per_worker=$MAX_TASKS_PER_WORKER"
    printf '%s  %s\n' "$(sha256_path "$TRAIN_DATA")" "$TRAIN_DATA"
    printf '%s  %s\n' \
        "$(sha256_path "$RELEASE_HOLDOUT_DIR/policy.json")" \
        "$RELEASE_HOLDOUT_DIR/policy.json"
    printf '%s  %s\n' \
        "$(sha256_path "$RELEASE_HOLDOUT_DIR/processing_samples.csv")" \
        "$RELEASE_HOLDOUT_DIR/processing_samples.csv"
    printf '%s  %s\n' "$(sha256_path "$OUT/manifest.json")" \
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
            --save-validation-predictions \
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

run_comparison() {
    local name="$1"
    local a_condition="$2"
    local b_spec="$3"
    local comparison="$OUT/comparisons/$name"
    if [ -f "$comparison/.done" ]; then
        return
    fi
    mkdir -p "$comparison"
    mhcflurry eval compare-models \
        --a "$OUT/$a_condition" \
        --a-label "$a_condition" \
        --b "$b_spec" \
        --b-label "$(basename "$b_spec")" \
        --data-dir "$DATA_EVAL_DIR" \
        --release-holdout-dir "$RELEASE_HOLDOUT_DIR" \
        --include processing \
        --processing-modes short_flanks \
        --out "$comparison" \
        "${COMMON_PARALLELISM_ARGS[@]}"
    mhcflurry eval plot-comparison \
        --input "$comparison" \
        --components processing \
        --summary-pdf "$comparison/plots/model_comparison_figures.pdf"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$comparison/.done"
}

if [ "$EVALUATION" = all ]; then
    mkdir -p "$OUT/comparisons"

    # Score the large public ensemble only once. Every remaining condition is
    # paired to its architecture-matched 5-aa control, except the large 5-aa
    # control itself, which is paired to the small 5-aa anchor.
    run_comparison \
        small_tanh__legacy_5aa-vs-public \
        small_tanh__legacy_5aa public
    for condition in \
        small_tanh__legacy_no_flank \
        small_tanh__compact_5x2 \
        small_tanh__extended_5x5
    do
        run_comparison \
            "$condition-vs-small_tanh__legacy_5aa" \
            "$condition" "$OUT/small_tanh__legacy_5aa"
    done
    run_comparison \
        large_relu__legacy_5aa-vs-small_tanh__legacy_5aa \
        large_relu__legacy_5aa "$OUT/small_tanh__legacy_5aa"
    for condition in \
        large_relu__legacy_no_flank \
        large_relu__compact_5x2 \
        large_relu__extended_5x5
    do
        run_comparison \
            "$condition-vs-large_relu__legacy_5aa" \
            "$condition" "$OUT/large_relu__legacy_5aa"
    done

    prediction_suffix="processing/predictions_short_flanks.csv.bz2"
    small_anchor="$OUT/comparisons/small_tanh__legacy_5aa-vs-public/$prediction_suffix"
    large_anchor="$OUT/comparisons/large_relu__legacy_5aa-vs-small_tanh__legacy_5aa/$prediction_suffix"
    affinity_args=(
        --score "public_5aa=$small_anchor:b_processing_score"
        --score "small_tanh__legacy_5aa=$small_anchor:a_processing_score"
        --score "large_relu__legacy_5aa=$large_anchor:a_processing_score"
    )
    for condition in \
        small_tanh__legacy_no_flank \
        small_tanh__compact_5x2 \
        small_tanh__extended_5x5
    do
        path="$OUT/comparisons/$condition-vs-small_tanh__legacy_5aa/$prediction_suffix"
        affinity_args+=(--score "$condition=$path:a_processing_score")
    done
    for condition in \
        large_relu__legacy_no_flank \
        large_relu__compact_5x2 \
        large_relu__extended_5x5
    do
        path="$OUT/comparisons/$condition-vs-large_relu__legacy_5aa/$prediction_suffix"
        affinity_args+=(--score "$condition=$path:a_processing_score")
    done
    mhcflurry eval processing-affinity-control \
        "${affinity_args[@]}" \
        --baseline public_5aa \
        --data-dir "$DATA_EVAL_DIR" \
        --decoys-per-hit 10 \
        --same-protein-caliper 0.25 \
        --out "$OUT/affinity_controlled"
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/completed_at_utc.txt"
