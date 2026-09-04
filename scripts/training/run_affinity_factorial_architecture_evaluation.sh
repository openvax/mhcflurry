#!/usr/bin/env bash
# Evaluate every affinity-factorial condition separately by architecture.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_affinity_factorial_architecture_evaluation.sh [OPTIONS]

Required:
  --factorial-dir PATH               Completed affinity-factorial output
  --data-eval-dir PATH               Evaluation data directory
  --release-holdout-dir PATH         Frozen holdout manifests
  --training-source-commit COMMIT    Commit used to train the models
  --analysis-source-commit COMMIT    Commit used for this postprocessing

Execution controls:
  --gpus INTEGER|auto                GPU count (default: auto)
  --max-workers-per-gpu INTEGER|auto Worker density (default: auto)
  --max-tasks-per-worker INTEGER     Worker recycling interval (default: 12)
  --torch-compile VALUE              auto, 0, or 1 (default: 0)
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

FACTORIAL_DIR=""
DATA_EVAL_DIR=""
RELEASE_HOLDOUT_DIR=""
TRAINING_SOURCE_COMMIT=""
ANALYSIS_SOURCE_COMMIT=""
GPUS="auto"
MAX_WORKERS_PER_GPU="auto"
MAX_TASKS_PER_WORKER=12
TORCH_COMPILE=0
MATMUL_PRECISION="highest"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --factorial-dir)
            require_value "$@"; FACTORIAL_DIR="$2"; shift 2 ;;
        --data-eval-dir)
            require_value "$@"; DATA_EVAL_DIR="$2"; shift 2 ;;
        --release-holdout-dir)
            require_value "$@"; RELEASE_HOLDOUT_DIR="$2"; shift 2 ;;
        --training-source-commit)
            require_value "$@"; TRAINING_SOURCE_COMMIT="$2"; shift 2 ;;
        --analysis-source-commit)
            require_value "$@"; ANALYSIS_SOURCE_COMMIT="$2"; shift 2 ;;
        --gpus)
            require_value "$@"; GPUS="$2"; shift 2 ;;
        --max-workers-per-gpu)
            require_value "$@"; MAX_WORKERS_PER_GPU="$2"; shift 2 ;;
        --max-tasks-per-worker)
            require_value "$@"; MAX_TASKS_PER_WORKER="$2"; shift 2 ;;
        --torch-compile)
            require_value "$@"; TORCH_COMPILE="$2"; shift 2 ;;
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

for required in \
        FACTORIAL_DIR DATA_EVAL_DIR RELEASE_HOLDOUT_DIR \
        TRAINING_SOURCE_COMMIT ANALYSIS_SOURCE_COMMIT; do
    if [ -z "${!required}" ]; then
        printf 'Missing required argument for %s\n' "$required" >&2
        usage >&2
        exit 2
    fi
done
for directory in "$FACTORIAL_DIR" "$DATA_EVAL_DIR" "$RELEASE_HOLDOUT_DIR"; do
    if [ ! -d "$directory" ]; then
        printf 'Not a directory: %s\n' "$directory" >&2
        exit 2
    fi
done
for file in \
        "$FACTORIAL_DIR/manifest.json" \
        "$FACTORIAL_DIR/manifest.csv" \
        "$RELEASE_HOLDOUT_DIR/policy.json" \
        "$RELEASE_HOLDOUT_DIR/affinity_samples.csv" \
        "$RELEASE_HOLDOUT_DIR/affinity_pmhcs.csv"; do
    if [ ! -f "$file" ]; then
        printf 'Missing required file: %s\n' "$file" >&2
        exit 2
    fi
done
if ! [[ "$MAX_TASKS_PER_WORKER" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer\n' '--max-tasks-per-worker' >&2
    exit 2
fi
if [ "$MAX_WORKERS_PER_GPU" != "auto" ] && \
        ! [[ "$MAX_WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be auto or a positive integer\n' \
        '--max-workers-per-gpu' >&2
    exit 2
fi
if [ "$GPUS" != "auto" ] && ! [[ "$GPUS" =~ ^[0-9]+$ ]]; then
    printf '%s must be auto or a nonnegative integer\n' '--gpus' >&2
    exit 2
fi
case "$TORCH_COMPILE" in
    auto|0|1) ;;
    *) printf 'Invalid --torch-compile: %s\n' "$TORCH_COMPILE" >&2; exit 2 ;;
esac
case "$MATMUL_PRECISION" in
    none|highest|high|medium) ;;
    *)
        printf 'Invalid --matmul-precision: %s\n' \
            "$MATMUL_PRECISION" >&2
        exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH_OUT="$FACTORIAL_DIR/architecture_evaluation"
mkdir -p "$ARCH_OUT/subsets" "$ARCH_OUT/comparisons"

if [ "$GPUS" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
    else
        GPUS=0
    fi
fi

BASELINE_CONDITION="$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["baseline_condition"])' \
    "$FACTORIAL_DIR/manifest.json")"

{
    printf '%s\n' \
        "schema_version=1" \
        "training_source_commit=$TRAINING_SOURCE_COMMIT" \
        "analysis_source_commit=$ANALYSIS_SOURCE_COMMIT" \
        "baseline_condition=$BASELINE_CONDITION" \
        "gpus=$GPUS" \
        "max_workers_per_gpu=$MAX_WORKERS_PER_GPU" \
        "max_tasks_per_worker=$MAX_TASKS_PER_WORKER" \
        "torch_compile=$TORCH_COMPILE" \
        "matmul_precision=$MATMUL_PRECISION"
    sha256sum \
        "$FACTORIAL_DIR/provenance.txt" \
        "$FACTORIAL_DIR/manifest.json" \
        "$RELEASE_HOLDOUT_DIR/policy.json" \
        "$RELEASE_HOLDOUT_DIR/affinity_samples.csv" \
        "$RELEASE_HOLDOUT_DIR/affinity_pmhcs.csv"
} > "$ARCH_OUT/provenance.txt"

tail -n +2 "$FACTORIAL_DIR/manifest.csv" | cut -d, -f1 | \
while IFS= read -r condition; do
    condition_dir="$FACTORIAL_DIR/$condition"
    if [ ! -f "$condition_dir/.train.done" ]; then
        printf 'Factorial training is incomplete: %s\n' "$condition" >&2
        exit 1
    fi
    python "$SCRIPT_DIR/split_affinity_architectures.py" \
        "$condition_dir/models.unselected.combined" \
        "$ARCH_OUT/subsets/$condition" \
        > "$ARCH_OUT/subsets/$condition.stdout.json"
done

baseline_subsets="$ARCH_OUT/subsets/$BASELINE_CONDITION"
for baseline in "$baseline_subsets"/architecture_*; do
    architecture="$(basename "$baseline")"
    comparison="$ARCH_OUT/baseline-vs-public/$architecture"
    if [ ! -f "$comparison/.done" ]; then
        mkdir -p "$comparison"
        mhcflurry eval compare-models \
            --a "$baseline" \
            --a-label "$BASELINE_CONDITION/$architecture" \
            --b public \
            --data-dir "$DATA_EVAL_DIR" \
            --release-holdout-dir "$RELEASE_HOLDOUT_DIR" \
            --affinity-training-overlap-policy audit \
            --skip-affinity-predictions \
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
done

tail -n +2 "$FACTORIAL_DIR/manifest.csv" | cut -d, -f1 | \
while IFS= read -r condition; do
    if [ "$condition" = "$BASELINE_CONDITION" ]; then
        continue
    fi
    for candidate in "$ARCH_OUT/subsets/$condition"/architecture_*; do
        architecture="$(basename "$candidate")"
        baseline="$baseline_subsets/$architecture"
        if [ ! -d "$baseline" ]; then
            printf 'Missing baseline subset: %s\n' "$baseline" >&2
            exit 1
        fi
        comparison="${ARCH_OUT}/comparisons/${condition}/${architecture}-vs-baseline"
        if [ ! -f "$comparison/.done" ]; then
            mkdir -p "$comparison"
            mhcflurry eval compare-models \
                --a "$candidate" \
                --a-label "$condition/$architecture" \
                --b "$baseline" \
                --b-label "$BASELINE_CONDITION/$architecture" \
                --data-dir "$DATA_EVAL_DIR" \
                --release-holdout-dir "$RELEASE_HOLDOUT_DIR" \
                --affinity-training-overlap-policy audit \
                --skip-affinity-predictions \
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
    done
    python "$SCRIPT_DIR/summarize_affinity_factorial_architectures.py" \
        "$FACTORIAL_DIR" > "$ARCH_OUT/summary.stdout.json"
done

python "$SCRIPT_DIR/summarize_affinity_factorial_architectures.py" \
    "$FACTORIAL_DIR" > "$ARCH_OUT/summary.stdout.json"
