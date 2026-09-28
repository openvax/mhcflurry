#!/usr/bin/env bash
# Compare 3- and 4-residue peptide-side cleavage windows on one frozen base.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_processing_cleavage_boundary_radius.sh [OPTIONS]

Required:
  --out PATH                  Follow-up output directory
  --base-run PATH             Completed processing-cleavage-boundaries run
  --architecture NAME         small_tanh or large_relu
  --train-data PATH           Frozen processing training table
  --data-eval-dir PATH        Evaluation data directory
  --release-holdout-dir PATH  Frozen release-holdout manifests
  --source-commit COMMIT      Exact follow-up source commit

Execution controls:
  --random-seed INTEGER       Shared folds/fits seed (default: 42)
  --gpus INTEGER              GPU count (default: auto)
  --num-jobs INTEGER          Concurrent jobs per condition (default: auto)
  --dataloader-num-workers N  Workers per dataloader (default: auto)
  --parallel-conditions N     Conditions trained concurrently (default: 1)
  --affinity-control MODE     all or none (default: all)
  -h, --help                  Show this help

The scientific contrast is fixed to 5 external residues and either 3 or 4
peptide-side residues at both cleavage sites. The two conditions use the same
data, folds, seed, and recipe as the supplied base run. Set
--parallel-conditions 2 only after the seeded packing probe passes.
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
BASE_RUN=""
ARCHITECTURE=""
TRAIN_DATA=""
DATA_EVAL_DIR=""
RELEASE_HOLDOUT_DIR=""
SOURCE_COMMIT=""
RANDOM_SEED=42
GPUS=auto
NUM_JOBS=auto
DATALOADER_NUM_WORKERS=auto
PARALLEL_CONDITIONS=1
AFFINITY_CONTROL=all
ORIGINAL_ARGS=("$@")

while [ "$#" -gt 0 ]; do
    case "$1" in
        --out) require_value "$@"; OUT="$2"; shift 2 ;;
        --base-run) require_value "$@"; BASE_RUN="$2"; shift 2 ;;
        --architecture) require_value "$@"; ARCHITECTURE="$2"; shift 2 ;;
        --train-data) require_value "$@"; TRAIN_DATA="$2"; shift 2 ;;
        --data-eval-dir) require_value "$@"; DATA_EVAL_DIR="$2"; shift 2 ;;
        --release-holdout-dir)
            require_value "$@"; RELEASE_HOLDOUT_DIR="$2"; shift 2 ;;
        --source-commit) require_value "$@"; SOURCE_COMMIT="$2"; shift 2 ;;
        --random-seed) require_value "$@"; RANDOM_SEED="$2"; shift 2 ;;
        --gpus) require_value "$@"; GPUS="$2"; shift 2 ;;
        --num-jobs) require_value "$@"; NUM_JOBS="$2"; shift 2 ;;
        --dataloader-num-workers)
            require_value "$@"; DATALOADER_NUM_WORKERS="$2"; shift 2 ;;
        --parallel-conditions)
            require_value "$@"; PARALLEL_CONDITIONS="$2"; shift 2 ;;
        --affinity-control)
            require_value "$@"; AFFINITY_CONTROL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for required in \
    OUT BASE_RUN ARCHITECTURE TRAIN_DATA DATA_EVAL_DIR \
    RELEASE_HOLDOUT_DIR SOURCE_COMMIT
do
    if [ -z "${!required}" ]; then
        printf 'Missing required argument for %s\n' "$required" >&2
        exit 2
    fi
done
case "$ARCHITECTURE" in
    small_tanh|large_relu) ;;
    *) printf 'Invalid architecture: %s\n' "$ARCHITECTURE" >&2; exit 2 ;;
esac
case "$PARALLEL_CONDITIONS" in
    1|2) ;;
    *) printf 'Parallel conditions must be 1 or 2\n' >&2; exit 2 ;;
esac
case "$AFFINITY_CONTROL" in
    all|none) ;;
    *) printf 'Affinity control must be all or none\n' >&2; exit 2 ;;
esac
for path in \
    "$TRAIN_DATA" \
    "$BASE_RUN/manifest.json" \
    "$RELEASE_HOLDOUT_DIR/policy.json" \
    "$RELEASE_HOLDOUT_DIR/processing_samples.csv"
do
    if [ ! -f "$path" ]; then
        printf 'Missing required file: %s\n' "$path" >&2
        exit 2
    fi
done
for condition in legacy_5aa compact_5x2 extended_5x5; do
    path="$BASE_RUN/${ARCHITECTURE}__${condition}/predictor_path.txt"
    if [ ! -f "$path" ]; then
        printf 'Base run lacks %s\n' "$path" >&2
        exit 2
    fi
done

mhcflurry train validate-processing-data --data "$TRAIN_DATA"
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
    "$OUT" \
    --architectures "$ARCHITECTURE" \
    --peptide-context-lengths 3 4 \
    --no-controls \
    --design processing-cleavage-boundary-radius \
    > "$OUT/manifest.stdout.json"

{
    printf '%s\n' \
        "schema_version=1" \
        "source_commit=$SOURCE_COMMIT" \
        "design=processing-cleavage-boundary-radius" \
        "base_run=$BASE_RUN" \
        "architecture=$ARCHITECTURE" \
        "random_seed=$RANDOM_SEED" \
        "gpus=$GPUS" \
        "num_jobs=$NUM_JOBS" \
        "dataloader_num_workers=$DATALOADER_NUM_WORKERS" \
        "parallel_conditions=$PARALLEL_CONDITIONS" \
        "affinity_control=$AFFINITY_CONTROL"
    printf '%s  %s\n' "$(sha256_path "$TRAIN_DATA")" "$TRAIN_DATA"
    printf '%s  %s\n' "$(sha256_path "$BASE_RUN/manifest.json")" \
        "$BASE_RUN/manifest.json"
    printf '%s  %s\n' "$(sha256_path "$OUT/manifest.json")" \
        "$OUT/manifest.json"
} > "$OUT/provenance.txt"

COMMON_ARGS=(
    --num-jobs "$NUM_JOBS"
    --gpus "$GPUS"
    --max-workers-per-gpu 1
    --max-tasks-per-worker 12
    --torch-compile 0
    --matmul-precision highest
)
TRAINING_ARGS=(
    "${COMMON_ARGS[@]}"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/gpu_telemetry.sh"
GPU_TELEMETRY_PID=""
trap stop_gpu_telemetry EXIT
start_gpu_telemetry "$OUT/gpu_occupancy.csv"

train_condition() {
    local condition="$1"
    local condition_out="$OUT/$condition"
    local processing_out="$condition_out/processing"
    local unselected="$processing_out/models.unselected.short_flanks"
    local hyperparameters="$OUT/conditions/$condition.yaml"
    mkdir -p "$processing_out"
    mhcflurry-class1-train-processing-models \
        --data "$TRAIN_DATA" \
        --held-out-samples 10 \
        --num-folds 4 \
        --random-seed "$RANDOM_SEED" \
        --hyperparameters "$hyperparameters" \
        --out-models-dir "$unselected" \
        --worker-log-dir "$processing_out" \
        "${TRAINING_ARGS[@]}"
}

CONDITIONS=(
    "${ARCHITECTURE}__intermediate_5x3"
    "${ARCHITECTURE}__intermediate_5x4"
)
if [ "$PARALLEL_CONDITIONS" -eq 2 ]; then
    train_condition "${CONDITIONS[0]}" > "$OUT/${CONDITIONS[0]}.runner.log" 2>&1 &
    first_pid=$!
    train_condition "${CONDITIONS[1]}" > "$OUT/${CONDITIONS[1]}.runner.log" 2>&1 &
    second_pid=$!
    first_status=0
    second_status=0
    wait "$first_pid" || first_status=$?
    wait "$second_pid" || second_status=$?
    if [ "$first_status" -ne 0 ] || [ "$second_status" -ne 0 ]; then
        printf 'Concurrent training failed: %s=%d %s=%d\n' \
            "${CONDITIONS[0]}" "$first_status" \
            "${CONDITIONS[1]}" "$second_status" >&2
        exit 1
    fi
else
    for condition in "${CONDITIONS[@]}"; do
        train_condition "$condition" 2>&1 | tee "$OUT/$condition.runner.log"
    done
fi

for condition in "${CONDITIONS[@]}"; do
    condition_out="$OUT/$condition"
    processing_out="$condition_out/processing"
    unselected="$processing_out/models.unselected.short_flanks"
    selected="$processing_out/models.selected.short_flanks"
    mhcflurry-class1-select-processing-models \
        --data "$unselected/train_data.csv.bz2" \
        --models-dir "$unselected" \
        --out-models-dir "$selected" \
        --min-models-per-fold 1 \
        --max-models-per-fold 1 \
        --save-validation-predictions \
        "${TRAINING_ARGS[@]}"
    cp "$unselected/train_data.csv.bz2" "$selected/train_data.csv.bz2"
    printf '%s\n' "$condition_out" > "$condition_out/predictor_path.txt"
    mhcflurry train plot-loss-curves \
        --selected-dir "$selected" \
        --unselected-dir "$unselected" \
        --out "$condition_out/loss_plots"
done
stop_gpu_telemetry
GPU_TELEMETRY_PID=""

mkdir -p "$OUT/comparisons"
# One paired invocation loads the frozen 2.05-million-row cohort once and
# scores each new condition once. Comparisons to all prior anchors are derived
# from these saved row-identical columns by the combined affinity control.
radius_comparison="$OUT/comparisons/${CONDITIONS[0]}-vs-${CONDITIONS[1]}"
mhcflurry eval compare-models \
    --a "$OUT/${CONDITIONS[0]}" \
    --a-label "${CONDITIONS[0]}" \
    --b "$OUT/${CONDITIONS[1]}" \
    --b-label "${CONDITIONS[1]}" \
    --data-dir "$DATA_EVAL_DIR" \
    --release-holdout-dir "$RELEASE_HOLDOUT_DIR" \
    --include processing \
    --processing-modes short_flanks \
    --out "$radius_comparison" \
    "${COMMON_ARGS[@]}"
mhcflurry eval plot-comparison \
    --input "$radius_comparison" \
    --components processing \
    --summary-pdf "$radius_comparison/plots/model_comparison_figures.pdf"

prediction_suffix="processing/predictions_short_flanks.csv.bz2"
if [ "$AFFINITY_CONTROL" = all ]; then
    small_anchor="$BASE_RUN/comparisons/small_tanh__legacy_5aa-vs-public/$prediction_suffix"
    large_anchor="$BASE_RUN/comparisons/large_relu__legacy_5aa-vs-small_tanh__legacy_5aa/$prediction_suffix"
    affinity_args=(
        --score "public_2_1_selected=$small_anchor:b_processing_score"
        --score "small_tanh__legacy_5aa=$small_anchor:a_processing_score"
        --score "large_relu__legacy_5aa=$large_anchor:a_processing_score"
    )
    for architecture in small_tanh large_relu; do
        anchor="${architecture}__legacy_5aa"
        for window in legacy_no_flank compact_5x2 extended_5x5; do
            condition="${architecture}__${window}"
            path="$BASE_RUN/comparisons/$condition-vs-$anchor/$prediction_suffix"
            affinity_args+=(--score "$condition=$path:a_processing_score")
        done
    done
    path="$radius_comparison/$prediction_suffix"
    affinity_args+=(--score "${CONDITIONS[0]}=$path:a_processing_score")
    affinity_args+=(--score "${CONDITIONS[1]}=$path:b_processing_score")
    mhcflurry eval processing-affinity-control \
        "${affinity_args[@]}" \
        --baseline public_2_1_selected \
        --data-dir "$DATA_EVAL_DIR" \
        --decoys-per-hit 10 \
        --same-protein-caliper 0.25 \
        --out "$OUT/affinity_controlled"
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/completed_at_utc.txt"
