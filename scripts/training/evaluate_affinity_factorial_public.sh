#!/usr/bin/env bash
# Compare every trained affinity-factorial condition directly to public weights.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: evaluate_affinity_factorial_public.sh [OPTIONS]

Required:
  --factorial-dir PATH               Completed or partially completed factorial
  --public-affinity-dir PATH         Official models.no_additional_ms predictor
  --data-eval-dir PATH               Evaluation data directory
  --release-holdout-dir PATH         Frozen release-holdout manifests
  --analysis-source-commit COMMIT    Exact comparison-code commit

Optional:
  --external-predictions FILE        Benchmark-aligned NetMHCpan/MixMHCpred
                                      table for combined candidate figures
  --public-label LABEL               Report label (default: public-no-additional-ms)
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
PUBLIC_AFFINITY_DIR=""
DATA_EVAL_DIR=""
RELEASE_HOLDOUT_DIR=""
ANALYSIS_SOURCE_COMMIT=""
EXTERNAL_PREDICTIONS=""
PUBLIC_LABEL="public-no-additional-ms"
GPUS="auto"
MAX_WORKERS_PER_GPU="auto"
MAX_TASKS_PER_WORKER=12
TORCH_COMPILE=0
MATMUL_PRECISION="highest"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --factorial-dir)
            require_value "$@"; FACTORIAL_DIR="$2"; shift 2 ;;
        --public-affinity-dir)
            require_value "$@"; PUBLIC_AFFINITY_DIR="$2"; shift 2 ;;
        --data-eval-dir)
            require_value "$@"; DATA_EVAL_DIR="$2"; shift 2 ;;
        --release-holdout-dir)
            require_value "$@"; RELEASE_HOLDOUT_DIR="$2"; shift 2 ;;
        --analysis-source-commit)
            require_value "$@"; ANALYSIS_SOURCE_COMMIT="$2"; shift 2 ;;
        --external-predictions)
            require_value "$@"; EXTERNAL_PREDICTIONS="$2"; shift 2 ;;
        --public-label)
            require_value "$@"; PUBLIC_LABEL="$2"; shift 2 ;;
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
if [ -n "$EXTERNAL_PREDICTIONS" ] && [ ! -f "$EXTERNAL_PREDICTIONS" ]; then
    printf 'External predictions file does not exist: %s\n' \
        "$EXTERNAL_PREDICTIONS" >&2
    exit 2
fi

for required in \
        FACTORIAL_DIR PUBLIC_AFFINITY_DIR DATA_EVAL_DIR \
        RELEASE_HOLDOUT_DIR ANALYSIS_SOURCE_COMMIT; do
    if [ -z "${!required}" ]; then
        printf 'Missing required argument for %s\n' "$required" >&2
        usage >&2
        exit 2
    fi
done
for directory in \
        "$FACTORIAL_DIR" "$PUBLIC_AFFINITY_DIR" \
        "$DATA_EVAL_DIR" "$RELEASE_HOLDOUT_DIR"; do
    if [ ! -d "$directory" ]; then
        printf 'Not a directory: %s\n' "$directory" >&2
        exit 2
    fi
done
for required_file in \
        "$FACTORIAL_DIR/manifest.json" \
        "$FACTORIAL_DIR/manifest.csv" \
        "$PUBLIC_AFFINITY_DIR/manifest.csv" \
        "$PUBLIC_AFFINITY_DIR/train_data.csv.bz2" \
        "$RELEASE_HOLDOUT_DIR/policy.json" \
        "$RELEASE_HOLDOUT_DIR/affinity_samples.csv" \
        "$RELEASE_HOLDOUT_DIR/affinity_pmhcs.csv"; do
    if [ ! -f "$required_file" ]; then
        printf 'Missing required file: %s\n' "$required_file" >&2
        exit 2
    fi
done
case "$TORCH_COMPILE" in
    auto|0|1) ;;
    *) printf 'Invalid --torch-compile: %s\n' "$TORCH_COMPILE" >&2; exit 2 ;;
esac
case "$MATMUL_PRECISION" in
    none|highest|high|medium) ;;
    *) printf 'Invalid --matmul-precision: %s\n' "$MATMUL_PRECISION" >&2; exit 2 ;;
esac
if [ "$GPUS" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
    else
        GPUS=0
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_CONDITION="$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["baseline_condition"])' \
    "$FACTORIAL_DIR/manifest.json")"
PROVENANCE="$FACTORIAL_DIR/public_comparison_provenance.txt"
{
    printf '%s\n' \
        "schema_version=1" \
        "analysis_source_commit=$ANALYSIS_SOURCE_COMMIT" \
        "public_affinity_dir=$PUBLIC_AFFINITY_DIR" \
        "public_label=$PUBLIC_LABEL" \
        "affinity_source=no_additional_ms" \
        "affinity_training_overlap_policy=exclude"
    sha256sum \
        "$FACTORIAL_DIR/manifest.json" \
        "$RELEASE_HOLDOUT_DIR/policy.json" \
        "$RELEASE_HOLDOUT_DIR/affinity_samples.csv" \
        "$RELEASE_HOLDOUT_DIR/affinity_pmhcs.csv"
    while IFS= read -r public_file; do
        sha256sum "$public_file"
    done < <(find "$PUBLIC_AFFINITY_DIR" -maxdepth 1 -type f | sort)
    if [ -n "$EXTERNAL_PREDICTIONS" ]; then
        sha256sum "$EXTERNAL_PREDICTIONS"
        if [ -f "$EXTERNAL_PREDICTIONS.provenance.json" ]; then
            sha256sum "$EXTERNAL_PREDICTIONS.provenance.json"
        fi
    fi
} > "$PROVENANCE"

conditions=("$BASELINE_CONDITION")
while IFS= read -r condition; do
    if [ "$condition" != "$BASELINE_CONDITION" ] && \
            [ -f "$FACTORIAL_DIR/$condition/predictor_path.txt" ]; then
        conditions+=("$condition")
    fi
done < <(tail -n +2 "$FACTORIAL_DIR/manifest.csv" | cut -d, -f1)

public_predictions=""
for condition in "${conditions[@]}"; do
    condition_out="$FACTORIAL_DIR/$condition"
    predictor_path_file="$condition_out/predictor_path.txt"
    if [ ! -f "$predictor_path_file" ]; then
        continue
    fi
    predictor="$(cat "$predictor_path_file")"
    if [ ! -d "$predictor" ]; then
        printf 'Predictor path is not a directory for %s: %s\n' \
            "$condition" "$predictor" >&2
        exit 2
    fi
    unselected="$condition_out/models.unselected.combined"
    if [ ! -f "$condition_out/.loss-plots.done" ]; then
        loss_plot_args=(
            --selected-dir "$predictor"
            --out "$condition_out/loss_plots"
        )
        if [ "$predictor" != "$unselected" ]; then
            loss_plot_args+=(--unselected-dir "$unselected")
        fi
        mhcflurry train plot-loss-curves "${loss_plot_args[@]}"
        date -u +%Y-%m-%dT%H:%M:%SZ \
            > "$condition_out/.loss-plots.done"
    fi
    if [ "$condition" = "$BASELINE_CONDITION" ]; then
        comparison="$FACTORIAL_DIR/baseline-vs-public-no-additional-ms"
    else
        comparison="$condition_out/comparison-vs-public-no-additional-ms"
    fi
    if [ ! -f "$comparison/.done" ]; then
        mkdir -p "$comparison"
        reuse_public_args=()
        if [ "$condition" != "$BASELINE_CONDITION" ]; then
            if [ ! -f "$public_predictions" ]; then
                printf 'Reusable public predictions are missing: %s\n' \
                    "$public_predictions" >&2
                exit 2
            fi
            reuse_public_args=(
                --b-affinity-predictions "$public_predictions"
                --b-affinity-prediction-column b_pred
            )
        fi
        mhcflurry eval compare-models \
            --a "$predictor" \
            --a-label "$condition" \
            --b public \
            --b-label "$PUBLIC_LABEL" \
            --b-affinity-dir "$PUBLIC_AFFINITY_DIR" \
            --data-dir "$DATA_EVAL_DIR" \
            --release-holdout-dir "$RELEASE_HOLDOUT_DIR" \
            --affinity-source no_additional_ms \
            --affinity-training-overlap-policy exclude \
            "${reuse_public_args[@]}" \
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
    if [ "$condition" = "$BASELINE_CONDITION" ]; then
        public_predictions="$comparison/affinity/predictions.csv.bz2"
        if [ ! -f "$public_predictions" ]; then
            printf 'Baseline comparison did not save public predictions: %s\n' \
                "$public_predictions" >&2
            exit 2
        fi
    fi
    if [ ! -f "$comparison/.plots.done" ]; then
        mhcflurry plot-model-comparison \
            --input "$comparison" \
            --components affinity \
            --summary-pdf "$comparison/plots/model_comparison_figures.pdf"
        date -u +%Y-%m-%dT%H:%M:%SZ > "$comparison/.plots.done"
    fi
    printf 'Verified direct public comparison: %s\n' "$condition"
done

python "$SCRIPT_DIR/summarize_affinity_factorial.py" \
    "$FACTORIAL_DIR" > "$FACTORIAL_DIR/summary.stdout.json"

candidate_figure_out="$FACTORIAL_DIR/candidate_figures-vs-public-2.2"
if [ ! -f "$candidate_figure_out/.done" ]; then
    candidate_figure_args=()
    for condition in "${conditions[@]}"; do
        candidate_figure_args+=(--condition "$condition")
    done
    if [ "${#candidate_figure_args[@]}" -eq 0 ]; then
        printf 'No trained predictor paths were available for figures.\n' >&2
        exit 2
    fi
    if [ -n "$EXTERNAL_PREDICTIONS" ]; then
        candidate_figure_args+=(
            --external-predictions "$EXTERNAL_PREDICTIONS"
        )
    fi
    mhcflurry eval affinity-candidate-figures \
        --factorial-dir "$FACTORIAL_DIR" \
        --out "$candidate_figure_out" \
        --public-predictor-name mhcflurry_public_2_2 \
        "${candidate_figure_args[@]}"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$candidate_figure_out/.done"
fi
date -u +%Y-%m-%dT%H:%M:%SZ \
    > "$FACTORIAL_DIR/public_comparisons.completed_at_utc.txt"
