#!/usr/bin/env bash
# Run the paired processing initializer/optimizer panel on the frozen holdout.
# RELEASE_RANDOM_SEED controls shared decoys, folds, and every fit (default 42).
set -euo pipefail

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/../.." && pwd)}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/gpu_telemetry.sh"
GPU_TELEMETRY_PID=""
trap stop_gpu_telemetry EXIT

export PYTHONUNBUFFERED=1
export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-0}"
export MHCFLURRY_TORCH_COMPILE_LOSS="${MHCFLURRY_TORCH_COMPILE_LOSS:-0}"
export MHCFLURRY_MATMUL_PRECISION="${MHCFLURRY_MATMUL_PRECISION:-highest}"

PANEL_DIR="$MHCFLURRY_OUT/hyperparameter_panels"
HOLDOUT_DIR="$MHCFLURRY_OUT/release_holdout"
SHARED_DIR="$MHCFLURRY_OUT/processing.shared"
PAIRWISE_DIR="$MHCFLURRY_OUT/paired_comparisons"
mkdir -p "$PANEL_DIR" "$HOLDOUT_DIR" "$SHARED_DIR" "$PAIRWISE_DIR"

python "$SCRIPT_DIR/generate_release_hyperparameter_ablations.py" "$PANEL_DIR" \
    > "$MHCFLURRY_OUT/panel_manifest.stdout.json"

mhcflurry-downloads fetch \
    data_evaluation data_curated data_mass_spec_annotated data_references \
    models_class1_pan models_class1_processing
mhcflurry train release-holdout build \
    --data-dir "$(mhcflurry-downloads path data_evaluation)" \
    --training-data "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" \
    --mass-spec-data "$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2" \
    --out-dir "$HOLDOUT_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
else
    GPUS=0
fi
if [ "$GPUS" -eq 0 ]; then
    NUM_JOBS="${NUM_JOBS:-1}"
    MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-1}"
else
    NUM_JOBS="${NUM_JOBS:-auto}"
    MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-auto}"
fi
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-auto}"
RELEASE_RANDOM_SEED="${RELEASE_RANDOM_SEED:-42}"

COMMON_PARALLELISM_ARGS=(
    --num-jobs "$NUM_JOBS"
    --max-tasks-per-worker 1000
    --gpus "$GPUS"
    --max-workers-per-gpu "$MAX_WORKERS_PER_GPU"
    --torch-compile 0
    --matmul-precision highest
)
TRAINING_PARALLELISM_ARGS=(
    "${COMMON_PARALLELISM_ARGS[@]}"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
)

compress_csv_bzip2() {
    local path="$1"
    if command -v lbzip2 >/dev/null 2>&1; then
        lbzip2 -f "$path"
    elif command -v pbzip2 >/dev/null 2>&1; then
        pbzip2 -f "$path"
    else
        bzip2 -f "$path"
    fi
}

# Hold affinity fixed at the public release while constructing processing
# decoys, so this panel isolates processing initialization and optimizer
# equations. Generate the expensive shared training table only once.
AFFINITY_PREDICTOR="${AFFINITY_PREDICTOR:-$(mhcflurry-downloads path models_class1_pan)/models.combined}"
if [ ! -f "$SHARED_DIR/train_data.csv.bz2" ]; then
    python "$REPO/downloads-generation/models_class1_processing/annotate_hits_with_expression.py" \
        --hits "$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2" \
        --expression "$(mhcflurry-downloads path data_curated)/rna_expression.csv.bz2" \
        --out "$SHARED_DIR/hits_with_tpm.csv"
    compress_csv_bzip2 "$SHARED_DIR/hits_with_tpm.csv"

    python "$SCRIPT_DIR/release_exact/make_train_data.processing.py" \
        --hits "$SHARED_DIR/hits_with_tpm.csv.bz2" \
        --affinity-predictor "$AFFINITY_PREDICTOR" \
        --proteome-reference-csv "$(mhcflurry-downloads path data_references)/uniprot_proteins.csv.bz2" \
        --ppv-multiplier 100 \
        --hit-multiplier-to-take 2 \
        --exclude-samples-file "$HOLDOUT_DIR/processing_samples.csv" \
        --random-seed "$RELEASE_RANDOM_SEED" \
        --out "$SHARED_DIR/train_data.csv" \
        "${TRAINING_PARALLELISM_ARGS[@]}"
    compress_csv_bzip2 "$SHARED_DIR/train_data.csv"
fi

conditions=(
    glorot_keras_adam
    kaiming_keras_adam
    glorot_pytorch_adam
    kaiming_pytorch_adam
)
read -r -a variants <<< \
    "${PROCESSING_ABLATION_VARIANTS:-with_flanks no_flank}"
if [ "${#variants[@]}" -eq 0 ]; then
    echo "PROCESSING_ABLATION_VARIANTS must name at least one variant" >&2
    exit 2
fi
for variant in "${variants[@]}"; do
    case "$variant" in
        with_flanks|no_flank|short_flanks) ;;
        *)
            echo "Unknown processing ablation variant: $variant" >&2
            exit 2
            ;;
    esac
done
PROCESSING_ABLATION_COMPARE_MODES="${PROCESSING_ABLATION_COMPARE_MODES:-}"
if [ -z "$PROCESSING_ABLATION_COMPARE_MODES" ]; then
    PROCESSING_ABLATION_COMPARE_MODES="${variants[0]}"
    for variant in "${variants[@]:1}"; do
        PROCESSING_ABLATION_COMPARE_MODES+=",$variant"
    done
fi

train_processing_panel() {
    local condition_root="$1"
    local variant="$2"
    local hyperparameters="$3"
    local processing_root="$condition_root/processing"
    local unselected="$processing_root/models.unselected.$variant"
    local selected="$processing_root/models.selected.$variant"

    mkdir -p "$processing_root"
    if [ -f "$unselected/manifest.csv" ]; then
        mhcflurry-class1-train-processing-models \
            --out-models-dir "$unselected" \
            --continue-incomplete \
            "${TRAINING_PARALLELISM_ARGS[@]}"
    else
        mhcflurry-class1-train-processing-models \
            --data "$SHARED_DIR/train_data.csv.bz2" \
            --held-out-samples 10 \
            --num-folds 4 \
            --random-seed "$RELEASE_RANDOM_SEED" \
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

for condition in "${conditions[@]}"; do
    condition_root="$MHCFLURRY_OUT/processing.$condition"
    processing_root="$condition_root/processing"
    mkdir -p "$processing_root"
    start_gpu_telemetry "$processing_root/gpu_occupancy.csv"
    for variant in "${variants[@]}"; do
        hyperparameters="$PANEL_DIR/processing.$condition.$variant.yaml"
        train_processing_panel \
            "$condition_root" "$variant" "$hyperparameters"
    done
    stop_gpu_telemetry
done

batch_sweep_variants=(short_flanks no_flank)
for minibatch_size in 128 256 512 1024; do
    condition_root="$MHCFLURRY_OUT/processing_batch_sweep.batch$minibatch_size"
    processing_root="$condition_root/processing"
    mkdir -p "$processing_root"
    start_gpu_telemetry "$processing_root/gpu_occupancy.csv"
    for variant in "${batch_sweep_variants[@]}"; do
        hyperparameters="$PANEL_DIR/processing_batch_sweep.batch$minibatch_size.$variant.yaml"
        train_processing_panel \
            "$condition_root" "$variant" "$hyperparameters"
    done
    stop_gpu_telemetry
done

DATA_EVAL_DIR="$(mhcflurry-downloads path data_evaluation)"
baseline_root="$MHCFLURRY_OUT/processing.glorot_keras_adam"
for condition in \
    kaiming_keras_adam \
    glorot_pytorch_adam \
    kaiming_pytorch_adam
do
    mhcflurry eval compare-models \
        --a "$MHCFLURRY_OUT/processing.$condition" \
        --a-label "$condition" \
        --b "$baseline_root" \
        --b-label glorot_keras_adam \
        --data-dir "$DATA_EVAL_DIR" \
        --release-holdout-dir "$HOLDOUT_DIR" \
        --include processing \
        --processing-modes "$PROCESSING_ABLATION_COMPARE_MODES" \
        --out "$PAIRWISE_DIR/$condition-vs-glorot_keras_adam" \
        "${COMMON_PARALLELISM_ARGS[@]}"
done

# Keep one external anchor: the retrained historical-parity condition versus
# the currently published processing ensemble on the identical frozen rows.
mhcflurry eval compare-models \
    --a "$baseline_root" \
    --a-label glorot_keras_adam \
    --b public \
    --b-label public \
    --data-dir "$DATA_EVAL_DIR" \
    --release-holdout-dir "$HOLDOUT_DIR" \
    --include processing \
    --processing-modes "$PROCESSING_ABLATION_COMPARE_MODES" \
    --out "$PAIRWISE_DIR/glorot_keras_adam-vs-public" \
    "${COMMON_PARALLELISM_ARGS[@]}"

batch_sweep_baseline="$MHCFLURRY_OUT/processing_batch_sweep.batch512"
for minibatch_size in 128 256 1024; do
    mhcflurry eval compare-models \
        --a "$MHCFLURRY_OUT/processing_batch_sweep.batch$minibatch_size" \
        --a-label "processing_batch_$minibatch_size" \
        --b "$batch_sweep_baseline" \
        --b-label processing_batch_512 \
        --data-dir "$DATA_EVAL_DIR" \
        --release-holdout-dir "$HOLDOUT_DIR" \
        --include processing \
        --processing-modes short_flanks,no_flank \
        --out "$PAIRWISE_DIR/processing_batch_$minibatch_size-vs-512" \
        "${COMMON_PARALLELISM_ARGS[@]}"
done

mhcflurry eval compare-models \
    --a "$batch_sweep_baseline" \
    --a-label processing_batch_512 \
    --b public \
    --b-label public \
    --data-dir "$DATA_EVAL_DIR" \
    --release-holdout-dir "$HOLDOUT_DIR" \
    --include processing \
    --processing-modes short_flanks,no_flank \
    --out "$PAIRWISE_DIR/processing_batch_512-vs-public" \
    "${COMMON_PARALLELISM_ARGS[@]}"
