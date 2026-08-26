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

PARALLELISM_ARGS=(
    --num-jobs "$NUM_JOBS"
    --max-tasks-per-worker 1000
    --gpus "$GPUS"
    --max-workers-per-gpu "$MAX_WORKERS_PER_GPU"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --torch-compile 0
    --matmul-precision highest
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
        "${PARALLELISM_ARGS[@]}"
    compress_csv_bzip2 "$SHARED_DIR/train_data.csv"
fi

conditions=(
    glorot_keras_adam
    kaiming_keras_adam
    glorot_pytorch_adam
    kaiming_pytorch_adam
)
variants=(with_flanks no_flank)

for condition in "${conditions[@]}"; do
    condition_root="$MHCFLURRY_OUT/processing.$condition"
    processing_root="$condition_root/processing"
    mkdir -p "$processing_root"
    start_gpu_telemetry "$processing_root/gpu_occupancy.csv"
    for variant in "${variants[@]}"; do
        unselected="$processing_root/models.unselected.$variant"
        selected="$processing_root/models.selected.$variant"
        hyperparameters="$PANEL_DIR/processing.$condition.$variant.yaml"

        if [ -f "$unselected/manifest.csv" ]; then
            mhcflurry-class1-train-processing-models \
                --out-models-dir "$unselected" \
                --continue-incomplete \
                "${PARALLELISM_ARGS[@]}"
        else
            mhcflurry-class1-train-processing-models \
                --data "$SHARED_DIR/train_data.csv.bz2" \
                --held-out-samples 10 \
                --num-folds 4 \
                --random-seed "$RELEASE_RANDOM_SEED" \
                --hyperparameters "$hyperparameters" \
                --out-models-dir "$unselected" \
                --worker-log-dir "$processing_root" \
                "${PARALLELISM_ARGS[@]}"
        fi

        if [ ! -f "$selected/train_data.csv.bz2" ]; then
            mhcflurry-class1-select-processing-models \
                --data "$unselected/train_data.csv.bz2" \
                --models-dir "$unselected" \
                --out-models-dir "$selected" \
                --min-models-per-fold 1 \
                --max-models-per-fold 2 \
                "${PARALLELISM_ARGS[@]}"
            cp "$unselected/train_data.csv.bz2" \
                "$selected/train_data.csv.bz2"
        fi
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
        --processing-modes with_flanks,no_flank \
        --out "$PAIRWISE_DIR/$condition-vs-glorot_keras_adam" \
        "${PARALLELISM_ARGS[@]}"
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
    --processing-modes with_flanks,no_flank \
    --out "$PAIRWISE_DIR/glorot_keras_adam-vs-public" \
    "${PARALLELISM_ARGS[@]}"
