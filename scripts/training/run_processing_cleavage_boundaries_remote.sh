#!/usr/bin/env bash
# Build frozen prerequisites and run the cleavage-boundary panel on runplz.
set -euo pipefail

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/../.." && pwd)}"

export PYTHONUNBUFFERED=1
export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-0}"
export MHCFLURRY_TORCH_COMPILE_LOSS="${MHCFLURRY_TORCH_COMPILE_LOSS:-0}"
export MHCFLURRY_MATMUL_PRECISION="${MHCFLURRY_MATMUL_PRECISION:-highest}"

HOLDOUT_DIR="$MHCFLURRY_OUT/release_holdout"
SHARED_DIR="$MHCFLURRY_OUT/processing.shared"
RUN_DIR="$MHCFLURRY_OUT/processing-cleavage-boundaries"
mkdir -p "$HOLDOUT_DIR" "$SHARED_DIR" "$RUN_DIR"

mhcflurry-downloads fetch \
    data_evaluation data_curated data_mass_spec_annotated data_references \
    models_class1_pan models_class1_processing
mhcflurry train release-holdout build \
    --data-dir "$(mhcflurry-downloads path data_evaluation)" \
    --training-data \
        "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" \
    --mass-spec-data \
        "$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2" \
    --out-dir "$HOLDOUT_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
else
    GPUS=0
fi
NUM_JOBS="${NUM_JOBS:-auto}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-auto}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-auto}"
MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-12}"
RELEASE_RANDOM_SEED="${RELEASE_RANDOM_SEED:-42}"

TRAINING_PARALLELISM_ARGS=(
    --num-jobs "$NUM_JOBS"
    --max-tasks-per-worker "$MAX_TASKS_PER_WORKER"
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

AFFINITY_PREDICTOR="${AFFINITY_PREDICTOR:-$(mhcflurry-downloads path models_class1_pan)/models.combined}"
if [ ! -f "$SHARED_DIR/train_data.csv.bz2" ]; then
    python "$REPO/downloads-generation/models_class1_processing/annotate_hits_with_expression.py" \
        --hits \
            "$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2" \
        --expression \
            "$(mhcflurry-downloads path data_curated)/rna_expression.csv.bz2" \
        --out "$SHARED_DIR/hits_with_tpm.csv"
    compress_csv_bzip2 "$SHARED_DIR/hits_with_tpm.csv"

    python "$SCRIPT_DIR/release_exact/make_train_data.processing.py" \
        --hits "$SHARED_DIR/hits_with_tpm.csv.bz2" \
        --affinity-predictor "$AFFINITY_PREDICTOR" \
        --proteome-reference-csv \
            "$(mhcflurry-downloads path data_references)/uniprot_proteins.csv.bz2" \
        --ppv-multiplier 100 \
        --hit-multiplier-to-take 2 \
        --exclude-samples-file "$HOLDOUT_DIR/processing_samples.csv" \
        --random-seed "$RELEASE_RANDOM_SEED" \
        --out "$SHARED_DIR/train_data.csv" \
        "${TRAINING_PARALLELISM_ARGS[@]}"
    compress_csv_bzip2 "$SHARED_DIR/train_data.csv"
fi

SOURCE_COMMIT="${MHCFLURRY_RELEASE_GIT_COMMIT:-}"
if [ -z "$SOURCE_COMMIT" ]; then
    SOURCE_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
fi

bash "$SCRIPT_DIR/run_processing_cleavage_boundaries.sh" \
    --out "$RUN_DIR" \
    --train-data "$SHARED_DIR/train_data.csv.bz2" \
    --data-eval-dir "$(mhcflurry-downloads path data_evaluation)" \
    --release-holdout-dir "$HOLDOUT_DIR" \
    --source-commit "$SOURCE_COMMIT" \
    --random-seed "$RELEASE_RANDOM_SEED" \
    --gpus "$GPUS" \
    --num-jobs "$NUM_JOBS" \
    --max-workers-per-gpu "$MAX_WORKERS_PER_GPU" \
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
    --max-tasks-per-worker "$MAX_TASKS_PER_WORKER"
