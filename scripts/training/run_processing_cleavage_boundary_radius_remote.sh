#!/usr/bin/env bash
# Run the focused peptide-side cleavage-radius follow-up on a shared volume.
set -euo pipefail

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/../.." && pwd)}"

export PYTHONUNBUFFERED=1
export MHCFLURRY_TORCH_COMPILE=0
export MHCFLURRY_TORCH_COMPILE_LOSS=0
export MHCFLURRY_MATMUL_PRECISION=highest

BASE_RUN="${BOUNDARY_RADIUS_BASE_RUN:-$MHCFLURRY_OUT/processing-cleavage-boundaries}"
FOLLOWUP_OUT="${BOUNDARY_RADIUS_OUT:-$MHCFLURRY_OUT/processing-cleavage-boundary-radius}"
TRAIN_DATA="${BOUNDARY_RADIUS_TRAIN_DATA:-$MHCFLURRY_OUT/processing.shared/train_data.csv.bz2}"
HOLDOUT_DIR="${BOUNDARY_RADIUS_HOLDOUT_DIR:-$MHCFLURRY_OUT/release_holdout}"
ARCHITECTURE="${BOUNDARY_RADIUS_ARCHITECTURE:-large_relu}"
PARALLEL_CONDITIONS="${BOUNDARY_RADIUS_PARALLEL_CONDITIONS:-1}"
AFFINITY_CONTROL="${BOUNDARY_RADIUS_AFFINITY_CONTROL:-none}"

mhcflurry-downloads fetch data_evaluation

if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
else
    GPUS=0
fi
SOURCE_COMMIT="${MHCFLURRY_RELEASE_GIT_COMMIT:-}"
if [ -z "$SOURCE_COMMIT" ]; then
    SOURCE_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
fi

bash "$SCRIPT_DIR/run_processing_cleavage_boundary_radius.sh" \
    --out "$FOLLOWUP_OUT" \
    --base-run "$BASE_RUN" \
    --architecture "$ARCHITECTURE" \
    --train-data "$TRAIN_DATA" \
    --data-eval-dir "$(mhcflurry-downloads path data_evaluation)" \
    --release-holdout-dir "$HOLDOUT_DIR" \
    --source-commit "$SOURCE_COMMIT" \
    --random-seed "${RELEASE_RANDOM_SEED:-42}" \
    --gpus "$GPUS" \
    --num-jobs "${NUM_JOBS:-auto}" \
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-auto}" \
    --parallel-conditions "$PARALLEL_CONDITIONS" \
    --affinity-control "$AFFINITY_CONTROL"
