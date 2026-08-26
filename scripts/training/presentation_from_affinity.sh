#!/usr/bin/env bash
#
# Train processing + presentation predictors against an *already trained*
# affinity ensemble (models.combined dir). Mirrors stages 2-3 of
# pan_allele_release_full.sh but skips stage 1, so it's cheap to run as
# a tail-on after a sweep that has already produced the affinity stack.
#
# Required env:
#   AFFINITY_PREDICTOR   path to existing models.combined dir
#   BASE_OUT             where to write processing/, presentation/
#
# Optional env (all have defaults compatible with release_full):
#   REPO, GPUS, MAX_WORKERS_PER_GPU, NUM_JOBS, DATALOADER_NUM_WORKERS,
#   MATMUL_PRECISION, MHCFLURRY_TORCH_COMPILE,
#   PROCESSING_NUM_JOBS, PROCESSING_MAX_WORKERS_PER_GPU,
#   PROCESSING_HELD_OUT_SAMPLES, PRESENTATION_DECOYS_PER_HIT,
#   PRESENTATION_FEATURE_CHUNK_SIZE, TRAINING_MINIBATCH_SIZE,
#   PROCESSING_MINIBATCH_SIZE, PROCESSING_VARIANTS,
#   PRESENTATION_PROCESSING_WITH_FLANKS_KIND,
#   RELEASE_RANDOM_SEED,
#   MHCFLURRY_GPU_TELEMETRY, MHCFLURRY_GPU_TELEMETRY_SECONDS
set -euo pipefail
set -x

: "${AFFINITY_PREDICTOR:?AFFINITY_PREDICTOR must be set}"
: "${BASE_OUT:?BASE_OUT must be set}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$SCRIPT_DIR/release_exact"
: "${REPO:=$(cd "$SCRIPT_DIR/../.." && pwd)}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/gpu_telemetry.sh"
GPU_TELEMETRY_PID=""
trap stop_gpu_telemetry EXIT

export PYTHONUNBUFFERED=1
export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-0}"
export MHCFLURRY_TORCH_COMPILE_LOSS="${MHCFLURRY_TORCH_COMPILE_LOSS:-0}"
export MHCFLURRY_MATMUL_PRECISION="${MHCFLURRY_MATMUL_PRECISION:-highest}"

mkdir -p "$BASE_OUT/processing" "$BASE_OUT/presentation"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    GPUS=0
fi
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-auto}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-auto}"
PROCESSING_HELD_OUT_SAMPLES="${PROCESSING_HELD_OUT_SAMPLES:-10}"
PRESENTATION_DECOYS_PER_HIT="${PRESENTATION_DECOYS_PER_HIT:-2}"
PRESENTATION_SAMPLE_FRACTION="${PRESENTATION_SAMPLE_FRACTION:-0.1}"
PRESENTATION_FEATURE_CHUNK_SIZE="${PRESENTATION_FEATURE_CHUNK_SIZE:-250000}"
TRAINING_MINIBATCH_SIZE="${TRAINING_MINIBATCH_SIZE:-1024}"
PROCESSING_MINIBATCH_SIZE="${PROCESSING_MINIBATCH_SIZE:-512}"
PROCESSING_VARIANTS="${PROCESSING_VARIANTS:-with_flanks no_flank short_flanks}"
PRESENTATION_PROCESSING_WITH_FLANKS_KIND="${PRESENTATION_PROCESSING_WITH_FLANKS_KIND:-short_flanks}"
RELEASE_RANDOM_SEED="${RELEASE_RANDOM_SEED:-42}"

processing_variant_enabled() {
    case " $PROCESSING_VARIANTS " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

seen_processing_variants=" "
for processing_variant in $PROCESSING_VARIANTS; do
    case "$processing_variant" in
        with_flanks|no_flank|short_flanks) ;;
        *)
            echo "Unknown PROCESSING_VARIANTS entry: $processing_variant" >&2
            exit 2
            ;;
    esac
    case "$seen_processing_variants" in
        *" $processing_variant "*)
            echo "Duplicate PROCESSING_VARIANTS entry: $processing_variant" >&2
            exit 2
            ;;
    esac
    seen_processing_variants="$seen_processing_variants$processing_variant "
done
processing_variant_enabled no_flank || {
    echo "PROCESSING_VARIANTS must include no_flank for presentation training." >&2
    exit 2
}
case "$PRESENTATION_PROCESSING_WITH_FLANKS_KIND" in
    with_flanks|short_flanks) ;;
    *)
        echo "PRESENTATION_PROCESSING_WITH_FLANKS_KIND must be with_flanks or short_flanks." >&2
        exit 2
        ;;
esac
processing_variant_enabled "$PRESENTATION_PROCESSING_WITH_FLANKS_KIND" || {
    echo "PROCESSING_VARIANTS must include PRESENTATION_PROCESSING_WITH_FLANKS_KIND=$PRESENTATION_PROCESSING_WITH_FLANKS_KIND." >&2
    exit 2
}

if [ "$GPUS" -eq 0 ]; then
    NUM_JOBS=1
    MAX_WORKERS_PER_GPU=1
else
    NUM_JOBS="${NUM_JOBS:-auto}"
    case "$NUM_JOBS" in
        auto) ;;
        *[!0-9]*)
            echo "NUM_JOBS must be auto or an integer; got '$NUM_JOBS'." >&2
            exit 2
            ;;
        *)
            if [ "$NUM_JOBS" -lt 1 ]; then
                echo "NUM_JOBS must be at least 1; got '$NUM_JOBS'." >&2
                exit 2
            fi
            ;;
    esac
fi

DATALOADER_NUM_WORKERS_REQUESTED="$DATALOADER_NUM_WORKERS"

COMMON_PARALLELISM_ARGS=(
    --num-jobs "$NUM_JOBS"
    --max-tasks-per-worker 1000
    --gpus "$GPUS"
    --max-workers-per-gpu "$MAX_WORKERS_PER_GPU"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --torch-compile "${TORCH_COMPILE_CLI:-0}"
    --matmul-precision "${MATMUL_PRECISION:-highest}"
)
[ "${MHCFLURRY_ENABLE_TIMING:-0}" = "1" ] && COMMON_PARALLELISM_ARGS+=(--enable-timing)

PROCESSING_NUM_JOBS="${PROCESSING_NUM_JOBS:-auto}"
PROCESSING_MAX_WORKERS_PER_GPU="${PROCESSING_MAX_WORKERS_PER_GPU:-auto}"
PROCESSING_PARALLELISM_ARGS=(
    --num-jobs "$PROCESSING_NUM_JOBS"
    --max-tasks-per-worker 1000
    --gpus "$GPUS"
    --max-workers-per-gpu "$PROCESSING_MAX_WORKERS_PER_GPU"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --torch-compile "${TORCH_COMPILE_CLI:-0}"
    --matmul-precision "${MATMUL_PRECISION:-highest}"
)
[ "${MHCFLURRY_ENABLE_TIMING:-0}" = "1" ] && PROCESSING_PARALLELISM_ARGS+=(--enable-timing)

PRESENTATION_NUM_JOBS="${PRESENTATION_NUM_JOBS:-auto}"
PRESENTATION_MAX_WORKERS_PER_GPU="${PRESENTATION_MAX_WORKERS_PER_GPU:-auto}"
PRESENTATION_PARALLELISM_ARGS=(
    --num-jobs "$PRESENTATION_NUM_JOBS"
    --max-tasks-per-worker 1000
    --gpus "$GPUS"
    --max-workers-per-gpu "$PRESENTATION_MAX_WORKERS_PER_GPU"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --torch-compile "${TORCH_COMPILE_CLI:-0}"
    --matmul-precision "${MATMUL_PRECISION:-highest}"
)
[ "${MHCFLURRY_ENABLE_TIMING:-0}" = "1" ] && PRESENTATION_PARALLELISM_ARGS+=(--enable-timing)

PRESENTATION_CALIBRATION_NUM_JOBS="${PRESENTATION_CALIBRATION_NUM_JOBS:-auto}"
PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU="${PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU:-auto}"
PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE="${PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE:-auto}"
PRESENTATION_CALIBRATION_PARALLELISM_ARGS=(
    --num-jobs "$PRESENTATION_CALIBRATION_NUM_JOBS"
    --max-tasks-per-worker 1000
    --gpus "$GPUS"
    --max-workers-per-gpu "$PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --torch-compile "${TORCH_COMPILE_CLI:-0}"
    --matmul-precision "${MATMUL_PRECISION:-highest}"
)
[ "${MHCFLURRY_ENABLE_TIMING:-0}" = "1" ] && PRESENTATION_CALIBRATION_PARALLELISM_ARGS+=(--enable-timing)

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

# ============================================================
# STAGE 2 — PROCESSING
# ============================================================
STAGE2_START=$(date +%s)
start_gpu_telemetry "$BASE_OUT/processing/gpu_occupancy.csv"
cd "$BASE_OUT/processing"

mhcflurry-downloads fetch data_mass_spec_annotated data_references

cp "$REPO/downloads-generation/models_class1_processing/annotate_hits_with_expression.py" .
cp "$RECIPE_DIR/make_train_data.processing.py" .

python annotate_hits_with_expression.py \
    --hits "$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2" \
    --expression "$(mhcflurry-downloads path data_curated)/rna_expression.csv.bz2" \
    --out "$(pwd)/hits_with_tpm.csv"
compress_csv_bzip2 "$(pwd)/hits_with_tpm.csv"

python make_train_data.processing.py \
    --hits "$(pwd)/hits_with_tpm.csv.bz2" \
    --affinity-predictor "$AFFINITY_PREDICTOR" \
    --proteome-reference-csv "$(mhcflurry-downloads path data_references)/uniprot_proteins.csv.bz2" \
    --ppv-multiplier 100 \
    --hit-multiplier-to-take 2 \
    --random-seed "$RELEASE_RANDOM_SEED" \
    --out "$(pwd)/train_data.csv" \
    "${COMMON_PARALLELISM_ARGS[@]}"
compress_csv_bzip2 "$(pwd)/train_data.csv"

mhcflurry class1-generate-training-hyperparameters processing-base \
    --minibatch-size "$PROCESSING_MINIBATCH_SIZE" \
    > hyperparameters.base.yaml

for kind in $PROCESSING_VARIANTS; do
    mhcflurry class1-generate-training-hyperparameters processing-variant \
        hyperparameters.base.yaml "$kind" \
        > "hyperparameters.$kind.yaml"

    mhcflurry-class1-train-processing-models \
        --data "$(pwd)/train_data.csv.bz2" \
        --held-out-samples "$PROCESSING_HELD_OUT_SAMPLES" \
        --num-folds 4 \
        --random-seed "$RELEASE_RANDOM_SEED" \
        --hyperparameters "hyperparameters.$kind.yaml" \
        --out-models-dir "$(pwd)/models.unselected.$kind" \
        --worker-log-dir "$BASE_OUT/processing" \
        "${PROCESSING_PARALLELISM_ARGS[@]}"

    mhcflurry-class1-select-processing-models \
        --data "$(pwd)/models.unselected.$kind/train_data.csv.bz2" \
        --models-dir "$(pwd)/models.unselected.$kind" \
        --out-models-dir "$(pwd)/models.selected.$kind" \
        --min-models-per-fold 1 \
        --max-models-per-fold 2 \
        "${PROCESSING_PARALLELISM_ARGS[@]}"
    cp "$(pwd)/models.unselected.$kind/train_data.csv.bz2" \
        "$(pwd)/models.selected.$kind/train_data.csv.bz2"
done

stop_gpu_telemetry
echo "STAGE 2 duration: $(( $(date +%s) - STAGE2_START )) sec"

# ============================================================
# STAGE 3 — PRESENTATION
# ============================================================
STAGE3_START=$(date +%s)
start_gpu_telemetry "$BASE_OUT/presentation/gpu_occupancy.csv"
cd "$BASE_OUT/presentation"

cp "$RECIPE_DIR/make_train_data.presentation.py" \
    make_train_data.presentation.py

python make_train_data.presentation.py \
    --hits "$BASE_OUT/processing/hits_with_tpm.csv.bz2" \
    --proteome-reference-csv "$(mhcflurry-downloads path data_references)/uniprot_proteins.csv.bz2" \
    --decoys-per-hit "$PRESENTATION_DECOYS_PER_HIT" \
    --exclude-pmid 31844290 31495665 31154438 \
    --only-format MULTIALLELIC \
    --sample-fraction "$PRESENTATION_SAMPLE_FRACTION" \
    --random-seed "$RELEASE_RANDOM_SEED" \
    --out "$(pwd)/train_data.csv"
compress_csv_bzip2 "$(pwd)/train_data.csv"

mhcflurry-class1-train-presentation-models \
    --data "$(pwd)/train_data.csv.bz2" \
    --affinity-predictor "$AFFINITY_PREDICTOR" \
    --processing-predictor-with-flanks "$BASE_OUT/processing/models.selected.$PRESENTATION_PROCESSING_WITH_FLANKS_KIND" \
    --processing-predictor-without-flanks "$BASE_OUT/processing/models.selected.no_flank" \
    --out-models-dir "$(pwd)/models" \
    --random-seed "$RELEASE_RANDOM_SEED" \
    --feature-chunk-size "$PRESENTATION_FEATURE_CHUNK_SIZE" \
    "${PRESENTATION_PARALLELISM_ARGS[@]}"

mhcflurry-calibrate-percentile-ranks \
    --models-dir "$(pwd)/models" \
    --match-amino-acid-distribution-data "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    --alleles-file "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    --predictor-kind class1_presentation \
    --num-peptides-per-length 10000 \
    --alleles-per-genotype 1 \
    --num-genotypes 50 \
    --prediction-batch-size "$PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE" \
    --random-seed "$RELEASE_RANDOM_SEED" \
    --verbosity 1 \
    "${PRESENTATION_CALIBRATION_PARALLELISM_ARGS[@]}"

cp "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    "$(pwd)/models/affinity_predictor_train_data.csv.bz2"
cp "$BASE_OUT/processing/models.selected.$PRESENTATION_PROCESSING_WITH_FLANKS_KIND/train_data.csv.bz2" \
    "$(pwd)/models/processing_predictor_with_flanks_train_data.csv.bz2"
cp "$BASE_OUT/processing/models.selected.no_flank/train_data.csv.bz2" \
    "$(pwd)/models/processing_predictor_no_flank_train_data.csv.bz2"

stop_gpu_telemetry
echo "STAGE 3 duration: $(( $(date +%s) - STAGE3_START )) sec"
echo "=== presentation predictor at $BASE_OUT/presentation/models ==="
