#!/usr/bin/env bash
#
# Full mhcflurry release pipeline: affinity → processing → presentation,
# end-to-end at full architecture-sweep size.
#
# Composition:
#   - Stage 1 (affinity) delegates to pan_allele_release_affinity.sh,
#     which carries the heartbeat / write_snapshot / log_release_event
#     instrumentation, --continue-incomplete resume logic, and the
#     calibrate + eval-against-public-release phases.
#   - Stages 2-3 (processing + presentation) are inlined here, with
#     the full architecture sweep (no truncation).
#
# Each stage writes to its own subdirectory under MHCFLURRY_OUT so
# artifacts don't collide (affinity/, processing/, presentation/). The
# downstream eval step in stage 3 uses all three predictors together.
#
# Resumption: re-running this script reuses any models that the affinity
# stage already trained (via --continue-incomplete inside that stage).
# Stages 2-3 are not yet incremental — they re-run from scratch each
# time. The dominant wall-time is in stage 1, so this is OK in practice.
#
# Env (caller-tunable; all have sensible defaults):
#   MHCFLURRY_OUT              required — root for all artifacts
#   REPO                       path to the mhcflurry repo
#                              (default: this checkout)
#   MAX_WORKERS_PER_GPU        default per-GPU worker cap for shared stages
#   AFFINITY_MAX_WORKERS_PER_GPU
#                              affinity per-GPU worker cap (default auto)
#   DATALOADER_NUM_WORKERS     'auto' (default) lets the orchestrator pick
#   PROCESSING_NUM_JOBS        processing worker count (default auto)
#   PROCESSING_MAX_WORKERS_PER_GPU
#                              processing per-GPU worker cap (default auto)
#   PROCESSING_HELD_OUT_SAMPLES  (default 50; subset script uses 10)
#   PRESENTATION_DECOYS_PER_HIT (default 99 to match release; subset uses 2)
#   TRAINING_MINIBATCH_SIZE    shared affinity/processing default (default 1024)
#   AFFINITY_MINIBATCH_SIZE    affinity-specific override
#   PROCESSING_MINIBATCH_SIZE  processing-specific override
#   PROCESSING_VARIANTS        space-separated variants to train
#                              (default "with_flanks no_flank short_flanks")
#   PRESENTATION_PROCESSING_WITH_FLANKS_KIND
#                              processing variant used as presentation's
#                              with-flanks predictor (default with_flanks)
#   MHCFLURRY_GPU_TELEMETRY    0 disables processing/presentation GPU CSVs
#   MHCFLURRY_GPU_TELEMETRY_SECONDS
#                              telemetry sampling interval (default 30)
set -euo pipefail
set -x

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$SCRIPT_DIR/release_exact"
: "${REPO:=$(cd "$SCRIPT_DIR/../.." && pwd)}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/set_cpu_threads.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/gpu_telemetry.sh"
GPU_TELEMETRY_PID=""
trap stop_gpu_telemetry EXIT

export PYTHONUNBUFFERED=1
# Same default as the affinity stage; the orchestrator's CLI flag
# (--torch-compile auto) reads this when set.
export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-1}"

BASE_OUT="$MHCFLURRY_OUT"
mkdir -p "$BASE_OUT/affinity" "$BASE_OUT/processing" "$BASE_OUT/presentation"

# Detect GPU count once; reuse for all stages.
if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    GPUS=0
fi
# Default to auto so each training command exercises the orchestrator's
# workload-aware resolver. Affinity training in particular has a
# minibatch-sensitive validation footprint; leave worker packing to the
# in-process training command, which sees the hyperparameters and row count.
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-auto}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-auto}"
PROCESSING_HELD_OUT_SAMPLES="${PROCESSING_HELD_OUT_SAMPLES:-50}"
PRESENTATION_DECOYS_PER_HIT="${PRESENTATION_DECOYS_PER_HIT:-99}"
PRESENTATION_FEATURE_CHUNK_SIZE="${PRESENTATION_FEATURE_CHUNK_SIZE:-250000}"
TRAINING_MINIBATCH_SIZE="${TRAINING_MINIBATCH_SIZE:-1024}"
AFFINITY_MINIBATCH_SIZE="${AFFINITY_MINIBATCH_SIZE:-$TRAINING_MINIBATCH_SIZE}"
AFFINITY_MAX_WORKERS_PER_GPU="${AFFINITY_MAX_WORKERS_PER_GPU:-auto}"
PROCESSING_MINIBATCH_SIZE="${PROCESSING_MINIBATCH_SIZE:-$TRAINING_MINIBATCH_SIZE}"
PROCESSING_VARIANTS="${PROCESSING_VARIANTS:-with_flanks no_flank short_flanks}"
PRESENTATION_PROCESSING_WITH_FLANKS_KIND="${PRESENTATION_PROCESSING_WITH_FLANKS_KIND:-with_flanks}"

processing_variant_enabled() {
    case " $PROCESSING_VARIANTS " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

processing_variant_enabled no_flank || {
    echo "PROCESSING_VARIANTS must include no_flank for presentation training." >&2
    exit 2
}
processing_variant_enabled "$PRESENTATION_PROCESSING_WITH_FLANKS_KIND" || {
    echo "PROCESSING_VARIANTS must include PRESENTATION_PROCESSING_WITH_FLANKS_KIND=$PRESENTATION_PROCESSING_WITH_FLANKS_KIND." >&2
    exit 2
}

if [ "$GPUS" -eq 0 ]; then
    NUM_JOBS=1
    MAX_WORKERS_PER_GPU=1
elif [ "$MAX_WORKERS_PER_GPU" = "auto" ]; then
    # Pre-resolve via the orchestrator's helper so the rest of the
    # script (set_cpu_threads helper, COMMON_PARALLELISM_ARGS, log
    # banners) can use a numeric value. Pass num_jobs=0 to skip the
    # by_jobs clamp inside auto_max_workers_per_gpu — we want the
    # resolver to pick on VRAM + hard_cap alone, then derive num_jobs
    # from the picked MWPG.
    MAX_WORKERS_PER_GPU="$(
        GPUS="$GPUS" python - <<'PY'
import os
from mhcflurry.parallelism import auto_max_workers_per_gpu
print(auto_max_workers_per_gpu(
    num_jobs=0,
    num_gpus=int(os.environ["GPUS"]),
    backend="auto",
))
PY
    )"
    NUM_JOBS="$(( GPUS * MAX_WORKERS_PER_GPU ))"
    echo "Resolved MAX_WORKERS_PER_GPU=auto to $MAX_WORKERS_PER_GPU; NUM_JOBS=$NUM_JOBS"
else
    case "${NUM_JOBS:-auto}" in
        auto)
            NUM_JOBS="$(( GPUS * MAX_WORKERS_PER_GPU ))"
            ;;
        *[!0-9]*)
            echo "NUM_JOBS must be auto or an integer; got '$NUM_JOBS'." >&2
            exit 2
            ;;
        *)
            if [ "$NUM_JOBS" -lt 1 ]; then
                echo "NUM_JOBS must be at least 1; got '$NUM_JOBS'." >&2
                exit 2
            fi
            capacity="$(( GPUS * MAX_WORKERS_PER_GPU ))"
            if [ "$NUM_JOBS" -gt "$capacity" ]; then
                echo "Clamping NUM_JOBS=$NUM_JOBS to GPU capacity $capacity." >&2
                NUM_JOBS="$capacity"
            fi
            ;;
    esac
fi

# Resolve DataLoader prefetch workers before building parallelism args so the
# shell-side CPU thread budget matches the mhcflurry worker hyperparameters.
DATALOADER_NUM_WORKERS_REQUESTED="$DATALOADER_NUM_WORKERS"
DATALOADER_NUM_WORKERS="$(resolve_dataloader_num_workers "$NUM_JOBS")"
printf >&2 \
    "[pan_allele_release_full.sh] DATALOADER_NUM_WORKERS=%s resolved to %s\n" \
    "$DATALOADER_NUM_WORKERS_REQUESTED" "$DATALOADER_NUM_WORKERS"

# Shared parallelism args for the later stages. The affinity stage uses its own
# worker cap and job count below. --torch-compile auto reads
# MHCFLURRY_TORCH_COMPILE env (set above), so the env path and the CLI path
# produce identical orchestrator state.
COMMON_PARALLELISM_ARGS=(
    --num-jobs "$NUM_JOBS"
    --max-tasks-per-worker 1000
    --gpus "$GPUS"
    --max-workers-per-gpu "$MAX_WORKERS_PER_GPU"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --torch-compile auto
    --matmul-precision "${MATMUL_PRECISION:-none}"
)
if [ "${MHCFLURRY_ENABLE_TIMING:-0}" = "1" ]; then
    COMMON_PARALLELISM_ARGS+=(--enable-timing)
fi

# Processing has a different VRAM profile from affinity training: Conv1d
# flank models keep encoded sequence tensors resident and run post-fit AUC
# prediction over large fold splits. Leave worker packing as "auto" here so
# mhcflurry-class1-train-processing-models can resolve it from the processing
# data + hyperparameter sweep instead of inheriting affinity's resolved value.
PROCESSING_NUM_JOBS="${PROCESSING_NUM_JOBS:-auto}"
PROCESSING_MAX_WORKERS_PER_GPU="${PROCESSING_MAX_WORKERS_PER_GPU:-auto}"
PROCESSING_PARALLELISM_ARGS=(
    --num-jobs "$PROCESSING_NUM_JOBS"
    --max-tasks-per-worker 1000
    --gpus "$GPUS"
    --max-workers-per-gpu "$PROCESSING_MAX_WORKERS_PER_GPU"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --torch-compile auto
    --matmul-precision "${MATMUL_PRECISION:-none}"
)
if [ "${MHCFLURRY_ENABLE_TIMING:-0}" = "1" ]; then
    PROCESSING_PARALLELISM_ARGS+=(--enable-timing)
fi

# Presentation training is mostly feature generation over a large
# presentation CSV. Each worker loads the merged affinity predictor plus the
# processing ensembles, so default to one worker per GPU; callers can raise
# this after validating VRAM headroom for a given release artifact.
PRESENTATION_NUM_JOBS="${PRESENTATION_NUM_JOBS:-auto}"
PRESENTATION_MAX_WORKERS_PER_GPU="${PRESENTATION_MAX_WORKERS_PER_GPU:-1}"
PRESENTATION_PARALLELISM_ARGS=(
    --num-jobs "$PRESENTATION_NUM_JOBS"
    --max-tasks-per-worker 1000
    --gpus "$GPUS"
    --max-workers-per-gpu "$PRESENTATION_MAX_WORKERS_PER_GPU"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --torch-compile auto
    --matmul-precision "${MATMUL_PRECISION:-none}"
)
if [ "${MHCFLURRY_ENABLE_TIMING:-0}" = "1" ]; then
    PRESENTATION_PARALLELISM_ARGS+=(--enable-timing)
fi

# Presentation percentile calibration has the same resident predictor stack as
# presentation feature generation, but a different transient memory profile.
# Leave its worker packing and prediction batch size as auto by default so the
# calibrate command's workload-specific VRAM hint can resolve both together.
PRESENTATION_CALIBRATION_NUM_JOBS="${PRESENTATION_CALIBRATION_NUM_JOBS:-auto}"
PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU="${PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU:-auto}"
PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE="${PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE:-auto}"
PRESENTATION_CALIBRATION_PARALLELISM_ARGS=(
    --num-jobs "$PRESENTATION_CALIBRATION_NUM_JOBS"
    --max-tasks-per-worker 1000
    --gpus "$GPUS"
    --max-workers-per-gpu "$PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --torch-compile auto
    --matmul-precision "${MATMUL_PRECISION:-none}"
)
if [ "${MHCFLURRY_ENABLE_TIMING:-0}" = "1" ]; then
    PRESENTATION_CALIBRATION_PARALLELISM_ARGS+=(--enable-timing)
fi

NUM_JOBS="$NUM_JOBS" GPUS="$GPUS" MAX_WORKERS_PER_GPU="$MAX_WORKERS_PER_GPU" \
    DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" set_cpu_threads

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
# STAGE 1 — AFFINITY
# ============================================================
echo "=== STAGE 1: AFFINITY ==="
STAGE1_START=$(date +%s)
AFFINITY_NUM_JOBS="${AFFINITY_NUM_JOBS:-}"
if [ -z "$AFFINITY_NUM_JOBS" ] && [ "$GPUS" -eq 0 ]; then
    AFFINITY_NUM_JOBS=1
elif [ -z "$AFFINITY_NUM_JOBS" ] && \
        [ "$AFFINITY_MAX_WORKERS_PER_GPU" != "auto" ]; then
    AFFINITY_NUM_JOBS="$(( GPUS * AFFINITY_MAX_WORKERS_PER_GPU ))"
fi
AFFINITY_ENV=(
    "MHCFLURRY_OUT=$BASE_OUT/affinity"
    "GPUS=$GPUS"
    "MAX_WORKERS_PER_GPU=$AFFINITY_MAX_WORKERS_PER_GPU"
    "DATALOADER_NUM_WORKERS=${AFFINITY_DATALOADER_NUM_WORKERS:-$DATALOADER_NUM_WORKERS_REQUESTED}"
    "SKIP_EVAL=${SKIP_EVAL:-0}"
    "SKIP_PLOTS=${SKIP_PLOTS:-0}"
    "TRAINING_MINIBATCH_SIZE=$TRAINING_MINIBATCH_SIZE"
    "AFFINITY_MINIBATCH_SIZE=$AFFINITY_MINIBATCH_SIZE"
)
if [ -n "$AFFINITY_NUM_JOBS" ]; then
    AFFINITY_ENV+=("NUM_JOBS=$AFFINITY_NUM_JOBS")
fi
env "${AFFINITY_ENV[@]}" bash "$SCRIPT_DIR/pan_allele_release_affinity.sh"
AFFINITY_PREDICTOR="$BASE_OUT/affinity/models.combined"
echo "STAGE 1 duration: $(( $(date +%s) - STAGE1_START )) sec"
echo "affinity predictor: $AFFINITY_PREDICTOR"

# ============================================================
# STAGE 2 — PROCESSING
# Trains the configured processing variants. Presentation consumes no_flank and
# the configured with-flanks source (with_flanks by default).
# ============================================================
echo "=== STAGE 2: PROCESSING ==="
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
    ARCH_COUNT=$(python -c \
        "import yaml; print(len(yaml.safe_load(open('hyperparameters.$kind.yaml'))))")
    echo "processing.$kind: using $ARCH_COUNT architectures"

    mhcflurry-class1-train-processing-models \
        --data "$(pwd)/train_data.csv.bz2" \
        --held-out-samples "$PROCESSING_HELD_OUT_SAMPLES" \
        --num-folds 4 \
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
# STAGE 3 — PRESENTATION (Class1PresentationPredictor)
# ============================================================
echo "=== STAGE 3: PRESENTATION ==="
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
    --out "$(pwd)/train_data.csv"
compress_csv_bzip2 "$(pwd)/train_data.csv"

mhcflurry-class1-train-presentation-models \
    --data "$(pwd)/train_data.csv.bz2" \
    --affinity-predictor "$AFFINITY_PREDICTOR" \
    --processing-predictor-with-flanks "$BASE_OUT/processing/models.selected.$PRESENTATION_PROCESSING_WITH_FLANKS_KIND" \
    --processing-predictor-without-flanks "$BASE_OUT/processing/models.selected.no_flank" \
    --out-models-dir "$(pwd)/models" \
    --feature-chunk-size "$PRESENTATION_FEATURE_CHUNK_SIZE" \
    "${PRESENTATION_PARALLELISM_ARGS[@]}"

mhcflurry-calibrate-percentile-ranks \
    --models-dir "$(pwd)/models" \
    --match-amino-acid-distribution-data "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    --alleles-file "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    --predictor-kind class1_presentation \
    --num-peptides-per-length 100000 \
    --alleles-per-genotype 1 \
    --num-genotypes 50 \
    --prediction-batch-size "$PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE" \
    --verbosity 1 \
    "${PRESENTATION_CALIBRATION_PARALLELISM_ARGS[@]}"

# Bundle training-data CSVs into the final presentation predictor dir
# so it's self-contained for distribution.
cp "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    "$(pwd)/models/affinity_predictor_train_data.csv.bz2"
cp "$BASE_OUT/processing/models.selected.$PRESENTATION_PROCESSING_WITH_FLANKS_KIND/train_data.csv.bz2" \
    "$(pwd)/models/processing_predictor_with_flanks_train_data.csv.bz2"
cp "$BASE_OUT/processing/models.selected.no_flank/train_data.csv.bz2" \
    "$(pwd)/models/processing_predictor_no_flank_train_data.csv.bz2"

stop_gpu_telemetry
echo "STAGE 3 duration: $(( $(date +%s) - STAGE3_START )) sec"

echo "=== DONE ==="
echo "affinity:     $AFFINITY_PREDICTOR"
echo "processing:   $BASE_OUT/processing/models.selected.{${PROCESSING_VARIANTS// /,}}"
echo "presentation: $BASE_OUT/presentation/models"
ls -la "$BASE_OUT/presentation/models" | head -20
