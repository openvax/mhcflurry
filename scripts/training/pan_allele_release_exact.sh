#!/usr/bin/env bash
#
# Exact replication of the public mhcflurry pan-allele release training
# recipe (models_class1_pan, 2.2.0's GENERATE.sh — local mode, not
# cluster). Uses the same `generate_hyperparameters.py`,
# `reassign_mass_spec_training_data.py`, `additional_alleles.txt`, the
# same curated_training_data and pretrain data (random_peptide_predictions),
# same 4 folds, same architecture sweep, same --min-models/--max-models
# selection, and runs percentile-rank calibration on the final ensemble.
#
# Env:
#   MHCFLURRY_OUT              required — where artifacts are written.
#                              Final selected ensemble lives at
#                              $MHCFLURRY_OUT/models.combined/.
#   NUM_JOBS=N                 parallelism (default: $GPUS). Controls
#                              how many networks train concurrently;
#                              mhcflurry pairs jobs 1:1 with GPUs.
set -euo pipefail
set -x

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"

export PYTHONUNBUFFERED=1
# torch.compile on by default after the Phase-4 sweep landed it.
# Compile cost (~30-60s codegen) is paid once per worker. We deliberately
# recycle workers after a moderate number of tasks to avoid the mysterious
# long-lived-worker death mode seen on multi-day runs, while still
# amortizing compile / CUDA init across several networks.
export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-1}"
# OMP/MKL/OPENBLAS set via set_cpu_threads helper below — auto-computes
# from nproc + GPU/worker layout. Manually override any of them before
# calling this script to pin an explicit value.

SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_ABSOLUTE_PATH")"
RECIPE_DIR="$SCRIPT_DIR/release_exact"

mkdir -p "$MHCFLURRY_OUT"
cd "$MHCFLURRY_OUT"

# ---- GPU occupancy sampler (background, for later analysis) ----------
# Sample nvidia-smi every 30 s into a CSV that rsync-down can pull back.
# Captures per-GPU util, memory, and timestamp so we can answer "what
# was the steady-state occupancy?" without having to SSH live during
# the run. Auto-exits when the training process ends (watches its pid).
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LOG="$MHCFLURRY_OUT/gpu_occupancy.csv"
    {
        echo "timestamp,gpu_index,util_percent,mem_used_mib,mem_total_mib"
        while :; do
            ts=$(date +%s)
            nvidia-smi \
                --query-gpu=index,utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits 2>/dev/null \
              | while IFS=',' read -r idx util mem_used mem_total; do
                  echo "$ts,${idx// /},${util// /},${mem_used// /},${mem_total// /}"
              done
            sleep 30
        done
    } > "$GPU_LOG" 2>/dev/null &
    GPU_SAMPLER_PID=$!
    echo "GPU occupancy sampler PID: $GPU_SAMPLER_PID → $GPU_LOG"
    # Make sure the sampler dies when the script exits (clean or error).
    trap "kill $GPU_SAMPLER_PID 2>/dev/null; wait $GPU_SAMPLER_PID 2>/dev/null" EXIT
fi

# ---- parallelism -----------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    GPUS=0
fi
echo "Detected GPUS: $GPUS"

PROCESSORS=$(getconf _NPROCESSORS_ONLN)
echo "Detected processors: $PROCESSORS"

# Per-worker peak is 16-22 GB on mhcflurry pan-allele training (observed on
# A100-80GB). Safe: 2 on 80GB, 1 on 40GB. 4 OOMs on 80GB.
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-2}"
if [ "$GPUS" -eq "0" ]; then
    NUM_JOBS="${NUM_JOBS-1}"
else
    NUM_JOBS="${NUM_JOBS-$(( GPUS * MAX_WORKERS_PER_GPU ))}"
fi
echo "Num jobs: $NUM_JOBS (max-workers-per-gpu=$MAX_WORKERS_PER_GPU)"
# Recycle after a bounded number of tasks so compile is still amortized
# but leaks / descriptor creep / orphaned-runtime state cannot accumulate
# indefinitely inside one worker. Override with MAX_TASKS_PER_WORKER.
MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-12}"
PARALLELISM_ARGS="--num-jobs $NUM_JOBS --max-tasks-per-worker $MAX_TASKS_PER_WORKER --gpus $GPUS --max-workers-per-gpu $MAX_WORKERS_PER_GPU"

# Phase 1 (#268): enable the BLOSUM62 encoding cache + DataLoader prefetch
# by default. USE_ENCODING_CACHE=0 or DATALOADER_NUM_WORKERS=0 restores
# the pre-#268 legacy path (bit-identical; verified by Phase 1 tests).
USE_ENCODING_CACHE="${USE_ENCODING_CACHE:-1}"
# DataLoader prefetch workers per training worker. v11 sweep winner
# (batch=512, dl=1, wpg=2 on L40S) showed dl=1 marginally beat dl=2;
# dl=2 added overhead without parallelism benefit at single-item
# granularity. Default to 1 now.
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
CACHE_ARGS=()
if [ "$USE_ENCODING_CACHE" = "1" ]; then
    CACHE_ARGS=(--use-encoding-cache)
fi

# Auto-configure OMP / MKL / OPENBLAS thread budget uniformly based on
# nproc, GPU count, worker count, dataloader worker count. User can
# override any of {OMP,MKL,OPENBLAS}_NUM_THREADS before invoking this
# script; the helper respects manual settings.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/set_cpu_threads.sh"
GPUS="$GPUS" MAX_WORKERS_PER_GPU="$MAX_WORKERS_PER_GPU" \
    DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" set_cpu_threads

# ---- data ------------------------------------------------------------
mhcflurry-downloads fetch data_curated allele_sequences random_peptide_predictions

# Reassign mass-spec training rows per release recipe.
cp "$RECIPE_DIR/reassign_mass_spec_training_data.py" .
cp "$RECIPE_DIR/additional_alleles.txt" .

python reassign_mass_spec_training_data.py \
    "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" \
    --set-measurement-value 100 \
    --out-csv "$(pwd)/train_data.csv"
bzip2 -f "$(pwd)/train_data.csv"
TRAINING_DATA="$(pwd)/train_data.csv.bz2"

# ---- hyperparameters -------------------------------------------------
cp "$RECIPE_DIR/generate_hyperparameters.py" .
python generate_hyperparameters.py > hyperparameters.full.yaml
# Inject dataloader_num_workers into every arch entry (Phase 1 #268).
python - <<PY
import yaml
hp = yaml.safe_load(open("hyperparameters.full.yaml"))
for d in hp:
    d["dataloader_num_workers"] = $DATALOADER_NUM_WORKERS
with open("hyperparameters.yaml", "w") as f:
    yaml.safe_dump(hp, f)
print(f"release_exact: {len(hp)} archs with dataloader_num_workers=$DATALOADER_NUM_WORKERS")
PY
ARCH_COUNT=$(python -c "import yaml; print(len(yaml.safe_load(open('hyperparameters.yaml'))))")
echo "Architectures in sweep: $ARCH_COUNT"

ALLELE_SEQUENCES="$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv"
PRETRAIN_DATA="$(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2"

# ---- train all candidates (4 folds × ARCH_COUNT architectures) ------
# `combined` is the only kind the public release trains (see GENERATE.sh).
for kind in combined
do
    UNSELECTED_DIR="models.unselected.${kind}"

    CONTINUE_ARGS=""
    if [ -d "$UNSELECTED_DIR" ]; then
        echo "Found existing $UNSELECTED_DIR — continuing incomplete run"
        CONTINUE_ARGS="--continue-incomplete"
    fi

    mhcflurry-class1-train-pan-allele-models \
        --data "$TRAINING_DATA" \
        --allele-sequences "$ALLELE_SEQUENCES" \
        --pretrain-data "$PRETRAIN_DATA" \
        --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
        --num-folds 4 \
        --hyperparameters hyperparameters.yaml \
        --out-models-dir "$(pwd)/$UNSELECTED_DIR" \
        --worker-log-dir "$MHCFLURRY_OUT" \
        "${CACHE_ARGS[@]}" \
        $PARALLELISM_ARGS $CONTINUE_ARGS
done
echo "Done training. Beginning model selection."

# ---- select ---------------------------------------------------------
for kind in combined
do
    UNSELECTED_DIR="models.unselected.${kind}"
    SELECTED_DIR="models.${kind}"

    mhcflurry-class1-select-pan-allele-models \
        --data "$UNSELECTED_DIR/train_data.csv.bz2" \
        --models-dir "$UNSELECTED_DIR" \
        --out-models-dir "$SELECTED_DIR" \
        --min-models 2 \
        --max-models 8 \
        $PARALLELISM_ARGS
    cp "$UNSELECTED_DIR/train_data.csv.bz2" "$SELECTED_DIR/train_data.csv.bz2"

    # ---- percentile rank calibration (matches release) ---------------
    time mhcflurry-calibrate-percentile-ranks \
        --models-dir "$SELECTED_DIR" \
        --match-amino-acid-distribution-data "$UNSELECTED_DIR/train_data.csv.bz2" \
        --motif-summary \
        --num-peptides-per-length 100000 \
        --alleles-per-work-chunk 10 \
        --verbosity 1 \
        $PARALLELISM_ARGS
done

echo "Full release-exact training completed."
echo "Final ensemble: $MHCFLURRY_OUT/models.combined/"
ls -la "$MHCFLURRY_OUT/models.combined" | head -30
