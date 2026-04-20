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

# CRITICAL: matches GENERATE.sh in the public release. Without this,
# numpy/MKL in each worker multi-threads to all available cores. With
# 8 parallel workers on a 120-vCPU box that gave us load avg ~713
# (6x oversubscription), 20-40x slowdown, and near-idle GPUs. One thread
# per worker lets the GPU actually be the limiting factor.
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_ABSOLUTE_PATH")"
RECIPE_DIR="$SCRIPT_DIR/release_exact"

mkdir -p "$MHCFLURRY_OUT"
cd "$MHCFLURRY_OUT"

# ---- parallelism -----------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    GPUS=0
fi
echo "Detected GPUS: $GPUS"

PROCESSORS=$(getconf _NPROCESSORS_ONLN)
echo "Detected processors: $PROCESSORS"

if [ "$GPUS" -eq "0" ]; then
    NUM_JOBS="${NUM_JOBS-1}"
else
    NUM_JOBS="${NUM_JOBS-$GPUS}"
fi
echo "Num jobs: $NUM_JOBS"
PARALLELISM_ARGS="--num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1"

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
python generate_hyperparameters.py > hyperparameters.yaml
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
