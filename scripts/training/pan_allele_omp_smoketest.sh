#!/usr/bin/env bash
#
# Quickly measure per-epoch wall time with OMP_NUM_THREADS=1. Same
# pipeline as pan_allele_release_affinity.sh but caps work:
#   - 1 fold (not 4)
#   - 2 architectures (not 35 from generate_hyperparameters.py)
#   - no percentile-rank calibration
#   - no model selection step
#   - cap epochs at 20 per network to bound wall time
#
# Target signal: per-epoch time drops from the 200-450s seen without
# OMP_NUM_THREADS=1 to <20s. If we see <20s on an A100 with OMP=1, the
# full release-exact run is worth launching.
set -euo pipefail
set -x

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_ABSOLUTE_PATH")"
RECIPE_DIR="$SCRIPT_DIR/release_exact"

mkdir -p "$MHCFLURRY_OUT"
cd "$MHCFLURRY_OUT"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    GPUS=0
fi
echo "Detected GPUS: $GPUS"
PARALLELISM_ARGS="--num-jobs $GPUS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1"

# ---- data (same as public recipe) -----------------------------------
mhcflurry-downloads fetch data_curated allele_sequences random_peptide_predictions

cp "$RECIPE_DIR/reassign_mass_spec_training_data.py" .
cp "$RECIPE_DIR/additional_alleles.txt" .

python reassign_mass_spec_training_data.py \
    "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" \
    --set-measurement-value 100 \
    --out-csv "$(pwd)/train_data.csv"
bzip2 -f "$(pwd)/train_data.csv"
TRAINING_DATA="$(pwd)/train_data.csv.bz2"

# ---- sub-sampled hyperparameters (2 arches, capped epochs) ----------
cp "$RECIPE_DIR/generate_hyperparameters.py" .
# Generate full sweep, then subset to 2 architectures with capped epochs.
python generate_hyperparameters.py > hyperparameters.full.yaml
python - <<'PY'
import yaml
with open("hyperparameters.full.yaml") as f:
    all_hp = yaml.safe_load(f)
sub = all_hp[:2]
for hp in sub:
    hp["max_epochs"] = 20
    if "train_data" in hp and isinstance(hp["train_data"], dict):
        if "pretrain_max_epochs" in hp["train_data"]:
            hp["train_data"]["pretrain_max_epochs"] = 5
with open("hyperparameters.yaml", "w") as f:
    yaml.safe_dump(sub, f)
PY

ARCH_COUNT=$(python -c "import yaml; print(len(yaml.safe_load(open('hyperparameters.yaml'))))")
echo "Architectures in smoketest sweep: $ARCH_COUNT"

ALLELE_SEQUENCES="$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv"
PRETRAIN_DATA="$(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2"

mhcflurry-class1-train-pan-allele-models \
    --data "$TRAINING_DATA" \
    --allele-sequences "$ALLELE_SEQUENCES" \
    --pretrain-data "$PRETRAIN_DATA" \
    --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
    --num-folds 1 \
    --hyperparameters hyperparameters.yaml \
    --out-models-dir "$(pwd)/models.smoketest" \
    --worker-log-dir "$MHCFLURRY_OUT" \
    $PARALLELISM_ARGS

echo "Smoketest training completed."
echo "Per-epoch timing summary (look for [53 sec] or similar in brackets):"
for f in "$MHCFLURRY_OUT"/LOG-worker.*.txt; do
    [ -f "$f" ] || continue
    grep -E "Epoch  +[0-9]+ / +[0-9]+ \[" "$f" | head -5
done
echo "--- done ---"
