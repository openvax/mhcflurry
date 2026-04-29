#!/usr/bin/env bash
#
# Minibatch-size sweep on a single architecture / single fold to measure
# the wall-clock + GPU-occupancy + val-loss tradeoff. Used to validate
# whether the production minibatch=128 (affinity recipe) or =4096
# (release_exact override) leaves performance on the table at higher
# values like 16384 or 32768.
#
# Cheap test: ~5 min per minibatch × 4 settings = ~20 min total on a
# single A100. Each run trains one network for a fixed budget of epochs
# (no early-stopping, deterministic seed) so the only varying input is
# minibatch_size.
#
# Outputs $MHCFLURRY_OUT/<minibatch>/{train.log,nvidia-smi.log,
# fit_info.json,val_loss.txt} per minibatch.
#
# Env:
#   MHCFLURRY_OUT         required — root for per-minibatch artifacts
#   MINIBATCH_SIZES       space-separated list (default: "4096 16384")
#   MAX_EPOCHS            cap epochs per run (default: 10)
#   PATIENCE              early-stopping patience (default: 999 = effectively off)
#   RANDOM_SEED           seed for reproducible comparison (default: 0)
#
# Reads $MHCFLURRY_OUT/summary.csv when done — same script appends a row
# per (minibatch, wall_seconds, mean_gpu_util, final_val_loss).
set -euo pipefail
set -x

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
MINIBATCH_SIZES="${MINIBATCH_SIZES:-4096 16384}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
PATIENCE="${PATIENCE:-999}"
RANDOM_SEED="${RANDOM_SEED:-0}"

# Reuse the production-recipe peptide-side encoding so the test
# parallels the real workload's compute profile. Single-arch /
# single-fold keeps the experiment focused on minibatch effects.
export PYTHONUNBUFFERED=1
export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-1}"
# torch.compile codegen cost is the same regardless of minibatch; ignore
# in the timing comparison by warming up first (see WARMUP below).
export MHCFLURRY_ENABLE_TIMING=1

if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
else
    GPUS=0
fi
echo "Detected GPUs: $GPUS"

mkdir -p "$MHCFLURRY_OUT"
cd "$MHCFLURRY_OUT"

mhcflurry-downloads fetch data_curated allele_sequences

TRAINING_DATA="$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2"
ALLELE_SEQUENCES="$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv"

# Initialize summary CSV (overwrite each run; appends are per-minibatch).
echo "minibatch,wall_seconds,mean_gpu_util,peak_gpu_util,final_val_loss,epochs_run" > summary.csv

for MB in $MINIBATCH_SIZES; do
    echo "=== minibatch_size=$MB ==="
    SUBDIR="$MHCFLURRY_OUT/mb_$MB"
    mkdir -p "$SUBDIR"
    cd "$SUBDIR"

    # Generate hyperparameters with the production architecture but a
    # patched minibatch_size. patience=$PATIENCE (default 999) effectively
    # disables early-stopping so all runs see the same number of epochs.
    python - <<PY > hyperparameters.yaml
from yaml import dump
hp = [{
    'activation': 'tanh',
    'allele_dense_layer_sizes': [],
    'batch_normalization': False,
    'dense_layer_l1_regularization': 0.0,
    'dense_layer_l2_regularization': 0.0,
    'dropout_probability': 0.5,
    'early_stopping': True,
    'init': 'glorot_uniform',
    'layer_sizes': [1024, 512],
    'learning_rate': 0.001,
    'locally_connected_layers': [],
    'topology': 'feedforward',
    'loss': 'custom:mse_with_inequalities',
    'max_epochs': $MAX_EPOCHS,
    'minibatch_size': $MB,
    'optimizer': 'rmsprop',
    'output_activation': 'sigmoid',
    'patience': $PATIENCE,
    'min_delta': 0.0,
    'peptide_encoding': {
        'vector_encoding_name': 'BLOSUM62',
        'alignment_method': 'left_pad_centered_right_pad',
        'max_length': 15,
    },
    'peptide_allele_merge_activation': '',
    'peptide_allele_merge_method': 'concatenate',
    'peptide_amino_acid_encoding': 'BLOSUM62',
    'peptide_dense_layer_sizes': [],
    'random_negative_affinity_max': 50000.0,
    'random_negative_affinity_min': 30000.0,
    'random_negative_constant': 1,
    'random_negative_distribution_smoothing': 0.0,
    'random_negative_match_distribution': True,
    'random_negative_rate': 1.0,
    'random_negative_method': 'by_allele_equalize_nonbinders',
    'random_negative_binder_threshold': 500.0,
    'train_data': {'pretrain': False},
    'validation_split': 0.1,
    'data_dependent_initialization_method': 'lsuv',
}]
print(dump(hp))
PY

    # Side-thread nvidia-smi sampler at 1 Hz — captures GPU util and memory
    # for the whole training run. Killed after training exits.
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
        --format=csv,noheader \
        -lms 1000 > nvidia-smi.log &
    NVSMI_PID=$!

    # Capture pre-warmup wall start; the train command pays a one-time
    # torch.compile cost the first iteration. We accept that as part of
    # the per-minibatch wall cost (it should be roughly equal across
    # minibatches since the compile graph is shape-polymorphic via
    # MHCFLURRY_TORCH_COMPILE_DYNAMIC=1).
    START_TS=$(date +%s)

    set +e
    mhcflurry-class1-train-pan-allele-models \
        --data "$TRAINING_DATA" \
        --allele-sequences "$ALLELE_SEQUENCES" \
        --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
        --num-folds 1 \
        --num-replicates 1 \
        --max-epochs "$MAX_EPOCHS" \
        --hyperparameters hyperparameters.yaml \
        --out-models-dir "$SUBDIR/models" \
        --num-jobs 0 \
        --gpus "$GPUS" \
        2>&1 | tee train.log
    TRAIN_RC=${PIPESTATUS[0]}
    set -e

    END_TS=$(date +%s)
    WALL=$(( END_TS - START_TS ))

    # Stop the nvidia-smi sampler and compute mean / peak GPU util for
    # the run window. Filters to GPU 0 since num_jobs=0 runs in main proc.
    kill "$NVSMI_PID" 2>/dev/null || true
    wait "$NVSMI_PID" 2>/dev/null || true

    MEAN_UTIL=$(awk -F',' '$2 ~ /^ *0$/ {gsub(/ %/, "", $3); s+=$3; n++} END {if (n>0) printf "%.1f", s/n; else print "0.0"}' nvidia-smi.log)
    PEAK_UTIL=$(awk -F',' '$2 ~ /^ *0$/ {gsub(/ %/, "", $3); if ($3+0 > p) p=$3+0} END {printf "%d", p}' nvidia-smi.log)

    # Pull final val_loss + epoch count from the saved fit_info.
    FINAL_VAL_LOSS=$(python - "$SUBDIR/models" <<'PY'
import json, sys, os, glob
models_dir = sys.argv[1]
manifest = os.path.join(models_dir, "manifest.csv")
import csv
with open(manifest) as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("nan"); sys.exit(0)
config = json.loads(rows[0]["config_json"])
fit_info = config.get("fit_info") or [{}]
val_losses = fit_info[0].get("val_loss") or []
print(val_losses[-1] if val_losses else "nan")
PY
)
    EPOCHS_RUN=$(python - "$SUBDIR/models" <<'PY'
import json, sys, os, csv
models_dir = sys.argv[1]
with open(os.path.join(models_dir, "manifest.csv")) as f:
    rows = list(csv.DictReader(f))
config = json.loads(rows[0]["config_json"])
fit_info = config.get("fit_info") or [{}]
print(len(fit_info[0].get("loss") or []))
PY
)

    cd "$MHCFLURRY_OUT"
    echo "$MB,$WALL,$MEAN_UTIL,$PEAK_UTIL,$FINAL_VAL_LOSS,$EPOCHS_RUN" >> summary.csv

    if [ "$TRAIN_RC" -ne 0 ]; then
        echo "minibatch=$MB FAILED with rc=$TRAIN_RC; see $SUBDIR/train.log"
        # Don't bail — let the loop record what it can for the others.
    fi
done

echo "=== summary ==="
cat "$MHCFLURRY_OUT/summary.csv"
