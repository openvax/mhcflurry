#!/usr/bin/env bash
#
# Train a full 4 fold × 4 replicate = 16-network pan-allele MHCflurry
# ensemble on the curated_training_data. Matches the structure of the
# public release so we can compare predictions head-to-head.
#
# Env:
#   MHCFLURRY_OUT              required — where artifacts are written
#   ENSEMBLE_MAX_EPOCHS=N      cap epochs per network (default: 500;
#                              early stopping usually ends long before)
#   ENSEMBLE_PATIENCE=N        early-stopping patience (default: 20)
#   ENSEMBLE_NUM_FOLDS=N       override fold count (default: 4)
#   ENSEMBLE_NUM_REPLICATES=N  override replicate count (default: 4)
#   ENSEMBLE_NUM_JOBS=N        mhcflurry --num-jobs value (default: 0 =
#                              inline; >=1 forks a Pool, which has hit
#                              CUDA-fork interactions — only use when
#                              you know GPU-per-worker is sane).
set -euo pipefail
set -x

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
MAX_EPOCHS="${ENSEMBLE_MAX_EPOCHS:-500}"
PATIENCE="${ENSEMBLE_PATIENCE:-20}"
NUM_FOLDS="${ENSEMBLE_NUM_FOLDS:-4}"
NUM_REPLICATES="${ENSEMBLE_NUM_REPLICATES:-4}"
NUM_JOBS="${ENSEMBLE_NUM_JOBS:-0}"

mkdir -p "$MHCFLURRY_OUT"
cd "$MHCFLURRY_OUT"

mhcflurry-downloads fetch data_curated allele_sequences

TRAINING_DATA="$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2"
ALLELE_SEQUENCES="$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv"

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
    'max_epochs': ${MAX_EPOCHS},
    'minibatch_size': 128,
    'optimizer': 'rmsprop',
    'output_activation': 'sigmoid',
    'patience': ${PATIENCE},
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

if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
else
    GPUS=0
fi
echo "Detected GPUs: $GPUS"

mhcflurry-class1-train-pan-allele-models \
    --data "$TRAINING_DATA" \
    --allele-sequences "$ALLELE_SEQUENCES" \
    --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
    --num-folds "$NUM_FOLDS" \
    --num-replicates "$NUM_REPLICATES" \
    --max-epochs "$MAX_EPOCHS" \
    --hyperparameters hyperparameters.yaml \
    --out-models-dir "$MHCFLURRY_OUT/models.ensemble" \
    --num-jobs "$NUM_JOBS" \
    --gpus "$GPUS"

echo "Ensemble training completed. Artifacts in: $MHCFLURRY_OUT/models.ensemble"
ls -la "$MHCFLURRY_OUT/models.ensemble"
