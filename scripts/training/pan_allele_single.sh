#!/usr/bin/env bash
#
# Train one pan-allele MHCflurry network on the full curated_training_data.
# Real hyperparameters (1024x512 dense, early stopping) but only one fold x
# one replicate — this is for validating the runner pipeline against real
# training, not for producing a release.
#
# Env:
#   MHCFLURRY_OUT            required — where artifacts are written
#   SINGLE_MAX_EPOCHS=N      cap epochs (default: 500; early stopping usually
#                            ends long before this on full data)
#   SINGLE_PATIENCE=N        early-stopping patience (default: 20)
set -euo pipefail
set -x

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
MAX_EPOCHS="${SINGLE_MAX_EPOCHS:-500}"
PATIENCE="${SINGLE_PATIENCE:-20}"

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
    --num-folds 1 \
    --num-replicates 1 \
    --max-epochs "$MAX_EPOCHS" \
    --hyperparameters hyperparameters.yaml \
    --out-models-dir "$MHCFLURRY_OUT/models.single" \
    --num-jobs 0 \
    --gpus "$GPUS"

echo "Single-model training completed. Artifacts in: $MHCFLURRY_OUT/models.single"
ls -la "$MHCFLURRY_OUT/models.single"
