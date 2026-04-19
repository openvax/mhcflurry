#!/usr/bin/env bash
#
# Pan-allele training smoketest — verifies the training pipeline runs end-to-end.
# Not for model quality; hyperparameters and data are tiny.
#
# Env:
#   MHCFLURRY_OUT               required — where artifacts are written
#   MHCFLURRY_DOWNLOADS_DIR     optional — host mount for mhcflurry-downloads
#   TINY=1                      filter to 2 alleles, ~200 rows, 2 epochs
#   SMOKETEST_MAX_EPOCHS=N      override epochs (default: 2)
#   SMOKETEST_NUM_FOLDS=N       override folds  (default: 1)
#   SMOKETEST_NUM_REPLICATES=N  override reps   (default: 1)
set -euo pipefail
set -x

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
TINY="${TINY:-0}"
MAX_EPOCHS="${SMOKETEST_MAX_EPOCHS:-2}"
NUM_FOLDS="${SMOKETEST_NUM_FOLDS:-1}"
NUM_REPLICATES="${SMOKETEST_NUM_REPLICATES:-1}"

mkdir -p "$MHCFLURRY_OUT"
cd "$MHCFLURRY_OUT"

# Ensure mhcflurry-downloads data is available. Idempotent: fetch skips
# anything already on disk. Needed in containers where MHCFLURRY_DOWNLOADS_DIR
# is set but the directory is empty.
mhcflurry-downloads fetch data_curated allele_sequences

TRAINING_DATA_SRC="$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2"
ALLELE_SEQUENCES_SRC="$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv"

# Build the training data slice.
if [ "$TINY" = "1" ]; then
    python - "$TRAINING_DATA_SRC" "$ALLELE_SEQUENCES_SRC" <<'PY'
import sys, pandas as pd
src_data, src_alleles = sys.argv[1], sys.argv[2]
alleles = ["HLA-A*02:01", "HLA-B*07:02"]
df = pd.read_csv(src_data)
sub = df[df["allele"].isin(alleles)].groupby("allele").head(100)
sub.to_csv("train_data.csv", index=False)
# Allele sequences uses the `normalized_allele` column name.
seqs = pd.read_csv(src_alleles)
seqs[seqs["normalized_allele"].isin(alleles)].to_csv("allele_sequences.tiny.csv", index=False)
print(f"TINY: {len(sub)} rows across {sub['allele'].nunique()} alleles")
PY
    TRAINING_DATA="$(pwd)/train_data.csv"
    ALLELE_SEQUENCES="$(pwd)/allele_sequences.tiny.csv"
else
    TRAINING_DATA="$TRAINING_DATA_SRC"
    ALLELE_SEQUENCES="$ALLELE_SEQUENCES_SRC"
fi

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
    'layer_sizes': [64, 32],
    'learning_rate': 0.001,
    'locally_connected_layers': [],
    'topology': 'feedforward',
    'loss': 'custom:mse_with_inequalities',
    'max_epochs': ${MAX_EPOCHS},
    'minibatch_size': 64,
    'optimizer': 'rmsprop',
    'output_activation': 'sigmoid',
    'patience': 2,
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
    --held-out-measurements-per-allele-fraction-and-max 0.25 20 \
    --num-folds "$NUM_FOLDS" \
    --num-replicates "$NUM_REPLICATES" \
    --max-epochs "$MAX_EPOCHS" \
    --hyperparameters hyperparameters.yaml \
    --out-models-dir "$MHCFLURRY_OUT/models.smoketest" \
    --num-jobs 1 \
    --gpus "$GPUS"

echo "Smoketest training completed. Artifacts in: $MHCFLURRY_OUT/models.smoketest"
ls -la "$MHCFLURRY_OUT/models.smoketest"
