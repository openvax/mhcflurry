#!/bin/bash
#
# Train standard MHCflurry Class I models.
# Calls mhcflurry-class1-train-allele-specific-models on curated training data
# using the hyperparameters in "hyperparameters.yaml".
#
set -e
set -x

DOWNLOAD_NAME=models_class1
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR=$(dirname "$SCRIPT_ABSOLUTE_PATH")

mkdir -p "$SCRATCH_DIR"
rm -rf "$SCRATCH_DIR/$DOWNLOAD_NAME"
mkdir "$SCRATCH_DIR/$DOWNLOAD_NAME"

# Send stdout and stderr to a logfile included with the archive.
exec >  >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt")
exec 2> >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt" >&2)

# Log some environment info
date
pip freeze
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

mkdir models

python $SCRIPT_DIR/generate_hyperparameters.py > hyperparameters.yaml

time mhcflurry-class1-train-allele-specific-models \
    --data "$(mhcflurry-downloads path data_curated)/curated_training_data.with_mass_spec.csv.bz2" \
    --hyperparameters hyperparameters.yaml \
    --out-models-dir models \
    --percent-rank-calibration-num-peptides-per-length 1000000 \
    --min-measurements-per-allele 75 \
    --num-jobs 32 8

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
