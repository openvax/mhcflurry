#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'WARNING: This script is intended to be called with additional arguments to pass to mhcflurry-class1-allele-specific-cv-and-train'
    echo 'See README.md'
fi

set -e
set -x

DOWNLOAD_NAME=models_class1_allele_specific_ensemble
SCRATCH_DIR=/tmp/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR=$(dirname "$SCRIPT_ABSOLUTE_PATH")
export PYTHONUNBUFFERED=1

mkdir -p "$SCRATCH_DIR"
rm -rf "$SCRATCH_DIR/$DOWNLOAD_NAME"
mkdir "$SCRATCH_DIR/$DOWNLOAD_NAME"

# Send stdout and stderr to a logfile included with the archive.
exec >  >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt")
exec 2> >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt" >&2)

# Log some environment info
date
pip freeze
git rev-parse HEAD
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

mkdir models

cp $SCRIPT_DIR/models.py .
python models.py > models.json

time mhcflurry-class1-allele-specific-ensemble-train \
    --ensemble-size 16 \
    --model-architectures models.json \
    --train-data "$(mhcflurry-downloads path data_combined_iedb_kim2014)/combined_human_class1_dataset.csv" \
    --min-samples-per-allele 200 \
    --out-manifest models.csv \
    --out-models models \
    --verbose \
    "$@"

cp $SCRIPT_ABSOLUTE_PATH .
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
