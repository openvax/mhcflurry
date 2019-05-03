#!/bin/bash
#
# Generate predictions for random peptides. Used for pre-training some models.
#
set -e
set -x

DOWNLOAD_NAME=random_peptide_predictions
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
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
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME
cp $SCRIPT_DIR/random_predictions.py .
cp $SCRIPT_ABSOLUTE_PATH .

time python random_predictions.py \
    --num-peptides 5000000 \
    --models "$(mhcflurry-downloads path models_class1_selected_no_mass_spec)/models" \
    --out predictions.csv

bzip2 predictions.csv
bzip2 LOG.txt
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
