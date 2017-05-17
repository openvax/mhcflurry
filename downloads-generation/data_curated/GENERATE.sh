#!/bin/bash

set -e
set -x

DOWNLOAD_NAME=data_curated
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
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

cp $SCRIPT_DIR/curate.py .

time python curate.py \
    --data-iedb \
        "$(mhcflurry-downloads path data_iedb)/mhc_ligand_full.csv.bz2" \
    --data-kim2014 \
        "$(mhcflurry-downloads path data_kim2014)/bdata.20130222.mhci.public.1.txt" \
    --out-csv curated_training_data.csv

bzip2 curated_training_data.csv
cp $SCRIPT_ABSOLUTE_PATH .
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
