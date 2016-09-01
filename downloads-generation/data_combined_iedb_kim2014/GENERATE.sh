#!/bin/bash

set -e
set -x

DOWNLOAD_NAME=data_combined_iedb_kim2014
SCRATCH_DIR=/tmp/mhcflurry-downloads-generation
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
git rev-parse HEAD
git status

cd "$SCRATCH_DIR/$DOWNLOAD_NAME"

mkdir .tmp  # By starting with a dot, we won't include it in the tar archive
cd .tmp

wget --quiet http://www.iedb.org/doc/mhc_ligand_full.zip
unzip mhc_ligand_full.zip

$SCRIPT_DIR/create-iedb-class1-dataset.py \
    --input-csv mhc_ligand_full.csv \
    --output-pickle-filename iedb_human_class1_assay_datasets.pickle

$SCRIPT_DIR/create-combined-class1-dataset.py \
    --iedb-pickle-path iedb_human_class1_assay_datasets.pickle \
    --netmhcpan-csv-path "$(mhcflurry-downloads path data_kim2014)/bdata.20130222.mhci.public.1.txt" \
    --output-csv-filename ../combined_human_class1_dataset.csv

cd ..
cp $SCRIPT_ABSOLUTE_PATH .
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
