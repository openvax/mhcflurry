#!/bin/bash
#
#
set -e
set -x

DOWNLOAD_NAME=data_mass_spec_benchmark
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR=$(dirname "$SCRIPT_ABSOLUTE_PATH")
export PYTHONUNBUFFERED=1

mkdir -p "$SCRATCH_DIR"
rm -rf "$SCRATCH_DIR/$DOWNLOAD_NAME"
mkdir "$SCRATCH_DIR/$DOWNLOAD_NAME"

# Send stdout and stderr to a logfile included with the archive.
#exec >  >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt")
#exec 2> >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt" >&2)

# Log some environment info
date
pip freeze
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

cp $SCRIPT_DIR/write_proteome_peptides.py .

PEPTIDES=$(mhcflurry-downloads path data_mass_spec_annotated)/all_sequences.csv.bz2
REFERENCES_DIR=$(mhcflurry-downloads path data_references)

python write_proteome_peptides.py \
    "$PEPTIDES" \
    "${REFERENCES_DIR}/uniprot_proteins.csv.bz2" \
    --out proteome_peptides.csv
ls -lh proteome_peptides.csv
bzip2 proteome_peptides.csv

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
