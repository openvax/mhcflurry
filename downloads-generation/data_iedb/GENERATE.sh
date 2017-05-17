#!/bin/bash

set -e
set -x

DOWNLOAD_NAME=data_iedb
SCRATCH_DIR=/tmp/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

mkdir -p "$SCRATCH_DIR"
rm -rf "$SCRATCH_DIR/$DOWNLOAD_NAME"
mkdir "$SCRATCH_DIR/$DOWNLOAD_NAME"

# Send stdout and stderr to a logfile included with the archive.
exec >  >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt")
exec 2> >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt" >&2)

# Log some environment info
date

cd $SCRATCH_DIR/$DOWNLOAD_NAME

wget --quiet http://www.iedb.org/doc/mhc_ligand_full.zip
unzip mhc_ligand_full.zip
rm mhc_ligand_full.zip

cp $SCRIPT_ABSOLUTE_PATH .
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
