#!/bin/bash
#
# Download some published MHC I ligands identified by mass-spec
#
#
set -e
set -x

DOWNLOAD_NAME=data_systemhcatlas
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

mkdir -p "$SCRATCH_DIR"
rm -rf "$SCRATCH_DIR/$DOWNLOAD_NAME"
mkdir "$SCRATCH_DIR/$DOWNLOAD_NAME"

# Send stdout and stderr to a logfile included with the archive.
exec >  >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt")
exec 2> >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt" >&2)

# Log some environment info
date
pip freeze
# git rev-parse HEAD
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

wget --quiet https://github.com/openvax/mhcflurry/releases/download/pre-1.1/systemhc.20171121.combined.csv.bz2

mv systemhc.20171121.combined.csv.bz2 data.csv.bz2

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
