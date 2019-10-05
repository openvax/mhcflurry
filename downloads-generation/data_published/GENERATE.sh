#!/bin/bash
#
# Download published non-IEDB MHC I ligand data. Most data has made its way into
# IEDB but not all. Here we gather up the rest.
#
#
set -e
set -x

DOWNLOAD_NAME=data_published
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
# git rev-parse HEAD
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

# Kim et al 2014 [PMID 25017736]
wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/bdata.2009.mhci.public.1.txt
wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/bdata.20130222.mhci.public.1.txt
wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/bdata.2013.mhci.public.blind.1.txt

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
