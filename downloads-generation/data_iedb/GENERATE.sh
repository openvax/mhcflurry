#!/bin/bash
#
# Download latest MHC I ligand data from IEDB.
#
set -e
set -x

DOWNLOAD_NAME=data_iedb
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

cd $SCRATCH_DIR/$DOWNLOAD_NAME

wget -q http://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip -O mhc_ligand_full.zip
wget -q http://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip -O tcell_full_v3.zip

unzip mhc_ligand_full.zip
rm mhc_ligand_full.zip
bzip2 mhc_ligand_full.csv

unzip tcell_full_v3.zip
rm tcell_full_v3.zip
bzip2 tcell_full_v3.csv

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
