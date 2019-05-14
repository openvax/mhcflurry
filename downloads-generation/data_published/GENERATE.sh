#!/bin/bash
#
# Download some published MHC I ligand data
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
cp $SCRIPT_DIR/parse.py .

# Kim et al 2014 [PMID 25017736]
wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/bdata.2009.mhci.public.1.txt
wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/bdata.20130222.mhci.public.1.txt
wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/bdata.2013.mhci.public.blind.1.txt

# Abelin et al 2017 [PMID 28228285]
# This is now in IEDB, so commenting out.
# wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/abelin2017.hits.csv.bz2

#
# For the supplementary tables downloaded below, the ID indicates the PMID.
#
# These have all been incorporated into IEDB so we now leave them commented out.
#ID=28904123  # Di Marco et al 2017
#wget -q http://www.jimmunol.org/highwire/filestream/347380/field_highwire_adjunct_files/1/JI_1700938_Supplemental_Table_1.xlsx -O "${ID}.xlsx"
#python parse.py --format "$ID" --input "${ID}.xlsx" --out-csv "${ID}.csv"

#ID=30410026  # Illing et al 2018
#wget -q https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-07109-w/MediaObjects/41467_2018_7109_MOESM3_ESM.xlsx -O "${ID}.xlsx"
#python parse.py --format "$ID" --input "${ID}.xlsx" --out-csv "${ID}.csv"

#ID=28855257  # Mobbs et al 2017
#wget -q http://www.jbc.org/lookup/suppl/doi:10.1074/jbc.M117.806976/-/DC1/jbc.M117.806976-1.xlsx -O "${ID}.xlsx"
#python parse.py --format "$ID" --input "${ID}.xlsx" --out-csv "${ID}.csv"

#ID=29437277  # Ramarathinam et al 2018
#wget -q https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fpmic.201700253&file=pmic12831-sup-0002-Data.xlsx -O "${ID}.xlsx"
#python parse.py --format "$ID" --input "${ID}.xlsx" --out-csv "${ID}.csv"

#ID=28218747  # Pymm et al 2017
#wget -q https://media.nature.com/original/nature-assets/nsmb/journal/v24/n4/extref/nsmb.3381-S2.xlsx -O "${ID}.xlsx"
#python parse.py --format "$ID" --input "${ID}.xlsx" --out-csv "${ID}.csv"

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
