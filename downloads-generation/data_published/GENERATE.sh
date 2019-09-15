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

date

cd $SCRATCH_DIR/$DOWNLOAD_NAME

############################################
# BINDING AFFINITIES
############################################
#
# Kim et al 2014 [PMID 25017736]
wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/bdata.2009.mhci.public.1.txt
wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/bdata.20130222.mhci.public.1.txt
wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/bdata.2013.mhci.public.blind.1.txt

mkdir raw

############################################
# MS: Multiallelic
############################################
# Bassani-Sternberg, ..., Gfeller PLOS Comp. Bio. 2017 [PMID 28832583]
# The first dataset is from this work. The second dataset is originally from:
#   Pearson, ..., Perreault JCI 2016 [PMID 27841757]
# but was reanalyzed in this work, and we download the reanalyzed version here.
PMID=28832583
mkdir -p raw/$PMID
wget -q https://doi.org/10.1371/journal.pcbi.1005725.s002 -P raw/$PMID # data generated in this work
wget -q https://doi.org/10.1371/journal.pcbi.1005725.s003 -P raw/$PMID # data reanalyzed in this work

# Bassani-Sternberg, ..., Mann Mol Cell Proteomics 2015 [PMID 25576301]
PMID=25576301
mkdir -p raw/$PMID
wget -q https://www.mcponline.org/highwire/filestream/35026/field_highwire_adjunct_files/7/mcp.M114.042812-4.xlsx -P raw/$PMID

# Mommen, ..., Heck PNAS 2014 [PMID 24616531]
PMID=24616531
mkdir -p raw/$PMID
wget -q https://www.pnas.org/highwire/filestream/615485/field_highwire_adjunct_files/1/sd01.xlsx -P raw/$PMID

# Gloger, ..., Neri Cancer Immunol Immunother 2016 [PMID 27600516]
# Data extracted from supplemental PDF table.
PMID=27600516
mkdir -p raw/$PMID
wget -q https://github.com/openvax/mhcflurry/releases/download/pan-dev1/27600516.peptides.csv -P raw/$PMID

# Ritz, ..., Fugmann Proteomics 2016 [PMID 26992070]
# Supplemental zip downloaded from publication
PMID=26992070
mkdir -p raw/$PMID
wget -q https://github.com/openvax/mhcflurry/releases/download/pan-dev1/pmic12297-sup-0001-supinfo.zip -P raw/$PMID
cd raw/$PMID
unzip pmic12297-sup-0001-supinfo.zip
cd ../..

# Shraibman, ..., Admon Mol Cell Proteomics	2016 [PMID 27412690]
PMID=27412690
mkdir -p raw/$PMID
wget -q https://www.mcponline.org/lookup/suppl/doi:10.1074/mcp.M116.060350/-/DC1/mcp.M116.060350-2.xlsx -P raw/$PMID

# Pearson, ..., Perreault J Clin Invest 2016 [PMID 27841757]
# Note: we do not use the original data from this publicaton, we use 28832583's reanalysis of it.
#

# Hassan, ..., van Veelen Mol Cell Proteomics 2015 [PMID 23481700]
PMID=23481700
mkdir -p raw/$PMID
wget -q https://www.mcponline.org/highwire/filestream/34681/field_highwire_adjunct_files/1/mcp.M112.024810-2.xls  -P raw/$PMID



cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
