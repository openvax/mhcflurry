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
# BINDING AFFINITIES: class I
############################################
#
# Kim et al 2014 [PMID 25017736]
wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/bdata.2009.mhci.public.1.txt
wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/bdata.20130222.mhci.public.1.txt
wget -q https://github.com/openvax/mhcflurry/releases/download/pre-1.1/bdata.2013.mhci.public.blind.1.txt

mkdir ms

############################################
# MS: Class I
############################################
# Bassani-Sternberg, ..., Gfeller PLOS Comp. Bio. 2017 [PMID 28832583]
# The first dataset is from this work. The second dataset is originally from:
#   Pearson, ..., Perreault JCI 2016 [PMID 27841757]
# but was reanalyzed in this work, and we download the reanalyzed version here.
PMID=28832583
mkdir -p ms/$PMID
wget -q https://doi.org/10.1371/journal.pcbi.1005725.s002 -P ms/$PMID # data generated in this work
wget -q https://doi.org/10.1371/journal.pcbi.1005725.s003 -P ms/$PMID # data reanalyzed in this work
cd ms/$PMID
unzip *.s002
unzip *.s003
mkdir saved
mv Dataset*/Dataset*.txt saved
rm -rf Dataset* *.s002 *.s003 _*
mv saved/* .
rmdir saved
cd ../..

# Bassani-Sternberg, ..., Mann Mol Cell Proteomics 2015 [PMID 25576301]
PMID=25576301
mkdir -p ms/$PMID
wget -q https://www.mcponline.org/highwire/filestream/35026/field_highwire_adjunct_files/7/mcp.M114.042812-4.xlsx -P ms/$PMID

# Mommen, ..., Heck PNAS 2014 [PMID 24616531]
PMID=24616531
mkdir -p ms/$PMID
wget -q https://www.pnas.org/highwire/filestream/615485/field_highwire_adjunct_files/1/sd01.xlsx -P ms/$PMID

# Gloger, ..., Neri Cancer Immunol Immunother 2016 [PMID 27600516]
# Data extracted from supplemental PDF table.
PMID=27600516
mkdir -p ms/$PMID
wget -q https://github.com/openvax/mhcflurry/releases/download/pan-dev1/27600516.peptides.csv -P ms/$PMID

# Ritz, ..., Fugmann Proteomics 2016 [PMID 26992070]
# Supplemental zip downloaded from publication
PMID=26992070
mkdir -p ms/$PMID
wget -q https://github.com/openvax/mhcflurry/releases/download/pan-dev1/pmic12297-sup-0001-supinfo.zip -P ms/$PMID
cd ms/$PMID
unzip pmic12297-sup-0001-supinfo.zip
cd ../..

# Shraibman, ..., Admon Mol Cell Proteomics	2016 [PMID 27412690]
PMID=27412690
mkdir -p ms/$PMID
wget -q https://www.mcponline.org/lookup/suppl/doi:10.1074/mcp.M116.060350/-/DC1/mcp.M116.060350-2.xlsx -P ms/$PMID

# Pearson, ..., Perreault J Clin Invest 2016 [PMID 27841757]
# Note: we do not use the original data from this publicaton, we use 28832583's reanalysis of it.
#

# Hassan, ..., van Veelen Mol Cell Proteomics 2015 [PMID 23481700]
PMID=23481700
mkdir -p ms/$PMID
wget -q https://www.mcponline.org/highwire/filestream/34681/field_highwire_adjunct_files/1/mcp.M112.024810-2.xls -P ms/$PMID

# Shraibman, ..., Admon Mol Cell Proteomics 2019 [PMID 31154438]
PMID=31154438
mkdir -p ms/$PMID
wget -q https://www.mcponline.org/highwire/filestream/51948/field_highwire_adjunct_files/3/zjw006195963st2.txt -P ms/$PMID
wget -q https://www.mcponline.org/highwire/filestream/51948/field_highwire_adjunct_files/1/zjw006195963st1.xlsx -P ms/$PMID

# Bassani-Sternberg, ..., Krackhardt Nature Comm. 2016 [PMID 27869121]
PMID=27869121
mkdir -p ms/$PMID
wget -q "https://static-content.springer.com/esm/art%3A10.1038%2Fncomms13404/MediaObjects/41467_2016_BFncomms13404_MOESM1318_ESM.xlsx" -P ms/$PMID

# Sarkizova, ..., Keskin Nature Biotechnology 2019 [PMID 31844290]
PMID=31844290
mkdir -p ms/$PMID
# Monoallelic:
wget -q "https://static-content.springer.com/esm/art%3A10.1038%2Fs41587-019-0322-9/MediaObjects/41587_2019_322_MOESM3_ESM.xlsx" -P ms/$PMID
# Multiallelic:
wget -q "https://static-content.springer.com/esm/art%3A10.1038%2Fs41587-019-0322-9/MediaObjects/41587_2019_322_MOESM4_ESM.xlsx" -P ms/$PMID


############################################
# MS: Class II
############################################
# Abelin, ..., Rooney Immunity 2019 [PMID 31495665]
PMID=31495665
mkdir -p ms/$PMID
wget -q https://ars.els-cdn.com/content/image/1-s2.0-S1074761319303632-mmc2.xlsx -P ms/$PMID


############################################
# RNA-seq expression data (TPMs)
############################################
# CCLE as processed by expression atlas
DATASET=expression-atlas-22460905
mkdir -p expression/$DATASET
wget -q https://www.ebi.ac.uk/gxa/experiments-content/E-MTAB-2770/resources/ExperimentDownloadSupplier.RnaSeqBaseline/tpms.tsv -P expression/$DATASET

# Human protein atlas
DATASET=human-protein-atlas
mkdir -p expression/$DATASET
cd expression/$DATASET
wget -q https://www.proteinatlas.org/download/rna_celline.tsv.zip
wget -q https://www.proteinatlas.org/download/rna_blood_cell_sample_tpm_m.tsv.zip
wget -q https://www.proteinatlas.org/download/rna_tissue_gtex.tsv.zip
for i in $(ls *.zip)
do
    unzip $i
    rm $i
done
cd ../..

# Melanoma. Original publication
# Barry, ..., Krummel Nature Medicine 2018 [PMID 29942093].
DATASET=GSE113126
mkdir -p expression/$DATASET 
cd expression/$DATASET
wget -q "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE113126&format=file" -O GSE113126_RAW.tar
tar -xvf GSE113126_RAW.tar
rm GSE113126_RAW.tar
cd ../..

############################################
# T cell epitopes: class I
############################################
#
# Koşaloğlu-Yalçın, ..., Peters. Oncoimmunology 2018 [PMID 30377561]
#
PMID=30377561
mkdir -p epitopes/$PMID
wget -q https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6204999/bin/koni-07-11-1492508-s001.zip -P epitopes/$PMID
cd epitopes/$PMID
unzip *.zip
rm -f *.jpg
cd ../..

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"

