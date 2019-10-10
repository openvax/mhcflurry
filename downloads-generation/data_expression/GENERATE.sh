#!/bin/bash
#
# Download published gene expression data corresponding to some of our mass
# spec datasets.
#
#
set -e
set -x

DOWNLOAD_NAME=data_expression
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

# Many cell line sequencing data is available from:
# Available from SRA [access required] at:
# https://www.ebi.ac.uk/ega/studies/EGAS00001000610

# CCLE as processed by expression atlas
DATASET=expression-atlas-22460905
mkdir $DATASET
cd $DATASET
#wget -q https://www.ebi.ac.uk/gxa/experiments-content/E-MTAB-2770/resources/ExperimentDownloadSupplier.RnaSeqBaseline/tpms.tsv
ls -lh .
cd ..

# Human protein atlas
DATASET=human-protein-atlas
mkdir $DATASET
cd $DATASET
wget -q https://www.proteinatlas.org/download/rna_celline.tsv.zip
wget -q https://www.proteinatlas.org/download/rna_blood_cell_schmiedel.tsv.zip
wget -q https://www.proteinatlas.org/download/rna_blood_cell_sample_tpm_m.tsv.zip
wget -q https://www.proteinatlas.org/download/rna_tissue_gtex.tsv.zip
for i in $(ls *.zip)
do
    unzip $i
    rm $i
done
ls -lh .
cd ..

#DATASET=HEK293
#mkdir $DATASET
#cd $DATASET
#wget -q  "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE85161&format=file" -O GSE85161_RAW.tar
#tar -xvf GSE85161_RAW.tar
#rm GSE85161_RAW.tar
#ls -lh .
#cd ..

# CCLE cell lines
# Other source of CCLE information
#DATASET=ccle
#mkdir $DATASET
#cd $DATASET
#wget -q https://data.broadinstitute.org/ccle/CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz
#wget -q https://data.broadinstitute.org/ccle/CCLE_miRNA_20181103.gct
#cd ..

# PBMC
DATASET=pbmc
mkdir $DATASET
cd $DATASET
wget -q ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE107nnn/GSE107011/suppl/GSE107011_Processed_data_TPM.txt.gz
cd ..

# B721.221
#DATASET=b721221
#mkdir $DATASET
#cd $DATASET
#wget -q https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE93315&format=file
#cd ..

#DATASET=pancan-xena
#mkdir $DATASET
#cd $DATASET
#wget -q https://pancanatlas.xenahubs.net/download/probeMap/hugo_gencode_good_hg19_V24lift37_probemap
#wget -q https://pancanatlas.xenahubs.net/download/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz
#cd ..

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
