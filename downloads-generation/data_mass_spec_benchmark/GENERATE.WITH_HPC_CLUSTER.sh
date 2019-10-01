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
exec >  >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt")
exec 2> >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt" >&2)

# Log some environment info
date
pip freeze
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

cp $SCRIPT_DIR/write_proteome_peptides.py .
cp $SCRIPT_DIR/run_mhcflurry.py .
cp $SCRIPT_DIR/write_allele_list.py .

PEPTIDES=$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2
REFERENCES_DIR=$(mhcflurry-downloads path data_references)

python write_allele_list.py "$PEPTIDES" --out alleles.txt
mkdir predictions

# First just chr1 peptides
python write_proteome_peptides.py \
    "$PEPTIDES" \
    "${REFERENCES_DIR}/uniprot_proteins.csv.bz2" \
    --chromosome 1 \
    --out proteome_peptides.chr1.csv

for kind in with_mass_spec no_mass_spec
do
    python run_mhcflurry.py \
        proteome_peptides.chr1.csv \
        --chunk-size 100000 \
        --batch-size 65536 \
        --models-dir "$(mhcflurry-downloads path models_class1_pan)/models.$kind" \
        --allele $(cat alleles.txt) \
        --out "predictions/chr1.mhcflurry.$kind" \
        --verbosity 1 \
        --worker-log-dir "$SCRATCH_DIR/$DOWNLOAD_NAME" \
        --cluster-parallelism \
        --cluster-max-retries 15 \
        --cluster-submit-command bsub \
        --cluster-results-workdir ~/mhcflurry-scratch \
        --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.lsf
done

# Now all peptides
python write_proteome_peptides.py \
    "$PEPTIDES" \
    "${REFERENCES_DIR}/uniprot_proteins.csv.bz2" \
    --out proteome_peptides.all.csv

for kind in with_mass_spec no_mass_spec
do
    python run_mhcflurry.py \
        proteome_peptides.all.csv \
        --chunk-size 500000 \
        --batch-size 65536 \
        --models-dir "$(mhcflurry-downloads path models_class1_pan)/models.$kind" \
        --allele $(cat alleles.txt) \
        --out "predictions/all.mhcflurry.$kind" \
        --verbosity 1 \
        --worker-log-dir "$SCRATCH_DIR/$DOWNLOAD_NAME" \
        --cluster-parallelism \
        --cluster-max-retries 15 \
        --cluster-submit-command bsub \
        --cluster-results-workdir ~/mhcflurry-scratch \
        --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.lsf
done


bzip2 proteome_peptides.chr1.csv
bzip2 proteome_peptides.all.csv

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
