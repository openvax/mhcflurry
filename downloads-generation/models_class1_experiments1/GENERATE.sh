#!/bin/bash

set -e
set -x

DOWNLOAD_NAME=models_class1_experiments1
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
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

ALLELES="HLA-A*01:01 HLA-A*02:01 HLA-A*02:03 HLA-A*02:07 HLA-A*03:01 HLA-A*11:01 HLA-A*24:02 HLA-A*29:02 HLA-A*31:01 HLA-A*68:02 HLA-B*07:02 HLA-B*15:01 HLA-B*35:01 HLA-B*44:02 HLA-B*44:03 HLA-B*51:01 HLA-B*54:01 HLA-B*57:01"

# Standard architecture on quantitative only
cp $SCRIPT_DIR/hyperparameters-standard.json .
mkdir models-standard-quantitative
time mhcflurry-class1-train-allele-specific-models \
    --data "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" \
    --only-quantitative \
    --hyperparameters hyperparameters-standard.json \
    --out-models-dir models-standard-quantitative \
    --percent-rank-calibration-num-peptides-per-length 0 \
    --alleles $ALLELES &

# Model variations on qualitative + quantitative
for mod in 0local_noL1 0local 1local dense16 dense64 noL1 
do
    cp $SCRIPT_DIR/hyperparameters-${mod}.json .
    mkdir models-${mod}
    time mhcflurry-class1-train-allele-specific-models \
        --data "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" \
        --hyperparameters hyperparameters-${mod}.json \
        --out-models-dir models-${mod} \
        --percent-rank-calibration-num-peptides-per-length 0 \
        --alleles $ALLELES &
done
wait

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
