#!/bin/bash
#
# Train pan-allele MHCflurry Class I models.
#
set -e
set -x

DOWNLOAD_NAME=models_class1_pan
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR=$(dirname "$SCRIPT_ABSOLUTE_PATH")

mkdir -p "$SCRATCH_DIR"
rm -rf "$SCRATCH_DIR/$DOWNLOAD_NAME"
mkdir "$SCRATCH_DIR/$DOWNLOAD_NAME"

# Send stdout and stderr to a logfile included with the archive.
#exec >  >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt")
#exec 2> >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt" >&2)

# Log some environment info
date
pip freeze
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

mkdir models

python $SCRIPT_DIR/generate_hyperparameters.py > hyperparameters.yaml

GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
echo "Detected GPUS: $GPUS"

PROCESSORS=$(getconf _NPROCESSORS_ONLN)
echo "Detected processors: $PROCESSORS"

time mhcflurry-class1-train-pan-allele-models \
    --data "$(mhcflurry-downloads path data_curated)/curated_training_data.with_mass_spec.csv.bz2" \
    --allele-sequences "$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv" \
    --pretrain-data "$(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2" \
    --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
    --ensemble-size 4 \
    --hyperparameters hyperparameters.yaml \
    --out-models-dir models-unselected \


    #--num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
tar -cjf "../${DOWNLOAD_NAME}.with_unselected.tar.bz2" *
echo "Created archive: $SCRATCH_DIR/${DOWNLOAD_NAME}.with_unselected.tar.bz2"

ls * | grep -v models-unselected | xargs -I {} tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" {}
echo "Created archive: $SCRATCH_DIR/${DOWNLOAD_NAME}.tar.bz2"