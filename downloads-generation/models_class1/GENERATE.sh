#!/bin/bash
#
# Model select standard MHCflurry Class I models.
#
set -e
set -x

DOWNLOAD_NAME=models_class1
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
cp $SCRIPT_DIR/write_validation_data.py .

mkdir models

GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
echo "Detected GPUS: $GPUS"

PROCESSORS=$(getconf _NPROCESSORS_ONLN)
echo "Detected processors: $PROCESSORS"

python ./write_validation_data.py \
    --include "$(mhcflurry-downloads path data_curated)/curated_training_data.with_mass_spec.csv.bz2" \
    --exclude "$(mhcflurry-downloads path models_class1_unselected)/models/train_data.csv.bz2" \
    --only-alleles-present-in-exclude \
    --out-data test.csv \
    --out-summary test.summary.csv

wc -l test.csv

time mhcflurry-class1-select-allele-specific-models \
    --data test.csv \
    --models-dir "$(mhcflurry-downloads path models_class1_unselected)/models" \
    --out-models-dir models \
    --scoring combined:mass-spec,mse,consensus \
    --consensus-num-peptides-per-length 10000 \
    --combined-min-models 8 \
    --combined-max-models 16 \
    --unselected-accuracy-scorer combined:mass-spec,mse \
    --unselected-accuracy-percentile-threshold 95 \
    --mass-spec-min-measurements 500 \
    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 1

time mhcflurry-calibrate-percentile-ranks \
    --models-dir models \
    --num-peptides-per-length 100000 \
    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
