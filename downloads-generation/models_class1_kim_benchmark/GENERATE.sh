#!/bin/bash
#
set -x

DOWNLOAD_NAME=models_class1_kim_benchmark
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

cp $SCRIPT_DIR/curate.py .
cp $SCRIPT_DIR/write_validation_data.py .

time python curate.py \
    --data-kim2014 \
        "$(mhcflurry-downloads path data_published)/bdata.2009.mhci.public.1.txt" \
    --out-csv train.csv

bzip2 train.csv

mkdir models
cp $SCRIPT_DIR/class1_pseudosequences.csv .
python $SCRIPT_DIR/generate_hyperparameters.py > hyperparameters.yaml

GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
echo "Detected GPUS: $GPUS"

PROCESSORS=$(getconf _NPROCESSORS_ONLN)
echo "Detected processors: $PROCESSORS"

time mhcflurry-class1-train-allele-specific-models \
    --data "train.csv.bz2" \
    --allele-sequences class1_pseudosequences.csv \
    --hyperparameters hyperparameters.yaml \
    --out-models-dir models \
    --held-out-fraction-reciprocal 10 \
    --min-measurements-per-allele 20 \
    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50

time python ./write_validation_data.py \
    --include "train.csv.bz2" \
    --exclude "models/train_data.csv.bz2" \
    --only-alleles-present-in-exclude \
    --out-data test.csv \
    --out-summary test.summary.csv

wc -l test.csv

mkdir selected-models
time mhcflurry-class1-select-allele-specific-models \
    --data test.csv \
    --models-dir models \
    --out-models-dir selected-models \
    --scoring combined:mse,consensus \
    --consensus-num-peptides-per-length 10000 \
    --combined-min-models 8 \
    --combined-max-models 16 \
    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 5

time mhcflurry-calibrate-percentile-ranks \
    --models-dir selected-models \
    --num-peptides-per-length 100000 \
    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
