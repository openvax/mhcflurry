#!/bin/bash
#
# Model select based on consensus (agreement with full ensemble).
# Uses models trained in models_class1_unselected download.
#
set -e
set -x

DOWNLOAD_NAME=models_class1_consensus
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

mkdir models

GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
echo "Detected GPUS: $GPUS"

PROCESSORS=$(getconf _NPROCESSORS_ONLN)
echo "Detected processors: $PROCESSORS"

time mhcflurry-class1-select-allele-specific-models \
    --models-dir "$(mhcflurry-downloads path models_class1_unselected)/models" \
    --out-models-dir models \
    --scoring consensus \
    --consensus-num-peptides-per-length 10000 \
    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50

time mhcflurry-calibrate-percentile-ranks \
    --models-dir models \
    --num-peptides-per-length 100000 \
    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
