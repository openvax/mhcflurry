#!/bin/bash
# Model select pan-allele MHCflurry Class I models and calibrate percentile ranks.
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
exec >  >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt")
exec 2> >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt" >&2)

# Log some environment info
date
pip freeze
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

cp $SCRIPT_ABSOLUTE_PATH .

GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
echo "Detected GPUS: $GPUS"

PROCESSORS=$(getconf _NPROCESSORS_ONLN)
echo "Detected processors: $PROCESSORS"

NUM_JOBS=$GPUS
if [ "$NUM_JOBS" -eq "0" ]; then
   NUM_JOBS=1
fi
echo "Num jobs: $NUM_JOBS"

export PYTHONUNBUFFERED=1

UNSELECTED_PATH="$(mhcflurry-downloads path models_class1_pan_unselected)"

for kind in with_mass_spec #no_mass_spec
do
    MODELS_DIR="$UNSELECTED_PATH/models.${kind}"
    time mhcflurry-class1-select-pan-allele-models \
        --data "$MODELS_DIR/train_data.csv.bz2" \
        --models-dir "$MODELS_DIR" \
        --out-models-dir models.${kind} \
        --min-models 8 \
        --max-models 32 \
        --num-jobs 0 \
        --num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1

    time mhcflurry-calibrate-percentile-ranks \
        --models-dir models.${kind} \
        --match-amino-acid-distribution-data "$MODELS_DIR/train_data.csv.bz2" \
        --motif-summary \
        --num-peptides-per-length 100000 \
        --verbosity 1 \
        --num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1
done

bzip2 LOG.txt
for i in $(ls LOG-worker.*.txt) ; do bzip2 $i ; done
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
