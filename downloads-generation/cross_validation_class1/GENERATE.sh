#!/bin/bash

set -e
set -x

DOWNLOAD_NAME=cross_validation_class1
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR=$(dirname "$SCRIPT_ABSOLUTE_PATH")

NFOLDS=5

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

cp $SCRIPT_DIR/hyperparameters.yaml .
cp $SCRIPT_DIR/split_folds.py .
cp $SCRIPT_DIR/score.py .

time python split_folds.py \
    "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" \
    --min-measurements-per-allele 100 \
    --folds $NFOLDS \
    --random-state 1 \
    --output-pattern-test "./test.fold_{}.csv" \
    --output-pattern-train "./train.fold_{}.csv"

# Kill child processes if parent exits:
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

for fold in $(seq 0 $(expr $NFOLDS - 1))
do
    mhcflurry-class1-train-allele-specific-models \
        --data train.fold_${fold}.csv \
        --hyperparameters hyperparameters.yaml \
        --out-models-dir models.fold_${fold} \
        --min-measurements-per-allele 0 \
        --percent-rank-calibration-num-peptides-per-length 0 \
    2>&1 | tee -a LOG.train.fold_${fold}.txt &
done
wait

echo "DONE TRAINING. NOW PREDICTING."

for fold in $(seq 0 $(expr $NFOLDS - 1))
do
    mhcflurry-predict \
        test.fold_${fold}.csv \
        --models models.fold_${fold} \
        --no-throw \
        --include-individual-model-predictions \
        --out predictions.fold_${fold}.csv &
done
wait

time python score.py \
    predictions.fold_*.csv \
    --out-combined predictions.combined.csv \
    --out-scores scores.csv \
    --out-summary summary.all.csv

grep -v single summary.all.csv > summary.ensemble.csv

cp $SCRIPT_ABSOLUTE_PATH .
for i in $(ls *.txt)
do
    bzip2 $i
done
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
