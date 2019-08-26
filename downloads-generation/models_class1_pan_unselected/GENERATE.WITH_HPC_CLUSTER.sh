#!/bin/bash
#
# Train pan-allele MHCflurry Class I models.
#
# Uses an HPC cluster (Mount Sinai chimera cluster, which uses lsf job
# scheduler). This would need to be modified for other sites.
#
set -e
set -x

DOWNLOAD_NAME=models_class1_pan_unselected
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR=$(dirname "$SCRIPT_ABSOLUTE_PATH")

mkdir -p "$SCRATCH_DIR"
if [ "$1" != "continue-incomplete" ]
then
    echo "Fresh run"
    rm -rf "$SCRATCH_DIR/$DOWNLOAD_NAME"
    mkdir "$SCRATCH_DIR/$DOWNLOAD_NAME"
else
    echo "Continuing incomplete run"
fi

# Send stdout and stderr to a logfile included with the archive.
LOG="$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.$(date +%s).txt"
exec >  >(tee -ia "$LOG")
exec 2> >(tee -ia "$LOG" >&2)

# Log some environment info
echo "Invocation: $0 $@"
date
pip freeze
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ "$1" != "continue-incomplete" ]
then
    cp $SCRIPT_DIR/generate_hyperparameters.py .
    python generate_hyperparameters.py > hyperparameters.yaml
fi

for kind in with_mass_spec no_mass_spec
do
    EXTRA_TRAIN_ARGS=""
    if [ "$1" == "continue-incomplete" ] && [ -d "models.${kind}" ]
    then
        echo "Will continue existing run: $kind"
        EXTRA_TRAIN_ARGS="--continue-incomplete"
    fi

    mhcflurry-class1-train-pan-allele-models \
        --data "$(mhcflurry-downloads path data_curated)/curated_training_data.${kind}.csv.bz2" \
        --allele-sequences "$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv" \
        --pretrain-data "$(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2" \
        --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
        --ensemble-size 4 \
        --hyperparameters hyperparameters.yaml \
        --out-models-dir $(pwd)/models.${kind} \
        --worker-log-dir "$SCRATCH_DIR/$DOWNLOAD_NAME" \
        --verbosity 0 \
        --cluster-parallelism \
        --cluster-submit-command bsub \
        --cluster-results-workdir ~/mhcflurry-scratch \
        --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.lsf \
        $EXTRA_TRAIN_ARGS
done

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 -f "$LOG"
for i in $(ls LOG-worker.*.txt) ; do bzip2 -f $i ; done
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"

# Split into <2GB chunks for GitHub
PARTS="${RESULT}.part."
# Check for pre-existing part files and rename them.
for i in $(ls "${PARTS}"* )
do
    DEST="${i}.OLD.$(date +%s)"
    echo "WARNING: already exists: $i . Moving to $DEST"
    mv $i $DEST
done
split -b 2000M "$RESULT" "$PARTS"
echo "Split into parts:"
ls -lh "${PARTS}"*
