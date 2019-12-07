#!/bin/bash
#
# Uses an HPC cluster (Mount Sinai chimera cluster, which uses lsf job
# scheduler). This would need to be modified for other sites.
#
# Usage: GENERATE.sh <local|cluster>
#
set -e
set -x

DOWNLOAD_NAME=models_class1_pan_refined
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR=$(dirname "$SCRIPT_ABSOLUTE_PATH")

if [ "$1" != "cluster" ]
then
    GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
    echo "Detected GPUS: $GPUS"

    PROCESSORS=$(getconf _NPROCESSORS_ONLN)
    echo "Detected processors: $PROCESSORS"

    if [ "$GPUS" -eq "0" ]; then
       NUM_JOBS=${NUM_JOBS-1}
    else
        NUM_JOBS=${NUM_JOBS-$GPUS}
    fi
    echo "Num jobs: $NUM_JOBS"
    PARALLELISM_ARGS+=" --num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1"
else
    PARALLELISM_ARGS+=" --cluster-parallelism --cluster-max-retries 3 --cluster-submit-command bsub --cluster-results-workdir $HOME/mhcflurry-scratch --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.gpu.lsf"
fi

rm -rf "$SCRATCH_DIR/$DOWNLOAD_NAME"
mkdir -p "$SCRATCH_DIR/$DOWNLOAD_NAME"

# Send stdout and stderr to a logfile included with the archive.
#LOG="$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.$(date +%s).txt"
#exec >  >(tee -ia "$LOG")
#exec 2> >(tee -ia "$LOG" >&2)

# Log some environment info
#echo "Invocation: $0 $@"
#date
#pip freeze
#git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# CAHNGE TO CP
ln -s $SCRIPT_DIR/make_multiallelic_training_data.py .

time python make_multiallelic_training_data.py \
    --hits "$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2" \
    --expression "$(mhcflurry-downloads path data_curated)/rna_expression.csv.bz2" \
    --out train.multiallelic.csv

time mhcflurry-multiallelic-refinement \
    --monoallelic-data "$(mhcflurry-downloads path data_curated)/curated_training_data.with_mass_spec.csv.bz2" \
    --multiallelic-data train.multiallelic.csv \
    --models-dir "$(mhcflurry-downloads path models_class1_pan)/models.with_mass_spec"
    --hyperparameters hyperparameters.yaml \
    --out-affinity-predictor-dir $(pwd)/models.affinity \
    --out-presentation-predictor-dir $(pwd)/models.presentation \
    --worker-log-dir "$SCRATCH_DIR/$DOWNLOAD_NAME" \
    $PARALLELISM_ARGS

echo "Done training."

bzip2 train.multiallelic.csv

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 -f "$LOG"
for i in $(ls LOG-worker.*.txt) ; do bzip2 -f $i ; done
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"

