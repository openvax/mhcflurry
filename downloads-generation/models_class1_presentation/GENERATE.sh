#!/bin/bash
#
#
# Usage: GENERATE.sh <local|cluster> <fresh|continue-incomplete>
#
# cluster mode uses an HPC cluster (Mount Sinai chimera cluster, which uses lsf job
# scheduler). This would need to be modified for other sites.
#
set -e
set -x

DOWNLOAD_NAME=models_class1_presentation
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
    PARALLELISM_ARGS+=" --cluster-parallelism --cluster-max-retries 3 --cluster-submit-command bsub --cluster-results-workdir $HOME/mhcflurry-scratch --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.lsf"
fi

mkdir -p "$SCRATCH_DIR"
if [ "$2" != "continue-incomplete" ]
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
mhcflurry-downloads info

cd $SCRATCH_DIR/$DOWNLOAD_NAME

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ "$2" == "continue-incomplete" ] && [ -f "hits_with_tpm.csv.bz2" ]
then
    echo "Reusing existing expression-annotated hits data"
else
    cp $SCRIPT_DIR/annotate_hits_with_expression.py .
    time python annotate_hits_with_expression.py \
        --hits "$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2" \
        --expression "$(mhcflurry-downloads path data_curated)/rna_expression.csv.bz2" \
        --out "$(pwd)/hits_with_tpm.csv"
    bzip2 -f hits_with_tpm.csv
fi

if [ "$2" == "continue-incomplete" ] && [ -f "train_data.csv.bz2" ]
then
    echo "Reusing existing training data"
else
    cp $SCRIPT_DIR/make_benchmark.py .
    time python make_benchmark.py \
        --hits "$(pwd)/hits_with_tpm.csv.bz2" \
        --proteome-peptides "$(mhcflurry-downloads path data_mass_spec_benchmark)/proteome_peptides.all.csv.bz2" \
        --decoys-per-hit 99 \
        --exclude-pmid 31844290 31495665 31154438 \
        --only-format MULTIALLELIC \
        --out "$(pwd)/train_data.csv"
    bzip2 -f train_data.csv
fi

mhcflurry-class1-train-presentation-models \
    --data "$(pwd)/train_data.csv.bz2" \
    --affinity-predictor "$(mhcflurry-downloads path models_class1_pan)/models.combined" \
    --cleavage-predictor-with-flanks "$(mhcflurry-downloads path models_class1_cleavage)/models.selected" \
    --cleavage-predictor-without-flanks "$(mhcflurry-downloads path models_class1_cleavage_variants)/models.selected.no_flank" \
    --out-models-dir "$(pwd)/models"

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 -f "$LOG"
for i in $(ls LOG-worker.*.txt) ; do bzip2 -f $i ; done
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
