#!/bin/bash
# Model select pan-allele MHCflurry Class I models and calibrate percentile ranks.
#
# Usage: GENERATE.sh <local|cluster>
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
cp $SCRIPT_DIR/additional_alleles.txt .


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


export PYTHONUNBUFFERED=1

UNSELECTED_PATH="$(mhcflurry-downloads path models_class1_pan_unselected)"

# For now we calibrate percentile ranks only for alleles for which there
# is training data. Calibrating all alleles would be too slow.
# This could be improved though.
ALLELE_LIST=$(bzcat "$UNSELECTED_PATH/models.with_mass_spec/train_data.csv.bz2" | cut -f 1 -d , | grep -v allele | uniq | sort | uniq)
ALLELE_LIST+=$(echo " " $(cat additional_alleles.txt | grep -v '#') )

for kind in combined
do
    MODELS_DIR="$UNSELECTED_PATH/models.${kind}"
    time mhcflurry-class1-select-pan-allele-models \
        --data "$MODELS_DIR/train_data.csv.bz2" \
        --models-dir "$MODELS_DIR" \
        --out-models-dir models.${kind} \
        --min-models 2 \
        --max-models 8 \
        $PARALLELISM_ARGS

    cp "$MODELS_DIR/train_data.csv.bz2" "models.${kind}/"

    # For now we calibrate percentile ranks only for alleles for which there
    # is training data. Calibrating all alleles would be too slow.
    # This could be improved though.
    time mhcflurry-calibrate-percentile-ranks \
        --models-dir models.${kind} \
        --match-amino-acid-distribution-data "$MODELS_DIR/train_data.csv.bz2" \
        --motif-summary \
        --num-peptides-per-length 100000 \
        --allele $ALLELE_LIST \
        --verbosity 1 \
        $PARALLELISM_ARGS
done

bzip2 LOG.txt
for i in $(ls LOG-worker.*.txt) ; do bzip2 $i ; done
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
