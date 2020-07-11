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

DOWNLOAD_NAME=analysis_predictor_info
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

cd $SCRATCH_DIR/$DOWNLOAD_NAME

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

####### GENERATION OF BINDING MOTIFS AND OTHER ARTIFACTS #######

if [ "$2" == "continue-incomplete" ] && [ -f "motifs/artifacts.csv" ]
then
    echo "Reusing existing artifacts"
else
    echo "Using affinity predictor:"
    cat "$(mhcflurry-downloads path models_class1_pan)/models.combined/info.txt"

    mkdir motifs
    cp "$(mhcflurry-downloads path models_class1_pan)/models.combined/info.txt" motifs/

    cp $SCRIPT_DIR/generate_artifacts.py .
    time python generate_artifacts.py \
        --affinity-predictor "$(mhcflurry-downloads path models_class1_pan)/models.combined" \
        --out "$(pwd)/motifs" \
        $PARALLELISM_ARGS
fi

####### EVALUATION ON MODEL SELECTION DATA #######

if [ "$2" == "continue-incomplete" ] && [ -f "model_selection_with_decoys.csv.bz2" ]
then
    echo "Reusing existing model_selection_with_decoys data"
else
    echo "Using affinity predictor:"
    cat "$(mhcflurry-downloads path models_class1_pan)/models.combined/info.txt"

    cp $SCRIPT_DIR/generate_model_selection_with_decoys.py .
    time python generate_model_selection_with_decoys.py \
        "$(mhcflurry-downloads path models_class1_pan)/models.combined/model_selection_data.csv.bz2" \
        --proteome-peptides "$(mhcflurry-downloads path data_references)/uniprot_proteins.csv.bz2" \
        --out "$(pwd)/model_selection_with_decoys.csv"
    bzip2 -f model_selection_with_decoys.csv
fi

if [ "$2" == "continue-incomplete" ] && [ -f "model_selection_with_decoys.predictions.selected.csv.bz2" ]
then
    echo "Reusing existing model_selection_with_decoys.predictions.selected.data"
else
    echo "Using affinity predictor:"
    cat "$(mhcflurry-downloads path models_class1_pan)/models.combined/info.txt"

    cp $SCRIPT_DIR/predict_on_model_selection_data.py .
    time python predict_on_model_selection_data.py \
        "$(mhcflurry-downloads path models_class1_pan)/models.combined" \
        --data "$(pwd)/model_selection_with_decoys.csv.bz2" \
        --out "$(pwd)/model_selection_with_decoys.predictions.selected.csv" \
        $PARALLELISM_ARGS
    bzip2 -f model_selection_with_decoys.predictions.selected.csv
fi

if [ "$2" == "continue-incomplete" ] && [ -f "model_selection_with_decoys.predictions.unselected.csv.bz2" ]
then
    echo "Reusing existing model_selection_with_decoys.predictions.unselected data"
else
    echo "Using affinity predictor:"
    cat "$(mhcflurry-downloads path models_class1_pan_unselected)/models.unselected.combined/info.txt"

    cp $SCRIPT_DIR/predict_on_model_selection_data.py .
    time python predict_on_model_selection_data.py \
        "$(mhcflurry-downloads path models_class1_pan_unselected)/models.unselected.combined" \
        --data "$(pwd)/model_selection_with_decoys.csv.bz2" \
        --out "$(pwd)/model_selection_with_decoys.predictions.unselected.csv" \
        $PARALLELISM_ARGS
    bzip2 -f model_selection_with_decoys.predictions.unselected.csv
fi


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
