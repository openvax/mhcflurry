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

DOWNLOAD_NAME=models_class1_processing
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

VARIANTS=( short_flanks with_flanks no_n_flank no_c_flank no_flank )

cp $SCRIPT_DIR/generate_hyperparameters.variants.py .
cp $SCRIPT_DIR/generate_hyperparameters.base.py .
python generate_hyperparameters.base.py > hyperparameters.base.yaml
for kind in "${VARIANTS[@]}"
do
    python generate_hyperparameters.variants.py hyperparameters.base.yaml $kind > hyperparameters.$kind.yaml
done

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
    cp $SCRIPT_DIR/make_train_data.py .
    time python make_train_data.py \
        --hits "$(pwd)/hits_with_tpm.csv.bz2" \
        --predictions "$(mhcflurry-downloads path data_predictions)/predictions/all.mhcflurry.combined" \
        --proteome-peptides "$(mhcflurry-downloads path data_predictions)/proteome_peptides.all.csv.bz2" \
        --ppv-multiplier 100 \
        --hit-multiplier-to-take 2 \
        --out "$(pwd)/train_data.csv"
    bzip2 -f train_data.csv
fi
TRAIN_DATA="$(pwd)/train_data.csv.bz2"


for kind in "${VARIANTS[@]}"
do
    CONTINUE_INCOMPLETE_ARGS=""
    if [ "$2" == "continue-incomplete" ] && [ -d "models.unselected.$kind" ]
    then
        echo "Will continue existing run: $kind"
        CONTINUE_INCOMPLETE_ARGS="--continue-incomplete"
    fi

    mhcflurry-class1-train-processing-models \
        --data "$TRAIN_DATA" \
        --held-out-samples 10 \
        --num-folds 4 \
        --hyperparameters "$(pwd)/hyperparameters.$kind.yaml" \
        --out-models-dir "$(pwd)/models.unselected.$kind" \
        --worker-log-dir "$SCRATCH_DIR/$DOWNLOAD_NAME" \
        $PARALLELISM_ARGS $CONTINUE_INCOMPLETE_ARGS
done

echo "Done training. Beginning model selection."

for kind in "${VARIANTS[@]}"
do
    MODELS_DIR="$(pwd)/models.unselected.$kind"
    mhcflurry-class1-select-processing-models \
        --data "$MODELS_DIR/train_data.csv.bz2" \
        --models-dir "$MODELS_DIR" \
        --out-models-dir "$(pwd)/models.selected.$kind" \
        --min-models 1 \
        --max-models 2 \
        $PARALLELISM_ARGS
    cp "$MODELS_DIR/train_data.csv.bz2" "models.selected.$kind/train_data.csv.bz2"
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

# Write out just the selected models
# Move unselected into a hidden dir so it is excluded in the glob (*).
mkdir .ignored
mv models.unselected.* .ignored/
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.selected.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
mv .ignored/* . && rmdir .ignored
echo "Created archive: $RESULT"
