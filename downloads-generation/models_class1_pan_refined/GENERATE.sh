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
    PARALLELISM_ARGS+=" --cluster-parallelism --cluster-max-retries 3 --cluster-submit-command bsub --cluster-results-workdir $HOME/mhcflurry-scratch --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.lsf"
fi

rm -rf "$SCRATCH_DIR/$DOWNLOAD_NAME"
mkdir -p "$SCRATCH_DIR/$DOWNLOAD_NAME"

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

cp $SCRIPT_DIR/make_multiallelic_training_data.py .
cp $SCRIPT_DIR/hyperparameters.yaml .

MONOALLELIC_TRAIN="$(mhcflurry-downloads path models_class1_pan)/models.with_mass_spec/train_data.csv.bz2"

# ********************************************************
# First we refine a single model excluding chromosome 1.
if false; then
    echo "Beginning testing run."
    time python make_multiallelic_training_data.py \
        --hits "$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2" \
        --expression "$(mhcflurry-downloads path data_curated)/rna_expression.csv.bz2" \
        --exclude-contig "1" \
        --decoys-per-hit 1 \
        --out train.multiallelic.no_chr1.csv

    time mhcflurry-multiallelic-refinement \
        --monoallelic-data "$MONOALLELIC_TRAIN" \
        --multiallelic-data train.multiallelic.no_chr1.csv \
        --models-dir "$(mhcflurry-downloads path models_class1_pan)/models.with_mass_spec" \
        --hyperparameters hyperparameters.yaml \
        --out-affinity-predictor-dir $(pwd)/test_models.no_chr1.affinity \
        --out-presentation-predictor-dir $(pwd)/test_models.no_chr1.presentation \
        --worker-log-dir "$SCRATCH_DIR/$DOWNLOAD_NAME" \
        $PARALLELISM_ARGS

    time mhcflurry-calibrate-percentile-ranks \
        --models-dir $(pwd)/test_models.no_chr1.affinity   \
        --match-amino-acid-distribution-data "$MONOALLELIC_TRAIN" \
        --motif-summary \
        --num-peptides-per-length 100000 \
        --allele "HLA-A*02:01" "HLA-A*02:20" "HLA-C*02:10" \
        --verbosity 1 \
        $PARALLELISM_ARGS
fi

# ********************************************************
echo "Beginning production run"
if [ -f "$SCRIPT_DIR/train.multiallelic.csv" ]; then
    echo "Using existing multiallelic train data."
    cp "$SCRIPT_DIR/train.multiallelic.csv" .
else
    time python make_multiallelic_training_data.py \
        --hits "$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2" \
        --expression "$(mhcflurry-downloads path data_curated)/rna_expression.csv.bz2" \
        --decoys-per-hit 1 \
        --out train.multiallelic.csv \
        --alleles "HLA-A*02:20"
fi

ALLELE_LIST=$(bzcat "$MONOALLELIC_TRAIN" | cut -f 1 -d , | grep -v allele | uniq | sort | uniq)
ALLELE_LIST+=$(cat train.multiallelic.csv | cut -f 7 -d , | gerp -v hla | uniq | tr ' ' '\n' | sort | uniq)
ALLELE_LIST+=$(echo " " $(cat $SCRIPT_DIR/additional_alleles.txt | grep -v '#') )

time mhcflurry-multiallelic-refinement \
    --monoallelic-data "$MONOALLELIC_TRAIN" \
    --multiallelic-data train.multiallelic.csv \
    --models-dir "$(mhcflurry-downloads path models_class1_pan)/models.with_mass_spec" \
    --hyperparameters hyperparameters.yaml \
    --out-affinity-predictor-dir $(pwd)/models.affinity \
    --out-presentation-predictor-dir $(pwd)/models.presentation \
    --worker-log-dir "$SCRATCH_DIR/$DOWNLOAD_NAME" \
    --only-alleles-with-mass-spec \
    $PARALLELISM_ARGS

time mhcflurry-calibrate-percentile-ranks \
    --models-dir $(pwd)/models.affinity  \
    --match-amino-acid-distribution-data "$MONOALLELIC_TRAIN" \
    --motif-summary \
    --num-peptides-per-length 100000 \
    --allele $ALLELE_LIST \
    --verbosity 1 \
    $PARALLELISM_ARGS

echo "Done training."

#rm train.multiallelic.*

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 -f "$LOG"
for i in $(ls LOG-worker.*.txt) ; do bzip2 -f $i ; done
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
