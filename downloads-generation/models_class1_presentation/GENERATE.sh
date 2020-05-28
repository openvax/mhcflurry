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
        --proteome-peptides "$(mhcflurry-downloads path data_predictions)/proteome_peptides.all.csv.bz2" \
        --decoys-per-hit 2 \
        --exclude-pmid 31844290 31495665 31154438 \
        --only-format MULTIALLELIC \
        --out "$(pwd)/train_data.csv"
    bzip2 -f train_data.csv
fi

rm -rf commands
mkdir commands

if [ "$2" == "continue-incomplete" ] && [ -f "models/weights.csv" ]
then
    echo "Reusing existing trained predictor"
else
    echo time mhcflurry-class1-train-presentation-models \
        --data "$(pwd)/train_data.csv.bz2" \
        --affinity-predictor \""$(mhcflurry-downloads path models_class1_pan)/models.combined"\" \
        --processing-predictor-with-flanks \""$(mhcflurry-downloads path models_class1_processing)/models.selected.with_flanks"\" \
        --processing-predictor-without-flanks \""$(mhcflurry-downloads path models_class1_processing)/models.selected.no_flank"\" \
        --out-models-dir "$(pwd)/models" >> commands/train.sh
fi

if [ "$2" == "continue-incomplete" ] && [ -f "models/percent_ranks.csv" ]
then
    echo "Reusing existing percentile ranks"
else
    echo time mhcflurry-calibrate-percentile-ranks \
        --models-dir "$(pwd)/models" \
        --match-amino-acid-distribution-data \""$(mhcflurry-downloads path models_class1_pan)/models.combined/train_data.csv.bz2"\" \
        --alleles-file \""$(mhcflurry-downloads path models_class1_pan)/models.combined/train_data.csv.bz2"\" \
        --predictor-kind class1_presentation \
        --num-peptides-per-length 100000 \
        --alleles-per-genotype 1 \
        --verbosity 1 >> commands/calibrate_percentile_ranks.sh
fi

ls -lh commands

if [ "$1" != "cluster" ]
then
    echo "Running locally"
    for i in $(ls commands/*.sh)
    do
        echo "# *******"
        echo "# Command $i"
        cat $i
        bash $i
    done
else
    echo "Running on cluster"
    for i in $(ls commands/*.sh)
    do
        echo "# *******"
        echo "# Command $i"
        cat $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.lsf > ${i}.lsf
        echo cd "$(pwd)" >> ${i}.lsf
        cat $i >> ${i}.lsf
        cat ${i}.lsf
        bsub -K < "${i}.lsf" &
    done
    wait
fi

cp "$(mhcflurry-downloads path models_class1_pan)/models.combined/train_data.csv.bz2" models/affinity_predictor_train_data.csv.bz2
cp "$(mhcflurry-downloads path models_class1_processing)/models.selected.with_flanks/train_data.csv.bz2" models/processing_predictor_train_data.csv.bz2
cp "$(mhcflurry-downloads path models_class1_processing)/models.selected.no_flank/train_data.csv.bz2" models/processing_predictor_no_flank_train_data.csv.bz2

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 -f "$LOG"
for i in $(ls LOG-worker.*.txt) ; do bzip2 -f $i ; done
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
mkdir .ignored
mv hits_with_tpm.csv.bz2 .ignored/
tar -cjf "$RESULT" *
mv .ignored/* . && rmdir .ignored
echo "Created archive: $RESULT"
