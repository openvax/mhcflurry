#!/bin/bash
#
# This download includes predictions for MHCflurry and NetMHCpan 4.0 over a
# large number of peptides encompassing almost the full proteome.
#
# Usage:
# GENERATE.sh <local|cluster> <reuse-all|reuse-none|reuse-predictions|reuse-predictions-except-mhcflurry>
#
# The first choice listed above for each argument is the default.
#
# Meanings for these arguments:
#
# FIRST ARGUMENT: where to run
# local             - run locally using NUM_JOBS cores.
# cluster           - run on cluster.
#
# SECOND ARGUMENT: whether to reuse predictions from existing downloaded data
# reuse-all         - reuse predictions and peptide / allele lists from existing
#                     downloaded data_mass_spec_benchmark.
# reuse-none        - fully self-contained run; do not reuse anything.
# reuse-predictions - reuse predictions but not peptide or allele lists. Any
#                     new peptides not already included will be run.
# reuse-predictions-except-mhcflurry
#                   - Reuse predictions except for mhcflurry.
#
set -e
set -x

DOWNLOAD_NAME=data_mass_spec_benchmark
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR=$(dirname "$SCRIPT_ABSOLUTE_PATH")
export PYTHONUNBUFFERED=1

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

cp $SCRIPT_DIR/write_proteome_peptides.py .
cp $SCRIPT_DIR/write_allele_list.py .
cp $SCRIPT_DIR/run_predictors.py .

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
    EXTRA_ARGS+=" --num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1"
else
    EXTRA_ARGS+=" --cluster-parallelism --cluster-max-retries 3 --cluster-submit-command bsub --cluster-results-workdir $HOME/mhcflurry-scratch"
fi

PEPTIDES=$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2
REFERENCES_DIR=$(mhcflurry-downloads path data_references)

if [ "${2:-reuse-none}" != "reuse-none" ]
then
    EXISTING_DATA="$(mhcflurry-downloads path $DOWNLOAD_NAME)"
    echo "Will reuse data from $EXISTING_DATA"
else
    EXISTING_DATA=""
    echo "Will NOT reuse any data"
fi

mkdir predictions

# Write out alleles
if [ "$2" == "reuse-all" ]
then
    echo "Reusing allele list"
    cp "$EXISTING_DATA/alleles.txt" .
else
    echo "Generating allele list"
    python write_allele_list.py "$PEPTIDES" --out alleles.txt
fi

# Write out and process peptides.
# First just chr1 peptides, then all peptides.
# TODO: switch this back
for subset in chr1 all
do
    if [ "$2" == "reuse-all" ]
    then
        echo "Reusing peptide list"
        cp "$EXISTING_DATA/proteome_peptides.$subset.csv.bz2" .
    else
        echo "Generating peptide list"
        SUBSET_ARG=""
        if [ "$subset" == "chr1" ]
        then
            SUBSET_ARG="--chromosome 1"
        fi
        python write_proteome_peptides.py \
            "$PEPTIDES" \
            "${REFERENCES_DIR}/uniprot_proteins.csv.bz2" \
            --out proteome_peptides.$subset.csv $SUBSET_ARG
        bzip2 proteome_peptides.$subset.csv
    fi

    # Run netmhcpan4
    for kind in el ba
    do
        OUT_DIR=predictions/${subset}.netmhcpan4.$kind
        REUSE1=""
        REUSE2=""
        if [ "$subset" == "all" ]
        then
            REUSE1="predictions/chr1.netmhcpan4.$kind"
        fi
        if [ "${2:-reuse-none}" != "reuse-none" ]
        then
            REUSE2="$EXISTING_DATA"/$OUT_DIR
        fi

        python run_predictors.py \
            proteome_peptides.$subset.csv.bz2 \
            --result-dtype "float16" \
            --predictor netmhcpan4-$kind \
            --chunk-size 1000 \
            --allele $(cat alleles.txt | grep -v '31:0102') \
            --out "$OUT_DIR" \
            --worker-log-dir "$SCRATCH_DIR/$DOWNLOAD_NAME" \
            --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.nogpu.lsf \
            --reuse-predictions "$REUSE1" "$REUSE2" $EXTRA_ARGS
    done


    # Run MHCflurry
    for kind in combined
    do
        OUT_DIR=predictions/${subset}.mhcflurry.${kind}
        REUSE1=""
        REUSE2=""
        if [ "$subset" == "all" ]
        then
            REUSE1="predictions/chr1.mhcflurry.${kind}"
        fi
        if [ "${2:-reuse-none}" != "reuse-none" ] && [ "${2:-reuse-none}" != "reuse-predictions-except-mhcflurry" ]
        then
            REUSE2="$EXISTING_DATA"/$OUT_DIR
        fi

        python run_predictors.py \
            proteome_peptides.${subset}.csv.bz2 \
            --result-dtype "float16" \
            --predictor mhcflurry \
            --chunk-size 500000 \
            --mhcflurry-batch-size 65536 \
            --mhcflurry-models-dir "$(mhcflurry-downloads path models_class1_pan)/models.$kind" \
            --allele $(cat alleles.txt) \
            --out "$OUT_DIR" \
            --worker-log-dir "$SCRATCH_DIR/$DOWNLOAD_NAME" \
            --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.gpu.lsf \
            --reuse-predictions "$REUSE1" "$REUSE2" $EXTRA_ARGS
    done
done

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
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
split -b 2000m "$RESULT" "$PARTS"
echo "Split into parts:"
ls -lh "${PARTS}"*
