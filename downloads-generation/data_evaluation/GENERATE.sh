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

DOWNLOAD_NAME=data_evaluation
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

### GENERATE BENCHMARK: MONOALLELIC
if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.monoallelic.csv.bz2" ]
then
    echo "Reusing existing monoallelic benchmark"
else
    cp $SCRIPT_DIR/make_benchmark.py .
    time python make_benchmark.py \
        --hits "$(pwd)/hits_with_tpm.csv.bz2" \
        --proteome-peptides "$(mhcflurry-downloads path data_mass_spec_benchmark)/proteome_peptides.all.csv.bz2" \
        --decoys-per-hit 99 \
        --only-format MONOALLELIC \
        --out "$(pwd)/benchmark.monoallelic.csv"
    bzip2 -f benchmark.monoallelic.csv
    rm -f benchmark.monoallelic.predictions.csv.bz2
fi

### AFFINITY PREDICTOR VARIANT: MONOALLELIC
if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.monoallelic.predictions.csv.bz2" ]
then
    echo "Reusing existing monoallelic benchmark predictions"
else
    time mhcflurry-predict \
        benchmark.monoallelic.csv.bz2 \
        --allele-column hla \
        --prediction-column-prefix no_additional_ms_ \
        --models "$(mhcflurry-downloads path models_class1_pan_variants)/models.no_additional_ms" \
        --affinity-only \
        --no-affinity-percentile \
        --out benchmark.monoallelic.predictions.csv \
        --no-throw
    bzip2 -f benchmark.monoallelic.predictions.csv
    ls -lh benchmark.monoallelic.predictions.csv.bz2
fi

### GENERATE BENCHMARK: MULTIALLELIC
if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.csv.bz2" ]
then
    echo "Reusing existing multiallelic benchmark"
else
    cp $SCRIPT_DIR/make_benchmark.py .
    time python make_benchmark.py \
        --hits "$(pwd)/hits_with_tpm.csv.bz2" \
        --proteome-peptides "$(mhcflurry-downloads path data_mass_spec_benchmark)/proteome_peptides.all.csv.bz2" \
        --decoys-per-hit 99 \
        --only-format MULTIALLELIC \
        --out "$(pwd)/benchmark.multiallelic.csv"
    bzip2 -f benchmark.multiallelic.csv
    rm -f benchmark.multiallelic.predictions1.csv.bz2
fi

### AFFINITY PREDICTORS: MULTIALLELIC
if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.predictions1.csv.bz2" ]
then
    echo "Reusing existing multiallelic predictions"
else
    cp $SCRIPT_DIR/predict.py .
    time mhcflurry-predict \
        benchmark.multiallelic.csv.bz2 \
        --allele-column hla \
        --prediction-column-prefix mhcflurry_production_ \
        --models "$(mhcflurry-downloads path models_class1_pan)/models.combined" \
        --affinity-only \
        --no-affinity-percentile \
        --out "$(pwd)/benchmark.multiallelic.predictions1.csv"

    for variant in no_additional_ms compact_peptide affinity_only no_pretrain single_hidden_no_pretrain
    do
        time mhcflurry-predict \
            "$(pwd)/benchmark.multiallelic.predictions1.csv" \
            --allele-column hla \
            --prediction-column-prefix "${variant}_" \
            --models "$(mhcflurry-downloads path models_class1_pan_variants)/models.$variant" \
            --affinity-only \
            --no-affinity-percentile \
            --out "$(pwd)/benchmark.multiallelic.predictions1.csv"
    done

    bzip2 -f benchmark.multiallelic.predictions1.csv
    rm -f benchmark.multiallelic.predictions2.csv.bz2
fi


### PRESENTATION: WITH FLANKS
if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.predictions2.csv.bz2" ]
then
    echo "Reusing existing multiallelic predictions2"
else
    time mhcflurry-predict \
        "$(pwd)/benchmark.multiallelic.predictions1.csv.bz2" \
        --allele-column hla \
        --prediction-column-prefix presentation_with_flanks_ \
        --models "$(mhcflurry-downloads path models_class1_presentation)/models" \
        --no-affinity-percentile \
        --out "$(pwd)/benchmark.multiallelic.predictions2.csv"

    bzip2 -f benchmark.multiallelic.predictions2.csv
    rm -f benchmark.multiallelic.predictions3.csv.bz2
fi

### PRESENTATION: NO FLANKS
if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.predictions3.csv.bz2" ]
then
    echo "Reusing existing multiallelic predictions3"
else
    time mhcflurry-predict \
        "$(pwd)/benchmark.multiallelic.predictions2.csv.bz2" \
        --allele-column hla \
        --prediction-column-prefix presentation_with_flanks_ \
        --models "$(mhcflurry-downloads path models_class1_presentation)/models" \
        --no-affinity-percentile \
        --no-flanking \
        --out "$(pwd)/benchmark.multiallelic.predictions3.csv"

    bzip2 -f benchmark.multiallelic.predictions3.csv
fi


cp $SCRIPT_ABSOLUTE_PATH .
bzip2 -f "$LOG"
for i in $(ls LOG-worker.*.txt) ; do bzip2 -f $i ; done
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
