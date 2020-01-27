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
export MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE=16384

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
        --exclude-train-data "$(mhcflurry-downloads path models_class1_pan_variants)/models.no_additional_ms/train_data.csv.bz2" \
        --only-format MONOALLELIC \
        --out "$(pwd)/benchmark.monoallelic.csv"
    bzip2 -f benchmark.monoallelic.csv
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
        --exclude-train-data "$(mhcflurry-downloads path models_class1_pan)/models.combined/train_data.csv.bz2" \
        --decoys-per-hit 99 \
        --only-format MULTIALLELIC \
        --out "$(pwd)/benchmark.multiallelic.csv"
    bzip2 -f benchmark.multiallelic.csv
fi

rm -rf commands
mkdir commands

### AFFINITY PREDICTOR VARIANT: MONOALLELIC
if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.monoallelic.predictions.csv.bz2" ]
then
    echo "Reusing existing monoallelic benchmark predictions"
else
    echo time mhcflurry-predict \
        "$(pwd)/benchmark.monoallelic.csv.bz2" \
        --allele-column hla \
        --prediction-column-prefix no_additional_ms_ \
        --models \""$(mhcflurry-downloads path models_class1_pan_variants)/models.no_additional_ms"\" \
        --affinity-only \
        --no-affinity-percentile \
        --out "$(pwd)/benchmark.monoallelic.predictions.csv" \
        --no-throw >> commands/monoallelic.sh
    echo bzip2 -f "$(pwd)/benchmark.monoallelic.predictions.csv" >> commands/monoallelic.sh
fi



### AFFINITY PREDICTORS: MULTIALLELIC
if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.production.csv.bz2" ]
then
    echo "Reusing existing multiallelic predictions"
else
    echo time mhcflurry-predict \
        "$(pwd)/benchmark.multiallelic.csv.bz2" \
        --allele-column hla \
        --prediction-column-prefix mhcflurry_production_ \
        --models \""$(mhcflurry-downloads path models_class1_pan)/models.combined"\" \
        --affinity-only \
        --no-affinity-percentile \
        --out "$(pwd)/benchmark.multiallelic.production.csv" >> commands/multiallelic.production.sh
    echo bzip2 -f "$(pwd)/benchmark.multiallelic.production.csv" >> commands/multiallelic.production.sh
fi

for variant in no_additional_ms compact_peptide affinity_only no_pretrain single_hidden_no_pretrain
do
    if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.${variant}.csv.bz2" ]
    then
        echo "Reusing existing multiallelic predictions: ${variant}"
    else
        echo time mhcflurry-predict \
            "$(pwd)/benchmark.multiallelic.csv.bz2" \
            --allele-column hla \
            --prediction-column-prefix "${variant}_" \
            --models \""$(mhcflurry-downloads path models_class1_pan_variants)/models.$variant"\" \
            --affinity-only \
            --no-affinity-percentile \
            --out "$(pwd)/benchmark.multiallelic.${variant}.csv" >> commands/multiallelic.${variant}.sh
        echo bzip2 -f "$(pwd)/benchmark.multiallelic.${variant}.csv" >> commands/multiallelic.${variant}.sh
    fi
done


### PRESENTATION: WITH FLANKS
if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.presentation_with_flanks.csv.bz2" ]
then
    echo "Reusing existing multiallelic presentation with flanks"
else
    echo time mhcflurry-predict \
        "$(pwd)/benchmark.multiallelic.csv.bz2" \
        --allele-column hla \
        --prediction-column-prefix presentation_with_flanks_ \
        --models \""$(mhcflurry-downloads path models_class1_presentation)/models"\" \
        --no-affinity-percentile \
        --out "$(pwd)/benchmark.multiallelic.presentation_with_flanks.csv" >> commands/multiallelic.presentation_with_flanks.sh
    echo bzip2 -f "$(pwd)/benchmark.multiallelic.presentation_with_flanks.csv"  >> commands/multiallelic.presentation_with_flanks.sh
fi

### PRESENTATION: NO FLANKS
if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.presentation_without_flanks.csv.bz2" ]
then
    echo "Reusing existing multiallelic presentation without flanks"
else
    echo time mhcflurry-predict \
        "$(pwd)/benchmark.multiallelic.csv.bz2" \
        --allele-column hla \
        --prediction-column-prefix presentation_without_flanks_ \
        --models \""$(mhcflurry-downloads path models_class1_presentation)/models"\" \
        --no-affinity-percentile \
        --no-flanking \
        --out "$(pwd)/benchmark.multiallelic.presentation_without_flanks.csv" >> commands/multiallelic.presentation_without_flanks.sh
    echo bzip2 -f "$(pwd)/benchmark.multiallelic.presentation_without_flanks.csv"  >> commands/multiallelic.presentation_without_flanks.sh
fi

### PRECOMPUTED ####
for variant in netmhcpan4.ba netmhcpan4.el mixmhcpred
do
    if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.monoallelic.${variant}.csv.bz2" ]
    then
        echo "Reusing existing monoallelic ${variant}"
    else
        cp $SCRIPT_DIR/join_with_precomputed.py .
        echo time python join_with_precomputed.py \
            \""$(pwd)/benchmark.monoallelic.csv.bz2"\" \
            ${variant} \
            --out "$(pwd)/benchmark.monoallelic.${variant}.csv" >> commands/monoallelic.${variant}.sh
        echo bzip2 -f "$(pwd)/benchmark.monoallelic.${variant}.csv"  >> commands/monoallelic.${variant}.sh
    fi

    if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.${variant}.csv.bz2" ]
    then
        echo "Reusing existing multiallelic ${variant}"
    else
        cp $SCRIPT_DIR/join_with_precomputed.py .
        echo time python join_with_precomputed.py \
            \""$(pwd)/benchmark.multiallelic.csv.bz2"\" \
            ${variant} \
            --out "$(pwd)/benchmark.multiallelic.${variant}.csv" >> commands/multiallelic.${variant}.sh
        echo bzip2 -f "$(pwd)/benchmark.multiallelic.${variant}.csv"  >> commands/multiallelic.${variant}.sh
    fi
done

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

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 -f "$LOG"
for i in $(ls LOG-worker.*.txt) ; do bzip2 -f $i ; done
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
