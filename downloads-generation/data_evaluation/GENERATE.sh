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

## GENERATE BENCHMARK: MONOALLELIC
#for kind in train_excluded all
for kind in train_excluded
do
    EXCLUDE_TRAIN_DATA=""
    if [ "$kind" == "train_excluded" ]
    then
        EXCLUDE_TRAIN_DATA="$(mhcflurry-downloads path models_class1_pan_variants)/models.no_additional_ms/train_data.csv.bz2"
    fi

    if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.monoallelic.$kind.csv.bz2" ]
    then
        echo "Reusing existing monoallelic benchmark: benchmark.monoallelic.$kind.csv.bz2"
    else
        cp $SCRIPT_DIR/make_benchmark.py .
        time python make_benchmark.py \
            --hits "$(mhcflurry-downloads path models_class1_processing)/hits_with_tpm.csv.bz2" \
            --proteome-peptides "$(mhcflurry-downloads path models_class1_processing)/proteome_peptides.csv.bz2" \
            --decoys-per-hit 110 \
            --exclude-train-data "$EXCLUDE_TRAIN_DATA" \
            --only-format MONOALLELIC \
            --out "$(pwd)/benchmark.monoallelic.$kind.csv"
        bzip2 -f benchmark.monoallelic.$kind.csv
    fi
done

### GENERATE BENCHMARK: MULTIALLELIC
#for kind in train_excluded all
for kind in train_excluded
do
    EXCLUDE_TRAIN_DATA=""
    if [ "$kind" == "train_excluded" ]
    then
        EXCLUDE_TRAIN_DATA="$(mhcflurry-downloads path models_class1_pan)/models.combined/train_data.csv.bz2"
    fi

    if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.$kind.csv.bz2" ]
    then
        echo "Reusing existing multiallelic benchmark"
    else
        cp $SCRIPT_DIR/make_benchmark.py .
        time python make_benchmark.py \
            --hits "$(mhcflurry-downloads path models_class1_processing)/hits_with_tpm.csv.bz2" \
            --proteome-peptides "$(mhcflurry-downloads path models_class1_processing)/proteome_peptides.csv.bz2" \
            --decoys-per-hit 110 \
            --exclude-train-data "$EXCLUDE_TRAIN_DATA" \
            --only-format MULTIALLELIC \
            --out "$(pwd)/benchmark.multiallelic.$kind.csv"
        bzip2 -f benchmark.multiallelic.$kind.csv
    fi
done

for kind in train_excluded
do
    ### SPLIT BENCHMARK: MONOALLELIC
    if [ "$2" == "continue-incomplete" ] && [ -f "MONOALLELIC_SAMPLES" ]
    then
        echo "Reusing existing monoallelic $kind benchmark pieces"
    else
        cp $SCRIPT_DIR/split_by_sample.py .
        time python split_by_sample.py \
            "$(pwd)/benchmark.monoallelic.$kind.csv.bz2" \
            --out "$(pwd)/benchmark.monoallelic.$kind.%s.csv" \
            --out-samples MONOALLELIC_SAMPLES

        for sample in $(cat MONOALLELIC_SAMPLES)
        do
            bzip2 -f "$(pwd)/benchmark.monoallelic.$kind.$sample.csv"
        done
    fi

    ### SPLIT BENCHMARK: MULTIALLELIC
    if [ "$2" == "continue-incomplete" ] && [ -f "MULTIALLELIC_SAMPLES" ]
    then
        echo "Reusing existing multiallelic $kind benchmark pieces"
    else
        cp $SCRIPT_DIR/split_by_sample.py .
        time python split_by_sample.py \
            "$(pwd)/benchmark.multiallelic.$kind.csv.bz2" \
            --out "$(pwd)/benchmark.multiallelic.$kind.%s.csv" \
            --out-samples MULTIALLELIC_SAMPLES

        for sample in $(cat MULTIALLELIC_SAMPLES)
        do
            bzip2 -f "$(pwd)/benchmark.multiallelic.$kind.$sample.csv"
        done
    fi
done

rm -rf commands
mkdir commands

#for kind in train_excluded all
for kind in train_excluded
do
    MONOALLELIC_AFFINITY_PREDICTOR="$(mhcflurry-downloads path models_class1_pan_variants)/models.no_additional_ms"
    echo "MONOALLELIC: Using affinity predictor: MONOALLELIC_AFFINITY_PREDICTOR"
    cat "$MONOALLELIC_AFFINITY_PREDICTOR/info.txt"

    for sample in $(cat MONOALLELIC_SAMPLES)
    do
        ## AFFINITY PREDICTOR VARIANT: MONOALLELIC
        if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.monoallelic.no_additional_ms.$kind.$sample.csv.bz2" ]
        then
            echo "Reusing existing monoallelic benchmark predictions $sample"
        else
            echo time mhcflurry-predict \
                "$(pwd)/benchmark.monoallelic.$kind.$sample.csv.bz2" \
                --allele-column hla \
                --prediction-column-prefix no_additional_ms_ \
                --models \""$MONOALLELIC_AFFINITY_PREDICTOR"\" \
                --affinity-only \
                --no-affinity-percentile \
                --out "$(pwd)/benchmark.monoallelic.no_additional_ms.$kind.$sample.csv" \
                --no-throw >> commands/monoallelic.$kind.$sample.sh
            echo bzip2 -f "$(pwd)/benchmark.monoallelic.no_additional_ms.$kind.$sample.csv" >> commands/monoallelic.$kind.$sample.sh
        fi
    done

    MULTIALLELIC_AFFINITY_PREDICTOR="$(mhcflurry-downloads path models_class1_pan)/models.combined"
    echo "MULTIALLELIC: Using affinity predictor: MULTIALLELIC_AFFINITY_PREDICTOR"
    cat "$MULTIALLELIC_AFFINITY_PREDICTOR/info.txt"

    for sample in $(cat MULTIALLELIC_SAMPLES)
    do
        ### AFFINITY PREDICTORS: MULTIALLELIC
        if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.production.$kind.$sample.csv.bz2" ]
        then
            echo "Reusing existing multiallelic predictions $sample"
        else
            echo time mhcflurry-predict \
                "$(pwd)/benchmark.multiallelic.$kind.$sample.csv.bz2" \
                --allele-column hla \
                --prediction-column-prefix mhcflurry_production_ \
                --models \""$MULTIALLELIC_AFFINITY_PREDICTOR"\" \
                --affinity-only \
                --no-affinity-percentile \
                --out "$(pwd)/benchmark.multiallelic.production.$kind.$sample.csv" >> commands/multiallelic.production.$kind.$sample.sh
            echo bzip2 -f "$(pwd)/benchmark.multiallelic.production.$kind.$sample.csv" >> commands/multiallelic.production.$kind.$sample.sh
        fi
    done

    #for sample in $(cat MULTIALLELIC_SAMPLES)
    #do
        #for variant in no_additional_ms compact_peptide affinity_only no_pretrain single_hidden_no_pretrain 500nm
        #for variant in 50nm
        #do
        #    if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.${variant}.$kind.$sample.csv.bz2" ]
        #    then
        #        echo "Reusing existing multiallelic predictions: ${variant} $sample"
        #    else
        #        echo time mhcflurry-predict \
        #            "$(pwd)/benchmark.multiallelic.$kind.$sample.csv.bz2" \
        #            --allele-column hla \
        #            --prediction-column-prefix "${variant}_" \
        #            --models \""$(mhcflurry-downloads path models_class1_pan_variants)/models.$variant"\" \
        #            --affinity-only \
        #            --no-affinity-percentile \
        #            --out "$(pwd)/benchmark.multiallelic.${variant}.$kind.$sample.csv" >> commands/multiallelic.${variant}.$kind.$sample.sh
        #        echo bzip2 -f "$(pwd)/benchmark.multiallelic.${variant}.$kind.$sample.csv" >> commands/multiallelic.${variant}.$kind.$sample.sh
        #    fi
        #done
    #done

    PRESENTATION_PREDICTOR="$(mhcflurry-downloads path models_class1_presentation)/models"
    echo "Using presentation predictor: $PRESENTATION_PREDICTOR"
    cat "$PRESENTATION_PREDICTOR/info.txt"

    for sample in $(cat MULTIALLELIC_SAMPLES)
    do
        ### PRESENTATION: WITH FLANKS
        if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.presentation_with_flanks.$kind.$sample.csv.bz2" ]
        then
            echo "Reusing existing multiallelic presentation with flanks $sample"
        else
            echo time mhcflurry-predict \
                "$(pwd)/benchmark.multiallelic.$kind.$sample.csv.bz2" \
                --allele-column hla \
                --prediction-column-prefix presentation_with_flanks_ \
                --models \""$PRESENTATION_PREDICTOR"\" \
                --no-affinity-percentile \
                --out "$(pwd)/benchmark.multiallelic.presentation_with_flanks.$kind.$sample.csv" >> commands/multiallelic.presentation_with_flanks.$kind.$sample.sh
            echo bzip2 -f "$(pwd)/benchmark.multiallelic.presentation_with_flanks.$kind.$sample.csv"  >> commands/multiallelic.presentation_with_flanks.$kind.$sample.sh
        fi
    done

    for sample in $(cat MULTIALLELIC_SAMPLES)
    do
        ### PRESENTATION: NO FLANKS
        if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.presentation_without_flanks.$kind.$sample.csv.bz2" ]
        then
            echo "Reusing existing multiallelic presentation without flanks $sample"
        else
            echo time mhcflurry-predict \
                "$(pwd)/benchmark.multiallelic.$kind.$sample.csv.bz2" \
                --allele-column hla \
                --prediction-column-prefix presentation_without_flanks_ \
                --models \""$PRESENTATION_PREDICTOR"\" \
                --no-affinity-percentile \
                --no-flanking \
                --out "$(pwd)/benchmark.multiallelic.presentation_without_flanks.$kind.$sample.csv" >> commands/multiallelic.presentation_without_flanks.$kind.$sample.sh
            echo bzip2 -f "$(pwd)/benchmark.multiallelic.presentation_without_flanks.$kind.$sample.csv"  >> commands/multiallelic.presentation_without_flanks.$kind.$sample.sh
        fi
    done

    ### PRECOMPUTED ####
    for variant in netmhcpan4.ba netmhcpan4.el mixmhcpred
    do
        for sample in $(cat MONOALLELIC_SAMPLES)
        do
            if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.monoallelic.${variant}.$kind.$sample.csv.bz2" ]
            then
                echo "Reusing existing monoallelic ${variant} $sample"
            else
                cp $SCRIPT_DIR/join_with_precomputed.py .
                echo time python join_with_precomputed.py \
                    \""$(pwd)/benchmark.monoallelic.$kind.$sample.csv.bz2"\" \
                    ${variant} \
                    --out "$(pwd)/benchmark.monoallelic.${variant}.$kind.$sample.csv" >> commands/monoallelic.${variant}.$kind.$sample.sh
                echo bzip2 -f "$(pwd)/benchmark.monoallelic.${variant}.$kind.$sample.csv"  >> commands/monoallelic.${variant}.$kind.$sample.sh
            fi
        done

        for sample in $(cat MULTIALLELIC_SAMPLES)
        do
            if [ "$2" == "continue-incomplete" ] && [ -f "benchmark.multiallelic.${variant}.$kind.$sample.csv.bz2" ]
            then
                echo "Reusing existing multiallelic ${variant} $sample"
            else
                cp $SCRIPT_DIR/join_with_precomputed.py .
                echo time python join_with_precomputed.py \
                    \""$(pwd)/benchmark.multiallelic.$kind.$sample.csv.bz2"\" \
                    ${variant} \
                    --out "$(pwd)/benchmark.multiallelic.${variant}.$kind.$sample.csv" >> commands/multiallelic.${variant}.$kind.$sample.sh
                echo bzip2 -f "$(pwd)/benchmark.multiallelic.${variant}.$kind.$sample.csv"  >> commands/multiallelic.${variant}.$kind.$sample.sh
            fi
        done
    done
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
        sleep 0.2
    done
    wait
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
split -b 2000m "$RESULT" "$PARTS"
echo "Split into parts:"
ls -lh "${PARTS}"*