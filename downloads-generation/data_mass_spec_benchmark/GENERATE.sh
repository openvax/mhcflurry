#!/bin/bash
#
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
#exec >  >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt")
#exec 2> >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt" >&2)

# Log some environment info
date
pip freeze
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

cp $SCRIPT_DIR/write_proteome_peptides.py .
cp $SCRIPT_DIR/run_mhcflurry.py .
cp $SCRIPT_DIR/write_allele_list.py .

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


PEPTIDES=$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2
REFERENCES_DIR=$(mhcflurry-downloads path data_references)

#python write_proteome_peptides.py \
#    "$PEPTIDES" \
#    "${REFERENCES_DIR}/uniprot_proteins.csv.bz2" \
#    --out proteome_peptides.csv
#ls -lh proteome_peptides.csv
#bzip2 proteome_peptides.csv
ln -s ~/Dropbox/sinai/projects/201808-mhcflurry-pan/20190622-models/proteome_peptides.csv.bz2 proteome_peptides.csv.bz2

python write_allele_list.py "$PEPTIDES" --out alleles.txt

mkdir predictions

for kind in with_mass_spec no_mass_spec
do
    python run_mhcflurry.py \
        proteome_peptides.csv.bz2 \
        --chunk-size 10000000 \
        --models-dir "$(mhcflurry-downloads path models_class1_pan)/models.$kind" \
        --allele $(cat alleles.txt) \
        --out "predictions/mhcflurry.$kind" \
        --num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1
done

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
