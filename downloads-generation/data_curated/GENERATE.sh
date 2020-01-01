#!/bin/bash
#
# Create "curated" training data, which combines an IEDB download with additional
# published data, removes unusable entries, normalizes allele name, and performs
# other filtering and standardization.
#
set -e
set -x

DOWNLOAD_NAME=data_curated
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

cp $SCRIPT_DIR/curate.py .
cp $SCRIPT_DIR/curate_ms_by_pmid.py .

MS_DIR="$(mhcflurry-downloads path data_published)/ms"
cp -r "$MS_DIR" .

EXPRESSION_DIR="$(mhcflurry-downloads path data_published)/expression"
cp -r "$EXPRESSION_DIR" .

CURATE_BY_PMID_ARGS=""
for pmid in $(ls ms)
do
    CURATE_BY_PMID_ARGS+=$(echo --ms-item $pmid ms/$pmid/* ' ')
done
for item in $(ls expression)
do
    CURATE_BY_PMID_ARGS+=$(echo --expression-item $item expression/$item/* ' ')
done

time python curate_ms_by_pmid.py $CURATE_BY_PMID_ARGS \
    --ms-out ms.by_pmid.csv \
    --expression-out rna_expression.csv \
    --expression-metadata-out rna_expression.metadata.csv

bzip2 ms.by_pmid.csv
bzip2 rna_expression.csv

rm -rf ms

time python curate.py \
    --data-iedb \
        "$(mhcflurry-downloads path data_iedb)/mhc_ligand_full.csv.bz2" \
    --data-kim2014 \
        "$(mhcflurry-downloads path data_published)/bdata.20130222.mhci.public.1.txt" \
    --data-systemhc-atlas \
        "$(mhcflurry-downloads path data_systemhcatlas)/data.csv.bz2" \
    --data-additional-ms "$(pwd)/ms.by_pmid.csv.bz2" \
    --out-csv curated_training_data.csv \
    --out-affinity-csv curated_training_data.affinity.csv \
    --out-mass-spec-csv curated_training_data.mass_spec.csv

time python curate.py \
    --data-iedb \
        "$(mhcflurry-downloads path data_iedb)/mhc_ligand_full.csv.bz2" \
    --data-kim2014 \
        "$(mhcflurry-downloads path data_published)/bdata.20130222.mhci.public.1.txt" \
    --data-systemhc-atlas \
        "$(mhcflurry-downloads path data_systemhcatlas)/data.csv.bz2" \
    --out-csv curated_training_data.no_additional_ms.csv

for i in $(ls *.csv)
do
    bzip2 $i
done

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
