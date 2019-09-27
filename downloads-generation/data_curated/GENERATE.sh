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
cp $SCRIPT_DIR/curate_by_pmid.py .

RAW_DIR="$(mhcflurry-downloads path data_published)/raw"
cp -r "$RAW_DIR" .

CURATE_BY_PMID_ARGS=""
for pmid in $(ls raw)
do
    CURATE_BY_PMID_ARGS+=$(echo --item $pmid raw/$pmid/* ' ')
done

time python curate_by_pmid.py $CURATE_BY_PMID_ARGS \
    --out nontraining_curated.by_pmid.csv

bzip2 nontraining_curated.by_pmid.csv

rm -rf raw

# No mass-spec data
time python curate.py \
    --data-iedb \
        "$(mhcflurry-downloads path data_iedb)/mhc_ligand_full.csv.bz2" \
    --data-kim2014 \
        "$(mhcflurry-downloads path data_published)/bdata.20130222.mhci.public.1.txt" \
    --out-csv curated_training_data.no_mass_spec.csv

# With mass-spec data
time python curate.py \
    --data-iedb \
        "$(mhcflurry-downloads path data_iedb)/mhc_ligand_full.csv.bz2" \
    --data-kim2014 \
        "$(mhcflurry-downloads path data_published)/bdata.20130222.mhci.public.1.txt" \
    --data-systemhc-atlas \
        "$(mhcflurry-downloads path data_systemhcatlas)/data.csv.bz2" \
    --include-iedb-mass-spec \
    --out-csv curated_training_data.with_mass_spec.csv

bzip2 curated_training_data.no_mass_spec.csv
bzip2 curated_training_data.with_mass_spec.csv

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"
