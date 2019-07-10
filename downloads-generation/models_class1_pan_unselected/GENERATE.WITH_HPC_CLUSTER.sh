#!/bin/bash
#
# Train pan-allele MHCflurry Class I models.
#
# Uses an HPC cluster (Mount Sinai chimera cluster, which uses lsf job
# scheduler). This would need to be modified for other sites.
#
set -e
set -x

DOWNLOAD_NAME=models_class1_pan_unselected
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR=$(dirname "$SCRIPT_ABSOLUTE_PATH")

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

cp $SCRIPT_DIR/generate_hyperparameters.py .
python generate_hyperparameters.py > hyperparameters.yaml

for kind in with_mass_spec no_mass_spec
do
    mhcflurry-class1-train-pan-allele-models \
        --data "$(mhcflurry-downloads path data_curated)/curated_training_data.${kind}.csv.bz2" \
        --allele-sequences "$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv" \
        --pretrain-data "$(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2" \
        --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
        --ensemble-size 4 \
        --hyperparameters hyperparameters.yaml \
        --out-models-dir models.${kind} \
        --worker-log-dir "$SCRATCH_DIR/$DOWNLOAD_NAME" \
        --verbosity 1 \
        --cluster-parallelism \
        --cluster-submit-command bsub \
        --cluster-results-workdir ~/mhcflurry-model-training \
        --cluster-script-prefix-path ./cluster_submit_script_header.mssm_hpc.lsf
done

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
for i in $(ls LOG-worker.*.txt) ; do bzip2 $i ; done
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *
echo "Created archive: $SCRATCH_DIR/${DOWNLOAD_NAME}.tar.bz2"
