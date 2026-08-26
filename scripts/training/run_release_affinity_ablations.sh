#!/usr/bin/env bash
# Run the paired affinity hyperparameter panel on the frozen release holdout.
set -euo pipefail

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/../.." && pwd)}"

PANEL_DIR="$MHCFLURRY_OUT/hyperparameter_panels"
HOLDOUT_DIR="$MHCFLURRY_OUT/release_holdout"
mkdir -p "$PANEL_DIR" "$HOLDOUT_DIR"

python "$SCRIPT_DIR/generate_release_hyperparameter_ablations.py" "$PANEL_DIR" \
    > "$MHCFLURRY_OUT/panel_manifest.stdout.json"

mhcflurry-downloads fetch data_evaluation data_curated data_mass_spec_annotated
mhcflurry train release-holdout build \
    --data-dir "$(mhcflurry-downloads path data_evaluation)" \
    --training-data "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" \
    --mass-spec-data "$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2" \
    --out-dir "$HOLDOUT_DIR"

for condition in \
    published_parity \
    proposed_release \
    pre_activation_lsuv \
    no_lsuv \
    pytorch_rmsprop
do
    condition_out="$MHCFLURRY_OUT/affinity.$condition"
    env \
        MHCFLURRY_OUT="$condition_out" \
        REPO="$REPO" \
        RELEASE_HOLDOUT_DIR="$HOLDOUT_DIR" \
        AFFINITY_HYPERPARAMETERS_FILE="$PANEL_DIR/affinity.$condition.yaml" \
        SKIP_CALIBRATE=1 \
        SKIP_EVAL=1 \
        SKIP_PLOTS=0 \
        bash "$SCRIPT_DIR/pan_allele_release_affinity.sh"
done

DATA_EVAL_DIR="$(mhcflurry-downloads path data_evaluation)"
PAIRWISE_DIR="$MHCFLURRY_OUT/paired_comparisons"
mkdir -p "$PAIRWISE_DIR"
for condition in \
    proposed_release \
    pre_activation_lsuv \
    no_lsuv \
    pytorch_rmsprop
do
    mhcflurry eval compare-models \
        --a "$MHCFLURRY_OUT/affinity.$condition/models.combined" \
        --a-label "$condition" \
        --b "$MHCFLURRY_OUT/affinity.published_parity/models.combined" \
        --b-label published_parity \
        --data-dir "$DATA_EVAL_DIR" \
        --release-holdout-dir "$HOLDOUT_DIR" \
        --affinity-training-overlap-policy audit \
        --include affinity \
        --out "$PAIRWISE_DIR/$condition-vs-published_parity"
done
