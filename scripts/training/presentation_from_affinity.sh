#!/usr/bin/env bash
#
# Train processing + presentation predictors against an *already trained*
# affinity ensemble (models.combined dir). Mirrors stages 2-3 of
# pan_allele_release_full.sh but skips stage 1, so it's cheap to run as
# a tail-on after a sweep that has already produced the affinity stack.
#
# Required env:
#   AFFINITY_PREDICTOR   path to existing models.combined dir
#   BASE_OUT             where to write processing/, presentation/
#   REPO                 mhcflurry repo root (for downloads-generation/)
#
# Optional env (all have defaults compatible with release_full):
#   GPUS, MAX_WORKERS_PER_GPU, NUM_JOBS, DATALOADER_NUM_WORKERS,
#   USE_ENCODING_CACHE, MATMUL_PRECISION, MHCFLURRY_TORCH_COMPILE,
#   PROCESSING_HELD_OUT_SAMPLES, PRESENTATION_DECOYS_PER_HIT
set -euo pipefail
set -x

: "${AFFINITY_PREDICTOR:?AFFINITY_PREDICTOR must be set}"
: "${BASE_OUT:?BASE_OUT must be set}"
: "${REPO:?REPO must be set}"

export PYTHONUNBUFFERED=1
export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-1}"

mkdir -p "$BASE_OUT/processing" "$BASE_OUT/presentation"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    GPUS=0
fi
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-auto}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-auto}"
USE_ENCODING_CACHE="${USE_ENCODING_CACHE:-1}"
PROCESSING_HELD_OUT_SAMPLES="${PROCESSING_HELD_OUT_SAMPLES:-50}"
PRESENTATION_DECOYS_PER_HIT="${PRESENTATION_DECOYS_PER_HIT:-99}"

if [ "$GPUS" -eq 0 ]; then
    NUM_JOBS=1
    MAX_WORKERS_PER_GPU=1
elif [ "$MAX_WORKERS_PER_GPU" = "auto" ]; then
    MAX_WORKERS_PER_GPU="$(
        GPUS="$GPUS" python - <<'PY'
import os
from mhcflurry.local_parallelism import auto_max_workers_per_gpu
print(auto_max_workers_per_gpu(
    num_jobs=0, num_gpus=int(os.environ["GPUS"]), backend="auto"))
PY
    )"
    NUM_JOBS="$(( GPUS * MAX_WORKERS_PER_GPU ))"
else
    NUM_JOBS="${NUM_JOBS:-$(( GPUS * MAX_WORKERS_PER_GPU ))}"
fi

COMMON_PARALLELISM_ARGS=(
    --num-jobs "$NUM_JOBS"
    --max-tasks-per-worker 1000
    --gpus "$GPUS"
    --max-workers-per-gpu "$MAX_WORKERS_PER_GPU"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --torch-compile auto
    --matmul-precision "${MATMUL_PRECISION:-none}"
)
[ "${MHCFLURRY_ENABLE_TIMING:-0}" = "1" ] && COMMON_PARALLELISM_ARGS+=(--enable-timing)

# ============================================================
# STAGE 2 — PROCESSING
# ============================================================
STAGE2_START=$(date +%s)
cd "$BASE_OUT/processing"

mhcflurry-downloads fetch data_mass_spec_annotated data_references

cp "$REPO/downloads-generation/models_class1_processing/annotate_hits_with_expression.py" .
cp "$REPO/downloads-generation/models_class1_processing/write_proteome_peptides.py" .
cp "$REPO/downloads-generation/models_class1_processing/make_train_data.py" make_train_data.processing.py
cp "$REPO/downloads-generation/models_class1_processing/generate_hyperparameters.base.py" .
cp "$REPO/downloads-generation/models_class1_processing/generate_hyperparameters.variants.py" .

python annotate_hits_with_expression.py \
    --hits "$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2" \
    --expression "$(mhcflurry-downloads path data_curated)/rna_expression.csv.bz2" \
    --out "$(pwd)/hits_with_tpm.csv"
bzip2 -f "$(pwd)/hits_with_tpm.csv"

python write_proteome_peptides.py \
    "$(mhcflurry-downloads path data_mass_spec_annotated)/annotated_ms.csv.bz2" \
    "$(mhcflurry-downloads path data_references)/uniprot_proteins.csv.bz2" \
    --out "$(pwd)/proteome_peptides.csv"
bzip2 -f "$(pwd)/proteome_peptides.csv"

python make_train_data.processing.py \
    --hits "$(pwd)/hits_with_tpm.csv.bz2" \
    --affinity-predictor "$AFFINITY_PREDICTOR" \
    --proteome-peptides "$(pwd)/proteome_peptides.csv.bz2" \
    --ppv-multiplier 100 \
    --hit-multiplier-to-take 2 \
    --out "$(pwd)/train_data.csv" \
    "${COMMON_PARALLELISM_ARGS[@]}"
bzip2 -f "$(pwd)/train_data.csv"

python generate_hyperparameters.base.py > hyperparameters.base.yaml

for kind in no_flank short_flanks; do
    python generate_hyperparameters.variants.py hyperparameters.base.yaml "$kind" \
        > "hyperparameters.$kind.full.yaml"
    python - "$kind" <<'PY'
import sys, yaml
kind = sys.argv[1]
hp = yaml.unsafe_load(open(f"hyperparameters.{kind}.full.yaml"))
def detuple(x):
    if isinstance(x, tuple): return list(x)
    if isinstance(x, list):  return [detuple(e) for e in x]
    if isinstance(x, dict):  return {k: detuple(v) for k, v in x.items()}
    return x
with open(f"hyperparameters.{kind}.yaml", "w") as f:
    yaml.safe_dump([detuple(d) for d in hp], f)
PY

    mhcflurry-class1-train-processing-models \
        --data "$(pwd)/train_data.csv.bz2" \
        --held-out-samples "$PROCESSING_HELD_OUT_SAMPLES" \
        --num-folds 4 \
        --hyperparameters "hyperparameters.$kind.yaml" \
        --out-models-dir "$(pwd)/models.unselected.$kind" \
        --worker-log-dir "$BASE_OUT/processing" \
        "${COMMON_PARALLELISM_ARGS[@]}"

    mhcflurry-class1-select-processing-models \
        --data "$(pwd)/models.unselected.$kind/train_data.csv.bz2" \
        --models-dir "$(pwd)/models.unselected.$kind" \
        --out-models-dir "$(pwd)/models.selected.$kind" \
        --min-models 1 \
        --max-models 2 \
        "${COMMON_PARALLELISM_ARGS[@]}"
    cp "$(pwd)/models.unselected.$kind/train_data.csv.bz2" \
        "$(pwd)/models.selected.$kind/train_data.csv.bz2"
done

echo "STAGE 2 duration: $(( $(date +%s) - STAGE2_START )) sec"

# ============================================================
# STAGE 3 — PRESENTATION
# ============================================================
STAGE3_START=$(date +%s)
cd "$BASE_OUT/presentation"

cp "$REPO/downloads-generation/models_class1_presentation/make_train_data.py" \
    make_train_data.presentation.py

python make_train_data.presentation.py \
    --hits "$BASE_OUT/processing/hits_with_tpm.csv.bz2" \
    --proteome-peptides "$BASE_OUT/processing/proteome_peptides.csv.bz2" \
    --decoys-per-hit "$PRESENTATION_DECOYS_PER_HIT" \
    --exclude-pmid 31844290 31495665 31154438 \
    --only-format MULTIALLELIC \
    --out "$(pwd)/train_data.csv"
bzip2 -f "$(pwd)/train_data.csv"

mhcflurry-class1-train-presentation-models \
    --data "$(pwd)/train_data.csv.bz2" \
    --affinity-predictor "$AFFINITY_PREDICTOR" \
    --processing-predictor-with-flanks "$BASE_OUT/processing/models.selected.short_flanks" \
    --processing-predictor-without-flanks "$BASE_OUT/processing/models.selected.no_flank" \
    --out-models-dir "$(pwd)/models"

mhcflurry-calibrate-percentile-ranks \
    --models-dir "$(pwd)/models" \
    --match-amino-acid-distribution-data "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    --alleles-file "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    --predictor-kind class1_presentation \
    --num-peptides-per-length 100000 \
    --alleles-per-genotype 1 \
    --num-genotypes 50 \
    --verbosity 1 \
    "${COMMON_PARALLELISM_ARGS[@]}"

cp "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    "$(pwd)/models/affinity_predictor_train_data.csv.bz2"
cp "$BASE_OUT/processing/models.selected.short_flanks/train_data.csv.bz2" \
    "$(pwd)/models/processing_predictor_with_flanks_train_data.csv.bz2"
cp "$BASE_OUT/processing/models.selected.no_flank/train_data.csv.bz2" \
    "$(pwd)/models/processing_predictor_no_flank_train_data.csv.bz2"

echo "STAGE 3 duration: $(( $(date +%s) - STAGE3_START )) sec"
echo "=== presentation predictor at $BASE_OUT/presentation/models ==="
