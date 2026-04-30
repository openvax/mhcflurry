#!/usr/bin/env bash
#
# Full mhcflurry release pipeline: affinity → processing → presentation,
# end-to-end at full architecture-sweep size.
#
# Composition:
#   - Stage 1 (affinity) delegates to pan_allele_release_affinity.sh,
#     which carries the heartbeat / write_snapshot / log_release_event
#     instrumentation, --continue-incomplete resume logic, and the
#     calibrate + eval-against-public-release phases.
#   - Stages 2-3 (processing + presentation) are inlined here, modeled
#     on pan_allele_presentation_subset.sh but without the SUBSET_ARCHS
#     truncation so the full architecture sweep trains.
#
# Each stage writes to its own subdirectory under MHCFLURRY_OUT so
# artifacts don't collide (affinity/, processing/, presentation/). The
# downstream eval step in stage 3 uses all three predictors together.
#
# Resumption: re-running this script reuses any models that the affinity
# stage already trained (via --continue-incomplete inside that stage).
# Stages 2-3 are not yet incremental — they re-run from scratch each
# time. The dominant wall-time is in stage 1, so this is OK in practice.
#
# Env (caller-tunable; all have sensible defaults):
#   MHCFLURRY_OUT              required — root for all artifacts
#   REPO                       path to the rsynced mhcflurry repo
#                              (default: $HOME/runplz-repo)
#   MAX_WORKERS_PER_GPU        per-GPU worker cap (default 2 on 80GB cards)
#   USE_ENCODING_CACHE         enable Phase-1 mmap encoding cache (default 1)
#   DATALOADER_NUM_WORKERS     'auto' (default) lets the orchestrator pick
#   PROCESSING_HELD_OUT_SAMPLES  (default 50; subset script uses 10)
#   PRESENTATION_DECOYS_PER_HIT (default 99 to match release; subset uses 2)
set -euo pipefail
set -x

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
: "${REPO:=$HOME/runplz-repo}"

export PYTHONUNBUFFERED=1
# Same default as the affinity stage; the orchestrator's CLI flag
# (--torch-compile auto) reads this when set.
export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-1}"

BASE_OUT="$MHCFLURRY_OUT"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$BASE_OUT/affinity" "$BASE_OUT/processing" "$BASE_OUT/presentation"

# Detect GPU count once; reuse for all stages.
if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    GPUS=0
fi
# Default to auto so the new (2.3.0) full release exercises the
# orchestrator's hardware-aware resolver. On 8x80GB this lands at 4
# workers/GPU = 32 fit workers (vs the affinity-only release_exact's
# pinned 2/GPU = 16, which preserves bit-for-bit replication of 2.2.0).
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-auto}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-auto}"
USE_ENCODING_CACHE="${USE_ENCODING_CACHE:-1}"
PROCESSING_HELD_OUT_SAMPLES="${PROCESSING_HELD_OUT_SAMPLES:-50}"
PRESENTATION_DECOYS_PER_HIT="${PRESENTATION_DECOYS_PER_HIT:-99}"

if [ "$GPUS" -eq 0 ]; then
    NUM_JOBS=1
    MAX_WORKERS_PER_GPU=1
elif [ "$MAX_WORKERS_PER_GPU" = "auto" ]; then
    # Pre-resolve via the orchestrator's helper so the rest of the
    # script (set_cpu_threads helper, COMMON_PARALLELISM_ARGS, log
    # banners) can use a numeric value. Pass num_jobs=0 to skip the
    # by_jobs clamp inside auto_max_workers_per_gpu — we want the
    # resolver to pick on VRAM + hard_cap alone, then derive num_jobs
    # from the picked MWPG.
    MAX_WORKERS_PER_GPU="$(
        GPUS="$GPUS" python - <<'PY'
import os
from mhcflurry.local_parallelism import auto_max_workers_per_gpu
print(auto_max_workers_per_gpu(
    num_jobs=0,
    num_gpus=int(os.environ["GPUS"]),
    backend="auto",
))
PY
    )"
    NUM_JOBS="$(( GPUS * MAX_WORKERS_PER_GPU ))"
    echo "Resolved MAX_WORKERS_PER_GPU=auto to $MAX_WORKERS_PER_GPU; NUM_JOBS=$NUM_JOBS"
else
    NUM_JOBS="${NUM_JOBS:-$(( GPUS * MAX_WORKERS_PER_GPU ))}"
fi

# Same parallelism args as stage 1; processing/presentation stages
# below pick this up. --torch-compile auto reads MHCFLURRY_TORCH_COMPILE
# env (set above), so the env path and the CLI path produce identical
# orchestrator state.
COMMON_PARALLELISM_ARGS=(
    --num-jobs "$NUM_JOBS"
    --max-tasks-per-worker 1000
    --gpus "$GPUS"
    --max-workers-per-gpu "$MAX_WORKERS_PER_GPU"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --torch-compile auto
    --matmul-precision "${MATMUL_PRECISION:-none}"
)
if [ "${MHCFLURRY_ENABLE_TIMING:-0}" = "1" ]; then
    COMMON_PARALLELISM_ARGS+=(--enable-timing)
fi

CACHE_ARGS=()
if [ "$USE_ENCODING_CACHE" = "1" ]; then
    CACHE_ARGS=(--use-encoding-cache)
fi

# ============================================================
# STAGE 1 — AFFINITY
# ============================================================
echo "=== STAGE 1: AFFINITY ==="
STAGE1_START=$(date +%s)
MHCFLURRY_OUT="$BASE_OUT/affinity" \
    NUM_JOBS="$NUM_JOBS" \
    GPUS="$GPUS" \
    MAX_WORKERS_PER_GPU="$MAX_WORKERS_PER_GPU" \
    DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" \
    USE_ENCODING_CACHE="$USE_ENCODING_CACHE" \
    bash "$SCRIPT_DIR/pan_allele_release_affinity.sh"
AFFINITY_PREDICTOR="$BASE_OUT/affinity/models.combined"
echo "STAGE 1 duration: $(( $(date +%s) - STAGE1_START )) sec"
echo "affinity predictor: $AFFINITY_PREDICTOR"

# ============================================================
# STAGE 2 — PROCESSING (no_flank + short_flanks variants)
# Both variants are inputs to the presentation predictor.
# ============================================================
echo "=== STAGE 2: PROCESSING ==="
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
    # Round-trip through safe_dump to strip python/tuple tags so
    # downstream readers can use yaml.safe_load.
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
print(f"processing.{kind}: using {len(hp)} architectures")
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
# STAGE 3 — PRESENTATION (Class1PresentationPredictor)
# ============================================================
echo "=== STAGE 3: PRESENTATION ==="
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

# Bundle training-data CSVs into the final presentation predictor dir
# so it's self-contained for distribution.
cp "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    "$(pwd)/models/affinity_predictor_train_data.csv.bz2"
cp "$BASE_OUT/processing/models.selected.short_flanks/train_data.csv.bz2" \
    "$(pwd)/models/processing_predictor_with_flanks_train_data.csv.bz2"
cp "$BASE_OUT/processing/models.selected.no_flank/train_data.csv.bz2" \
    "$(pwd)/models/processing_predictor_no_flank_train_data.csv.bz2"

echo "STAGE 3 duration: $(( $(date +%s) - STAGE3_START )) sec"

echo "=== DONE ==="
echo "affinity:     $AFFINITY_PREDICTOR"
echo "processing:   $BASE_OUT/processing/models.selected.{no_flank,short_flanks}"
echo "presentation: $BASE_OUT/presentation/models"
ls -la "$BASE_OUT/presentation/models" | head -20
