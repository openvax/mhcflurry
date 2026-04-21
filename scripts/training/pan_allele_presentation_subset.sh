#!/usr/bin/env bash
#
# Subset of the full release-exact pipeline: affinity → processing →
# presentation end-to-end with a small architecture sweep per variant.
# Produces a usable Class1PresentationPredictor for Phase D comparison
# against the public release, and gives us timing / cost signal for
# extrapolating the full-run budget.
#
# Default subset:
#   Affinity:    4 folds × 8 architectures = 32 networks
#   Processing:  4 folds × 8 architectures × 2 variants (no_flank,
#                short_flanks) = 64 networks
#   Presentation: 1 logistic-regression on top of the above (cheap)
#
# Environment:
#   MHCFLURRY_OUT              required — directory for all artifacts
#   SUBSET_ARCHS=N             architectures per variant (default 8)
#   REPO                       path to the rsynced mhcflurry repo (used
#                              to find downloads-generation recipe files)
#                              Defaults to $HOME/runplz-repo.
set -euo pipefail
set -x

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SUBSET_ARCHS="${SUBSET_ARCHS:-8}"
# Per-worker GPU memory during mhcflurry training + validation inference
# peaks at 16-22 GB (activation tensors, not weights). On A100-80GB, 2
# workers fit comfortably (~44 GB); 4 workers OOMs deterministically.
# On A100-40GB, 1 worker is the safe ceiling.
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-2}"
# Phase 1 (#268) encoding cache: pre-encode BLOSUM62 peptide vectors
# once and share across workers via mmap. Opt-in for safety; set
# USE_ENCODING_CACHE=0 to run the legacy path. Cache materializes under
# $MHCFLURRY_OUT/affinity/encoding_cache.
USE_ENCODING_CACHE="${USE_ENCODING_CACHE:-1}"
# DataLoader prefetch workers per training process. 0 = no prefetch
# (bit-identical to pre-#268). 4 overlaps CPU data-prep with GPU
# compute; each adds ~100-500 MB RSS.
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
REPO="${REPO:-$HOME/runplz-repo}"

# Inject dataloader_num_workers into hyperparameters for all training
# stages. Applied post-YAML-generation so we don't need to edit
# generate_hyperparameters.py upstream.
HYPERPARAMETER_INJECT=(
    "dataloader_num_workers=$DATALOADER_NUM_WORKERS"
)

# Extra flags for mhcflurry-class1-train-pan-allele-models to enable the
# encoding cache.
CACHE_ARGS=()
if [ "$USE_ENCODING_CACHE" = "1" ]; then
    CACHE_ARGS=(--use-encoding-cache)
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    GPUS=0
fi
echo "Detected GPUS: $GPUS"
if [ "$GPUS" -eq 0 ]; then
    NUM_JOBS=1
else
    NUM_JOBS="$(( GPUS * MAX_WORKERS_PER_GPU ))"
fi
PARALLELISM_ARGS="--num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu $MAX_WORKERS_PER_GPU"

mkdir -p "$MHCFLURRY_OUT/affinity" "$MHCFLURRY_OUT/processing" "$MHCFLURRY_OUT/presentation"

# ---- optional Nsight Systems profiling wrapper ----------------------
# Profile only stage 1 (affinity), for a bounded duration. Everything
# else runs bare so the 4-6 hr full run isn't slowed/ballooned.
#
# The pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel image installs nsys at
# /usr/local/cuda/bin/nsys but does NOT add that dir to the default PATH,
# so a plain `command -v nsys` misses it (this bit us on the 2026-04-21
# A100 subset run — profiling silently skipped). Probe both the default
# PATH and the canonical CUDA bin path before giving up.
NSYS_WRAP=()
if [ "${NSYS_PROFILE:-0}" = "1" ]; then
    NSYS_BIN=""
    if command -v nsys >/dev/null 2>&1; then
        NSYS_BIN="$(command -v nsys)"
    elif [ -x /usr/local/cuda/bin/nsys ]; then
        NSYS_BIN="/usr/local/cuda/bin/nsys"
    elif [ -x /opt/nvidia/nsight-systems/bin/nsys ]; then
        NSYS_BIN="/opt/nvidia/nsight-systems/bin/nsys"
    fi

    if [ -n "$NSYS_BIN" ]; then
        NSYS_WRAP=(
            "$NSYS_BIN" profile
            --duration=180        # 3 min capture window
            --trace=cuda,nvtx,osrt
            --follow-fork=true    # mhcflurry uses multiprocessing.Pool
            --sample=none         # cheaper; we care about GPU vs CPU not PC
            --output="$MHCFLURRY_OUT/nsys_profile_stage1_affinity"
            --force-overwrite=true
        )
        echo "NSYS_PROFILE=1: will wrap stage 1 affinity training with: ${NSYS_WRAP[*]}"
    else
        echo "WARNING: NSYS_PROFILE=1 but no nsys found in PATH, /usr/local/cuda/bin, or /opt/nvidia/nsight-systems/bin; continuing without profiling. Install with 'apt-get install -y nsight-systems-cli' on Ubuntu images."
    fi
fi

# ============================================================
# STAGE 1 — AFFINITY (Class1AffinityPredictor)
# ============================================================
echo "=== STAGE 1: AFFINITY ==="
STAGE1_START=$(date +%s)

cd "$MHCFLURRY_OUT/affinity"
mhcflurry-downloads fetch data_curated allele_sequences random_peptide_predictions

cp "$REPO/downloads-generation/models_class1_pan/reassign_mass_spec_training_data.py" .
cp "$REPO/downloads-generation/models_class1_pan/additional_alleles.txt" .
python reassign_mass_spec_training_data.py \
    "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" \
    --set-measurement-value 100 \
    --out-csv "$(pwd)/train_data.csv"
bzip2 -f "$(pwd)/train_data.csv"

cp "$REPO/downloads-generation/models_class1_pan/generate_hyperparameters.py" .
python generate_hyperparameters.py > hyperparameters.full.yaml
python - <<PY
import yaml
hp = yaml.safe_load(open("hyperparameters.full.yaml"))[:${SUBSET_ARCHS}]
# Inject per-run overrides (e.g. dataloader_num_workers from issue #268)
# without having to edit generate_hyperparameters.py upstream.
overrides = dict(kv.split("=", 1) for kv in """${HYPERPARAMETER_INJECT[@]}""".split() if "=" in kv)
for k, v in overrides.items():
    # Int-typed overrides (dataloader_num_workers) must parse as int;
    # fall back to string if not numeric.
    try:
        parsed = int(v)
    except ValueError:
        try:
            parsed = float(v)
        except ValueError:
            parsed = v
    for d in hp:
        d[k] = parsed
with open("hyperparameters.yaml", "w") as f:
    yaml.safe_dump(hp, f)
print(f"affinity: using {len(hp)} architectures; overrides={overrides}")
PY

"${NSYS_WRAP[@]}" mhcflurry-class1-train-pan-allele-models \
    --data "$(pwd)/train_data.csv.bz2" \
    --allele-sequences "$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv" \
    --pretrain-data "$(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2" \
    --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
    --num-folds 4 \
    --hyperparameters hyperparameters.yaml \
    --out-models-dir "$(pwd)/models.unselected" \
    --worker-log-dir "$MHCFLURRY_OUT/affinity" \
    "${CACHE_ARGS[@]}" \
    $PARALLELISM_ARGS

mhcflurry-class1-select-pan-allele-models \
    --data "$(pwd)/models.unselected/train_data.csv.bz2" \
    --models-dir "$(pwd)/models.unselected" \
    --out-models-dir "$(pwd)/models.combined" \
    --min-models 1 \
    --max-models 4 \
    $PARALLELISM_ARGS
cp "$(pwd)/models.unselected/train_data.csv.bz2" "$(pwd)/models.combined/train_data.csv.bz2"

AFFINITY_PREDICTOR="$MHCFLURRY_OUT/affinity/models.combined"
echo "affinity predictor at: $AFFINITY_PREDICTOR"
echo "STAGE 1 duration: $(( $(date +%s) - STAGE1_START )) sec"

# ============================================================
# STAGE 2 — PROCESSING (no_flank + short_flanks)
# Only those two variants are consumed by the presentation predictor.
# ============================================================
echo "=== STAGE 2: PROCESSING ==="
STAGE2_START=$(date +%s)

cd "$MHCFLURRY_OUT/processing"
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
    $PARALLELISM_ARGS
bzip2 -f "$(pwd)/train_data.csv"

python generate_hyperparameters.base.py > hyperparameters.base.yaml

for kind in no_flank short_flanks; do
    python generate_hyperparameters.variants.py hyperparameters.base.yaml $kind \
        > hyperparameters.$kind.full.yaml
    python - <<PY
import yaml
hp = yaml.unsafe_load(open("hyperparameters.$kind.full.yaml"))[:${SUBSET_ARCHS}]
# Avoid python/tuple tags on write so downstream readers can safe_load.
for d in hp:
    def _detuple(x):
        if isinstance(x, tuple): return list(x)
        if isinstance(x, list):  return [_detuple(e) for e in x]
        if isinstance(x, dict):  return {k: _detuple(v) for k, v in x.items()}
        return x
    hp_list_safe = [_detuple(d) for d in hp]
with open("hyperparameters.$kind.yaml", "w") as f:
    yaml.safe_dump([_detuple(d) for d in hp], f)
print(f"processing.$kind: using {len(hp)} architectures")
PY

    mhcflurry-class1-train-processing-models \
        --data "$(pwd)/train_data.csv.bz2" \
        --held-out-samples 10 \
        --num-folds 4 \
        --hyperparameters hyperparameters.$kind.yaml \
        --out-models-dir "$(pwd)/models.unselected.$kind" \
        --worker-log-dir "$MHCFLURRY_OUT/processing" \
        $PARALLELISM_ARGS

    mhcflurry-class1-select-processing-models \
        --data "$(pwd)/models.unselected.$kind/train_data.csv.bz2" \
        --models-dir "$(pwd)/models.unselected.$kind" \
        --out-models-dir "$(pwd)/models.selected.$kind" \
        --min-models 1 \
        --max-models 2 \
        $PARALLELISM_ARGS
    cp "$(pwd)/models.unselected.$kind/train_data.csv.bz2" \
        "$(pwd)/models.selected.$kind/train_data.csv.bz2"
done

echo "STAGE 2 duration: $(( $(date +%s) - STAGE2_START )) sec"

# ============================================================
# STAGE 3 — PRESENTATION (Class1PresentationPredictor)
# ============================================================
echo "=== STAGE 3: PRESENTATION ==="
STAGE3_START=$(date +%s)

cd "$MHCFLURRY_OUT/presentation"
cp "$REPO/downloads-generation/models_class1_presentation/make_train_data.py" \
    make_train_data.presentation.py

python make_train_data.presentation.py \
    --hits "$MHCFLURRY_OUT/processing/hits_with_tpm.csv.bz2" \
    --proteome-peptides "$MHCFLURRY_OUT/processing/proteome_peptides.csv.bz2" \
    --decoys-per-hit 2 \
    --exclude-pmid 31844290 31495665 31154438 \
    --only-format MULTIALLELIC \
    --sample-fraction 0.1 \
    --out "$(pwd)/train_data.csv"
bzip2 -f "$(pwd)/train_data.csv"

time mhcflurry-class1-train-presentation-models \
    --data "$(pwd)/train_data.csv.bz2" \
    --affinity-predictor "$AFFINITY_PREDICTOR" \
    --processing-predictor-with-flanks "$MHCFLURRY_OUT/processing/models.selected.short_flanks" \
    --processing-predictor-without-flanks "$MHCFLURRY_OUT/processing/models.selected.no_flank" \
    --out-models-dir "$(pwd)/models"

time mhcflurry-calibrate-percentile-ranks \
    --models-dir "$(pwd)/models" \
    --match-amino-acid-distribution-data "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    --alleles-file "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    --predictor-kind class1_presentation \
    --num-peptides-per-length 10000 \
    --alleles-per-genotype 1 \
    --num-genotypes 50 \
    --verbosity 1 \
    $PARALLELISM_ARGS

# Convenience: copy training data into the final bundle like the
# release does.
cp "$AFFINITY_PREDICTOR/train_data.csv.bz2" \
    "$(pwd)/models/affinity_predictor_train_data.csv.bz2"
cp "$MHCFLURRY_OUT/processing/models.selected.short_flanks/train_data.csv.bz2" \
    "$(pwd)/models/processing_predictor_with_flanks_train_data.csv.bz2"
cp "$MHCFLURRY_OUT/processing/models.selected.no_flank/train_data.csv.bz2" \
    "$(pwd)/models/processing_predictor_no_flank_train_data.csv.bz2"

echo "STAGE 3 duration: $(( $(date +%s) - STAGE3_START )) sec"

echo "=== DONE ==="
echo "Final presentation predictor: $MHCFLURRY_OUT/presentation/models/"
ls -la "$MHCFLURRY_OUT/presentation/models" | head -20
