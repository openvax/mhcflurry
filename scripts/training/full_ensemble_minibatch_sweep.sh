#!/usr/bin/env bash
#
# Full-ensemble minibatch-size sweep. For each minibatch in
# ${MINIBATCH_SIZES}, runs the full release_affinity pipeline
# (train -> select -> calibrate -> eval) into a per-size subdirectory
# and records phase wall times + eval metrics into sweep_summary.csv.
#
# Reuses the existing v7 box's downloaded data (training data, allele
# sequences, pretrain predictions, eval benchmarks, public predictor)
# so each iteration only re-trains and re-evaluates -- no fresh box
# bring-up per size.
#
# Required env (defaults assume the v7 release-exact run dir layout):
#   MHCFLURRY_OUT     base output dir (e.g. .../out/affinity)
#   SWEEP_OUT         sweep root (default: $MHCFLURRY_OUT/full_sweep)
#   MINIBATCH_SIZES   space-separated list (default: "256 512 1024 2048 4096 8192 16384")
#   TRAIN_DATA        train_data.csv.bz2 from prior run
#   HYPERPARAMS_TPL   template hyperparameters.yaml to patch
#   ALLELE_SEQUENCES  path to a pseudosequence CSV
#   PRETRAIN_DATA     path to random_peptide_predictions/predictions.csv.bz2
#   DATA_EVAL_DIR     data_evaluation/ for benchmark hits/decoys
#   GPUS, MAX_TASKS_PER_WORKER
#   MHCFLURRY_SCALE_LR_BASE_MB  reference minibatch for sqrt LR scaling
#                                (default 128, matching the historical
#                                2.0.0–2.2.x pan-allele recipe — the
#                                old documented value of 64 was wrong).
#   MHCFLURRY_SCALE_LR  if "1", multiply learning_rate by sqrt(mb/BASE_MB) for
#                       mb>BASE_MB (square-root LR scaling, Goyal et al. 2017
#                       Appendix B). Default "0" leaves LR untouched, so
#                       the sweep isolates the effect of minibatch size at
#                       fixed LR -- run twice (once with, once without)
#                       into separate $SWEEP_OUT roots to compare.
# (--num-jobs and --max-workers-per-gpu auto-resolve per phase)
#
# Output layout:
#   $SWEEP_OUT/
#     mb_1024/{hyperparameters.yaml,train.log,select.log,calibrate.log,eval.log,
#              models.unselected.combined/, models.combined/, eval_comparison/}
#     mb_2048/...
#     ...
#     sweep_summary.csv  -- one row per minibatch
#
set -euo pipefail
set -x

pseudosequence_lookup() {
  python -c 'from mhcflurry.pseudosequences import main; main()' "$@"
}

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
SWEEP_OUT="${SWEEP_OUT:-$MHCFLURRY_OUT/full_sweep}"
MINIBATCH_SIZES="${MINIBATCH_SIZES:-256 512 1024 2048 4096 8192 16384}"
TRAIN_DATA="${TRAIN_DATA:-$MHCFLURRY_OUT/train_data.csv.bz2}"
HYPERPARAMS_TPL="${HYPERPARAMS_TPL:-$MHCFLURRY_OUT/hyperparameters.yaml}"
if [ -z "${ALLELE_SEQUENCES:-}" ]; then
  ALLELE_SEQUENCES_DIR="$(mhcflurry-downloads path allele_sequences)"
  ALLELE_SEQUENCES="$(pseudosequence_lookup path \
    --directory "$ALLELE_SEQUENCES_DIR" \
    --length 39 \
    --fallback-legacy)"
fi
PRETRAIN_DATA="${PRETRAIN_DATA:-$(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2}"
DATA_EVAL_DIR="${DATA_EVAL_DIR:-$(mhcflurry-downloads path data_evaluation)}"
GPUS="${GPUS:-8}"
MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-12}"
# Model selection bounds. mhcflurry-class1-select-pan-allele-models picks
# between MIN_MODELS_PER_FOLD and MAX_MODELS_PER_FOLD models per training
# fold; with 4 folds, total ensemble size lands in
# [4*MIN_MODELS_PER_FOLD, 4*MAX_MODELS_PER_FOLD]. Default range matches
# the historical 2.x recipe; pin both to the same value to fix ensemble
# size (e.g. =3 → 12 models = 3/fold × 4 folds).
MIN_MODELS_PER_FOLD="${MIN_MODELS_PER_FOLD:-2}"
MAX_MODELS_PER_FOLD="${MAX_MODELS_PER_FOLD:-8}"
MHCFLURRY_SCALE_LR="${MHCFLURRY_SCALE_LR:-0}"
MHCFLURRY_SCALE_LR_BASE_MB="${MHCFLURRY_SCALE_LR_BASE_MB:-128}"
# When 1, skip the calibrate phase entirely (eval doesn't need percentile
# ranks, only the predict outputs from models.combined). Useful for fast
# multi-cell sweeps where calibrate would be ~10-15min/cell of pure
# overhead. Run calibrate manually for the winning cell at the end.
MHCFLURRY_SKIP_CALIBRATE="${MHCFLURRY_SKIP_CALIBRATE:-0}"
# --num-jobs and --max-workers-per-gpu are passed as ``auto`` so that
# the resolver in mhcflurry/parallelism picks values that match
# the box (free VRAM, GPU count, per-worker memory) instead of being
# pinned to fixed sweep-time defaults that drift away from reality.

mkdir -p "$SWEEP_OUT"
SUMMARY="$SWEEP_OUT/sweep_summary.csv"
SUMMARY_HEADER="minibatch,n_models,params_M,train_seconds,select_seconds,calibrate_seconds,eval_seconds,total_seconds,n_alleles_reported,n_hits,n_rows,roc_auc_macro_new,roc_auc_macro_public,pr_auc_macro_new,pr_auc_macro_public,ppv_at_n_macro_new,ppv_at_n_macro_public,roc_auc_micro_new,roc_auc_micro_public,pr_auc_micro_new,pr_auc_micro_public,ppv_at_n_micro_new,ppv_at_n_micro_public,new_better_roc_auc,new_better_pr_auc,new_better_ppv_at_n,public_better_roc_auc,public_better_pr_auc,public_better_ppv_at_n"

# Rebuild sweep_summary.csv from the durable per-cell row.csv files (header +
# each cell's row in MINIBATCH_SIZES order). Written atomically via a temp
# file so a crash never leaves a half-written summary, and regenerated from
# the per-cell rows every time, so a completed cell's row can never be lost.
rebuild_summary() {
    local mb
    {
        echo "$SUMMARY_HEADER"
        for mb in $MINIBATCH_SIZES; do
            [ -f "$SWEEP_OUT/mb_$mb/row.csv" ] && cat "$SWEEP_OUT/mb_$mb/row.csv"
        done
    } > "$SUMMARY.tmp"
    mv "$SUMMARY.tmp" "$SUMMARY"
}

for MB in $MINIBATCH_SIZES; do
    SIZE_OUT="$SWEEP_OUT/mb_$MB"
    # Per-cell completion is marked by row.csv, written (atomically) only
    # after metrics are extracted -- NOT by summary.json, which compare-models
    # writes partway through eval. A cell that died after eval but before its
    # row was recorded therefore re-enters, skips every expensive phase via
    # its .<phase>.done sentinel, and just regenerates the row.
    if [ -f "$SIZE_OUT/row.csv" ]; then
        echo "=== minibatch=$MB already complete, skipping ==="
        continue
    fi
    mkdir -p "$SIZE_OUT"
    cd "$SIZE_OUT"

    # Phase-level idempotency: each phase writes a sentinel
    # ($SIZE_OUT/.<phase>.done) on success, recording its wall time
    # for the summary CSV. A killed-mid-phase rerun skips already-done
    # phases (avoids re-training when only calibrate failed last time).
    TRAIN_SENTINEL="$SIZE_OUT/.train.done"
    SELECT_SENTINEL="$SIZE_OUT/.select.done"
    CALIBRATE_SENTINEL="$SIZE_OUT/.calibrate.done"
    EVAL_SENTINEL="$SIZE_OUT/.eval.done"

    # Patch the hyperparameters template's minibatch_size in every spec.
    # When MHCFLURRY_SCALE_LR=1 and mb > MHCFLURRY_SCALE_LR_BASE_MB,
    # also scale learning_rate by sqrt(mb/BASE_MB) (square-root LR
    # scaling, Goyal et al. 2017). BASE_MB defaults to 128 to match the
    # 2.0.0–2.2.x pan-allele recipe; sweeps at or below the baseline
    # don't get downscaled because smaller batches already train fine
    # at the baseline LR.
    if [ ! -f "$SIZE_OUT/hyperparameters.yaml" ]; then
        python3 - "$HYPERPARAMS_TPL" "$MB" "$MHCFLURRY_SCALE_LR" \
            "$MHCFLURRY_SCALE_LR_BASE_MB" \
            <<'PY' > "$SIZE_OUT/hyperparameters.yaml"
import math, sys, yaml
src = sys.argv[1]
mb = int(sys.argv[2])
scale_lr = sys.argv[3] == "1"
base_mb = int(sys.argv[4])
with open(src) as f:
    specs = yaml.safe_load(f)
for spec in specs:
    spec["minibatch_size"] = mb
    if (scale_lr and mb > base_mb
            and spec.get("learning_rate") is not None):
        spec["learning_rate"] = (
            float(spec["learning_rate"]) * math.sqrt(mb / base_mb))
print(yaml.safe_dump(specs))
PY
    fi

    # ---- train ----
    if [ -f "$TRAIN_SENTINEL" ]; then
        train_sec=$(cat "$TRAIN_SENTINEL")
        echo "=== minibatch=$MB train already complete (${train_sec}s), skipping ==="
    else
    train_start=$(date +%s)
    mhcflurry-class1-train-pan-allele-models \
        --data "$TRAIN_DATA" \
        --allele-sequences "$ALLELE_SEQUENCES" \
        --pretrain-data "$PRETRAIN_DATA" \
        --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
        --num-folds 4 \
        --hyperparameters hyperparameters.yaml \
        --out-models-dir "$SIZE_OUT/models.unselected.combined" \
        --worker-log-dir "$SIZE_OUT" \
        --num-jobs auto \
        --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
        --gpus "$GPUS" \
        --max-workers-per-gpu auto \
        --dataloader-num-workers 1 \
        --torch-compile auto \
        --matmul-precision none \
        --enable-timing \
        2>&1 | tee "$SIZE_OUT/train.log"
    train_end=$(date +%s)
    train_sec=$(( train_end - train_start ))
    echo "$train_sec" > "$TRAIN_SENTINEL"
    fi

    # ---- select ----
    if [ -f "$SELECT_SENTINEL" ]; then
        select_sec=$(cat "$SELECT_SENTINEL")
        echo "=== minibatch=$MB select already complete (${select_sec}s), skipping ==="
    else
    select_start=$(date +%s)
    cd "$SIZE_OUT"
    mhcflurry-class1-select-pan-allele-models \
        --data "$SIZE_OUT/models.unselected.combined/train_data.csv.bz2" \
        --models-dir "$SIZE_OUT/models.unselected.combined" \
        --out-models-dir "$SIZE_OUT/models.combined" \
        --min-models-per-fold "$MIN_MODELS_PER_FOLD" \
        --max-models-per-fold "$MAX_MODELS_PER_FOLD" \
        --num-jobs auto \
        --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
        --gpus "$GPUS" \
        --max-workers-per-gpu auto \
        --dataloader-num-workers 1 \
        --torch-compile auto \
        --matmul-precision none \
        --enable-timing \
        2>&1 | tee "$SIZE_OUT/select.log"
    select_end=$(date +%s)
    select_sec=$(( select_end - select_start ))
    echo "$select_sec" > "$SELECT_SENTINEL"
    fi

    # ---- calibrate (auto MWPG accounts for cartesian forward + cache) ----
    if [ "$MHCFLURRY_SKIP_CALIBRATE" = "1" ]; then
        cal_sec=0
        echo "0" > "$CALIBRATE_SENTINEL"
        echo "=== minibatch=$MB calibrate skipped (MHCFLURRY_SKIP_CALIBRATE=1) ==="
    elif [ -f "$CALIBRATE_SENTINEL" ]; then
        cal_sec=$(cat "$CALIBRATE_SENTINEL")
        echo "=== minibatch=$MB calibrate already complete (${cal_sec}s), skipping ==="
    else
    cal_start=$(date +%s)
    mhcflurry-calibrate-percentile-ranks \
        --models-dir "$SIZE_OUT/models.combined" \
        --match-amino-acid-distribution-data "$SIZE_OUT/models.unselected.combined/train_data.csv.bz2" \
        --motif-summary --num-peptides-per-length 50000 \
        --alleles-per-work-chunk 30 --gpu-batched --verbosity 1 \
        --num-jobs auto \
        --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
        --gpus "$GPUS" \
        --max-workers-per-gpu auto \
        --dataloader-num-workers 1 \
        --torch-compile auto \
        --matmul-precision none \
        --enable-timing \
        2>&1 | tee "$SIZE_OUT/calibrate.log"
    cal_end=$(date +%s)
    cal_sec=$(( cal_end - cal_start ))
    echo "$cal_sec" > "$CALIBRATE_SENTINEL"
    fi

    # ---- eval (compare against public 2.2.0) ----
    if [ -f "$EVAL_SENTINEL" ]; then
        eval_sec=$(cat "$EVAL_SENTINEL")
        echo "=== minibatch=$MB eval already complete (${eval_sec}s), skipping ==="
    else
    eval_start=$(date +%s)
    mkdir -p "$SIZE_OUT/eval_comparison"
    mhcflurry compare-models \
        --a "$SIZE_OUT/models.combined" \
        --a-label "mb_$MB" \
        --b public \
        --data-dir "$DATA_EVAL_DIR" \
        --include affinity \
        --out "$SIZE_OUT/eval_comparison" \
        2>&1 | tee "$SIZE_OUT/eval.log"
    eval_end=$(date +%s)
    eval_sec=$(( eval_end - eval_start ))
    echo "$eval_sec" > "$EVAL_SENTINEL"
    fi

    total_sec=$(( train_sec + select_sec + cal_sec + eval_sec ))

    # Pull metrics out of summary.json into the cell's own row.csv (written
    # atomically; the sweep_summary.csv is rebuilt from these rows below).
    # n_models + params_M are derived from models.combined: row count of
    # manifest.csv gives ensemble size, and summing the element counts across
    # each weights_<name>.npz gives the total number of saved parameters (in
    # millions).
    python3 - "$SIZE_OUT/eval_comparison/affinity/summary.json" "$MB" "$train_sec" \
        "$select_sec" "$cal_sec" "$eval_sec" "$total_sec" "$SIZE_OUT/row.csv" \
        "$SIZE_OUT/models.combined" <<'PY'
import json, os, sys
import numpy
import pandas
(summary_path, mb, train_s, sel_s, cal_s, eval_s, tot_s, csv_path,
 models_dir) = sys.argv[1:10]
with open(summary_path) as f:
    s = json.load(f)
mac = s["macro_mean_over_alleles"]
mic = s["micro_pooled"]
ac = s["allele_count"]
manifest = pandas.read_csv(os.path.join(models_dir, "manifest.csv"))
n_models = len(manifest)
total_params = 0
for name in manifest["model_name"]:
    weights_path = os.path.join(models_dir, f"weights_{name}.npz")
    with numpy.load(weights_path) as wf:
        for key in wf.files:
            total_params += int(wf[key].size)
params_M = total_params / 1_000_000.0
row = [
    mb, n_models, f"{params_M:.3f}",
    train_s, sel_s, cal_s, eval_s, tot_s,
    s["n_alleles_reported"], s["n_hits"], s["n_rows"],
    mac["roc_auc"]["a"], mac["roc_auc"]["b"],
    mac["pr_auc"]["a"],  mac["pr_auc"]["b"],
    mac["ppv_at_n"]["a"],mac["ppv_at_n"]["b"],
    mic["a"]["roc_auc"], mic["b"]["roc_auc"],
    mic["a"]["pr_auc"],  mic["b"]["pr_auc"],
    mic["a"]["ppv_at_n"],mic["b"]["ppv_at_n"],
    ac["a_better_roc_auc"], ac["a_better_pr_auc"], ac["a_better_ppv_at_n"],
    ac["b_better_roc_auc"], ac["b_better_pr_auc"], ac["b_better_ppv_at_n"],
]
# Write the single row atomically (temp file + os.replace) so a crash
# mid-write can never leave a truncated line that a later run would treat as
# a completed cell.
tmp_path = csv_path + ".tmp"
with open(tmp_path, "w") as f:
    f.write(",".join(str(x) for x in row) + "\n")
os.replace(tmp_path, csv_path)
print(f"=== minibatch={mb} done: n_models={n_models} params={params_M:.2f}M "
      f"train={train_s}s select={sel_s}s calibrate={cal_s}s "
      f"eval={eval_s}s total={tot_s}s ===")
PY

    rebuild_summary
    cd "$MHCFLURRY_OUT"
done

# Final rebuild so a run where every cell was already complete (and therefore
# skipped above) still refreshes the summary from the per-cell rows.
rebuild_summary

echo "=== sweep complete ==="
cat "$SUMMARY"
