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
#   ALLELE_SEQUENCES  path to allele_sequences.csv
#   PRETRAIN_DATA     path to random_peptide_predictions/predictions.csv.bz2
#   PUBLIC_MODELS_DIR public 2.2.0 models.combined for eval comparison
#   DATA_EVAL_DIR     data_evaluation/ for benchmark hits/decoys
#   COMPARE_SCRIPT    path to compare_new_vs_public.py
#   GPUS, MAX_TASKS_PER_WORKER, ENCODING_CACHE_DIR
#   MHCFLURRY_SCALE_LR  if "1", multiply learning_rate by sqrt(mb/64) for
#                       mb>64 (square-root LR scaling, Goyal et al. 2017
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

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
SWEEP_OUT="${SWEEP_OUT:-$MHCFLURRY_OUT/full_sweep}"
MINIBATCH_SIZES="${MINIBATCH_SIZES:-256 512 1024 2048 4096 8192 16384}"
TRAIN_DATA="${TRAIN_DATA:-$MHCFLURRY_OUT/train_data.csv.bz2}"
HYPERPARAMS_TPL="${HYPERPARAMS_TPL:-$MHCFLURRY_OUT/hyperparameters.yaml}"
ALLELE_SEQUENCES="${ALLELE_SEQUENCES:-$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv}"
PRETRAIN_DATA="${PRETRAIN_DATA:-$(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2}"
PUBLIC_MODELS_DIR="${PUBLIC_MODELS_DIR:-$(mhcflurry-downloads path models_class1_pan)/models.combined}"
DATA_EVAL_DIR="${DATA_EVAL_DIR:-$(mhcflurry-downloads path data_evaluation)}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:?COMPARE_SCRIPT must point at compare_new_vs_public.py}"
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
ENCODING_CACHE_DIR="${ENCODING_CACHE_DIR:-$HOME/runplz-cache/encoding_cache}"
MHCFLURRY_SCALE_LR="${MHCFLURRY_SCALE_LR:-0}"
# When 1, skip the calibrate phase entirely (eval doesn't need percentile
# ranks, only the predict outputs from models.combined). Useful for fast
# multi-cell sweeps where calibrate would be ~10-15min/cell of pure
# overhead. Run calibrate manually for the winning cell at the end.
MHCFLURRY_SKIP_CALIBRATE="${MHCFLURRY_SKIP_CALIBRATE:-0}"
# --num-jobs and --max-workers-per-gpu are passed as ``auto`` so that
# the resolver in mhcflurry/local_parallelism.py picks values that match
# the box (free VRAM, GPU count, per-worker memory) instead of being
# pinned to fixed sweep-time defaults that drift away from reality.

mkdir -p "$SWEEP_OUT"
SUMMARY="$SWEEP_OUT/sweep_summary.csv"

if [ ! -f "$SUMMARY" ]; then
    echo "minibatch,n_models,params_M,train_seconds,select_seconds,calibrate_seconds,eval_seconds,total_seconds,n_alleles_reported,n_hits,n_rows,roc_auc_macro_new,roc_auc_macro_public,pr_auc_macro_new,pr_auc_macro_public,ppv_at_n_macro_new,ppv_at_n_macro_public,roc_auc_micro_new,roc_auc_micro_public,pr_auc_micro_new,pr_auc_micro_public,ppv_at_n_micro_new,ppv_at_n_micro_public,new_better_roc_auc,new_better_pr_auc,new_better_ppv_at_n,public_better_roc_auc,public_better_pr_auc,public_better_ppv_at_n" > "$SUMMARY"
fi

for MB in $MINIBATCH_SIZES; do
    SIZE_OUT="$SWEEP_OUT/mb_$MB"
    if [ -f "$SIZE_OUT/eval_comparison/summary.json" ]; then
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
    # When MHCFLURRY_SCALE_LR=1 and mb>64, also scale learning_rate by
    # sqrt(mb/64) (square-root LR scaling). 64 is the historical mhcflurry
    # baseline batch size; sweeps below 64 don't get downscaled because
    # smaller batches already train fine at the baseline LR.
    if [ ! -f "$SIZE_OUT/hyperparameters.yaml" ]; then
        python3 - "$HYPERPARAMS_TPL" "$MB" "$MHCFLURRY_SCALE_LR" \
            <<'PY' > "$SIZE_OUT/hyperparameters.yaml"
import math, sys, yaml
src, mb, scale_lr = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
with open(src) as f:
    specs = yaml.safe_load(f)
for spec in specs:
    spec["minibatch_size"] = mb
    if scale_lr and mb > 64 and "learning_rate" in spec:
        spec["learning_rate"] = float(spec["learning_rate"]) * math.sqrt(mb / 64.0)
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
        --use-encoding-cache --encoding-cache-dir "$ENCODING_CACHE_DIR" \
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
    python3 "$COMPARE_SCRIPT" \
        --new-models-dir "$SIZE_OUT/models.combined" \
        --public-models-dir "$PUBLIC_MODELS_DIR" \
        --data-dir "$DATA_EVAL_DIR" \
        --out "$SIZE_OUT/eval_comparison" \
        2>&1 | tee "$SIZE_OUT/eval.log"
    eval_end=$(date +%s)
    eval_sec=$(( eval_end - eval_start ))
    echo "$eval_sec" > "$EVAL_SENTINEL"
    fi

    total_sec=$(( train_sec + select_sec + cal_sec + eval_sec ))

    # Pull metrics out of summary.json into the sweep CSV. n_models +
    # params_M are derived from models.combined: row count of manifest.csv
    # gives ensemble size, and summing parameter counts across each
    # weights_<name>.npz gives total trainable params (in millions).
    python3 - "$SIZE_OUT/eval_comparison/summary.json" "$MB" "$train_sec" \
        "$select_sec" "$cal_sec" "$eval_sec" "$total_sec" "$SUMMARY" \
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
    mac["roc_auc"]["new"], mac["roc_auc"]["public"],
    mac["pr_auc"]["new"],  mac["pr_auc"]["public"],
    mac["ppv_at_n"]["new"],mac["ppv_at_n"]["public"],
    mic["new"]["roc_auc"], mic["public"]["roc_auc"],
    mic["new"]["pr_auc"],  mic["public"]["pr_auc"],
    mic["new"]["ppv_at_n"],mic["public"]["ppv_at_n"],
    ac["new_better_roc_auc"], ac["new_better_pr_auc"], ac["new_better_ppv_at_n"],
    ac["public_better_roc_auc"], ac["public_better_pr_auc"], ac["public_better_ppv_at_n"],
]
with open(csv_path, "a") as f:
    f.write(",".join(str(x) for x in row) + "\n")
print(f"=== minibatch={mb} done: n_models={n_models} params={params_M:.2f}M "
      f"train={train_s}s select={sel_s}s calibrate={cal_s}s "
      f"eval={eval_s}s total={tot_s}s ===")
PY

    cd "$MHCFLURRY_OUT"
done

echo "=== sweep complete ==="
cat "$SUMMARY"
