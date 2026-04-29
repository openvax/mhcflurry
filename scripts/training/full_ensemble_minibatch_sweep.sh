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
#   MINIBATCH_SIZES   space-separated list (default: "1024 2048 4096 8192 16384")
#   TRAIN_DATA        train_data.csv.bz2 from prior run
#   HYPERPARAMS_TPL   template hyperparameters.yaml to patch
#   ALLELE_SEQUENCES  path to allele_sequences.csv
#   PRETRAIN_DATA     path to random_peptide_predictions/predictions.csv.bz2
#   PUBLIC_MODELS_DIR public 2.2.0 models.combined for eval comparison
#   DATA_EVAL_DIR     data_evaluation/ for benchmark hits/decoys
#   COMPARE_SCRIPT    path to compare_new_vs_public.py
#   GPUS, NUM_JOBS, MAX_TASKS_PER_WORKER, MAX_WORKERS_PER_GPU,
#   CALIBRATE_MAX_WORKERS_PER_GPU, ENCODING_CACHE_DIR
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
MINIBATCH_SIZES="${MINIBATCH_SIZES:-1024 2048 4096 8192 16384}"
TRAIN_DATA="${TRAIN_DATA:-$MHCFLURRY_OUT/train_data.csv.bz2}"
HYPERPARAMS_TPL="${HYPERPARAMS_TPL:-$MHCFLURRY_OUT/hyperparameters.yaml}"
ALLELE_SEQUENCES="${ALLELE_SEQUENCES:-$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv}"
PRETRAIN_DATA="${PRETRAIN_DATA:-$(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2}"
PUBLIC_MODELS_DIR="${PUBLIC_MODELS_DIR:-$(mhcflurry-downloads path models_class1_pan)/models.combined}"
DATA_EVAL_DIR="${DATA_EVAL_DIR:-$(mhcflurry-downloads path data_evaluation)}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:?COMPARE_SCRIPT must point at compare_new_vs_public.py}"
GPUS="${GPUS:-8}"
NUM_JOBS="${NUM_JOBS:-32}"
MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-12}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-4}"
CALIBRATE_MAX_WORKERS_PER_GPU="${CALIBRATE_MAX_WORKERS_PER_GPU:-1}"
ENCODING_CACHE_DIR="${ENCODING_CACHE_DIR:-$HOME/runplz-cache/encoding_cache}"

mkdir -p "$SWEEP_OUT"
SUMMARY="$SWEEP_OUT/sweep_summary.csv"

if [ ! -f "$SUMMARY" ]; then
    echo "minibatch,train_seconds,select_seconds,calibrate_seconds,eval_seconds,total_seconds,n_alleles_reported,n_hits,n_rows,roc_auc_macro_new,roc_auc_macro_public,pr_auc_macro_new,pr_auc_macro_public,ppv_at_n_macro_new,ppv_at_n_macro_public,roc_auc_micro_new,roc_auc_micro_public,pr_auc_micro_new,pr_auc_micro_public,ppv_at_n_micro_new,ppv_at_n_micro_public,new_better_roc_auc,new_better_pr_auc,new_better_ppv_at_n,public_better_roc_auc,public_better_pr_auc,public_better_ppv_at_n" > "$SUMMARY"
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
    if [ ! -f "$SIZE_OUT/hyperparameters.yaml" ]; then
        python3 - "$HYPERPARAMS_TPL" "$MB" <<'PY' > "$SIZE_OUT/hyperparameters.yaml"
import sys, yaml
src, mb = sys.argv[1], int(sys.argv[2])
with open(src) as f:
    specs = yaml.safe_load(f)
for spec in specs:
    spec["minibatch_size"] = mb
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
        --num-jobs "$NUM_JOBS" \
        --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
        --gpus "$GPUS" \
        --max-workers-per-gpu "$MAX_WORKERS_PER_GPU" \
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
        --min-models 2 --max-models 8 \
        --num-jobs "$NUM_JOBS" \
        --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
        --gpus "$GPUS" \
        --max-workers-per-gpu "$MAX_WORKERS_PER_GPU" \
        --dataloader-num-workers 1 \
        --torch-compile auto \
        --matmul-precision none \
        --enable-timing \
        2>&1 | tee "$SIZE_OUT/select.log"
    select_end=$(date +%s)
    select_sec=$(( select_end - select_start ))
    echo "$select_sec" > "$SELECT_SENTINEL"
    fi

    # ---- calibrate (max-workers-per-gpu pinned to avoid auto-OOM) ----
    if [ -f "$CALIBRATE_SENTINEL" ]; then
        cal_sec=$(cat "$CALIBRATE_SENTINEL")
        echo "=== minibatch=$MB calibrate already complete (${cal_sec}s), skipping ==="
    else
    cal_start=$(date +%s)
    mhcflurry-calibrate-percentile-ranks \
        --models-dir "$SIZE_OUT/models.combined" \
        --match-amino-acid-distribution-data "$SIZE_OUT/models.unselected.combined/train_data.csv.bz2" \
        --motif-summary --num-peptides-per-length 50000 \
        --alleles-per-work-chunk 30 --gpu-batched --verbosity 1 \
        --num-jobs "$NUM_JOBS" \
        --max-tasks-per-worker "$MAX_TASKS_PER_WORKER" \
        --gpus "$GPUS" \
        --max-workers-per-gpu "$CALIBRATE_MAX_WORKERS_PER_GPU" \
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

    # Pull metrics out of summary.json into the sweep CSV.
    python3 - "$SIZE_OUT/eval_comparison/summary.json" "$MB" "$train_sec" \
        "$select_sec" "$cal_sec" "$eval_sec" "$total_sec" "$SUMMARY" <<'PY'
import json, sys
summary_path, mb, train_s, sel_s, cal_s, eval_s, tot_s, csv_path = sys.argv[1:9]
with open(summary_path) as f:
    s = json.load(f)
mac = s["macro_mean_over_alleles"]
mic = s["micro_pooled"]
ac = s["allele_count"]
row = [
    mb, train_s, sel_s, cal_s, eval_s, tot_s,
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
print(f"=== minibatch={mb} done: train={train_s}s select={sel_s}s "
      f"calibrate={cal_s}s eval={eval_s}s total={tot_s}s ===")
PY

    cd "$MHCFLURRY_OUT"
done

echo "=== sweep complete ==="
cat "$SUMMARY"
