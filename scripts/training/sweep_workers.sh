#!/usr/bin/env bash
#
# Single-network sweep — measures per-network training wall time across
# cheap, zero-startup-overhead perf knobs.
#
# Fixed:
#   - 1 network per cell (1 fold × 1 arch × 1 replicate)
#   - TF32 on (default on any CUDA device via
#     _configure_matmul_precision)
#   - cudnn.benchmark on (same helper)
#   - torch.compile OFF (startup codegen cost doesn't amortize over a
#     single network)
#   - maxtasksperchild=1000 (single worker does the single item anyway)
#   - gpu_workers=1, num_jobs=1 (only 1 item to process; parallelism
#     axis is moot)
#
# Varied:
#   - dataloader_num_workers ∈ {0, 2, 4}
#   - minibatch_size ∈ {512, 1024, 2048}
#   - OMP_NUM_THREADS ∈ {1, auto}
#
# 3 × 3 × 2 = 18 cells, ~3-5 min each → ~1.5h total.
set -euo pipefail

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
SWEEP_RESULTS_CSV="${SWEEP_RESULTS_CSV:-$MHCFLURRY_OUT/sweep_results.csv}"
REPO="${REPO:-$HOME/runplz-repo}"

export PYTHONUNBUFFERED=1
# Compile stays off; TF32 + cudnn.benchmark run unconditionally on CUDA.
# OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS are set
# per-cell via set_cpu_threads below (they vary with OMP axis and the
# auto-allocator reacts to GPUS / MAX_WORKERS_PER_GPU / DATALOADER).

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/set_cpu_threads.sh"

cd "$MHCFLURRY_OUT"

# ---- one-time data prep (idempotent) ----
if [ ! -f "$MHCFLURRY_OUT/train_data.csv.bz2" ]; then
    mhcflurry-downloads fetch data_curated allele_sequences random_peptide_predictions
    cp "$REPO/downloads-generation/models_class1_pan/reassign_mass_spec_training_data.py" .
    cp "$REPO/downloads-generation/models_class1_pan/additional_alleles.txt" .
    DATA_CURATED=$(mhcflurry-downloads path data_curated)
    python reassign_mass_spec_training_data.py \
        "$DATA_CURATED/curated_training_data.csv.bz2" \
        --set-measurement-value 100 \
        --out-csv train_data.csv
    bzip2 -f train_data.csv
fi

# Always regenerate the base hyperparameters (we'll overwrite batch
# size per-cell) — the file is small and this avoids stale state.
cp "$REPO/downloads-generation/models_class1_pan/generate_hyperparameters.py" .
python generate_hyperparameters.py > hyperparameters.full.yaml

ALLELE_SEQUENCES=$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv
RANDOM_PRED=$(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2

echo "phase,dataloader_workers,minibatch_size,threads,workers_per_gpu,wall_seconds,setup_seconds,training_seconds,exit_code" > "$SWEEP_RESULTS_CSV"

GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
if [ "$GPUS" -eq 0 ]; then echo "No GPU"; exit 1; fi

# Single network per cell: 1 fold × 1 arch × 1 replicate.
NUM_FOLDS=1
ITEMS_PER_CELL=1

run_cell() {
    local DL_W=$1 MINIBATCH=$2 THREADS_LABEL=$3 WPG=$4
    local CELL="$MHCFLURRY_OUT/cell_dl${DL_W}_mb${MINIBATCH}_threads${THREADS_LABEL}_wpg${WPG}"
    mkdir -p "$CELL"; cd "$CELL"

    # Fixed epochs: exactly 2 pretrain + 2 finetune. Patience disabled
    # (set to a large number so early stopping never fires within 2
    # epochs). This removes the variable-tail-length confound — every
    # cell does the same amount of training work, so wall time
    # directly reflects config throughput.
    python - <<PY > "$CELL/hyperparameters.cell.yaml"
import yaml
hp = yaml.safe_load(open("$MHCFLURRY_OUT/hyperparameters.full.yaml"))[:1]
for entry in hp:
    entry["max_epochs"] = 2
    entry["patience"] = 999
    entry["min_delta"] = 0.0
    entry["early_stopping"] = False
    entry["minibatch_size"] = $MINIBATCH
    entry["dataloader_num_workers"] = $DL_W
    entry["train_data"] = dict(entry.get("train_data", {}))
    entry["train_data"]["pretrain_max_epochs"] = 2
    entry["train_data"]["pretrain_min_epochs"] = 2
    entry["train_data"]["pretrain_patience"] = 999
    entry["train_data"]["pretrain_max_val_loss"] = 1.0  # effectively disabled
print(yaml.dump(hp))
PY

    # Sweep axes: dataloader workers, batch size, thread budget.
    # Thread axis: "1" = force minimal, "auto" = let helper compute
    # from core count. In both cases set_cpu_threads exports OMP, MKL,
    # and OPENBLAS consistently (manual per-var overrides still
    # respected if the user sets them pre-script).
    export MAX_WORKERS_PER_GPU=$WPG
    export DATALOADER_NUM_WORKERS=$DL_W
    export MHCFLURRY_TORCH_COMPILE=0
    # Clear any lingering env from a prior cell so set_cpu_threads
    # re-auto-computes cleanly.
    unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS MHCFLURRY_CPU_THREADS_PER_WORKER
    if [ "$THREADS_LABEL" = "1" ]; then
        export MHCFLURRY_CPU_THREADS_PER_WORKER=1
    fi
    set_cpu_threads

    echo ""
    echo "=== CELL: dl=$DL_W batch=$MINIBATCH threads=$THREADS_LABEL wpg=$WPG ==="

    set +e
    local start=$(date +%s)
    mhcflurry-class1-train-pan-allele-models \
        --data "$MHCFLURRY_OUT/train_data.csv.bz2" \
        --allele-sequences "$ALLELE_SEQUENCES" \
        --pretrain-data "$RANDOM_PRED" \
        --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
        --num-folds "$NUM_FOLDS" \
        --hyperparameters "$CELL/hyperparameters.cell.yaml" \
        --out-models-dir "$CELL/models" \
        --worker-log-dir "$CELL" \
        --use-encoding-cache \
        --num-jobs "$WPG" \
        --max-tasks-per-worker 1000 \
        --gpus "$GPUS" \
        --max-workers-per-gpu "$WPG" \
        > "$CELL/stdout.log" 2> "$CELL/stderr.log"
    local exit_code=$?
    local end=$(date +%s)
    set -e

    local wall=$(( end - start ))

    # Pull TIMING_MARKER lines out of stdout to separate setup vs
    # training time. Format: "TIMING_MARKER <phase> <epoch_seconds>".
    # See train_pan_allele_models_command.py for emit sites.
    local setup_sec=-1
    local training_sec=-1
    if [ -f "$CELL/stdout.log" ]; then
        local ts_start ts_setup ts_train
        ts_start=$(grep "TIMING_MARKER start " "$CELL/stdout.log" | awk '{print $3}' | tail -1)
        ts_setup=$(grep "TIMING_MARKER setup_done " "$CELL/stdout.log" | awk '{print $3}' | tail -1)
        ts_train=$(grep "TIMING_MARKER training_done " "$CELL/stdout.log" | awk '{print $3}' | tail -1)
        if [ -n "$ts_start" ] && [ -n "$ts_setup" ]; then
            setup_sec=$(awk "BEGIN {printf \"%.1f\", $ts_setup - $ts_start}")
        fi
        if [ -n "$ts_setup" ] && [ -n "$ts_train" ]; then
            training_sec=$(awk "BEGIN {printf \"%.1f\", $ts_train - $ts_setup}")
        fi
    fi

    echo "=== done: dl=$DL_W batch=$MINIBATCH threads=$THREADS_LABEL wpg=$WPG wall=${wall}s setup=${setup_sec}s training=${training_sec}s exit=${exit_code} ==="
    echo "G,$DL_W,$MINIBATCH,$THREADS_LABEL,$WPG,$wall,$setup_sec,$training_sec,$exit_code" >> "$SWEEP_RESULTS_CSV"
    cd "$MHCFLURRY_OUT"
}

# ---- matrix ----
# threads × dataloader × workers_per_gpu × batch = 2×2×2×2 = 16 cells.
# Purpose: isolate the effect of each knob with all others pinned; each
# cell trains exactly 2 networks (num_folds=2, 1 arch) for 2 epochs
# each so wall is fully comparable across cells.
for DL in 1 2; do
    for WPG in 1 2; do
        for MB in 128 512; do
            for THREADS in 1 auto; do
                run_cell "$DL" "$MB" "$THREADS" "$WPG"
            done
        done
    done
done

echo ""
echo "=== SWEEP COMPLETE ==="
cat "$SWEEP_RESULTS_CSV"

python - <<PY
import csv
rows = []
with open("$SWEEP_RESULTS_CSV") as f:
    for r in csv.DictReader(f):
        if r["exit_code"] == "0":
            rows.append(r)
if not rows:
    print("NO WINNER"); raise SystemExit(1)
# Rank by training_seconds if available (config-dependent work),
# else fall back to total wall. Setup time is a constant spawn/data-
# prep tax that's roughly the same per cell and shouldn't drive the
# winner choice.
def sort_key(r):
    try:
        ts = float(r["training_seconds"])
        if ts > 0:
            return ts
    except (ValueError, KeyError):
        pass
    return float(r["wall_seconds"])
rows.sort(key=sort_key)
print("\nTop 6 fastest configurations (ranked by training time; wall and setup shown for context):")
for r in rows[:6]:
    print(
        f"  dl={r['dataloader_workers']} batch={r['minibatch_size']} "
        f"threads={r['threads']} wpg={r['workers_per_gpu']}  "
        f"training={r.get('training_seconds','?')}s  "
        f"setup={r.get('setup_seconds','?')}s  wall={r['wall_seconds']}s"
    )
best = rows[0]
print(
    f"\nWINNER: dl={best['dataloader_workers']} batch={best['minibatch_size']} "
    f"threads={best['threads']} wpg={best['workers_per_gpu']}  "
    f"training={best.get('training_seconds','?')}s  wall={best['wall_seconds']}s"
)
PY
