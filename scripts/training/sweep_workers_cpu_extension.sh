#!/usr/bin/env bash
#
# Extends sweep_workers.sh with CPU-worker cells.
#
# The first sweep only varied MAX_WORKERS_PER_GPU (GPU workers), with
# --num-jobs matching --max-workers-per-gpu (no overflow CPU workers).
# This script appends cells where --num-jobs > max-workers-per-gpu, so
# the excess workers run training entirely on CPU.
#
# Matrix:
#   gpu_workers ∈ {0, 1, 2, 3}
#   cpu_workers ∈ {1, 2, 4}    (overflow workers, backend=cpu)
#   omp         ∈ {1, unset}
#
# Appends to the same $SWEEP_RESULTS_CSV so a single pick-winner pass
# covers both sweeps.
#
# Expects the first sweep to have already run (train_data.csv.bz2,
# hyperparameters.sweep.yaml present in $MHCFLURRY_OUT).
set -euo pipefail

: "${MHCFLURRY_OUT:?MHCFLURRY_OUT must be set}"
SWEEP_RESULTS_CSV="${SWEEP_RESULTS_CSV:-$MHCFLURRY_OUT/sweep_results.csv}"

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MHCFLURRY_TORCH_COMPILE=1

ALLELE_SEQUENCES=$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv
RANDOM_PRED=$(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2

cd "$MHCFLURRY_OUT"

GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')

# Write header only if CSV missing (should already exist from sweep 1)
if [ ! -f "$SWEEP_RESULTS_CSV" ]; then
    echo "max_workers_per_gpu,omp_threads,wall_seconds,items_per_hour,exit_code,cpu_workers,num_jobs" > "$SWEEP_RESULTS_CSV"
fi

for OMP in 1 0; do
    if [ "$OMP" = "0" ]; then
        omp_label="auto"
    else
        omp_label="$OMP"
    fi
    for GPU_WORKERS in 0 1 2 3; do
        for CPU_WORKERS in 1 2 4; do
            NUM_JOBS=$(( GPU_WORKERS + CPU_WORKERS ))
            if [ "$NUM_JOBS" -eq 0 ]; then
                continue
            fi
            EFFECTIVE_GPUS=$GPUS
            if [ "$GPU_WORKERS" -eq 0 ]; then
                EFFECTIVE_GPUS=0
            fi

            CELL_DIR="$MHCFLURRY_OUT/cell_g${GPU_WORKERS}_c${CPU_WORKERS}_omp${omp_label}"
            mkdir -p "$CELL_DIR"
            cd "$CELL_DIR"

            export MAX_WORKERS_PER_GPU="$GPU_WORKERS"
            if [ "$OMP" = "0" ]; then
                unset OMP_NUM_THREADS 2>/dev/null || true
            else
                export OMP_NUM_THREADS="$OMP"
            fi

            echo ""
            echo "=== CELL: gpu=$GPU_WORKERS cpu=$CPU_WORKERS num_jobs=$NUM_JOBS omp=$omp_label ==="

            set +e
            start=$(date +%s)
            mhcflurry-class1-train-pan-allele-models \
                --data "$MHCFLURRY_OUT/train_data.csv.bz2" \
                --allele-sequences "$ALLELE_SEQUENCES" \
                --pretrain-data "$RANDOM_PRED" \
                --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
                --num-folds 4 \
                --hyperparameters "$MHCFLURRY_OUT/hyperparameters.sweep.yaml" \
                --out-models-dir "$CELL_DIR/models" \
                --worker-log-dir "$CELL_DIR" \
                --use-encoding-cache \
                --num-jobs "$NUM_JOBS" \
                --max-tasks-per-worker 1 \
                --gpus "$EFFECTIVE_GPUS" \
                --max-workers-per-gpu "$GPU_WORKERS" \
                > "$CELL_DIR/stdout.log" 2> "$CELL_DIR/stderr.log"
            exit_code=$?
            end=$(date +%s)
            set -e

            wall=$(( end - start ))
            if [ "$wall" -gt 0 ] && [ "$exit_code" -eq 0 ]; then
                iph=$(awk "BEGIN {printf \"%.2f\", 4.0 / ($wall / 3600.0)}")
            else
                iph="0"
            fi

            echo "=== CELL done: gpu=$GPU_WORKERS cpu=$CPU_WORKERS omp=$omp_label wall=${wall}s items/hr=${iph} exit=${exit_code} ==="
            echo "$GPU_WORKERS,$omp_label,$wall,$iph,$exit_code,$CPU_WORKERS,$NUM_JOBS" >> "$SWEEP_RESULTS_CSV"

            cd "$MHCFLURRY_OUT"
        done
    done
done

echo ""
echo "=== CPU-EXTENSION SWEEP COMPLETE ==="
cat "$SWEEP_RESULTS_CSV"

python - <<PY
import csv
rows = []
with open("$SWEEP_RESULTS_CSV") as f:
    for row in csv.DictReader(f):
        if row["exit_code"] == "0" and float(row["items_per_hour"]) > 0:
            rows.append(row)
if not rows:
    print("NO WINNER")
    raise SystemExit(1)
rows.sort(key=lambda r: float(r["items_per_hour"]), reverse=True)
print("\nTop 5 configurations:")
for r in rows[:5]:
    gpu = r["max_workers_per_gpu"]
    cpu = r.get("cpu_workers", "0") or "0"
    omp = r["omp_threads"]
    iph = r["items_per_hour"]
    wall = r["wall_seconds"]
    print(f"  gpu={gpu} cpu={cpu} omp={omp}  items/hr={iph}  wall={wall}s")
best = rows[0]
print(f"\nWINNER: gpu={best['max_workers_per_gpu']} cpu={best.get('cpu_workers','0')} omp={best['omp_threads']} items/hr={best['items_per_hour']}")
PY
