#!/usr/bin/env bash
#
# Auto-compute the per-training-worker CPU thread budget and set
# OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS uniformly.
#
# All three env vars control the same concern — per-process BLAS / OpenMP
# thread count — but they live in different libraries (Intel MKL,
# OpenBLAS, OpenMP runtime) so PyTorch + numpy + scikit-learn route to
# whichever happens to back the workload. Setting them independently
# is a footgun. This script gives one knob:
#
#   Inputs (env, all must be set before sourcing):
#     GPUS                        number of visible CUDA devices (0 allowed)
#     MAX_WORKERS_PER_GPU         training workers per GPU
#     DATALOADER_NUM_WORKERS      per-training-worker DataLoader prefetch procs
#
#   Optional override:
#     MHCFLURRY_CPU_THREADS_PER_WORKER
#         explicit per-training-worker thread count; skips auto-calc.
#
# Allocation model:
#   total_cores = nproc()
#   total_training_workers = max(1, GPUS * MAX_WORKERS_PER_GPU)
#   reserved = 2 (OS / orchestrator) + total_training_workers * DATALOADER_NUM_WORKERS
#   per_worker = max(1, (total_cores - reserved) / total_training_workers)
#
# Then exports OMP / MKL / OPENBLAS = per_worker, and prints a summary.
# User-set env vars are respected (only set if unset).
#
# Prints one human-readable summary line on stderr so logs show the
# decision and inputs that led to it.

set_cpu_threads() {
    local cores
    cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

    local gpus=${GPUS:-0}
    local wpg=${MAX_WORKERS_PER_GPU:-1}
    local dl=${DATALOADER_NUM_WORKERS:-0}

    local total_workers=$(( gpus * wpg ))
    if [ "$total_workers" -lt 1 ]; then
        total_workers=1
    fi

    local per_worker
    if [ -n "${MHCFLURRY_CPU_THREADS_PER_WORKER:-}" ]; then
        per_worker="$MHCFLURRY_CPU_THREADS_PER_WORKER"
    else
        # Reserve 2 cores for OS + orchestrator driver process, plus
        # one core per DataLoader worker process across all training
        # workers.
        local reserved=$(( 2 + total_workers * dl ))
        local available=$(( cores - reserved ))
        if [ "$available" -lt "$total_workers" ]; then
            # CPU is oversubscribed; give each training worker 1 thread.
            # This matches the "release" convention and avoids numpy
            # spawning 22 BLAS threads per process on a box that can't
            # schedule them all.
            per_worker=1
        else
            per_worker=$(( available / total_workers ))
            if [ "$per_worker" -lt 1 ]; then per_worker=1; fi
        fi
    fi

    # User overrides take precedence — only default if unset.
    : "${OMP_NUM_THREADS:=$per_worker}"
    : "${MKL_NUM_THREADS:=$per_worker}"
    : "${OPENBLAS_NUM_THREADS:=$per_worker}"
    export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

    printf >&2 \
        '[set_cpu_threads] cores=%d gpus=%d workers_per_gpu=%d dl_workers=%d → per_worker=%d (OMP=%s MKL=%s OPENBLAS=%s)\n' \
        "$cores" "$gpus" "$wpg" "$dl" "$per_worker" \
        "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS"
}
