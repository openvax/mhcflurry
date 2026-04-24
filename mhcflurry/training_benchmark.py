"""Training benchmark / profiling harness for pan-allele runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy
import pandas


TIMING_MARKER_RE = re.compile(r"TIMING_MARKER\s+(\S+)\s+([0-9]+(?:\.[0-9]+)?)")
PROCESS_TELEMETRY_RE = re.compile(
    r"PROCESS_TELEMETRY pid=(\d+) marker=(\S+) rss_mb=([^\s]+) num_fds=([^\s]+)"
)
GPU_TELEMETRY_RE = re.compile(
    r"GPU_MEMORY_TELEMETRY pid=(\d+) task=(\S+) marker=(\S+) "
    r"allocated_gb=([0-9.]+) reserved_gb=([0-9.]+) max_allocated_gb=([0-9.]+)"
)


def _scalar(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return float(value[0])
    if isinstance(value, (int, float, numpy.integer, numpy.floating)):
        return float(value)
    return default


def _series(info, key):
    value = info.get(key, [])
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, tuple):
        return [float(v) for v in value]
    if isinstance(value, (int, float, numpy.integer, numpy.floating)):
        return [float(value)]
    return []


def _sum_series(info, key):
    return float(sum(_series(info, key)))


def _safe_div(numerator, denominator):
    if not denominator:
        return None
    return numerator / denominator


def _flatten_fit_records(models_dir: str | os.PathLike[str]):
    manifest_path = Path(models_dir) / "manifest.csv"
    manifest_df = pandas.read_csv(manifest_path)
    records = []
    for _, row in manifest_df.iterrows():
        config = json.loads(row["config_json"])
        fit_infos = config.get("fit_info", []) or []
        for fit_index, info in enumerate(fit_infos):
            training_info = info.get("training_info", {}) or {}
            records.append(
                {
                    "model_name": row["model_name"],
                    "fit_index": fit_index,
                    "phase": training_info.get("phase", "unknown"),
                    "info": info,
                    "training_info": training_info,
                }
            )
    return records


def _summarize_fit_records(records):
    if not records:
        return {
            "num_fits": 0,
            "total_time_s": 0.0,
        }

    total_time = sum(_scalar(record["info"].get("time")) for record in records)
    total_epochs = sum(len(record["info"].get("loss", [])) for record in records)
    total_train_time = sum(_sum_series(record["info"], "epoch_train_time") for record in records)
    total_fetch_time = sum(_sum_series(record["info"], "epoch_fetch_time") for record in records)
    total_h2d_time = sum(_sum_series(record["info"], "epoch_h2d_time") for record in records)
    total_loss_sync_time = sum(
        _sum_series(record["info"], "epoch_loss_sync_time") for record in records
    )
    total_input_build_time = sum(
        _sum_series(record["info"], "epoch_input_build_time") for record in records
    )
    total_initialization_time = sum(
        _sum_series(record["info"], "epoch_initialization_time") for record in records
    )
    total_shuffle_time = sum(
        _sum_series(record["info"], "epoch_shuffle_dataset_time") for record in records
    )
    total_dataloader_setup_time = sum(
        _sum_series(record["info"], "epoch_dataloader_setup_time") for record in records
    )
    total_validation_materialize_time = sum(
        _sum_series(record["info"], "epoch_validation_materialize_time")
        for record in records
    )
    total_validation_compute_time = sum(
        _sum_series(record["info"], "epoch_validation_time") for record in records
    )
    total_callback_time = sum(
        _sum_series(record["info"], "epoch_callback_time") for record in records
    )
    total_gc_time = sum(_sum_series(record["info"], "epoch_gc_time") for record in records)
    total_epoch_time = sum(_sum_series(record["info"], "epoch_total_time") for record in records)
    total_train_batches = int(
        sum(sum(record["info"].get("epoch_num_train_batches", [])) for record in records)
    )
    total_train_rows = int(
        sum(sum(record["info"].get("epoch_num_train_rows", [])) for record in records)
    )
    total_tail_rows = int(
        sum(sum(record["info"].get("epoch_tail_train_rows", [])) for record in records)
    )
    total_validation_batches = int(
        sum(sum(record["info"].get("epoch_num_validation_batches", [])) for record in records)
    )

    compile_overhead_estimates = []
    first_batch_times = []
    for record in records:
        info = record["info"]
        first_batch_time = info.get("first_batch_time")
        epoch_train_times = _series(info, "epoch_train_time")
        epoch_train_batches = info.get("epoch_num_train_batches", [])
        if first_batch_time is None:
            continue
        first_batch_times.append(float(first_batch_time))
        total_batches = int(sum(epoch_train_batches)) if epoch_train_batches else 0
        if total_batches <= 1 or not epoch_train_times:
            continue
        steady_batch_time = _safe_div(sum(epoch_train_times) - float(first_batch_time), total_batches - 1)
        if steady_batch_time is None:
            continue
        compile_overhead_estimates.append(max(0.0, float(first_batch_time) - steady_batch_time))

    validation_rows_total = 0
    effective_validation_batch_sizes = []
    validation_cache_reused_count = 0
    dataloader_num_workers = []
    iterator_setup_time = 0.0
    for record in records:
        info = record["info"]
        epochs = len(info.get("loss", []))
        validation_rows = int(_scalar(info.get("validation_rows"), default=0))
        validation_rows_total += validation_rows * epochs
        batch_size = info.get("effective_validation_batch_size")
        if batch_size is not None:
            effective_validation_batch_sizes.append(int(batch_size))
        if info.get("validation_cache_reused"):
            validation_cache_reused_count += 1
        dataloader_num_workers.append(int(_scalar(info.get("dataloader_num_workers"), default=0)))
        iterator_setup_time += _scalar(info.get("iterator_setup_time"), default=0.0)

    validation_total_time = total_validation_materialize_time + total_validation_compute_time
    return {
        "num_fits": len(records),
        "total_time_s": total_time,
        "total_epochs": total_epochs,
        "total_epoch_timing_s": total_epoch_time,
        "train_time_s": total_train_time,
        "fetch_time_s": total_fetch_time,
        "h2d_time_s": total_h2d_time,
        "loss_sync_time_s": total_loss_sync_time,
        "input_build_time_s": total_input_build_time,
        "initialization_time_s": total_initialization_time,
        "shuffle_time_s": total_shuffle_time,
        "dataloader_setup_time_s": total_dataloader_setup_time,
        "iterator_setup_time_s": iterator_setup_time,
        "validation_materialize_time_s": total_validation_materialize_time,
        "validation_compute_time_s": total_validation_compute_time,
        "validation_total_time_s": validation_total_time,
        "callback_time_s": total_callback_time,
        "gc_time_s": total_gc_time,
        "train_time_fraction_of_epoch_timing": _safe_div(total_train_time, total_epoch_time),
        "validation_time_fraction_of_epoch_timing": _safe_div(validation_total_time, total_epoch_time),
        "fetch_time_fraction_of_epoch_timing": _safe_div(total_fetch_time, total_epoch_time),
        "input_build_fraction_of_epoch_timing": _safe_div(total_input_build_time, total_epoch_time),
        "h2d_time_fraction_of_epoch_timing": _safe_div(total_h2d_time, total_epoch_time),
        "loss_sync_time_fraction_of_epoch_timing": _safe_div(
            total_loss_sync_time, total_epoch_time
        ),
        "total_train_batches": total_train_batches,
        "total_train_rows": total_train_rows,
        "total_tail_train_rows": total_tail_rows,
        "total_validation_batches": total_validation_batches,
        "validation_rows_seen": validation_rows_total,
        "train_rows_per_second": _safe_div(total_train_rows, total_train_time),
        "validation_rows_per_second": _safe_div(validation_rows_total, validation_total_time),
        "steady_state_train_batch_time_s": _safe_div(
            total_train_time - sum(first_batch_times),
            total_train_batches - len(first_batch_times),
        ) if total_train_batches > len(first_batch_times) else None,
        "first_batch_time_s_median": float(numpy.median(first_batch_times))
        if first_batch_times else None,
        "first_batch_time_s_max": max(first_batch_times) if first_batch_times else None,
        "compile_warmup_overhead_s_sum": float(sum(compile_overhead_estimates)),
        "compile_warmup_overhead_s_median": float(numpy.median(compile_overhead_estimates))
        if compile_overhead_estimates else None,
        "compile_warmup_overhead_s_max": max(compile_overhead_estimates)
        if compile_overhead_estimates else None,
        "effective_validation_batch_sizes": sorted(set(effective_validation_batch_sizes)),
        "validation_cache_reused_fit_count": validation_cache_reused_count,
        "dataloader_num_workers": sorted(set(dataloader_num_workers)),
    }


def _parse_log_file(log_path: str | os.PathLike[str] | None):
    if log_path is None:
        return {
            "path": None,
            "timing_markers": {},
            "process_telemetry": [],
            "gpu_telemetry": [],
        }

    path = Path(log_path)
    markers = {}
    process_telemetry = []
    gpu_telemetry = []
    with path.open() as fd:
        for line in fd:
            match = TIMING_MARKER_RE.search(line)
            if match:
                markers[match.group(1)] = float(match.group(2))
            match = PROCESS_TELEMETRY_RE.search(line)
            if match:
                rss_value = None if match.group(3) == "na" else float(match.group(3))
                fd_value = None if match.group(4) == "na" else int(match.group(4))
                process_telemetry.append(
                    {
                        "pid": int(match.group(1)),
                        "marker": match.group(2),
                        "rss_mb": rss_value,
                        "num_fds": fd_value,
                    }
                )
            match = GPU_TELEMETRY_RE.search(line)
            if match:
                gpu_telemetry.append(
                    {
                        "pid": int(match.group(1)),
                        "task": match.group(2),
                        "marker": match.group(3),
                        "allocated_gb": float(match.group(4)),
                        "reserved_gb": float(match.group(5)),
                        "max_allocated_gb": float(match.group(6)),
                    }
                )
    return {
        "path": str(path),
        "timing_markers": markers,
        "process_telemetry": process_telemetry,
        "gpu_telemetry": gpu_telemetry,
    }


def _summarize_telemetry(parsed_log):
    process_entries = parsed_log["process_telemetry"]
    gpu_entries = parsed_log["gpu_telemetry"]

    process_by_pid = {}
    for entry in process_entries:
        pid_entries = process_by_pid.setdefault(entry["pid"], {})
        pid_entries[entry["marker"]] = entry

    rss_growth = []
    fd_growth = []
    for pid, markers in process_by_pid.items():
        start_entry = markers.get("START")
        end_entry = markers.get("END")
        if start_entry and end_entry:
            if start_entry["rss_mb"] is not None and end_entry["rss_mb"] is not None:
                rss_growth.append(
                    {
                        "pid": pid,
                        "rss_mb_delta": end_entry["rss_mb"] - start_entry["rss_mb"],
                    }
                )
            if start_entry["num_fds"] is not None and end_entry["num_fds"] is not None:
                fd_growth.append(
                    {
                        "pid": pid,
                        "fd_delta": end_entry["num_fds"] - start_entry["num_fds"],
                    }
                )

    max_gpu_reserved = max((entry["reserved_gb"] for entry in gpu_entries), default=None)
    max_gpu_allocated = max((entry["allocated_gb"] for entry in gpu_entries), default=None)
    max_gpu_highwater = max((entry["max_allocated_gb"] for entry in gpu_entries), default=None)
    max_rss = max(
        (entry["rss_mb"] for entry in process_entries if entry["rss_mb"] is not None),
        default=None,
    )
    max_fds = max(
        (entry["num_fds"] for entry in process_entries if entry["num_fds"] is not None),
        default=None,
    )

    return {
        "num_process_samples": len(process_entries),
        "num_gpu_samples": len(gpu_entries),
        "max_rss_mb": max_rss,
        "max_num_fds": max_fds,
        "max_gpu_reserved_gb": max_gpu_reserved,
        "max_gpu_allocated_gb": max_gpu_allocated,
        "max_gpu_highwater_gb": max_gpu_highwater,
        "rss_growth_by_pid": rss_growth,
        "fd_growth_by_pid": fd_growth,
    }


def _derive_top_level_wall_times(train_log, selection_log):
    train_markers = train_log["timing_markers"]
    selection_markers = selection_log["timing_markers"]
    return {
        "data_load_s": _safe_div(
            train_markers.get("data_loaded", 0.0) - train_markers.get("start", 0.0),
            1.0,
        ) if "start" in train_markers and "data_loaded" in train_markers else None,
        "setup_s": _safe_div(
            train_markers.get("setup_done", 0.0) - train_markers.get("data_loaded", 0.0),
            1.0,
        ) if "data_loaded" in train_markers and "setup_done" in train_markers else None,
        "training_s": _safe_div(
            train_markers.get("training_done", 0.0) - train_markers.get("setup_done", 0.0),
            1.0,
        ) if "setup_done" in train_markers and "training_done" in train_markers else None,
        "selection_s": _safe_div(
            selection_markers.get("selection_done", 0.0) - selection_markers.get("selection_start", 0.0),
            1.0,
        ) if "selection_start" in selection_markers and "selection_done" in selection_markers else None,
    }


def analyze_training_run(
    models_dir: str | os.PathLike[str],
    *,
    train_log: str | os.PathLike[str] | None = None,
    selection_log: str | os.PathLike[str] | None = None,
):
    records = _flatten_fit_records(models_dir)
    phase_to_records = {}
    for record in records:
        phase_to_records.setdefault(record["phase"], []).append(record)

    parsed_train_log = _parse_log_file(train_log)
    parsed_selection_log = _parse_log_file(selection_log)
    summary = {
        "models_dir": str(models_dir),
        "num_models": len({record["model_name"] for record in records}),
        "num_fit_records": len(records),
        "top_level_wall_times_s": _derive_top_level_wall_times(
            parsed_train_log, parsed_selection_log
        ),
        "train_log": parsed_train_log,
        "selection_log": parsed_selection_log,
        "telemetry": {
            "train": _summarize_telemetry(parsed_train_log),
            "selection": _summarize_telemetry(parsed_selection_log),
        },
        "phase_summaries": {
            phase: _summarize_fit_records(phase_records)
            for phase, phase_records in sorted(phase_to_records.items())
        },
        "all_fit_summary": _summarize_fit_records(records),
    }
    return summary


def _write_json(result, out_json):
    payload = json.dumps(result, indent=2, sort_keys=True)
    if out_json:
        Path(out_json).write_text(payload + "\n")
    print(payload)


def _run_command(command, *, log_path, cwd):
    args = shlex.split(command)
    env = os.environ.copy()
    env.setdefault("MHCFLURRY_ENABLE_TIMING", "1")
    with Path(log_path).open("w") as fd:
        subprocess.run(
            args,
            cwd=cwd,
            env=env,
            stdout=fd,
            stderr=subprocess.STDOUT,
            check=True,
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze an existing training/models directory and timing logs.",
    )
    analyze_parser.add_argument("--models-dir", required=True)
    analyze_parser.add_argument("--train-log")
    analyze_parser.add_argument("--selection-log")
    analyze_parser.add_argument("--out-json")

    run_parser = subparsers.add_parser(
        "run",
        help="Run profiled train/select commands and emit a JSON summary.",
    )
    run_parser.add_argument("--models-dir", required=True)
    run_parser.add_argument("--train-command", required=True)
    run_parser.add_argument("--selection-command")
    run_parser.add_argument("--cwd", default=".")
    run_parser.add_argument("--log-dir")
    run_parser.add_argument("--out-json")

    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        result = analyze_training_run(
            args.models_dir,
            train_log=args.train_log,
            selection_log=args.selection_log,
        )
        _write_json(result, args.out_json)
        return 0

    log_dir = Path(args.log_dir or (Path(args.models_dir) / "benchmark_profile_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    train_log = log_dir / "train.log"
    selection_log = log_dir / "select.log"

    _run_command(args.train_command, log_path=train_log, cwd=args.cwd)
    if args.selection_command:
        _run_command(args.selection_command, log_path=selection_log, cwd=args.cwd)
        selection_log_path = str(selection_log)
    else:
        selection_log_path = None

    result = analyze_training_run(
        args.models_dir,
        train_log=str(train_log),
        selection_log=selection_log_path,
    )
    _write_json(result, args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
