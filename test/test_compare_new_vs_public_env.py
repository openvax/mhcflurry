"""Tests for ``scripts/training/compare_new_vs_public.py``."""
import argparse
import importlib.util
import pathlib

import numpy

from mhcflurry.workload_planning import WORKLOAD_AFFINITY_INFERENCE


def _load_module():
    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "scripts" / "training" / "compare_new_vs_public.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_new_vs_public", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args():
    return argparse.Namespace(
        num_jobs="auto",
        gpus=None,
        max_workers_per_gpu="auto",
        backend="auto",
        max_tasks_per_worker=None,
        worker_log_dir=None,
        torch_compile="auto",
        matmul_precision="none",
    )


def test_parallel_predict_uses_shared_worker_pool(monkeypatch):
    mod = _load_module()
    calls = {}

    class FakePool:
        def imap_unordered(self, fn, work_items, chunksize=1):
            del fn
            calls["work_items"] = list(work_items)
            calls["chunksize"] = chunksize
            return [
                (item["chunk_num"], numpy.asarray([item["chunk_num"]]))
                for item in reversed(work_items)
            ]

        def close(self):
            calls["closed"] = True

        def join(self):
            calls["joined"] = True

    def fake_worker_pool(args, workload_name, workload_hints, start_method=None):
        calls["workload_name"] = workload_name
        calls["workload_hints"] = workload_hints
        calls["start_method"] = start_method
        args.num_jobs = 2
        args.gpus = 2
        args.max_workers_per_gpu = 1
        args.backend = "gpu"
        return FakePool()

    monkeypatch.setattr(
        mod,
        "worker_pool_with_gpu_assignments_from_args",
        fake_worker_pool)

    predictions = mod.parallel_predict(
        _args(),
        predictor_dir="/models",
        peptides=numpy.asarray(["A", "B", "C", "D"]),
        alleles=numpy.asarray(["HLA-A*02:01"] * 4),
    )

    assert calls["workload_name"] == WORKLOAD_AFFINITY_INFERENCE
    assert calls["workload_hints"] == {"prediction_rows": 4}
    assert calls["start_method"] == "spawn"
    assert calls["chunksize"] == 1
    assert calls["closed"] is True
    assert calls["joined"] is True
    assert [item["chunk_num"] for item in calls["work_items"]] == [0, 1, 2, 3]
    assert predictions.tolist() == [0, 1, 2, 3]


def test_parallel_predict_serial_path_uses_configured_process(monkeypatch):
    mod = _load_module()
    calls = {}

    def fake_worker_pool(args, workload_name, workload_hints, start_method=None):
        del workload_name, workload_hints, start_method
        args.num_jobs = 0
        args.gpus = 0
        args.max_workers_per_gpu = 1
        args.backend = "cpu"
        return None

    def fake_predict_chunk(predictor_dir, peptides, alleles, chunk_num):
        calls["predictor_dir"] = predictor_dir
        calls["peptides"] = peptides.tolist()
        calls["alleles"] = alleles.tolist()
        calls["chunk_num"] = chunk_num
        return chunk_num, numpy.asarray([10.0, 20.0])

    monkeypatch.setattr(
        mod,
        "worker_pool_with_gpu_assignments_from_args",
        fake_worker_pool)
    monkeypatch.setattr(mod, "_predict_chunk", fake_predict_chunk)

    predictions = mod.parallel_predict(
        _args(),
        predictor_dir="/models",
        peptides=numpy.asarray(["A", "B"]),
        alleles=numpy.asarray(["HLA-A*02:01", "HLA-B*07:02"]),
    )

    assert calls == {
        "predictor_dir": "/models",
        "peptides": ["A", "B"],
        "alleles": ["HLA-A*02:01", "HLA-B*07:02"],
        "chunk_num": 0,
    }
    assert predictions.tolist() == [10.0, 20.0]
