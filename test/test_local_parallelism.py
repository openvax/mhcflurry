from argparse import ArgumentParser

import pytest

from mhcflurry.local_parallelism import (
    add_local_parallelism_args,
    validate_worker_pool_args,
    worker_init_kwargs_for_scheduler,
)


def test_worker_init_kwargs_round_robin_across_gpus():
    assert worker_init_kwargs_for_scheduler(
        num_jobs=5,
        num_gpus=2,
        backend="auto",
        max_workers_per_gpu=2,
    ) == [
        {"backend": "gpu", "gpu_device_nums": [0]},
        {"backend": "gpu", "gpu_device_nums": [1]},
        {"backend": "gpu", "gpu_device_nums": [0]},
        {"backend": "gpu", "gpu_device_nums": [1]},
        {"backend": "cpu", "gpu_device_nums": []},
    ]


def test_worker_init_kwargs_without_gpu_scheduling_uses_backend():
    assert worker_init_kwargs_for_scheduler(
        num_jobs=3,
        num_gpus=0,
        backend="mps",
        max_workers_per_gpu=2,
    ) == [
        {"backend": "mps"},
        {"backend": "mps"},
        {"backend": "mps"},
    ]


def test_worker_init_kwargs_normalizes_default_backend_alias():
    assert worker_init_kwargs_for_scheduler(
        num_jobs=2,
        num_gpus=0,
        backend="default",
        max_workers_per_gpu=2,
    ) == [
        {"backend": "auto"},
        {"backend": "auto"},
    ]


def test_worker_init_kwargs_with_gpus_normalizes_default_backend_alias():
    assert worker_init_kwargs_for_scheduler(
        num_jobs=3,
        num_gpus=1,
        backend="default",
        max_workers_per_gpu=2,
    ) == [
        {"backend": "gpu", "gpu_device_nums": [0]},
        {"backend": "gpu", "gpu_device_nums": [0]},
        {"backend": "cpu", "gpu_device_nums": []},
    ]


def test_backend_default_alias_parses():
    parser = ArgumentParser()
    add_local_parallelism_args(parser)
    args = parser.parse_args(["--backend", "default"])
    assert args.backend == "default"


def test_validate_worker_pool_args_requires_parallelism_for_gpus():
    with pytest.raises(ValueError, match="num_jobs > 0"):
        validate_worker_pool_args(
            num_jobs=0,
            num_gpus=1,
            backend="auto",
            max_workers_per_gpu=1,
        )


def test_validate_worker_pool_args_rejects_non_cuda_backends_for_gpus():
    with pytest.raises(ValueError, match="backend 'auto' or 'gpu'"):
        validate_worker_pool_args(
            num_jobs=2,
            num_gpus=1,
            backend="mps",
            max_workers_per_gpu=1,
        )


def test_validate_worker_pool_args_rejects_invalid_backend():
    with pytest.raises(ValueError, match="Invalid backend"):
        validate_worker_pool_args(
            num_jobs=2,
            num_gpus=0,
            backend="gpuu",
            max_workers_per_gpu=1,
        )
