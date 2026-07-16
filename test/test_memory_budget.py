import torch

from mhcflurry.memory_budget import (
    GIB,
    memory_reserve_bytes,
    memory_worker_capacity,
    module_tensor_bytes,
    per_worker_memory_budget_bytes,
    training_module_bytes,
)


def test_shared_memory_reserve_scales_but_has_a_floor():
    assert memory_reserve_bytes(40 * GIB) == 4 * GIB
    assert memory_reserve_bytes(4 * GIB) == 1 * GIB
    assert per_worker_memory_budget_bytes(40 * GIB, 2) == 18 * GIB


def test_memory_worker_capacity_has_no_default_performance_cap():
    assert memory_worker_capacity(80, 4) == 18
    assert memory_worker_capacity(40, 17) == 2


def test_module_and_training_state_bytes_are_shape_derived():
    module = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=True))
    module.register_buffer("extra", torch.zeros(5, dtype=torch.float32))
    parameter_bytes = sum(
        p.nelement() * p.element_size() for p in module.parameters())
    buffer_bytes = sum(
        b.nelement() * b.element_size() for b in module.buffers())
    assert module_tensor_bytes(module) == parameter_bytes + buffer_bytes
    assert training_module_bytes(
        module, optimizer_state_copies=2
    ) == parameter_bytes * 4 + buffer_bytes
