# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared memory-budget calculations for training and inference.

Launch-time worker packing and runtime batch sizing must use the same budget.
Otherwise each layer can independently claim the same free memory and a plan
that looks conservative on paper can still OOM. This module is deliberately
free of torch imports so the parent process can size a spawn/fork worker pool
without initializing CUDA.
"""

GIB = float(1 << 30)

# Keep one shared reserve for CUDA contexts, allocator fragmentation, and
# allocations racing between co-resident workers. The reserve is a small
# fraction on large accelerators but never disappears on small ones.
MEMORY_RESERVE_FRACTION = 0.10
DEVICE_MIN_RESERVE_BYTES = 1 << 30
HOST_MIN_RESERVE_BYTES = 2 << 30


def memory_reserve_bytes(
        available_bytes,
        *,
        min_reserve_bytes=DEVICE_MIN_RESERVE_BYTES,
        reserve_fraction=MEMORY_RESERVE_FRACTION):
    """Return the shared reserve to leave outside an automatic plan."""
    available_bytes = max(int(available_bytes), 0)
    if not available_bytes:
        return 0
    reserve = max(
        int(min_reserve_bytes),
        int(available_bytes * float(reserve_fraction)),
    )
    return min(reserve, available_bytes)


def usable_memory_bytes(
        available_bytes,
        *,
        min_reserve_bytes=DEVICE_MIN_RESERVE_BYTES,
        reserve_fraction=MEMORY_RESERVE_FRACTION):
    """Memory an automatic plan may spend after the shared reserve."""
    available_bytes = max(int(available_bytes), 0)
    return max(
        available_bytes - memory_reserve_bytes(
            available_bytes,
            min_reserve_bytes=min_reserve_bytes,
            reserve_fraction=reserve_fraction,
        ),
        0,
    )


def per_worker_memory_budget_bytes(
        available_bytes,
        num_workers,
        *,
        min_reserve_bytes=DEVICE_MIN_RESERVE_BYTES,
        reserve_fraction=MEMORY_RESERVE_FRACTION):
    """Split usable memory evenly between co-resident workers."""
    return usable_memory_bytes(
        available_bytes,
        min_reserve_bytes=min_reserve_bytes,
        reserve_fraction=reserve_fraction,
    ) // max(int(num_workers), 1)


def memory_worker_capacity(
        available_gb,
        per_worker_gb,
        *,
        min_reserve_bytes=DEVICE_MIN_RESERVE_BYTES,
        reserve_fraction=MEMORY_RESERVE_FRACTION):
    """Return how many complete worker working sets fit in available memory.

    At least one worker is returned: a caller with a GPU should attempt one
    worker and let its normal preflight produce the detailed undersized-device
    warning. There is intentionally no default performance hard cap here.
    """
    available_bytes = max(float(available_gb), 0.0) * GIB
    per_worker_bytes = max(float(per_worker_gb), 0.0) * GIB
    if per_worker_bytes <= 0:
        return 1
    return max(
        1,
        int(usable_memory_bytes(
            available_bytes,
            min_reserve_bytes=min_reserve_bytes,
            reserve_fraction=reserve_fraction,
        ) // per_worker_bytes),
    )


def module_tensor_bytes(module):
    """Return unique parameter and buffer bytes for a torch-like module."""
    seen = set()
    total = 0
    tensors = list(module.parameters(recurse=True))
    tensors.extend(module.buffers(recurse=True))
    for tensor in tensors:
        try:
            key = (str(tensor.device), int(tensor.data_ptr()))
        except RuntimeError:
            key = id(tensor)
        if key in seen:
            continue
        seen.add(key)
        total += int(tensor.nelement()) * int(tensor.element_size())
    return total


def training_module_bytes(module, optimizer_state_copies=2):
    """Estimate model, gradient, and optimizer-state bytes for training.

    Parameters and buffers are counted exactly. Trainable parameters also need
    one gradient tensor and, for Adam/RMSProp-like optimizers, one or more
    parameter-sized state tensors. The caller supplies that state count.
    """
    parameter_bytes = sum(
        int(parameter.nelement()) * int(parameter.element_size())
        for parameter in module.parameters(recurse=True)
    )
    buffer_bytes = sum(
        int(buffer.nelement()) * int(buffer.element_size())
        for buffer in module.buffers(recurse=True)
    )
    return (
        buffer_bytes
        + parameter_bytes * (2 + max(int(optimizer_state_copies), 0))
    )
