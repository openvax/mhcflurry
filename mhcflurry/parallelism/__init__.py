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

"""Local parallelism helpers.

This package contains the implementation formerly held in
``mhcflurry.local_parallelism``. The old module remains as an import
compatibility shim.
"""

from importlib import import_module

_SUBMODULES = ("cli_args", "planning", "torch_compile", "worker_pool", "worker_runtime")
_EXPORTS = {
    name: module
    for module, names in {
        "cli_args": (
            "add_local_parallelism_args",
            "add_prediction_parallelism_args",
        ),
        "planning": (
            "apply_dataloader_num_workers_to_work_items",
            "apply_random_negative_pool_epochs_to_work_items",
            "apply_resolved_training_hyperparameters_to_work_items",
            "auto_dataloader_num_workers",
            "auto_max_workers_per_gpu",
            "auto_num_jobs",
            "auto_random_negative_pool_epochs",
            "detect_free_vram_per_gpu_gb",
            "free_vram_per_gpu_from_nvidia_smi_gb",
            "num_workers_per_gpu_from_args",
            "refresh_device_memory_budget",
            "refine_local_parallelism_from_spawn_context",
            "refine_local_parallelism_from_warmup",
            "resolve_cpu_thread_budget",
            "resolve_cpu_threads_per_worker",
            "resolve_dataloader_num_workers",
            "resolve_local_parallelism_args",
            "resolve_max_workers_per_gpu",
        ),
        "torch_compile": (
            "configure_cluster_worker_torch_compile_threads",
            "hoist_torchinductor_compile_threads",
            "resolve_torchinductor_compile_threads_env",
            "run_single_worker_resource_probe",
            "run_single_worker_torch_compile_warmup",
        ),
        "worker_pool": (
            "NonDaemonContext",
            "NonDaemonPool",
            "NonDaemonProcess",
            "NonDaemonSpawnContext",
            "NonDaemonSpawnProcess",
            "attach_constant_data_to_work_items_if_needed",
            "chunk_ranges_for_local_parallelism",
            "estimate_worker_context_bytes",
            "refine_local_parallelism_from_worker_context",
            "make_worker_pool",
            "non_daemon_context",
            "validate_worker_pool_args",
            "worker_pool_uses_fork",
            "worker_pool_with_gpu_assignments",
            "worker_pool_with_gpu_assignments_from_args",
            "worker_init_kwargs_for_scheduler",
        ),
        "worker_runtime": (
            "WrapException",
            "call_wrapped",
            "call_wrapped_kwargs",
            "worker_init",
            "worker_init_entry_point",
        ),
    }.items()
    for name in names
}

__all__ = sorted((*_SUBMODULES, *_EXPORTS))


def __getattr__(name):
    """Import worker/runtime helpers only when callers request them."""
    if name in _SUBMODULES:
        value = import_module("." + name, __name__)
    elif name in _EXPORTS:
        value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    else:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
