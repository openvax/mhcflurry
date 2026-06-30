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

from . import cli_args as cli_args
from . import planning as planning
from . import torch_compile as torch_compile
from . import worker_pool as worker_pool
from . import worker_runtime as worker_runtime
from .cli_args import add_local_parallelism_args, add_prediction_parallelism_args
from .planning import (
    apply_dataloader_num_workers_to_work_items,
    apply_random_negative_pool_epochs_to_work_items,
    apply_resolved_training_hyperparameters_to_work_items,
    auto_dataloader_num_workers,
    auto_max_workers_per_gpu,
    auto_num_jobs,
    auto_random_negative_pool_epochs,
    detect_free_vram_per_gpu_gb,
    num_workers_per_gpu_from_args,
    resolve_dataloader_num_workers,
    resolve_local_parallelism_args,
    resolve_max_workers_per_gpu,
)
from .torch_compile import (
    configure_cluster_worker_torch_compile_threads,
    hoist_torchinductor_compile_threads,
    resolve_torchinductor_compile_threads_env,
    run_single_worker_torch_compile_warmup,
)
from .worker_pool import (
    NonDaemonContext,
    NonDaemonPool,
    NonDaemonProcess,
    NonDaemonSpawnContext,
    NonDaemonSpawnProcess,
    attach_constant_data_to_work_items_if_needed,
    chunk_ranges_for_local_parallelism,
    make_worker_pool,
    non_daemon_context,
    validate_worker_pool_args,
    worker_pool_uses_fork,
    worker_pool_with_gpu_assignments,
    worker_pool_with_gpu_assignments_from_args,
    worker_init_kwargs_for_scheduler,
)
from .worker_runtime import (
    WrapException,
    call_wrapped,
    call_wrapped_kwargs,
    worker_init,
    worker_init_entry_point,
)

__all__ = [
    "NonDaemonContext",
    "NonDaemonPool",
    "NonDaemonProcess",
    "NonDaemonSpawnContext",
    "NonDaemonSpawnProcess",
    "WrapException",
    "add_local_parallelism_args",
    "add_prediction_parallelism_args",
    "apply_dataloader_num_workers_to_work_items",
    "apply_random_negative_pool_epochs_to_work_items",
    "apply_resolved_training_hyperparameters_to_work_items",
    "attach_constant_data_to_work_items_if_needed",
    "auto_dataloader_num_workers",
    "auto_max_workers_per_gpu",
    "auto_num_jobs",
    "auto_random_negative_pool_epochs",
    "call_wrapped",
    "call_wrapped_kwargs",
    "chunk_ranges_for_local_parallelism",
    "cli_args",
    "configure_cluster_worker_torch_compile_threads",
    "detect_free_vram_per_gpu_gb",
    "hoist_torchinductor_compile_threads",
    "make_worker_pool",
    "non_daemon_context",
    "num_workers_per_gpu_from_args",
    "planning",
    "resolve_dataloader_num_workers",
    "resolve_local_parallelism_args",
    "resolve_max_workers_per_gpu",
    "resolve_torchinductor_compile_threads_env",
    "run_single_worker_torch_compile_warmup",
    "torch_compile",
    "validate_worker_pool_args",
    "worker_pool",
    "worker_init",
    "worker_init_entry_point",
    "worker_init_kwargs_for_scheduler",
    "worker_pool_uses_fork",
    "worker_pool_with_gpu_assignments",
    "worker_pool_with_gpu_assignments_from_args",
    "worker_runtime",
]
