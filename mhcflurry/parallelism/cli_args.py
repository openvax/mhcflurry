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

"""Argument-parser helpers for local parallelism."""


def _max_workers_per_gpu_arg(value):
    """argparse type for ``--max-workers-per-gpu``. Accepts ``"auto"`` or int>=1."""
    import argparse
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "--max-workers-per-gpu must be 'auto' or an integer >= 1, got %r"
            % (value,)
        )
    if v < 1:
        raise argparse.ArgumentTypeError(
            "--max-workers-per-gpu must be >= 1, got %d" % (v,)
        )
    return v


def _num_jobs_arg(value):
    """argparse type for ``--num-jobs``. Accepts ``"auto"`` or int>=0."""
    import argparse
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "--num-jobs must be 'auto' or an integer >= 0, got %r" % (value,)
        )
    if v < 0:
        raise argparse.ArgumentTypeError(
            "--num-jobs must be >= 0, got %d" % (v,)
        )
    return v


def _dataloader_num_workers_arg(value):
    """argparse type for ``--dataloader-num-workers``. ``"auto"`` or int>=0."""
    import argparse
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "--dataloader-num-workers must be 'auto' or an integer >= 0, "
            "got %r" % (value,)
        )
    if v < 0:
        raise argparse.ArgumentTypeError(
            "--dataloader-num-workers must be >= 0, got %d" % (v,)
        )
    return v


def _random_negative_pool_epochs_arg(value):
    """argparse type for ``--random-negative-pool-epochs``. ``"auto"`` or int>=1."""
    import argparse
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "--random-negative-pool-epochs must be 'auto' or an integer >= 1, "
            "got %r" % (value,)
        )
    if v < 1:
        raise argparse.ArgumentTypeError(
            "--random-negative-pool-epochs must be >= 1, got %d" % (v,)
        )
    return v

def add_local_parallelism_args(parser):
    """
    Add local parallelism arguments to the given argparse.ArgumentParser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    group = parser.add_argument_group("Local parallelism")

    group.add_argument(
        "--num-jobs",
        default="auto",
        type=_num_jobs_arg,
        metavar="N",
        help="Number of local processes to parallelize training over. "
             "Pass 'auto' (default) to derive from "
             "``--gpus * --max-workers-per-gpu`` once the latter is "
             "resolved (so workers never overflow to CPU silently). "
             "Pass 0 for serial run, or an int to pin.")
    group.add_argument(
        "--backend",
        choices=("auto", "default", "gpu", "mps", "cpu"),
        default="auto",
        help="Device backend. 'default' is a legacy alias for 'auto'. 'gpu' "
             "means CUDA. 'auto' (default) selects the "
             "best available device: GPU > MPS > CPU. When --gpus is set, "
             "GPU-assigned workers use CUDA and overflow workers are forced "
             "to CPU.")
    group.add_argument(
        "--gpus",
        type=int,
        metavar="N",
        help="Number of CUDA GPUs to assign across parallel workers. When "
             "CUDA_VISIBLE_DEVICES is set, this is a count within that "
             "scheduler-visible mask. Requires --num-jobs > 0. Each assigned "
             "worker gets one GPU; workers beyond "
             "--gpus * --max-workers-per-gpu run on CPU.")
    group.add_argument(
        "--max-workers-per-gpu",
        type=_max_workers_per_gpu_arg,
        metavar="N",
        default="auto",
        help="Maximum number of workers to assign to a GPU. Pass 'auto' "
             "(default) to pick a value based on detected free VRAM, the "
             "per-worker VRAM upper bound, and a 4-worker hard cap "
             "(see ``auto_max_workers_per_gpu``). Pass an integer to pin. "
             "Workers beyond ``--gpus * --max-workers-per-gpu`` run on CPU.")
    group.add_argument(
        "--max-tasks-per-worker",
        type=int,
        metavar="N",
        default=None,
        help="Restart workers after N tasks. Workaround for memory leaks.")
    group.add_argument(
        "--worker-log-dir",
        default=None,
        help="Write worker stdout and stderr logs to given directory.")
    group.add_argument(
        "--dataloader-num-workers",
        type=_dataloader_num_workers_arg,
        metavar="N",
        default="auto",
        help="Per-fit-worker DataLoader child count for streaming "
             "pretraining. Pass "
             "'auto' (default) to derive from box vCPUs / RAM / "
             "fit-worker plan via "
             "``mhcflurry.parallelism.auto_dataloader_num_workers`` "
             "(empirical hard cap = 4). Pass an integer to pin (0 builds "
             "pretraining batches in-process). Overrides any "
             "``dataloader_num_workers`` set in component-model "
             "hyperparameters when applicable; non-affinity train commands "
             "accept the flag for uniformity but currently no-op.")
    group.add_argument(
        "--random-negative-pool-epochs",
        type=_random_negative_pool_epochs_arg,
        metavar="N",
        default="auto",
        help="Number of consecutive epochs that share a pre-encoded "
             "random-negative pool. Pass 'auto' (default) to size from "
             "system RAM / fit-worker plan via "
             "``mhcflurry.parallelism.auto_random_negative_pool_epochs`` "
             "(hard cap = 10). Pass an integer to pin (1 means fresh "
             "random negatives every epoch). Overrides any "
             "``random_negative_pool_epochs`` set in component-model "
             "hyperparameters.")
    group.add_argument(
        "--torch-compile",
        choices=("auto", "0", "1"),
        default="auto",
        help="Enable torch.compile for forward kernels. '1' on, '0' off, "
             "'auto' (default) reads MHCFLURRY_TORCH_COMPILE env (off when "
             "unset). When on, the orchestrator also auto-sizes "
             "TORCHINDUCTOR_COMPILE_THREADS — see "
             "hoist_torchinductor_compile_threads.")
    group.add_argument(
        "--torch-compile-loss",
        choices=("auto", "0", "1"),
        default="auto",
        help="Enable torch.compile for training loss modules. 'auto' "
             "(default) reads MHCFLURRY_TORCH_COMPILE_LOSS env; when unset, "
             "loss compilation defaults on inside maybe_compile_loss. CUDA "
             "workers run a one-op autograd warmup before compiling losses to "
             "avoid the PyTorch 2.4 / Triton invalid-device-context bug.")
    group.add_argument(
        "--matmul-precision",
        choices=("none", "highest", "high", "medium"),
        default="none",
        help="torch.set_float32_matmul_precision setting + cudnn.benchmark "
             "enable. 'highest' keeps full fp32 numerics with cudnn "
             "auto-tuning; 'high'/'medium' enable TF32 on Ampere+ (~2x "
             "matmul speedup, fp32 accumulation preserved, input-mantissa "
             "truncated). 'none' (default) leaves PyTorch's default "
             "untouched. CPU/MPS: no-op.")
    group.add_argument(
        "--enable-timing",
        action="store_true",
        default=False,
        help="Populate per-epoch timing arrays in fit_info "
             "(epoch_fetch_time, epoch_train_time, epoch_validation_time). "
             "Persisted in the model's config_json for post-hoc breakdown. "
             "No runtime cost beyond a few timestamp records per epoch.")


def add_prediction_parallelism_args(parser):
    """
    Add prediction-time local parallelism arguments to an argparse parser.

    This is the inference subset of ``add_local_parallelism_args``: the worker
    scheduler, backend selection, and torch forward-kernel knobs, without
    training-only DataLoader/random-negative options.
    """
    group = parser.add_argument_group("Prediction parallelism")

    group.add_argument(
        "--num-jobs",
        default="auto",
        type=_num_jobs_arg,
        metavar="N",
        help="Number of local prediction worker processes. Pass 'auto' "
             "(default) to use ``--gpus * --max-workers-per-gpu`` when "
             "CUDA GPUs are specified, otherwise run serially. Pass 0 for "
             "serial prediction.")
    group.add_argument(
        "--backend",
        choices=("auto", "default", "gpu", "mps", "cpu"),
        default="auto",
        help="Device backend. 'auto' (default) selects GPU > MPS > CPU. "
             "When --gpus is set, GPU-assigned workers use CUDA.")
    group.add_argument(
        "--gpus",
        type=int,
        metavar="N",
        help="Number of CUDA GPUs to assign across parallel prediction "
             "workers. When CUDA_VISIBLE_DEVICES is set, this is a count "
             "within that scheduler-visible mask. Requires --num-jobs > 0.")
    group.add_argument(
        "--max-workers-per-gpu",
        type=_max_workers_per_gpu_arg,
        metavar="N",
        default="auto",
        help="Maximum prediction workers to assign to each CUDA GPU. Pass "
             "'auto' (default) to choose from detected free VRAM, or an int "
             "to pin.")
    group.add_argument(
        "--max-tasks-per-worker",
        type=int,
        metavar="N",
        default=None,
        help="Restart workers after N prediction chunks.")
    group.add_argument(
        "--worker-log-dir",
        default=None,
        help="Write prediction worker stdout and stderr logs to this directory.")
    group.add_argument(
        "--torch-compile",
        choices=("auto", "0", "1"),
        default="auto",
        help="Enable torch.compile for forward kernels. 'auto' reads "
             "MHCFLURRY_TORCH_COMPILE.")
    group.add_argument(
        "--matmul-precision",
        choices=("none", "highest", "high", "medium"),
        default="none",
        help="torch.set_float32_matmul_precision setting. CPU/MPS: no-op.")
