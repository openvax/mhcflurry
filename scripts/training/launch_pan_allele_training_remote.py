#!/usr/bin/env python
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

"""
Launch pan-allele training on remote GPU machines.

This is the maintained runplz wrapper for pan_allele_release_full.sh. The
release workflow chooses whether Brev should use an existing instance or
intentionally provision one by setting RUNPLZ_BREV_* environment variables
before invoking ``runplz brev``. Local runs should call the shell script
directly.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

try:
    from runplz import App, Image
    from runplz.config import BrevConfig
except ImportError as e:
    raise SystemExit(
        "runplz is required for this launcher. Install it with "
        "`pip install runplz` or run pan_allele_release_full.sh locally."
    ) from e


APP_NAME = os.environ.get("RUNPLZ_APP_NAME", "mhcflurry-pan-allele-training")
GPU_TYPE = os.environ.get("RUNPLZ_GPU", "A100")
NUM_GPUS = int(os.environ.get("RUNPLZ_NUM_GPUS", "4"))
MIN_GPU_MEMORY = int(os.environ.get("RUNPLZ_MIN_GPU_MEMORY", "35"))
MIN_CPU = int(os.environ.get("RUNPLZ_MIN_CPU", "32"))
MIN_MEMORY = int(os.environ.get("RUNPLZ_MIN_MEMORY", "300"))
MIN_DISK = int(os.environ.get("RUNPLZ_MIN_DISK", "1000"))
DEFAULT_OUT = os.environ.get(
    "MHCFLURRY_OUT", "/root/mhcflurry-pan-allele-training-run"
)

TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def env_bool(environ, name, default=False):
    value = environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in TRUE_ENV_VALUES:
        return True
    if normalized in FALSE_ENV_VALUES:
        return False
    raise ValueError(
        "%s must be one of %s or %s; got %r" % (
            name,
            sorted(TRUE_ENV_VALUES),
            sorted(FALSE_ENV_VALUES),
            value,
        )
    )


def env_int_optional(environ, name):
    value = environ.get(name)
    if value is None or not value.strip():
        return None
    return int(value)


def env_csv_tuple(environ, name, default):
    value = environ.get(name)
    if value is None:
        return default
    if not value.strip():
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def brev_config_from_env(environ=os.environ):
    return BrevConfig(
        auto_create_instances=env_bool(
            environ, "RUNPLZ_BREV_AUTO_CREATE", default=False
        ),
        instance_type=(
            environ.get("RUNPLZ_BREV_INSTANCE_TYPE")
            or environ.get("BREV_INSTANCE_TYPE")
            or None
        ),
        mode=environ.get("RUNPLZ_BREV_MODE", "container"),
        on_finish=environ.get("RUNPLZ_BREV_ON_FINISH", "leave"),
        max_runtime_seconds=env_int_optional(
            environ, "RUNPLZ_BREV_MAX_RUNTIME_SECONDS"
        ),
        ssh_ready_wait_seconds=int(
            environ.get("RUNPLZ_BREV_SSH_READY_WAIT_SECONDS", "2400")
        ),
        instance_type_fallback_count=int(
            environ.get("RUNPLZ_BREV_INSTANCE_TYPE_FALLBACK_COUNT", "3")
        ),
        exclude_providers=env_csv_tuple(
            environ, "RUNPLZ_BREV_EXCLUDE_PROVIDERS", ("oci",)
        ),
    )


def remote_training_env(environ=os.environ):
    env = {
        "DATALOADER_NUM_WORKERS": environ.get("DATALOADER_NUM_WORKERS", "auto"),
        "MAX_TASKS_PER_WORKER": environ.get("MAX_TASKS_PER_WORKER", "12"),
        "MAX_WORKERS_PER_GPU": environ.get("MAX_WORKERS_PER_GPU", "auto"),
        # PyTorch/Inductor workers load GNU OpenMP (libgomp). The PyTorch
        # conda image also includes mkl-service, whose INTEL threading default
        # aborts when libgomp is already loaded.
        "MKL_THREADING_LAYER": environ.get("MKL_THREADING_LAYER", "GNU"),
        "MHCFLURRY_ENABLE_TIMING": environ.get("MHCFLURRY_ENABLE_TIMING", "1"),
        "MHCFLURRY_TORCH_COMPILE": environ.get("MHCFLURRY_TORCH_COMPILE", "1"),
        "MHCFLURRY_TORCH_COMPILE_LOSS": environ.get(
            "MHCFLURRY_TORCH_COMPILE_LOSS", "1"
        ),
        "MHCFLURRY_MATMUL_PRECISION": environ.get(
            "MHCFLURRY_MATMUL_PRECISION", "high"
        ),
        "MATMUL_PRECISION": environ.get("MATMUL_PRECISION", "high"),
        "MATMUL_PRECISION_CLI": environ.get("MATMUL_PRECISION_CLI", "high"),
        "PRESENTATION_PROCESSING_WITH_FLANKS_KIND": environ.get(
            "PRESENTATION_PROCESSING_WITH_FLANKS_KIND", "with_flanks"
        ),
        "PROCESSING_VARIANTS": environ.get(
            "PROCESSING_VARIANTS", "with_flanks no_flank short_flanks"
        ),
        "TORCHINDUCTOR_COMPILE_THREADS": environ.get(
            "TORCHINDUCTOR_COMPILE_THREADS", "auto"
        ),
        "TRAINING_MINIBATCH_SIZE": environ.get("TRAINING_MINIBATCH_SIZE", "1024"),
    }
    for name in ("AFFINITY_MINIBATCH_SIZE", "PROCESSING_MINIBATCH_SIZE"):
        if name in environ:
            env[name] = environ[name]
    return env


image = (
    Image.from_registry(
        os.environ.get(
            "RUNPLZ_IMAGE",
            "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        )
    )
    .apt_install(
        "python-is-python3",
        "bzip2",
        "wget",
        "rsync",
        "build-essential",
        "git",
        "libhdf5-dev",
        "libxml2-dev",
        "libxslt1-dev",
        "procps",
    )
    .pip_install("runplz>=3.11.0")
    .pip_install_local_dir(".", editable=True)
)

app = App(
    APP_NAME,
    brev_config=brev_config_from_env(),
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    num_gpus=NUM_GPUS,
    min_gpu_memory=MIN_GPU_MEMORY,
    min_cpu=MIN_CPU,
    min_memory=MIN_MEMORY,
    min_disk=MIN_DISK,
    timeout=60 * 60 * 24 * 14,
    env=remote_training_env(),
)
def train_release_full():
    """Run the maintained full release training script."""
    repo = Path.cwd()
    out = Path(
        os.environ.get("RUNPLZ_OUT")
        or os.environ.get("MHCFLURRY_OUT")
        or DEFAULT_OUT
    ).resolve()
    env = os.environ.copy()
    env.update({"MHCFLURRY_OUT": str(out), "REPO": str(repo)})
    subprocess.run(
        ["bash", "scripts/training/pan_allele_release_full.sh"],
        check=True,
        cwd=repo,
        env=env,
    )


@app.local_entrypoint()
def main():
    train_release_full.remote()
