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

This is the maintained remote wrapper for pan_allele_release_full.sh. It uses
runplz as the transport, with Brev configuration when runplz is pointed at Brev
instances. Local runs should call the shell script directly.
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


APP_NAME = os.environ.get("RUNPLZ_APP_NAME", "mhcflurry-release-full")
GPU_TYPE = os.environ.get("RUNPLZ_GPU", "A100")
NUM_GPUS = int(os.environ.get("RUNPLZ_NUM_GPUS", "4"))
MIN_GPU_MEMORY = int(os.environ.get("RUNPLZ_MIN_GPU_MEMORY", "35"))
MIN_CPU = int(os.environ.get("RUNPLZ_MIN_CPU", "32"))
MIN_MEMORY = int(os.environ.get("RUNPLZ_MIN_MEMORY", "300"))
MIN_DISK = int(os.environ.get("RUNPLZ_MIN_DISK", "1000"))
DEFAULT_OUT = os.environ.get("MHCFLURRY_OUT", "/root/mhcflurry-release-run")


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
    brev_config=BrevConfig(
        auto_create_instances=False,
        mode="container",
        on_finish="leave",
        ssh_ready_wait_seconds=2400,
    ),
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
    env={
        "AFFINITY_MINIBATCH_SIZE": os.environ.get("AFFINITY_MINIBATCH_SIZE", "1024"),
        "DATALOADER_NUM_WORKERS": os.environ.get("DATALOADER_NUM_WORKERS", "auto"),
        "MAX_TASKS_PER_WORKER": os.environ.get("MAX_TASKS_PER_WORKER", "12"),
        "MAX_WORKERS_PER_GPU": os.environ.get("MAX_WORKERS_PER_GPU", "auto"),
        "MHCFLURRY_ENABLE_TIMING": os.environ.get("MHCFLURRY_ENABLE_TIMING", "1"),
        "MHCFLURRY_TORCH_COMPILE": os.environ.get("MHCFLURRY_TORCH_COMPILE", "1"),
        "MHCFLURRY_TORCH_COMPILE_LOSS": os.environ.get(
            "MHCFLURRY_TORCH_COMPILE_LOSS", "1"
        ),
        "MHCFLURRY_MATMUL_PRECISION": os.environ.get(
            "MHCFLURRY_MATMUL_PRECISION", "high"
        ),
        "MATMUL_PRECISION": os.environ.get("MATMUL_PRECISION", "high"),
        "MATMUL_PRECISION_CLI": os.environ.get("MATMUL_PRECISION_CLI", "high"),
        "PRESENTATION_PROCESSING_WITH_FLANKS_KIND": os.environ.get(
            "PRESENTATION_PROCESSING_WITH_FLANKS_KIND", "with_flanks"
        ),
        "PROCESSING_MINIBATCH_SIZE": os.environ.get(
            "PROCESSING_MINIBATCH_SIZE", "1024"
        ),
        "PROCESSING_VARIANTS": os.environ.get(
            "PROCESSING_VARIANTS", "with_flanks no_flank short_flanks"
        ),
        "TORCHINDUCTOR_COMPILE_THREADS": os.environ.get(
            "TORCHINDUCTOR_COMPILE_THREADS", "auto"
        ),
        "TRAINING_MINIBATCH_SIZE": os.environ.get("TRAINING_MINIBATCH_SIZE", "1024"),
    },
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
