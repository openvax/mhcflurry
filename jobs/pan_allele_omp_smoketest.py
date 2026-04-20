"""OMP smoke test — verifies OMP_NUM_THREADS=1 gives the expected
per-epoch speedup on a single A100 before committing to the expensive
full release-exact run.

Launch:
    runplz brev --instance mhcflurry-omp-smoketest jobs/pan_allele_omp_smoketest.py

Expected wall time: ~30-60 min (fetch + 1 fold × 2 arches × ≤20 epochs
with pretrain capped at 5 epochs). Budget: ~$3 on denvr_A100_sxm4.

Success signal: per-epoch times in LOG-worker.*.txt show <20 sec/epoch
(ideally <10). Failure signal: >50 sec/epoch means OMP fix didn't help
or there's a different bottleneck.
"""

import os
import subprocess

from runplz import App, BrevConfig, Image

app = App(
    "pan-allele-omp-smoketest",
    brev=BrevConfig(mode="container", instance_type="denvr_A100_sxm4"),
)

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("bzip2", "wget", "rsync", "build-essential", "git")
    .pip_install(
        "runplz>=1.5.0",
        "pandas>=2.0",
        "appdirs",
        "scikit-learn",
        "mhcgnomes>=3.0.1",
        "numpy>=1.22.4",
        "pyyaml",
        "tqdm",
    )
    .pip_install_local_dir(".", editable=True)
)


@app.function(
    image=image,
    gpu="A100",
    min_cpu=4,
    min_memory=26,
    min_gpu_memory=40,
    min_disk=100,
    timeout=2 * 60 * 60,
)
def train():
    env = os.environ.copy()
    env["MHCFLURRY_OUT"] = env["RUNPLZ_OUT"]
    subprocess.run(
        ["bash", "scripts/training/pan_allele_omp_smoketest.sh"],
        check=True,
        env=env,
    )


@app.local_entrypoint()
def main():
    train.remote()
