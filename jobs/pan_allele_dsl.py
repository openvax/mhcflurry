"""Pan-allele single-network training using the Modal-shaped Image DSL.

Image layers declared once and reused across backends:
    mhcflurry-run local  jobs/pan_allele_dsl.py    # docker build locally
    mhcflurry-run brev   --instance <gpu> jobs/pan_allele_dsl.py   # `brev create --mode container`
    mhcflurry-run modal  jobs/pan_allele_dsl.py    # modal.Image chain

On Modal the layers build on Modal's cluster and cache per-hash.
On Brev with BrevConfig(mode="container") the layers run inline at boot
on top of the pre-provisioned container image. On local docker, the
layers synthesize a Dockerfile that `docker build` consumes.
"""

from mhcflurry.runners import App, BrevConfig, Image

app = App(
    "pan-allele-dsl",
    brev=BrevConfig(
        instance_type="g2-standard-4:nvidia-l4:1",
        auto_create=False,
        mode="vm",
    ),
)

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("bzip2", "wget", "rsync", "build-essential", "git")
    .pip_install(
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
    gpu="L4",
    timeout=6 * 60 * 60,
    env={
        "SINGLE_MAX_EPOCHS": "500",
        "SINGLE_PATIENCE": "20",
    },
)
def train():
    import subprocess
    subprocess.run(
        ["bash", "scripts/training/pan_allele_single.sh"],
        check=True,
    )


@app.local_entrypoint()
def main():
    train.remote()
