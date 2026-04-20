"""Pan-allele single-network training using the Modal-shaped Image DSL.

Image layers declared once and reused across backends:
    runplz local  jobs/pan_allele_dsl.py    # docker build locally
    runplz brev   --instance <gpu> jobs/pan_allele_dsl.py   # `brev create --mode container`
    runplz modal  jobs/pan_allele_dsl.py    # modal.Image chain

On Modal the layers build on Modal's cluster and cache per-hash.
On Brev with BrevConfig(mode="container") the layers run inline at boot
on top of the pre-provisioned container image. On local docker, the
layers synthesize a Dockerfile that `docker build` consumes.
"""

from runplz import App, BrevConfig, Image

app = App(
    "pan-allele-dsl",
    brev=BrevConfig(
        auto_create=True,
        mode="container",
        # Pin Denvr A100 (40GB) — known-reliable provider. The default
        # cheapest match is MassedCompute, whose DGX-class boxes have
        # been observed "RUNNING" on Brev's side but unreachable on
        # port 2222 for extended periods.
        instance_type="denvr_A100_sxm4",
    ),
)

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("bzip2", "wget", "rsync", "build-essential", "git")
    .pip_install(
        "runplz>=1.5.0",  # the in-container bootstrap runs `python -m runplz._bootstrap`
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
    # Resource requests — GB for everything memory/disk-related.
    # 26 GB RAM avoids the SIGKILL we hit on 16 GB boxes during
    # mhcflurry's full-data peptide encoding.
    gpu="A100",
    min_cpu=4,
    min_memory=26,         # GB
    min_gpu_memory=40,     # GB VRAM (A100-40GB)
    min_disk=100,          # GB
    timeout=6 * 60 * 60,
    env={
        "MHCFLURRY_OUT": "/out",
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
