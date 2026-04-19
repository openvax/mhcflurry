"""One-epoch pan-allele training on Brev GPU via docker (repro case).

Exercises the failing config: `docker run --gpus all` + real pytorch
training with multiprocessing pool + disk I/O. Used to check whether
larger-vCPU instances still hit the SSH-banner-wedge bug.
"""

from mhcflurry.runners import App, BrevConfig, Image

app = App(
    "pan-allele-gpu-one-epoch",
    brev=BrevConfig(
        instance_type="g2-standard-8:nvidia-l4:1",
        auto_create=False,
    ),
)

image = Image.from_dockerfile("docker/Dockerfile.train")


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    env={
        "SINGLE_MAX_EPOCHS": "1",
        "SINGLE_PATIENCE": "1",
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
