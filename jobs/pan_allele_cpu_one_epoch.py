"""One-epoch CPU training on Brev via docker — scenario validation.

Same workload as pan_allele_one_epoch.py but explicit instance type
is CPU (n2d-highmem-2), to verify the docker+CPU path fully completes
a training epoch (not just "SSH stays up while sleeping").
"""

from mhcflurry.runners import App, BrevConfig, Image

app = App(
    "pan-allele-cpu-one-epoch",
    brev=BrevConfig(instance_type="n2d-highmem-2", auto_create=False),
)

image = Image.from_dockerfile("docker/Dockerfile.train")


@app.function(
    image=image,
    gpu=None,
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
