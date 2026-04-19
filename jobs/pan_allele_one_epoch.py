"""One-epoch pan-allele training — probe whether Brev SSH drops on CPU.

Same as pan_allele_single.py but caps training at 1 epoch and runs it
on a CPU box, so the workload lasts long enough to trigger any
Brev-side session instability without using --gpus all.
"""

from mhcflurry.runners import App, BrevConfig, Image

app = App(
    "pan-allele-one-epoch",
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
