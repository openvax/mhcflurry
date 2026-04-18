"""Pan-allele single-network training — Modal-shaped entry point.

Trains one pan-allele network with real hyperparameters on the full
curated training data. One fold, one replicate, early stopping.

Run:
    mhcflurry-run brev --instance <gpu-box> jobs/pan_allele_single.py
"""

from mhcflurry.runners import App, BrevConfig, Image

app = App(
    "pan-allele-single",
    brev=BrevConfig(instance_type="a100-40gb", auto_create=False),
)

image = Image.from_dockerfile("docker/Dockerfile.train")


@app.function(
    image=image,
    gpu="T4",
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
