"""Pan-allele training smoketest — Modal-shaped entry point.

Run:
    mhcflurry-run local jobs/pan_allele_smoketest.py
    mhcflurry-run brev  --instance my-gpu-box jobs/pan_allele_smoketest.py
"""

from mhcflurry.runners import App, BrevConfig, Image

app = App(
    "pan-allele-smoketest",
    brev=BrevConfig(instance_type="a100-40gb", auto_create=False),
)

image = Image.from_dockerfile("docker/Dockerfile.train")


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    env={
        "TINY": "1",
        "SMOKETEST_MAX_EPOCHS": "2",
        "SMOKETEST_NUM_FOLDS": "1",
        "SMOKETEST_NUM_REPLICATES": "1",
    },
)
def train():
    import subprocess
    subprocess.run(
        ["bash", "scripts/training/pan_allele_smoketest.sh"],
        check=True,
    )


@app.local_entrypoint()
def main():
    train.remote()
