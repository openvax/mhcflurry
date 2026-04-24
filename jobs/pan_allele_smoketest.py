"""Pan-allele training smoketest — Modal-shaped entry point.

Run:
    runplz local jobs/pan_allele_smoketest.py
    runplz brev  --instance my-gpu-box jobs/pan_allele_smoketest.py
"""

from runplz import App, BrevConfig, Image

app = App(
    "pan-allele-smoketest",
    brev_config=BrevConfig(auto_create_instances=False),
)

image = Image.from_dockerfile("docker/Dockerfile.train")


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    env={
        "MHCFLURRY_OUT": "/out",
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
