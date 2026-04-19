"""One-epoch real mhcflurry training on the diagnostic Brev box.

Triggers the failing workload (multiprocessing.Pool + disk I/O + docker
--gpus all) on the mhcflurry-diag host where the privileged diagnostic
container is already dumping host state every 15 s to
$HOME/diag-out/brev-host-diag.log. Purpose: catch the SSH wedge with
host-side evidence in hand.
"""

from mhcflurry.runners import App, BrevConfig, Image

app = App(
    "pan-allele-diag",
    brev=BrevConfig(instance_type="g2-standard-4:nvidia-l4:1", auto_create=False),
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
