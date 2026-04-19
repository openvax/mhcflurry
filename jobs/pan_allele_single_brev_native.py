"""Pan-allele single-network training on Brev, **without docker**.

Same as pan_allele_single.py but sets BrevConfig(use_docker=False).
The Brev backend skips docker build/run entirely and installs
mhcflurry + PyTorch into a venv on the box instead. Workaround for
the Brev-specific SSH-dies-under-`docker run --gpus all` bug; once
that's fixed, the regular docker path will work too.

Run:
    mhcflurry-run brev --instance <gpu-box> jobs/pan_allele_single_brev_native.py
"""

from mhcflurry.runners import App, BrevConfig, Image

app = App(
    "pan-allele-single-brev-native",
    brev=BrevConfig(
        instance_type="a100-40gb",
        auto_create=False,
        use_docker=False,
    ),
)

# Declared for API symmetry (so this job runs unchanged on Modal/local).
# Ignored by the Brev backend when use_docker=False.
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
