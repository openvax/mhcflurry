"""Small Modal-shaped API for mhcflurry training/prediction jobs.

Pattern:
    from mhcflurry.runners import App, Image, BrevConfig

    app = App("my-job", brev=BrevConfig(instance_type="a100-40gb", auto_create=True))
    image = Image.from_dockerfile("docker/Dockerfile.train")

    @app.function(image=image, gpu="A100")
    def train():
        import subprocess
        subprocess.run(["bash", "scripts/training/pan_allele_smoketest.sh"], check=True)

    @app.local_entrypoint()
    def main():
        train.remote()

Then: `mhcflurry-run local path/to/script.py` (or `brev --instance ...`).
"""

from mhcflurry.runners.app import App, Function
from mhcflurry.runners.config import BrevConfig, ModalConfig
from mhcflurry.runners.image import Image

__all__ = ["App", "Function", "Image", "BrevConfig", "ModalConfig"]
