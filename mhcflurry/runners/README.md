# mhcflurry.runners

Small Modal-shaped API for launching mhcflurry training (and eventually
prediction) jobs on **local Docker**, **Brev**, or **Modal**. All three
backends share `docker/Dockerfile.train` so the runtime is identical across them.

## Shape

```python
from mhcflurry.runners import App, BrevConfig, Image

app = App(
    "pan-allele-smoketest",
    brev=BrevConfig(instance_type="a100-40gb", auto_create=True),
)

image = Image.from_dockerfile("docker/Dockerfile.train")

@app.function(image=image, gpu="A100", timeout=3600,
              env={"TINY": "1"})
def train():
    import subprocess
    subprocess.run(
        ["bash", "scripts/training/pan_allele_smoketest.sh"],
        check=True,
    )

@app.local_entrypoint()
def main():
    train.remote()
```

Run:
```bash
mhcflurry-run local  jobs/pan_allele_smoketest.py
mhcflurry-run brev   --instance my-gpu-box  jobs/pan_allele_smoketest.py
mhcflurry-run modal  jobs/pan_allele_smoketest.py    # stub
```

## Contract

Every backend honors the same contract: **image + command + gpu + env +
outputs_dir**. Training writes to `$MHCFLURRY_OUT` inside the container; the
runner collects that directory back to `./out/` on the host.

`.remote(...)` args must be JSON-serializable — no closures, no custom
objects. This is the main concession to keeping the surface small.

## Credentials

- **Modal**: reads `~/.modal.toml` (`modal setup` once).
- **Brev**: uses `brev` CLI state from `brev login`. No API key needed.
  `BrevConfig(api_key_env="BREV_API_KEY", auth="rest")` is a reserved hook
  for a future REST fallback — not implemented yet.

## Brev quickstart

```bash
sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"
brev login
# Either pre-create an instance in the web console, or set auto_create=True in BrevConfig.
mhcflurry-run brev --instance my-gpu-box jobs/pan_allele_smoketest.py
```

## Layout

```
docker/Dockerfile.train                     # shared image
mhcflurry/runners/
  __init__.py                               # App, Image, BrevConfig, ModalConfig
  app.py                                    # App + Function
  image.py                                  # Image.from_dockerfile
  config.py                                 # BrevConfig, ModalConfig
  _local.py  _brev.py  _modal.py            # backends
  _bootstrap.py                             # in-container entrypoint
  _cli.py                                   # mhcflurry-run
jobs/pan_allele_smoketest.py                # example job
scripts/training/pan_allele_smoketest.sh    # backend-agnostic training
```

## Tiny sanity check

The smoketest script supports `TINY=1` — filters to 2 alleles / ~200 rows,
runs a few epochs. Finishes in ~1 minute on CPU. Useful both for local
iteration and for a sanity pass on a new Brev instance before committing
to a full run.
