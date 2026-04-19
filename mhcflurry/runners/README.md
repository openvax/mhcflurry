# mhcflurry.runners

Small Modal-shaped Python API for launching mhcflurry training (and
eventually prediction) jobs on **local Docker**, **Brev**, or **Modal**.
One declaration, three backends.

## Shape

```python
from mhcflurry.runners import App, BrevConfig, Image

app = App(
    "pan-allele-single",
    brev=BrevConfig(instance_type="g2-standard-4:nvidia-l4:1",
                    auto_create=False, mode="vm"),
)

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("bzip2", "rsync", "build-essential")
    .pip_install("pandas>=2.0", "scikit-learn", "mhcgnomes>=3.0.1",
                 "pyyaml", "tqdm", "appdirs")
    .pip_install_local_dir(".", editable=True)
)

@app.function(
    image=image,
    gpu="T4",
    min_cpu=4,
    min_memory=26,          # GB
    min_gpu_memory=16,      # GB VRAM
    min_disk=100,           # GB
    timeout=60 * 60,
    env={"SINGLE_MAX_EPOCHS": "500", "SINGLE_PATIENCE": "20"},
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
```

Run:

```bash
mhcflurry-run local jobs/pan_allele_dsl.py
mhcflurry-run brev  --instance my-gpu-box jobs/pan_allele_dsl.py
mhcflurry-run modal jobs/pan_allele_dsl.py
```

## Image DSL

The `Image` API mirrors `modal.Image`:

- `Image.from_registry(ref)` — start from a public image
- `Image.from_dockerfile(path, context=...)` — use a pre-existing Dockerfile
- `.apt_install(*packages)` — apt layer
- `.pip_install(*packages, index_url=...)` — pip layer
- `.pip_install_local_dir(path, editable=True)` — copy the repo in and `pip install -e`
- `.run_commands(*cmds)` — arbitrary shell

Each backend translates ops the way it needs to:
- **Modal**: `modal.Image.from_registry(...).apt_install(...)...` (server-side build, cached by hash)
- **Brev `mode="vm"` + docker**: synthesizes a Dockerfile, `docker build` on remote
- **Brev `mode="container"`**: box IS the base image; ops run inline via ssh at first invocation
- **local**: same as Brev vm mode, built locally

## Resource requests

`@app.function(gpu=..., min_cpu=..., min_memory=..., min_gpu_memory=..., min_disk=...)`

All memory/disk values in **GB**. On Modal the fields forward directly;
on Brev they drive `brev search` to auto-pick a matching instance type
(when `BrevConfig.auto_create=True`). `BrevConfig.instance_type` is
the fallback.

Common `gpu=` labels (Modal-accepted, also recognized by the Brev
translator): `T4`, `L4`, `L40S`, `A10`/`A10G`, `A100-40GB`,
`A100-80GB`, `H100`, `H200`, `V100`.

## Contract

Every backend honors: **image + command + gpu + env + outputs_dir**.
Training writes to `$MHCFLURRY_OUT` inside the container; the runner
collects that directory back to `./out/` on the host.

`.remote(...)` args must be JSON-serializable — no closures, no custom
objects. This is the main concession to keeping the surface small.

## Credentials

- **Modal**: `~/.modal.toml` (`modal setup` once).
- **Brev**: `brev` CLI state from `brev login`. No API key needed.
  `BrevConfig(api_key_env="BREV_API_KEY", auth="rest")` is a reserved
  hook for a future REST fallback — not implemented yet.

## Brev specifics

Two provisioning modes:

| Mode | Box shape | Who runs user code | Good for |
|---|---|---|---|
| `mode="vm"` (default) | Full VM + Brev sidecar stack (grafana, jupyter, cloudflared) + docker runtime | Our `docker run --gpus all` | Most training; image caching across boxes |
| `mode="container"` | Box *is* your image (passed via `brev create --mode container --container-image <base>`); no sidecars | Python process directly in the box | Fastest cold-start; avoids nvidia-container-toolkit |

Also `BrevConfig(use_docker=False)` (mode="vm" only): legacy escape
hatch that skips docker entirely and installs mhcflurry natively into
a venv on the VM. Kept because at one point `docker run --gpus all`
had a reproducible bug on Brev GPU boxes (see
[docs/brev-ssh-bug-report.md](../../docs/brev-ssh-bug-report.md)). The
bug does not reproduce at time of writing but the escape hatch remains.

## Modal specifics

- `pip install 'mhcflurry[runners-modal]'` to pull `modal>=1.1,<2`.
- First run on a given `Image` takes ~1–2 min for Modal to build the
  image; cached after that.
- Function return value currently goes through a tar blob (256 MB cap).
  Fine for single-model training. For ensemble / heavy multi-fold runs
  we need to switch to `modal.Volume` — see TODO in `_modal.py`.

## Layout

```
docker/Dockerfile.train                     # optional shared image
mhcflurry/runners/
  __init__.py                               # App, Image, BrevConfig, ModalConfig
  app.py                                    # App + Function + resource request fields
  image.py                                  # Image DSL (from_registry + ops chain)
  config.py                                 # BrevConfig, ModalConfig
  _local.py  _brev.py  _modal.py            # backends
  _bootstrap.py                             # in-container entrypoint
  _cli.py                                   # `mhcflurry-run` entry
jobs/
  pan_allele_smoketest.py                   # TINY sanity check
  pan_allele_dsl.py                         # real-data single-network training
scripts/training/
  pan_allele_smoketest.sh                   # TINY=1 + hyperparameters.yaml
  pan_allele_single.sh                      # full-data single-network
scripts/validate_against_public.py          # compare a trained predictor vs. public release
docs/brev-ssh-bug-report.md                 # historical bug record for Brev support
test/test_runners.py                        # offline tests for DSL + config + picker
```

## Tests

```bash
pytest test/test_runners.py
```

Tests are offline — they don't provision any cloud resources. The
closest thing to an integration test is running the TINY smoketest
through each backend manually.

## Tiny sanity check

`scripts/training/pan_allele_smoketest.sh` supports `TINY=1` — filters
to 2 alleles / ~200 rows, runs 2 epochs. Finishes in about a minute on
CPU. Useful both for local iteration and for sanity-checking a new
Brev instance before committing to a real run:

```bash
mhcflurry-run brev --instance my-cpu-box jobs/pan_allele_smoketest.py
```
