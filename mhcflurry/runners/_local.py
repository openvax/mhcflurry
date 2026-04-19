"""Local backend: docker build + docker run.

Auto-detects NVIDIA runtime via `docker info`; passes `--gpus all` when
available, omits otherwise. The training library handles CPU/MPS/CUDA
selection on its own.
"""

import json
import subprocess
from pathlib import Path


IMAGE_TAG_DEFAULT = "mhcflurry-train:local"


def run(app, function, args, kwargs, *,
        image_tag: str = IMAGE_TAG_DEFAULT,
        build: bool = True,
        outputs_dir: str = "out"):
    repo = app._repo_root
    if repo is None:
        raise RuntimeError("App repo_root not set (CLI should have set this).")

    host_out = (repo / outputs_dir).resolve()
    host_out.mkdir(parents=True, exist_ok=True)

    df, ctx = function.image.resolve(repo)
    if build:
        _docker_build(df, ctx, image_tag)

    script_in_container = _container_path_for(function.module_file, repo)

    cmd = [
        "docker", "run", "--rm",
        "--name", f"mhcflurry-{app.name}-{function.name}",
        "-v", f"{host_out}:/out",
        "-w", "/workspace",
        "-e", "MHCFLURRY_OUT=/out",
        "-e", f"MHCFLURRY_RUNNER_SCRIPT={script_in_container}",
        "-e", f"MHCFLURRY_RUNNER_FUNCTION={function.name}",
        "-e", f"MHCFLURRY_RUNNER_ARGS={json.dumps(args)}",
        "-e", f"MHCFLURRY_RUNNER_KWARGS={json.dumps(kwargs)}",
    ]
    if _nvidia_available():
        cmd += ["--gpus", "all"]
    for k, v in function.env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [image_tag, "python", "-m", "mhcflurry.runners._bootstrap"]

    _print_cmd(cmd)
    subprocess.run(cmd, check=True)


def _docker_build(dockerfile: Path, context: Path, tag: str):
    cmd = ["docker", "build", "-f", str(dockerfile), "-t", tag, str(context)]
    _print_cmd(cmd)
    subprocess.run(cmd, check=True)


def _nvidia_available() -> bool:
    r = subprocess.run(["docker", "info", "--format", "{{json .Runtimes}}"],
                       capture_output=True, text=True)
    return r.returncode == 0 and "nvidia" in r.stdout


def _container_path_for(host_path: str, repo: Path) -> str:
    # Assumes the image bakes the repo at /workspace (our Dockerfile does).
    rel = Path(host_path).resolve().relative_to(repo)
    return str(Path("/workspace") / rel)


def _print_cmd(cmd):
    # flush=True matches the convention in _brev/_modal — lets users tailing
    # a log file see status prints land before the subprocess output.
    print("+ " + " ".join(cmd), flush=True)
