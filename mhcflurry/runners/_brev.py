"""Brev backend: provision (optional) → rsync repo → ssh docker build+run → rsync outputs.

Assumes `brev` CLI is installed and `brev login` has been run. Uses Brev's
managed SSH config (`brev refresh` updates ~/.ssh/config).
"""

import json
import shlex
import subprocess
from pathlib import Path


REMOTE_REPO_DIR = "mhcflurry"           # relative to $HOME on the instance
REMOTE_OUT_DIR = "mhcflurry-out"
REMOTE_IMAGE_TAG = "mhcflurry-train:brev"


def run(app, function, args, kwargs, *,
        instance: str,
        outputs_dir: str = "out"):
    _require_brev_cli()

    cfg = app.brev
    if cfg.auto_create and not _instance_exists(instance):
        _create_instance(instance, cfg.instance_type)
    _refresh_ssh()

    repo = app._repo_root
    host_out = (repo / outputs_dir).resolve()
    host_out.mkdir(parents=True, exist_ok=True)

    _rsync_up(repo, instance)
    rel_script = Path(function.module_file).resolve().relative_to(repo)

    remote_cmd = _build_remote_command(
        function=function,
        rel_script=str(rel_script),
        args=args,
        kwargs=kwargs,
    )
    _ssh(instance, remote_cmd)
    _rsync_down(instance, host_out)


def _build_remote_command(*, function, rel_script: str, args, kwargs) -> str:
    df = function.image.dockerfile
    env_flags = " ".join(
        f"-e {shlex.quote(f'{k}={v}')}" for k, v in function.env.items()
    )
    runner_env = (
        f"-e MHCFLURRY_OUT=/out "
        f"-e MHCFLURRY_RUNNER_SCRIPT={shlex.quote('/workspace/' + rel_script)} "
        f"-e MHCFLURRY_RUNNER_FUNCTION={shlex.quote(function.name)} "
        f"-e MHCFLURRY_RUNNER_ARGS={shlex.quote(json.dumps(args))} "
        f"-e MHCFLURRY_RUNNER_KWARGS={shlex.quote(json.dumps(kwargs))}"
    )
    return (
        f"set -euo pipefail; "
        f"cd ~/{REMOTE_REPO_DIR} && "
        f"docker build -f {shlex.quote(df)} -t {REMOTE_IMAGE_TAG} . && "
        f"mkdir -p ~/{REMOTE_OUT_DIR} && "
        f"docker run --rm --gpus all "
        f"-v ~/{REMOTE_OUT_DIR}:/out "
        f"{runner_env} {env_flags} "
        f"{REMOTE_IMAGE_TAG} python -m mhcflurry.runners._bootstrap"
    )


def _require_brev_cli():
    if subprocess.run(["which", "brev"], capture_output=True).returncode != 0:
        raise RuntimeError(
            "`brev` CLI not found. Install via:\n"
            '  sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"\n'
            "Then run: brev login"
        )


def _instance_exists(name: str) -> bool:
    r = subprocess.run(["brev", "ls"], capture_output=True, text=True)
    return r.returncode == 0 and name in r.stdout


def _create_instance(name: str, instance_type: str):
    _sh(["brev", "create", name, "--instance-type", instance_type])


def _refresh_ssh():
    _sh(["brev", "refresh"])


def _rsync_up(repo: Path, instance: str):
    _sh([
        "rsync", "-az", "--delete",
        "--exclude=.git", "--exclude=.venv", "--exclude=__pycache__",
        "--exclude=*.egg-info", "--exclude=build", "--exclude=dist",
        "--exclude=out",
        f"{repo}/", f"{instance}:~/{REMOTE_REPO_DIR}/",
    ])


def _rsync_down(instance: str, local_out: Path):
    _sh(["rsync", "-az", f"{instance}:~/{REMOTE_OUT_DIR}/", f"{local_out}/"])


def _ssh(instance: str, remote_cmd: str):
    _sh(["ssh", instance, "bash", "-lc", remote_cmd])


def _sh(cmd):
    print("+ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)
