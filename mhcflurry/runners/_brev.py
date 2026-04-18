"""Brev backend: provision (optional) → rsync repo → ssh docker build+run → rsync outputs.

Assumes `brev` CLI is installed and `brev login` has been run. Uses Brev's
managed SSH config (`brev refresh` populates ~/.brev/ssh_config, which
~/.ssh/config Includes).
"""

import json
import shlex
import subprocess
from pathlib import Path


REMOTE_REPO_DIR = "mhcflurry"
REMOTE_OUT_DIR = "mhcflurry-out"
REMOTE_IMAGE_TAG = "mhcflurry-train:brev"

# Brev's CLI hangs on an interactive walkthrough once an instance exists in the
# org, blocking `brev ls`/`brev refresh`. Overwriting this file with the
# completed state skips it. The file lives under the user's home — cheap and
# harmless.
_BREV_ONBOARDING = Path.home() / ".brev" / "onboarding_step.json"
_BREV_ONBOARDING_DONE = {
    "step": 999,
    "hasRunBrevShell": True,
    "hasRunBrevOpen": True,
}


def run(app, function, args, kwargs, *,
        instance: str,
        outputs_dir: str = "out"):
    _require_brev_cli()
    _skip_onboarding()

    cfg = app.brev
    if not _instance_exists(instance):
        if cfg.auto_create:
            _create_instance(instance, cfg.instance_type)
        else:
            raise RuntimeError(
                f"Brev instance {instance!r} not found and auto_create=False. "
                f"Create it first with `brev create {instance} --type {cfg.instance_type}`."
            )
    _refresh_ssh()

    repo = app._repo_root
    host_out = (repo / outputs_dir).resolve()
    host_out.mkdir(parents=True, exist_ok=True)

    _ensure_docker(instance)
    _rsync_up(repo, instance)

    rel_script = Path(function.module_file).resolve().relative_to(repo)
    gpu_flag = "--gpus all" if _remote_has_nvidia(instance) else ""
    remote_cmd = _build_remote_command(
        function=function,
        rel_script=str(rel_script),
        args=args,
        kwargs=kwargs,
        gpu_flag=gpu_flag,
    )
    _ssh(instance, remote_cmd)
    _rsync_down(instance, host_out)


def _build_remote_command(*, function, rel_script: str, args, kwargs,
                          gpu_flag: str) -> str:
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
        f"sudo docker build -f {shlex.quote(df)} -t {REMOTE_IMAGE_TAG} . && "
        f"mkdir -p ~/{REMOTE_OUT_DIR} && "
        f"sudo docker run --rm {gpu_flag} "
        f"-v $HOME/{REMOTE_OUT_DIR}:/out "
        f"{runner_env} {env_flags} "
        f"{REMOTE_IMAGE_TAG} python -m mhcflurry.runners._bootstrap"
    )


def _require_brev_cli():
    if subprocess.run(["which", "brev"], capture_output=True).returncode != 0:
        raise RuntimeError(
            "`brev` CLI not found. Install via `brew install brev` (macOS) "
            "or the script at https://developer.nvidia.com/brev, then run "
            "`brev login`."
        )


def _skip_onboarding():
    try:
        _BREV_ONBOARDING.parent.mkdir(parents=True, exist_ok=True)
        _BREV_ONBOARDING.write_text(json.dumps(_BREV_ONBOARDING_DONE))
    except OSError:
        pass


def _instance_exists(name: str) -> bool:
    r = subprocess.run(["brev", "ls", "--json"], capture_output=True, text=True,
                       timeout=60)
    if r.returncode != 0:
        return False
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return False
    instances = data if isinstance(data, list) else data.get("instances", [])
    return any(i.get("name") == name for i in instances)


def _create_instance(name: str, instance_type: str):
    _sh(["brev", "create", name, "--type", instance_type])


def _refresh_ssh():
    _sh(["brev", "refresh"])


def _ensure_docker(instance: str, timeout_s: int = 420):
    # Brev's own bootstrap installs docker shortly after the instance comes up.
    # Poll for it (up to ~7 min) before falling back to get-docker.sh.
    print(f"+ waiting for docker on {instance} (up to {timeout_s}s)")
    wait_script = (
        "for i in $(seq 1 60); do "
        "command -v docker >/dev/null 2>&1 && exit 0; "
        "sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 "
        "  && echo 'apt busy, waiting' || echo 'no docker yet'; "
        "sleep 7; "
        "done; exit 1"
    )
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
         instance, wait_script],
        timeout=timeout_s,
    )
    if r.returncode != 0:
        print("+ docker still missing — falling back to get-docker.sh")
        _sh(["ssh", instance, "curl -fsSL https://get.docker.com | sudo sh"])


def _remote_has_nvidia(instance: str) -> bool:
    # nvidia-smi is often pre-installed on Brev boxes even without a GPU;
    # the reliable signal is /proc/driver/nvidia, which only exists when the
    # kernel module is loaded against real hardware.
    r = subprocess.run(
        ["ssh", instance, "test -d /proc/driver/nvidia && echo y || echo n"],
        capture_output=True, text=True, timeout=30,
    )
    return r.returncode == 0 and r.stdout.strip() == "y"


def _rsync_up(repo: Path, instance: str):
    _sh([
        "rsync", "-az", "--delete",
        "--exclude=.git", "--exclude=.venv", "--exclude=__pycache__",
        "--exclude=*.egg-info", "--exclude=build", "--exclude=dist",
        "--exclude=out",
        f"{repo}/", f"{instance}:{REMOTE_REPO_DIR}/",
    ])


def _rsync_down(instance: str, local_out: Path):
    _sh(["rsync", "-az", f"{instance}:{REMOTE_OUT_DIR}/", f"{local_out}/"])


def _ssh(instance: str, remote_cmd: str):
    _sh(["ssh", instance, "bash", "-lc", remote_cmd])


def _sh(cmd):
    print("+ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)
