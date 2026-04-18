"""Brev backend: provision (optional) → rsync repo → ssh docker build+run → rsync outputs.

Assumes `brev` CLI is installed and `brev login` has been run. Uses Brev's
managed SSH config (`brev refresh` populates ~/.brev/ssh_config, which
~/.ssh/config Includes).

Long training runs are resilient to SSH disconnects: the remote docker
container is started detached (`docker run -d`), logs are streamed via
`docker logs -f` in a reconnect loop, and we wait for the container to
exit via `docker wait` (which tolerates transient tunnel drops).
"""

import json
import shlex
import subprocess
import time
import uuid
from pathlib import Path


REMOTE_REPO_DIR = "mhcflurry"
REMOTE_OUT_DIR = "mhcflurry-out"
REMOTE_IMAGE_TAG = "mhcflurry-train:brev"

# Aggressive keepalives so Brev's SSH tunnel doesn't drop the connection
# during a multi-hour docker run. ServerAliveInterval=30 sends a probe every
# 30s if the session is idle; ServerAliveCountMax=240 tolerates 2 hours of
# unacked probes before giving up.
_SSH_KEEPALIVE = [
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=240",
    "-o", "TCPKeepAlive=yes",
]

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

    container_name = f"mhcflurry-{function.name}-{uuid.uuid4().hex[:8]}"
    _build_image(instance, function.image.dockerfile)
    _run_container_detached(
        instance=instance,
        container_name=container_name,
        function=function,
        rel_script=str(rel_script),
        args=args,
        kwargs=kwargs,
        gpu_flag=gpu_flag,
    )
    exit_code = _stream_and_wait(instance, container_name)
    _ssh_capture(instance,
                 f"sudo docker rm {container_name} >/dev/null 2>&1 || true")
    _rsync_down(instance, host_out)
    if exit_code != 0:
        raise RuntimeError(
            f"Remote container {container_name} exited with status {exit_code}"
        )


def _build_image(instance: str, dockerfile: str):
    build = (
        f"set -euo pipefail; "
        f"cd ~/{REMOTE_REPO_DIR} && "
        f"sudo docker build -f {shlex.quote(dockerfile)} "
        f"-t {REMOTE_IMAGE_TAG} ."
    )
    _ssh(instance, build)


def _run_container_detached(*, instance, container_name, function, rel_script,
                            args, kwargs, gpu_flag):
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
    start = (
        f"set -euo pipefail; "
        f"mkdir -p ~/{REMOTE_OUT_DIR} && "
        f"sudo docker run -d --name {container_name} {gpu_flag} "
        f"-v $HOME/{REMOTE_OUT_DIR}:/out "
        f"{runner_env} {env_flags} "
        f"{REMOTE_IMAGE_TAG} python -m mhcflurry.runners._bootstrap"
    )
    _ssh(instance, start)


def _stream_and_wait(instance: str, container_name: str) -> int:
    """Stream container logs and return its exit code.

    `docker logs -f` exits when the container stops, so we loop across SSH
    reconnects: if ssh drops mid-stream we re-attach with `--tail 0` to
    pick up where we left off, then call `docker wait` for the exit code.
    """
    print(f"+ streaming logs from {container_name} (resilient to reconnects)",
          flush=True)
    tail = "all"
    while True:
        cmd = f"sudo docker logs -f --tail {tail} {container_name}"
        r = subprocess.run(
            ["ssh", *_SSH_KEEPALIVE, instance, cmd],
        )
        # Container may have exited cleanly (rc=0) or ssh may have dropped.
        # Either way, check whether the container is still running.
        running = _container_running(instance, container_name)
        if not running:
            break
        print(f"+ ssh disconnected (rc={r.returncode}); container still "
              f"running, reconnecting log stream", flush=True)
        tail = "0"
        time.sleep(2)
    # Container stopped. Get its exit code (docker wait returns immediately
    # for stopped containers and prints the exit code).
    r = subprocess.run(
        ["ssh", *_SSH_KEEPALIVE, instance,
         f"sudo docker wait {container_name}"],
        capture_output=True, text=True,
    )
    try:
        return int(r.stdout.strip() or "1")
    except ValueError:
        return 1


def _container_running(instance: str, container_name: str) -> bool:
    # Treat ssh hangs / errors as "assume still running" so the caller keeps
    # retrying the log stream instead of giving up. If the box really did die
    # mid-training, docker wait at the end will return its final exit code.
    try:
        r = subprocess.run(
            ["ssh", *_SSH_KEEPALIVE, instance,
             f"sudo docker inspect --format '{{{{.State.Running}}}}' {container_name}"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return True
    if r.returncode != 0:
        return True
    return r.stdout.strip() == "true"


def _ssh_capture(instance: str, remote_cmd: str) -> str:
    r = subprocess.run(
        ["ssh", *_SSH_KEEPALIVE, instance, remote_cmd],
        capture_output=True, text=True, timeout=60,
    )
    return r.stdout


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
    # On GPU boxes the binary appears before dockerd is listening, so poll
    # `docker info` (daemon reachable) rather than just presence of the binary.
    print(f"+ waiting for docker daemon on {instance} (up to {timeout_s}s)", flush=True)
    wait_script = (
        "for i in $(seq 1 60); do "
        "if command -v docker >/dev/null 2>&1 && "
        "   sudo docker info >/dev/null 2>&1; then exit 0; fi; "
        "sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 "
        "  && echo 'apt busy, waiting' || echo 'waiting for docker daemon'; "
        "sleep 7; "
        "done; exit 1"
    )
    r = subprocess.run(
        ["ssh", *_SSH_KEEPALIVE,
         "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
         instance, wait_script],
        timeout=timeout_s,
    )
    if r.returncode != 0:
        print("+ docker daemon not ready — falling back to get-docker.sh",
              flush=True)
        _sh(["ssh", instance, "curl -fsSL https://get.docker.com | sudo sh"])


def _remote_has_nvidia(instance: str) -> bool:
    # nvidia-smi is often pre-installed on Brev boxes even without a GPU;
    # the reliable signal is /proc/driver/nvidia, which only exists when the
    # kernel module is loaded against real hardware.
    r = subprocess.run(
        ["ssh", *_SSH_KEEPALIVE, instance,
         "test -d /proc/driver/nvidia && echo y || echo n"],
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
    # Pass the whole pipeline as a SINGLE arg to ssh. If we pass
    # ["ssh", host, "bash", "-lc", cmd] instead, ssh space-joins the trailing
    # argv before sending to the remote shell, which then re-parses — turning
    # `bash -lc 'set -euo pipefail; X'` into `bash -lc set -euo pipefail; X`
    # (i.e. `set` runs with no args as the -c command, X runs in the outer
    # shell without errexit). Quoting with shlex.quote around the whole
    # command string avoids that.
    _sh(["ssh", *_SSH_KEEPALIVE, instance, f"bash -lc {shlex.quote(remote_cmd)}"])


def _sh(cmd):
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)
