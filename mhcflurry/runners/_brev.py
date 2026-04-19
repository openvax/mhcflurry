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
from typing import Optional


REMOTE_REPO_DIR = "mhcflurry"
REMOTE_OUT_DIR = "mhcflurry-out"
REMOTE_IMAGE_TAG = "mhcflurry-train:brev"

# Brev's ~/.brev/ssh_config sets `ControlMaster auto` (connection
# multiplexing). That's fast for short repeated calls but catastrophic
# for our workload: a long-lived `docker logs -f` ssh session holds the
# master, the underlying TCP goes stale (common on GCP N1/G2 GPU boxes),
# and every subsequent ssh call — including our `docker inspect` health
# probe — hangs for ~5 minutes waiting for the dead master to time out.
# Force a fresh TCP connection per ssh call to sidestep that entirely.
#
# ServerAliveInterval=30 + large ServerAliveCountMax keeps each
# individual session alive during idle stretches (docker image pulls,
# data downloads, between-epoch pauses).
_SSH_OPTS = [
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
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
    _validate_config(cfg)
    if not _instance_exists(instance):
        if cfg.auto_create:
            _create_instance(instance, cfg.instance_type, cfg=cfg,
                             image=function.image, function=function)
        else:
            hint = (
                f"brev create {instance} --type {cfg.instance_type}"
                if cfg.mode == "vm" else
                f"brev create {instance} --mode container --type {cfg.instance_type} "
                f"--container-image {function.image.base or '<base>'}"
            )
            raise RuntimeError(
                f"Brev instance {instance!r} not found and auto_create=False. "
                f"Create it first with `{hint}`."
            )
    _refresh_ssh()

    repo = app._repo_root
    host_out = (repo / outputs_dir).resolve()
    host_out.mkdir(parents=True, exist_ok=True)

    if cfg.mode == "container":
        # Pre-built container images (e.g. pytorch/pytorch) don't ship with
        # rsync. Install it before the first rsync call, otherwise we get
        # "rsync: command not found" on the remote end.
        _ensure_remote_rsync(instance)
    _rsync_up(repo, instance)
    rel_script = Path(function.module_file).resolve().relative_to(repo)

    if cfg.mode == "container":
        # Brev `--mode container` box IS the user's container image. Apply
        # our Image DSL ops inline (apt_install, pip_install,
        # pip_install_local_dir, run_commands) via ssh, then invoke the
        # bootstrap. No docker-in-docker, no nvidia-container-toolkit,
        # no Brev VM sidecar stack.
        exit_code = _run_container_mode(
            instance=instance,
            function=function,
            rel_script=str(rel_script),
            args=args,
            kwargs=kwargs,
        )
    elif cfg.use_docker:
        _ensure_docker(instance)
        gpu_flag = "--gpus all" if _remote_has_nvidia(instance) else ""
        container_name = f"mhcflurry-{function.name}-{uuid.uuid4().hex[:8]}"
        _build_image(instance, function.image)
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
        _ssh_capture(
            instance,
            f"sudo docker rm {container_name} >/dev/null 2>&1 || true",
        )
    else:
        # Legacy native path. Skips docker; installs python + torch +
        # mhcflurry into a venv on a plain VM-mode Brev box. Use this only
        # if you can't use mode="container" (e.g. specific provider flow).
        exit_code = _run_native(
            instance=instance,
            function=function,
            rel_script=str(rel_script),
            args=args,
            kwargs=kwargs,
            has_nvidia=_remote_has_nvidia(instance),
        )

    _rsync_down(instance, host_out)
    if exit_code != 0:
        raise RuntimeError(f"Remote run exited with status {exit_code}")


_NATIVE_VENV = "$HOME/mhcflurry-venv"
_NATIVE_OUT = f"$HOME/{REMOTE_OUT_DIR}"


def _ensure_remote_rsync(instance: str):
    """Install rsync on the remote if missing (pytorch/pytorch image and
    similar are slim and don't ship with rsync)."""
    cmd = (
        "command -v rsync >/dev/null 2>&1 && exit 0; "
        "export DEBIAN_FRONTEND=noninteractive; "
        "sudo apt-get update -qq && "
        "sudo apt-get install -y -qq --no-install-recommends rsync"
    )
    _ssh(instance, cmd)


def _run_container_mode(*, instance, function, rel_script, args, kwargs):
    """Brev `--mode container`: the SSH box IS the user's container image.

    Runs the user's Image DSL ops (apt_install, pip_install, ...) inline
    over ssh, then invokes the bootstrap. Because the box is already the
    user-selected pytorch/cuda image, there's no docker-in-docker, no
    `--gpus all` nvidia-container-toolkit path, no Brev VM-mode sidecars.
    """
    ops_script = _render_ops_script(function.image)
    if ops_script:
        _ssh(instance, ops_script)

    # Values that contain $HOME must use double-quoted exports so the
    # remote shell expands them. shlex.quote would wrap in single quotes,
    # preventing expansion. Non-path values (JSON, name) go through
    # shlex.quote since they may contain shell metacharacters.
    user_env_exports = " ".join(
        f"export {k}={shlex.quote(str(v))};" for k, v in function.env.items()
    )
    remote = (
        "set -euo pipefail; "
        "export PATH=/opt/conda/bin:$PATH; "
        f'export MHCFLURRY_OUT="$HOME/{REMOTE_OUT_DIR}"; '
        f'export MHCFLURRY_RUNNER_SCRIPT="$HOME/{REMOTE_REPO_DIR}/{rel_script}"; '
        f"export MHCFLURRY_RUNNER_FUNCTION={shlex.quote(function.name)}; "
        f"export MHCFLURRY_RUNNER_ARGS={shlex.quote(json.dumps(args))}; "
        f"export MHCFLURRY_RUNNER_KWARGS={shlex.quote(json.dumps(kwargs))}; "
        f"{user_env_exports} "
        'mkdir -p "$MHCFLURRY_OUT"; '
        # cd to the rsync'd repo so user-level `subprocess.run(["bash",
        # "scripts/..."])` calls with relative paths find their scripts.
        f'cd "$HOME/{REMOTE_REPO_DIR}"; '
        "python -m mhcflurry.runners._bootstrap"
    )
    r = subprocess.run(
        ["ssh", *_SSH_OPTS, instance, f"bash -lc {shlex.quote(remote)}"]
    )
    return r.returncode


def _render_ops_script(image) -> str:
    """Translate Image DSL ops into a bash script that runs them in
    sequence on the remote container-mode box. Idempotent: apt/pip on
    already-present packages is a cheap no-op."""
    if image.dockerfile is not None:
        # Dockerfile-based images don't translate to inline ops. Caller
        # must either switch to Image.from_registry(...) + DSL, or use
        # mode="vm" with use_docker=True.
        raise RuntimeError(
            "BrevConfig(mode='container') requires Image.from_registry(...) "
            "with DSL ops. Image.from_dockerfile() doesn't translate to "
            "inline installs."
        )
    lines = ["set -euo pipefail"]
    # Start on an apt-free note: wait for any apt from box bootstrap.
    lines.append(
        "for i in $(seq 1 60); do "
        "  sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 "
        "    && { echo waiting for apt; sleep 10; } "
        "    || break; "
        "done"
    )
    lines.append("export DEBIAN_FRONTEND=noninteractive")
    lines.append("export PATH=/opt/conda/bin:$PATH")

    apt_packages_seen: list[str] = []
    pip_packages_seen: list[str] = []
    for op in image.ops:
        kw = op.kwargs_dict()
        if op.kind == "apt_install" and op.args:
            apt_packages_seen.extend(op.args)
            pkgs = " ".join(shlex.quote(p) for p in op.args)
            lines.append(
                f"sudo apt-get update -qq && sudo apt-get install -y -qq "
                f"--no-install-recommends {pkgs}"
            )
        elif op.kind == "pip_install" and op.args:
            pip_packages_seen.extend(op.args)
            pkgs = " ".join(shlex.quote(p) for p in op.args)
            idx = ""
            if "index_url" in kw:
                idx = f" --index-url {shlex.quote(kw['index_url'])}"
            lines.append(f"pip install --quiet{idx} {pkgs}")
        elif op.kind == "pip_install_local_dir":
            path = kw.get("path", ".")
            editable = kw.get("editable", "1") == "1"
            flags = "-e " if editable else ""
            # Use the rsync'd repo (we rsync it before running). Keep $HOME
            # unquoted so the remote shell expands it; only quote `path`.
            rel = path.lstrip("./")
            sub = f"/{rel}" if rel else ""
            lines.append(
                f'pip install --quiet {flags}"$HOME/{REMOTE_REPO_DIR}{sub}"'
            )
        elif op.kind == "run" and op.args:
            for cmd in op.args:
                lines.append(cmd)
    return "; ".join(lines)


def _run_native(*, instance, function, rel_script, args, kwargs, has_nvidia):
    """Install mhcflurry natively and run the job over ssh.

    Two ssh sessions: (1) idempotent setup — wait for Brev's own apt, then
    apt-get + venv + pip install; (2) actually run the user's function
    via the bootstrap, with env vars set for this specific invocation.
    """
    torch_index = (
        "https://download.pytorch.org/whl/cu121" if has_nvidia else
        "https://download.pytorch.org/whl/cpu"
    )
    setup = (
        "set -euo pipefail; "
        # Wait out Brev's own apt activity (installs docker + nvidia
        # runtime at first boot even when we don't want them).
        "for i in $(seq 1 120); do "
        "  sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 "
        "    && { echo waiting for apt; sleep 10; } "
        "    || break; "
        "done; "
        "sudo apt-get update -qq; "
        "sudo apt-get install -y -qq --no-install-recommends "
        "  python3 python3-venv python3-pip bzip2 wget rsync build-essential; "
        f"python3 -m venv {_NATIVE_VENV}; "
        f"source {_NATIVE_VENV}/bin/activate; "
        "pip install --quiet --upgrade pip; "
        f"pip install --quiet torch --index-url {torch_index}; "
        f"pip install --quiet -e $HOME/{REMOTE_REPO_DIR}"
    )
    _ssh(instance, setup)

    user_env_exports = " ".join(
        f"export {k}={shlex.quote(str(v))};" for k, v in function.env.items()
    )
    remote = (
        "set -euo pipefail; "
        f'source "$HOME/mhcflurry-venv/bin/activate"; '
        f'export MHCFLURRY_OUT="$HOME/{REMOTE_OUT_DIR}"; '
        f'export MHCFLURRY_RUNNER_SCRIPT="$HOME/{REMOTE_REPO_DIR}/{rel_script}"; '
        f"export MHCFLURRY_RUNNER_FUNCTION={shlex.quote(function.name)}; "
        f"export MHCFLURRY_RUNNER_ARGS={shlex.quote(json.dumps(args))}; "
        f"export MHCFLURRY_RUNNER_KWARGS={shlex.quote(json.dumps(kwargs))}; "
        f"{user_env_exports} "
        'mkdir -p "$MHCFLURRY_OUT"; '
        f'cd "$HOME/{REMOTE_REPO_DIR}"; '
        "python -m mhcflurry.runners._bootstrap"
    )
    r = subprocess.run(
        ["ssh", *_SSH_OPTS, instance, f"bash -lc {shlex.quote(remote)}"]
    )
    return r.returncode


def _build_image(instance: str, image):
    """Build the image on the remote, either from a user Dockerfile or
    from our Image DSL (by synthesizing a Dockerfile on the fly)."""
    if image.dockerfile is not None:
        build = (
            f"set -euo pipefail; "
            f"cd ~/{REMOTE_REPO_DIR} && "
            f"sudo docker build -f {shlex.quote(image.dockerfile)} "
            f"-t {REMOTE_IMAGE_TAG} ."
        )
    else:
        df = image.render_dockerfile()
        # Pipe the synthesized Dockerfile to docker build via stdin,
        # using the repo as context so pip_install_local_dir can COPY it.
        build = (
            f"set -euo pipefail; "
            f"cd ~/{REMOTE_REPO_DIR} && "
            f"cat <<'__EOF__' | sudo docker build -f - -t {REMOTE_IMAGE_TAG} .\n"
            f"{df}\n"
            f"__EOF__"
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
    # --network=host bypasses docker's bridge + iptables NAT machinery for the
    # container. We tried this specifically to see if conntrack saturation
    # or nvidia-container-toolkit netfilter rules were behind Brev GPU's
    # SSH-goes-dark-after-a-few-minutes bug; it wasn't (probing showed the
    # port-22 SYN/ACK still completes but the banner bytes never arrive,
    # even with host networking). Keeping the flag anyway: simpler
    # networking, no NAT overhead, and it rules out a whole class of
    # failure modes if we ever see this again.
    start = (
        f"set -euo pipefail; "
        f"mkdir -p ~/{REMOTE_OUT_DIR} && "
        f"sudo docker run -d --name {container_name} --network=host {gpu_flag} "
        f"-v $HOME/{REMOTE_OUT_DIR}:/out "
        f"{runner_env} {env_flags} "
        f"{REMOTE_IMAGE_TAG} python -m mhcflurry.runners._bootstrap"
    )
    _ssh(instance, start)


def _stream_and_wait(instance: str, container_name: str,
                     max_reconnects: int = 20) -> int:
    """Stream container logs and return its exit code.

    `docker logs -f` exits when the container stops, so we loop across SSH
    reconnects: if ssh drops mid-stream we re-attach with `--tail 0` to
    pick up where we left off, then call `docker wait` for the exit code.
    Gives up after `max_reconnects` consecutive reconnect attempts so we
    don't loop forever if the box is permanently unreachable.
    """
    print(f"+ streaming logs from {container_name} (resilient to reconnects)",
          flush=True)
    tail = "all"
    reconnects = 0
    while True:
        cmd = f"sudo docker logs -f --tail {tail} {container_name}"
        r = subprocess.run(
            ["ssh", *_SSH_OPTS, instance, cmd],
        )
        # Container may have exited cleanly (rc=0) or ssh may have dropped.
        # Either way, check whether the container is still running.
        running = _container_running(instance, container_name)
        if not running:
            break
        reconnects += 1
        if reconnects > max_reconnects:
            print(
                f"+ ssh reconnected {max_reconnects} times without finishing; "
                f"giving up on log stream. Container {container_name} may "
                f"still be running — check with `brev exec {instance} "
                f"'sudo docker logs {container_name}'`.",
                flush=True,
            )
            break
        print(f"+ ssh disconnected (rc={r.returncode}); container still "
              f"running, reconnecting log stream "
              f"({reconnects}/{max_reconnects})",
              flush=True)
        tail = "0"
        time.sleep(2)
    # Container stopped. Get its exit code (docker wait returns immediately
    # for stopped containers and prints the exit code).
    r = subprocess.run(
        ["ssh", *_SSH_OPTS, instance,
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
            ["ssh", *_SSH_OPTS, instance,
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
        ["ssh", *_SSH_OPTS, instance, remote_cmd],
        capture_output=True, text=True, timeout=60,
    )
    return r.stdout


def _validate_config(cfg):
    if cfg.mode not in ("vm", "container"):
        raise ValueError(
            f"BrevConfig.mode must be 'vm' or 'container'; got {cfg.mode!r}."
        )
    if cfg.mode == "container" and not cfg.use_docker:
        # `use_docker` is meaningless when mode='container' (the box IS the
        # container — there is no separate docker runtime inside). Flag it so
        # users don't silently get surprising behavior.
        raise ValueError(
            "BrevConfig(mode='container', use_docker=False) is contradictory. "
            "use_docker only applies to mode='vm'. In mode='container' the "
            "box itself is the user's image — there's no inner docker to skip."
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
        existed = _BREV_ONBOARDING.exists()
        _BREV_ONBOARDING.write_text(json.dumps(_BREV_ONBOARDING_DONE))
        if not existed:
            print(
                f"+ wrote {_BREV_ONBOARDING} to skip Brev CLI walkthrough "
                f"(prevents `brev ls` from hanging once instances exist)",
                flush=True,
            )
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


def _create_instance(name: str, instance_type: str, *,
                     cfg=None, image=None, function=None):
    """Provision a Brev instance.

    If `function` carries resource requests (cpu/memory/min_gpu_memory/
    min_disk/gpu), we translate those into `brev search` filters, pick
    the cheapest matching instance type, and create with it — overriding
    the explicit `cfg.instance_type` default. This is the mechanism that
    makes `@app.function(gpu="T4", memory=26000, cpu=4)` do the same
    thing on Brev that it does on Modal.
    """
    has_resource_request = function is not None and any([
        function.min_cpu is not None,
        function.min_memory is not None,
        function.min_gpu_memory is not None,
        function.min_disk is not None,
        function.gpu is not None and instance_type is None,
    ])
    if has_resource_request:
        picked = _pick_instance_type(function)
        if picked:
            instance_type = picked

    cmd = ["brev", "create", name, "--type", instance_type]
    # Propagate --min-disk from function if set (also controls
    # actual provisioned disk size on Brev, not just filter).
    if function is not None and function.min_disk is not None:
        cmd += ["--min-disk", str(function.min_disk)]
    if cfg is not None and cfg.mode == "container":
        if image is None or image.base is None:
            raise RuntimeError(
                "BrevConfig(mode='container') requires Image.from_registry(...) "
                "so we know which container image to provision the box from."
            )
        cmd += ["--mode", "container", "--container-image", image.base]
    _sh(cmd)


def _brev_gpu_name(modal_name: str) -> str:
    """Translate Modal-style GPU labels to Brev `--gpu-name` filter strings.

    Modal accepts things like "A100-40GB"; brev search wants a base name
    like "A100" and separately filters by VRAM. Stripping the suffix is
    good enough for matching.
    """
    n = modal_name.upper()
    # Strip any trailing "-40GB" / "-80GB" etc.
    for suffix in ("-40GB", "-80GB", "-16GB", "-24GB"):
        if n.endswith(suffix):
            return n[: -len(suffix)]
    return n


def _pick_instance_type(function) -> Optional[str]:
    """Run `brev search gpu` (or cpu) with filters from `function`'s
    resource requests; return the cheapest matching TYPE string.
    Returns None if no match."""
    mode = "gpu" if function.gpu is not None else "cpu"
    cmd = ["brev", "search", mode, "--json", "--sort", "price"]
    if function.gpu:
        cmd += ["--gpu-name", _brev_gpu_name(function.gpu)]
    if function.min_gpu_memory is not None:
        cmd += ["--min-vram", str(function.min_gpu_memory)]
    if function.min_cpu is not None:
        cmd += ["--min-vcpu", str(int(function.min_cpu))]
    if function.min_memory is not None:
        cmd += ["--min-ram", str(function.min_memory)]
    if function.min_disk is not None:
        cmd += ["--min-disk", str(function.min_disk)]
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    try:
        results = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    # `brev search --json` returns a list of matches sorted by $/hr.
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    return first.get("type") or first.get("Type") or first.get("name")


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
        ["ssh", *_SSH_OPTS,
         "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
         instance, wait_script],
        timeout=timeout_s,
    )
    if r.returncode != 0:
        print(
            f"+ docker daemon not reachable on {instance} after {timeout_s}s "
            f"(Brev's bootstrap may be stuck — check `brev ls` / "
            f"`brev exec {instance} 'systemctl status docker'`); "
            f"falling back to get-docker.sh",
            flush=True,
        )
        _sh(["ssh", instance, "curl -fsSL https://get.docker.com | sudo sh"])


def _remote_has_nvidia(instance: str) -> bool:
    # nvidia-smi is often pre-installed on Brev boxes even without a GPU;
    # the reliable signal is /proc/driver/nvidia, which only exists when the
    # kernel module is loaded against real hardware.
    r = subprocess.run(
        ["ssh", *_SSH_OPTS, instance,
         "test -d /proc/driver/nvidia && echo y || echo n"],
        capture_output=True, text=True, timeout=30,
    )
    return r.returncode == 0 and r.stdout.strip() == "y"


def _rsync_up(repo: Path, instance: str):
    # Intentionally no --delete: a user who sshes in and leaves files under
    # ~/mhcflurry/ (logs, probe scripts, local edits) shouldn't have those
    # wiped by the next run. Stale files on the remote are cheap; accidental
    # user-data loss is not.
    _sh([
        "rsync", "-az",
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
    _sh(["ssh", *_SSH_OPTS, instance, f"bash -lc {shlex.quote(remote_cmd)}"])


def _sh(cmd):
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)
