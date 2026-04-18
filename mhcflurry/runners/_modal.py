"""Modal backend.

We generate a per-invocation Python file that declares a module-scope
Modal `App` and `@app.function`/`@app.local_entrypoint`, then invoke
`modal run <generated>.py::main`. Using module-scope decorators (vs
`serialized=True`) avoids Modal's requirement that the local Python
version match the image's. Config (dockerfile, gpu, env, etc.) is
baked into the generated file as Python literals so it's picked up
identically when Modal imports the entrypoint locally and in the
container.

Outputs: the remote function tars `/out` and returns the bytes; the
local entrypoint writes them to a file we extract to the host.
"""

import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path


_ENTRYPOINT_TEMPLATE = '''\
"""Generated Modal entrypoint for mhcflurry.runners. Do not edit."""

import io
import os
import subprocess
import tarfile

import modal


_DOCKERFILE = {dockerfile!r}
_CTX = {ctx!r}
_APP_NAME = {app_name!r}
_GPU = {gpu!r}
_TIMEOUT = {timeout!r}
_OUT_BLOB = {out_blob!r}
_CONTAINER_ENV = {container_env!r}


image = modal.Image.from_dockerfile(_DOCKERFILE, context_dir=_CTX).env(_CONTAINER_ENV)
app = modal.App(_APP_NAME)


@app.function(image=image, gpu=_GPU, timeout=_TIMEOUT)
def runner() -> bytes:
    os.makedirs("/out", exist_ok=True)
    subprocess.run(
        ["python", "-m", "mhcflurry.runners._bootstrap"],
        check=True,
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("/out", arcname=".")
    return buf.getvalue()


@app.local_entrypoint()
def main():
    blob = runner.remote()
    with open(_OUT_BLOB, "wb") as f:
        f.write(blob)
    print(f"[runner] wrote {{len(blob)}} bytes to {{_OUT_BLOB}}")
'''


def run(app, function, args, kwargs, *, outputs_dir: str = "out"):
    try:
        import modal  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Modal backend requires `pip install modal` and `modal setup` (run once)."
        ) from exc

    repo = app._repo_root
    if repo is None:
        raise RuntimeError("App repo_root not set (CLI should have set this).")

    df, ctx = function.image.resolve(repo)
    host_out = (repo / outputs_dir).resolve()
    host_out.mkdir(parents=True, exist_ok=True)

    rel_script = Path(function.module_file).resolve().relative_to(repo)

    # The remote container sees /out from its ENV. All runner-config env
    # vars get baked into the image via Image.env() so the bootstrap picks
    # them up without caller intervention.
    container_env = {
        "MHCFLURRY_OUT": "/out",
        "MHCFLURRY_RUNNER_SCRIPT": f"/workspace/{rel_script}",
        "MHCFLURRY_RUNNER_FUNCTION": function.name,
        "MHCFLURRY_RUNNER_ARGS": json.dumps(args),
        "MHCFLURRY_RUNNER_KWARGS": json.dumps(kwargs),
        **function.env,
    }

    blob_path = tempfile.NamedTemporaryFile(
        suffix=".tar.gz", prefix="mhcflurry-modal-", delete=False
    ).name

    entrypoint_src = _ENTRYPOINT_TEMPLATE.format(
        dockerfile=str(df),
        ctx=str(ctx),
        app_name=f"{app.name}-{function.name}",
        gpu=function.gpu,
        timeout=function.timeout,
        out_blob=blob_path,
        container_env=container_env,
    )

    entry_file = tempfile.NamedTemporaryFile(
        suffix="_modal_entry.py", prefix="mhcflurry-", delete=False, mode="w"
    )
    entry_file.write(entrypoint_src)
    entry_file.close()

    print(f"+ modal run {entry_file.name}::main", flush=True)
    try:
        subprocess.run(
            ["modal", "run", f"{entry_file.name}::main"],
            check=True,
        )
    finally:
        try:
            os.unlink(entry_file.name)
        except OSError:
            pass

    _extract_tar(blob_path, host_out)
    try:
        os.unlink(blob_path)
    except OSError:
        pass
    print(f"Modal run complete. Outputs in {host_out}", flush=True)


def _extract_tar(blob_path: str, dest: Path):
    with open(blob_path, "rb") as f:
        blob = f.read()
    buf = io.BytesIO(blob)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(dest)
