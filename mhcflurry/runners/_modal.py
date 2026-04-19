"""Modal backend.

Generates a per-invocation Python file with a module-scope `modal.App`
+ `@app.function` + `@app.local_entrypoint`, then runs
`modal run <generated>.py::main`. Module-scope decorators avoid the
Python-version-matching requirement that `serialized=True` carries.

Two image shapes supported:
- `Image.from_dockerfile(path, context=...)`: rendered as
  `modal.Image.from_dockerfile(path, context_dir=context)`.
- `Image.from_registry(ref).apt_install(...).pip_install(...)
  .pip_install_local_dir(".")`: rendered as a `modal.Image` op chain.
  All image build layers run on Modal's build cluster and are cached.

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


_APP_NAME = {app_name!r}
_GPU = {gpu!r}
_CPU = {cpu!r}
_MEMORY = {memory!r}
_TIMEOUT = {timeout!r}
_OUT_BLOB = {out_blob!r}
_CONTAINER_ENV = {container_env!r}


{image_construction}


app = modal.App(_APP_NAME)


# Modal accepts None for cpu/memory — picks a default. Exact GPU string
# passed through.
@app.function(image=image, gpu=_GPU, cpu=_CPU, memory=_MEMORY, timeout=_TIMEOUT)
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

    host_out = (repo / outputs_dir).resolve()
    host_out.mkdir(parents=True, exist_ok=True)

    rel_script = Path(function.module_file).resolve().relative_to(repo)
    container_env = {
        "MHCFLURRY_OUT": "/out",
        "MHCFLURRY_RUNNER_SCRIPT": f"/workspace/{rel_script}",
        "MHCFLURRY_RUNNER_FUNCTION": function.name,
        "MHCFLURRY_RUNNER_ARGS": json.dumps(args),
        "MHCFLURRY_RUNNER_KWARGS": json.dumps(kwargs),
        **function.env,
    }

    image_src = _render_modal_image(function.image, repo=repo)
    image_src += "\nimage = image.env(_CONTAINER_ENV)"

    blob_path = tempfile.NamedTemporaryFile(
        suffix=".tar.gz", prefix="mhcflurry-modal-", delete=False
    ).name

    # Modal's @app.function accepts memory in MB; our API uses GB. Convert.
    modal_memory = (
        int(function.min_memory * 1024) if function.min_memory is not None else None
    )
    entrypoint_src = _ENTRYPOINT_TEMPLATE.format(
        app_name=f"{app.name}-{function.name}",
        gpu=function.gpu,
        cpu=function.min_cpu,
        memory=modal_memory,
        timeout=function.timeout,
        out_blob=blob_path,
        container_env=container_env,
        image_construction=image_src,
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


def _render_modal_image(image, *, repo: Path) -> str:
    """Render an `image = ...` assignment using Modal's Image DSL.

    Maps our Image layer ops 1:1 onto modal.Image methods.
    """
    if image.dockerfile is not None:
        df, ctx = image.resolve(repo)
        return (
            f"image = modal.Image.from_dockerfile({str(df)!r}, "
            f"context_dir={str(ctx)!r})"
        )
    if image.base is None:
        raise ValueError("Image has neither base nor dockerfile")
    lines = [f"image = modal.Image.from_registry({image.base!r})"]
    for op in image.ops:
        kw = op.kwargs_dict()
        if op.kind == "apt_install" and op.args:
            args = ", ".join(repr(a) for a in op.args)
            lines.append(f"image = image.apt_install({args})")
        elif op.kind == "pip_install" and op.args:
            args = ", ".join(repr(a) for a in op.args)
            extra = f", index_url={kw['index_url']!r}" if "index_url" in kw else ""
            lines.append(f"image = image.pip_install({args}{extra})")
        elif op.kind == "pip_install_local_dir":
            path = kw.get("path", ".")
            editable = kw.get("editable", "1") == "1"
            local_dir = (repo / path).resolve()
            flags = "-e " if editable else ""
            lines.append(
                f"image = image.add_local_dir({str(local_dir)!r}, "
                f"remote_path='/workspace', copy=True).run_commands("
                f"'pip install {flags}/workspace')"
            )
        elif op.kind == "run" and op.args:
            args = ", ".join(repr(a) for a in op.args)
            lines.append(f"image = image.run_commands({args})")
    return "\n".join(lines)


def _extract_tar(blob_path: str, dest: Path):
    with open(blob_path, "rb") as f:
        blob = f.read()
    buf = io.BytesIO(blob)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(dest)
