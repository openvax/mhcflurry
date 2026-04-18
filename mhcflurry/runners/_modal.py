"""Modal backend.

Builds a Modal image from `function.image`, runs the in-container
bootstrap with the runner env vars, and returns the contents of
`/out` as a tar blob that gets extracted on the host.

Tar-return is fine for smoketest-sized outputs; heavy training should
switch to `modal.Volume` when we get there.
"""

import io
import json
import tarfile
from pathlib import Path


def run(app, function, args, kwargs, *, outputs_dir: str = "out"):
    try:
        import modal
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

    script_in_container = _container_path(function.module_file, repo)

    image = modal.Image.from_dockerfile(str(df), context_dir=str(ctx))
    modal_app = modal.App(app.name)

    env_vars = {
        "MHCFLURRY_OUT": "/out",
        "MHCFLURRY_RUNNER_SCRIPT": script_in_container,
        "MHCFLURRY_RUNNER_FUNCTION": function.name,
        "MHCFLURRY_RUNNER_ARGS": json.dumps(args),
        "MHCFLURRY_RUNNER_KWARGS": json.dumps(kwargs),
        **function.env,
    }

    @modal_app.function(
        image=image,
        gpu=function.gpu,
        timeout=function.timeout,
        name=f"{app.name}-{function.name}",
    )
    def _runner(env_overrides: dict) -> bytes:
        import os
        import subprocess
        os.makedirs("/out", exist_ok=True)
        subprocess.run(
            ["python", "-m", "mhcflurry.runners._bootstrap"],
            env={**os.environ, **env_overrides},
            check=True,
        )
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add("/out", arcname=".")
        return buf.getvalue()

    with modal_app.run():
        blob = _runner.remote(env_vars)

    _extract_tar(blob, host_out)
    print(f"Modal run complete. Outputs in {host_out}")


def _container_path(host_path: str, repo: Path) -> str:
    rel = Path(host_path).resolve().relative_to(repo)
    return str(Path("/workspace") / rel)


def _extract_tar(blob: bytes, dest: Path):
    buf = io.BytesIO(blob)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(dest)
