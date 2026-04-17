"""Modal backend (stub).

Kept in sync with the local/brev contract so swapping backends is trivial
once we validate it. Not yet exercised end-to-end.
"""

import json
import subprocess


def run(app, function, args, kwargs, *, outputs_dir: str = "out"):
    try:
        import modal
    except ImportError as exc:
        raise RuntimeError(
            "Modal backend requires `pip install modal` and `modal setup` (run once)."
        ) from exc

    repo = app._repo_root
    df, ctx = function.image.resolve(repo)
    image = modal.Image.from_dockerfile(str(df), context_dir=str(ctx))
    modal_app = modal.App(app.name, image=image)

    gpu_spec = function.gpu  # e.g. "A100" or "A100:2"

    @modal_app.function(gpu=gpu_spec, timeout=function.timeout)
    def _runner():
        env = {
            "MHCFLURRY_OUT": "/out",
            "MHCFLURRY_RUNNER_SCRIPT": f"/workspace/{function.module_file.split('/workspace/')[-1]}",
            "MHCFLURRY_RUNNER_FUNCTION": function.name,
            "MHCFLURRY_RUNNER_ARGS": json.dumps(args),
            "MHCFLURRY_RUNNER_KWARGS": json.dumps(kwargs),
            **function.env,
        }
        subprocess.run(
            ["python", "-m", "mhcflurry.runners._bootstrap"],
            env={**env}, check=True,
        )

    with modal_app.run():
        _runner.remote()
