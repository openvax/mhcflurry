"""`mhcflurry-run` CLI.

Usage:
    mhcflurry-run local <script.py>
    mhcflurry-run brev --instance my-gpu-box <script.py>
    mhcflurry-run modal <script.py>

Loads the user's script, finds its @app.local_entrypoint, sets the backend,
and invokes it.
"""

import argparse
import importlib.util
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="mhcflurry-run",
                                description="Run an mhcflurry training/prediction job.")
    p.add_argument("backend", choices=["local", "brev", "modal"])
    p.add_argument("script", help="Path to a job script defining an App with @local_entrypoint.")
    p.add_argument("--outputs-dir", default="out",
                   help="Host directory to collect outputs into (default: out/).")
    p.add_argument("--instance", help="[brev] Brev instance name.")
    p.add_argument("--no-build", action="store_true",
                   help="[local] Skip docker build (reuse tagged image).")
    args = p.parse_args(argv)

    script_path = Path(args.script).resolve()
    if not script_path.is_file():
        p.error(f"script not found: {script_path}")

    app = _load_app(script_path)

    # repo_root = git root or script's parent
    app._repo_root = _repo_root_for(script_path)
    app._backend = args.backend
    app._backend_kwargs = {"outputs_dir": args.outputs_dir}

    if args.backend == "brev":
        if not args.instance:
            p.error("--instance is required for the brev backend")
        app._backend_kwargs["instance"] = args.instance
    if args.backend == "local" and args.no_build:
        app._backend_kwargs["build"] = False

    if app._entrypoint is None:
        p.error(f"{script_path} has no @app.local_entrypoint() function")
    app._entrypoint()


def _load_app(script_path: Path):
    spec = importlib.util.spec_from_file_location("_mhcflurry_user_job", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mhcflurry_user_job"] = module
    spec.loader.exec_module(module)

    from mhcflurry.runners.app import App
    apps = [v for v in vars(module).values() if isinstance(v, App)]
    if not apps:
        raise SystemExit(f"No App found in {script_path}")
    if len(apps) > 1:
        raise SystemExit(f"Multiple Apps found in {script_path}; expected exactly one.")
    return apps[0]


def _repo_root_for(script_path: Path) -> Path:
    # Walk up looking for a .git dir; fall back to script's parent's parent.
    for parent in [script_path.parent, *script_path.parents]:
        if (parent / ".git").exists():
            return parent
    return script_path.parent


if __name__ == "__main__":
    main()
