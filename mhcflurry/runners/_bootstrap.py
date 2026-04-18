"""In-container bootstrap.

Invoked by the runner inside the container/VM. Imports the user's job script
by file path, then calls the target Function's .local(*args, **kwargs).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path


def main():
    script_path = os.environ["MHCFLURRY_RUNNER_SCRIPT"]
    function_name = os.environ["MHCFLURRY_RUNNER_FUNCTION"]
    args = json.loads(os.environ.get("MHCFLURRY_RUNNER_ARGS", "[]"))
    kwargs = json.loads(os.environ.get("MHCFLURRY_RUNNER_KWARGS", "{}"))

    script_path = str(Path(script_path).resolve())
    spec = importlib.util.spec_from_file_location("_mhcflurry_user_job", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load user job from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mhcflurry_user_job"] = module
    spec.loader.exec_module(module)

    fn = getattr(module, function_name, None)
    if fn is None:
        raise RuntimeError(
            f"Function {function_name!r} not found in {script_path}"
        )
    # fn is a Function wrapper; call the underlying callable directly.
    result = fn.local(*args, **kwargs)
    if result is not None:
        # Emit a sentinel for CLI consumers, but keep stdout human-readable.
        print(f"[runner] result: {result!r}")


if __name__ == "__main__":
    main()
