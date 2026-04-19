"""App and Function — Modal-shaped surface.

Intentionally minimal:
  - @app.function(image, gpu, timeout, env) decorates a module-level function
  - fn.local(...) calls it in the current process
  - fn.remote(...) dispatches to the selected backend
  - @app.local_entrypoint() marks the function `mhcflurry-run` invokes

Args passed to .remote(...) must be JSON-serializable. No closures, no locals.
"""

import inspect
import json
from pathlib import Path
from typing import Callable, Optional

from mhcflurry.runners.config import BrevConfig, ModalConfig
from mhcflurry.runners.image import Image


class Function:
    def __init__(self, app: "App", fn: Callable, *, image: Image,
                 gpu: Optional[str], timeout: int, env: dict,
                 cpu: Optional[float] = None,
                 memory: Optional[int] = None,
                 min_gpu_memory: Optional[int] = None,
                 min_disk: Optional[int] = None):
        self.app = app
        self.fn = fn
        self.image = image
        # Resource requests — interpret as *minimums*. Each backend picks a
        # matching instance (Modal: direct; Brev: via `brev search`).
        self.gpu = gpu                         # exact GPU name: "T4", "L4", "A100"
        self.cpu = cpu                         # min vCPUs (float for fractional)
        self.memory = memory                   # min RAM in MB (Modal convention)
        self.min_gpu_memory = min_gpu_memory   # min VRAM per GPU in GB
        self.min_disk = min_disk               # min disk in GB
        self.timeout = timeout
        self.env = dict(env or {})
        self.name = fn.__name__
        self.module_file = str(Path(inspect.getfile(fn)).resolve())

    def local(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def remote(self, *args, **kwargs):
        _ensure_json_safe(args, kwargs)
        return self.app._dispatch(self, list(args), dict(kwargs))

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"Call {self.name}.local(...) or {self.name}.remote(...), not {self.name}(...)."
        )


class App:
    def __init__(self, name: str, *,
                 brev: Optional[BrevConfig] = None,
                 modal: Optional[ModalConfig] = None):
        self.name = name
        self.brev = brev or BrevConfig()
        self.modal = modal or ModalConfig()
        self.functions: dict[str, Function] = {}
        self._entrypoint: Optional[Callable] = None

        # Runtime-populated by the CLI before local_entrypoint fires.
        self._backend: Optional[str] = None
        self._backend_kwargs: dict = {}
        self._repo_root: Optional[Path] = None

    def function(self, *, image: Image, gpu: Optional[str] = None,
                 cpu: Optional[float] = None,
                 memory: Optional[int] = None,
                 min_gpu_memory: Optional[int] = None,
                 min_disk: Optional[int] = None,
                 timeout: int = 60 * 60, env: Optional[dict] = None):
        def decorator(fn: Callable) -> Function:
            f = Function(
                self, fn,
                image=image,
                gpu=gpu, cpu=cpu, memory=memory,
                min_gpu_memory=min_gpu_memory, min_disk=min_disk,
                timeout=timeout, env=env or {},
            )
            self.functions[f.name] = f
            return f
        return decorator

    def local_entrypoint(self):
        def decorator(fn: Callable) -> Callable:
            self._entrypoint = fn
            return fn
        return decorator

    def _dispatch(self, function: Function, args: list, kwargs: dict):
        if self._backend is None:
            raise RuntimeError(
                "No backend selected. Invoke via `mhcflurry-run <backend> <script>`."
            )
        backend = self._backend
        if backend == "local":
            from mhcflurry.runners import _local
            return _local.run(self, function, args, kwargs, **self._backend_kwargs)
        if backend == "brev":
            from mhcflurry.runners import _brev
            return _brev.run(self, function, args, kwargs, **self._backend_kwargs)
        if backend == "modal":
            from mhcflurry.runners import _modal
            return _modal.run(self, function, args, kwargs, **self._backend_kwargs)
        raise ValueError(f"Unknown backend: {backend!r}")


def _ensure_json_safe(args, kwargs):
    try:
        json.dumps([list(args), dict(kwargs)])
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Function.remote(...) args must be JSON-serializable. "
            "Use primitives/lists/dicts, not closures or custom objects."
        ) from exc
