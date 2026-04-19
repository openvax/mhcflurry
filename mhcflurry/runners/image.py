"""Image: Modal-shaped layered image builder.

Same surface as `modal.Image`: start from `from_registry(...)` or
`from_dockerfile(...)`, chain `.apt_install(...)`, `.pip_install(...)`,
`.pip_install_local_dir(...)`, `.run_commands(...)`.

Each backend picks up the ops how it likes:
- Modal: translate 1:1 to `modal.Image` chain (server-side build).
- Local docker / Brev VM mode: synthesize a Dockerfile.
- Brev `--mode container`: `brev create --mode container
  --container-image <base>` then run the ops inline over ssh at boot.

The old `from_dockerfile(...)` path still works (a single-op "include
dockerfile" image). This is additive.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class ImageOp:
    kind: str                          # "apt_install" | "pip_install" | "pip_install_local_dir" | "run"
    args: Tuple[str, ...] = ()
    kwargs: Tuple[Tuple[str, str], ...] = ()   # (key, value) pairs; hashable

    def kwargs_dict(self) -> dict:
        return dict(self.kwargs)


@dataclass(frozen=True)
class Image:
    # Exactly one of `base` / `dockerfile` is set.
    base: Optional[str] = None
    dockerfile: Optional[str] = None
    context: Optional[str] = None
    ops: Tuple[ImageOp, ...] = field(default_factory=tuple)

    # --- constructors ---

    @classmethod
    def from_registry(cls, ref: str) -> "Image":
        """Start from a public image like `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`."""
        return cls(base=ref)

    @classmethod
    def from_dockerfile(cls, dockerfile: str,
                        context: Optional[str] = None) -> "Image":
        """Use an existing Dockerfile. No DSL ops are applied on top."""
        return cls(dockerfile=dockerfile, context=context)

    # --- layer ops ---

    def apt_install(self, *packages: str) -> "Image":
        return self._with_op(ImageOp(kind="apt_install", args=tuple(packages)))

    def pip_install(self, *packages: str,
                    index_url: Optional[str] = None) -> "Image":
        kw: Tuple[Tuple[str, str], ...] = ()
        if index_url is not None:
            kw = (("index_url", index_url),)
        return self._with_op(
            ImageOp(kind="pip_install", args=tuple(packages), kwargs=kw)
        )

    def pip_install_local_dir(self, path: str = ".",
                              editable: bool = True) -> "Image":
        kw = (("path", path), ("editable", "1" if editable else "0"))
        return self._with_op(ImageOp(kind="pip_install_local_dir", kwargs=kw))

    def run_commands(self, *commands: str) -> "Image":
        return self._with_op(ImageOp(kind="run", args=tuple(commands)))

    def _with_op(self, op: ImageOp) -> "Image":
        if self.dockerfile is not None:
            raise ValueError(
                "Cannot chain layer ops on a from_dockerfile() image; build your "
                "image with Image.from_registry(...) + .apt_install/.pip_install/... "
                "instead, or bake everything into the Dockerfile."
            )
        return replace(self, ops=self.ops + (op,))

    # --- resolution for the Docker-building backends ---

    def resolve(self, repo_root: Path) -> Tuple[Path, Path]:
        """For legacy Dockerfile-based images: return (dockerfile, context)."""
        if self.dockerfile is None:
            raise ValueError(
                "resolve() requires a from_dockerfile() image; use render_dockerfile() "
                "for layered images."
            )
        df = (repo_root / self.dockerfile).resolve()
        if not df.is_file():
            raise FileNotFoundError(f"Dockerfile not found: {df}")
        ctx = (repo_root / (self.context or ".")).resolve()
        return df, ctx

    def render_dockerfile(self) -> str:
        """Synthesize a Dockerfile from the layered ops. Used by backends
        that build images (local docker, Brev VM mode, stock `docker build`).

        Uses exec-form `RUN ["cmd", "arg", ...]` for ops that take a
        package list, so Dockerfile's default /bin/sh -c parsing (dash
        on Debian/Ubuntu base images) doesn't mangle version specifiers
        like `pandas>=2.0` — sh/dash would read `>=` as a redirect
        operator in shell-form RUN and silently install the unversioned
        package while creating a file named `=2.0` in the image.
        """
        if self.base is None:
            raise ValueError("render_dockerfile() requires from_registry()")
        import json as _json
        lines = [f"FROM {self.base}", ""]
        lines.append("ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1")
        for op in self.ops:
            if op.kind == "apt_install" and op.args:
                # apt-get syntax is safe for shell-form, but exec-form keeps
                # us consistent; use a /bin/sh -c wrapper.
                cmd = (
                    "apt-get update && apt-get install -y --no-install-recommends "
                    + " ".join(op.args)
                    + " && rm -rf /var/lib/apt/lists/*"
                )
                lines.append(f"RUN {_json.dumps(['/bin/sh', '-c', cmd])}")
            elif op.kind == "pip_install" and op.args:
                kw = op.kwargs_dict()
                argv = ["pip", "install", "--no-cache-dir"]
                if "index_url" in kw:
                    argv += ["--index-url", kw["index_url"]]
                argv += list(op.args)
                lines.append(f"RUN {_json.dumps(argv)}")
            elif op.kind == "pip_install_local_dir":
                kw = op.kwargs_dict()
                path = kw.get("path", ".")
                editable = kw.get("editable", "1") == "1"
                lines.append(f"COPY {path} /workspace")
                argv = ["pip", "install", "--no-cache-dir"]
                if editable:
                    argv.append("-e")
                argv.append("/workspace")
                lines.append(f"RUN {_json.dumps(argv)}")
            elif op.kind == "run" and op.args:
                for cmd in op.args:
                    # run_commands() is explicitly shell-form — users may
                    # use shell features (pipes, &&). That's their choice.
                    lines.append(f"RUN {cmd}")
        return "\n".join(lines) + "\n"
